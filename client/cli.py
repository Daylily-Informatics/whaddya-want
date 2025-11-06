#!/usr/bin/env python3
"""
Minimal+ voice client for the cloud companion — with local speaker-ID and bark detection.

Features
- Streams microphone audio to Amazon Transcribe (Streaming) via amazon-transcribe SDK
- Emits ONLY FINAL transcripts
- Local speaker recognition (SpeechBrain ECAPA-TDNN)
- Local dog-bark detection (YAMNet on TF Hub)
- Voice enrollment phrase: "enroll my voice as <Name>"
- Sends context to broker: {"speaker_id": "...", "acoustic_event": "dog_bark"}
- Sends client_event on enrollment: {"enrolled_speaker": "<Name>"}
- Writes broker TTS mp3 to ./output/

Install
  pip install amazon-transcribe sounddevice numpy requests awscrt
  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
  pip install speechbrain tensorflow==2.16.1 tensorflow_hub tensorflow_io==0.36.0

Env
  AWS_REGION / AWS_DEFAULT_REGION (or pass --region)

Run
  python cli.py --broker-url https://.../ingest/audio --region us-west-2
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import os
import queue
import re
import signal
import sys
import threading
import uuid
from collections import deque
from pathlib import Path
from typing import AsyncGenerator, Optional, Tuple

import numpy as np
import requests
import sounddevice as sd

# Transcribe Streaming SDK (NOT boto3)
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptResultStream

# ---------- Speaker-ID (SpeechBrain ECAPA) ----------
_SPK_READY = False
try:
    import torch  # noqa: F401
    from speechbrain.pretrained import EncoderClassifier  # type: ignore
    _SPK_READY = True
except Exception:
    _SPK_READY = False

_SPK_DIR = Path(os.path.expanduser("~/.whaddya/speakers"))
_SPK_DIR.mkdir(parents=True, exist_ok=True)

class SpeakerID:
    """Lightweight wrapper around SpeechBrain ECAPA for enroll/identify."""
    def __init__(self):
        self.enabled = _SPK_READY
        self.model = None
        self.cache = {}  # name -> np.ndarray embedding

    def _ensure_model(self):
        if not self.enabled:
            return
        if self.model is None:
            # Lazy-load to avoid startup latency
            self.model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": "cpu"}
            )
            # Load any existing embeddings
            for p in _SPK_DIR.glob("*.npy"):
                try:
                    self.cache[p.stem] = np.load(p)
                except Exception:
                    pass

    def enroll(self, name: str, wav16: np.ndarray) -> bool:
        """Store/overwrite embedding for `name`. wav16: float32 mono 16kHz."""
        if not self.enabled:
            return False
        self._ensure_model()
        if self.model is None:
            return False
        with torch.no_grad():
            emb = self.model.encode_batch(torch.from_numpy(wav16).unsqueeze(0))
        vec = emb.squeeze(0).squeeze(0).cpu().numpy()
        np.save(_SPK_DIR / f"{name}.npy", vec)
        self.cache[name] = vec
        return True

    def identify(self, wav16: np.ndarray, threshold: float = 0.65) -> Optional[str]:
        """Return best-match name if above cosine-sim threshold, else None."""
        if not self.enabled:
            return None
        self._ensure_model()
        if self.model is None or not self.cache:
            return None
        with torch.no_grad():
            probe = self.model.encode_batch(torch.from_numpy(wav16).unsqueeze(0))
        pv = probe.squeeze(0).squeeze(0).cpu().numpy()
        best, best_name = -1.0, None
        for name, ref in self.cache.items():
            sim = float(np.dot(pv, ref) / (np.linalg.norm(pv) * np.linalg.norm(ref) + 1e-9))
            if sim > best:
                best, best_name = sim, name
        return best_name if best >= threshold else None

# ---------- Bark detector (YAMNet) ----------
_YAM_READY = False
try:
    import tensorflow as tf  # noqa: F401
    import tensorflow_hub as hub  # type: ignore
    import tensorflow_io as tfio  # noqa: F401
    _YAM_READY = True
except Exception:
    _YAM_READY = False

class BarkDetector:
    """Binary bark-present detector using YAMNet class scores."""
    def __init__(self):
        self.enabled = _YAM_READY
        self.model = None
        self.labels = None

    def _ensure_model(self):
        if not self.enabled or self.model is not None:
            return
        self.model = hub.load("https://tfhub.dev/google/yamnet/1")
        # Load class map (display_name column)
        labels_path = hub.resolve("https://tfhub.dev/google/yamnet/1") + "/assets/yamnet_class_map.csv"
        raw = tf.io.read_file(labels_path).numpy().decode().splitlines()
        self.labels = [r.split(",")[-1] for r in raw]

    def detect(self, wav16: np.ndarray, prob_threshold: float = 0.25) -> bool:
        if not self.enabled:
            return False
        self._ensure_model()
        if self.model is None or self.labels is None:
            return False
        waveform = tf.convert_to_tensor(wav16, dtype=tf.float32)
        scores, _, _ = self.model(waveform)
        mean_scores = tf.reduce_mean(scores, axis=0).numpy()
        idx = {lbl: i for i, lbl in enumerate(self.labels)}
        p_dog = mean_scores[idx["Dog"]] if "Dog" in idx else 0.0
        p_bark = mean_scores[idx["Bark"]] if "Bark" in idx else 0.0
        return max(float(p_dog), float(p_bark)) >= prob_threshold

# ---------- Transcript handler (finals only) ----------
class FinalsOnlyHandler(TranscriptResultStreamHandler):
    def __init__(self, stream: TranscriptResultStream, out_q: asyncio.Queue[str]):
        super().__init__(stream)
        self.out_q = out_q

    async def handle_transcript_event(self, transcript_event):
        for result in transcript_event.transcript.results:
            if result.is_partial:
                continue
            text = "".join(alt.transcript for alt in result.alternatives)
            if text.strip():
                await self.out_q.put(text.strip())

# ---------- Microphone → Transcribe streaming ----------
def _bytes_to_float_mono_int16(data: bytes, channels: int) -> np.ndarray:
    """Convert interleaved int16 bytes to mono float32 [-1,1]"""
    arr = np.frombuffer(data, dtype=np.int16)
    if channels > 1:
        arr = arr.reshape(-1, channels).mean(axis=1).astype(np.int16)
    f = arr.astype(np.float32) / 32768.0
    return f

async def stream_microphone(
    *,
    region: str,
    language_code: str = "en-US",
    sample_rate: int = 16000,
    channels: int = 1,
    blocksize: int = 4096,
    input_device: int | None = None,
    analysis_buf: deque | None = None,
) -> AsyncGenerator[str, None]:
    """
    Async generator yielding FINAL transcripts from Amazon Transcribe Streaming.
    Also fills analysis_buf (float32 16k mono) for local models if provided.
    """
    client = TranscribeStreamingClient(region=region)
    stream = await client.start_stream_transcription(
        language_code=language_code,
        media_sample_rate_hz=sample_rate,
        media_encoding="pcm",
    )

    loop = asyncio.get_running_loop()

    # Bridge: PortAudio callback -> blocking queue -> asyncio queue
    raw_q: queue.Queue[bytes] = queue.Queue(maxsize=100)
    async_q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=100)

    def audio_cb(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        try:
            raw = bytes(indata)
            if analysis_buf is not None:
                fmono = _bytes_to_float_mono_int16(raw, channels)
                analysis_buf.extend(fmono.tolist())
            raw_q.put_nowait(raw)
        except queue.Full:
            pass

    def mic_thread():
        with sd.RawInputStream(
            samplerate=sample_rate,
            blocksize=blocksize,
            dtype="int16",
            channels=channels,
            device=input_device,
            callback=audio_cb,
        ):
            while True:
                chunk = raw_q.get()
                loop.call_soon_threadsafe(async_q.put_nowait, chunk)

    t = threading.Thread(target=mic_thread, daemon=True)
    t.start()

    async def sender():
        try:
            while True:
                chunk = await async_q.get()
                await stream.input_stream.send_audio_event(audio_chunk=chunk)
        finally:
            await stream.input_stream.end_stream()

    finals_q: asyncio.Queue[str] = asyncio.Queue()
    handler = FinalsOnlyHandler(stream.output_stream, finals_q)

    send_task = asyncio.create_task(sender())
    recv_task = asyncio.create_task(handler.handle_events())

    try:
        while True:
            text = await finals_q.get()
            yield text
    finally:
        send_task.cancel()
        recv_task.cancel()

# ---------- Helpers ----------
_ENROLL_RE = re.compile(r"\b(enrol+l?\s+my\s+voice\s+as|enrol+l?\s+me\s+as)\s+([A-Za-z][\w\s'\-]{0,31})\b", re.IGNORECASE)

def parse_enrollment(text: str) -> Optional[str]:
    m = _ENROLL_RE.search(text)
    if not m:
        return None
    name = m.group(2).strip()
    # Collapse spaces, title-case for display; keep original for ID
    name = re.sub(r"\s+", " ", name)
    return name

def take_latest_seconds(buf: deque, seconds: float, rate: int) -> Optional[np.ndarray]:
    need = int(seconds * rate)
    if len(buf) < need:
        return None
    arr = np.array([buf[i] for i in range(len(buf) - need, len(buf))], dtype=np.float32)
    # normalize lightly
    mx = float(np.max(np.abs(arr))) + 1e-9
    arr = arr / mx
    return arr

# ---------- CLI / Broker loop ----------
async def run():
    ap = argparse.ArgumentParser(description="Cloud AI companion voice client (+speaker-ID, bark detect)")
    ap.add_argument("--broker-url", required=True, help="API Gateway invoke URL (e.g., https://xxx.execute-api.../ingest/audio)")
    ap.add_argument("--session", default=str(uuid.uuid4()), help="Session UUID (auto if omitted)")
    ap.add_argument("--region", default=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-west-2")
    ap.add_argument("--language", default="en-US", help="Transcribe language code (default: en-US)")
    ap.add_argument("--input-device", type=int, default=None, help="sounddevice input device index (optional)")
    ap.add_argument("--rate", type=int, default=16000, help="sample rate (default 16000)")
    ap.add_argument("--channels", type=int, default=1, help="mic channels (default 1)")
    ap.add_argument("--id-threshold", type=float, default=0.65, help="speaker cosine threshold (default 0.65)")
    ap.add_argument("--id-window", type=float, default=2.0, help="seconds of audio for ID/enroll (default 2.0)")
    args = ap.parse_args()

    print(f"Starting session {args.session} (region={args.region}, lang={args.language})")
    print("Tip: list devices with:\n  python - <<'PY'\nimport sounddevice as sd; print(sd.query_devices())\nPY\n")

    # graceful Ctrl-C
    stop = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            asyncio.get_running_loop().add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass  # Windows

    # Rolling analysis buffer at 16k mono floats
    analysis_buf = deque(maxlen=int(args.rate * 6))  # keep last ~6 seconds

    # Lazy-init models
    spkid = SpeakerID()
    bark = BarkDetector()

    async for transcript in stream_microphone(
        region=args.region,
        language_code=args.language,
        sample_rate=args.rate,
        channels=args.channels,
        input_device=args.input_device,
        analysis_buf=analysis_buf,
    ):
        if stop.is_set():
            break

        print(f"YOU: {transcript}")

        # 1) Enrollment phrase?
        enrolled_name = None
        name = parse_enrollment(transcript)
        if name:
            wav = take_latest_seconds(analysis_buf, args.id_window, args.rate)
            if wav is not None and spkid.enroll(name, wav):
                enrolled_name = name
                print(f"[enrolled voice] {name}")
            else:
                print("[enrollment failed] Not enough audio or model unavailable.", file=sys.stderr)

        # 2) Build context from buffer
        context = {}
        wav_id = take_latest_seconds(analysis_buf, args.id_window, args.rate)
        if wav_id is not None:
            who = spkid.identify(wav_id, threshold=args.id_threshold)
            if who:
                context["speaker_id"] = who
            if bark.detect(wav_id):
                context["acoustic_event"] = "dog_bark"

        client_event = {}
        if enrolled_name:
            client_event["enrolled_speaker"] = enrolled_name

        # 3) POST to broker
        payload = {"session_id": args.session, "text": transcript}
        if context:
            payload["context"] = context
        if client_event:
            payload["client_event"] = client_event

        try:
            r = requests.post(args.broker_url, json=payload, timeout=60)
            r.raise_for_status()
            body = r.json()
        except Exception as e:
            print(f"[broker error] {e}", file=sys.stderr)
            continue

        # Expect: {"text": "...", "audio": {"audio_base64": "..."}}
        ai_text = body.get("text", "")
        if context.get("speaker_id"):
            print(f"[ctx] speaker={context['speaker_id']}", end="")
            if context.get("acoustic_event"):
                print(f"  event={context['acoustic_event']}")
            else:
                print()
        elif context.get("acoustic_event"):
            print(f"[ctx] event={context['acoustic_event']}")
        if client_event:
            print(f"[event] enrolled_speaker={client_event['enrolled_speaker']}")

        print(f"AI:  {ai_text}")

        audio_b64 = (body.get("audio") or {}).get("audio_base64")
        if audio_b64:
            outdir = Path("output")
            outdir.mkdir(exist_ok=True)
            out = outdir / f"response-{uuid.uuid4()}.mp3"
            out.write_bytes(base64.b64decode(audio_b64))
            print(f"[saved] {out}")

        if stop.is_set():
            break

    print("Exiting.")

def main():
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
