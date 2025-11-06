#!/usr/bin/env python3
"""
Minimal+ voice client for the cloud companion — with local speaker-ID and bark detection.

Features
- Streams microphone audio to Amazon Transcribe (Streaming) via amazon-transcribe SDK
- Emits ONLY FINAL transcripts
- Local speaker recognition (SpeechBrain ECAPA-TDNN)
- Local dog-bark detection (YAMNet on TF Hub)
- Voice enrollment phrase(s): "enroll my voice as <Name>", "enroll me as <Name>",
  tolerant variants like "and roll my voice as <Name>", plus "call me <Name>"
- Manual enroll: press ENTER to enroll last ~3s as 'Major', or type a name then ENTER
- Optional --force-enroll <Name> to enroll once automatically
- Voice exit commands (exit/quit/stop listening/goodbye/stop now)
- Keyboard quit: type 'q' + ENTER
- Sends context: {"speaker_id": "...", "acoustic_event": "dog_bark"}
- Sends client_event on enrollment: {"enrolled_speaker": "<Name>"}
- Honors broker client_event {"command":"exit"}
- Writes broker TTS mp3 to ./output/

Install (CPU-only reference set)
  pip install amazon-transcribe sounddevice numpy requests awscrt
  pip install torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cpu
  pip install speechbrain==0.5.16 "huggingface_hub<0.21"
  pip install tensorflow==2.16.1 tensorflow_hub tensorflow_io==0.36.0

Env
  AWS_REGION / AWS_DEFAULT_REGION (or pass --region)

Run
  python client/cli.py --broker-url https://.../ingest/audio --region us-west-2
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import os
import queue
import re
import signal
import sys
import threading
import uuid
import warnings
from collections import deque
from pathlib import Path
from typing import AsyncGenerator, Optional

import numpy as np
import requests
import sounddevice as sd

# Suppress harmless noise
warnings.filterwarnings("ignore", message="torchvision is not available")
warnings.filterwarnings(
    "ignore",
    message=r"You are using `torch\.load` with `weights_only=False`",
    category=FutureWarning,
    module="speechbrain",
)
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

# ---- Amazon Transcribe (streaming) ----
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptResultStream

# ---- Speaker-ID (SpeechBrain ECAPA) ----
_SPK_READY = False
try:
    import torch  # noqa: F401
    from speechbrain.pretrained import EncoderClassifier  # type: ignore
    _SPK_READY = True
except Exception as e:
    print("[diag] speechbrain import error:", repr(e), file=sys.stderr)
    _SPK_READY = False

_SPK_DIR = Path(os.path.expanduser("~/.whaddya/speakers"))
_SPK_DIR.mkdir(parents=True, exist_ok=True)


# ---- AI audio playback (mp3 → speaker) ----
_MP3_READY = False
try:
    from pydub import AudioSegment  # type: ignore

    _MP3_READY = True
except Exception as e:
    print("[diag] mp3 playback disabled (pydub import failed):", repr(e), file=sys.stderr)
    _MP3_READY = False


class AudioPlayer:
    """Decode mp3 bytes and play them through the default sounddevice output."""

    def __init__(self, mute_guard: threading.Event | None = None) -> None:
        self.enabled = _MP3_READY
        self._warned_decode_error = False
        self._mute_guard = mute_guard

    def _decode(self, data: bytes) -> tuple[np.ndarray, int] | tuple[None, None]:
        if not self.enabled:
            return (None, None)
        try:
            segment = AudioSegment.from_file(io.BytesIO(data), format="mp3")
        except Exception as exc:
            if not self._warned_decode_error:
                print(
                    "[diag] unable to decode mp3 audio — install ffmpeg or libav for pydub:",
                    exc,
                    file=sys.stderr,
                )
                self._warned_decode_error = True
            return (None, None)
        samples = np.array(segment.get_array_of_samples())
        if segment.channels > 1:
            samples = samples.reshape((-1, segment.channels))
        else:
            samples = samples.reshape((-1, 1))
        scale = float(1 << (8 * segment.sample_width - 1))
        audio = samples.astype(np.float32) / scale
        return audio, int(segment.frame_rate)

    async def play(self, data: bytes) -> bool:
        if not self.enabled:
            return False
        if self._mute_guard is not None:
            self._mute_guard.set()
        try:
            loop = asyncio.get_running_loop()
            audio, rate = await loop.run_in_executor(None, self._decode, data)
            if audio is None or rate is None:
                return False
            sd.play(audio, rate)
            await loop.run_in_executor(None, sd.wait)
            if self._mute_guard is not None:
                await asyncio.sleep(0.15)
        finally:
            if self._mute_guard is not None:
                self._mute_guard.clear()
        return True


class SpeakerID:
    """Lightweight wrapper around SpeechBrain ECAPA for enroll/identify."""
    def __init__(self):
        self.enabled = _SPK_READY
        self.model = None
        self.cache: dict[str, np.ndarray] = {}

    def _ensure_model(self):
        if not self.enabled:
            return
        if self.model is None:
            self.model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": "cpu"},
            )
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


# ---- Bark detector (TFHub YAMNet) ----
_YAM_READY = False
try:
    import tensorflow as tf  # noqa: F401
    import tensorflow_hub as hub  # type: ignore
    import tensorflow_io as tfio  # noqa: F401
    _YAM_READY = True
except Exception as e:
    print("[diag] yamnet import error:", repr(e), file=sys.stderr)
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
        scores, _, _ = self.model(waveform)              # [frames, 521]
        mean_scores = tf.reduce_mean(scores, axis=0).numpy()  # [521]
        idx = {lbl: i for i, lbl in enumerate(self.labels)}

        p_dog  = float(mean_scores[idx["Dog"]])  if "Dog"  in idx else 0.0
        p_bark = float(mean_scores[idx["Bark"]]) if "Bark" in idx else 0.0

        return max(p_dog, p_bark) >= prob_threshold



# ---- Transcript handler (finals only) ----
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


# ---- Mic → Transcribe streaming ----
def _bytes_to_float_mono_int16(data: bytes, channels: int) -> np.ndarray:
    """Convert interleaved int16 bytes to mono float32 [-1,1]."""
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
    mute_event: threading.Event | None = None,
) -> AsyncGenerator[str, None]:
    """Yield FINAL transcripts; optionally fill analysis_buf with float32 @16k mono."""
    client = TranscribeStreamingClient(region=region)
    stream = await client.start_stream_transcription(
        language_code=language_code,
        media_sample_rate_hz=sample_rate,
        media_encoding="pcm",
    )

    loop = asyncio.get_running_loop()
    raw_q: queue.Queue[bytes] = queue.Queue(maxsize=100)
    async_q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=100)

    def audio_cb(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        if mute_event is not None and mute_event.is_set():
            return
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

    threading.Thread(target=mic_thread, daemon=True).start()

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


# ---- Helpers ----
# Accept STT goofs: "and roll", "in roll", "role", etc., plus "call me <Name>"
_ENROLL_RE = re.compile(
    r"""(?xi)
    \b(?:enrol+|in\s*rol+|and\s*rol+|role)\s+
      (?:my\s+voice|me)\s*
      (?:as\s+)?                               # optional 'as'
      (?:
        (?:i'?m|i\s+am|this\s+is|says)\s+      # optional bridge
      )?
      ([A-Za-z][\w\s'\-]{0,31})\b
    |
    \bcall\s+me\s+([A-Za-z][\w\s'\-]{0,31})\b
    """,
)

_EXIT_RE = re.compile(r"\b(exit|quit|stop listening|goodbye|stop now)\b", re.I)


def parse_enrollment(text: str) -> Optional[str]:
    m = _ENROLL_RE.search(text)
    if not m:
        return None
    name = (m.group(1) or m.group(2) or "").strip()
    name = re.sub(r"\s+", " ", name)
    return name or None


def take_latest_seconds(buf: deque, seconds: float, rate: int) -> Optional[np.ndarray]:
    need = int(seconds * rate)
    if len(buf) < need:
        return None
    arr = np.array([buf[i] for i in range(len(buf) - need, len(buf))], dtype=np.float32)
    mx = float(np.max(np.abs(arr))) + 1e-9
    arr = arr / mx
    return arr


# ---- CLI / Broker loop ----
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
    ap.add_argument("--force-enroll", default=None, help="Enroll given name from the first usable buffer (one-shot)")
    ap.add_argument(
        "--voice",
        default=os.getenv("POLLY_VOICE"),
        help="Optional Amazon Polly voice ID override (default: server configuration)",
    )
    args = ap.parse_args()

    voice_id = (args.voice or "").strip() or None

    if voice_id:
        print(f"[diag] overriding Polly voice to '{voice_id}'")

    print(f"Starting session {args.session} (region={args.region}, lang={args.language})")
    print("Tip: list devices with:\n  python - <<'PY'\nimport sounddevice as sd; print(sd.query_devices())\nPY\n")

    # graceful Ctrl-C
    stop = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            asyncio.get_running_loop().add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass

    # Rolling analysis buffer at 16k mono floats
    analysis_buf = deque(maxlen=int(args.rate * 6))  # keep last ~6 seconds

    # Models
    spkid = SpeakerID()
    bark = BarkDetector()
    playback_mute = threading.Event()
    player = AudioPlayer(mute_guard=playback_mute)
    print(
        "[diag] speaker-id enabled={}  bark-detector enabled={}  playback enabled={}".format(
            spkid.enabled, bark.enabled, player.enabled
        )
    )
    print("[hint] Press ENTER to enroll last ~3s as 'Major' (or type a name then ENTER). Type 'q' + ENTER to quit.")

    # Manual enroll / quit via stdin
    manual_q = queue.Queue()

    def _stdin_watcher():
        try:
            while True:
                line = sys.stdin.readline()
                if not line:
                    break
                s = line.strip()
                if s.lower() in ("q", "quit", "exit"):
                    manual_q.put("__QUIT__")
                else:
                    manual_q.put(s)
        except Exception:
            pass

    threading.Thread(target=_stdin_watcher, daemon=True).start()

    forced_done = False
    playback_warned = False

    async for transcript in stream_microphone(
        region=args.region,
        language_code=args.language,
        sample_rate=args.rate,
        channels=args.channels,
        input_device=args.input_device,
        analysis_buf=analysis_buf,
        mute_event=playback_mute,
    ):
        if stop.is_set():
            break

        print(f"YOU: {transcript}")

        # Local voice exit
        if _EXIT_RE.search(transcript):
            print("[voice] exit requested — shutting down.")
            break

        need = int(args.id_window * args.rate)
        print(f"[diag] buf_samples={len(analysis_buf)} need~{need}")

        # One-shot forced enrollment
        if args.force_enroll and not forced_done:
            wav_fe = take_latest_seconds(analysis_buf, args.id_window, args.rate)
            if wav_fe is not None and spkid.enroll(args.force_enroll, wav_fe):
                forced_done = True
                print(f"[enrolled voice] {args.force_enroll} (forced)")

        # Manual enroll / quit via stdin
        while not manual_q.empty():
            typed = manual_q.get()
            if typed == "__QUIT__":
                print("[keyboard] quit requested — shutting down.")
                stop.set()
                break
            target = typed if typed else "Major"
            wav_m = take_latest_seconds(analysis_buf, max(args.id_window, 3.0), args.rate)
            if wav_m is not None and spkid.enroll(target, wav_m):
                print(f"[enrolled voice] {target} (manual)")
            else:
                print("[enrollment failed] buffer too short or model unavailable", file=sys.stderr)
        if stop.is_set():
            break

        # Enrollment phrase from transcript
        enrolled_name = None
        name = parse_enrollment(transcript)
        if name:
            wav = take_latest_seconds(analysis_buf, args.id_window, args.rate)
            if wav is not None and spkid.enroll(name, wav):
                enrolled_name = name
                print(f"[enrolled voice] {name}")
            else:
                print("[enrollment failed] not enough audio or model unavailable.", file=sys.stderr)

        # Build context for broker
        context = {}
        wav_id = take_latest_seconds(analysis_buf, args.id_window, args.rate)
        if wav_id is not None:
            who = spkid.identify(wav_id, threshold=args.id_threshold)
            if who:
                context["speaker_id"] = who
            try:
                if bark.detect(wav_id):
                    context["acoustic_event"] = "dog_bark"
            except Exception as e:
                print("[diag] bark detect error:", e, file=sys.stderr)

        client_event = {}
        if enrolled_name:
            client_event["enrolled_speaker"] = enrolled_name

        # POST to broker
        payload = {"session_id": args.session, "text": transcript}
        if context:
            payload["context"] = context
        if client_event:
            payload["client_event"] = client_event
        if voice_id:
            payload["voice_id"] = voice_id

        try:
            r = requests.post(args.broker_url, json=payload, timeout=60)
            r.raise_for_status()
            body = r.json()
        except Exception as e:
            print(f"[broker error] {e}", file=sys.stderr)
            continue

        # Honor broker-requested exit
        ce = body.get("client_event") or {}
        if isinstance(ce, dict) and ce.get("command") == "exit":
            print("[broker] requested exit — shutting down.")
            break

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
            audio_bytes = base64.b64decode(audio_b64)
            out.write_bytes(audio_bytes)
            print(f"[saved] {out}")
            played = await player.play(audio_bytes)
            if not played and not playback_warned:
                print(
                    "[hint] Install 'pydub' and ffmpeg (or libav) to enable speaker playback.",
                    file=sys.stderr,
                )
                playback_warned = True

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
