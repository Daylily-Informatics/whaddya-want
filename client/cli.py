#!/usr/bin/env python3
"""
Minimal voice client for the cloud companion.

- Streams microphone audio to Amazon Transcribe (Streaming) using the amazon-transcribe SDK
- Sends FINAL transcripts to a broker (API Gateway/Lambda) as JSON
- Prints AI text and writes returned MP3 (base64) to ./output/

Deps:
  pip install amazon-transcribe sounddevice numpy requests awscrt

Env:
  AWS_REGION / AWS_DEFAULT_REGION (or use --region)
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import os
import queue
import signal
import sys
import threading
import uuid
from pathlib import Path
from typing import AsyncGenerator

import requests
import sounddevice as sd

# Transcribe Streaming SDK (NOT boto3)
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptResultStream


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
async def stream_microphone(
    *,
    region: str,
    language_code: str = "en-US",
    sample_rate: int = 16000,
    channels: int = 1,
    blocksize: int = 4096,
    input_device: int | None = None,
) -> AsyncGenerator[str, None]:
    """
    Async generator yielding FINAL transcripts from Amazon Transcribe Streaming.
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
            # Non-fatal; prints XRuns, etc.
            print(status, file=sys.stderr)
        try:
            raw_q.put_nowait(bytes(indata))
        except queue.Full:
            pass  # drop if back-pressured

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


# ---------- CLI / Broker loop ----------
async def run():
    ap = argparse.ArgumentParser(description="Cloud AI companion voice client")
    ap.add_argument("--broker-url", required=True, help="API Gateway invoke URL (e.g., https://xxx.execute-api.../ingest/audio)")
    ap.add_argument("--session", default=str(uuid.uuid4()), help="Session UUID (auto if omitted)")
    ap.add_argument("--region", default=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-west-2")
    ap.add_argument("--language", default="en-US", help="Transcribe language code (default: en-US)")
    ap.add_argument("--input-device", type=int, default=None, help="sounddevice input device index (optional)")
    ap.add_argument("--rate", type=int, default=16000, help="sample rate (default 16000)")
    args = ap.parse_args()

    print(f"Starting session {args.session} (region={args.region}, lang={args.language})")
    print("Tip: list devices with: python - <<'PY'\nimport sounddevice as sd; print(sd.query_devices())\nPY")

    # graceful Ctrl-C
    stop = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            asyncio.get_running_loop().add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass  # Windows

    async for transcript in stream_microphone(
        region=args.region,
        language_code=args.language,
        sample_rate=args.rate,
        input_device=args.input_device,
    ):
        if stop.is_set():
            break
        print(f"YOU: {transcript}")
        # Post to broker
        try:
            r = requests.post(
                args.broker_url,
                json={"session_id": args.session, "text": transcript},
                timeout=60,
            )
            r.raise_for_status()
            body = r.json()
        except Exception as e:
            print(f"[broker error] {e}", file=sys.stderr)
            continue

        # Expect: {"text": "...", "audio": {"audio_base64": "..."}}
        ai_text = body.get("text", "")
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
