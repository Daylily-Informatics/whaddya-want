"""Minimal CLI client that streams audio chunks to Amazon Transcribe and calls the broker."""
from __future__ import annotations

import argparse
import base64
import queue
import threading
import uuid
from pathlib import Path

import boto3
import requests
import sounddevice as sd


def stream_microphone(language_code: str = "en-US", sample_rate: int = 16000):
    transcribe = boto3.client("transcribe")
    q: queue.Queue[bytes] = queue.Queue()

    def callback(indata, frames, time, status):  # pragma: no cover - audio callback
        if status:
            print(status)
        q.put(bytes(indata))

    stream = transcribe.start_stream_transcription(
        LanguageCode=language_code,
        MediaSampleRateHertz=sample_rate,
        MediaEncoding="pcm",
    )

    def producer():  # pragma: no cover - network stream
        with sd.RawInputStream(
            samplerate=sample_rate,
            blocksize=4096,
            dtype="int16",
            channels=1,
            callback=callback,
        ):
            while True:
                chunk = q.get()
                stream.send_audio_event(AudioChunk=chunk)

    threading.Thread(target=producer, daemon=True).start()
    for event in stream:  # pragma: no cover - stream consumption
        if "Transcript" in event:
            results = event["Transcript"]["Results"]
            for result in results:
                if not result.get("IsPartial"):
                    yield result["Alternatives"][0]["Transcript"]


def main() -> None:  # pragma: no cover - CLI entry
    parser = argparse.ArgumentParser(description="Interact with the cloud AI companion")
    parser.add_argument("--broker-url", required=True, help="Invoke URL of the broker Lambda")
    parser.add_argument("--session", default=str(uuid.uuid4()))
    args = parser.parse_args()

    session = args.session
    print(f"Starting session {session}")

    for transcript in stream_microphone():
        payload = {"session_id": session, "text": transcript}
        response = requests.post(args.broker_url, json=payload, timeout=30)
        response.raise_for_status()
        body = response.json()
        print(f"AI: {body['text']}")
        audio_path = Path("output")
        audio_path.mkdir(exist_ok=True)
        file_path = audio_path / f"response-{uuid.uuid4()}.mp3"
        audio_bytes = base64.b64decode(body["audio"]["audio_base64"])
        file_path.write_bytes(audio_bytes)
        print(f"Saved audio to {file_path}")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
