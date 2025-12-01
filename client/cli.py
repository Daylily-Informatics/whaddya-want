import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import boto3
import numpy as np
import sounddevice as sd
from amazon_transcribe.client import TranscribeStreamingClient

from agent_core.logging_utils import configure_logging


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device selection and persistence
# ---------------------------------------------------------------------------

CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".whaddya_want")
CONFIG_PATH = os.path.join(CONFIG_DIR, "devices.json")


def _load_device_config() -> Dict[str, Any]:
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_device_config(data: Dict[str, Any]) -> None:
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _pick_device(kind: str, candidates: List[int]) -> int:
    """Interactively prompt the user to choose an input/output device."""
    devices = sd.query_devices()
    print(f"\nAvailable {kind} devices:")
    for idx in candidates:
        info = devices[idx]
        caps = []
        if info["max_input_channels"] > 0:
            caps.append(f"in={info['max_input_channels']}")
        if info["max_output_channels"] > 0:
            caps.append(f"out={info['max_output_channels']}")
        caps_str = "/".join(caps)
        print(f"  [{idx}] {info['name']} (hostapi={info['hostapi']}, {caps_str})")

    while True:
        raw = input(f"Select {kind} device index: ").strip()
        if not raw:
            continue
        try:
            choice = int(raw)
        except ValueError:
            print("Please enter an integer index from the list.")
            continue
        if choice not in candidates:
            print("Invalid index; please choose from the listed device indices.")
            continue
        return choice


def select_devices(setup_devices: bool) -> Tuple[int, int]:
    """Return (input_device_index, output_device_index), prompting if needed."""
    devices = sd.query_devices()
    input_candidates = [i for i, d in enumerate(devices) if d["max_input_channels"] > 0]
    output_candidates = [i for i, d in enumerate(devices) if d["max_output_channels"] > 0]

    if not input_candidates:
        raise RuntimeError("No audio input devices found.")
    if not output_candidates:
        raise RuntimeError("No audio output devices found.")

    cfg = _load_device_config()
    cached_in = cfg.get("input_device")
    cached_out = cfg.get("output_device")

    if not setup_devices and cached_in in input_candidates and cached_out in output_candidates:
        logger.info(
            "Using cached devices: input=%s output=%s",
            cached_in,
            cached_out,
        )
        return cached_in, cached_out

    # Prompt the user.
    in_dev = _pick_device("input", input_candidates)
    out_dev = _pick_device("output", output_candidates)

    cfg["input_device"] = in_dev
    cfg["output_device"] = out_dev
    _save_device_config(cfg)

    return in_dev, out_dev


# ---------------------------------------------------------------------------
# STT via Amazon Transcribe Streaming
# ---------------------------------------------------------------------------

RATE = 16000
CHANNELS = 1
LANGUAGE_CODE = "en-US"


async def _transcribe_once(duration_s: float, device_index: int) -> str:
    """Capture microphone audio for `duration_s` and return a transcript string."""
    client = TranscribeStreamingClient(region=os.getenv("AWS_REGION") or "us-west-2"
                                       )
    stream = await client.start_stream_transcription(
        language_code=LANGUAGE_CODE,
        media_sample_rate_hz=RATE,
        media_encoding="pcm"    )

    final_text: str = ""

    async def mic() -> None:
        nonlocal stream
        frames_per_chunk = int(RATE * 0.2)  # 0.2s chunks
        start = time.time()
        with sd.InputStream(
            samplerate=RATE,
            channels=CHANNELS,
            dtype="int16",
            device=device_index,
        ) as s:
            while time.time() - start < duration_s:
                audio = s.read(frames_per_chunk)[0].tobytes()
                await stream.input_stream.send_audio_event(audio_chunk=audio)
        await stream.input_stream.end_stream()

    async def listener() -> None:
        nonlocal final_text
        async for event in stream.output_stream:
            if not hasattr(event, "transcript"):
                continue
            for result in event.transcript.results:
                if result.is_partial:
                    continue
                # Take the most recent non-partial result as the utterance.
                final_text = "".join(a.transcript for a in result.alternatives)

    await asyncio.gather(mic(), listener())
    return final_text.strip()


def transcribe_once(duration_s: float, device_index: int) -> str:
    """Synchronous wrapper around _transcribe_once."""
    try:
        return asyncio.run(_transcribe_once(duration_s, device_index))
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        logger.warning("Transcribe failed: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# TTS via Polly + sounddevice
# ---------------------------------------------------------------------------

def synthesize_and_play(
    text: str,
    voice_id: str,
    output_device: int,
    region_name: Optional[str] = None,
    sample_rate: int = RATE,
) -> None:
    """Use Polly to synthesize `text` and play it via sounddevice."""
    if not text:
        return

    region = region_name or os.getenv("AWS_REGION") or os.getenv("REGION") or "us-west-2"
    polly = boto3.client("polly", region_name=region)
    resp = polly.synthesize_speech(
        Text=text,
        VoiceId=voice_id,
        OutputFormat="pcm",
        SampleRate=str(sample_rate),
    )
    audio_bytes = resp["AudioStream"].read()
    data = np.frombuffer(audio_bytes, dtype=np.int16)

    logger.debug("Playing %d samples at %d Hz on device %s", len(data), sample_rate, output_device)
    sd.play(data, samplerate=sample_rate, device=output_device)
    sd.wait()


# ---------------------------------------------------------------------------
# HTTP helper to call the broker
# ---------------------------------------------------------------------------

def call_broker(
    broker_url: str,
    session_id: str,
    transcript: str,
    channel: str = "audio",
    voice_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Send a transcript to the broker HTTP endpoint and return the JSON response."""
    import urllib.request
    import urllib.error

    payload = {
        "transcript": transcript,
        "channel": channel,
    }
    if voice_id is not None:
        payload["voice_id"] = str(voice_id)

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        broker_url,
        data=data,
        headers={"Content-Type": "application/json", "X-Session-Id": session_id},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw)
    except Exception as exc:
        logger.error("Error calling broker at %s: %s", broker_url, exc)
        return {"reply_text": "I had trouble reaching the conversation broker.", "actions": []}


# ---------------------------------------------------------------------------
# Main loops
# ---------------------------------------------------------------------------

def audio_loop(args: argparse.Namespace) -> None:
    """Main audio interaction loop: STT -> broker -> Polly TTS."""
    try:
        in_dev, out_dev = select_devices(setup_devices=args.setup_devices)
    except Exception as exc:
        logger.error("Audio device setup failed: %s", exc)
        sys.exit(1)

    print("\nAudio mode: press Ctrl+C to exit.")
    print(f"Using input device #{in_dev} and output device #{out_dev}.")
    print()

    voice_id = args.voice or os.getenv("AGENT_VOICE_ID") or "Matthew"

    while True:
        try:
            print("Speak now...")
            utter = transcribe_once(duration_s=args.utterance_seconds, device_index=in_dev)
            if not utter:
                print("(No speech detected)")
                continue

            if args.verbose:
                print(f"[YOU] {utter}")

            resp = call_broker(
                broker_url=args.broker_url,
                session_id=args.session,
                transcript=utter,
                channel="audio",
                voice_id=in_dev,
            )
            reply_text = resp.get("reply_text") or ""
            print(f"[AGENT] {reply_text}")
            synthesize_and_play(
                text=reply_text,
                voice_id=voice_id,
                output_device=out_dev,
                region_name=os.getenv("AWS_REGION") or os.getenv("REGION"),
            )
        except KeyboardInterrupt:
            print("\nExiting audio loop.")
            break


def text_loop(args: argparse.Namespace) -> None:
    """Simple text-only REPL that talks to the broker."""
    print("Text-only mode. Ctrl+D or Ctrl+C to exit.")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not line:
            continue

        resp = call_broker(
            broker_url=args.broker_url,
            session_id=args.session,
            transcript=line,
            channel="cli",
        )
        reply_text = resp.get("reply_text") or ""
        print(reply_text)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Local CLI front-end for the whaddya-want agent.")
    parser.add_argument(
        "--session",
        required=True,
        help="Session identifier for grouping conversations on the broker side.",
    )
    parser.add_argument(
        "--broker-url",
        required=True,
        help="HTTP URL for the conversation broker endpoint.",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Use text-only interaction (disable microphone and audio playback).",
    )
    parser.add_argument(
        "--setup-devices",
        action="store_true",
        help="Force prompting for input/output audio devices, even if cached devices exist.",
    )
    parser.add_argument(
        "--voice",
        default=None,
        help="Polly VoiceId to use for the agent voice (default: Matthew).",
    )
    parser.add_argument(
        "--utterance-seconds",
        type=float,
        default=8.0,
        help="Maximum length of each recorded utterance in seconds.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity; can be specified multiple times.",
    )

    # Backwards-compat arguments (ignored but accepted)
    parser.add_argument(
        "--voice-mode",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--self-voice-name",
        default=None,
        help=argparse.SUPPRESS,
    )

    args = parser.parse_args()

    # Configure logging based on -v level.
    configure_logging(args.verbose)

    if args.text_only:
        text_loop(args)
    else:
        audio_loop(args)


if __name__ == "__main__":
    main()
