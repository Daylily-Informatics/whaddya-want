import argparse
import asyncio
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import boto3
import numpy as np
import sounddevice as sd
from amazon_transcribe.client import TranscribeStreamingClient

from agent_core.logging_utils import configure_logging
from . import identity  # NEW: voice identity helpers

NAME_STOPWORDS = {"it's", "its", "what", "you", "i", "call", "should", "name"}


def extract_name_heuristic(utterance: str) -> Optional[str]:
    """
    Heuristic extractor for a speaker's self-reported name.

    Handles simple patterns like:
      - "my name is Major"
      - "I'm Major"
      - "I am Major"
      - "Major"  (single-token reply)

    Returns the extracted name string, or None if no good candidate is found.
    """
    if not utterance:
        return None

    text = utterance.strip()
    lower = text.lower().strip()

    # If it's clearly a question back at the agent, it's not a name.
    if lower.endswith("?"):
        return None

    # Strip common leading phrases
    prefixes = [
        "my name is ",
        "i am ",
        "i'm ",
        "its ",
        "it's ",
        "call me ",
    ]
    for p in prefixes:
        if lower.startswith(p):
            name_part = text[len(p):].strip()
            break
    else:
        name_part = text

    # Take first token up to punctuation as candidate
    import re as _re
    name_part = name_part.strip(" .,:;!?\"'")
    if not name_part:
        return None
    # Split on punctuation to discard trailing phrases
    token = _re.split(r"[,.!?]", name_part, maxsplit=1)[0].strip()
    if not token:
        return None

    # Very short or stopword-y tokens are not acceptable names.
    if token.lower() in NAME_STOPWORDS:
        return None
    if len(token) > 40:
        return None

    # Single token only (for now)
    if " " in token:
        token = token.split()[0]

    return token or None


def llm_extract_name(utterance: str) -> Optional[str]:
    """
    Optional LLM-based name extractor.

    Expects the model to return either:
      - a single name token, e.g.  Major
      - or the literal token  NONE

    If the OpenAI client is not available or the call fails, returns None.
    """
    return None


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

    in_dev = _pick_device("input", input_candidates)
    out_dev = _pick_device("output", output_candidates)

    cfg["input_device"] = in_dev
    cfg["output_device"] = out_dev
    _save_device_config(cfg)

    return in_dev, out_dev


# ---------------------------------------------------------------------------
# STT via Amazon Transcribe Streaming + naive speaker embedding
# ---------------------------------------------------------------------------

RATE = 16000
CHANNELS = 1
LANGUAGE_CODE = "en-US"


def _compute_embedding_from_samples(samples: np.ndarray, num_bins: int = 32) -> Optional[List[float]]:
    """Compute a simple, fixed-size embedding from a 1D float32 sample array."""
    if samples.size == 0:
        return None

    if samples.ndim > 1:
        samples = samples.reshape(-1)

    length = samples.shape[0]
    bin_size = max(1, length // num_bins)
    emb: List[float] = []
    for i in range(num_bins):
        start_i = i * bin_size
        end_i = (i + 1) * bin_size if i < num_bins - 1 else length
        if start_i >= length:
            emb.append(0.0)
            continue
        segment = samples[start_i:end_i]
        emb.append(float(np.mean(np.abs(segment))))

    norm = float(np.linalg.norm(emb))
    if norm > 1e-6:
        emb = [v / norm for v in emb]
    return emb


async def _transcribe_once(duration_s: float, device_index: int) -> Tuple[str, Optional[List[float]]]:
    """Capture microphone audio for `duration_s` and return (transcript, embedding)."""
    region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-west-2"
    client = TranscribeStreamingClient(region=region)
    stream = await client.start_stream_transcription(
        language_code=LANGUAGE_CODE,
        media_sample_rate_hz=RATE,
        media_encoding="pcm",
    )

    final_text: str = ""
    all_frames: List[np.ndarray] = []

    async def mic() -> None:
        nonlocal stream, all_frames
        frames_per_chunk = int(RATE * 0.2)  # 0.2s chunks
        start = time.time()
        with sd.InputStream(
            samplerate=RATE,
            channels=CHANNELS,
            dtype="int16",
            device=device_index,
        ) as s:
            while time.time() - start < duration_s:
                frames, _overflowed = s.read(frames_per_chunk)
                all_frames.append(frames.copy())
                audio_bytes = frames.tobytes()
                await stream.input_stream.send_audio_event(audio_chunk=audio_bytes)
        await stream.input_stream.end_stream()

    async def listener() -> None:
        nonlocal final_text
        async for event in stream.output_stream:
            if not hasattr(event, "transcript"):
                continue
            for result in event.transcript.results:
                if result.is_partial:
                    continue
                final_text = "".join(a.transcript for a in result.alternatives)

    await asyncio.gather(mic(), listener())

    transcript = final_text.strip()
    embedding: Optional[List[float]] = None
    if all_frames:
        concat = np.concatenate(all_frames, axis=0).astype(np.float32)
        embedding = _compute_embedding_from_samples(concat)

    return transcript, embedding


def transcribe_once(duration_s: float, device_index: int) -> Tuple[str, Optional[List[float]]]:
    """Synchronous wrapper around _transcribe_once."""
    try:
        return asyncio.run(_transcribe_once(duration_s, device_index))
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        logger.warning("Transcribe failed: %s", exc)
        return "", None


# ---------------------------------------------------------------------------
# TTS via Polly + sounddevice
# ---------------------------------------------------------------------------

def synthesize_and_play(
    text: str,
    voice_id: str,
    output_device: int,
    region_name: Optional[str] = None,
    sample_rate: int = RATE,
) -> float:
    """Use Polly to synthesize `text` and play it via sounddevice (non-blocking).

    Returns an approximate playback duration in seconds.
    """
    if not text:
        return 0.0

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

    logger.info("[AGENT_VOICE_PLAY] %s", text)
    logger.debug("Playing %d samples at %d Hz on device %s", len(data), sample_rate, output_device)
    sd.play(data, samplerate=sample_rate, device=output_device)
    play_sec = len(data) / float(sample_rate)
    return play_sec


# ---------------------------------------------------------------------------
# HTTP helper to call the broker
# ---------------------------------------------------------------------------

def call_broker(
    broker_url: str,
    session_id: str,
    transcript: str,
    channel: str = "audio",
    voice_id: Optional[int] = None,
    embedding: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Send a transcript + optional embedding to the broker and return the JSON response."""
    import urllib.request
    import urllib.error

    payload: Dict[str, Any] = {
        "transcript": transcript,
        "channel": channel,
    }
    if voice_id is not None:
        payload["voice_id"] = str(voice_id)
    if embedding is not None:
        payload["voice_embedding"] = embedding

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
# Reply filtering: separate spoken content from debug/info
# ---------------------------------------------------------------------------

def split_speech_and_debug(text: str) -> Tuple[str, str]:
    """Split agent reply into (spoken_text, debug_text)."""
    if not text:
        return "", ""

    lines = text.splitlines()
    speech_lines: List[str] = []
    debug_lines: List[str] = []

    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()

        is_debug = False

        if lower.startswith("stored memory:"):
            is_debug = True
        if lower.startswith("i now have the following memories"):
            is_debug = True
        if stripped.startswith("{") and stripped.endswith("}"):
            is_debug = True
        if "\"item_type\": \"MEMORY\"" in stripped or "'item_type': 'MEMORY'" in stripped:
            is_debug = True

        if debug_lines and stripped.startswith(("* ", "- ")):
            is_debug = True

        if is_debug:
            debug_lines.append(line)
        else:
            speech_lines.append(line)

    speech_text = "\n".join(speech_lines).strip()
    debug_text = "\n".join(debug_lines).strip()
    return speech_text, debug_text


# ---------------------------------------------------------------------------
# Echo suppression
# ---------------------------------------------------------------------------

def is_echo(utter: str, last_agent: str) -> bool:
    """Return True if `utter` looks like an echo of `last_agent`.

    More aggressive than before:
    - If utter == last_agent -> echo.
    - If utter is a non-trivial substring (>=10 chars) of last_agent -> echo.
    """
    if not utter or not last_agent:
        return False

    u = utter.lower().strip()
    a = last_agent.lower().strip()
    if not u or not a:
        return False

    if u == a:
        return True

    # Non-trivial substring (catch tails like "I'll store this as a fact memory.")
    if len(u) >= 10 and u in a:
        return True
    if len(a) >= 10 and a in u:
        return True

    return False


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

    # Track which speaker we think is active, based on voice embeddings.
    current_speaker_name: Optional[str] = None
    last_ack_speaker_name: Optional[str] = None

    last_agent_speech: str = ""
    agent_playing_until: float = 0.0

    while True:
        try:
            print("Speak now...")
            utter, embedding = transcribe_once(duration_s=args.utterance_seconds, device_index=in_dev)

            now = time.time()

            if not utter:
                print("(No speech detected)")
                continue

            logger.info("[MIC] %s", utter)
            if args.verbose:
                print(f"[YOU] {utter}")

            lower = utter.lower()

            # Stop command: interrupt agent speech and resume listening.
            if (
                "marvin stop" in lower
                or "marvin, stop" in lower
                or "marvin please stop" in lower
                or "marvin, please stop" in lower
            ):
                sd.stop()
                agent_playing_until = 0.0
                print("[system] Stopped agent speech.")
                continue

            # If we're still in the playback window, almost certainly echo.
            if now < agent_playing_until:
                logger.info("[DURING_PLAYBACK_IGNORED] %s", utter)
                if args.verbose:
                    print("[system] Ignoring utterance during agent playback window.")
                continue

            # If utter looks like the last agent speech, treat as echo even if playback ended.
            if last_agent_speech and is_echo(utter, last_agent_speech):
                logger.info("[ECHO_IGNORED] %s", utter)
                if args.verbose:
                    print("[system] Ignoring echo of agent speech (similar text).")
                continue

            # Try to resolve the speaker identity from the voice embedding.
            pending_greet_name: Optional[str] = None
            if embedding is not None:
                try:
                    resolved_name, is_new = identity.resolve_voice(
                        embedding=embedding,
                        voice_id=None,
                        claimed_name=None,
                    )
                except Exception as exc:
                    logger.error("Voice resolution failed: %s", exc)
                    resolved_name, is_new = None, False

                if resolved_name is None and is_new:
                    # Unknown speaker: prompt for a name and enroll.
                    prompt_text = "I don't recognize your voice yet. What should I call you?"
                    logger.info("[VOICE_ENROLL_PROMPT]")

                    # 1. Speak the prompt
                    play_sec = synthesize_and_play(
                        text=prompt_text,
                        voice_id=voice_id,
                        output_device=out_dev,
                        region_name=os.getenv("AWS_REGION") or os.getenv("REGION"),
                    )

                    # 2. Wait for playback to finish before listening for the name
                    #    Add a small margin to be safe against timing drift.
                    time.sleep(play_sec + 0.1)

                    # 3. Now capture the user's spoken name
                    name_utter, name_embedding = transcribe_once(
                        duration_s=3.0,
                        device_index=in_dev,
                    )


                    # Name enrollment state: run heuristic + optional LLM to extract a self-name.
                    heuristic_name = extract_name_heuristic(name_utter or "")
                    llm_name = llm_extract_name(name_utter or "")
                    final_name = heuristic_name or llm_name

                    logger.info(
                        "[NAME_ENROLL] stt=%r heuristic=%r llm=%r final=%r",
                        name_utter,
                        heuristic_name,
                        llm_name,
                        final_name,
                    )
                    if args.verbose:
                        print(
                            f"[DEBUG] name_enroll stt={name_utter!r} "
                            f"heuristic={heuristic_name!r} llm={llm_name!r} final={final_name!r}"
                        )

                    if final_name:
                        claimed_name = final_name
                        try:
                            # Use name clip embedding if available, else fall back to original.
                            resolved_name, _ = identity.resolve_voice(
                                embedding=name_embedding or embedding,
                                voice_id=None,
                                claimed_name=claimed_name,
                            )
                        except Exception as exc:
                            logger.error("Failed to register new voice profile: %s", exc)
                            resolved_name = claimed_name
                    else:
                        # No usable name; keep resolved_name as None so we'll try again later.
                        resolved_name = None

                if resolved_name:
                    current_speaker_name = resolved_name
                    logger.info("[VOICE_IDENTITY] resolved speaker=%s", resolved_name)

            # Set up one-time greeting when the active speaker changes.
            if current_speaker_name and current_speaker_name != last_ack_speaker_name:
                pending_greet_name = current_speaker_name

            # Normal path: send to broker
            resp = call_broker(
                broker_url=args.broker_url,
                session_id=args.session,
                transcript=utter,
                channel="audio",
                voice_id=in_dev,
                embedding=embedding,
            )
            full_reply = resp.get("reply_text") or ""
            speech_text, debug_text = split_speech_and_debug(full_reply)

            if not speech_text and full_reply:
                speech_text = full_reply

            # Prepend greeting once per new speaker.
            if pending_greet_name and speech_text:
                speech_text = f"{pending_greet_name}, {speech_text}"
                last_ack_speaker_name = pending_greet_name

            last_agent_speech = speech_text or ""

            logger.info("[AGENT] %s", speech_text)
            if args.verbose:
                print(f"[AGENT] {speech_text}")
                if debug_text:
                    print("[DEBUG]")
                    print(debug_text)
            else:
                print(f"[AGENT] {speech_text}")

            play_sec = synthesize_and_play(
                text=speech_text,
                voice_id=voice_id,
                output_device=out_dev,
                region_name=os.getenv("AWS_REGION") or os.getenv("REGION"),
            )
            agent_playing_until = time.time() + play_sec

        except KeyboardInterrupt:
            print("\nExiting audio loop.")
            break


def text_loop(args: argparse.Namespace) -> None:
    """Simple text-only REPL that talks to the broker, with debug filtering."""
    print("Text-only mode. Ctrl+D or Ctrl+C to exit.")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not line:
            continue

        logger.info("[MIC_TEXT] %s", line)

        resp = call_broker(
            broker_url=args.broker_url,
            session_id=args.session,
            transcript=line,
            channel="cli",
        )
        reply_text = resp.get("reply_text") or ""
        speech_text, debug_text = split_speech_and_debug(reply_text)

        if not speech_text and reply_text:
            speech_text = reply_text

        logger.info("[AGENT_TEXT] %s", speech_text)

        print(speech_text)
        if args.verbose and debug_text:
            print("[DEBUG]")
            print(debug_text)


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
        default=6.0,
        help="Maximum length of each recorded utterance in seconds (upper bound).",
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
    configure_logging(args.verbose)

    if args.text_only:
        text_loop(args)
    else:
        audio_loop(args)


if __name__ == "__main__":
    main()
