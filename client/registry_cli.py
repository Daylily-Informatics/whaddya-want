#!/usr/bin/env python3
"""Command-line manager for the local AI identity registry.

This tool manages the registry stored at ``~/.whaddya/registry.json`` via the
``client.identity`` helpers. It supports listing, inspecting, adding/updating,
and deleting voice and face profiles.

Examples
--------
List all enrolled profiles as a table:
    python client/registry_cli.py list

Add or update a profile from embeddings:
    python client/registry_cli.py add --name "Ada" --voice-npy ada_voice.npy --face-npy ada_face.npy

Enroll from raw assets (requires optional dependencies):
    python client/registry_cli.py add --name "Ada" --voice-audio ada.wav --face-image ada.jpg

Inspect a single entry as JSON:
    python client/registry_cli.py show --name Ada --json

Delete a face profile:
    python client/registry_cli.py delete --name Ada --type face
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from client import identity
from client.speaker_id import SpeakerEmbedder


def _resample_mono(audio: np.ndarray, src_rate: int, target_rate: int = 16000) -> np.ndarray:
    """Lightweight resampler using linear interpolation."""
    if src_rate == target_rate:
        return audio.astype(np.float32)
    duration = len(audio) / float(src_rate)
    t_old = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    t_new = np.linspace(0.0, duration, num=int(duration * target_rate), endpoint=False)
    return np.interp(t_new, t_old, audio).astype(np.float32)


def _load_audio_mono_16k(path: Path) -> np.ndarray:
    """Load audio as mono 16 kHz float32 samples."""
    if importlib.util.find_spec("soundfile") is not None:
        import soundfile as sf

        data, rate = sf.read(path)
    else:
        import wave

        with wave.open(path, "rb") as wf:
            rate = wf.getframerate()
            channels = wf.getnchannels()
            frames = wf.readframes(wf.getnframes())
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        if channels > 1:
            data = data.reshape(-1, channels).mean(axis=1)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return _resample_mono(np.asarray(data, dtype=np.float32), rate)


def _load_face_from_image(path: Path) -> np.ndarray:
    if importlib.util.find_spec("face_recognition") is None:
        raise SystemExit("face_recognition is required for --face-image inputs.")
    import face_recognition

    image = face_recognition.load_image_file(path)
    encodings = face_recognition.face_encodings(image)
    if not encodings:
        raise SystemExit(f"No faces found in {path}")
    if len(encodings) > 1:
        print(
            f"[warn] Multiple faces found in {path}; using the first encoding.",
            file=sys.stderr,
        )
    return np.asarray(encodings[0], dtype=np.float32)


def _load_npy_vector(path: Path) -> np.ndarray:
    vec = np.load(path)
    return np.asarray(vec, dtype=np.float32)


def _list_entries(_: argparse.Namespace) -> None:
    entries = identity.list_entries()
    if _.json:
        print(json.dumps(entries, indent=2))
        return
    if not entries:
        print("No profiles are currently enrolled.")
        return
    print(f"{'Name':20} {'Type':8} Details")
    print("-" * 60)
    for e in entries:
        details: List[str] = []
        if e.get("voice"):
            details.append("voice")
        if e.get("face"):
            details.append("face")
        print(f"{(e.get('name') or ''):20} {(e.get('type') or ''):8} {', '.join(details) or '—'}")


def _show_entry(args: argparse.Namespace) -> None:
    entries = identity.list_entries()
    match = next(
        (e for e in entries if (e.get("name") or "").lower() == args.name.strip().lower()),
        None,
    )
    if match is None:
        raise SystemExit(f"No entry found for '{args.name}'.")
    if args.json:
        print(json.dumps(match, indent=2))
    else:
        details = [
            f"type: {match.get('type')}",
            "voice: present" if match.get("voice") else "voice: none",
            "face: present" if match.get("face") else "face: none",
        ]
        print(f"Entry for {match.get('name')}:\n  " + "\n  ".join(details))


def _add_or_update(args: argparse.Namespace) -> None:
    name = args.name.strip()
    voice_vec: Optional[np.ndarray] = None
    face_vec: Optional[np.ndarray] = None

    if args.voice_npy:
        voice_vec = _load_npy_vector(Path(args.voice_npy))
    elif args.voice_audio:
        embedder = SpeakerEmbedder()
        if not embedder.enabled:
            raise SystemExit("Speaker embedding model is unavailable; install speechbrain/torch.")
        audio = _load_audio_mono_16k(Path(args.voice_audio))
        voice_vec = embedder.embed(audio)
        if voice_vec is None:
            raise SystemExit("Failed to compute speaker embedding from audio.")

    if args.face_npy:
        face_vec = _load_npy_vector(Path(args.face_npy))
    elif args.face_image:
        face_vec = _load_face_from_image(Path(args.face_image))

    if voice_vec is None and face_vec is None:
        raise SystemExit("Provide at least one of --voice-* or --face-* to enroll.")

    if voice_vec is not None:
        identity.enroll_voice(name, voice_vec)
        print(f"Enrolled voice for '{name}'.")
    if face_vec is not None:
        identity.enroll_face(name, face_vec)
        print(f"Enrolled face for '{name}'.")


def _delete_entry(args: argparse.Namespace) -> None:
    name = args.name.strip()
    types = [args.type] if args.type in {"voice", "face"} else ["voice", "face"]
    removed: List[str] = []
    for kind in types:
        if kind == "voice" and identity.delete_voice(name):
            removed.append("voice")
        elif kind == "face" and identity.delete_face(name):
            removed.append("face")
    if not removed:
        raise SystemExit(f"No matching profiles found for '{name}'.")
    print(f"Removed {', '.join(removed)} profile(s) for '{name}'.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage voice and face registrations stored in ~/.whaddya/registry.json",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List all registered profiles")
    p_list.add_argument("--json", action="store_true", help="Output raw registry JSON")
    p_list.set_defaults(func=_list_entries)

    p_show = sub.add_parser("show", help="Show one profile by name")
    p_show.add_argument("--name", required=True, help="Name to inspect (case-insensitive)")
    p_show.add_argument("--json", action="store_true", help="Output the entry as JSON")
    p_show.set_defaults(func=_show_entry)

    p_add = sub.add_parser("add", help="Add or update a profile with voice/face data")
    p_add.add_argument("--name", required=True, help="Profile name")
    p_add.add_argument("--voice-npy", help="Path to a .npy speaker embedding")
    p_add.add_argument("--voice-audio", help="Path to a WAV/FLAC audio file for embedding")
    p_add.add_argument("--face-npy", help="Path to a .npy face embedding")
    p_add.add_argument("--face-image", help="Path to an image file for face encoding")
    p_add.set_defaults(func=_add_or_update)

    p_del = sub.add_parser("delete", help="Delete voice/face data for a profile")
    p_del.add_argument("--name", required=True, help="Profile name to remove")
    p_del.add_argument(
        "--type",
        choices=["voice", "face", "both"],
        default="both",
        help="Which profile elements to delete",
    )
    p_del.set_defaults(func=_delete_entry)

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler: Callable[[argparse.Namespace], None] = getattr(args, "func")
    handler(args)


if __name__ == "__main__":
    main()
