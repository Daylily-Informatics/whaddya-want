#!/usr/bin/env python3
"""Shared broker + audio helpers for the CLI and monitor."""
from __future__ import annotations

import asyncio
import base64
import io
import json
import sys
import threading
from typing import Any, Dict, Optional, Callable, Awaitable, Tuple

import numpy as np
import requests

_audio_loop: asyncio.AbstractEventLoop | None = None
_audio_thread: threading.Thread | None = None
_audio_lock = threading.Lock()


def _run_audio_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


def _ensure_audio_loop() -> asyncio.AbstractEventLoop:
    global _audio_loop, _audio_thread
    if _audio_loop is None:
        with _audio_lock:
            if _audio_loop is None:
                loop = asyncio.new_event_loop()
                thread = threading.Thread(target=_run_audio_loop, args=(loop,), daemon=True)
                thread.start()
                _audio_loop = loop
                _audio_thread = thread
    assert _audio_loop is not None
    return _audio_loop

_MP3_READY = False
try:
    from pydub import AudioSegment  # type: ignore

    _MP3_READY = True
except Exception as e:
    print("[diag] pydub import failed:", repr(e), file=sys.stderr)
    _MP3_READY = False

import sounddevice as sd  # noqa: E402


class AudioPlayer:
    def __init__(self, mute_guard=None) -> None:
        self.enabled = _MP3_READY
        self._warned = False
        self._mute_guard = mute_guard
        # Thread-safe abort flag signalled from any thread
        self._abort_flag = threading.Event()

    def _resample(self, audio: np.ndarray, src_rate: int) -> tuple[np.ndarray, int]:
        """Resample to the output device's preferred rate to avoid crackle."""

        try:
            target_rate = int(sd.query_devices(kind="output")["default_samplerate"])
        except Exception:
            target_rate = src_rate

        if target_rate <= 0 or target_rate == src_rate:
            return audio, src_rate

        frames = audio.shape[0]
        new_frames = int(round(frames * float(target_rate) / float(src_rate)))
        if new_frames <= 0:
            return audio, src_rate

        # Linear interpolation resample keeps dependencies light and works for mono/stereo.
        x_old = np.linspace(0.0, 1.0, frames, endpoint=False, dtype=np.float64)
        x_new = np.linspace(0.0, 1.0, new_frames, endpoint=False, dtype=np.float64)
        if audio.ndim == 1 or audio.shape[1] == 1:
            resampled = np.interp(x_new, x_old, audio.reshape(-1)).astype(np.float32)
            resampled = resampled.reshape(-1, 1)
        else:
            resampled = np.stack(
                [np.interp(x_new, x_old, audio[:, ch]) for ch in range(audio.shape[1])],
                axis=1,
            ).astype(np.float32)
        return resampled, target_rate

    def _decode(self, data: bytes):
        if not self.enabled:
            return (None, None)
        try:
            seg = AudioSegment.from_file(io.BytesIO(data), format="mp3")
        except Exception as exc:
            if not self._warned:
                print(
                    "[diag] unable to decode mp3 — install ffmpeg/libav for pydub:",
                    exc,
                    file=sys.stderr,
                )
                self._warned = True
            return (None, None)
        samples = np.array(seg.get_array_of_samples())
        if seg.channels > 1:
            samples = samples.reshape((-1, seg.channels))
        else:
            samples = samples.reshape((-1, 1))
        scale = float(1 << (8 * seg.sample_width - 1))
        audio = samples.astype(np.float32) / scale
        return self._resample(audio, int(seg.frame_rate))

    async def play(self, data: bytes) -> bool:
        if not self.enabled:
            return False
        if self._mute_guard is not None:
            self._mute_guard.set()
        loop = asyncio.get_running_loop()
        self._abort_flag.clear()
        try:
            audio, rate = await loop.run_in_executor(None, self._decode, data)
            if audio is None or rate is None:
                return False
            sd.play(audio, rate)
            duration = float(audio.shape[0]) / float(rate)
            # Poll the abort flag without touching asyncio primitives across threads.
            end_time = loop.time() + duration + 0.25
            while loop.time() < end_time and not self._abort_flag.is_set():
                await asyncio.sleep(0.05)
            sd.stop()
            await asyncio.sleep(0.05)
            if self._mute_guard is not None:
                # Give a small grace period where the mic capture thread ignores audio.
                await asyncio.sleep(0.15)
        finally:
            if self._mute_guard is not None:
                self._mute_guard.clear()
        return True

    def stop(self) -> None:
        # Can be called safely from any thread
        self._abort_flag.set()


async def _play_on_audio_loop(player: "AudioPlayer", data: bytes) -> bool:
    """Run player.play on the dedicated audio loop and await completion."""
    fut = asyncio.run_coroutine_threadsafe(player.play(data), _ensure_audio_loop())
    return await asyncio.wrap_future(fut)


# ---- Shared player / mute guard helpers ----
_shared_playback_mute: threading.Event | None = None
_shared_player: AudioPlayer | None = None


def get_shared_audio() -> Tuple[AudioPlayer, threading.Event]:
    """Return the process-wide audio player and playback mute guard.

    Both the CLI and monitor reuse these objects so that Marvin's voice output and
    the "Marvin is talking" signal stay consistent across subsystems.
    """

    global _shared_playback_mute, _shared_player
    if _shared_playback_mute is None:
        _shared_playback_mute = threading.Event()
    if _shared_player is None:
        _shared_player = AudioPlayer(mute_guard=_shared_playback_mute)
    return _shared_player, _shared_playback_mute


async def speak_via_broker(
    *,
    broker_url: str,
    session_id: str,
    text: str,
    voice_id: Optional[str],
    voice_mode: str,
    player: "AudioPlayer",
    playback_mute: threading.Event,
    context: Optional[Dict[str, Any]] = None,
    text_only: bool = False,
    timeout: int = 60,
    verbose: bool = False,
    barge_monitor: Optional[Callable[[], Awaitable[None]]] = None,
) -> Optional[Dict[str, Any]]:
    """Shared broker interaction: POST text and optionally play returned audio.

    The HTTP call is offloaded to a thread to avoid blocking event loops. Playback is
    always driven through the shared audio loop so callers from different threads
    and event loops share the same machinery.
    """

    payload: Dict[str, Any] = {
        "session_id": session_id,
        "text": text,
        "voice_mode": voice_mode,
    }
    if voice_id:
        payload["voice_id"] = voice_id
    if context:
        payload["context"] = context
    if text_only:
        payload["text_only"] = True

    if verbose:
        print(f"[diag] POST {broker_url}")
        print("[diag] payload=", json.dumps(payload, indent=2))

    try:
        r = await asyncio.to_thread(
            requests.post,
            broker_url,
            json=payload,
            timeout=timeout,
            headers={"X-Client-Session": session_id},
        )
    except requests.RequestException as e:
        print(f"[broker error] request failed: {getattr(e,'args',[repr(e)])[0]}", file=sys.stderr)
        return None

    if not r.ok:
        rid = r.headers.get("x-amzn-RequestId") or r.headers.get("x-amz-apigw-id") or "?"
        print(f"[broker error] HTTP {r.status_code} {r.reason}  request-id={rid}", file=sys.stderr)
        h_subset = {
            k: r.headers.get(k)
            for k in ["x-amzn-RequestId", "x-amz-apigw-id", "content-type", "date"]
            if r.headers.get(k)
        }
        if h_subset:
            print(f"[broker headers] {h_subset}", file=sys.stderr)
        try:
            errj = r.json()
            print("[broker body.json]", json.dumps(errj, indent=2), file=sys.stderr)
        except Exception:
            bt = (r.text or "").strip()
            if bt:
                print(
                    "[broker body.text]",
                    bt[:4000] + ("…(truncated)…" if len(bt) > 4000 else ""),
                    file=sys.stderr,
                )
        return None

    try:
        body = r.json()
    except Exception as e:
        print(f"[broker error] invalid JSON response: {e}", file=sys.stderr)
        return None

    audio_b64 = (body.get("audio") or {}).get("audio_base64")
    if audio_b64 and not text_only:
        data = base64.b64decode(audio_b64)

        async def _play():
            mon = asyncio.create_task(barge_monitor()) if barge_monitor else None
            try:
                return await _play_on_audio_loop(player, data)
            finally:
                if mon:
                    mon.cancel()

        try:
            await _play()
        except Exception as e:
            print(f"[speech] playback error: {e}", file=sys.stderr)

    return body


def say_via_broker_sync(
    *,
    broker_url: str,
    session_id: str,
    text: str,
    voice_id: Optional[str],
    voice_mode: str,
    player: "AudioPlayer",
    playback_mute: threading.Event,
    context: Optional[Dict[str, Any]] = None,
    text_only: bool = False,
    timeout: int = 60,
    verbose: bool = False,
) -> Optional[Dict[str, Any]]:
    """Synchronous wrapper around speak_via_broker for threading callers."""

    return asyncio.run(
        speak_via_broker(
            broker_url=broker_url,
            session_id=session_id,
            text=text,
            voice_id=voice_id,
            voice_mode=voice_mode,
            player=player,
            playback_mute=playback_mute,
            context=context,
            text_only=text_only,
            timeout=timeout,
            verbose=verbose,
        )
    )
