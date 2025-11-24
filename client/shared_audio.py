#!/usr/bin/env python3
"""Shared abortable MP3 player for Polly output (used by cli + monitor)."""
from __future__ import annotations
import asyncio, io, sys, threading
import numpy as np


# Global audio event loop
_audio_loop = asyncio.new_event_loop()

def _run_audio_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()
    
_audio_thread = threading.Thread(target=_run_audio_loop, args=(_audio_loop,), daemon=True)
_audio_thread.start()

_MP3_READY = False
try:
    from pydub import AudioSegment  # type: ignore
    _MP3_READY = True
except Exception as e:
    print("[diag] pydub import failed:", repr(e), file=sys.stderr)
    _MP3_READY = False

import sounddevice as sd

class AudioPlayer:
    def __init__(self, mute_guard=None) -> None:
        self.enabled = _MP3_READY
        self._warned = False
        self._mute_guard = mute_guard
        self._abort_evt = asyncio.Event()

    def _decode(self, data: bytes):
        if not self.enabled:
            return (None, None)
        try:
            seg = AudioSegment.from_file(io.BytesIO(data), format="mp3")
        except Exception as exc:
            if not self._warned:
                print("[diag] unable to decode mp3 — install ffmpeg/libav for pydub:", exc, file=sys.stderr)
                self._warned = True
            return (None, None)
        samples = np.array(seg.get_array_of_samples())
        if seg.channels > 1:
            samples = samples.reshape((-1, seg.channels))
        else:
            samples = samples.reshape((-1, 1))
        scale = float(1 << (8 * seg.sample_width - 1))
        audio = samples.astype(np.float32) / scale
        return audio, int(seg.frame_rate)

    async def play(self, data: bytes) -> bool:
        if not self.enabled:
            return False
        if self._mute_guard is not None:
            self._mute_guard.set()
        loop = asyncio.get_running_loop()
        self._abort_evt.clear()
        try:
            audio, rate = await loop.run_in_executor(None, self._decode, data)
            if audio is None or rate is None:
                return False
            sd.play(audio, rate)
            duration = float(audio.shape[0]) / float(rate)
            try:
                await asyncio.wait_for(self._abort_evt.wait(), timeout=duration + 0.25)
            except asyncio.TimeoutError:
                pass
            finally:
                sd.stop()
                await asyncio.sleep(0.05)
            if self._mute_guard is not None:
                await asyncio.sleep(0.15)
        finally:
            if self._mute_guard is not None:
                self._mute_guard.clear()
        return True

    def stop(self) -> None:
        try:
            self._abort_evt.set()
        except Exception:
            pass



import base64, json, requests  # at top with other imports
from typing import Optional, Dict, Any

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
) -> Optional[Dict[str, Any]]:
    """Synchronous helper: POST text to broker and play any returned audio on the global audio loop."""
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

    try:
        r = requests.post(
            broker_url,
            json=payload,
            timeout=timeout,
            headers={"X-Client-Session": session_id},
        )
        if not r.ok:
            print(f"[speech] broker HTTP {r.status_code} {r.text}", file=sys.stderr)
            return None
        body = r.json()
    except Exception as e:
        print(f"[speech] broker call error: {e}", file=sys.stderr)
        return None

    audio_b64 = (body.get("audio") or {}).get("audio_base64")
    if audio_b64 and not text_only:
        data = base64.b64decode(audio_b64)
        try:
            asyncio.run_coroutine_threadsafe(player.play(data), _audio_loop)
        except Exception as e:
            print(f"[speech] playback error: {e}", file=sys.stderr)

    return body
