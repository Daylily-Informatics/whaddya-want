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
