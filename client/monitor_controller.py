from __future__ import annotations
import subprocess
import sys
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

from client import monitor as monitor_module
from client.shared_audio import AudioPlayer


@dataclass
class MonitorConfig:
    broker_url: str
    session: str
    voice_id: Optional[str]
    voice_mode: str
    camera_index: Optional[int]
    mic_index: Optional[int]
    speaker_index: Optional[int]


class MonitorController:
    """Manage lifecycle of the monitor as a companion to the voice client."""

    def __init__(self, player: AudioPlayer, playback_mute: threading.Event):
        self._player = player
        self._playback_mute = playback_mute
        self._config: MonitorConfig | None = None
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._process: subprocess.Popen | None = None

    def configure(self, config: MonitorConfig) -> None:
        self._config = replace(config)

    def launch(self, config: MonitorConfig | None = None) -> None:
        if config is not None:
            self.configure(config)
        if self._config is None:
            raise RuntimeError("Monitor configuration not set.")

        self.stop()

        if sys.platform == "darwin":
            self._launch_subprocess()
        else:
            self._launch_thread()

    def stop(self) -> None:
        if self._thread is not None:
            if self._stop_event is not None:
                self._stop_event.set()
            self._thread.join(timeout=5)
        self._thread = None
        self._stop_event = None

        if self._process is not None:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            finally:
                self._process = None

    # ---- internals ----
    def _launch_thread(self) -> None:
        assert self._config is not None
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=monitor_module.start_monitor,
            kwargs=dict(
                broker_url=self._config.broker_url,
                session=self._config.session,
                voice=self._config.voice_id or "",
                voice_mode=self._config.voice_mode,
                camera_index=self._config.camera_index,
                mic_index=self._config.mic_index,
                speaker_index=self._config.speaker_index,
                stop_event=self._stop_event,
                player=self._player,
                playback_mute=self._playback_mute,
            ),
            daemon=True,
        )
        self._thread.start()
        print("[monitor] started (press 'q' in its window to quit).")

    def _launch_subprocess(self) -> None:
        assert self._config is not None
        monitor_path = Path(__file__).resolve().parent / "monitor.py"
        args = [
            sys.executable or "python3",
            str(monitor_path),
            "--broker-url",
            self._config.broker_url,
            "--session",
            self._config.session,
            "--voice",
            self._config.voice_id or "",
            "--voice-mode",
            self._config.voice_mode,
        ]
        if self._config.camera_index is not None:
            args += ["--camera-index", str(self._config.camera_index)]
        if self._config.mic_index is not None:
            args += ["--mic-index", str(self._config.mic_index)]
        if self._config.speaker_index is not None:
            args += ["--speaker-index", str(self._config.speaker_index)]

        self._process = subprocess.Popen(args)
        print(f"[monitor] subprocess started with pid={self._process.pid} (macOS main-thread GUI)")
