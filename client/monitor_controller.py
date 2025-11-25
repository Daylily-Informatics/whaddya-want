from __future__ import annotations
import asyncio
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import cv2
import sounddevice as sd

from client import monitor as monitor_module
from client.shared_audio import AudioPlayer, speak_via_broker


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

    def __init__(
        self, player: AudioPlayer, playback_mute: threading.Event, loop: asyncio.AbstractEventLoop | None = None
    ):
        self._player = player
        self._playback_mute = playback_mute
        self._config: MonitorConfig | None = None
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._process: subprocess.Popen | None = None
        self._loop = loop
        self._task: asyncio.Task | None = None

    def configure(self, config: MonitorConfig) -> None:
        self._config = replace(config)

    def launch(self, config: MonitorConfig | None = None) -> None:
        if config is not None:
            self.configure(config)
        if self._config is None:
            raise RuntimeError("Monitor configuration not set.")

        self.stop()

        if self._loop is not None:
            self._launch_main_thread()
        elif sys.platform == "darwin":
            self._launch_subprocess()
        else:
            self._launch_thread()

    def stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=5)
        self._thread = None

        if self._task is not None:
            self._task.cancel()
            self._task = None

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
    def _launch_main_thread(self) -> None:
        assert self._config is not None
        if self._loop is None:
            raise RuntimeError("Asyncio loop is required for main-thread monitor mode.")
        self._stop_event = threading.Event()
        self._task = self._loop.create_task(self._run_main_thread_monitor())
        print("[monitor] started on main thread (async OpenCV loop).")

    async def _run_main_thread_monitor(self) -> None:
        assert self._config is not None

        args = monitor_module.build_args(
            broker_url=self._config.broker_url,
            session=self._config.session,
            voice=self._config.voice_id or "",
            voice_mode=self._config.voice_mode,
            camera_index=self._config.camera_index,
            mic_index=self._config.mic_index,
            speaker_index=self._config.speaker_index,
        )

        cam_s, mic_s, spk_s = monitor_module.load_saved_devices()
        cam_idx = args.camera_index if args.camera_index is not None else (cam_s if cam_s is not None else 0)
        mic_idx = args.mic_index if args.mic_index is not None else mic_s
        spk_idx = args.speaker_index if args.speaker_index is not None else spk_s

        try:
            if mic_idx is not None or spk_idx is not None:
                sd.default.device = (mic_idx, spk_idx)
        except Exception:
            pass

        stop_signal = self._stop_event or threading.Event()

        events: "queue.Queue[tuple]" = queue.Queue()
        threading.Thread(
            target=monitor_module.asr_listener,
            args=(events, stop_signal, mic_idx, self._playback_mute),
            daemon=True,
        ).start()

        if self._loop is None:
            raise RuntimeError("Event loop required for monitor async run")
        monitor_events: asyncio.Queue[str] = asyncio.Queue()

        def _emit_text(txt: str):
            self._loop.call_soon_threadsafe(monitor_events.put_nowait, txt)

        engine = monitor_module.MonitorEngine(
            args,
            player=self._player,
            playback_mute=self._playback_mute,
            stop_event=stop_signal,
            event_callback=_emit_text,
            defer_name_capture=True,
        )

        cap = cv2.VideoCapture(cam_idx)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {cam_idx}")
        print("[monitor] ready — press 'q' to quit.")

        window_title = "Marvin Monitor (press q to quit)"
        display_enabled = True
        try:
            if (threading.current_thread() is not threading.main_thread()) and sys.platform.startswith("linux"):
                cv2.startWindowThread()
            cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        except cv2.error as e:
            display_enabled = False
            print(
                "[monitor] GUI display unavailable; running headless (no preview window). "
                f"OpenCV error: {e}",
                file=sys.stderr,
            )

        try:
            while not stop_signal.is_set():
                try:
                    while True:
                        kind, val = events.get_nowait()
                        if kind == "cmd" and engine.handle_command(val):
                            break
                except queue.Empty:
                    pass

                ret, frame = cap.read()
                if not ret:
                    print("[monitor] frame grab failed")
                    break

                result = engine.process_frame(frame)
                frame = result.frame

                while not monitor_events.empty():
                    text = await monitor_events.get()
                    await speak_via_broker(
                        broker_url=args.broker_url,
                        session_id=args.session,
                        text=text,
                        voice_id=args.voice,
                        voice_mode=args.voice_mode,
                        player=self._player,
                        playback_mute=self._playback_mute,
                        context={"intro_already_sent": True},
                        text_only=False,
                        timeout=30,
                        verbose=False,
                        barge_monitor=None,
                    )

                if result.needs_human_name_capture and result.human_unknown_target:
                    nm = await asyncio.to_thread(monitor_module.listen_for_name, mic_idx, 7.0)
                    engine.apply_name_capture_results(
                        human_name=nm,
                        human_unknown_target=result.human_unknown_target,
                        animal_captures=None,
                    )

                animal_captures: list[tuple[str, str, object]] = []
                for et, sig in result.animal_name_capture_targets or []:
                    nm = await asyncio.to_thread(monitor_module.listen_for_name, mic_idx, 7.0)
                    if nm:
                        animal_captures.append((et, nm, sig))
                if animal_captures:
                    engine.apply_name_capture_results(
                        human_name=None,
                        human_unknown_target=None,
                        animal_captures=animal_captures,
                    )

                if display_enabled:
                    try:
                        cv2.imshow(window_title, frame)
                        if (cv2.waitKey(1) & 0xFF) == ord("q"):
                            stop_signal.set()
                            break
                    except cv2.error:
                        display_enabled = False
                        print(
                            "[monitor] GUI display disabled after OpenCV error; continuing headless.",
                            file=sys.stderr,
                        )

                await asyncio.sleep(0)
        except asyncio.CancelledError:
            stop_signal.set()
            raise
        finally:
            stop_signal.set()
            try:
                cap.release()
            except Exception:
                pass
            if display_enabled:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass
            try:
                sd.stop()
            except Exception:
                pass

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
