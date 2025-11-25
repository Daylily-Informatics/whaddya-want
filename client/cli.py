#!/usr/bin/env python3
from __future__ import annotations
import argparse, asyncio, contextlib, io, json, os, queue, re, signal, subprocess, sys, traceback, threading, uuid, warnings
from collections import deque
from pathlib import Path
from typing import AsyncGenerator, Optional, Tuple, Dict, Any

import numpy as np
import sounddevice as sd

warnings.filterwarnings("ignore", message="torchvision is not available")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptResultStream

# Shared identity + audio
from client import identity
from client import monitor as monitor_module
from client.shared_audio import AudioPlayer, speak_via_broker

# ---- Speaker embedding (SpeechBrain) ----
_SPK_READY = False
try:
    import torch  # noqa: F401
    from speechbrain.pretrained import EncoderClassifier  # type: ignore
    _SPK_READY = True
except Exception as e:
    print("[diag] speechbrain import error:", repr(e), file=sys.stderr)
    _SPK_READY = False

_SPK_DIR = Path(os.path.expanduser("~/.whaddya/speakers"))
_SPK_DIR.mkdir(parents=True, exist_ok=True)

# ---- Optional face recog for image test ----
_FACE_OK = False
try:
    import face_recognition
    _FACE_OK = True
except Exception:
    _FACE_OK = False

# ---- Persisted device choices ----
_STATE_DIR = Path(os.path.expanduser("~/.whaddya"))
_DEVICES_JSON = _STATE_DIR / "devices.json"
_STATE_DIR.mkdir(parents=True, exist_ok=True)

# ---- Finals-only handler ----
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

# ---- Mic → Transcribe ----
def _bytes_to_float_mono_int16(data: bytes, channels: int) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.int16)
    if channels > 1:
        arr = arr.reshape(-1, channels).mean(axis=1).astype(np.int16)
    return arr.astype(np.float32) / 32768.0

async def stream_microphone(*, region: str, language_code="en-US", sample_rate=16000, channels=1,
                            blocksize=4096, input_device: int|None=None,
                            analysis_buf: deque|None=None, mute_event: threading.Event|None=None,
                            verbose=False) -> AsyncGenerator[str, None]:
    client = TranscribeStreamingClient(region=region)
    stream = await client.start_stream_transcription(
        language_code=language_code, media_sample_rate_hz=sample_rate, media_encoding="pcm",
    )
    loop = asyncio.get_running_loop()
    raw_q: queue.Queue[bytes] = queue.Queue(maxsize=100)
    async_q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=100)
    def audio_cb(indata, frames, time_info, status):
        if status and verbose:
            print(status, file=sys.stderr)
        try:
            raw = bytes(indata)
            if mute_event is not None and mute_event.is_set():
                return
            if analysis_buf is not None:
                fmono = _bytes_to_float_mono_int16(raw, channels)
                analysis_buf.extend(fmono.tolist())
            raw_q.put_nowait(raw)
        except queue.Full:
            pass
    def mic_thread():
        with sd.RawInputStream(samplerate=sample_rate, blocksize=blocksize, dtype="int16",
                               channels=channels, device=input_device, callback=audio_cb):
            while True:
                chunk = raw_q.get()
                loop.call_soon_threadsafe(async_q.put_nowait, chunk)
    threading.Thread(target=mic_thread, daemon=True).start()
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

# ---- Helpers / commands ----
_REGISTER_RE = re.compile(
    r"""(?xi)
    \b(?:register|enrol+|in\s*rol+|and\s*rol+|role)\s+(?:my\s+voice|me)\s*
      (?:as\s+)?(?:(?:i'?m|i\s+am|this\s+is|says)\s+)?([A-Za-z][\w\s'\-]{0,31})\b
    | \bcall\s+me\s+([A-Za-z][\w\s'\-]{0,31})\b
    """
)
_EXIT_RE = re.compile(r"\b(exit|quit|stop listening|goodbye|stop now)\b", re.I)
_WAKE_RE = re.compile(r"\b(?:hey|ok|okay)\s+marvin\b", re.I)
_MONITOR_RE = re.compile(r"^\s*marvin[, ]+monitor\b", re.I)
_RESET_RE = re.compile(r"^\s*marvin[, ]+reset\b", re.I)
_HELP_RE = re.compile(r"^\s*marvin[, ]+help\b", re.I)
_DEVICE_CMD_RE = re.compile(
    r"""(?xi)
    \b(?:switch|change|set|use|select)\s+(?:the\s+)?
    (camera|mic|microphone|speaker|speakers|output)\s*
    (?:to|number|device)?\s*([\w-]+)
    """
)

_NUM_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

def parse_registration(text: str) -> Optional[str]:
    m = _REGISTER_RE.search(text)
    if not m:
        return None
    name = (m.group(1) or m.group(2) or "").strip()
    name = re.sub(r"\s+", " ", name)
    return name or None

def take_latest_seconds(buf: deque, seconds: float, rate: int) -> Optional[np.ndarray]:
    need = int(seconds * rate)
    if len(buf) < need:
        return None
    arr = np.array([buf[i] for i in range(len(buf) - need, len(buf))], dtype=np.float32)
    mx = float(np.max(np.abs(arr))) + 1e-9
    return arr / mx

# ---- Device helpers ----
def _camera_device_name(idx: int) -> Optional[str]:
    sysfs_name = Path(f"/sys/class/video4linux/video{idx}/name")
    try:
        return sysfs_name.read_text().strip()
    except OSError:
        return None


def _avfoundation_camera_names() -> dict[str, str]:
    """Return camera names on macOS using ffmpeg/avfoundation, if available."""
    if sys.platform != "darwin":
        return {}
    try:
        # ffmpeg prints avfoundation devices to stderr
        proc = subprocess.run(
            ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return {}

    names: dict[str, str] = {}
    collecting = False
    for line in (proc.stderr or "").splitlines():
        if "AVFoundation video devices" in line:
            collecting = True
            continue
        if collecting:
            m = re.search(r"\[(\d+)\]\s+(.*)", line)
            if m:
                names[m.group(1)] = m.group(2).strip()
            elif line.strip() == "":
                break
    return names

def _detect_cameras(max_index: int = 10) -> list[dict]:
    import cv2
    friendly_names = _avfoundation_camera_names()
    found=[]
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        ok = cap is not None and cap.isOpened()
        if ok:
            ret,_ = cap.read()
            cap.release()
            if ret:
                name = friendly_names.get(str(idx)) or _camera_device_name(idx)
                label = name or f"Camera {idx}"
                found.append({"index": str(idx), "label": label})
        elif cap is not None:
            cap.release()
    return found

def _describe_audio_devices():
    try:
        devs = sd.query_devices()
    except Exception as exc:
        raise RuntimeError(f"Unable to query audio devices: {exc}") from exc
    hostapis = sd.query_hostapis()
    mics, spk = [], []
    for idx, d in enumerate(devs):
        ha = hostapis[d["hostapi"]]["name"] if hostapis and 0 <= d.get("hostapi",-1) < len(hostapis) else ""
        label = f"{d['name']} ({ha})" if ha else d["name"]
        ent = {"index": str(idx), "label": label}
        if d.get("max_input_channels",0)>0: mics.append(ent)
        if d.get("max_output_channels",0)>0: spk.append(ent)
    return mics, spk

def _prompt_choice(options, title):
    print(f"[{title}]")
    for i,opt in enumerate(options, start=1):
        print(f"  {i}. {opt['label']} (index {opt['index']})")
    default_idx=1
    while True:
        try: raw=input(f"Select {title.lower()} [default {default_idx}]: ").strip()
        except EOFError: raw=""
        choice = default_idx if not raw else (int(raw) if raw.isdigit() else None)
        if choice and 1<=choice<=len(options):
            sel = int(options[choice-1]["index"])
            print(f"[selected] {options[choice-1]['label']}")
            return sel
        print("Invalid selection, try again.")

def device_setup_interactive() -> Tuple[int,int,int]:
    cams = _detect_cameras()
    if not cams: raise RuntimeError("No usable cameras detected.")
    cam = _prompt_choice(cams, "Available Cameras")
    mics, sp = _describe_audio_devices()
    if not mics: raise RuntimeError("No microphones detected.")
    mic = _prompt_choice(mics, "Available Microphones")
    sd.check_input_settings(device=mic, samplerate=16000, channels=1)
    with sd.InputStream(device=mic, channels=1, samplerate=16000) as s: s.read(1)
    print("[ok] microphone test passed.")
    if not sp: raise RuntimeError("No speakers detected.")
    spk = _prompt_choice(sp, "Available Speakers")
    sd.check_output_settings(device=spk, samplerate=16000, channels=1)
    import numpy as np
    dur=0.2; t=np.linspace(0,dur,int(16000*dur),False); tone=0.2*np.sin(2*np.pi*880*t)
    sd.play(tone, samplerate=16000, device=spk); sd.wait()
    print("[ok] speaker test passed.")
    _DEVICES_JSON.write_text(json.dumps({"camera_index":cam,"mic_index":mic,"speaker_index":spk}, indent=2))
    print(f"[saved] {_DEVICES_JSON}")
    return cam, mic, spk

def load_saved_devices() -> Tuple[int|None,int|None,int|None]:
    if _DEVICES_JSON.exists():
        try:
            d=json.loads(_DEVICES_JSON.read_text())
            return d.get("camera_index"), d.get("mic_index"), d.get("speaker_index")
        except Exception:
            return None,None,None
    return None,None,None


def _persist_devices(camera: int | None, mic: int | None, speaker: int | None) -> None:
    try:
        _DEVICES_JSON.write_text(
            json.dumps(
                {
                    "camera_index": camera,
                    "mic_index": mic,
                    "speaker_index": speaker,
                },
                indent=2,
            )
        )
    except Exception:
        pass


def parse_device_command(text: str) -> tuple[str, int] | None:
    m = _DEVICE_CMD_RE.search(text)
    if not m:
        return None
    raw_kind = m.group(1).lower()
    idx_token = (m.group(2) or "").strip().lower()
    idx: int | None
    if idx_token.isdigit():
        idx = int(idx_token)
    else:
        idx = _NUM_WORDS.get(idx_token)
    if idx is None:
        return None
    if raw_kind.startswith("cam"):
        kind = "camera"
    elif raw_kind.startswith("mic") or raw_kind.startswith("micro"):
        kind = "microphone"
    else:
        kind = "speaker"
    return kind, idx

# ---- Voice embedding model (for identity) ----
class SpeakerEmbed:
    def __init__(self):
        self.enabled = _SPK_READY
        self.model = None
    def _ensure(self):
        if not self.enabled: return
        if self.model is None:
            self.model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device":"cpu"},
            )
    def embed(self, wav16: np.ndarray) -> Optional[np.ndarray]:
        if not self.enabled: return None
        self._ensure()
        if self.model is None: return None
        with torch.no_grad():
            emb = self.model.encode_batch(torch.from_numpy(wav16).unsqueeze(0))
        return emb.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)

# ---- CLI loop ----
async def run() -> bool:
    ap = argparse.ArgumentParser(description="Voice client (unified identity + monitor trigger)")
    ap.add_argument("--broker-url", required=True)
    ap.add_argument("--session", default=str(uuid.uuid4()))
    ap.add_argument("--region", default=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-west-2")
    ap.add_argument("--language", default="en-US")
    ap.add_argument("--input-device", type=int, default=None)
    ap.add_argument("--setup-devices", action="store_true")
    ap.add_argument("--identify_image", type=str, default=None, help="Optional: identify a face from an image and exit")
    ap.add_argument("--rate", type=int, default=16000)
    ap.add_argument("--channels", type=int, default=1)
    ap.add_argument("--id-threshold", type=float, default=0.65)
    ap.add_argument("--id-window", type=float, default=2.0)
    ap.add_argument(
        "--auto-register-name",
        type=str,
        default=None,
        help="If set, automatically register an unknown speaker with this name.",
    )
    ap.add_argument("--force-enroll", default=None)
    ap.add_argument("--voice", default=os.getenv("POLLY_VOICE"))
    ap.add_argument("--voice-mode", choices=["standard","neural","generative"], default="standard")
    ap.add_argument("--text-only", action="store_true", help="Request text-only responses from the broker")
    ap.add_argument("--save-audio", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--self-voice-name",
        type=str,
        default=os.getenv("SELF_VOICE_NAME"),
        help="If set, ignore transcripts whose speaker ID matches this enrolled voice (self-talk suppression).",
    )
    args = ap.parse_args()

    # Face ID one-shot (shared memory)
    if args.identify_image:
        if not _FACE_OK:
            print("[face] face_recognition not available.", file=sys.stderr)
            sys.exit(2)
        img = face_recognition.load_image_file(args.identify_image)  # noqa: E999 (dash)
        locs = face_recognition.face_locations(img, model="hog")
        if not locs:
            print("No face found.")
            sys.exit(1)
        encs = face_recognition.face_encodings(img, locs, num_jitters=1)
        name = identity.identify_face(encs[0])
        print(f"Identity: {name or 'Unknown'}")
        sys.exit(0)

    voice_id = (args.voice or "").strip() or None
    self_voice = (args.self_voice_name or "").strip().lower() or None
    verbose = bool(args.verbose)
    save_audio = bool(args.save_audio)

    # HTTP wire logs toggle
    if "--debug-http" in sys.argv:
        try:
            import http.client as _http_client
            _http_client.HTTPConnection.debuglevel = 1
            import logging
            logging.basicConfig()
            logging.getLogger("urllib3").setLevel(logging.DEBUG)
            logging.getLogger("urllib3").propagate = True
            verbose = True
        except Exception:
            pass

    def vprint(*vargs, **vkwargs):
        if verbose: print(*vargs, **vkwargs)

    if voice_id: vprint(f"[diag] Polly voice → {voice_id}")
    vprint(f"[diag] voice mode = {args.voice_mode}")

    cam_idx_s, mic_idx_s, spk_idx_s = load_saved_devices()
    cam_idx = cam_idx_s
    mic_idx = args.input_device if args.input_device is not None else mic_idx_s
    spk_idx = spk_idx_s
    if args.setup_devices or cam_idx is None or mic_idx is None or spk_idx is None:
        try:
            cam_idx, mic_idx, spk_idx = device_setup_interactive()
        except Exception as e:
            print(f"[device setup error] {e}", file=sys.stderr)
    try:
        if mic_idx is not None or spk_idx is not None:
            sd.default.device = (mic_idx, spk_idx)
    except Exception:
        pass
    print(f"Session {args.session}  region={args.region}  devices: camera={cam_idx} mic={mic_idx} spk={spk_idx}")

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    playback_mute = threading.Event()
    player = AudioPlayer(mute_guard=playback_mute)
    for sig in (signal.SIGINT, signal.SIGTERM):
        try: loop.add_signal_handler(sig, lambda s=sig: (player.stop(), stop.set()))
        except NotImplementedError: pass

    analysis_buf = deque(maxlen=int(args.rate * 6))
    spk_embed = SpeakerEmbed()
    auto_register_name = (args.auto_register_name or args.force_enroll or "").strip()
    help_lines = [
        "Say 'hey Marvin' or 'ok Marvin' to enable command mode for 8 seconds.",
        "While command mode is active, say 'exit' to stop the client.",
        "While command mode is active, say 'switch/change/set/use/select' the camera, microphone, or speaker to a device number.",
        "Say 'marvin monitor' to launch the monitor window.",
        "Say 'marvin reset' to close the monitor and restart the client.",
        "Say 'register my voice as <name>' or 'call me <name>' to save a voice profile.",
        "Say 'marvin help' to hear this list again.",
    ]

    intro_text = (
        "Hello, I'm Marvin. Say 'marvin help' any time to hear the available commands."
    )
    if not spk_embed.enabled:
        print(
            "[identity] Speaker embedding model unavailable; voice identification and auto-registration are disabled.",
            file=sys.stderr,
        )

    monitor_thread: threading.Thread | None = None
    monitor_stop: threading.Event | None = None
    mic: AsyncGenerator[str, None] | None = None
    mic_task: asyncio.Task[str] | None = None

    def _stop_monitor():
        nonlocal monitor_thread, monitor_stop
        if monitor_thread is None:
            return
        if monitor_stop is not None:
            monitor_stop.set()
        monitor_thread.join(timeout=5)
        monitor_thread = None
        monitor_stop = None

    def _launch_monitor():
        nonlocal monitor_thread, monitor_stop
        _stop_monitor()
        monitor_stop = threading.Event()
        try:
            monitor_thread = threading.Thread(
                target=monitor_module.start_monitor,
                kwargs=dict(
                    broker_url=args.broker_url,
                    session=args.session,
                    voice=voice_id or "",
                    voice_mode=args.voice_mode,
                    camera_index=cam_idx if cam_idx is not None else 0,
                    mic_index=mic_idx if mic_idx is not None else -1,
                    speaker_index=spk_idx if spk_idx is not None else -1,
                    stop_event=monitor_stop,
                    player=player,
                    playback_mute=playback_mute,
                ),
                daemon=True,
            )
            monitor_thread.start()
            print("[monitor] started (press 'q' in its window to quit).")
        except Exception as e:
            monitor_stop = None
            print(f"[monitor error] {e}", file=sys.stderr)

    print(f"AI:  {intro_text}")
    await speak_via_broker(
        broker_url=args.broker_url,
        session_id=args.session,
        text=intro_text,
        voice_id=voice_id,
        voice_mode=args.voice_mode,
        player=player,
        playback_mute=playback_mute,
        context=None,
        text_only=args.text_only,
        timeout=30,
        verbose=verbose,
        barge_monitor=None,
    )

    def _start_microphone(new_idx: int | None):
        nonlocal mic, mic_task, mic_idx
        mic = stream_microphone(
            region=args.region,
            language_code=args.language,
            sample_rate=args.rate,
            channels=args.channels,
            input_device=new_idx,
            analysis_buf=analysis_buf,
            mute_event=playback_mute,
            verbose=verbose,
        )
        mic_task = asyncio.create_task(mic.__anext__())
        mic_idx = new_idx

    async def _switch_microphone(new_idx: int) -> bool:
        nonlocal mic, mic_task, mic_idx
        prev_idx = mic_idx
        mic_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, StopAsyncIteration):
            await mic_task
        with contextlib.suppress(Exception):
            await mic.aclose()

        try:
            sd.check_input_settings(device=new_idx, samplerate=args.rate, channels=args.channels)
            with sd.InputStream(device=new_idx, channels=args.channels, samplerate=args.rate) as s:
                s.read(1)
        except Exception as exc:
            print(f"[voice] unable to switch microphone: {exc}", file=sys.stderr)
            _start_microphone(prev_idx)
            return False

        analysis_buf.clear()
        _start_microphone(new_idx)
        try:
            sd.default.device = (mic_idx, spk_idx)
        except Exception:
            pass
        _persist_devices(cam_idx, mic_idx, spk_idx)
        print(f"[voice] microphone switched to index {mic_idx}.")
        return True

    async def _switch_speaker(new_idx: int) -> bool:
        nonlocal spk_idx
        player.stop()
        with contextlib.suppress(Exception):
            sd.stop()
        try:
            sd.check_output_settings(device=new_idx, samplerate=args.rate, channels=1)
        except Exception as exc:
            print(f"[voice] unable to switch speaker: {exc}", file=sys.stderr)
            return False

        spk_idx = new_idx
        try:
            sd.default.device = (mic_idx, spk_idx)
        except Exception:
            pass
        _persist_devices(cam_idx, mic_idx, spk_idx)
        print(f"[voice] speaker switched to index {spk_idx}.")
        return True

    async def _handle_command(cmd_spec: Dict[str, Any]) -> None:
        """
        Execute a structured command emitted by the broker.

        Expected shape:
            {"name": "launch_monitor" | "set_device" | "noop",
             "args": {...}}
        """
        nonlocal cam_idx, mic_idx, spk_idx

        if not isinstance(cmd_spec, dict):
            return
        name = cmd_spec.get("name")
        args = cmd_spec.get("args") or {}

        if not isinstance(name, str) or not isinstance(args, dict):
            return

        if name == "noop":
            return

        if name == "launch_monitor":
            _launch_monitor()
            return

        if name == "set_device":
            kind = args.get("kind")
            idx = args.get("index")
            try:
                idx = int(idx)
            except (TypeError, ValueError):
                return

            _stop_monitor()

            if kind == "camera":
                cam_idx = idx
                _persist_devices(cam_idx, mic_idx, spk_idx)
                print(f"[voice] camera switched to index {cam_idx} (via command).")
                return
            elif kind == "microphone":
                await _switch_microphone(idx)
                return
            elif kind == "speaker":
                await _switch_speaker(idx)
                return

            # unknown kind → ignore
            return

    # stdin watcher
    manual_q: asyncio.Queue[str] = asyncio.Queue()
    def _stdin_watcher():
        try:
            while True:
                line = sys.stdin.readline()
                if not line: break
                s = line.strip()
                try:
                    if s.lower() in ("q","quit","exit"):
                        loop.call_soon_threadsafe(manual_q.put_nowait, "__QUIT__")
                    elif s.lower() in ("r","reset","restart"):
                        loop.call_soon_threadsafe(manual_q.put_nowait, "__RESET__")
                    else:
                        loop.call_soon_threadsafe(manual_q.put_nowait, s)
                except RuntimeError:
                    break
        except Exception:
            pass
    threading.Thread(target=_stdin_watcher, daemon=True).start()

    # Mic stream
    _start_microphone(mic_idx)
    reset_requested = False
    cmd_window_until = 0.0
    forced_done = False
    auto_registered_name: str | None = None

    try:
        while not stop.is_set():
            # manual keys
            handled_manual=False
            while True:
                try: typed = manual_q.get_nowait()
                except asyncio.QueueEmpty: break
                handled_manual=True
                if typed=="__QUIT__":
                    print("[keyboard] quit — shutting down."); stop.set(); break
                if typed=="__RESET__":
                    print("[keyboard] reset listener."); analysis_buf.clear(); reset_requested=True; _stop_monitor(); stop.set(); break
                # manual enroll: last ~3s
                wav_m = take_latest_seconds(analysis_buf, max(args.id_window, 3.0), args.rate)
                if wav_m is None:
                    print("[enroll] not enough audio.")
                else:
                    vec = spk_embed.embed(wav_m)
                    if vec is not None:
                        identity.enroll_voice(typed or "Major", vec)
                        print(f"[enrolled voice] {typed or 'Major'}")
                    else:
                        print("[enroll] speaker embed unavailable.", file=sys.stderr)
            if stop.is_set() or reset_requested: break
            if handled_manual: continue

            # next transcript
            try: transcript = await asyncio.wait_for(asyncio.shield(mic_task), timeout=0.2)
            except asyncio.TimeoutError: continue
            except StopAsyncIteration: stop.set(); break
            mic_task = asyncio.create_task(mic.__anext__())
            print(f"YOU: {transcript}")

            if _HELP_RE.search(transcript):
                print("[marvin help] Available commands:")
                for line in help_lines:
                    print(f"  - {line}")
                help_text = "Here are the Marvin commands: " + " ".join(help_lines)
                await speak_via_broker(
                    broker_url=args.broker_url,
                    session_id=args.session,
                    text=help_text,
                    voice_id=voice_id,
                    voice_mode=args.voice_mode,
                    player=player,
                    playback_mute=playback_mute,
                    context=None,
                    text_only=args.text_only,
                    timeout=30,
                    verbose=verbose,
                    barge_monitor=None,
                )
                continue

            # monitor trigger
            if _MONITOR_RE.search(transcript):
                _launch_monitor()
                continue

            if _RESET_RE.search(transcript):
                print("[voice] reset requested — restarting client.")
                analysis_buf.clear()
                _stop_monitor()
                reset_requested = True
                stop.set()
                break

            # wake-gated exit
            now = loop.time()
            if _WAKE_RE.search(transcript):
                cmd_window_until = now + 8.0
                print("[marvin] command mode enabled (8s).")
                continue
            if _EXIT_RE.search(transcript):
                if now <= cmd_window_until:
                    print("[voice] exit requested — shutting down."); stop.set(); break
                else:
                    print("[voice] 'exit' ignored (say 'hey Marvin' first).")

            device_cmd = parse_device_command(transcript)
            if device_cmd:
                if now > cmd_window_until:
                    print("[voice] device change ignored (say 'hey Marvin' first).")
                    continue
                kind, idx = device_cmd
                _stop_monitor()
                if kind == "camera":
                    cam_idx = idx
                    _persist_devices(cam_idx, mic_idx, spk_idx)
                    print(f"[voice] camera switched to index {cam_idx}.")
                elif kind == "microphone":
                    await _switch_microphone(idx)
                else:
                    await _switch_speaker(idx)
                continue

            # enrollment via phrase
            name = parse_registration(transcript)
            if name:
                wav = take_latest_seconds(analysis_buf, args.id_window, args.rate)
                if wav is not None:
                    vec = spk_embed.embed(wav)
                    if vec is not None:
                        identity.enroll_voice(name, vec); print(f"[enrolled voice] {name}")
                # no else: silent

            # build context with voice ID from shared registry
            wav_id = take_latest_seconds(analysis_buf, args.id_window, args.rate)
            context: Dict[str, Any] = {}
            is_self_voice = False
            if wav_id is not None:
                if not spk_embed.enabled:
                    vprint("[identity] Speaker embedding disabled; skipping voice matching.")
                else:
                    vec = spk_embed.embed(wav_id)
                    if vec is None:
                        vprint("[identity] Unable to embed current voice sample.")
                    else:
                        who = identity.identify_voice(vec, threshold=args.id_threshold)
                        if who:
                            if self_voice and who.strip().lower() == self_voice:
                                print(f"[voice] Ignoring self voice signature ({who}).")
                                is_self_voice = True
                            else:
                                context["speaker_id"] = who
                                print(f"[identity] Recognized speaker: {who}")
                        elif auto_registered_name is None:
                            new_name = auto_register_name or f"Guest-{args.session[:8]}"
                            identity.enroll_voice(new_name, vec)
                            auto_registered_name = new_name
                            context["speaker_id"] = new_name
                            print(f"[identity] Auto-registered voice as {new_name}.")

            if is_self_voice:
                continue

            async def _barge_monitor():
                need = int(0.3 * args.rate)
                while playback_mute.is_set() and not stop.is_set():
                    if len(analysis_buf) >= need:
                        arr = np.array([analysis_buf[i] for i in range(len(analysis_buf) - need, len(analysis_buf))], dtype=np.float32)
                        rms = float(np.sqrt(np.mean(arr * arr)))
                        if rms >= 0.04:
                            player.stop(); break
                    await asyncio.sleep(0.05)

            body = await speak_via_broker(
                broker_url=args.broker_url,
                session_id=args.session,
                text=transcript,
                voice_id=voice_id,
                voice_mode=args.voice_mode,
                player=player,
                playback_mute=playback_mute,
                context=context or None,
                text_only=args.text_only,
                timeout=60,
                verbose=verbose,
                barge_monitor=None if args.text_only else _barge_monitor,
            )

            if not body:
                continue

            ai_text = body.get("text", "")
            print(f"AI:  {ai_text}")

            cmd_spec = body.get("command")
            if isinstance(cmd_spec, dict):
                await _handle_command(cmd_spec)

    finally:
        mic_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, StopAsyncIteration):
            await mic_task
        with contextlib.suppress(Exception):
            await mic.aclose()
        _stop_monitor()
    print("Exiting.")
    return reset_requested

def main():
    try:
        while True:
            restart = asyncio.run(run())
            if not restart:
                break
            print("[system] reset complete; restarting Marvin client.")
    except KeyboardInterrupt:
        pass

if __name__=="__main__":
    main()

# Danielle
# Joanna
# Ruth
# Salli
# Matthew
# Ayanda
# Amy
# Olivia
# Stephen
# Kajal