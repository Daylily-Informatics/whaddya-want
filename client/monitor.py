#!/usr/bin/env python3
from __future__ import annotations
# Allow running as "python client/monitor.py" by adding repo root to sys.path (optional safety)
if __package__ is None or __package__ == "":
    import sys as _sys, pathlib as _pathlib
    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import argparse, json, os, sys, time, queue, re, threading, base64, asyncio, hashlib
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from ultralytics import YOLO
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import requests

from client import identity
from client.shared_audio import AudioPlayer, say_via_broker_sync, get_shared_audio

# Paths & state
STATE_DIR = Path(os.path.expanduser("~/.whaddya"))
STATE_DIR.mkdir(parents=True, exist_ok=True)
DEVICES_JSON = STATE_DIR / "devices.json"
IMAGES_DIR = (Path(__file__).resolve().parent.parent / "images")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

ASR_SAMPLE_RATE = 16000
COCO_PERSON, COCO_CAT, COCO_DOG, COCO_HORSE = 0,15,16,17
SPECIES_LABEL = {COCO_PERSON:"person", COCO_CAT:"cat", COCO_DOG:"dog", COCO_HORSE:"donkey"}
DETECT_CLASSES = [COCO_PERSON, COCO_CAT, COCO_DOG, COCO_HORSE]

REENTRY_ABSENCE_SEC = 2.0

# ---------- Animal signature (HSV+LBP) ----------
def _lbp_hist(gray: np.ndarray) -> np.ndarray:
    g=gray.astype(np.int16); c=g[1:-1,1:-1]
    codes=((g[0:-2,1:-1]>=c)<<0)|((g[0:-2,2:]>=c)<<1)|((g[1:-1,2:]>=c)<<2)|((g[2:,2:]>=c)<<3)|((g[2:,1:-1]>=c)<<4)|((g[2:,0:-2]>=c)<<5)|((g[1:-1,0:-2]>=c)<<6)|((g[0:-2,0:-2]>=c)<<7)
    h=np.bincount(codes.ravel(), minlength=256).astype(np.float32); h/= (h.sum()+1e-9); return h

def compute_animal_signature(bgr_roi: np.ndarray) -> np.ndarray:
    if bgr_roi.size==0: return np.zeros(768, np.float32)
    hsv=cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    hist=cv2.calcHist([hsv],[0,1,2],None,[8,8,8],[0,180,0,256,0,256]).astype(np.float32).ravel()
    hist/= (np.linalg.norm(hist)+1e-9)
    gray=cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2GRAY)
    lbp=_lbp_hist(gray)
    sig=np.concatenate([hist,lbp]).astype(np.float32); sig/= (np.linalg.norm(sig)+1e-9); return sig

# ---------- Faces ----------
_FACE_OK=False
try:
    import face_recognition
    _FACE_OK=True
except Exception:
    _FACE_OK=False

def analyze_faces(frame: np.ndarray):
    if not _FACE_OK: return []
    rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locs=face_recognition.face_locations(rgb, model="hog")
    if not locs: return []
    encs=face_recognition.face_encodings(rgb, locs, num_jitters=1)
    out=[]
    for loc,enc in zip(locs,encs):
        nm=identity.identify_face(enc, threshold=0.45)
        out.append({"location":loc,"encoding":enc,"name":nm})
    return out

# ---------- Vosk ASR for simple monitor commands ----------
def load_asr_model()->Model:
    for c in ["vosk-model-small-en-us-0.15","vosk-model-en-us-0.22"]:
        if os.path.isdir(c): return Model(c)
    raise RuntimeError("Vosk model not found (unzip in CWD).")

_CMD_PAUSE={"pause","hold","stop"}
_CMD_RESUME={"resume","continue","unpause"}
_CMD_EXIT={"exit","quit","bye","goodbye"}

def _normalize(s:str)->str:
    return re.sub(r"\s+"," ", re.sub(r"[^\w\s]"," ", s.lower())).strip()

def asr_listener(event_q: queue.Queue, stop_ev: threading.Event, mic_device: int|None=None,
                 playback_mute: threading.Event|None=None):
    rec=KaldiRecognizer(load_asr_model(), ASR_SAMPLE_RATE); rec.SetWords(False)
    def cb(indata, frames, timeinfo, status):
        if stop_ev.is_set(): raise sd.CallbackStop()
        if playback_mute is not None and playback_mute.is_set():
            return
        if not rec.AcceptWaveform(bytes(indata)): return
        phrase=(json.loads(rec.Result()).get("text") or "").strip()
        if not phrase: return
        toks=set(_normalize(phrase).split())
        if toks & _CMD_EXIT:   event_q.put(("cmd","exit"))
        elif toks & _CMD_PAUSE:event_q.put(("cmd","pause"))
        elif toks & _CMD_RESUME:event_q.put(("cmd","resume"))
    with sd.RawInputStream(samplerate=ASR_SAMPLE_RATE,blocksize=8000,dtype="int16",
                           channels=1,callback=cb,device=mic_device):
        while not stop_ev.is_set(): sd.sleep(100)

# ---------- YOLO utils ----------
def yolo_entities_strict(results, frame_shape, roi, person_conf, animal_conf, min_area_frac):
    H,W=frame_shape[:2]; area_min=min_area_frac*(W*H)
    ent=[]; det=results[0]
    for b in det.boxes:
        cls=int(b.cls[0].item()) if b.cls is not None else None
        if cls not in DETECT_CLASSES: continue
        conf=float(b.conf[0].item()) if b.conf is not None else 0.0
        need=person_conf if cls==COCO_PERSON else animal_conf
        if conf<need: continue
        x1,y1,x2,y2=map(int,b.xyxy[0].tolist())
        if (x2-x1)*(y2-y1)<area_min: continue
        rx1,ry1,rx2,ry2=roi
        if not (x1>=rx1*W and y1>=ry1*H and x2<=rx2*W and y2<=ry2*H): continue
        ent.append((SPECIES_LABEL.get(cls,"person"),(x1,y1,x2,y2)))
    return ent

def load_saved_devices()->Tuple[int|None,int|None,int|None]:
    if DEVICES_JSON.exists():
        try:
            d=json.loads(DEVICES_JSON.read_text())
            return d.get("camera_index"), d.get("mic_index"), d.get("speaker_index")
        except Exception:
            return None,None,None
    return None,None,None

def parse_args():
    p=argparse.ArgumentParser(description="Marvin on-device monitor (unified with broker)")
    p.add_argument("--broker-url", required=True)
    p.add_argument("--session", required=True)
    p.add_argument("--voice", default=os.getenv("POLLY_VOICE") or "Joanna")
    p.add_argument("--voice-mode", choices=["standard","neural","generative"], default="standard")
    p.add_argument("--camera-index", type=int, default=None)
    p.add_argument("--mic-index", type=int, default=None)
    p.add_argument("--speaker-index", type=int, default=None)
    p.add_argument("--person-conf", type=float, default=0.40)
    p.add_argument("--animal-conf", type=float, default=0.40)
    p.add_argument("--min-area-frac", type=float, default=0.005)
    p.add_argument("--persist-frames", type=int, default=1)
    p.add_argument("--entry-cooldown-s", type=float, default=1.0)
    p.add_argument("--roi", type=str, default="0,0,1,1")
    p.add_argument("--animal-match-thresh", type=float, default=0.22)
    return p.parse_args()

def build_args(*, broker_url: str, session: str, voice: str | None = None,
               voice_mode: str = "standard", camera_index: int | None = None,
               mic_index: int | None = None, speaker_index: int | None = None,
               person_conf: float = 0.40, animal_conf: float = 0.40,
               min_area_frac: float = 0.005, persist_frames: int = 1,
               entry_cooldown_s: float = 1.0, roi: str = "0,0,1,1",
               animal_match_thresh: float = 0.22):
    return argparse.Namespace(
        broker_url=broker_url,
        session=session,
        voice=voice or os.getenv("POLLY_VOICE") or "Joanna",
        voice_mode=voice_mode,
        camera_index=camera_index,
        mic_index=mic_index,
        speaker_index=speaker_index,
        person_conf=person_conf,
        animal_conf=animal_conf,
        min_area_frac=min_area_frac,
        persist_frames=persist_frames,
        entry_cooldown_s=entry_cooldown_s,
        roi=roi,
        animal_match_thresh=animal_match_thresh,
    )

def start_monitor(
    *,
    broker_url: str,
    session: str,
    voice: str | None = None,
    voice_mode: str = "standard",
    camera_index: int | None = None,
    mic_index: int | None = None,
    speaker_index: int | None = None,
    person_conf: float = 0.40,
    animal_conf: float = 0.40,
    min_area_frac: float = 0.005,
    persist_frames: int = 1,
    entry_cooldown_s: float = 1.0,
    roi: str = "0,0,1,1",
    animal_match_thresh: float = 0.22,
    stop_event: threading.Event | None = None,
    player: AudioPlayer | None = None,
    playback_mute: threading.Event | None = None,
):
    args = build_args(
        broker_url=broker_url,
        session=session,
        voice=voice,
        voice_mode=voice_mode,
        camera_index=camera_index,
        mic_index=mic_index,
        speaker_index=speaker_index,
        person_conf=person_conf,
        animal_conf=animal_conf,
        min_area_frac=min_area_frac,
        persist_frames=persist_frames,
        entry_cooldown_s=entry_cooldown_s,
        roi=roi,
        animal_match_thresh=animal_match_thresh,
    )
    return run_monitor(args, stop_event=stop_event, player=player, playback_mute=playback_mute)

def _parse_roi(s: str):
    try:
        x1,y1,x2,y2=[float(v) for v in s.split(",")]
        x1=max(0.0,min(1.0,x1)); y1=max(0.0,min(1.0,y1))
        x2=max(x1,min(1.0,x2));   y2=max(y1,min(1.0,y2))
        return x1,y1,x2,y2
    except Exception:
        return 0,0,1,1



# ---------- Name capture (Vosk) ----------
def _extract_name_freeform(text: str) -> str|None:
    t=_normalize(text)
    for trig in ["my name is","i am","i'm","call me","it's","its","name is"]:
        if trig in t:
            tail=t.split(trig,1)[1].strip()
            toks=tail.split()
            cand=" ".join(toks[:3]).strip().title()
            cand=re.sub(r"^(the|a|an)\s+","",cand)
            if len(cand)>=2: return cand[:60]
    toks=t.split()
    if 1<=len(toks)<=3 and all(len(x)>1 for x in toks):
        return " ".join(toks).title()
    return None

def listen_for_name(mic_device: int|None, timeout_s: float=7.0) -> str|None:
    rec=KaldiRecognizer(load_asr_model(), ASR_SAMPLE_RATE); rec.SetWords(False)
    got=[]; stop_ev=threading.Event()
    def cb(indata,frames,timeinfo,status):
        if stop_ev.is_set(): raise sd.CallbackStop()
        if not rec.AcceptWaveform(bytes(indata)): return
        phrase=(json.loads(rec.Result()).get("text") or "").strip()
        if phrase: got.append(phrase)
        nm=_extract_name_freeform(phrase or "")
        if nm:
            got.append(f"__NAME__:{nm}")
            stop_ev.set(); raise sd.CallbackStop()
    deadline=time.monotonic()+timeout_s
    try:
        with sd.RawInputStream(samplerate=ASR_SAMPLE_RATE,blocksize=8000,dtype="int16",
                               channels=1,callback=cb,device=mic_device):
            while not stop_ev.is_set() and time.monotonic()<deadline: sd.sleep(100)
    except sd.CallbackStop:
        pass
    for h in got:
        if h.startswith("__NAME__:"):
            return h.split(":",1)[1]
    if got:
        return _extract_name_freeform(" ".join(got))
    return None


def _unknown_key(person: dict) -> str:
    enc = person.get("encoding") if isinstance(person, dict) else None
    if enc is None:
        return "unknown_person_generic"
    try:
        data = np.asarray(enc, dtype=np.float32).tobytes()
    except Exception:
        return "unknown_person_generic"
    return hashlib.sha1(data).hexdigest()

# ---------- Main ----------
def run_monitor(
    args,
    stop_event: threading.Event | None = None,
    player: AudioPlayer | None = None,
    playback_mute: threading.Event | None = None,
):
    cam_s, mic_s, spk_s = load_saved_devices()
    cam_idx = args.camera_index if args.camera_index is not None else (cam_s if cam_s is not None else 0)
    mic_idx = args.mic_index if args.mic_index is not None else mic_s
    spk_idx = args.speaker_index if args.speaker_index is not None else spk_s
    try:
        if mic_idx is not None or spk_idx is not None:
            sd.default.device = (mic_idx, spk_idx)
    except Exception:
        pass

    stop_signal = stop_event or threading.Event()

    # Audio player (Polly mp3 from broker) — shared across the process
    if playback_mute is None or player is None:
        player, playback_mute = get_shared_audio()

    # Threads
    events: "queue.Queue[tuple]" = queue.Queue()
    threading.Thread(
        target=asr_listener,
        args=(events, stop_signal, mic_idx, playback_mute),
        daemon=True,
    ).start()

    # YOLO + camera
    model = YOLO("yolov8n.pt")
    cap   = cv2.VideoCapture(cam_idx)
    if not cap.isOpened(): raise RuntimeError(f"Cannot open camera index {cam_idx}")
    print("[monitor] ready — press 'q' to quit.")

    window_title = "Marvin Monitor (press q to quit)"
    display_enabled = True
    try:
        # On Linux/X11, startWindowThread can make background-thread windows usable.
        # On macOS (Cocoa) it’s flaky and can throw opaque C++ exceptions.
        if (
            threading.current_thread() is not threading.main_thread()
            and sys.platform.startswith("linux")
        ):
            cv2.startWindowThread()

        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    except cv2.error as e:
        display_enabled = False
        print(
            "[monitor] GUI display unavailable; running headless (no preview window). "
            f"OpenCV error: {e}",
            file=sys.stderr,
        )

    roi = _parse_roi(args.roi)
    present_frames=0
    prev_qualified=False
    absence_start=time.monotonic()
    last_entry_ts=0.0
    had_any=False
    state={"paused": False}
    presence_last_seen: dict[str, float] = {}
    last_spoken: dict[str, float] = {}
    unknown_prompted: set[str] = set()

    def _should_announce(label: str, *, now_mono: float, absent_ok: bool) -> bool:
        """Gate announcements so we only speak on meaningful arrivals."""

        last_seen = presence_last_seen.get(label)
        recently_spoken = (now_mono - last_spoken.get(label, 0.0)) < args.entry_cooldown_s
        recently_seen = last_seen is not None and (now_mono - last_seen) < REENTRY_ABSENCE_SEC
        return not recently_spoken and (not recently_seen or absent_ok)

    try:
        while True:
            if stop_signal.is_set():
                break

            # Commands
            try:
                while True:
                    kind, val = events.get_nowait()
                    if kind=="cmd":
                        if val=="exit":
                            say_via_broker_sync(
                                broker_url=args.broker_url,
                                session_id=args.session,
                                text="Exiting monitor.",
                                voice_id=args.voice,
                                voice_mode=args.voice_mode,
                                context={"intro_already_sent": True},
                                player=player,
                                playback_mute=playback_mute,
                            )
                            stop_signal.set()
                            break
                        elif val=="pause":
                            state["paused"]=True
                            say_via_broker_sync(
                                broker_url=args.broker_url,
                                session_id=args.session,
                                text="Paused detection.",
                                voice_id=args.voice,
                                voice_mode=args.voice_mode,
                                context={"intro_already_sent": True},
                                player=player,
                                playback_mute=playback_mute,
                            )
                        elif val=="resume":
                            state["paused"]=False
                            say_via_broker_sync(
                                broker_url=args.broker_url,
                                session_id=args.session,
                                text="Resumed detection.",
                                voice_id=args.voice,
                                voice_mode=args.voice_mode,
                                context={"intro_already_sent": True},
                                player=player,
                                playback_mute=playback_mute,
                            )
            except queue.Empty:
                pass

            if stop_signal.is_set():
                break

            ret, frame = cap.read()
            if not ret: print("[monitor] frame grab failed"); break

            # HUD ROI
            H,W = frame.shape[:2]
            rx1,ry1,rx2,ry2 = roi
            cv2.rectangle(frame, (int(rx1*W),int(ry1*H)), (int(rx2*W),int(ry2*H)), (80,80,80), 1)

            if not state["paused"]:
                results = model.predict(source=frame, verbose=False, classes=DETECT_CLASSES, conf=0.35)
                ents = yolo_entities_strict(results, frame.shape, roi, args.person_conf, args.animal_conf, args.min_area_frac)

                for et,(x1,y1,x2,y2) in ents:
                    color=(0,255,0) if et=="person" else (255,165,0) if et in {"dog","cat"} else (255,0,255)
                    cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                    cv2.putText(frame, et, (x1, max(20,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if ents:
                    present_frames += 1
                else:
                    if present_frames>0:
                        absence_start = time.monotonic()
                    present_frames = 0

                qualified = present_frames >= args.persist_frames
                absent_ok = (time.monotonic() - absence_start) >= REENTRY_ABSENCE_SEC

                if ents:
                    now_mono = time.monotonic()
                else:
                    # Clear presence cache after a brief absence so re-entries can trigger.
                    stale = [k for k, v in presence_last_seen.items() if (time.monotonic() - v) >= REENTRY_ABSENCE_SEC]
                    for k in stale:
                        presence_last_seen.pop(k, None)

                if qualified and not prev_qualified:
                    # DEBUG: did we ever actually fire an entry event?
                    print(
                        "                    [monitor] ENTRY EVENT fired; ents =",
                        ents,
                        "                      present_frames =",
                        present_frames,
                        flush=True,
                    )

                    now_ts = time.time()
                    now_mono = time.monotonic()
                    print(
                        "                    [monitor] ",
                        now_ts - last_entry_ts,
                        "s since last entry); ents =",
                        args.entry_cooldown_s,
                        " had any =",
                        had_any,
                        ".... absent_ok =",
                        absent_ok,
                        "....",
                        flush=True,
                    )

                    if (now_ts - last_entry_ts) >= args.entry_cooldown_s and (not had_any or absent_ok):
                        last_entry_ts = now_ts
                        had_any = True
                        # snapshot
                        try:
                            cv2.imwrite(str(IMAGES_DIR / f"entry_{time.strftime('%Y%m%d_%H%M%S')}.jpg"), frame)
                        except Exception:
                            pass

                        pending_names: list[str] = []
                        unknown_people: list[dict] = []
                        detected_labels: set[str] = set()

                        # Identify persons (shared assistant session)
                        if any(e == "person" for e, _ in ents):
                            faces = analyze_faces(frame)
                            if faces:
                                for f in faces:
                                    label = f["name"] or "unknown_person"
                                    detected_labels.add(label)
                                    if f["name"] and _should_announce(label, now_mono=now_mono, absent_ok=absent_ok):
                                        pending_names.append(f["name"])
                                        last_spoken[label] = now_mono
                                    elif not f["name"]:
                                        unknown_people.append(f)
                            elif _should_announce("unknown_person", now_mono=now_mono, absent_ok=absent_ok):
                                detected_labels.add("unknown_person")
                                unknown_people.append({"encoding": None})

                        if unknown_people:
                            detected_labels.add("unknown_person")
                            new_unknowns: list[tuple[str, dict]] = []
                            for person in unknown_people:
                                key = _unknown_key(person)
                                if key not in unknown_prompted:
                                    new_unknowns.append((key, person))

                            if new_unknowns and _should_announce("unknown_person", now_mono=now_mono, absent_ok=absent_ok):
                                key, first_unknown = new_unknowns[0]
                                unknown_prompted.add(key)
                                last_spoken["unknown_person"] = now_mono
                                say_via_broker_sync(
                                    broker_url=args.broker_url,
                                    session_id=args.session,
                                    text=(
                                        "The monitor sees someone entering, but I don't recognize them. "
                                        "Greet them in Marvin's voice, ask who they are, and request a name."
                                    ),
                                    voice_id=args.voice,
                                    voice_mode=args.voice_mode,
                                    context={"intro_already_sent": True, "monitor_event": "entry_unknown_person"},
                                    player=player,
                                    playback_mute=playback_mute,
                                )
                                nm = listen_for_name(mic_idx, timeout_s=7.0)
                                if nm:
                                    enc = first_unknown.get("encoding") if first_unknown else None
                                    if enc is not None:
                                        identity.enroll_face(nm, enc)
                                    pending_names.append(nm)
                                    presence_last_seen[nm] = now_mono

                        # Animals
                        for et, (x1, y1, x2, y2) in ents:
                            if et == "person":
                                continue
                            roi_img = frame[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
                            sig = compute_animal_signature(roi_img)
                            label = f"unknown_{et}"
                            who = identity.identify_animal(et, sig, max_dist=args.animal_match_thresh)
                            if who:
                                label = who
                                if _should_announce(label, now_mono=now_mono, absent_ok=absent_ok):
                                    pending_names.append(who)
                                    last_spoken[label] = now_mono
                            else:
                                if _should_announce(label, now_mono=now_mono, absent_ok=absent_ok):
                                    prompt = (
                                        f"The monitor spotted a {et} entering and doesn't recognize them. "
                                        "Greet them as Marvin and ask what name to use."
                                    )
                                    say_via_broker_sync(
                                        broker_url=args.broker_url,
                                        session_id=args.session,
                                        text=prompt,
                                        voice_id=args.voice,
                                        voice_mode=args.voice_mode,
                                        context={"intro_already_sent": True, "monitor_event": "entry_unknown_animal"},
                                        player=player,
                                        playback_mute=playback_mute,
                                    )
                                    nm = listen_for_name(mic_idx, timeout_s=7.0)
                                    if nm:
                                        identity.enroll_animal(nm, et, sig)
                                        pending_names.append(nm)
                                        label = nm
                                    last_spoken[label] = now_mono
                            detected_labels.add(label)

                        if pending_names:
                            arrivals = ", ".join(sorted(set(pending_names)))
                            say_via_broker_sync(
                                broker_url=args.broker_url,
                                session_id=args.session,
                                text=(
                                    f"The monitor sees {arrivals} entering. Greet them as Marvin, say their name, "
                                    "and welcome them."
                                ),
                                voice_id=args.voice,
                                voice_mode=args.voice_mode,
                                context={"intro_already_sent": True, "monitor_event": "entry_known"},
                                player=player,
                                playback_mute=playback_mute,
                            )

                        for label in detected_labels:
                            presence_last_seen[label] = now_mono

                prev_qualified = qualified

            if display_enabled:
                try:
                    cv2.imshow(window_title, frame)
                    if (cv2.waitKey(1) & 0xff)==ord('q'):
                        break
                except cv2.error:
                    display_enabled = False
                    print(
                        "[monitor] GUI display disabled after OpenCV error; continuing headless.",
                        file=sys.stderr,
                    )

    finally:
        stop_signal.set()
        try: cap.release()
        except Exception: pass
        if display_enabled:
            try: cv2.destroyAllWindows()
            except Exception: pass
        try: sd.stop()
        except Exception: pass


def main():
    args = parse_args()
    run_monitor(args)

if __name__=="__main__":
    main()
