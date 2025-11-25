#!/usr/bin/env python3
from __future__ import annotations
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from client import identity
from client.shared_audio import AudioPlayer, speak_via_broker

try:
    import face_recognition  # type: ignore

    _FACE_OK = True
except Exception:
    _FACE_OK = False

STATE_DIR = Path.home() / ".whaddya"
STATE_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR = STATE_DIR / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

COCO_PERSON, COCO_CAT, COCO_DOG, COCO_HORSE = 0, 15, 16, 17
SPECIES_LABEL = {
    COCO_PERSON: "person",
    COCO_CAT: "cat",
    COCO_DOG: "dog",
    COCO_HORSE: "donkey",
}
DETECT_CLASSES = [COCO_PERSON, COCO_CAT, COCO_DOG, COCO_HORSE]

REENTRY_ABSENCE_SEC = 2.0
FRAME_BUFFER_SIZE = 5


def compute_sharpness(bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def compute_animal_signature(bgr_roi: np.ndarray) -> np.ndarray:
    if bgr_roi.size == 0:
        return np.zeros(768, np.float32)
    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv],
        [0, 1, 2],
        None,
        [8, 8, 8],
        [0, 180, 0, 256, 0, 256],
    ).astype(np.float32).ravel()
    hist /= (np.linalg.norm(hist) + 1e-9)
    gray = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2GRAY)
    g = gray.astype(np.int16)
    c = g[1:-1, 1:-1]
    codes = (
        ((g[0:-2, 1:-1] >= c) << 0)
        | ((g[0:-2, 2:] >= c) << 1)
        | ((g[1:-1, 2:] >= c) << 2)
        | ((g[2:, 2:] >= c) << 3)
        | ((g[2:, 1:-1] >= c) << 4)
        | ((g[2:, 0:-2] >= c) << 5)
        | ((g[1:-1, 0:-2] >= c) << 6)
        | ((g[0:-2, 0:-2] >= c) << 7)
    )
    lbp_hist = np.bincount(codes.ravel(), minlength=256).astype(np.float32)
    lbp_hist /= (lbp_hist.sum() + 1e-9)
    sig = np.concatenate([hist, lbp_hist]).astype(np.float32)
    sig /= (np.linalg.norm(sig) + 1e-9)
    return sig


def analyze_faces(frame: np.ndarray) -> List[Dict[str, Any]]:
    if not _FACE_OK:
        return []
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locs = face_recognition.face_locations(rgb, model="hog")  # type: ignore[name-defined]
    if not locs:
        return []
    encs = face_recognition.face_encodings(rgb, locs, num_jitters=1)  # type: ignore[name-defined]
    out: List[Dict[str, Any]] = []
    for loc, enc in zip(locs, encs):
        nm = identity.identify_face(enc, threshold=0.45)
        out.append({"location": loc, "encoding": enc, "name": nm})
    return out


def yolo_entities_strict(
    results,
    frame_shape: Tuple[int, int, int],
    roi: Tuple[float, float, float, float],
    person_conf: float,
    animal_conf: float,
    min_area_frac: float,
) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    H, W = frame_shape[:2]
    area_min = min_area_frac * (W * H)
    ent: List[Tuple[str, Tuple[int, int, int, int]]] = []
    det = results[0]
    for b in det.boxes:
        cls = int(b.cls[0].item()) if b.cls is not None else None
        if cls not in DETECT_CLASSES:
            continue
        conf = float(b.conf[0].item()) if b.conf is not None else 0.0
        need = person_conf if cls == COCO_PERSON else animal_conf
        if conf < need:
            continue
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        if (x2 - x1) * (y2 - y1) < area_min:
            continue
        rx1, ry1, rx2, ry2 = roi
        if not (
            x1 >= rx1 * W
            and y1 >= ry1 * H
            and x2 <= rx2 * W
            and y2 <= ry2 * H
        ):
            continue
        ent.append((SPECIES_LABEL.get(cls, "person"), (x1, y1, x2, y2)))
    return ent


@dataclass
class MonitorConfig:
    broker_url: str
    session: str
    voice_id: Optional[str]
    voice_mode: str
    camera_index: int
    mic_index: Optional[int]
    # detection thresholds
    person_conf: float = 0.55
    animal_conf: float = 0.50
    min_area_frac: float = 0.005
    # require a few frames with a person before firing entry
    persist_frames: int = 3
    entry_cooldown_s: float = 5.0
    roi: str = "0,0,1,1"
    animal_match_thresh: float = 0.22


class MonitorEngine:
    """
    Headless monitor engine; step() is called from the asyncio loop on the main
    thread, and it handles camera read, detection, GUI overlay, and speech.
    """

    def __init__(
        self,
        cfg: MonitorConfig,
        player: AudioPlayer,
        playback_mute: threading.Event,
    ):
        self.cfg = cfg
        self.player = player
        self.playback_mute = playback_mute

        self.cap = cv2.VideoCapture(cfg.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {cfg.camera_index}")
        self.model = YOLO("yolov8n.pt")
        self.window_title = "Marvin Monitor (press q to quit)"
        cv2.namedWindow(self.window_title, cv2.WINDOW_NORMAL)

        self.roi = self._parse_roi(cfg.roi)
        self.present_frames = 0
        self.prev_qualified = False
        self.absence_start = time.monotonic()
        self.last_entry_ts = 0.0
        self.had_any = False
        self.paused = False

        self.presence_last_seen: Dict[str, float] = {}
        self.last_spoken: Dict[str, float] = {}
        self.unknown_prompted: set[str] = set()

        self.frame_buffer: List[Dict[str, Any]] = []
        self.display_id_labels: Dict[str, str] = {}
        self.stop_flag = False

        # pending unknown face encoding waiting for a name
        self.pending_face_enc: Optional[np.ndarray] = None

    def _parse_roi(self, s: str) -> Tuple[float, float, float, float]:
        try:
            x1, y1, x2, y2 = [float(v) for v in s.split(",")]
            x1 = max(0.0, min(1.0, x1))
            y1 = max(0.0, min(1.0, y1))
            x2 = max(x1, min(1.0, x2))
            y2 = max(y1, min(1.0, y2))
            return x1, y1, x2, y2
        except Exception:
            return 0.0, 0.0, 1.0, 1.0

    def _should_announce(self, label: str, *, now_mono: float, absent_ok: bool) -> bool:
        last_seen = self.presence_last_seen.get(label)
        recently_spoken = (now_mono - self.last_spoken.get(label, 0.0)) < self.cfg.entry_cooldown_s
        recently_seen = last_seen is not None and (now_mono - last_seen) < REENTRY_ABSENCE_SEC
        return not recently_spoken and (not recently_seen or absent_ok)

    def _mark_seen(self, label: str, now_mono: float) -> None:
        self.presence_last_seen[label] = now_mono

    def stop(self) -> None:
        self.stop_flag = True

    def cleanup(self) -> None:
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        try:
            cv2.destroyWindow(self.window_title)
        except Exception:
            pass

    def enroll_pending_face(self, name: str) -> bool:
        """
        Enroll the most recent unknown face under the given name.
        Called by the CLI when it successfully parses 'call me <name>' etc.
        """
        if self.pending_face_enc is None:
            print("[monitor] enroll_pending_face: no pending face.")
            return False
        try:
            identity.enroll_face(name, self.pending_face_enc)
            print(f"[monitor] enrolled pending face as {name}")
        finally:
            self.pending_face_enc = None
        return True

    def identify_current_face_name(self) -> Optional[str]:
        """
        Try to identify the most prominent face in the recent frames.
        Returns the name if recognized, or None.
        """
        if not _FACE_OK:
            return None
        id_frame_info: Optional[Dict[str, Any]] = None
        for item in self.frame_buffer:
            item_ents = item.get("ents") or []
            if any(e == "person" for e, _ in item_ents):
                if id_frame_info is None or item["sharpness"] > id_frame_info["sharpness"]:
                    id_frame_info = item
        if id_frame_info is None:
            return None
        frame = id_frame_info.get("frame")
        if frame is None:
            return None
        faces = analyze_faces(frame)
        for f in faces:
            nm = f.get("name")
            if nm:
                return nm
        return None

    async def step(self) -> None:
        if self.stop_flag:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        H, W = frame.shape[:2]
        rx1, ry1, rx2, ry2 = self.roi
        cv2.rectangle(
            frame,
            (int(rx1 * W), int(ry1 * H)),
            (int(rx2 * W), int(ry2 * H)),
            (80, 80, 80),
            1,
        )

        ents: List[Tuple[str, Tuple[int, int, int, int]]] = []

        if not self.paused:
            results = self.model.predict(
                source=frame,
                verbose=False,
                classes=DETECT_CLASSES,
                conf=0.35,
            )
            ents = yolo_entities_strict(
                results,
                frame.shape,
                self.roi,
                self.cfg.person_conf,
                self.cfg.animal_conf,
                self.cfg.min_area_frac,
            )

            sharpness = compute_sharpness(frame)
            self.frame_buffer.append(
                {"frame": frame.copy(), "sharpness": sharpness, "ents": ents}
            )
            if len(self.frame_buffer) > FRAME_BUFFER_SIZE:
                self.frame_buffer.pop(0)

            for et, (x1, y1, x2, y2) in ents:
                if et == "person":
                    color = (0, 255, 0)
                elif et in {"dog", "cat"}:
                    color = (255, 165, 0)
                else:
                    color = (255, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label_suffix = self.display_id_labels.get(et)
                if label_suffix:
                    label_text = f"{et}: {label_suffix}"
                else:
                    if et in {"person", "dog", "cat", "donkey"}:
                        label_text = f"{et}: unknown"
                    else:
                        label_text = et
                cv2.putText(
                    frame,
                    label_text,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

            if ents:
                self.present_frames += 1
            else:
                if self.present_frames > 0:
                    self.absence_start = time.monotonic()
                self.present_frames = 0

            qualified = self.present_frames >= self.cfg.persist_frames
            absent_ok = (time.monotonic() - self.absence_start) >= REENTRY_ABSENCE_SEC
            now_mono = time.monotonic()

            if not ents:
                stale = [
                    k
                    for k, v in self.presence_last_seen.items()
                    if (time.monotonic() - v) >= REENTRY_ABSENCE_SEC
                ]
                for k in stale:
                    self.presence_last_seen.pop(k, None)

            if qualified and not self.prev_qualified:
                now_ts = time.time()
                if (now_ts - self.last_entry_ts) >= self.cfg.entry_cooldown_s and (
                    not self.had_any or absent_ok
                ):
                    self.last_entry_ts = now_ts
                    self.had_any = True
                    try:
                        cv2.imwrite(
                            str(
                                IMAGES_DIR
                                / f"entry_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                            ),
                            frame,
                        )
                    except Exception:
                        pass
                    await self._handle_entry_event(ents, now_mono, absent_ok)

            self.prev_qualified = qualified

        cv2.imshow(self.window_title, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self.stop_flag = True

    async def _handle_entry_event(
        self,
        ents: List[Tuple[str, Tuple[int, int, int, int]]],
        now_mono: float,
        absent_ok: bool,
    ) -> None:
        if not ents:
            return

        # pick sharpest recent frame with a person
        id_frame_info: Optional[Dict[str, Any]] = None
        for item in self.frame_buffer:
            item_ents = item.get("ents") or []
            if any(e == "person" for e, _ in item_ents):
                if id_frame_info is None or item["sharpness"] > id_frame_info["sharpness"]:
                    id_frame_info = item
        if id_frame_info is None:
            id_frame_info = (
                self.frame_buffer[-1]
                if self.frame_buffer
                else {"frame": None, "sharpness": 0.0, "ents": ents}
            )

        id_frame = id_frame_info["frame"]
        id_ents = id_frame_info.get("ents") or ents
        if id_frame is None:
            return

        H_id, W_id = id_frame.shape[:2]

        known_humans: List[str] = []
        unknown_people: List[Dict[str, Any]] = []
        known_animals_by_type: Dict[str, List[str]] = {}
        unknown_animals_by_type: Dict[str, int] = {}
        detected_labels: set[str] = set()

        # Humans: identify faces, collect unknowns, stash encoding for enrollment
        if any(e == "person" for e, _ in id_ents):
            faces = analyze_faces(id_frame)
            if faces:
                for f in faces:
                    label = f["name"] or "unknown_person"
                    detected_labels.add(label)
                    if f["name"]:
                        known_humans.append(f["name"])
                        if self._should_announce(
                            label, now_mono=now_mono, absent_ok=absent_ok
                        ):
                            self.last_spoken[label] = now_mono
                    else:
                        unknown_people.append(f)
            elif self._should_announce(
                "unknown_person", now_mono=now_mono, absent_ok=absent_ok
            ):
                detected_labels.add("unknown_person")
                unknown_people.append({"encoding": None})

        if unknown_people:
            first = unknown_people[0]
            enc = first.get("encoding")
            if enc is not None:
                self.pending_face_enc = np.asarray(enc, dtype=np.float32)
                print("[monitor] stored pending unknown face encoding for enrollment")

        # Animals
        for et, (x1, y1, x2, y2) in id_ents:
            if et == "person":
                continue
            roi_img = id_frame[max(0, y1) : min(H_id, y2), max(0, x1) : min(W_id, x2)]
            sig = compute_animal_signature(roi_img)
            label = f"unknown_{et}"
            who = identity.identify_animal(et, sig, max_dist=self.cfg.animal_match_thresh)
            if who:
                label = who
                known_animals_by_type.setdefault(et, []).append(who)
                if self._should_announce(label, now_mono=now_mono, absent_ok=absent_ok):
                    self.last_spoken[label] = now_mono
            else:
                unknown_animals_by_type[et] = unknown_animals_by_type.get(et, 0) + 1
                if self._should_announce(label, now_mono=now_mono, absent_ok=absent_ok):
                    self.last_spoken[label] = now_mono
            detected_labels.add(label)

        humans_parts: List[str] = []
        if known_humans:
            humans_parts.append(
                "known=["
                + ", ".join(sorted(repr(h) for h in set(known_humans)))
                + "]"
            )
        if unknown_people:
            humans_parts.append(f"unknown_count={len(unknown_people)}")
        if not humans_parts:
            humans_parts.append("none")

        animals_parts: List[str] = []
        all_known_animals: List[str] = []
        for _, names in known_animals_by_type.items():
            all_known_animals.extend(names)
        if all_known_animals:
            animals_parts.append(
                "known=["
                + ", ".join(sorted(repr(a) for a in set(all_known_animals)))
                + "]"
            )
        all_unknown_species: List[str] = []
        for et, count in unknown_animals_by_type.items():
            all_unknown_species.extend([et] * count)
        if all_unknown_species:
            animals_parts.append(
                "unknown_species=["
                + ", ".join(sorted(repr(a) for a in set(all_unknown_species)))
                + "]"
            )
        if not animals_parts:
            animals_parts.append("none")

        task_bits: List[str] = []
        if known_humans:
            task_bits.append("Greet the known humans by name.")
        if unknown_people:
            task_bits.append(
                "Ask the unknown human(s), briefly and politely, what you should call them."
            )
        if all_known_animals or all_unknown_species:
            task_bits.append(
                "Optionally acknowledge animals in one short remark, then focus back on helping humans."
            )
        task_bits.append("Do NOT re-introduce yourself on monitor events.")

        lines = [
            "MONITOR_EVENT: entry",
            "HUMANS: " + " ".join(humans_parts),
            "ANIMALS: " + " ".join(animals_parts),
            "TASK: " + " ".join(task_bits),
        ]
        monitor_event_text = "\n".join(lines)

        await speak_via_broker(
            broker_url=self.cfg.broker_url,
            session_id=self.cfg.session,
            text=monitor_event_text,
            voice_id=self.cfg.voice_id,
            voice_mode=self.cfg.voice_mode,
            player=self.player,
            playback_mute=self.playback_mute,
            context={"intro_already_sent": True},
            text_only=False,
            timeout=30,
            verbose=False,
            barge_monitor=None,
        )

        self.display_id_labels.clear()
        if known_humans:
            self.display_id_labels["person"] = ", ".join(sorted(set(known_humans)))
        elif unknown_people:
            self.display_id_labels["person"] = "unknown"
        for et, names in known_animals_by_type.items():
            self.display_id_labels[et] = ", ".join(sorted(set(names)))
        for et in unknown_animals_by_type.keys():
            if et not in self.display_id_labels:
                self.display_id_labels[et] = "unknown"

        for label in detected_labels:
            self._mark_seen(label, now_mono)
