#!/usr/bin/env python3
from __future__ import annotations
import time
import threading
import asyncio
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


@dataclass
class FaceProbeResult:
    detected: bool
    recognized_name: Optional[str]
    captured_unknown: bool
    fallback_label: Optional[str]


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
    # vision / VLM integration
    #   "off"          → never send frames
    #   "entry_only"   → send one frame on entry events
    #   "entry_and_window" → send on entry + periodic snapshots during a short motion window
    vision_mode: str = "entry_and_window"
    vision_sample_interval_s: float = 5.0
    vision_motion_window_s: float = 15.0


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

        # vision / VLM sampling state
        self.vision_last_sample: float = 0.0
        self.vision_active_until: float = 0.0
        self.vision_busy: bool = False

    async def _speak(self, text: str, *, timeout: float = 30.0) -> None:
        await speak_via_broker(
            broker_url=self.cfg.broker_url,
            session_id=self.cfg.session,
            text=text,
            voice_id=self.cfg.voice_id,
            voice_mode=self.cfg.voice_mode,
            player=self.player,
            playback_mute=self.playback_mute,
            context={"intro_already_sent": True},
            text_only=False,
            timeout=timeout,
            verbose=False,
            barge_monitor=None,
        )

    async def _send_vision_snapshot(
        self,
        frame: np.ndarray,
        *,
        mode: str,
        context_extra: Optional[Dict[str, Any]] = None,
        text: Optional[str] = None,
    ) -> None:
        """
        Compress the current frame to JPEG and send it to the broker alongside a brief
        MONITOR_EVENT text. For `mode="entry"` this usually speaks; for `mode="update"`
        it defaults to text_only so Marvin doesn't chatter constantly.
        """
        if self.cfg.vision_mode == "off":
            return
        if self.vision_busy:
            return
        self.vision_busy = True
        try:
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if not ok:
                return
            jpeg_bytes = buf.tobytes()

            ctx: Dict[str, Any] = {"intro_already_sent": True}
            if context_extra:
                ctx.update(context_extra)

            if not text:
                if mode == "entry":
                    message_text = "MONITOR_EVENT: entry"
                else:
                    message_text = "MONITOR_EVENT: scene_update"
            else:
                message_text = text

            await speak_via_broker(
                broker_url=self.cfg.broker_url,
                session_id=self.cfg.session,
                text=message_text,
                voice_id=self.cfg.voice_id,
                voice_mode=self.cfg.voice_mode,
                player=self.player,
                playback_mute=self.playback_mute,
                context=ctx,
                text_only=(mode != "entry"),
                timeout=30,
                verbose=False,
                barge_monitor=None,
                image_jpeg=jpeg_bytes,
            )
        finally:
            self.vision_busy = False
            self.vision_last_sample = time.monotonic()

    def _record_frame(
        self,
        frame: np.ndarray,
        ents: List[Tuple[str, Tuple[int, int, int, int]]],
        sharpness: float,
    ) -> None:
        self.frame_buffer.append({"frame": frame.copy(), "sharpness": sharpness, "ents": ents})
        if len(self.frame_buffer) > FRAME_BUFFER_SIZE:
            self.frame_buffer.pop(0)

    def _capture_entities(
        self,
    ) -> Tuple[Optional[np.ndarray], List[Tuple[str, Tuple[int, int, int, int]]], float]:
        ret, frame = self.cap.read()
        if not ret:
            return None, [], 0.0
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
        self._record_frame(frame, ents, sharpness)
        return frame, ents, sharpness

    def _make_snotty_name(
        self, ents: List[Tuple[str, Tuple[int, int, int, int]]], frame_shape: Tuple[int, int, int]
    ) -> str:
        H, W = frame_shape[:2]
        person_boxes = [b for et, b in ents if et == "person"]
        if not person_boxes or H == 0 or W == 0:
            return "wiggly mystery blob"
        areas = []
        for x1, y1, x2, y2 in person_boxes:
            frac = max(0.0, (x2 - x1) * (y2 - y1)) / max(1.0, float(W * H))
            areas.append(frac)
        max_area = max(areas)
        if max_area > 0.25:
            return "restless close-up blur"
        if max_area > 0.1:
            return "wiggly silhouette"
        if max_area > 0.03:
            return "distant jitter shadow"
        return "tiny wandering speck"

    async def _collect_faces_with_retries(
        self,
        frame: np.ndarray,
        ents: List[Tuple[str, Tuple[int, int, int, int]]],
        *,
        retries: int = 2,
        wait_seconds: float = 2.0,
        hold_prompt: str = "Hold still so I can capture your face clearly.",
    ) -> Tuple[List[Dict[str, Any]], np.ndarray, List[Tuple[str, Tuple[int, int, int, int]]], Optional[str]]:
        faces = analyze_faces(frame)
        fallback_label: Optional[str] = None
        attempts = 0
        has_person = any(et == "person" for et, _ in ents)
        while not faces and has_person and attempts < retries:
            await self._speak(hold_prompt)
            await asyncio.sleep(wait_seconds)
            new_frame, new_ents, _ = self._capture_entities()
            if new_frame is None:
                break
            frame = new_frame
            ents = new_ents
            has_person = any(et == "person" for et, _ in ents)
            faces = analyze_faces(frame)
            attempts += 1

        if not faces and has_person:
            fallback_label = self._make_snotty_name(ents, frame.shape)

        return faces, frame, ents, fallback_label

    async def probe_face_identity(
        self,
        *,
        retries: int = 2,
        wait_seconds: float = 2.0,
    ) -> FaceProbeResult:
        was_paused = self.paused
        self.paused = True
        try:
            frame, ents, _ = self._capture_entities()
            if frame is None or not ents:
                return FaceProbeResult(False, None, False, None)

            faces, frame, ents, fallback_label = await self._collect_faces_with_retries(
                frame,
                ents,
                retries=retries,
                wait_seconds=wait_seconds,
            )

            recognized_name: Optional[str] = None
            captured_unknown = False
            for f in faces:
                nm = f.get("name")
                if nm:
                    recognized_name = nm
                    break
            if recognized_name:
                self.display_id_labels["person"] = recognized_name
                self._mark_seen(recognized_name, time.monotonic())
            elif faces:
                captured_unknown = True
                enc = faces[0].get("encoding")
                if enc is not None:
                    self.pending_face_enc = np.asarray(enc, dtype=np.float32)
            elif fallback_label:
                self.display_id_labels["person"] = fallback_label
                self._mark_seen(fallback_label, time.monotonic())

            return FaceProbeResult(True, recognized_name, captured_unknown, fallback_label)
        finally:
            self.paused = was_paused

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
        Enroll a face for `name`.

        Strategy:
        - If we previously captured an unknown face embedding at an entry event
          (pending_face_enc), use that.
        - Otherwise, grab the sharpest recent frame with a person, run
          face_recognition on it, and enroll the first face we see.
        """
        enc: Optional[np.ndarray] = None

        # 1) Use pending embedding from entry event if we have one
        if self.pending_face_enc is not None:
            enc = self.pending_face_enc
            print("[monitor] enroll_pending_face: using pending face embedding.")
            self.pending_face_enc = None
        else:
            print("[monitor] enroll_pending_face: no pending face, trying current frame buffer.")
            # 2) Fallback: choose the sharpest recent frame with a person and extract a face
            if not _FACE_OK:
                print("[monitor] enroll_pending_face: face_recognition not available.")
                return False

            id_frame_info: Optional[Dict[str, Any]] = None
            for item in self.frame_buffer:
                item_ents = item.get("ents") or []
                if any(e == "person" for e, _ in item_ents):
                    if id_frame_info is None or item["sharpness"] > id_frame_info["sharpness"]:
                        id_frame_info = item

            if id_frame_info is None:
                print("[monitor] enroll_pending_face: no recent frame with a person.")
                return False

            frame = id_frame_info.get("frame")
            if frame is None:
                print("[monitor] enroll_pending_face: best frame has no image data.")
                return False

            faces = analyze_faces(frame)
            if not faces:
                print("[monitor] enroll_pending_face: no faces found in best frame.")
                return False

            # Prefer an unknown face if any; otherwise just take the first
            chosen = None
            for f in faces:
                if not f.get("name"):
                    chosen = f
                    break
            if chosen is None:
                chosen = faces[0]

            enc = np.asarray(chosen.get("encoding"), dtype=np.float32)
            print("[monitor] enroll_pending_face: using fresh face encoding from current frame.")

        if enc is None:
            print("[monitor] enroll_pending_face: no usable encoding.")
            return False

        identity.enroll_face(name, enc)
        print(f"[monitor] enrolled face as {name}")
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
            self._record_frame(frame, ents, sharpness)

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
                    if self.cfg.vision_mode == "entry_and_window":
                        self.vision_active_until = time.monotonic() + self.cfg.vision_motion_window_s
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

            # Opportunistic scene snapshots while motion is ongoing / Marvin is talking
            if self.cfg.vision_mode == "entry_and_window":
                now = time.monotonic()
                should_sample = False
                if now < self.vision_active_until:
                    should_sample = True
                if self.playback_mute.is_set() and any(et == "person" for et, _ in ents):
                    should_sample = True

                if should_sample and (now - self.vision_last_sample) >= self.cfg.vision_sample_interval_s:
                    await self._send_vision_snapshot(
                        frame,
                        mode="update",
                        context_extra={"monitor_event": "update"},
                        text="MONITOR_EVENT: scene_update",
                    )

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
                if id_frame_info is None or item["sharpness"] > item.get("sharpness", 0.0):
                    id_frame_info = item
        if id_frame_info is None:
            id_frame_info = (
                self.frame_buffer[-1]
                if self.frame_buffer
                else {"frame": None, "sharpness": 0.0, "ents": ents}
            )

        id_frame = id_frame_info.get("frame")
        id_ents = id_frame_info.get("ents") or ents
        if id_frame is None:
            return

        H_id, W_id = id_frame.shape[:2]

        known_humans: List[str] = []
        unknown_people: List[Dict[str, Any]] = []
        known_animals_by_type: Dict[str, List[str]] = {}
        unknown_animals_by_type: Dict[str, int] = {}
        detected_labels: set[str] = set()

        fallback_label: Optional[str] = None

        # Humans: identify faces, collect unknowns, stash encoding for enrollment
        if any(e == "person" for e, _ in id_ents):
            faces, id_frame, id_ents, fallback_label = await self._collect_faces_with_retries(
                id_frame,
                id_ents,
                hold_prompt="Hold still for a moment so I can see your face.",
            )
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
            elif fallback_label:
                detected_labels.add(fallback_label)
                if self._should_announce(
                    fallback_label, now_mono=now_mono, absent_ok=absent_ok
                ):
                    self.last_spoken[fallback_label] = now_mono
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
            roi_img = id_frame[max(0, y1): min(H_id, y2), max(0, x1): min(W_id, x2)]
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

        # Build compact human/animal summaries for the text side
        humans_parts: List[str] = []
        if known_humans:
            humans_parts.append(
                "known=" + ", ".join(sorted(set(known_humans)))
            )
        if unknown_people:
            humans_parts.append(f"unknown_count={len(unknown_people)}")
        elif fallback_label:
            humans_parts.append(f"label={fallback_label}")
        if not humans_parts:
            humans_parts.append("none")
        humans_summary = "; ".join(humans_parts)

        animals_parts: List[str] = []
        all_known_animals: List[str] = []
        for _, names in known_animals_by_type.items():
            all_known_animals.extend(names)
        if all_known_animals:
            animals_parts.append(
                "known=" + ", ".join(sorted(set(all_known_animals)))
            )
        all_unknown_species: List[str] = []
        for et, count in unknown_animals_by_type.items():
            all_unknown_species.extend([et] * count)
        if all_unknown_species:
            animals_parts.append(
                "unknown_species=" + ", ".join(sorted(set(all_unknown_species)))
            )
        if not animals_parts:
            animals_parts.append("none")
        animals_summary = "; ".join(animals_parts)

        monitor_event_text = (
            "MONITOR_EVENT: entry\n"
            f"HUMANS: {humans_summary}\n"
            f"ANIMALS: {animals_summary}"
        )

        entry_ctx: Dict[str, Any] = {
            "intro_already_sent": True,
            "monitor_event": "entry",
            "known_humans": sorted(set(known_humans)),
            "unknown_human_count": len(unknown_people),
            "known_animals": sorted(set(all_known_animals)),
            "unknown_animals": sorted(set(all_unknown_species)),
            "fallback_label": fallback_label,
            "vision_hint": (
                f"Detected humans: known={sorted(set(known_humans))} "
                f"unknown_count={len(unknown_people)}; "
                f"animals: known={sorted(set(all_known_animals))} "
                f"unknown_species={sorted(set(all_unknown_species))}"
            ),
        }

        # Send initial entry snapshot to the broker's VLM if enabled
        await self._send_vision_snapshot(
            id_frame,
            mode="entry",
            context_extra=entry_ctx,
            text=monitor_event_text,
        )

        # Update overlay labels
        self.display_id_labels.clear()
        if known_humans:
            self.display_id_labels["person"] = ", ".join(sorted(set(known_humans)))
        elif unknown_people:
            self.display_id_labels["person"] = "unknown"
        elif fallback_label:
            self.display_id_labels["person"] = fallback_label
        for et, names in known_animals_by_type.items():
            self.display_id_labels[et] = ", ".join(sorted(set(names)))
        for et in unknown_animals_by_type.keys():
            if et not in self.display_id_labels:
                self.display_id_labels[et] = "unknown"

        for label in detected_labels:
            self._mark_seen(label, now_mono)
