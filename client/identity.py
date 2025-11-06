#!/usr/bin/env python3
"""
Unified identity registry for people and animals:
- People: face encodings (face_recognition) + voice embeddings (SpeechBrain ECAPA)
- Animals: HSV+LBP signatures (768-D)
Storage: ~/.whaddya/registry.json

Schema per entry:
{
  "name": "Major",
  "type": "person" | "dog" | "cat" | "donkey",
  "face": [128 floats]         # optional
  "voice": [192-dim? floats]   # optional (ECAPA embedding size)
  "sig": [768 floats]          # optional (animals)
}
"""
from __future__ import annotations
import json, os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Paths
STATE_DIR = Path(os.path.expanduser("~/.whaddya"))
STATE_DIR.mkdir(parents=True, exist_ok=True)
REGISTRY_PATH = STATE_DIR / "registry.json"

# -------- Core I/O --------
def _load() -> List[Dict[str, Any]]:
    if REGISTRY_PATH.exists():
        try:
            return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []

def _save(entries: List[Dict[str, Any]]) -> None:
    REGISTRY_PATH.write_text(json.dumps(entries, indent=2), encoding="utf-8")

def list_entries() -> List[Dict[str, Any]]:
    return _load()

def _ensure_unique(entries: List[Dict[str, Any]], name: str, etype: str) -> Optional[int]:
    nm = name.strip().lower()
    for i, e in enumerate(entries):
        if (e.get("type") == etype) and ((e.get("name") or "").strip().lower() == nm):
            return i
    return None

# -------- Cosine helpers --------
def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) + 1e-9) / (np.linalg.norm(b) + 1e-9))

# -------- Enroll --------
def enroll_face(name: str, face_vec: np.ndarray) -> None:
    entries = _load()
    idx = _ensure_unique(entries, name, "person")
    rec = (entries[idx] if idx is not None else {"name": name, "type": "person"})
    rec["face"] = face_vec.astype(float).tolist()
    if idx is None:
        entries.append(rec)
    _save(entries)

def enroll_voice(name: str, voice_vec: np.ndarray) -> None:
    entries = _load()
    idx = _ensure_unique(entries, name, "person")
    rec = (entries[idx] if idx is not None else {"name": name, "type": "person"})
    rec["voice"] = voice_vec.astype(float).tolist()
    if idx is None:
        entries.append(rec)
    _save(entries)

def enroll_animal(name: str, etype: str, sig_vec: np.ndarray) -> None:
    assert etype in {"dog", "cat", "donkey"}
    entries = _load()
    idx = _ensure_unique(entries, name, etype)
    rec = (entries[idx] if idx is not None else {"name": name, "type": etype})
    rec["sig"] = sig_vec.astype(float).tolist()
    if idx is None:
        entries.append(rec)
    _save(entries)

# -------- Identify --------
def identify_face(face_vec: np.ndarray, threshold: float = 0.45) -> Optional[str]:
    """threshold (kept for API parity); cosine sim cutoff applied internally."""
    entries = _load()
    best, best_name = -1.0, None
    fv_query = face_vec.astype(np.float32)
    for e in entries:
        fv = e.get("face")
        if e.get("type") != "person" or fv is None:
            continue
        sim = _cos(fv_query, np.array(fv, dtype=np.float32))
        if sim > best:
            best, best_name = sim, e.get("name")
    # empirical: sim >= 0.60 is a decent bar; adjust upstream if needed
    return best_name if best >= 0.60 else None

def identify_voice(voice_vec: np.ndarray, threshold: float = 0.65) -> Optional[str]:
    entries = _load()
    best, best_name = -1.0, None
    vv_query = voice_vec.astype(np.float32)
    for e in entries:
        vv = e.get("voice")
        if e.get("type") != "person" or vv is None:
            continue
        sim = _cos(vv_query, np.array(vv, dtype=np.float32))
        if sim > best:
            best, best_name = sim, e.get("name")
    return best_name if best >= threshold else None

def identify_animal(etype: str, sig_vec: np.ndarray, max_dist: float = 0.22) -> Optional[str]:
    """
    max_dist is cosine distance cutoff; we convert to cosine similarity need = 1 - max_dist.
    """
    entries = _load()
    best_sim, best_name = -1.0, None
    need = 1.0 - max_dist
    v = sig_vec.astype(np.float32)
    for e in entries:
        if e.get("type") != etype:
            continue
        s = e.get("sig")
        if s is None:
            continue
        sim = _cos(v, np.array(s, dtype=np.float32))
        if sim > best_sim:
            best_sim, best_name = sim, e.get("name")
    return best_name if best_sim >= need else None
