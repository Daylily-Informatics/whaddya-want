#!/usr/bin/env python3
"""
Unified identity registry for people and animals:

- People: face encodings (face_recognition) + voice embeddings (SpeechBrain ECAPA)
- Animals: HSV+LBP signatures (768-D)

Primary storage: ~/.whaddya/registry.json

Optional remote mirrors (best-effort, non-fatal if misconfigured/unavailable):

- DynamoDB table (JSON blob in a single row)
  - Configure via: IDENTITY_TABLE (table name)
  - Row schema: { "id": "identity_registry", "entries": [...] }

- S3 object (JSON file)
  - Configure via:
      IDENTITY_BUCKET  (preferred)  OR  AUDIO_BUCKET (fallback)
      IDENTITY_KEY     (optional, default "identity/registry.json")
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Optional AWS clients (for S3 / Dynamo mirror)
try:  # pragma: no cover - optional dependency
    import boto3  # type: ignore
    from botocore.exceptions import BotoCoreError, ClientError  # type: ignore
except Exception:  # pragma: no cover - offline / minimal env
    boto3 = None  # type: ignore
    BotoCoreError = ClientError = Exception  # type: ignore

# ---------------------------------------------------------------------------
# Local paths
# ---------------------------------------------------------------------------

STATE_DIR = Path(os.path.expanduser("~/.whaddya"))
STATE_DIR.mkdir(parents=True, exist_ok=True)
REGISTRY_PATH = STATE_DIR / "registry.json"

# ---------------------------------------------------------------------------
# Remote config (all optional)
# ---------------------------------------------------------------------------

IDENTITY_TABLE = os.getenv("IDENTITY_TABLE")  # dedicated Dynamo table name (optional)

IDENTITY_BUCKET = os.getenv("IDENTITY_BUCKET") or os.getenv("AUDIO_BUCKET")
IDENTITY_KEY = os.getenv("IDENTITY_KEY", "identity/registry.json")

_AWS_REGION = (
    os.getenv("AWS_REGION")
    or os.getenv("AWS_DEFAULT_REGION")
    or os.getenv("REGION")
    or "us-west-2"
)

_s3_client = None
_ddb_table = None


def _get_s3():
    """Lazy-init S3 client if bucket configured."""
    global _s3_client
    if _s3_client is None and boto3 is not None and IDENTITY_BUCKET:
        _s3_client = boto3.client("s3", region_name=_AWS_REGION)
    return _s3_client


def _get_ddb_table():
    """Lazy-init DynamoDB table if configured."""
    global _ddb_table
    if _ddb_table is None and boto3 is not None and IDENTITY_TABLE:
        ddb = boto3.resource("dynamodb", region_name=_AWS_REGION)
        _ddb_table = ddb.Table(IDENTITY_TABLE)
    return _ddb_table


# ---------------------------------------------------------------------------
# Core I/O
# ---------------------------------------------------------------------------

def _remote_load() -> Optional[List[Dict[str, Any]]]:
    """
    Best-effort remote load.

    Order:
    1. DynamoDB (if IDENTITY_TABLE is set)
    2. S3       (if IDENTITY_BUCKET is set)

    Returns a list of entries or None if remote is unavailable / misconfigured.
    """
    # DynamoDB first (if configured)
    tbl = _get_ddb_table()
    if tbl is not None:
        try:
            resp = tbl.get_item(Key={"id": "identity_registry"})
            item = resp.get("Item")
            if item and isinstance(item.get("entries"), list):
                return item["entries"]
        except (BotoCoreError, ClientError, Exception):
            # Don't crash the client if Dynamo is misconfigured
            pass

    # Then S3 (if configured)
    s3 = _get_s3()
    if s3 is not None and IDENTITY_BUCKET:
        try:
            obj = s3.get_object(Bucket=IDENTITY_BUCKET, Key=IDENTITY_KEY)
            data = obj["Body"].read().decode("utf-8")
            entries = json.loads(data)
            if isinstance(entries, list):
                return entries
        except (BotoCoreError, ClientError, Exception):
            pass

    return None


def _remote_save(entries: List[Dict[str, Any]]) -> None:
    """
    Best-effort remote save.

    - Writes to DynamoDB row id="identity_registry" if IDENTITY_TABLE is set.
    - Writes to S3 object if IDENTITY_BUCKET is set.
    - Swallows failures; local file is always the authoritative fallback.
    """
    # DynamoDB
    tbl = _get_ddb_table()
    if tbl is not None:
        try:
            tbl.put_item(
                Item={
                    "id": "identity_registry",
                    "entries": entries,
                }
            )
        except (BotoCoreError, ClientError, Exception):
            pass

    # S3
    s3 = _get_s3()
    if s3 is not None and IDENTITY_BUCKET:
        try:
            body = json.dumps(entries, indent=2).encode("utf-8")
            s3.put_object(
                Bucket=IDENTITY_BUCKET,
                Key=IDENTITY_KEY,
                Body=body,
                ContentType="application/json",
            )
        except (BotoCoreError, ClientError, Exception):
            pass


def _load() -> List[Dict[str, Any]]:
    """
    Load the identity registry.

    Precedence:
    1. Remote (Dynamo/S3) if available.
    2. Local ~/.whaddya/registry.json
    3. Empty list.
    """
    # Remote first, if configured
    remote = _remote_load()
    if isinstance(remote, list):
        return remote

    # Local fallback
    if REGISTRY_PATH.exists():
        try:
            return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []


def _save(entries: List[Dict[str, Any]]) -> None:
    """
    Save registry locally and attempt to mirror to remote.

    - Local write MUST succeed (or raise).
    - Remote writes are best-effort; failures are swallowed.
    """
    REGISTRY_PATH.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    _remote_save(entries)


def list_entries() -> List[Dict[str, Any]]:
    return _load()


def _ensure_unique(entries: List[Dict[str, Any]], name: str, etype: str) -> Optional[int]:
    nm = name.strip().lower()
    for i, e in enumerate(entries):
        if (e.get("type") == etype) and ((e.get("name") or "").strip().lower() == nm):
            return i
    return None


def _prune_person_record(rec: Dict[str, Any]) -> bool:
    """
    Remove empty person records.

    Returns True if the record still contains identifying data (voice or face),
    False if it should be dropped entirely.
    """
    return bool(rec.get("voice") or rec.get("face"))


# ---------------------------------------------------------------------------
# Cosine helpers
# ---------------------------------------------------------------------------

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) + 1e-9) / (np.linalg.norm(b) + 1e-9))


# ---------------------------------------------------------------------------
# Enroll
# ---------------------------------------------------------------------------

def enroll_face(name: str, face_vec: np.ndarray) -> None:
    """
    Enroll or update a person with a face embedding.

    - Updates local registry.json.
    - Best-effort mirror to Dynamo/S3 if configured.
    """
    entries = _load()
    idx = _ensure_unique(entries, name, "person")
    rec = (entries[idx] if idx is not None else {"name": name, "type": "person"})
    rec["face"] = np.asarray(face_vec, dtype=float).tolist()
    if idx is None:
        entries.append(rec)
    else:
        entries[idx] = rec
    _save(entries)


def enroll_voice(name: str, voice_vec: np.ndarray) -> None:
    """
    Enroll or update a person with a voice embedding.
    """
    entries = _load()
    idx = _ensure_unique(entries, name, "person")
    rec = (entries[idx] if idx is not None else {"name": name, "type": "person"})
    rec["voice"] = np.asarray(voice_vec, dtype=float).tolist()
    if idx is None:
        entries.append(rec)
    else:
        entries[idx] = rec
    _save(entries)


def enroll_animal(name: str, etype: str, sig_vec: np.ndarray) -> None:
    """
    Enroll or update an animal (dog/cat/donkey) with a 768-D signature.
    """
    assert etype in {"dog", "cat", "donkey"}
    entries = _load()
    idx = _ensure_unique(entries, name, etype)
    rec = (entries[idx] if idx is not None else {"name": name, "type": etype})
    rec["sig"] = np.asarray(sig_vec, dtype=float).tolist()
    if idx is None:
        entries.append(rec)
    else:
        entries[idx] = rec
    _save(entries)


# ---------------------------------------------------------------------------
# Delete / unenroll
# ---------------------------------------------------------------------------

def delete_face(name: str) -> bool:
    """
    Remove a person's face profile.

    Returns True if a face entry was removed.
    """
    entries = _load()
    idx = _ensure_unique(entries, name, "person")
    if idx is None:
        return False
    rec = entries[idx]
    if "face" not in rec:
        return False
    rec.pop("face", None)
    if not _prune_person_record(rec):
        entries.pop(idx)
    else:
        entries[idx] = rec
    _save(entries)
    return True


def delete_voice(name: str) -> bool:
    """
    Remove a person's voice profile.

    Returns True if a voice entry was removed.
    """
    entries = _load()
    idx = _ensure_unique(entries, name, "person")
    if idx is None:
        return False
    rec = entries[idx]
    if "voice" not in rec:
        return False
    rec.pop("voice", None)
    if not _prune_person_record(rec):
        entries.pop(idx)
    else:
        entries[idx] = rec
    _save(entries)
    return True


# ---------------------------------------------------------------------------
# Identify
# ---------------------------------------------------------------------------

def identify_face(face_vec: np.ndarray, threshold: float = 0.45) -> Optional[str]:
    """
    Identify a person by face embedding.

    NOTE: `threshold` is ignored; we use an internal cosine sim cutoff so the
    caller API stays stable while we tune the metric.

    Returns:
        - best-matching name (str) if similarity >= cutoff
        - None otherwise
    """
    entries = _load()
    best, best_name = -1.0, None
    fv_query = np.asarray(face_vec, dtype=np.float32)
    for e in entries:
        fv = e.get("face")
        if e.get("type") != "person" or fv is None:
            continue
        sim = _cos(fv_query, np.array(fv, dtype=np.float32))
        if sim > best:
            best, best_name = sim, e.get("name")
    # empirical: sim >= 0.60 is a decent bar; adjust upstream if needed
    return best_name if best >= 0.60 else None


def best_voice_match(voice_vec: np.ndarray) -> tuple[Optional[str], float]:
    """Return the closest voice match along with its cosine similarity."""

    entries = _load()
    best, best_name = -1.0, None
    vv_query = np.asarray(voice_vec, dtype=np.float32)
    for e in entries:
        vv = e.get("voice")
        if e.get("type") != "person" or vv is None:
            continue
        sim = _cos(vv_query, np.array(vv, dtype=np.float32))
        if sim > best:
            best, best_name = sim, e.get("name")
    return best_name, float(best)


def identify_voice(voice_vec: np.ndarray, threshold: float = 0.65) -> Optional[str]:
    name, score = best_voice_match(voice_vec)
    return name if score >= threshold else None


def identify_animal(etype: str, sig_vec: np.ndarray, max_dist: float = 0.22) -> Optional[str]:
    """
    Identify an animal of a given type via cosine similarity on the 768-D signature.

    max_dist is a cosine **distance** cutoff; we convert to a similarity requirement
    internally: need = 1 - max_dist.
    """
    entries = _load()
    best_sim, best_name = -1.0, None
    need = 1.0 - max_dist
    v = np.asarray(sig_vec, dtype=np.float32)
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
