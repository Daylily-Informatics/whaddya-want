from __future__ import annotations

"""Voice identity registry backed by the AgentStateTable DynamoDB table.

Supports:
- Mapping (agent_id, voice_id) -> VoiceProfile
- Storing one or more speaker embeddings per voice profile
- Nearest-neighbor lookup by embedding to fuzzily match new voices to
  existing profiles.

NOTE: DynamoDB does not support native Python float; we must store numeric
embeddings as Decimal and convert back to float on read.
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import boto3
from boto3.dynamodb.conditions import Key  # type: ignore[attr-defined]


logger = logging.getLogger(__name__)

_TABLE_NAME = os.environ.get("AGENT_STATE_TABLE")
if not _TABLE_NAME:
    raise RuntimeError("AGENT_STATE_TABLE environment variable is not set")

_dynamodb = boto3.resource("dynamodb")
_table = _dynamodb.Table(_TABLE_NAME)


@dataclass(slots=True)
class VoiceProfile:
    """In-memory representation of a voice profile."""

    agent_id: str
    voice_id: str
    name: str
    created_at: datetime
    last_seen_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_item(self) -> Dict[str, Any]:
        """Convert this profile to a DynamoDB-friendly item.

        In particular, we must ensure numeric embeddings are stored as Decimal.
        """
        meta = encode_metadata_for_dynamo(self.metadata)
        return {
            "pk": f"AGENT#{self.agent_id}",
            "sk": f"VOICE#{self.voice_id}",
            "item_type": "VOICE_PROFILE",
            "name": self.name,
            "voice_id": self.voice_id,
            "created_at": self.created_at.isoformat(),
            "last_seen_at": self.last_seen_at.isoformat(),
            "metadata": meta,
        }

    @classmethod
    def from_item(cls, item: Dict[str, Any]) -> "VoiceProfile":
        """Reconstruct a VoiceProfile from a DynamoDB item."""
        agent_tag = item.get("pk", "")
        if agent_tag.startswith("AGENT#"):
            agent_id = agent_tag.split("#", 1)[1]
        else:
            agent_id = agent_tag or "unknown"

        voice_tag = item.get("sk", "")
        if voice_tag.startswith("VOICE#"):
            voice_id = voice_tag.split("#", 1)[1]
        else:
            voice_id = item.get("voice_id") or "unknown"

        created_raw = item.get("created_at") or datetime.now(timezone.utc).isoformat()
        last_raw = item.get("last_seen_at") or created_raw

        created_at = datetime.fromisoformat(created_raw)
        last_seen_at = datetime.fromisoformat(last_raw)

        meta = item.get("metadata") or {}
        meta = decode_metadata_from_dynamo(meta)

        return cls(
            agent_id=agent_id,
            voice_id=voice_id,
            name=item.get("name", "Unknown"),
            created_at=created_at,
            last_seen_at=last_seen_at,
            metadata=meta,
        )


# ---------------------------------------------------------------------------
# Metadata (embedding) encoding helpers
# ---------------------------------------------------------------------------

def encode_metadata_for_dynamo(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of metadata where embeddings are Decimal-encoded.

    We expect embeddings to live under meta["embeddings"] as a list of lists
    of floats. DynamoDB requires Decimal for numeric values.
    """
    if not isinstance(meta, dict):
        return {}

    out: Dict[str, Any] = dict(meta)
    embs = out.get("embeddings")
    if embs is None:
        return out

    if not isinstance(embs, list):
        # If it's garbage, drop it rather than exploding the write.
        logger.warning("Unexpected embeddings type in metadata: %r", type(embs))
        out["embeddings"] = []
        return out

    encoded: List[List[Decimal]] = []
    for e in embs:
        if not isinstance(e, list):
            continue
        vec: List[Decimal] = []
        for v in e:
            try:
                vec.append(Decimal(str(float(v))))
            except Exception:
                # Skip non-numeric junk.
                continue
        if vec:
            encoded.append(vec)

    out["embeddings"] = encoded
    return out


def decode_metadata_from_dynamo(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Convert metadata read from Dynamo into a float-friendly structure.

    - Numbers inside embeddings (Decimal) are converted back to float.
    - Other fields are left as-is.
    """
    if not isinstance(meta, dict):
        return {}

    out: Dict[str, Any] = dict(meta)
    embs = out.get("embeddings")
    if embs is None:
        return out

    if not isinstance(embs, list):
        logger.warning("Unexpected embeddings type in stored metadata: %r", type(embs))
        out["embeddings"] = []
        return out

    decoded: List[List[float]] = []
    for e in embs:
        if not isinstance(e, list):
            continue
        vec: List[float] = []
        for v in e:
            try:
                if isinstance(v, Decimal):
                    vec.append(float(v))
                else:
                    vec.append(float(v))
            except Exception:
                continue
        if vec:
            decoded.append(vec)

    out["embeddings"] = decoded
    return out


def _append_embedding(meta: Dict[str, Any], embedding: List[float]) -> Dict[str, Any]:
    """Append an embedding (List[float]) to metadata["embeddings"] in float space.

    We keep embeddings as floats in memory; to_item() will convert them to Decimal
    for Dynamo.
    """
    meta = dict(meta) if meta is not None else {}
    embs = meta.get("embeddings") or []
    if not isinstance(embs, list):
        embs = []
    embs = list(embs)
    embs.append(list(embedding))
    meta["embeddings"] = embs
    return meta


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------

def get_by_voice_id(agent_id: str, voice_id: str) -> Optional[VoiceProfile]:
    """Return the VoiceProfile for (agent_id, voice_id) if it exists."""
    pk = f"AGENT#{agent_id}"
    sk = f"VOICE#{voice_id}"
    resp = _table.get_item(Key={"pk": pk, "sk": sk})
    item = resp.get("Item")
    if not item:
        return None
    if item.get("item_type") != "VOICE_PROFILE":
        return None
    return VoiceProfile.from_item(item)


def upsert_voice_profile(
    agent_id: str,
    voice_id: str,
    name: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> VoiceProfile:
    """Create or update a voice profile for the given voice_id."""
    existing = get_by_voice_id(agent_id, voice_id)
    now = datetime.now(timezone.utc)

    if existing:
        logger.debug(
            "Updating existing voice profile for agent=%s voice_id=%s name=%s -> %s",
            agent_id,
            voice_id,
            existing.name,
            name,
        )
        existing.name = name
        existing.last_seen_at = now
        if metadata:
            merged = dict(existing.metadata)
            # Don't blindly overwrite embeddings; if caller provided embeddings,
            # append the last one.
            if "embeddings" in metadata:
                emb_list = metadata.get("embeddings")
                if isinstance(emb_list, list) and emb_list:
                    last_emb = emb_list[-1]
                    if isinstance(last_emb, list):
                        merged = _append_embedding(merged, [float(x) for x in last_emb])
            for k, v in metadata.items():
                if k == "embeddings":
                    continue
                merged[k] = v
            existing.metadata = merged
        profile = existing
    else:
        logger.debug(
            "Creating new voice profile for agent=%s voice_id=%s name=%s",
            agent_id,
            voice_id,
            name,
        )
        clean_meta: Dict[str, Any] = {}
        if metadata:
            clean_meta.update(metadata)
        profile = VoiceProfile(
            agent_id=agent_id,
            voice_id=voice_id,
            name=name,
            created_at=now,
            last_seen_at=now,
            metadata=clean_meta,
        )

    _table.put_item(Item=profile.to_item())
    return profile


def touch_voice_profile(agent_id: str, voice_id: str) -> None:
    """Update last_seen_at for a profile if it exists; do nothing otherwise."""
    profile = get_by_voice_id(agent_id, voice_id)
    if not profile:
        return
    profile.last_seen_at = datetime.now(timezone.utc)
    _table.put_item(Item=profile.to_item())


def list_voice_profiles(agent_id: str) -> List[VoiceProfile]:
    """List all voice profiles for a given agent."""
    pk = f"AGENT#{agent_id}"
    items: List[Dict[str, Any]] = []
    start_key: Optional[Dict[str, Any]] = None
    while True:
        kwargs: Dict[str, Any] = {
            "KeyConditionExpression": Key("pk").eq(pk)
        }
        if start_key:
            kwargs["ExclusiveStartKey"] = start_key
        resp = _table.query(**kwargs)
        items.extend(resp.get("Items", []))
        start_key = resp.get("LastEvaluatedKey")
        if not start_key:
            break

    profiles: List[VoiceProfile] = []
    for item in items:
        if item.get("item_type") == "VOICE_PROFILE":
            try:
                profiles.append(VoiceProfile.from_item(item))
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to parse voice profile item: %s", exc)
                continue
    return profiles


def list_voice_names(agent_id: str) -> List[str]:
    """Return a sorted list of unique voice profile names for the agent."""
    profiles = list_voice_profiles(agent_id)
    names = sorted({p.name for p in profiles})
    return names


def delete_voice_profile(agent_id: str, name: str) -> bool:
    """Delete all voice profiles for this agent that match the given name."""
    name_norm = name.strip()
    if not name_norm:
        raise ValueError("Voice profile name must not be empty")

    profiles = list_voice_profiles(agent_id)
    deleted_any = False
    for p in profiles:
        if p.name == name_norm:
            _table.delete_item(
                Key={
                    "pk": f"AGENT#{agent_id}",
                    "sk": f"VOICE#{p.voice_id}",
                }
            )
            deleted_any = True

    return deleted_any


# ---------------------------------------------------------------------------
# Embedding-based nearest neighbor
# ---------------------------------------------------------------------------

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(n))
    norm_a = sum(a[i] * a[i] for i in range(n)) ** 0.5
    norm_b = sum(b[i] * b[i] for i in range(n)) ** 0.5
    if norm_a <= 1e-8 or norm_b <= 1e-8:
        return 0.0
    return dot / (norm_a * norm_b)


def find_best_match_by_embedding(
    agent_id: str,
    embedding: List[float],
    min_similarity: float = 0.8,
) -> Optional[VoiceProfile]:
    """Return the best matching VoiceProfile for the given embedding, or None."""
    profiles = list_voice_profiles(agent_id)
    best: Optional[VoiceProfile] = None
    best_sim = min_similarity
    for p in profiles:
        embs = p.metadata.get("embeddings") or []
        if not isinstance(embs, list):
            continue
        for e in embs:
            if not isinstance(e, list):
                continue
            vec = [float(x) for x in e]
            sim = _cosine_similarity(vec, embedding)
            if sim > best_sim:
                best_sim = sim
                best = p
    return best


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def resolve_voice(
    agent_id: str,
    voice_id: Optional[str],
    claimed_name: Optional[str] = None,
    embedding: Optional[List[float]] = None,
    similarity_threshold: float = 0.8,
) -> Tuple[Optional[str], bool]:
    """Resolve a voice to a name, using voice_id and/or embedding.

    Returns (name, is_new):

    - If voice_id is known: returns stored name, is_new=False.
    - Else if embedding matches an existing profile: returns that name, is_new=False,
      and creates/updates a mapping for this voice_id if provided.
    - Else if claimed_name is provided: creates a new profile and returns (claimed_name, True).
    - Else: (None, True) signalling that the caller should ask for the speaker's name.
    """
    # 1. If we know the voice_id directly, that's the strongest signal.
    if voice_id:
        existing = get_by_voice_id(agent_id, voice_id)
        if existing:
            logger.debug("Resolved voice_id=%s to existing profile name=%s", voice_id, existing.name)
            if embedding:
                existing.metadata = _append_embedding(existing.metadata, embedding)
                _table.put_item(Item=existing.to_item())
            else:
                touch_voice_profile(agent_id, voice_id)
            return existing.name, False

    # 2. Try nearest-neighbor on embedding.
    match: Optional[VoiceProfile] = None
    if embedding:
        match = find_best_match_by_embedding(agent_id, embedding, min_similarity=similarity_threshold)

    if match:
        logger.debug(
            "Embedding matched existing voice profile agent=%s voice_id=%s name=%s",
            agent_id,
            match.voice_id,
            match.name,
        )
        vid = voice_id or f"embed-{int(datetime.now(timezone.utc).timestamp())}"
        meta = _append_embedding(match.metadata, embedding)
        upsert_voice_profile(agent_id, vid, match.name, meta)
        return match.name, False

    # 3. No match; register a new profile if we have a claimed name.
    if claimed_name:
        vid = voice_id or f"voice-{int(datetime.now(timezone.utc).timestamp())}"
        meta: Dict[str, Any] = {}
        if embedding:
            meta = _append_embedding(meta, embedding)
        profile = upsert_voice_profile(agent_id, vid, claimed_name, meta)
        logger.debug(
            "Registered new voice profile agent=%s voice_id=%s name=%s",
            agent_id,
            vid,
            profile.name,
        )
        return profile.name, True

    # 4. Unknown voice; caller should ask for the speaker's name.
    logger.debug("Unknown voice (no match, no claimed name).")
    return None, True


__all__ = [
    "VoiceProfile",
    "get_by_voice_id",
    "upsert_voice_profile",
    "touch_voice_profile",
    "list_voice_profiles",
    "list_voice_names",
    "delete_voice_profile",
    "find_best_match_by_embedding",
    "resolve_voice",
]
