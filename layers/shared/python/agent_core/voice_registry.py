from __future__ import annotations

"""Voice identity registry backed by the AgentStateTable DynamoDB table.

This module provides a small API for registering and looking up voice profiles
for an agent. A *voice profile* links a stable voice identifier (for example,
a speaker embedding ID or device-level voice fingerprint) to a human-readable
name such as "Major" or "John".

Data is stored in the same DynamoDB table as events and memories
(AGENT_STATE_TABLE) using items with:

    pk  = f"AGENT#{agent_id}"
    sk  = f"VOICE#{voice_id}"
    item_type = "VOICE_PROFILE"

The minimal schema for a voice profile item is:

    {
        "pk": "AGENT#marvin",
        "sk": "VOICE#abc123",
        "item_type": "VOICE_PROFILE",
        "name": "Major",
        "voice_id": "abc123",
        "created_at": "2025-11-29T19:00:00+00:00",
        "last_seen_at": "2025-11-29T19:05:00+00:00",
        "metadata": {...}
    }

You can use this from Lambda (broker) and from local tools (via client.identity)
to keep a consistent mapping from voice_id -> name.

This module does *not* perform any acoustic analysis. It assumes that upstream
code (STT pipeline, audio front-end, etc.) can provide a stable voice_id for
each distinct speaker.
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
        return {
            "pk": f"AGENT#{self.agent_id}",
            "sk": f"VOICE#{self.voice_id}",
            "item_type": "VOICE_PROFILE",
            "name": self.name,
            "voice_id": self.voice_id,
            "created_at": self.created_at.isoformat(),
            "last_seen_at": self.last_seen_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_item(cls, item: Dict[str, Any]) -> "VoiceProfile":
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

        return cls(
            agent_id=agent_id,
            voice_id=voice_id,
            name=item.get("name", "Unknown"),
            created_at=created_at,
            last_seen_at=last_seen_at,
            metadata=item.get("metadata") or {},
        )


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
    """Create or update a voice profile for the given voice_id.

    If a profile already exists, its name and metadata are updated and
    last_seen_at is refreshed. Otherwise a new profile is created.
    """
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
            existing.metadata.update(metadata)
        profile = existing
    else:
        logger.debug(
            "Creating new voice profile for agent=%s voice_id=%s name=%s",
            agent_id,
            voice_id,
            name,
        )
        profile = VoiceProfile(
            agent_id=agent_id,
            voice_id=voice_id,
            name=name,
            created_at=now,
            last_seen_at=now,
            metadata=metadata or {},
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
    """Delete all voice profiles for this agent that match the given name.

    Returns True if at least one profile was deleted, False otherwise.
    """
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


def resolve_voice(
    agent_id: str,
    voice_id: Optional[str],
    claimed_name: Optional[str] = None,
) -> Tuple[Optional[str], bool]:
    """Resolve a voice_id to a name, optionally registering a new profile.

    Returns (name, is_new):

    - If voice_id is None, returns (claimed_name, False) and does not
      touch the registry.
    - If voice_id is known, returns (stored_name, False).
    - If voice_id is unknown and claimed_name is provided, registers a
      new profile and returns (claimed_name, True).
    - If voice_id is unknown and claimed_name is None, returns (None, True)
      to signal that the caller should ask the speaker for their name.
    """
    if not voice_id:
        # Nothing to resolve; rely on the claimed_name if provided.
        return claimed_name, False

    existing = get_by_voice_id(agent_id, voice_id)
    if existing:
        touch_voice_profile(agent_id, voice_id)
        return existing.name, False

    if claimed_name:
        profile = upsert_voice_profile(agent_id, voice_id, claimed_name)
        return profile.name, True

    # Unknown voice, no name yet – caller should ask.
    return None, True


__all__ = [
    "VoiceProfile",
    "get_by_voice_id",
    "upsert_voice_profile",
    "touch_voice_profile",
    "list_voice_profiles",
    "list_voice_names",
    "delete_voice_profile",
    "resolve_voice",
]
