"""
Unified state + memory store for the rank-4 agent.

Backed by the DynamoDB table whose name is provided via the
AGENT_STATE_TABLE environment variable and schema:

    pk      : partition key (S)
    sk      : sort key (S)
    gsi1pk  : GSI hash key (S)
    gsi1sk  : GSI range key (S)

We store both events and memories in this table.

Keying strategy
---------------
For all items we use:

    pk = f"AGENT#{agent_id}"

Events:

    sk      = f"EVENT#{timestamp_iso}"
    gsi1pk  = f"EVENT#{agent_id}"
    gsi1sk  = timestamp_iso

Memories:

    sk      = f"MEMORY#{timestamp_iso}"
    gsi1pk  = f"MEMORY#{agent_id}"
    gsi1sk  = timestamp_iso

Session information and other metadata are stored as attributes
(`session_id`, `memory_type`, `source`, etc.), not in the keys,
so we can still filter by session when desired without breaking
the GSI.

Public API (imported by agent_core.__init__):

    put_event(...)
    put_memory(...)
    recent_memories(...)
    query_memories(...)

These are written to be tolerant of existing call sites:

    put_memory("FACT", "some content")
    recent_memories(limit=10)

and optional keyword arguments for richer metadata:

    put_memory("FACT", "you have a dog named Chunk",
               source="user",
               session_id="jemxx1")
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import boto3
from boto3.dynamodb.conditions import Key

_TABLE = None  # cached DynamoDB Table object


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    """Return current UTC time in ISO 8601 format without microseconds."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _get_table():
    """Return the DynamoDB Table object, or raise if env is missing."""
    global _TABLE
    if _TABLE is not None:
        return _TABLE

    table_name = os.environ.get("AGENT_STATE_TABLE")
    if not table_name:
        raise RuntimeError("AGENT_STATE_TABLE environment variable is not set")

    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if not region:
        # Let boto3’s default resolution handle it (profile/config); this is fine
        dynamodb = boto3.resource("dynamodb")
    else:
        dynamodb = boto3.resource("dynamodb", region_name=region)

    _TABLE = dynamodb.Table(table_name)
    return _TABLE


def _resolve_agent_id(agent_id: Optional[str]) -> str:
    """Resolve agent_id from argument or env; fall back to 'default-agent'."""
    if agent_id:
        return agent_id
    env_agent = os.environ.get("AGENT_ID") or os.environ.get("AGENT") or ""
    return env_agent or "default-agent"


def _resolve_session_id(session_id: Optional[str]) -> str:
    """
    Resolve session_id from argument or env.
    We intentionally keep this as an attribute, not part of the keys.
    """
    if session_id:
        return session_id
    env_session = (
        os.environ.get("SESSION_ID")
        or os.environ.get("SESSION")
        or os.environ.get("CHAT_SESSION")
        or ""
    )
    return env_session or "default-session"


# ---------------------------------------------------------------------------
# Core write primitives
# ---------------------------------------------------------------------------

def put_event(
    event_type: str,
    content: str,
    *,
    source: str = "system",
    metadata: Optional[Dict[str, Any]] = None,
    agent_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Persist a generic event into AgentStateTable.

    Parameters
    ----------
    event_type : str
        High-level type label for the event (e.g., "HEARTBEAT", "ERROR").
    content : str
        Human-readable description of the event.
    source : str, default "system"
        Who produced the event ("system", "user", "ai", etc.).
    metadata : dict, optional
        Free-form JSON-serializable metadata.
    agent_id : str, optional
        Agent identity; defaults from env AGENT_ID or 'default-agent'.
    session_id : str, optional
        Session identity; defaults from env or 'default-session'.

    Returns
    -------
    dict
        The DynamoDB item that was written.
    """
    table = _get_table()
    agent_id = _resolve_agent_id(agent_id)
    session_id = _resolve_session_id(session_id)
    ts = _now_iso()

    item: Dict[str, Any] = {
        "pk": f"AGENT#{agent_id}",
        "sk": f"EVENT#{ts}",
        "gsi1pk": f"EVENT#{agent_id}",
        "gsi1sk": ts,
        "item_type": "EVENT",
        "event_type": event_type,
        "content": content,
        "source": source,
        "agent_id": agent_id,
        "session_id": session_id,
        "created_at": ts,
    }

    if metadata:
        # shallow copy so caller’s dict isn’t mutated
        item["metadata"] = dict(metadata)

    table.put_item(Item=item)
    return item


def put_memory(
    memory_type: str,
    content: str,
    *,
    source: str = "user",
    metadata: Optional[Dict[str, Any]] = None,
    agent_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Persist a memory item (FACT, SPECULATION, AI_INSIGHT) into AgentStateTable.

    Parameters
    ----------
    memory_type : str
        One of "FACT", "SPECULATION", "AI_INSIGHT" (or a future extension).
    content : str
        The natural-language content of the memory.
    source : str, default "user"
        Who asserted this memory ("user", "ai", "vision", etc.).
    metadata : dict, optional
        Free-form JSON-serializable metadata (e.g., source image id).
    agent_id : str, optional
        Agent identity; defaults from env AGENT_ID or 'default-agent'.
    session_id : str, optional
        Session identity; defaults from env or 'default-session'.

    Returns
    -------
    dict
        The DynamoDB item that was written.
    """
    table = _get_table()
    agent_id = _resolve_agent_id(agent_id)
    session_id = _resolve_session_id(session_id)
    ts = _now_iso()

    item: Dict[str, Any] = {
        "pk": f"AGENT#{agent_id}",
        "sk": f"MEMORY#{ts}",
        "gsi1pk": f"MEMORY#{agent_id}",
        "gsi1sk": ts,
        "item_type": "MEMORY",
        "memory_type": memory_type,
        "content": content,
        "source": source,
        "agent_id": agent_id,
        "session_id": session_id,
        "created_at": ts,
    }

    if metadata:
        item["metadata"] = dict(metadata)

    table.put_item(Item=item)
    return item


# ---------------------------------------------------------------------------
# Read/query helpers
# ---------------------------------------------------------------------------

def recent_memories(
    limit: int = 20,
    *,
    agent_id: Optional[str] = None,
    session_id: Optional[str] = None,
    memory_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Return up to `limit` most recent memory items for this agent/session.

    Parameters
    ----------
    limit : int, default 20
        Maximum number of memories to return.
    agent_id : str, optional
        Agent identity; defaults from env AGENT_ID if omitted.
    session_id : str, optional
        Session identity; if provided, we filter to this session_id.
    memory_type : str, optional
        If provided, filter by memory_type (e.g., "FACT").

    Returns
    -------
    list of dict
        Memory items ordered oldest → newest.
    """
    table = _get_table()
    agent_id = _resolve_agent_id(agent_id)
    session_id = _resolve_session_id(session_id)

    # Query the GSI for all MEMORY items for this agent, newest first.
    resp = table.query(
        IndexName="gsi1",
        KeyConditionExpression=Key("gsi1pk").eq(f"MEMORY#{agent_id}"),
        ScanIndexForward=False,  # newest first
        Limit=max(limit * 3, limit),  # grab some extra for filtering
    )
    items: List[Dict[str, Any]] = resp.get("Items", [])

    # Filter by session_id and memory_type in-memory (attributes, not keys)
    filtered: List[Dict[str, Any]] = []
    for item in items:
        if item.get("item_type") != "MEMORY":
            continue
        if session_id and item.get("session_id") not in (session_id, None):
            continue
        if memory_type and item.get("memory_type") != memory_type:
            continue
        filtered.append(item)
        if len(filtered) >= limit:
            break

    # Present oldest→newest which is more natural for prompts
    return list(reversed(filtered))


def query_memories(
    *,
    agent_id: Optional[str] = None,
    session_id: Optional[str] = None,
    memory_type: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """
    More flexible memory query API.

    Parameters
    ----------
    agent_id : str, optional
        Agent identity; defaults from env.
    session_id : str, optional
        If provided, filter to this session_id.
    memory_type : str, optional
        If provided, filter by memory_type (e.g., "FACT").
    since : str, optional
        Lower bound ISO8601 timestamp (inclusive) on created_at.
    until : str, optional
        Upper bound ISO8601 timestamp (inclusive) on created_at.
    limit : int, default 100
        Maximum number of records to return (after filtering).

    Returns
    -------
    list of dict
        Matching memory items ordered oldest → newest.
    """
    table = _get_table()
    agent_id = _resolve_agent_id(agent_id)
    session_id = _resolve_session_id(session_id)

    resp = table.query(
        IndexName="gsi1",
        KeyConditionExpression=Key("gsi1pk").eq(f"MEMORY#{agent_id}"),
        ScanIndexForward=False,
        Limit=max(limit * 5, limit),
    )
    items: List[Dict[str, Any]] = resp.get("Items", [])

    out: List[Dict[str, Any]] = []
    for item in items:
        if item.get("item_type") != "MEMORY":
            continue

        if session_id and item.get("session_id") not in (session_id, None):
            continue
        if memory_type and item.get("memory_type") != memory_type:
            continue

        created_at = item.get("created_at")
        if since and created_at and created_at < since:
            continue
        if until and created_at and created_at > until:
            continue

        out.append(item)
        if len(out) >= limit:
            break

    return list(reversed(out))
