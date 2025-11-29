from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import boto3
from boto3.dynamodb.conditions import Key

from .schema import Event, Memory, MemoryKind


# ---------------------------------------------------------------------------
# Table wiring
# ---------------------------------------------------------------------------

_TABLE_NAME = os.environ.get("AGENT_STATE_TABLE")
if not _TABLE_NAME:
    # It is better to fail loudly than to silently drop state.
    raise RuntimeError("AGENT_STATE_TABLE environment variable is not set")

# Let boto3 resolve region from env / config
_dynamo = boto3.resource("dynamodb")
_table = _dynamo.Table(_TABLE_NAME)


# ---------------------------------------------------------------------------
# Write APIs: operate on schema objects
# ---------------------------------------------------------------------------

def put_event(event: Event) -> None:
    """
    Persist an Event into the unified AgentStateTable.

    The Event dataclass is responsible for producing the correct
    DynamoDB item via its .to_item() method.
    """
    _table.put_item(Item=event.to_item())


def put_memory(memory: Memory) -> None:
    """
    Persist a Memory into the unified AgentStateTable.

    The Memory dataclass is responsible for producing the correct
    DynamoDB item via its .to_item() method.
    """
    _table.put_item(Item=memory.to_item())


# ---------------------------------------------------------------------------
# Read/query APIs
# ---------------------------------------------------------------------------

def recent_memories(
    agent_id: str,
    limit: int = 50,
    kinds: Optional[List[MemoryKind]] = None,
) -> List[Dict[str, Any]]:
    """
    Return up to `limit` most recent memories for a given agent.

    We rely on the key layout produced by Memory.to_item():

        pk = "AGENT#{agent_id}"
        sk = "TS#<iso-ts>#MEM#<memory_id>"

    So we query by pk and prefix on "TS#" and then filter by item_type
    and (optionally) memory_kind.
    """
    resp = _table.query(
        KeyConditionExpression=Key("pk").eq(f"AGENT#{agent_id}")
        & Key("sk").begins_with("TS#"),
        ScanIndexForward=False,      # newest first
        Limit=limit * 3,             # over-fetch a bit for post-filtering
    )
    items = [i for i in resp.get("Items", []) if i.get("item_type") == "MEMORY"]

    if kinds:
        allowed = {k.value for k in kinds}
        items = [i for i in items if i.get("memory_kind") in allowed]

    # Keep newest-first as the calling code expects; trim to limit
    return items[:limit]


def query_memories(
    agent_id: str,
    query: str,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """
    Very simple implementation: scan recent items and filter by substring.

    We search across:
      - `summary` if present
      - any string values in `content` dict
    """
    resp = _table.query(
        KeyConditionExpression=Key("pk").eq(f"AGENT#{agent_id}")
        & Key("sk").begins_with("TS#"),
        ScanIndexForward=False,
        Limit=limit * 10,   # over-fetch since we'll filter in Python
    )
    items = [i for i in resp.get("Items", []) if i.get("item_type") == "MEMORY"]

    q = query.lower()
    filtered: List[Dict[str, Any]] = []

    for item in items:
        text_chunks: List[str] = []

        # Optional summary field
        summary = item.get("summary")
        if isinstance(summary, str):
            text_chunks.append(summary)

        # Content can be a dict with arbitrary subfields
        content = item.get("content")
        if isinstance(content, dict):
            for v in content.values():
                if isinstance(v, str):
                    text_chunks.append(v)

        joined = " ".join(text_chunks).lower()
        if q in joined:
            filtered.append(item)
            if len(filtered) >= limit:
                break

    return filtered
