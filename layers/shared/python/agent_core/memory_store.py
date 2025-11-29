from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import boto3
from boto3.dynamodb.conditions import Key

from .schema import Event, Memory, MemoryKind


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Table wiring
# ---------------------------------------------------------------------------


_TABLE_NAME = os.environ.get("AGENT_STATE_TABLE")
if not _TABLE_NAME:
    # It is better to fail loudly than to silently drop state.
    raise RuntimeError("AGENT_STATE_TABLE environment variable is not set")

_dynamo = boto3.resource("dynamodb")
_table = _dynamo.Table(_TABLE_NAME)


def put_event(event: Event) -> None:
    """
    Persist an Event object into the DynamoDB table.

    Event.to_item() is responsible for producing the correct
    pk/sk layout for the unified state table.
    """
    item = event.to_item()
    logger.debug("Persisting event %s for agent %s", event.event_id, event.agent_id)
    _table.put_item(Item=item)


def put_memory(memory: Memory) -> None:
    """
    Persist a Memory object into the DynamoDB table.

    Memory.to_item() is responsible for producing the correct
    pk/sk/gsi layout and tagging with memory_kind, etc.
    """
    item = memory.to_item()
    logger.debug(
        "Persisting memory %s (kind=%s) for agent %s",
        memory.memory_id,
        memory.memory_kind,
        memory.agent_id,
    )
    _table.put_item(Item=item)


def recent_memories(
    agent_id: str,
    limit: int = 50,
    kinds: Optional[List[MemoryKind]] = None,
) -> List[Dict[str, Any]]:
    """
    Return up to `limit` most recent memory items for this agent.

    We rely on the key layout produced by Memory.to_item():

        pk = "AGENT#{agent_id}"
        sk = "TS#<iso-ts>#MEM#<memory_id>"

    So we query by pk and prefix on "TS#" and then filter down to
    MEMORY items (and optionally specific MemoryKind values).
    """
    logger.debug("Querying recent memories (limit=%s, kinds=%s)", limit, kinds)
    resp = _table.query(
        KeyConditionExpression=Key("pk").eq(f"AGENT#{agent_id}")
        & Key("sk").begins_with("TS#"),
        ScanIndexForward=False,  # newest first
        Limit=limit * 3,         # over-fetch a bit for post-filtering
    )
    items = [i for i in resp.get("Items", []) if i.get("item_type") == "MEMORY"]

    if kinds:
        allowed = {k.value for k in kinds}
        items = [i for i in items if i.get("memory_kind") in allowed]

    # Keep newest-first as the calling code expects; trim to limit
    trimmed = items[:limit]
    logger.debug("Returning %d memories after filtering", len(trimmed))
    return trimmed



def query_memories(
    agent_id: str,
    query: str,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """
    Very simple text search over recent memories for an agent.

    We grab a window of recent MEMORY items and then filter in Python
    by checking the query substring against:

      - `summary` (if present)
      - any string values in `content` (if it's a dict)
    """
    logger.debug("Querying memories for substring '%s' (limit=%s)", query, limit)
    resp = _table.query(
        KeyConditionExpression=Key("pk").eq(f"AGENT#{agent_id}")
        & Key("sk").begins_with("TS#"),
        ScanIndexForward=False,
        Limit=limit * 10,  # over-fetch since we filter in Python
    )
    items = [i for i in resp.get("Items", []) if i.get("item_type") == "MEMORY"]

    q = query.lower()
    filtered: List[Dict[str, Any]] = []

    for item in items:
        text_chunks: List[str] = []

        # Optional summary
        summary = item.get("summary")
        if isinstance(summary, str):
            text_chunks.append(summary)

        # Content dict with arbitrary subfields
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

    logger.debug("Found %d memories matching query", len(filtered))
    return filtered
