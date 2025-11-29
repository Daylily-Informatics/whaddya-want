from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

import boto3
from boto3.dynamodb.conditions import Key, Attr

from .schema import Event, Memory, MemoryKind


_TABLE_NAME = os.environ.get("AGENT_STATE_TABLE")
if not _TABLE_NAME:
    # It is better to fail loudly than to silently drop state.
    raise RuntimeError("AGENT_STATE_TABLE environment variable is not set")

_dynamo = boto3.resource("dynamodb")
_table = _dynamo.Table(_TABLE_NAME)


def put_event(event: Event) -> None:
    _table.put_item(Item=event.to_item())


def put_memory(memory: Memory) -> None:
    _table.put_item(Item=memory.to_item())


def recent_memories(
    agent_id: str,
    limit: int = 50,
    kinds: Optional[List[MemoryKind]] = None,
) -> List[Dict[str, Any]]:
    resp = _table.query(
        KeyConditionExpression=Key("pk").eq(f"AGENT#{agent_id}") & Key("sk").begins_with("TS#"),
        ScanIndexForward=False,
        Limit=limit * 3,
    )
    items = [i for i in resp.get("Items", []) if i.get("item_type") == "MEMORY"]
    if kinds:
        allowed = {k.value for k in kinds}
        items = [i for i in items if i.get("memory_kind") in allowed]
    return items[:limit]


def query_memories(
    agent_id: str,
    query: str,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    # Very simple implementation: scan recent items and filter by substring
    resp = _table.query(
        KeyConditionExpression=Key("pk").eq(f"AGENT#{agent_id}") & Key("sk").begins_with("TS#"),
        ScanIndexForward=False,
        Limit=limit * 10,
    )
    items = [i for i in resp.get("Items", []) if i.get("item_type") == "MEMORY"]
    q = query.lower()
    filtered: List[Dict[str, Any]] = []
    for item in items:
        text_chunks: List[str] = []
        if "summary" in item and isinstance(item["summary"], str):
            text_chunks.append(item["summary"])
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
