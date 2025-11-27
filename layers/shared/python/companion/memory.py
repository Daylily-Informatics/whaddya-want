"""Conversation memory helpers backed by DynamoDB.

This implementation is compatible with the existing `whaddyawant_session_memory`
table, which stores a single item per session:

    {
      "session_id": "...",
      "turns": [
        {"role": "user", "content": "...", "timestamp": "..."},
        {"role": "assistant", "content": "...", "timestamp": "..."},
        ...
      ],
      "ttl": <epoch-seconds>
    }

Older deployments may still have an earlier format where each element of
`turns` is a {"user": "...", "assistant": "..."} pair.

The AIS long-term memory store is separate, with one item per exchange:

    {
      "session_id": "...",
      "timestamp": "...",
      "user": {...},
      "assistant": {...},
      "metadata": {...},
      "ttl": <epoch-seconds>
    }
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List

import boto3
from boto3.dynamodb.conditions import Key


@dataclass(slots=True)
class ConversationTurn:
    """Represents a single exchange in the conversation."""

    role: str
    content: str
    timestamp: datetime


class ConversationStore:
    """Wrapper around DynamoDB for persisting conversation turns."""

    _TTL_DAYS = 7

    def __init__(self, table_name: str, region_name: str) -> None:
        self._table = boto3.resource("dynamodb", region_name=region_name).Table(table_name)

    # ---- Internal helpers ----
    @staticmethod
    def _decode_turns(raw_turns: Any) -> List[ConversationTurn]:
        """Decode stored `turns` data into a flat list of ConversationTurn.

        Supports both the new (role/content/timestamp) format and the
        legacy {"user": ..., "assistant": ...} pair format.
        """
        turns: list[ConversationTurn] = []
        if not isinstance(raw_turns, list):
            return turns

        # Default timestamp if none is stored
        now = datetime.now(timezone.utc)

        for entry in raw_turns:
            if not isinstance(entry, dict):
                continue
            # New format
            if "role" in entry and "content" in entry:
                ts_raw = entry.get("timestamp")
                try:
                    ts = datetime.fromisoformat(ts_raw) if ts_raw else now
                except Exception:
                    ts = now
                turns.append(
                    ConversationTurn(
                        role=str(entry["role"]),
                        content=str(entry["content"]),
                        timestamp=ts,
                    )
                )
                continue

            # Legacy format: {"user": "...", "assistant": "..."}
            user_text = entry.get("user")
            assistant_text = entry.get("assistant")
            if user_text is not None:
                turns.append(
                    ConversationTurn(
                        role="user",
                        content=str(user_text),
                        timestamp=now,
                    )
                )
            if assistant_text is not None:
                turns.append(
                    ConversationTurn(
                        role="assistant",
                        content=str(assistant_text),
                        timestamp=now,
                    )
                )

        return turns

    @staticmethod
    def _encode_turns(turns: List[ConversationTurn]) -> List[Dict[str, Any]]:
        """Encode ConversationTurn objects into a JSON-serializable list."""
        encoded: list[dict[str, Any]] = []
        for t in turns:
            encoded.append(
                {
                    "role": t.role,
                    "content": t.content,
                    "timestamp": t.timestamp.isoformat(),
                }
            )
        return encoded

    # ---- Public API ----
    def fetch_history(self, *, session_id: str, limit: int) -> List[ConversationTurn]:
        """Fetch up to `limit` recent turns for a session."""
        try:
            resp = self._table.get_item(Key={"session_id": session_id})
        except Exception:
            return []

        item = resp.get("Item")
        if not item:
            return []

        turns = self._decode_turns(item.get("turns"))
        if limit > 0 and len(turns) > limit:
            turns = turns[-limit:]
        return turns

    def append_turns(
        self,
        *,
        session_id: str,
        turns: List[ConversationTurn],
        limit: int,
    ) -> None:
        """Append new turns to the stored history for a session.

        This method re-fetches the existing history to keep things simple.
        For high-throughput workloads you might want to switch to a more
        sophisticated append-only model.
        """
        existing = self.fetch_history(session_id=session_id, limit=0)
        new_turns = existing + list(turns)
        if limit > 0 and len(new_turns) > limit:
            new_turns = new_turns[-limit:]

        now = datetime.now(timezone.utc)
        ttl = int(now.timestamp()) + self._TTL_DAYS * 24 * 3600
        item = {
            "session_id": session_id,
            "turns": self._encode_turns(new_turns),
            "ttl": ttl,
        }
        self._table.put_item(Item=item)


class AISLongTermMemoryStore:
    """Persists full exchanges into an AIS long-term memory table."""

    def __init__(self, table_name: str, region_name: str, ttl_days: int = 0) -> None:
        self._table = boto3.resource("dynamodb", region_name=region_name).Table(table_name)
        self._ttl_days = max(ttl_days, 0)

    @staticmethod
    def _json_sanitize(data: Dict[str, Any] | None) -> Dict[str, Any]:
        if not data:
            return {}
        try:
            return json.loads(json.dumps(data, default=str))
        except Exception:
            # Fallback: stringify everything if something could not be serialized
            return {"raw": str(data)}

    def record_exchange(
        self,
        *,
        session_id: str,
        timestamp: datetime,
        user_text: str,
        assistant_text: str,
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        item: Dict[str, Any] = {
            "session_id": session_id,
            "timestamp": timestamp.isoformat(),
            "user": {
                "role": "user",
                "content": user_text,
            },
            "assistant": {
                "role": "assistant",
                "content": assistant_text,
            },
        }

        sanitized = self._json_sanitize(metadata)
        if sanitized:
            item["metadata"] = sanitized

        if self._ttl_days:
            item["ttl"] = int(timestamp.timestamp()) + self._ttl_days * 24 * 3600

        self._table.put_item(Item=item)

    def search_exchanges(
        self,
        session_id: str,
        query: str | None = None,
        limit: int = 8,
    ) -> List[Dict[str, Any]]:
        """Return up to ``limit`` recent exchanges for a session.

        This is a very simple, in-memory scoring over the most recent items for
        the given ``session_id``. If ``query`` is provided, we rank by the
        number of query tokens that appear in the user/assistant text; if no
        match is found, we fall back to the newest items.
        """
        if not session_id:
            return []

        try:
            resp = self._table.query(
                KeyConditionExpression=Key("session_id").eq(session_id),
                ScanIndexForward=False,  # newest first
                Limit=100,
            )
        except Exception:
            # On any query failure, do not break the main flow; just return nothing.
            return []

        items = resp.get("Items", []) or []

        query_norm = (query or "").strip().lower()
        if not query_norm:
            selected = items[:limit]
        else:
            tokens = {t for t in query_norm.split() if t}
            scored: list[tuple[int, Dict[str, Any]]] = []
            for it in items:
                user = (it.get("user") or {}).get("content") or ""
                assistant = (it.get("assistant") or {}).get("content") or ""
                text = f"{user} {assistant}".lower()
                if not text:
                    continue
                score = sum(1 for t in tokens if t in text)
                if score > 0:
                    scored.append((score, it))

            if scored:
                scored.sort(key=lambda pair: pair[0], reverse=True)
                selected = [it for _, it in scored[:limit]]
            else:
                selected = items[:limit]

        # Present oldest -> newest for readability
        try:
            selected.sort(key=lambda it: it.get("timestamp") or "")
        except Exception:
            pass

        normalized: List[Dict[str, Any]] = []
        for it in selected:
            normalized.append(
                {
                    "session_id": it.get("session_id"),
                    "timestamp": it.get("timestamp"),
                    "user": it.get("user") or {},
                    "assistant": it.get("assistant") or {},
                    "metadata": it.get("metadata") or {},
                }
            )
        return normalized


__all__ = ["AISLongTermMemoryStore", "ConversationStore", "ConversationTurn"]
