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
`turns` was a {"user": "...", "assistant": "..."} pair. We transparently
convert those entries into the new per-message structure on read/write.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, List

import boto3


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
                    ts = datetime.fromisoformat(ts_raw) if isinstance(ts_raw, str) else now
                except Exception:
                    ts = now
                turns.append(
                    ConversationTurn(
                        role=str(entry.get("role") or "assistant"),
                        content=str(entry.get("content") or ""),
                        timestamp=ts,
                    )
                )
                continue

            # Legacy format: a single dict with "user" and "assistant" fields
            user_text = entry.get("user")
            if isinstance(user_text, str) and user_text.strip():
                turns.append(
                    ConversationTurn(
                        role="user",
                        content=user_text,
                        timestamp=now,
                    )
                )
            assistant_text = entry.get("assistant")
            if isinstance(assistant_text, str) and assistant_text.strip():
                turns.append(
                    ConversationTurn(
                        role="assistant",
                        content=assistant_text,
                        timestamp=now,
                    )
                )

        return turns

    @staticmethod
    def _encode_turns(turns: Iterable[ConversationTurn]) -> list[dict[str, Any]]:
        """Encode ConversationTurn objects for storage."""
        out: list[dict[str, Any]] = []
        for t in turns:
            out.append(
                {
                    "role": t.role,
                    "content": t.content,
                    "timestamp": t.timestamp.isoformat(),
                }
            )
        return out

    # ---- Public API ----
    def fetch_history(self, session_id: str, limit: int) -> list[ConversationTurn]:
        resp = self._table.get_item(Key={"session_id": session_id})
        item = resp.get("Item") or {}
        raw_turns = item.get("turns") or []
        turns = self._decode_turns(raw_turns)
        if limit > 0 and len(turns) > limit:
            turns = turns[-limit:]
        return turns

    def append_turns(
        self,
        session_id: str,
        turns: Iterable[ConversationTurn],
        *,
        limit: int,
    ) -> None:
        """Append new turns to the stored history, trimming to `limit`."""
        existing = self.fetch_history(session_id=session_id, limit=limit or 0)
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


__all__ = ["ConversationStore", "ConversationTurn"]
