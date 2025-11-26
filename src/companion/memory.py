"""Conversation memory helpers backed by DynamoDB."""
from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import boto3


@dataclass(slots=True)
class ConversationTurn:
    """Represents a single exchange in the conversation."""

    role: str
    content: str
    timestamp: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationTurn":
        return cls(
            role=data.get("role", "user"),
            content=data.get("content", ""),
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )


class ConversationStore:
    """Wrapper around DynamoDB for persisting conversation turns."""

    def __init__(self, table_name: str, region_name: str, ttl_seconds: int, history_limit: int = 100) -> None:
        self._table = boto3.resource("dynamodb", region_name=region_name).Table(table_name)
        self._ttl_seconds = ttl_seconds
        self._max_turns = max(1, history_limit)

    def fetch_history(self, session_id: str, limit: int) -> list[ConversationTurn]:
        response = self._table.get_item(Key={"session_id": session_id})
        item = response.get("Item") or {}
        turns = item.get("turns") or []
        history = []
        for raw in turns[-limit:]:
            try:
                history.append(ConversationTurn.from_dict(raw))
            except Exception:
                continue
        return history[-limit:]

    def append_turns(self, session_id: str, turns: Iterable[ConversationTurn]) -> None:
        existing = self.fetch_history(session_id=session_id, limit=self._max_turns)
        merged = [*existing, *list(turns)]
        limited = merged[-self._max_turns :]
        now_ts = int(time.time())
        payload = {
            "session_id": session_id,
            "turns": [t.to_dict() for t in limited],
            "updated_at": datetime.utcfromtimestamp(now_ts).isoformat() + "Z",
            "ttl": now_ts + self._ttl_seconds,
        }
        self._table.put_item(Item=payload)


__all__ = ["ConversationStore", "ConversationTurn"]
