"""Conversation memory helpers backed by DynamoDB."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import boto3
from boto3.dynamodb.conditions import Key


@dataclass(slots=True)
class ConversationTurn:
    """Represents a single exchange in the conversation."""

    role: str
    content: str
    timestamp: datetime

    def to_item(self, session_id: str, sequence: int) -> dict[str, Any]:
        return {
            "sessionId": session_id,
            "sequence": sequence,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
        }


class ConversationStore:
    """Wrapper around DynamoDB for persisting conversation turns."""

    def __init__(self, table_name: str, region_name: str) -> None:
        self._table = boto3.resource("dynamodb", region_name=region_name).Table(table_name)

    def fetch_history(self, session_id: str, limit: int) -> list[ConversationTurn]:
        response = self._table.query(
            KeyConditionExpression=Key("sessionId").eq(session_id),
            Limit=limit,
            ScanIndexForward=False,
        )
        items = response.get("Items", [])
        history = [
            ConversationTurn(
                role=item["role"],
                content=item["content"],
                timestamp=datetime.fromisoformat(item["timestamp"]),
            )
            for item in items
        ]
        history.reverse()
        return history

    def append_turns(self, session_id: str, turns: Iterable[ConversationTurn]) -> None:
        with self._table.batch_writer(overwrite_by_pkeys=("sessionId", "sequence")) as batch:
            for index, turn in enumerate(turns, start=int(datetime.now(timezone.utc).timestamp())):
                batch.put_item(Item=turn.to_item(session_id=session_id, sequence=index))


__all__ = ["ConversationStore", "ConversationTurn"]
