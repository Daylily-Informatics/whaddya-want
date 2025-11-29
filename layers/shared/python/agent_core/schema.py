from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List
from datetime import datetime, timezone
import uuid


class MemoryKind(str, Enum):
    FACT = "FACT"
    SPECULATION = "SPECULATION"
    AI_INSIGHT = "AI_INSIGHT"
    ACTION = "ACTION"
    META = "META"


@dataclass
class Event:
    agent_id: str
    session_id: str
    source: str
    channel: str
    ts: datetime
    payload: Dict[str, Any]
    event_id: str | None = None

    def to_item(self) -> Dict[str, Any]:
        if not self.event_id:
            self.event_id = str(uuid.uuid4())
        return {
            "pk": f"AGENT#{self.agent_id}",
            "sk": f"TS#{self.ts.isoformat()}#EVT#{self.event_id}",
            "gsi1pk": f"SESSION#{self.session_id}",
            "gsi1sk": self.ts.isoformat(),
            "item_type": "EVENT",
            "source": self.source,
            "channel": self.channel,
            "payload": self.payload,
        }


@dataclass
class Memory:
    agent_id: str
    session_id: str
    memory_kind: MemoryKind
    source: str
    ts: datetime
    summary: str
    content: Dict[str, Any]
    tags: List[str]
    links: List[Dict[str, str]]
    memory_id: str | None = None

    def to_item(self) -> Dict[str, Any]:
        if not self.memory_id:
            self.memory_id = str(uuid.uuid4())
        return {
            "pk": f"AGENT#{self.agent_id}",
            "sk": f"TS#{self.ts.isoformat()}#MEM#{self.memory_id}",
            "gsi1pk": f"SESSION#{self.session_id}",
            "gsi1sk": self.ts.isoformat(),
            "item_type": "MEMORY",
            "memory_kind": self.memory_kind.value,
            "source": self.source,
            "summary": self.summary,
            "content": self.content,
            "tags": self.tags,
            "links": self.links,
        }
