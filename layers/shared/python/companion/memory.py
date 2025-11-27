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
from typing import Any, Dict, List, Tuple

import json
import re
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
        """Return up to ``limit`` relevant exchanges for a session.

        Strategy:
        - Query up to a few hundred most recent items for this session_id.
        - Normalize the query into crude tokens (lowercased, stripped of punctuation,
          and with trailing 's' removed to collapse simple plural/singular).
        - Score each item by how many tokens appear in its user+assistant text.
        - Take the top-ranked items.
        - If there are not enough scored items, pad with the newest items so we
          still give the model some context even when the query is vague.
        """
        if not session_id:
            return []

        try:
            resp = self._table.query(
                KeyConditionExpression=Key("session_id").eq(session_id),
                ScanIndexForward=False,  # newest first
                Limit=500,               # allow deeper history within this session
            )
        except Exception:
            # On any query failure, do not break the main flow; just return nothing.
            return []

        items = resp.get("Items", []) or []
        if not items:
            return []

        # --- Normalize query into tokens -------------------------------------
        query_norm = (query or "").strip().lower()
        if query_norm:
            # remove most punctuation
            q_clean = re.sub(r"[^\w\s]", " ", query_norm)
            raw_tokens = q_clean.split()

            stopwords = {
                "the", "a", "an", "and", "or", "but",
                "to", "for", "of", "in", "on", "at", "about",
                "me", "you", "we", "i", "our", "your",
                "is", "are", "was", "were", "be", "been",
                "do", "did", "does", "have", "has", "had",
                "this", "that", "these", "those",
                "what", "which", "when", "where", "why", "how",
                "past", "before", "earlier", "remind", "remember",
            }
            tokens: set[str] = set()
            for t in raw_tokens:
                if not t or t in stopwords:
                    continue
                # crude singular/plural collapse
                base = t.rstrip("s")
                if base:
                    tokens.add(base)
        else:
            tokens = set()

        # --- Score items -----------------------------------------------------
        if not tokens:
            # No meaningful tokens: just take newest items as a fallback.
            selected = items[:limit]
        else:
            scored: list[Tuple[int, Dict[str, Any]]] = []
            for it in items:
                user = (it.get("user") or {}).get("content") or ""
                assistant = (it.get("assistant") or {}).get("content") or ""
                text = f"{user} {assistant}".lower()
                if not text:
                    continue
                # normalize text like we did for the query
                t_clean = re.sub(r"[^\w\s]", " ", text)
                # crude singular/plural collapse on the fly
                t_clean = " ".join(w.rstrip("s") for w in t_clean.split())
                score = sum(1 for tok in tokens if tok and tok in t_clean)
                if score > 0:
                    scored.append((score, it))

            scored.sort(key=lambda pair: pair[0], reverse=True)
            selected: List[Dict[str, Any]] = [it for _, it in scored[:limit]]

            # If we didn't get enough scored items, pad with newest unseen items.
            if len(selected) < limit:
                seen = set(id(it) for it in selected)
                for it in items:
                    if id(it) in seen:
                        continue
                    selected.append(it)
                    seen.add(id(it))
                    if len(selected) >= limit:
                        break

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
