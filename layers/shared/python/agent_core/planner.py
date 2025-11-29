from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from .schema import Event, Memory, MemoryKind
from . import memory_store
from .actions import actions_from_tool_calls, Action


logger = logging.getLogger(__name__)


def _coerce_memory_kind(value: str) -> MemoryKind:
    try:
        return MemoryKind(value)
    except ValueError:
        return MemoryKind.META


def _parse_tool_args(raw_args: Any) -> Dict[str, Any]:
    if raw_args is None:
        return {}
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        try:
            return json.loads(raw_args)
        except json.JSONDecodeError:
            return {"_raw": raw_args}
    return {"_raw": raw_args}


def handle_llm_result(
    llm_response: Dict[str, Any],
    event: Event,
) -> Tuple[List[Action], List[Memory], str]:
    """Interpret LLM output into actions + memories + speech.

    Returns (actions, memories, reply_text).
    """
    messages = llm_response.get("messages") or []
    if not messages:
        return [], [], ""

    logger.debug("Handling LLM response with %d messages", len(messages))
    last = messages[-1]
    reply_text = last.get("content") or ""
    tool_calls = last.get("tool_calls") or []
    logger.debug("Assistant proposed %d tool calls", len(tool_calls))

    new_memories: List[Memory] = []
    all_tool_calls: List[Dict[str, Any]] = []

    for tc in tool_calls:
        name = tc.get("name")
        args = _parse_tool_args(tc.get("arguments"))

        if name == "store_memory":
            kind_str = args.get("memory_kind", "META")
            mem = Memory(
                agent_id=event.agent_id,
                session_id=event.session_id,
                memory_kind=_coerce_memory_kind(kind_str),
                source=event.source,
                ts=datetime.now(timezone.utc),
                summary=args.get("summary", ""),
                content=args.get("content", {}),
                tags=args.get("tags") or [],
                links=args.get("links") or [],
            )
            new_memories.append(mem)
            logger.debug("Prepared new memory of kind %s from tool call", kind_str)

        elif name == "query_memory":
            query = args.get("query", "")
            if query:
                mem_items = memory_store.query_memories(
                    agent_id=event.agent_id,
                    query=query,
                    limit=30,
                )
                # We don't automatically modify reply_text here; the model
                # should be called again with these results if you want
                # multi-step planning. For now we just expose them as a
                # synthetic memory.
                synthetic = Memory(
                    agent_id=event.agent_id,
                    session_id=event.session_id,
                    memory_kind=MemoryKind.META,
                    source="system",
                    ts=datetime.now(timezone.utc),
                    summary=f"Query results for: {query}",
                    content={"query": query, "results": mem_items},
                    tags=["query_result"],
                    links=[],
                )
                new_memories.append(synthetic)
                logger.debug(
                    "Added synthetic memory containing %d query results", len(mem_items)
                )

        else:
            all_tool_calls.append({"name": name, "arguments": args})

    logger.debug(
        "Dispatching %d tool calls to action dispatcher", len(all_tool_calls)
    )
    actions = actions_from_tool_calls(all_tool_calls)
    return actions, new_memories, reply_text
