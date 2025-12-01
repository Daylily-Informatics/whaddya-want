from __future__ import annotations

from typing import Any, Dict, List


TOOLS_SPEC: List[Dict[str, Any]] = [
    {
        "name": "store_memory",
        "description": (
            "Store a durable memory about the current context. "
            "Use FACT for user-stated or observed truths, SPECULATION for guesses, "
            "and AI_INSIGHT for higher-level patterns."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "memory_kind": {
                    "type": "string",
                    "enum": ["FACT", "SPECULATION", "AI_INSIGHT", "ACTION", "META"],
                },
                "summary": {"type": "string"},
                "content": {"type": "object"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                },
                "links": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "event_id": {"type": "string"},
                            "relationship": {"type": "string"},
                        },
                        "required": ["event_id", "relationship"],
                    },
                    "default": [],
                },
            },
            "required": ["memory_kind", "summary", "content"],
        },
    },
    {
        "name": "query_memory",
        "description": (
            "Retrieve memories related to a topic. "
            "Use this when the user asks what you remember about something."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "schedule_task",
        "description": (
            "Request that the system schedule a future task (heartbeat, reminder, "
            "background check, etc.). The underlying implementation may use cron, "
            "EventBridge rules, or another scheduler."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {"type": "string"},
                "when": {"type": "string", "description": "Human-readable time description"},
            },
            "required": ["reason", "when"],
        },
    },
    {
        "name": "perform_action",
        "description": (
            "Act on the outside world (notifications, lights, emails, etc.). "
            "Some actions may require user confirmation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action_type": {"type": "string"},
                "payload": {"type": "object"},
                "require_user_confirm": {"type": "boolean", "default": True},
            },
            "required": ["action_type", "payload"],
        },
    },
]
