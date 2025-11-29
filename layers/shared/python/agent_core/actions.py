from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class Action:
    action_type: str
    payload: Dict[str, Any]
    require_user_confirm: bool = True


def actions_from_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[Action]:
    actions: List[Action] = []
    for call in tool_calls:
        name = call.get("name")
        args = call.get("arguments") or {}
        if name == "perform_action":
            actions.append(
                Action(
                    action_type=args.get("action_type", "UNKNOWN"),
                    payload=args.get("payload", {}),
                    require_user_confirm=bool(args.get("require_user_confirm", True)),
                )
            )
        elif name == "schedule_task":
            actions.append(
                Action(
                    action_type="SCHEDULE_TASK",
                    payload=args,
                    require_user_confirm=False,
                )
            )
        # store_memory and query_memory are handled in planner; we skip here.
    return actions


def dispatch_background_actions(actions: List[Action]) -> None:
    # Stub: hook this into SNS, SQS, email, home automation, etc.
    # For now we just log to stdout so you can see what the agent wants to do.
    if not actions:
        return
    print("[agent_core.actions] Dispatching actions:")
    for act in actions:
        print(f"  - {act.action_type} (confirm={act.require_user_confirm}): {act.payload}")
