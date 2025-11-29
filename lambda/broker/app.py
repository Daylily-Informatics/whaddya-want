from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict

from agent_core.schema import Event
from agent_core import memory_store, tools as agent_tools
from agent_core.planner import handle_llm_result
from agent_core.actions import dispatch_background_actions
from agent_core import llm_client
from agent_core.aws_model_client import AwsModelClient


AGENT_ID = os.environ.get("AGENT_ID", "marvin")



# Real model client backed by AWS Bedrock
_model_client = AwsModelClient.from_env()

def _derive_session_id(event: Dict[str, Any]) -> str:
    headers = event.get("headers") or {}
    src = headers.get("X-Session-Id") or headers.get("x-session-id")
    if src:
        return str(src)
    # Fallback: per-invocation session; adapt to your needs.
    return f"lambda-{event.get('requestContext', {}).get('requestId', 'unknown')}"


def _payload_from_event(event: Dict[str, Any]) -> Dict[str, Any]:
    body = event.get("body")
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except json.JSONDecodeError:
            body = {"raw": body}
    if not isinstance(body, dict):
        body = {}
    return body


def handler(event, context):
    body = _payload_from_event(event)
    session_id = _derive_session_id(event)
    source = body.get("source", "user")
    channel = body.get("channel", "audio")
    transcript = body.get("transcript") or body.get("text") or ""

    agent_event = Event(
        agent_id=AGENT_ID,
        session_id=session_id,
        source=source,
        channel=channel,
        ts=datetime.now(timezone.utc),
        payload={"transcript": transcript, "raw": body},
    )
    memory_store.put_event(agent_event)

    # Retrieve recent memories for context
    memories = memory_store.recent_memories(agent_id=AGENT_ID, limit=40)

    # Build messages for the model
    personality_prompt = body.get("personality_prompt") or (
        "You are Marvin, a slightly paranoid but helpful home/office AI."
    )
    system_prompt = llm_client.build_system_prompt(personality_prompt)

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": transcript or "The user said nothing, but an event was triggered.",
        },
        {
            "role": "system",
            "content": (
                "Recent memories (JSON): " + json.dumps(memories, default=str)[:6000]
            ),
        },
    ]

    llm_response = llm_client.chat_with_tools(
        model_client=_model_client,
        messages=messages,
        tools=agent_tools.TOOLS_SPEC,
    )

    actions, new_memories, reply_text = handle_llm_result(llm_response, agent_event)
    for mem in new_memories:
        memory_store.put_memory(mem)

    dispatch_background_actions(actions)

    resp_body = {
        "reply_text": reply_text,
        "actions": [a.__dict__ for a in actions],
    }

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(resp_body),
    }