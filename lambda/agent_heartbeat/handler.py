from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

from agent_core.schema import Event
from agent_core import memory_store, tools as agent_tools
from agent_core.planner import handle_llm_result
from agent_core.actions import dispatch_background_actions
from agent_core import llm_client
from agent_core.aws_model_client import AwsModelClient
from agent_core.logging_utils import setup_logging


AGENT_ID = os.environ.get("AGENT_ID", "marvin")


# Real model client backed by AWS Bedrock
_model_client = AwsModelClient.from_env()
setup_logging()
logger = logging.getLogger(__name__)


def handler(event, context):
    logger.debug("Heartbeat handler triggered", extra={"event": event})
    # Synthetic system heartbeat event
    agent_event = Event(
        agent_id=AGENT_ID,
        session_id="system-heartbeat",
        source="system",
        channel="timer",
        ts=datetime.now(timezone.utc),
        payload={"kind": "HEARTBEAT", "raw_event": event},
    )
    memory_store.put_event(agent_event)
    logger.debug("Stored heartbeat event for session %s", agent_event.session_id)

    memories = memory_store.recent_memories(agent_id=AGENT_ID, limit=100)
    logger.debug("Loaded %d memories for heartbeat planning", len(memories))

    personality_prompt = (
        "You are Marvin, a slightly paranoid but helpful home/office AI. "
        "This is a scheduled heartbeat: decide if any background work should be done."
    )
    system_prompt = llm_client.build_system_prompt(personality_prompt)

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "system",
            "content": "Heartbeat event payload: " + json.dumps(agent_event.payload),
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
    logger.debug("LLM heartbeat response contained %d messages", len(llm_response.get("messages", [])))

    actions, new_memories, reply_text = handle_llm_result(llm_response, agent_event)
    for mem in new_memories:
        memory_store.put_memory(mem)
    logger.debug("Persisted %d synthetic memories from heartbeat", len(new_memories))

    dispatch_background_actions(actions)
    logger.info("Heartbeat complete", extra={"actions": len(actions)})

    return {
        "statusCode": 200,
        "body": json.dumps(
            {"status": "ok", "reply_text": reply_text, "actions": [a.__dict__ for a in actions]}
        ),
    }