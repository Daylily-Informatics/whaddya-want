import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict

from agent_core.schema import Event
from agent_core import memory_store, tools as agent_tools
from agent_core.planner import handle_llm_result
from agent_core.actions import dispatch_background_actions
from agent_core import llm_client
from agent_core.aws_model_client import AwsModelClient
from agent_core.logging_utils import configure_logging
from agent_core import voice_registry


configure_logging(int(os.environ.get("VERBOSE", "0")))
logger = logging.getLogger(__name__)

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
    logger.info("Broker invoked with requestContext=%s", event.get("requestContext"))
    body = _payload_from_event(event)
    session_id = _derive_session_id(event)

    # Basic fields from the payload
    transcript = body.get("transcript") or body.get("text") or ""
    channel = body.get("channel", "audio")

    # Voice identity resolution
    voice_id = body.get("voice_id") or body.get("speaker_id")
    claimed_name = body.get("speaker_name") or body.get("user_name")
    speaker_name = None
    is_new_voice = False

    if voice_id or claimed_name:
        try:
            speaker_name, is_new_voice = voice_registry.resolve_voice(
                agent_id=AGENT_ID,
                voice_id=voice_id,
                claimed_name=claimed_name,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Error resolving voice profile: %s", exc)
            speaker_name, is_new_voice = claimed_name, False

    if speaker_name:
        source = speaker_name
    else:
        # If we have a voice_id but no known name yet, mark as unknown_voice.
        source = body.get("source") or ("unknown_voice" if voice_id else "user")

    # Persist the incoming event
    agent_event = Event(
        agent_id=AGENT_ID,
        session_id=session_id,
        source=source,
        channel=channel,
        ts=datetime.now(timezone.utc),
        payload={
            "transcript": transcript,
            "voice_id": voice_id,
            "speaker_name": speaker_name,
            "raw": body,
        },
    )
    memory_store.put_event(agent_event)
    logger.debug("Stored incoming event payload: %s", agent_event.payload)

    # Retrieve recent memories for context
    memories = memory_store.recent_memories(agent_id=AGENT_ID, limit=40)
    logger.debug("Loaded %s recent memories", len(memories))

    # Build messages for the model
    personality_prompt = body.get("personality_prompt") or (
        "You are Marvin, a slightly paranoid but helpful home/office AI."
    )
    system_prompt = llm_client.build_system_prompt(personality_prompt)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
    ]

    # Give the model explicit context about the current speaker.
    if speaker_name:
        messages.append(
            {
                "role": "system",
                "content": (
                    f"The current speaker is named '{speaker_name}'. When you store memories "
                    f"about this conversation, treat them as information about '{speaker_name}'."
                ),
            }
        )
    elif voice_id and is_new_voice:
        # Unknown voice: ask for their name before proceeding too far.
        messages.append(
            {
                "role": "system",
                "content": (
                    "You are hearing a new voice that has not been enrolled yet. Before "
                    "answering other substantive questions, politely ask the speaker for "
                    "their name so you can remember who they are."
                ),
            }
        )

    # User message
    messages.append(
        {
            "role": "user",
            "content": transcript
            or "The user said nothing, but an event was triggered.",
        }
    )

    # Memory dump for retrieval-augmented answers
    messages.append(
        {
            "role": "system",
            "content": "Recent memories (JSON): "
            + json.dumps(memories, default=str)[:6000],
        }
    )

    llm_response = llm_client.chat_with_tools(
        model_client=_model_client,
        messages=messages,
        tools=agent_tools.TOOLS_SPEC,
    )
    logger.debug("LLM response keys: %s", list(llm_response.keys()))

    actions, new_memories, reply_text = handle_llm_result(llm_response, agent_event)
    for mem in new_memories:
        memory_store.put_memory(mem)
        logger.debug("Persisted new memory from broker handler: %s", mem.summary)

    dispatch_background_actions(actions)
    logger.info("Broker produced %s actions", len(actions))

    resp_body = {
        "reply_text": reply_text,
        "actions": [a.__dict__ for a in actions],
    }

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(resp_body),
    }
