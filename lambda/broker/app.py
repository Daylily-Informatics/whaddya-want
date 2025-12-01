import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from agent_core.schema import Event
from agent_core import memory_store, tools as agent_tools
from agent_core.planner import handle_llm_result
from agent_core.actions import dispatch_background_actions
from agent_core import llm_client
from agent_core.aws_model_client import AwsModelClient
from agent_core.logging_utils import configure_logging
from agent_core import voice_registry
from agent_core.speech import SpeechSynthesizer


configure_logging(int(os.environ.get("VERBOSE", "0")))
logger = logging.getLogger(__name__)

AGENT_ID = os.environ.get("AGENT_ID", "marvin")

_model_client = AwsModelClient.from_env()

# Polly-based speech synthesizer (optional S3 backing).
try:
    _speech = SpeechSynthesizer(
        bucket=os.getenv("AUDIO_BUCKET"),
        voice_id=os.getenv("AGENT_VOICE_ID", "Matthew"),
        region_name=os.getenv("AWS_REGION") or os.getenv("REGION"),
    )
    logger.info(
        "Initialized SpeechSynthesizer with voice_id=%s bucket=%s",
        os.getenv("AGENT_VOICE_ID", "Matthew"),
        os.getenv("AUDIO_BUCKET"),
    )
except Exception as exc:  # pragma: no cover - defensive
    logger.warning("Failed to initialize SpeechSynthesizer: %s", exc)
    _speech = None


def _derive_session_id(event: Dict[str, Any]) -> str:
    headers = event.get("headers") or {}
    src = headers.get("X-Session-Id") or headers.get("x-session-id")
    if src:
        return str(src)
    rc = event.get("requestContext") or {}
    req_id = rc.get("requestId") or rc.get("requestID") or "unknown"
    return f"lambda-{req_id}"


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


def _extract_claimed_name_from_text(text: str) -> Optional[str]:
    """Heuristic to extract a name from utterances like 'my name is John'."""
    if not text:
        return None
    lower = text.lower()
    idx = lower.find("my name is")
    if idx == -1:
        return None
    after = text[idx + len("my name is") :].strip()
    for stop in (".", "!", "?", ","):
        s = after.find(stop)
        if s != -1:
            after = after[:s]
    name = after.strip(" '\"")
    if not name:
        return None
    if len(name.split()) > 5:
        return None
    return name


def handler(event, context):
    logger.info("Broker invoked with requestContext=%s", event.get("requestContext"))
    body = _payload_from_event(event)
    session_id = _derive_session_id(event)

    transcript: str = body.get("transcript") or body.get("text") or ""
    channel: str = body.get("channel", "audio")

    # Voice identity: raw ID + optional embedding + claimed name.
    voice_id = body.get("voice_id") or body.get("speaker_id")
    claimed_name = body.get("speaker_name") or body.get("user_name")
    embedding = body.get("voice_embedding")

    if voice_id and not claimed_name and transcript:
        inferred = _extract_claimed_name_from_text(transcript)
        if inferred:
            claimed_name = inferred

    speaker_name: Optional[str] = None
    is_new_voice = False

    if voice_id or claimed_name or embedding:
        try:
            speaker_name, is_new_voice = voice_registry.resolve_voice(
                agent_id=AGENT_ID,
                voice_id=str(voice_id) if voice_id is not None else None,
                claimed_name=claimed_name,
                embedding=embedding,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Error resolving voice profile: %s", exc)
            speaker_name, is_new_voice = claimed_name, False

    if speaker_name:
        source = speaker_name
    else:
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
            "voice_embedding": embedding,
            "raw": body,
        },
    )
    memory_store.put_event(agent_event)
    logger.debug("Stored incoming event payload: %s", agent_event.payload)

    memories = memory_store.recent_memories(agent_id=AGENT_ID, limit=40)
    logger.debug("Loaded %s recent memories", len(memories))

    personality_prompt = body.get("personality_prompt") or (
        "You are Marvin, a slightly paranoid but helpful home/office AI."
    )
    system_prompt = llm_client.build_system_prompt(personality_prompt)

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
    ]

    # Speaker context to the model.
    if speaker_name and not is_new_voice:
        messages.append(
            {
                "role": "system",
                "content": (
                    f"The current speaker's voice has been recognized as '{speaker_name}'. "
                    f"When you store memories about this conversation, treat them as "
                    f"information about '{speaker_name}'."
                ),
            }
        )
    elif speaker_name and is_new_voice:
        messages.append(
            {
                "role": "system",
                "content": (
                    f"You have just learned that the current speaker's name is '{speaker_name}'. "
                    f"Use this name when referring to them in conversation and memory."
                ),
            }
        )
    elif voice_id and is_new_voice:
        messages.append(
            {
                "role": "system",
                "content": (
                    "You are hearing a new voice that has not been enrolled yet. Before answering "
                    "other substantive questions, politely ask the speaker for their name so you "
                    "can remember who they are."
                ),
            }
        )

    messages.append(
        {
            "role": "system",
            "content": "Recent memories (JSON): " + json.dumps(memories, default=str)[:6000],
        }
    )

    messages.append(
        {
            "role": "user",
            "content": transcript
            or "The user said nothing, but an event was triggered.",
        }
    )

    logger.debug("Sending %s messages to llm_client", len(messages))
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

    audio_info: Optional[Dict[str, Any]] = None
    if _speech is not None and reply_text:
        try:
            audio_info = _speech.synthesize(
                text=reply_text,
                key_prefix=f"{AGENT_ID}/{session_id}",
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Speech synthesis failed: %s", exc)
            audio_info = None

    resp_body: Dict[str, Any] = {
        "reply_text": reply_text,
        "actions": [a.__dict__ for a in actions],
    }
    if audio_info:
        resp_body["audio"] = audio_info

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(resp_body),
    }
