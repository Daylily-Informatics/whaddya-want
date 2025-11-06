"""Core orchestration logic shared by the Lambda handler."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .config import RuntimeConfig
from .llm import LLMClient
from .memory import ConversationStore, ConversationTurn
from .speech import SpeechSynthesizer


class ConversationBroker:
    """Coordinates memory, LLM, and speech synthesis."""

    def __init__(self, config: RuntimeConfig) -> None:
        self._config = config
        self._memory = ConversationStore(
            table_name=config.conversation_table,
            region_name=config.region_name,
        )
        self._llm = LLMClient(
            secret_id=config.secrets_id,
            region_name=config.region_name,
        )
        self._speech = SpeechSynthesizer(
            bucket=config.audio_bucket,
            voice_id=config.voice_id,
            region_name=config.region_name,
        )

    def handle(self, session_id: str, user_text: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        history = self._memory.fetch_history(session_id=session_id, limit=self._config.history_limit)
        messages = [
            {
                "role": "system",
                "content": "You are Marvin, a helpful, proactive AI companion.",
            },
            *({"role": turn.role, "content": turn.content} for turn in history),
            {"role": "user", "content": user_text},
        ]
        if context:
            messages.append({"role": "system", "content": f"Context: {context}"})
        llm_response = self._llm.chat(messages=messages)

        timestamp = datetime.now(timezone.utc)
        turns = [
            ConversationTurn(role="user", content=user_text, timestamp=timestamp),
            ConversationTurn(role="assistant", content=llm_response.message, timestamp=timestamp),
        ]
        self._memory.append_turns(session_id=session_id, turns=turns)

        audio_payload = self._speech.synthesize(
            text=llm_response.message,
            session_id=session_id,
            response_id=str(int(timestamp.timestamp())),
        )

        return {
            "text": llm_response.message,
            "audio": audio_payload,
            "tool_calls": llm_response.tool_calls,
        }


__all__ = ["ConversationBroker"]
