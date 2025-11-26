"""Core orchestration logic shared by the Lambda handler."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .config import RuntimeConfig
from .prompts import load_personality_prompt
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
            provider="openai" if config.llm_provider == "openai" else "bedrock",
            model=config.llm_model_id,
            region_name=config.region_name,
            secret_id=config.secrets_id or None,
        )
        self._speech = SpeechSynthesizer(
            bucket=config.audio_bucket,
            voice_id=config.voice_id,
            region_name=config.region_name,
        )
        self._system_prompt = load_personality_prompt(config.prompts_path)

    def handle(
        self,
        session_id: str,
        user_text: str,
        context: dict[str, Any] | None = None,
        voice_id: str | None = None,
    ) -> dict[str, Any]:
        """Process a single user turn and return text + audio payload.

        Returns a mapping:

            {
              "text": "...",
              "audio": {...},              # see SpeechSynthesizer.synthesize
              "tool_calls": [...],         # if the provider supports tools
            }
        """
        if self._config.use_memory:
            history = self._memory.fetch_history(
                session_id=session_id,
                limit=self._config.history_limit,
            )
        else:
            history = []

        messages = [
            {
                "role": "system",
                "content": self._system_prompt,
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
        if self._config.use_memory:
            self._memory.append_turns(
                session_id=session_id,
                turns=turns,
                limit=self._config.history_limit,
            )

        audio_payload = self._speech.synthesize(
            text=llm_response.message,
            session_id=session_id,
            response_id=str(int(timestamp.timestamp())),
            voice_id=voice_id,
        )

        return {
            "text": llm_response.message,
            "audio": audio_payload,
            "tool_calls": llm_response.tool_calls,
        }


__all__ = ["ConversationBroker"]
