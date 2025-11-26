"""Core orchestration logic shared by the Lambda handler."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from .config import RuntimeConfig
from .llm import LLMClient
from .memory import ConversationStore, ConversationTurn
from .prompts import PromptSet, build_system_prompt, load_prompt_set
from .speech import SpeechSynthesizer


class ConversationBroker:
    """Coordinates memory, LLM, and speech synthesis."""

    def __init__(self, config: RuntimeConfig) -> None:
        self._config = config
        self._memory = ConversationStore(
            table_name=config.conversation_table,
            region_name=config.region_name,
            ttl_seconds=config.memory_ttl_seconds,
            history_limit=config.history_limit,
        )
        self._llm = LLMClient(
            secret_id=config.secrets_id,
            region_name=config.region_name,
            model=config.model_id,
        )
        self._speech = SpeechSynthesizer(
            bucket=config.audio_bucket,
            voice_id=config.voice_id,
            region_name=config.region_name,
        )
        self._prompts: PromptSet = load_prompt_set(config.prompts_path)
        self._allowed_commands = {"launch_monitor", "set_device", "noop"}

    def _extract_command(self, reply: str) -> tuple[str, dict[str, Any]]:
        default_cmd: dict[str, Any] = {"name": "noop", "args": {}}
        if not isinstance(reply, str):
            return str(reply), default_cmd

        m = re.search(r"^COMMAND:\s*(\{.*\})\s*$", reply, flags=re.MULTILINE)
        if m:
            cmd_json = m.group(1)
            clean = re.sub(r"^COMMAND:.*$", "", reply, flags=re.MULTILINE).rstrip()
            cmd = default_cmd
            try:
                parsed = json.loads(cmd_json)
                if isinstance(parsed, dict):
                    name = parsed.get("name") or "noop"
                    args = parsed.get("args") or {}
                    if (
                        isinstance(name, str)
                        and isinstance(args, dict)
                        and name in self._allowed_commands
                    ):
                        cmd = {"name": name, "args": args}
            except Exception:
                cmd = default_cmd
            return clean.strip(), cmd

        alt = re.search(
            r"^command\s+name[:\s]+([\w-]+)\s+args[:\s]+(.*)$",
            reply,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if alt:
            clean = re.sub(
                r"^command\s+name[:\s]+.*$", "", reply, flags=re.IGNORECASE | re.MULTILINE
            ).rstrip()
            name = alt.group(1).strip()
            args_raw = (alt.group(2) or "").strip()
            cmd = default_cmd
            args: dict[str, Any] = {}
            try:
                args = json.loads(args_raw) if args_raw else {}
            except Exception:
                args = {}

            if name in self._allowed_commands and isinstance(args, dict):
                cmd = {"name": name, "args": args}

            return clean.strip(), cmd

        return reply.strip(), default_cmd

    def handle(
        self,
        session_id: str,
        user_text: str,
        context: dict[str, Any] | None = None,
        *,
        text_only: bool = False,
        voice_id: str | None = None,
    ) -> dict[str, Any]:
        history = self._memory.fetch_history(session_id=session_id, limit=self._config.history_limit)
        system_prompt = build_system_prompt(
            self._prompts,
            speaker=(context or {}).get("speaker_id"),
            acoustic_event=(context or {}).get("acoustic_event"),
            intro_already_sent=bool((context or {}).get("intro_already_sent")),
        )
        messages = [
            {"role": "system", "content": system_prompt},
            *({"role": turn.role, "content": turn.content} for turn in history),
            {"role": "user", "content": user_text},
        ]
        llm_response = self._llm.chat(messages=messages)
        reply_text, command = self._extract_command(llm_response.message)

        timestamp = datetime.now(timezone.utc)
        turns = [
            ConversationTurn(role="user", content=user_text, timestamp=timestamp),
            ConversationTurn(role="assistant", content=reply_text, timestamp=timestamp),
        ]
        self._memory.append_turns(session_id=session_id, turns=turns)

        audio_payload = None
        if not text_only:
            audio_payload = self._speech.synthesize(
                text=reply_text,
                session_id=session_id,
                response_id=str(int(timestamp.timestamp() * 1000)),
                voice_id=voice_id,
            )

        return {
            "text": reply_text,
            "audio": audio_payload,
            "command": command,
            "tool_calls": llm_response.tool_calls,
        }


__all__ = ["ConversationBroker"]
