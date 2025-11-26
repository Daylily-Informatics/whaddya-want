"""Core orchestration logic shared by the Lambda handler."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, Tuple

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
        text_only: bool = False,
    ) -> dict[str, Any]:
        """Process a single user turn and return text + audio payload.

        Returns a mapping:

            {
              "text": "...",
              "audio": {...},              # see SpeechSynthesizer.synthesize
              "command": {...},            # parsed COMMAND line from the LLM reply
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

        reply_text, command = _extract_command(llm_response.message)

        if text_only and isinstance(audio_payload, dict):
            audio_payload = dict(audio_payload)
            audio_payload["audio_base64"] = None

        return {
            "text": reply_text,
            "audio": audio_payload,
            "command": command,
            "tool_calls": llm_response.tool_calls,
        }


def _extract_command(reply: str) -> Tuple[str, Dict[str, Any]]:
    """
    Parse an optional trailing 'COMMAND: {...}' line from the LLM reply.

    Returns (clean_text, command_dict). If no valid command is found, returns
    the original reply (stripped) and a default noop command.
    """
    default_cmd: Dict[str, Any] = {"name": "noop", "args": {}}
    if not isinstance(reply, str):
        return str(reply), default_cmd

    allowed_names = {"launch_monitor", "set_device", "noop"}

    # Look for a line starting with 'COMMAND:' and capture the JSON blob
    m = re.search(r"^COMMAND:\s*(\{.*\})\s*$", reply, flags=re.MULTILINE)
    if m:
        cmd_json = m.group(1)
        # Remove the COMMAND line from the visible text
        clean = re.sub(r"^COMMAND:.*$", "", reply, flags=re.MULTILINE).rstrip()

        cmd = default_cmd
        try:
            parsed = json.loads(cmd_json)
            if isinstance(parsed, dict):
                name = parsed.get("name") or "noop"
                args = parsed.get("args") or {}
                if isinstance(name, str) and isinstance(args, dict):
                    if name in allowed_names:
                        cmd = {"name": name, "args": args}
        except Exception:
            # On any parse error, fall back to noop but keep the cleaned text
            cmd = default_cmd

        return clean.strip(), cmd

    # Fallback: handle looser phrasing like "command name: noop args: {}"
    alt = re.search(
        r"^command\s+name[:\s]+([\w-]+)\s+args[:\s]+(.*)$",
        reply,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if alt:
        clean = re.sub(
            r"^command\s+name[:\s]+.*$",
            "",
            reply,
            flags=re.IGNORECASE | re.MULTILINE,
        ).rstrip()
        name = alt.group(1).strip()
        args_raw = (alt.group(2) or "").strip()
        cmd = default_cmd
        args: Dict[str, Any] = {}
        try:
            args = json.loads(args_raw) if args_raw else {}
        except Exception:
            args = {}

        if name in allowed_names and isinstance(args, dict):
            cmd = {"name": name, "args": args}

        return clean.strip(), cmd

    return reply.strip(), default_cmd


__all__ = ["ConversationBroker"]
