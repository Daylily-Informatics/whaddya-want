# layers/shared/python/companion/broker.py
"""Core orchestration logic shared by the Lambda handler."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, Tuple, List, Optional

from .config import RuntimeConfig
from .prompts import load_personality_prompt
from .llm import LLMClient
from .memory import AISLongTermMemoryStore, ConversationStore, ConversationTurn
from .speech import SpeechSynthesizer
from .actions import ActionManager


class ConversationBroker:
    """Coordinates memory, LLM, speech synthesis, and side-effectful actions."""

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

        # AIS long-term memory
        self._long_term_memory: AISLongTermMemoryStore | None = None
        if config.ais_memory_table:
            print(f"[ais-memory] initializing with table={config.ais_memory_table!r}")
            self._long_term_memory = AISLongTermMemoryStore(
                table_name=config.ais_memory_table,
                region_name=config.region_name,
                ttl_days=config.ais_memory_ttl_days,
            )
        else:
            print("[ais-memory] DISABLED (no AIS_MEMORY_TABLE configured)")

        self._long_term_memory: AISLongTermMemoryStore | None = None
        if config.ais_memory_table:
            self._long_term_memory = AISLongTermMemoryStore(
                table_name=config.ais_memory_table,
                region_name=config.region_name,
                ttl_days=config.ais_memory_ttl_days,
            )

        # Server-side actions (SMS/email/system commands)
        self._actions = ActionManager(region_name=config.region_name)

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
        # ---- Fetch short-term conversation history ----
        if self._config.use_memory:
            history = self._memory.fetch_history(
                session_id=session_id,
                limit=self._config.history_limit,
            )
        else:
            history = []

        # ---- Long-term memory lookup (AIS) ----
        memory_snippets_text: str | None = None
        memories: list[Dict[str, Any]] = []
        is_memory_query = self._looks_like_memory_query(user_text)

        if self._long_term_memory and is_memory_query:
            try:
                memories = self._long_term_memory.search_exchanges(
                    session_id=session_id,
                    query=user_text,
                    limit=8,
                )
                if memories:
                    print(f"[ais-memory] injecting {len(memories)} exchanges for session={session_id}")
                    memory_snippets_text = self._format_memory_snippets(memories)
            except Exception as exc:  # pragma: no cover - non-critical telemetry
                print(f"Warning: failed to retrieve AIS long-term memory: {exc}")
                memories = []
                memory_snippets_text = None

        # ---- Pure memory mode: bypass LLM for some queries ----
        if is_memory_query and self._is_pure_memory_question(user_text):
            if memories:
                reply_text = self._render_memory_reply(user_text, memories)
            else:
                reply_text = (
                    "I don't have any reliable memory of that topic. "
                    "My long-term log doesn't show anything relevant."
                )

            command: Dict[str, Any] = {"name": "noop", "args": {}}
            timestamp = datetime.now(timezone.utc)
            turns = [
                ConversationTurn(role="user", content=user_text, timestamp=timestamp),
                ConversationTurn(role="assistant", content=reply_text, timestamp=timestamp),
            ]

            # Persist short-term history
            if self._config.use_memory:
                self._memory.append_turns(
                    session_id=session_id,
                    turns=turns,
                    limit=self._config.history_limit,
                )

            # Record deterministic memory reply into AIS
            if self._long_term_memory:
                try:
                    self._long_term_memory.record_exchange(
                        session_id=session_id,
                        timestamp=timestamp,
                        user_text=user_text,
                        assistant_text=reply_text,
                        metadata={"server_action_result": None, "tool_calls": []},
                    )
                except Exception as exc:  # pragma: no cover
                    print(f"Warning: failed to persist AIS long-term memory: {exc}")

            audio_payload = self._speech.synthesize(
                text=reply_text,
                session_id=session_id,
                response_id=str(int(timestamp.timestamp())),
                voice_id=voice_id,
            )
            if text_only and isinstance(audio_payload, dict):
                audio_payload = dict(audio_payload)
                audio_payload["audio_base64"] = None

            return {
                "text": reply_text,
                "audio": audio_payload,
                "command": command,
                "tool_calls": [],
            }

        # ---- Build messages for the LLM ----
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": self._system_prompt,
            },
        ]

        if memory_snippets_text:
            messages.append(
                {
                    "role": "system",
                    "content": f"Relevant past memory:\n{memory_snippets_text}",
                }
            )

        messages.extend({"role": turn.role, "content": turn.content} for turn in history)
        messages.append({"role": "user", "content": user_text})

        vision_scene = None
        if isinstance(context, dict):
            vision_scene = context.get("vision_scene") if isinstance(context.get("vision_scene"), dict) else None

        if context:
            # Context is injected as a separate system message so it can carry
            # vision + environment info without polluting the main prompt text.
            messages.append({"role": "system", "content": f"Context: {context}"})

        vision_context_text = self._format_vision_scene_for_prompt(vision_scene)
        if vision_context_text:
            messages.append({"role": "system", "content": vision_context_text})

        # ---- Call the LLM ----
        llm_response = self._llm.chat(messages=messages)

        timestamp = datetime.now(timezone.utc)
        context_turns: list[ConversationTurn] = []
        if vision_context_text:
            context_turns.append(
                ConversationTurn(
                    role="system", content=vision_context_text, timestamp=timestamp
                )
            )

        turns = context_turns + [
            ConversationTurn(role="user", content=user_text, timestamp=timestamp),
            ConversationTurn(role="assistant", content=llm_response.message, timestamp=timestamp),
        ]

        # ---- Persist short-term history ----
        if self._config.use_memory:
            self._memory.append_turns(
                session_id=session_id,
                turns=turns,
                limit=self._config.history_limit,
            )

        # ---- Parse command from the LLM reply ----
        reply_text, command = _extract_command(llm_response.message)

        # ---- TTS ----
        audio_payload = self._speech.synthesize(
            text=reply_text,  # use cleaned text, not raw with COMMAND
            session_id=session_id,
            response_id=str(int(timestamp.timestamp())),
            voice_id=voice_id,
        )

        # ---- Execute server-side actions for certain commands ----
        server_action_result: Dict[str, Any] | None = None
        if isinstance(command, dict):
            cmd_name = command.get("name")
            cmd_args = command.get("args") or {}
            if cmd_name in {"send_text", "send_email", "run_command"}:
                try:
                    result = self._actions.run_server_action(str(cmd_name), cmd_args)
                    server_action_result = {
                        "name": result.name,
                        "executed": result.executed,
                        "error": result.error,
                        "detail": result.detail,
                    }
                    # Attach a small status line the user can see, if provided.
                    if result.visible_message:
                        reply_text = f"{reply_text}\n\n{result.visible_message}"
                    # Light-weight status inline with the command for clients that care.
                    command = dict(command)
                    command["server_result"] = server_action_result
                except Exception as exc:  # pragma: no cover - defensive
                    server_action_result = {
                        "name": cmd_name,
                        "executed": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    command = dict(command)
                    command["server_result"] = server_action_result

        # ---- Optionally strip inline audio for text-only calls ----
        if text_only and isinstance(audio_payload, dict):
            audio_payload = dict(audio_payload)
            audio_payload["audio_base64"] = None

        # ---- Long-term AIS memory ----
        if self._long_term_memory:
            metadata = {
                "vision_scene": (context or {}).get("vision_scene") if context else None,
                "vision_context_summary": vision_context_text,
                "command": command,
                "server_action_result": server_action_result,
                "tool_calls": llm_response.tool_calls,
                "assistant_visible_text": reply_text,
            }
            try:
                self._long_term_memory.record_exchange(
                    session_id=session_id,
                    timestamp=timestamp,
                    user_text=user_text,
                    assistant_text=llm_response.message,
                    metadata=metadata,
                )
            except Exception as exc:  # pragma: no cover - non-critical telemetry
                print(f"Warning: failed to persist AIS long-term memory: {exc}")

        return {
            "text": reply_text,
            "audio": audio_payload,
            "command": command,
            "tool_calls": llm_response.tool_calls,
        }

    def _looks_like_memory_query(self, user_text: str) -> bool:
        """Heuristic to decide whether the user is asking about past context.

        This is intentionally cheap and conservative; it only fires on obvious
        "what did we do before" / "remind me" style questions.
        """
        text = (user_text or "").lower()
        keywords = [
            "remind me",
            "remember when",
            "what did we decide",
            "what did we talk about",
            "what were we doing",
            "last time we",
            "previous conversation",
            "earlier in this conversation",
            "based on everything we've done",
            "summarize what we've done",
            # vision-ish memory cues
            "have you ever seen",
            "what have you seen",
            "pictures you've seen",
            "things you've seen in the past",
        ]
        return any(k in text for k in keywords)

    def _is_pure_memory_question(self, text: str) -> bool:
        """Return True if the user is clearly asking ONLY about past context."""
        t = (text or "").lower()
        # crude, but OK: no 'why/should/how do I' + has 'remind/remember/have we talked'
        if any(w in t for w in ["why", "should", "how do i"]):
            return False
        return any(
            phrase in t
            for phrase in [
                "remind me",
                "remember when",
                "have we discussed",
                "have we talked about",
                "what did we talk about",
                "what did i tell you",
                "what have we said about",
                "what things we've said about",
                "remind me if i asked you",
                "remind me what we talked about",
            ]
        )

    def _render_memory_reply(self, user_text: str, memories: List[Dict[str, Any]]) -> str:
        """Generate a deterministic, non-hallucinated summary of AIS memory."""
        lines: list[str] = []
        for ex in memories:
            ts = ex.get("timestamp") or ""
            user = (ex.get("user") or {}).get("content") or ""
            assistant = (ex.get("assistant") or {}).get("content") or ""
            if not user and not assistant:
                continue
            prefix = f"[{ts}] " if ts else ""
            if user:
                lines.append(f"{prefix}You said: {user}")
            if assistant:
                lines.append(f"{prefix}I replied: {assistant}")
        if not lines:
            return "I don't have any reliable memory on that topic in my log."
        return "Here's what I have in my long-term log about that:\n" + "\n".join(lines[:16])

    def _format_memory_snippets(self, exchanges: Any) -> str:
        """Format AIS memory exchanges into a compact, readable summary."""
        if not exchanges:
            return ""

        if not isinstance(exchanges, list):
            print(f"[ais-memory] expected list of dicts, got {type(exchanges)}: {repr(exchanges)[:500]}")
            return ""

        lines: list[str] = []
        for ex in exchanges:
            if not isinstance(ex, dict):
                print(f"[ais-memory] skipping non-dict memory item: {type(ex)} {repr(ex)[:200]}")
                continue

            ts = ex.get("timestamp") or ""
            user = (ex.get("user") or {}).get("content") or ""
            assistant = (ex.get("assistant") or {}).get("content") or ""

            # Truncate very long texts so we don't blow out the prompt
            if len(user) > 300:
                user = user[:297] + "..."
            if len(assistant) > 300:
                assistant = assistant[:297] + "..."
            header = f"- [{ts}] You: {user}" if ts else f"- You: {user}"
            lines.append(header)
            if assistant:
                lines.append(f"  Marvin: {assistant}")

        # Cap total lines to keep this block small
        return "\n".join(lines[:40])

    def _format_vision_scene_for_prompt(self, scene: Optional[Dict[str, Any]]) -> str:
        """Render a vision scene dictionary into a compact, LLM-friendly string."""
        if not isinstance(scene, dict):
            return ""

        lines: list[str] = []

        caption = scene.get("detailed_caption") or scene.get("caption")
        if caption:
            lines.append(f"Scene: {caption}")

        layout = scene.get("layout")
        if layout:
            lines.append(f"Layout: {layout}")

        lighting = scene.get("lighting")
        if lighting:
            lines.append(f"Lighting: {lighting}")

        def _format_entities(key: str) -> list[str]:
            out: list[str] = []
            entries = scene.get(key)
            if not isinstance(entries, list):
                return out
            for ent in entries:
                if not isinstance(ent, dict):
                    continue
                name = ent.get("name") or ent.get("species") or key.rstrip("s")
                count = ent.get("count") or ent.get("approx_count")
                attrs = ent.get("attributes") or ent.get("activities") or ent.get("notes")
                suffix = f" (x{count})" if isinstance(count, int) else ""
                detail = f": {attrs}" if attrs else ""
                out.append(f"- {name}{suffix}{detail}")
            return out

        people_lines = _format_entities("people")
        if people_lines:
            lines.append("People:")
            lines.extend(people_lines)

        animal_lines = _format_entities("animals")
        if animal_lines:
            lines.append("Animals:")
            lines.extend(animal_lines)

        object_lines = _format_entities("objects")
        if object_lines:
            lines.append("Objects:")
            lines.extend(object_lines)

        salient = scene.get("salient_facts") or scene.get("memory_bullets")
        if isinstance(salient, list) and salient:
            lines.append("Salient:")
            for fact in salient:
                if isinstance(fact, str):
                    lines.append(f"- {fact}")

        if not lines:
            return ""

        return "Vision observation (save for recall):\n" + "\n".join(lines)


def _extract_command(reply: str) -> Tuple[str, Dict[str, Any]]:
    """
    Parse an optional trailing 'COMMAND: {...}' line from the LLM reply.

    Returns (clean_text, command_dict). If no valid command is found, returns
    the original reply (stripped) and a default noop command.
    """
    default_cmd: Dict[str, Any] = {"name": "noop", "args": {}}
    if not isinstance(reply, str):
        return str(reply), default_cmd

    # Commands that are syntactically allowed. Some are executed only client-side
    # (launch_monitor, set_device); others are executed on the server (send_text,
    # send_email, run_command) with additional gating in ActionManager.
    allowed_names = {
        "launch_monitor",
        "set_device",
        "noop",
        "send_text",
        "send_email",
        "run_command",
    }

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
        args_raw = (alt.get(2) or "").strip() if hasattr(alt, "get") else (alt.group(2) or "").strip()
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
