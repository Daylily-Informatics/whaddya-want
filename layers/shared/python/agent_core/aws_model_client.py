from __future__ import annotations

"""AWS Bedrock-backed model client for the Rank-4 agent.

This client is responsible for:
- Reading MODEL_ID / AWS_REGION from the environment.
- Talking to Amazon Bedrock via the `converse` API.
- Returning responses in the structure expected by:
    - agent_core.llm_client.chat_with_tools
    - agent_core.planner.handle_llm_result

It also:
- Answers a few simple questions directly from long-term memory
  ("what is my name", "how old am I", "do I have dogs") to avoid
  wasting LLM calls.
- Supports *real* Bedrock tool-calling: when `tools` are provided we
  send a `toolConfig` to Bedrock and translate any returned toolCalls
  into the generic `tool_calls` structure that the planner expects.
- Optionally synthesizes a `store_memory` tool call when the model
  did not request one, so that basic long-term memory continues to
  function even if the model under-uses tools.

Returned response shape:

    {
        "messages": [
            ... original messages ...,
            {
                "role": "assistant",
                "content": reply_text,
                "tool_calls": [
                    {"name": "store_memory", "arguments": "{...}"},
                    ...
                ]
            }
        ]
    }

This is intentionally generic: it does *not* execute tools itself; it
just returns tool calls for the planner and action dispatcher.
"""

import json
import logging
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import boto3

from . import memory_store


logger = logging.getLogger(__name__)


class AwsModelClient:
    def __init__(self, model_id: str | None = None, region_name: str | None = None) -> None:
        # Prefer explicit args, then env. We do *not* silently default to Titan.
        self._model: str = (model_id or os.getenv("MODEL_ID") or "").strip()
        if not self._model:
            raise RuntimeError(
                "AwsModelClient: MODEL_ID must be configured (env var MODEL_ID or constructor). "
                "Choose a Bedrock model that supports system messages, e.g. meta.llama3-1-8b-instruct-v1:0."
            )
        self._region: str = (region_name or os.getenv("AWS_REGION") or "us-west-2").strip()
        self._bedrock = boto3.client("bedrock-runtime", region_name=self._region)
        logger.info("Initialized AwsModelClient model=%s region=%s", self._model, self._region)

    # ------------------------------------------------------------------
    # Internal helpers for Bedrock invocation
    # ------------------------------------------------------------------

    def _split_system_and_messages(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[str, List[Dict[str, str]]]:
        """Split a mixed message list into (system_prompt, non_system_messages)."""
        system_parts: List[str] = []
        non_system: List[Dict[str, str]] = []
        for m in messages:
            role = (m.get("role") or "").lower()
            content = m.get("content") or ""
            if role == "system":
                system_parts.append(str(content))
            else:
                non_system.append({"role": m.get("role", "user"), "content": str(content)})
        system_text = "\n\n".join(system_parts).strip()
        return system_text, non_system

    def _build_bedrock_messages(self, convo: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Convert generic messages into Bedrock `converse` message format."""
        messages: List[Dict[str, Any]] = []
        for msg in convo:
            role = msg["role"]
            text = msg["content"]
            messages.append(
                {
                    "role": role,
                    "content": [{"text": text}],
                }
            )
        return messages

    def _convert_tools_to_bedrock(self, tools: Optional[List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
        """Translate generic tool specs into Bedrock toolConfig structure."""
        if not tools:
            return None

        bedrock_tools: List[Dict[str, Any]] = []
        for spec in tools:
            name = spec.get("name")
            if not name:
                continue
            description = spec.get("description", "")
            params = spec.get("parameters") or {}
            bedrock_tools.append(
                {
                    "toolSpec": {
                        "name": name,
                        "description": description,
                        "inputSchema": {"json": params},
                    }
                }
            )

        if not bedrock_tools:
            return None

        return {
            "tools": bedrock_tools,
            "toolChoice": {"auto": {}},
        }

    def _invoke_converse(
        self,
        system: str,
        convo: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Low-level wrapper around Bedrock `converse` that returns the raw output."""
        messages = self._build_bedrock_messages(convo)
        kwargs: Dict[str, Any] = {
            "modelId": self._model,
            "messages": messages,
            "inferenceConfig": {
                "maxTokens": 300,
                "temperature": 0.4,
                "topP": 0.9,
            },
        }
        if system:
            kwargs["system"] = [{"text": system}]

        tool_config = self._convert_tools_to_bedrock(tools)
        if tool_config is not None:
            kwargs["toolConfig"] = tool_config

        try:
            logger.debug(
                "Invoking converse model %s with %s messages (system_present=%s, tools=%s)",
                self._model,
                len(messages),
                bool(system),
                bool(tool_config),
            )
            out = self._bedrock.converse(**kwargs)
            return out
        except Exception as e:  # pragma: no cover - defensive fallback
            msg = str(e)
            logger.warning("Bedrock converse error: %s", msg)

            # Explicitly surface "no system support" instead of silently stripping system.
            if "doesn't support system messages" in msg.lower():
                return {
                    "output": {
                        "message": {
                            "role": "assistant",
                            "content": [
                                {
                                    "text": (
                                        "Model configuration error: the configured MODEL_ID does not "
                                        "support system messages. Please choose a Bedrock model "
                                        "that supports system prompts (for example, a meta.llama3-based "
                                        "model) and set MODEL_ID accordingly."
                                    )
                                }
                            ],
                        },
                        "toolCalls": [],
                    }
                }

            # Last resort: surface a generic failure message.
            return {
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"text": "I had trouble talking to the language model backend."}
                        ],
                    },
                    "toolCalls": [],
                }
            }

    def _chat_bedrock(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Call Bedrock and return (reply_text, tool_calls)."""
        system, convo = self._split_system_and_messages(messages)
        if not self._model:
            raise RuntimeError("MODEL_ID must be set for Bedrock provider.")
        logger.debug(
            "Routing chat via converse for model %s (system_present=%s)",
            self._model,
            bool(system),
        )
        raw = self._invoke_converse(system, convo, tools)

        # Extract text
        reply_text = ""
        try:
            out_msg = raw["output"]["message"]
            contents = out_msg.get("content") or []
            for part in contents:
                if isinstance(part, dict) and "text" in part:
                    reply_text = str(part["text"])
                    break
        except Exception:  # pragma: no cover - defensive
            reply_text = "I couldn't generate a response."

        # Extract tool calls, if any
        tool_calls: List[Dict[str, Any]] = []
        try:
            raw_tool_calls = raw.get("output", {}).get("toolCalls", []) or []
            for tc in raw_tool_calls:
                name = tc.get("name")
                if not name:
                    continue
                # Bedrock uses `input` for tool arguments
                args = tc.get("input") or {}
                tool_calls.append({"name": name, "arguments": json.dumps(args)})
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to parse Bedrock toolCalls: %s", exc)

        return reply_text, tool_calls

    # ------------------------------------------------------------------
    # Helpers for memory semantics
    # ------------------------------------------------------------------

    def _get_agent_id(self) -> str:
        """Best-effort agent id; used as the partition key for memories."""
        return os.getenv("AGENT_ID") or "marvin"

    @staticmethod
    def _extract_text_fields(item: Dict[str, Any]) -> Iterable[str]:
        """Yield all relevant text fields from a memory item dict."""
        if not isinstance(item, dict):
            return []
        texts: List[str] = []
        summary = item.get("summary")
        if isinstance(summary, str):
            texts.append(summary)
        content = item.get("content")
        if isinstance(content, dict):
            for v in content.values():
                if isinstance(v, str):
                    texts.append(v)
        return texts

    def _answer_from_memory(self, user_text: str) -> Optional[str]:
        """Try to answer simple questions directly from MEMORY rows."""
        if not user_text:
            return None

        lower = user_text.strip().lower()
        agent_id = self._get_agent_id()

        # Normalize whitespace
        lower = re.sub(r"\s+", " ", lower)

        # --- Name questions ---
        if "what is my name" in lower or "what's my name" in lower or "who am i" in lower:
            candidates: List[str] = []
            mem_items = memory_store.recent_memories(agent_id=agent_id, limit=100)
            for item in mem_items:
                for txt in self._extract_text_fields(item):
                    t = txt.strip()
                    tl = t.lower()
                    m = re.search(r"\bmy name is\s+(.+)", tl, re.IGNORECASE)
                    if m:
                        name = t[m.start(1) : m.end(1)].strip(" .!?,")
                        candidates.append(name)
                        continue
                    m2 = re.search(r"\b(i[' ]?m|i am)\s+(.+)", tl, re.IGNORECASE)
                    if m2:
                        name = t[m2.start(2) : m2.end(2)].strip(" .!?,")
                        candidates.append(name)
            if candidates:
                name = candidates[0]
                if name:
                    return f"Based on what you've told me before, your name is {name}."

        # --- Age questions ---
        if (
            "how old am i" in lower
            or "what is my age" in lower
            or "what's my age" in lower
            or lower.strip(" ?!.") in {"my age", "age"}
        ):
            mem_items = memory_store.recent_memories(agent_id=agent_id, limit=100)
            ages: List[int] = []
            for item in mem_items:
                for txt in self._extract_text_fields(item):
                    tl = txt.lower()
                    m = re.search(
                        r"\b(i[' ]?m|i am)\s+(\d{1,3})(?:\s*(?:years? old|yo))?",
                        tl,
                        re.IGNORECASE,
                    )
                    if m:
                        try:
                            ages.append(int(m.group(2)))
                        except ValueError:
                            continue
            if ages:
                age = ages[0]
                return f"From my memory, you told me you are {age} years old."

        # --- Dog questions ---
        if "do i have dogs" in lower or "do i have a dog" in lower or "do i have any dogs" in lower:
            mem_items = memory_store.recent_memories(agent_id=agent_id, limit=200)
            dog_sentences: List[str] = []
            dog_names: List[str] = []
            for item in mem_items:
                for txt in self._extract_text_fields(item):
                    tl = txt.lower()
                    if "dog" in tl:
                        dog_sentences.append(txt)
                        # Try to pull out names after "named"
                        for m in re.finditer(
                            r"named\s+([A-Za-z][A-Za-z0-9_-]*)", txt, re.IGNORECASE
                        ):
                            dog_names.append(m.group(1))
            if dog_sentences:
                if dog_names:
                    unique_names: List[str] = []
                    for n in dog_names:
                        if n not in unique_names:
                            unique_names.append(n)
                    if len(unique_names) == 1:
                        return f"Yes. You've told me you have a dog named {unique_names[0]}."
                    else:
                        joined = ", ".join(unique_names)
                        return f"Yes. You've told me you have dogs, including {joined}."
                return "Yes, I remember you telling me you have dogs."

        return None

    # ------------------------------------------------------------------
    # Public API expected by agent_core.llm_client
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Return an LLM response shaped for agent_core.planner.handle_llm_result."""
        logger.debug("Chat called with %s messages; generating reply", len(messages))

        # Find the most recent user message; this is what we'll persist and may answer from.
        last_user_text: Optional[str] = None
        for msg in reversed(messages):
            if (msg.get("role") or "").lower() == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    last_user_text = content
                else:
                    last_user_text = str(content)
                break

        # First, see if we can answer directly from memory.
        reply_text: Optional[str] = None
        tool_calls: List[Dict[str, Any]] = []

        if last_user_text:
            mem_answer = self._answer_from_memory(last_user_text)
            if mem_answer:
                logger.debug(
                    "Answering from memory for user text preview='%s'", last_user_text[:60]
                )
                reply_text = mem_answer

        # If we couldn't answer from memory, fall back to the LLM + tools.
        if reply_text is None:
            reply_text, tool_calls = self._chat_bedrock(messages, tools)

        # Helper: decide whether a user utterance is worth persisting.
        def _should_store(text: str) -> bool:
            stripped = (text or "").strip()
            if not stripped:
                return False

            lower = stripped.lower()
            tokens = lower.split()

            noise_words = {
                "hi",
                "hey",
                "hello",
                "yo",
                "ok",
                "okay",
                "k",
                "kk",
                "lol",
                "help",
                "test",
                "exit",
                "quit",
                "bye",
                "goodbye",
            }
            if lower in noise_words:
                return False
            if len(tokens) == 1 and tokens[0] in noise_words:
                return False

            if lower.startswith(("command ", "cmd ", "run ", "shell ")):
                return False

            pronoun_keywords = {
                "i",
                "i'm",
                "im",
                "my",
                "me",
                "mine",
                "we",
                "our",
                "you",
                "your",
                "yours",
            }
            has_digit = any(ch.isdigit() for ch in lower)
            if len(tokens) <= 2 and not (set(tokens) & pronoun_keywords) and not has_digit:
                return False

            return True

        # Helper: classify the memory kind based on simple patterns.
        def _classify_memory_kind(text: str) -> str:
            lower = (text or "").lower()

            speculative_markers = [
                "i think",
                "i guess",
                "maybe",
                "probably",
                "i'm not sure",
                "im not sure",
                "i am not sure",
                "not sure",
                "i wonder",
                "might be",
            ]
            for marker in speculative_markers:
                if marker in lower:
                    return "SPECULATION"

            compliment_markers = [
                "you are the best",
                "you're the best",
                "youre the best",
                "you are great",
                "you are awesome",
                "you are amazing",
                "you are helpful",
                "you are very helpful",
            ]
            if any(marker in lower for marker in compliment_markers):
                return "AI_INSIGHT"

            if ("you are" in lower or "you're" in lower or "youre" in lower) and any(
                pw in lower
                for pw in [
                    "good",
                    "great",
                    "amazing",
                    "awesome",
                    "helpful",
                    "smart",
                    "nice",
                    "kind",
                    "cool",
                ]
            ):
                return "AI_INSIGHT"

            return "FACT"

        # Synthesize a store_memory call if appropriate and not already present.
        if last_user_text and _should_store(last_user_text):
            has_store = any((tc.get("name") == "store_memory") for tc in tool_calls)
            if not has_store:
                memory_kind = _classify_memory_kind(last_user_text)
                mem_args = {
                    "memory_kind": memory_kind,
                    "summary": last_user_text[:160],
                    "content": {"user_text": last_user_text},
                    "tags": [],
                    "links": [],
                }
                logger.debug(
                    "Synthesizing store_memory(kind=%s) for user text preview='%s'",
                    memory_kind,
                    last_user_text[:60],
                )
                tool_calls.append(
                    {
                        "name": "store_memory",
                        "arguments": json.dumps(mem_args),
                    }
                )
        elif last_user_text:
            logger.debug(
                "Skipping memory synthesis for noisy user text preview='%s'",
                last_user_text[:60],
            )

        assistant_msg: Dict[str, Any] = {
            "role": "assistant",
            "content": reply_text or "",
            "tool_calls": tool_calls,
        }
        logger.debug("Returning assistant message with %s tool calls", len(tool_calls))
        return {"messages": [*messages, assistant_msg]}

    @classmethod
    def from_env(cls) -> "AwsModelClient":
        """Factory using MODEL_ID / AWS_REGION from the Lambda/CLI environment."""
        return cls(model_id=os.getenv("MODEL_ID"), region_name=os.getenv("AWS_REGION"))
