from __future__ import annotations

"""
AWS Bedrock-backed model client for the Rank-4 agent.

Responsibilities:
- Read MODEL_ID / AWS_REGION from the environment.
- Talk to Bedrock via either the Titan text-express API shape or the generic
  `converse` API, depending on the model id.
- Return responses in the structure expected by agent_core.llm_client and
  agent_core.planner.handle_llm_result.

This version also synthesizes simple `store_memory` tool calls based on the
latest user message so that the planner will persist MEMORY rows into the
AgentStateTable, even though we are not yet using Bedrock's native tool
calling.
"""

import json
import logging
import os
from typing import Any, Dict, List, Tuple

import boto3

logger = logging.getLogger(__name__)


class AwsModelClient:
    def __init__(self, model_id: str | None = None, region_name: str | None = None) -> None:
        # Prefer explicit args, then env, then a sensible Bedrock default.
        self._model: str = (model_id or os.getenv("MODEL_ID") or "amazon.titan-text-express-v1").strip()
        self._region: str = (region_name or os.getenv("AWS_REGION") or "us-east-1").strip()
        if not self._model:
            raise RuntimeError("MODEL_ID must be configured (env var or constructor).")
        self._bedrock = boto3.client("bedrock-runtime", region_name=self._region)
        logger.info("Initialized AwsModelClient model=%s region=%s", self._model, self._region)

    # ---- Internal helpers (adapted from the 0.0.20 LLM client) ----

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

    def _chat_titan(self, system: str, convo: List[Dict[str, str]]) -> str:
        """Titan text-express style invocation: build a flat prompt from messages."""
        parts: List[str] = []
        if system:
            parts.append(f"System:\n{system}\n")
        for msg in convo:
            role = msg["role"]
            text = msg["content"]
            if role == "user":
                parts.append(f"User:\n{text}\n")
            elif role == "assistant":
                parts.append(f"Assistant:\n{text}\n")
            else:
                parts.append(f"{role.capitalize()}:\n{text}\n")

        # End with an Assistant: cue so Titan continues as assistant.
        parts.append("Assistant:\n")
        prompt = "\n".join(parts)

        payload: Dict[str, Any] = {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 300,
                "temperature": 0.4,
                "topP": 0.9,
                "stopSequences": ["\nUser:", "\nSystem:"],
            },
        }

        logger.debug("Invoking Titan model %s with %s convo messages", self._model, len(convo))
        resp = self._bedrock.invoke_model(
            modelId=self._model or "amazon.titan-text-express-v1",
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload),
        )
        out = json.loads(resp["body"].read())
        results = out.get("results") or []
        if not results:
            return "I couldn't generate a response."
        text = results[0].get("outputText") or "I couldn't generate a response."
        return text

    def _chat_converse(self, system: str, convo: List[Dict[str, str]]) -> str:
        """Generic Bedrock `converse` call that returns text only."""
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

        kwargs: Dict[str, Any] = {
            "modelId": self._model,
            "messages": messages,
            "inferenceConfig": {
                "maxTokens": 300,
                "temperature": 0.4,
                "topP": 0.9,
            },
        }
        had_system = bool(system)
        if system:
            kwargs["system"] = [{"text": system}]

        try:
            logger.debug(
                "Invoking converse model %s with %s messages (system_present=%s)",
                self._model,
                len(messages),
                had_system,
            )
            out = self._bedrock.converse(**kwargs)
            text = out["output"]["message"]["content"][0]["text"]
            return text
        except Exception as e:  # pragma: no cover - defensive fallback
            msg = str(e)
            logger.warning("Bedrock converse error: %s", msg)
            # Some models don't support system; retry without it.
            if had_system and ("system" in msg.lower() or "doesn't support system" in msg.lower()):
                kwargs.pop("system", None)
                out = self._bedrock.converse(**kwargs)
                text = out["output"]["message"]["content"][0]["text"]
                return text
            # Last resort: surface a generic failure message.
            return "I had trouble talking to the language model backend."

    def _chat_bedrock(self, messages: List[Dict[str, Any]]) -> str:
        """Route to Titan-style or Converse-style call based on model id."""
        system, convo = self._split_system_and_messages(messages)
        model_id = (self._model or "").strip()
        if not model_id:
            raise RuntimeError("MODEL_ID must be set for Bedrock provider.")
        logger.debug("Routing chat for model %s (system_present=%s)", model_id, bool(system))
        if model_id.startswith("amazon.titan-text-express"):
            return self._chat_titan(system, convo)
        # Default: use the generic converse API.
        return self._chat_converse(system, convo)

    # ---- Public API expected by agent_core.llm_client ----

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] | None = None,  # currently ignored for Bedrock
    ) -> Dict[str, Any]:
        """Return an LLM response shaped for agent_core.planner.handle_llm_result.

        The structure matches what handle_llm_result expects:

        {
          "messages": [
            ...original messages...,
            {
              "role": "assistant",
              "content": "<reply>",
              "tool_calls": [
                { "name": "store_memory", "arguments": "{...}" },
                ...
              ]
            }
          ]
        }

        For now we synthesize a `store_memory` tool call for the most
        recent user message so that the planner will persist it as a MEMORY
        row in the AgentStateTable. This is a stopgap until native Bedrock
        tool calling is wired through.

        This implementation also applies simple heuristics to decide
        whether to store a memory at all, and to classify it as FACT,
        SPECULATION, or AI_INSIGHT based on the text.
        """
        logger.debug("Chat called with %s messages; generating reply", len(messages))
        reply_text = self._chat_bedrock(messages)

        # Helper: decide whether a user utterance is worth persisting.
        def _should_store(text: str) -> bool:
            stripped = (text or "").strip()
            if not stripped:
                return False

            lower = stripped.lower()
            tokens = lower.split()

            # Very short noise / fillers.
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

            # Command-like chatter that shouldn't become long-term memory.
            if lower.startswith(("command ", "cmd ", "run ", "shell ")):
                return False

            # If it's extremely short and doesn't reference self/you and has
            # no digits, treat as noise.
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

            # Compliments or judgments about the AI or the interaction.
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

            # Default: treat as a concrete fact (often about the user).
            return "FACT"

        # Find the most recent user message; this is what we'll persist.
        last_user_text: str | None = None
        for msg in reversed(messages):
            if (msg.get("role") or "").lower() == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    last_user_text = content
                else:
                    last_user_text = str(content)
                break

        tool_calls: List[Dict[str, Any]] = []

        if last_user_text:
            if _should_store(last_user_text):
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
            else:
                logger.debug(
                    "Skipping memory synthesis for noisy user text preview='%s'",
                    last_user_text[:60],
                )

        assistant_msg: Dict[str, Any] = {
            "role": "assistant",
            "content": reply_text,
            "tool_calls": tool_calls,
        }
        logger.debug("Returning assistant message with %s tool calls", len(tool_calls))
        return {"messages": [*messages, assistant_msg]}

    @classmethod
    def from_env(cls) -> "AwsModelClient":
        """Factory using MODEL_ID / AWS_REGION from the Lambda/CLI environment."""
        return cls(model_id=os.getenv("MODEL_ID"), region_name=os.getenv("AWS_REGION"))
