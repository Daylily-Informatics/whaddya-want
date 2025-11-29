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
import os
from typing import Any, Dict, List, Tuple

import boto3


class AwsModelClient:
    def __init__(self, model_id: str | None = None, region_name: str | None = None) -> None:
        # Prefer explicit args, then env, then a sensible Bedrock default.
        self._model: str = (model_id or os.getenv("MODEL_ID") or "amazon.titan-text-express-v1").strip()
        self._region: str = (region_name or os.getenv("AWS_REGION") or "us-east-1").strip()
        if not self._model:
            raise RuntimeError("MODEL_ID must be configured (env var or constructor).")
        self._bedrock = boto3.client("bedrock-runtime", region_name=self._region)

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
            out = self._bedrock.converse(**kwargs)
            text = out["output"]["message"]["content"][0]["text"]
            return text
        except Exception as e:  # pragma: no cover - defensive fallback
            msg = str(e)
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

        For now we synthesize a single `store_memory` tool call for the most
        recent user message so that the planner will persist it as a MEMORY
        row in the AgentStateTable. This is a stopgap until native Bedrock
        tool calling is wired through.
        """
        reply_text = self._chat_bedrock(messages)

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
            mem_args = {
                "memory_kind": "FACT",
                "summary": last_user_text[:160],
                "content": {"user_text": last_user_text},
                "tags": [],
                "links": [],
            }
            tool_calls.append(
                {
                    "name": "store_memory",
                    "arguments": json.dumps(mem_args),
                }
            )

        assistant_msg: Dict[str, Any] = {
            "role": "assistant",
            "content": reply_text,
            "tool_calls": tool_calls,
        }
        return {"messages": [*messages, assistant_msg]}

    @classmethod
    def from_env(cls) -> "AwsModelClient":
        """Factory using MODEL_ID / AWS_REGION from the Lambda/CLI environment."""
        return cls(model_id=os.getenv("MODEL_ID"), region_name=os.getenv("AWS_REGION"))
