from __future__ import annotations

"""
AWS Bedrock-backed model client for the Rank-4 agent.

This is a minimal implementation that:
- Reads MODEL_ID from the environment (defaulting to a Titan text model).
- Uses the Bedrock Runtime client for text generation.
- Ignores the `tools` argument for now and returns a plain assistant reply with
  no tool calls. The Rank-4 planner will therefore not execute tools yet, but
  the conversational loop will work end-to-end.

If you want richer tool use, you can extend this client to:
- Encode `tools` into the system prompt, and/or
- Ask the model to return a JSON object with `reply_text` and `tool_calls`,
  then parse that here into the expected structure.
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
        self._model = (model_id or os.getenv("MODEL_ID") or "amazon.titan-text-express-v1").strip()
        self._region = (region_name or os.getenv("AWS_REGION") or "us-east-1").strip()
        if not self._model:
            raise RuntimeError("MODEL_ID must be configured (env var or constructor).")
        self._bedrock = boto3.client("bedrock-runtime", region_name=self._region)
        logger.debug("Initialized AwsModelClient", extra={"model": self._model, "region": self._region})

    # ---- Internal helpers (adapted from 0.0.20 LLM client) ----

    def _split_system_and_messages(
        self, messages: List[Dict[str, str]]
    ) -> Tuple[str, List[Dict[str, str]]]:
        system_parts: List[str] = []
        non_system: List[Dict[str, str]] = []
        for m in messages:
            role = m.get("role") or ""
            content = m.get("content") or ""
            if role == "system":
                system_parts.append(str(content))
            else:
                non_system.append({"role": role, "content": str(content)})
        system_text = "\n\n".join(system_parts).strip()
        return system_text, non_system

    def _chat_titan(self, system: str, convo: List[Dict[str, str]]) -> str:
        """
        Titan text-express style invocation: build a flat prompt from messages.

        This is copied and simplified from the 0.0.20 LLM client; it treats the
        conversation as alternating User/Assistant turns and appends an
        'Assistant:' at the end so Titan continues the assistant.
        """
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
                parts.append(f"{role.title()}:\n{text}\n")
        parts.append("Assistant:\n")
        prompt = "\n".join(parts)

        payload = {
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
        """
        Generic Bedrock Converse call using the `converse` API.

        This is intentionally conservative and does not use native tool
        support; it only returns text for now.
        """
        messages: List[Dict[str, Any]] = []
        for msg in convo:
            role = msg["role"]
            text = msg["content"]
            messages.append({"role": role, "content": [{"text": text}]})

        kwargs: Dict[str, Any] = {
            "modelId": self._model,
            "messages": messages,
            "inferenceConfig": {
                "maxTokens": 300,
                "temperature": 0.4,
                "topP": 0.9,
            },
        }

        had_system = False
        if system:
            kwargs["system"] = [{"text": system}]
            had_system = True

        try:
            out = self._bedrock.converse(**kwargs)
            text = out["output"]["message"]["content"][0]["text"]
            return text
        except Exception as e:  # pragma: no cover - defensive fallback
            msg = str(e)
            logger.warning("Bedrock converse call failed: %s", msg)
            # Some models don't support system; retry without it.
            if had_system and ("system" in msg.lower() or "doesn't support system" in msg.lower()):
                kwargs.pop("system", None)
                out = self._bedrock.converse(**kwargs)
                text = out["output"]["message"]["content"][0]["text"]
            return text
            # Last resort: surface a generic failure message.
            return "I had trouble talking to the language model backend."

    def _chat_bedrock(self, messages: List[Dict[str, str]]) -> str:
        system, convo = self._split_system_and_messages(messages)
        model_id = (self._model or "").strip()
        if not model_id:
            raise RuntimeError("MODEL_ID must be set for Bedrock provider.")
        if model_id.startswith("amazon.titan-text-express"):
            return self._chat_titan(system, convo)
        return self._chat_converse(system, convo)

    # ---- Public API expected by agent_core.llm_client ----

    def chat(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]] | None = None,  # noqa: ARG002 - currently unused
    ) -> Dict[str, Any]:
        """
        Return an LLM response shaped for agent_core.planner.handle_llm_result.

        Structure:
        {
          "messages": [
            ...original messages...,
            { "role": "assistant", "content": "<reply>", "tool_calls": [] }
          ]
        }
        """
        reply_text = self._chat_bedrock(messages)
        # For now we do not support tool calls, so this is always empty.
        assistant_msg: Dict[str, Any] = {
            "role": "assistant",
            "content": reply_text,
            "tool_calls": [],
        }
        return {"messages": [*messages, assistant_msg]}

    @classmethod
    def from_env(cls) -> "AwsModelClient":
        """Factory using MODEL_ID / AWS_REGION from the Lambda/CLI environment."""
        return cls(model_id=os.getenv("MODEL_ID"), region_name=os.getenv("AWS_REGION"))
