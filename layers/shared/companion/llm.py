"""Wrapper around LLM providers (OpenAI and Bedrock) for conversational responses."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, Optional

import boto3
from openai import OpenAI


Provider = Literal["openai", "bedrock"]


@dataclass(slots=True)
class LLMResponse:
    message: str
    tool_calls: list[dict[str, Any]]


class LLMClient:
    """Helper around OpenAI Chat Completions and Bedrock Converse/Titan."""

    def __init__(
        self,
        *,
        provider: Provider,
        model: str,
        region_name: str,
        secret_id: Optional[str] = None,
    ) -> None:
        self._provider: Provider = provider
        self._model = model
        self._region = region_name
        self._secret_id = (secret_id or "").strip()

        # Lazy-initialised clients
        self._secrets = (
            boto3.client("secretsmanager", region_name=region_name)
            if self._provider == "openai" and self._secret_id
            else None
        )
        self._openai_client: OpenAI | None = None
        self._bedrock = (
            boto3.client("bedrock-runtime", region_name=region_name)
            if self._provider == "bedrock"
            else None
        )

    # ---- OpenAI ----
    def _ensure_openai(self) -> None:
        if self._openai_client is not None:
            return
        if not self._secrets or not self._secret_id:
            raise RuntimeError("OpenAI provider selected but LLM_SECRET_ID is not configured.")
        secret = self._secrets.get_secret_value(SecretId=self._secret_id)
        payload = secret.get("SecretString")
        if not payload:
            raise RuntimeError("SecretString missing from Secrets Manager response")
        data = json.loads(payload)
        api_key = data.get("api_key") or data.get("openai_api_key")
        if not api_key:
            raise RuntimeError(
                "Secrets Manager payload must contain 'api_key' or 'openai_api_key'."
            )
        self._openai_client = OpenAI(api_key=api_key)

    def _chat_openai(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        self._ensure_openai()
        assert self._openai_client is not None  # for type checkers
        model = self._model or "gpt-4o-mini"
        response = self._openai_client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
        )
        choice = response.choices[0].message
        tool_calls = [call.model_dump() for call in choice.tool_calls or []]
        return LLMResponse(message=choice.content or "", tool_calls=tool_calls)

    # ---- Bedrock ----
    def _split_system_and_messages(
        self, messages: list[dict[str, str]]
    ) -> tuple[str, list[dict[str, str]]]:
        system_parts: list[str] = []
        non_system: list[dict[str, str]] = []
        for m in messages:
            role = m.get("role") or ""
            content = m.get("content") or ""
            if role == "system":
                system_parts.append(str(content))
            else:
                non_system.append({"role": role, "content": str(content)})
        system = "\n\n".join(p for p in system_parts if p.strip())
        return system, non_system

    def _chat_titan(
        self,
        system: str,
        convo: list[dict[str, str]],
    ) -> LLMResponse:
        """Bedrock Titan text-express: plain-text prompt."""
        if not self._bedrock:
            raise RuntimeError("Bedrock client not initialised")
        parts: list[str] = []
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
        # Ensure we end with an assistant turn
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
            return LLMResponse(message="I couldn’t generate a response.", tool_calls=[])
        text = results[0].get("outputText") or "I couldn’t generate a response."
        return LLMResponse(message=text, tool_calls=[])

    def _chat_converse(
        self,
        system: str,
        convo: list[dict[str, str]],
    ) -> LLMResponse:
        """Model-agnostic Bedrock Converse call."""
        if not self._bedrock:
            raise RuntimeError("Bedrock client not initialised")

        messages: list[dict[str, Any]] = []
        for msg in convo:
            role = msg["role"]
            text = msg["content"]
            messages.append({"role": role, "content": [{"text": text}]})

        kwargs: dict[str, Any] = {
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
            return LLMResponse(message=text, tool_calls=[])
        except Exception as e:
            msg = str(e)
            if had_system and ("system" in msg.lower() or "doesn't support system" in msg.lower()):
                kwargs.pop("system", None)
                out = self._bedrock.converse(**kwargs)
                text = out["output"]["message"]["content"][0]["text"]
                return LLMResponse(message=text, tool_calls=[])
            raise

    def _chat_bedrock(
        self,
        messages: list[dict[str, str]],
    ) -> LLMResponse:
        system, convo = self._split_system_and_messages(messages)
        model_id = (self._model or "").strip()
        if not model_id:
            raise RuntimeError("MODEL_ID must be set for Bedrock provider.")
        if model_id.startswith("amazon.titan-text-express"):
            return self._chat_titan(system, convo)
        return self._chat_converse(system, convo)

    # ---- Public API ----
    def chat(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        if self._provider == "openai":
            return self._chat_openai(messages, tools)
        return self._chat_bedrock(messages)


__all__ = ["LLMClient", "LLMResponse"]
