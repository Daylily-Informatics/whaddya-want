"""Wrapper around Bedrock/OpenAI APIs for conversational responses."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import boto3
from openai import OpenAI


@dataclass(slots=True)
class LLMResponse:
    message: str
    tool_calls: list[dict[str, Any]]


class LLMClient:
    """Helper around OpenAI Chat Completions and Bedrock Converse APIs."""

    def __init__(self, secret_id: str, region_name: str, model: str = "gpt-4o-mini") -> None:
        self._secrets = boto3.client("secretsmanager", region_name=region_name)
        self._secret_id = secret_id
        self._model = model
        self._client: OpenAI | None = None
        self._bedrock = boto3.client("bedrock-runtime", region_name=region_name)
        bedrock_prefixes = ("amazon.", "anthropic.", "meta.", "cohere.", "mistral.")
        self._provider = "bedrock" if model.startswith(bedrock_prefixes) else "openai"

    def _ensure_credentials(self) -> None:
        if self._provider != "openai" or self._client:
            return
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
        self._client = OpenAI(api_key=api_key)

    def _chat_openai(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> LLMResponse:
        self._ensure_credentials()
        assert self._client  # narrow type
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=tools,
        )
        choice = response.choices[0].message
        tool_calls = [call.model_dump() for call in choice.tool_calls or []]
        return LLMResponse(message=choice.content or "", tool_calls=tool_calls)

    def _chat_bedrock(self, messages: list[dict[str, Any]]) -> LLMResponse:
        sys_msgs = [{"text": m["content"]} for m in messages if m.get("role") == "system"]
        chat_msgs = [
            {"role": m["role"], "content": [{"text": m["content"]}]}
            for m in messages
            if m.get("role") != "system"
        ]
        kwargs: dict[str, Any] = {
            "modelId": self._model,
            "messages": chat_msgs,
            "inferenceConfig": {"maxTokens": 300, "temperature": 0.4, "topP": 0.9},
        }
        if sys_msgs:
            kwargs["system"] = sys_msgs
        response = self._bedrock.converse(**kwargs)
        msg = response["output"]["message"]["content"][0]["text"]
        return LLMResponse(message=msg, tool_calls=[])

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        if self._provider == "bedrock":
            return self._chat_bedrock(messages)
        return self._chat_openai(messages, tools)


__all__ = ["LLMClient", "LLMResponse"]
