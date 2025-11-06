"""Wrapper around the OpenAI API for conversational responses."""
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
    """Small helper around the OpenAI Chat Completions API."""

    def __init__(self, secret_id: str, region_name: str, model: str = "gpt-4o-mini") -> None:
        self._secrets = boto3.client("secretsmanager", region_name=region_name)
        self._secret_id = secret_id
        self._model = model
        self._client: OpenAI | None = None

    def _ensure_credentials(self) -> None:
        if self._client:
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

    def chat(self, messages: list[dict[str, str]], tools: list[dict[str, Any]] | None = None) -> LLMResponse:
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


__all__ = ["LLMClient", "LLMResponse"]
