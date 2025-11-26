"""Configuration helpers for the AI companion services."""
from __future__ import annotations

import os
from dataclasses import dataclass

from .prompts import find_config_file


def _env(name: str, default: str | None = None) -> str:
    """Read an environment variable or raise a helpful error."""
    try:
        value = os.environ[name]
    except KeyError as exc:  # pragma: no cover - simple configuration helper
        if default is not None:
            return default
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            "Set it in your Lambda configuration or local .env file."
        ) from exc
    return value


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Holds runtime configuration for the Lambda broker."""

    region_name: str
    conversation_table: str
    audio_bucket: str
    secrets_id: str
    voice_id: str
    model_id: str
    history_limit: int = 10
    memory_ttl_seconds: int = 7 * 24 * 3600
    prompts_path: str = str(find_config_file("prompts.yaml"))

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        """Create a configuration object from environment variables."""
        return cls(
            region_name=_env("AWS_REGION", default="us-east-1"),
            conversation_table=_env("CONVERSATION_TABLE"),
            audio_bucket=_env("AUDIO_BUCKET"),
            secrets_id=_env("LLM_SECRET_ID"),
            voice_id=_env("POLLY_VOICE", default="Joanna"),
            model_id=os.getenv("LLM_MODEL_ID")
            or os.getenv("MODEL_ID")
            or "gpt-4o-mini",
            history_limit=int(_env("HISTORY_LIMIT", default="10")),
            memory_ttl_seconds=int(_env("MEMORY_TTL_SECONDS", default=str(7 * 24 * 3600))),
            prompts_path=os.getenv("PROMPTS_CONFIG") or str(find_config_file("prompts.yaml")),
        )


__all__ = ["RuntimeConfig"]
