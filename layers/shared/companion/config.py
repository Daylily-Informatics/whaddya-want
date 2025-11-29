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
    history_limit: int = 10
    prompts_path: str = str(find_config_file("prompts.yaml"))
    llm_provider: str = "bedrock"
    llm_model_id: str = ""
    use_memory: bool = True
    vision_model_id: str = ""  # Bedrock multimodal model for vision (optional)
    ais_memory_table: str = ""
    ais_memory_ttl_days: int = 30

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        """Create a configuration object from environment variables."""
        region = _env("AWS_REGION", default="us-east-1")
        table = _env("CONVERSATION_TABLE")
        bucket = _env("AUDIO_BUCKET")
        secrets = os.getenv("LLM_SECRET_ID", "")
        voice = _env("POLLY_VOICE", default="Joanna")
        history = int(_env("HISTORY_LIMIT", default="10"))
        prompts_path = os.getenv("PROMPTS_CONFIG") or str(find_config_file("prompts.yaml"))

        model_id = _env("MODEL_ID", default="amazon.titan-text-express-v1")
        vision_model_id = os.getenv("VISION_MODEL_ID", "").strip()
        ais_memory_table = os.getenv("AIS_MEMORY_TABLE", "").strip()
        ais_memory_ttl_days = int(os.getenv("AIS_MEMORY_TTL_DAYS", "30"))

        provider = os.getenv("LLM_PROVIDER", "").strip().lower()
        if not provider:
            # Heuristic: if a secret is configured, assume OpenAI; otherwise Bedrock.
            provider = "openai" if secrets else "bedrock"
        if provider not in {"openai", "bedrock"}:
            raise RuntimeError("LLM_PROVIDER must be 'openai' or 'bedrock' if set.")

        use_memory = os.getenv("USE_MEMORY", "true").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }

        return cls(
            region_name=region,
            conversation_table=table,
            audio_bucket=bucket,
            secrets_id=secrets,
            voice_id=voice,
            history_limit=history,
            prompts_path=prompts_path,
            llm_provider=provider,
            llm_model_id=model_id,
            use_memory=use_memory,
            vision_model_id=vision_model_id,
            ais_memory_table=ais_memory_table,
            ais_memory_ttl_days=ais_memory_ttl_days,
        )


__all__ = ["RuntimeConfig"]
