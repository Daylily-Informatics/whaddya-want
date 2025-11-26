"""Prompt configuration helpers for the companion."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:  # Prefer PyYAML if available to allow flexible config files
        import yaml  # type: ignore

        data = yaml.safe_load(text) or {}
    except Exception:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:  # pragma: no cover - config error path
            raise RuntimeError(
                "Failed to parse prompt config; install PyYAML or use JSON-compatible content."
            ) from exc

    if not isinstance(data, dict):
        raise RuntimeError("Prompt config must be a mapping of keys to values.")
    return data


def find_config_file(filename: str) -> Path:
    """Locate a config file by walking up from this module's directory."""

    here = Path(__file__).resolve()
    for parent in [here.parent] + list(here.parents):
        candidate = parent / "config" / filename
        if candidate.exists():
            return candidate
    # Fallback to project root convention even if it doesn't exist yet
    return here.parent.parent.parent / "config" / filename


def load_personality_prompt(path: str | None = None) -> str:
    """Load the system/personality prompt from a config file."""

    config_path = Path(path) if path else find_config_file("prompts.yaml")
    data = _load_mapping(config_path)
    prompt = (
        data.get("personality_prompt")
        or data.get("system_prompt")
        or data.get("behavior_prompt")
    )
    if not isinstance(prompt, str) or not prompt.strip():
        raise RuntimeError("Prompt config must provide a non-empty personality prompt.")
    return prompt.strip()


__all__ = ["find_config_file", "load_personality_prompt"]