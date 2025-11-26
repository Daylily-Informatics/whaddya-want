"""Helpers for loading client configuration defaults."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def _load_mapping(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}

    text = path.read_text(encoding="utf-8")
    try:  # Prefer PyYAML if available
        import yaml  # type: ignore

        data = yaml.safe_load(text) or {}
    except Exception:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:  # pragma: no cover - config error path
            raise RuntimeError(
                "Failed to parse client params; install PyYAML or use JSON-compatible content."
            ) from exc

    if not isinstance(data, dict):
        raise RuntimeError("Client params must be defined as a mapping.")
    return data


def find_config_file(filename: str) -> Path:
    """Locate a config file by walking up from this module's directory."""

    here = Path(__file__).resolve()
    for parent in [here.parent] + list(here.parents):
        candidate = parent / "config" / filename
        if candidate.exists():
            return candidate
    return here.parent.parent / "config" / filename


def load_client_params() -> Dict[str, Any]:
    """Load defaults for the CLI from config/client_params.yaml."""

    config_path = find_config_file("client_params.yaml")
    return _load_mapping(config_path)


__all__ = ["find_config_file", "load_client_params"]
