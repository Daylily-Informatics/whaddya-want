from __future__ import annotations

"""
LLM client helpers for the Rank-4 agent.

Responsibilities:

- Load prompt templates from layers/shared/config/prompts.yaml
- Build the system prompt used for the model, based on that config
- Provide a simple chat_with_tools(model_client, messages, tools) wrapper
  that delegates to the model client (AwsModelClient or similar).

This file is the single place where we interpret prompts.yaml; all other
code should call build_system_prompt() instead of hard-coding personality
strings.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml  # PyYAML should be in the shared layer requirements


logger = logging.getLogger(__name__)

_PROMPTS_CACHE: Optional[Dict[str, Any]] = None


def _prompts_path() -> Path:
    """
    Return the absolute path to layers/shared/config/prompts.yaml.

    This file (llm_client.py) lives under:
        layers/shared/python/agent_core/llm_client.py

    The prompts.yaml file lives under:
        layers/shared/config/prompts.yaml

    So we go up two levels to "shared" and then into config/.
    """
    here = Path(__file__).resolve()
    shared_dir = here.parents[2]  # .../layers/shared
    return shared_dir / "config" / "prompts.yaml"


def load_prompts() -> Dict[str, Any]:
    """Load and cache the prompts.yaml configuration."""
    global _PROMPTS_CACHE
    if _PROMPTS_CACHE is not None:
        return _PROMPTS_CACHE

    path = _prompts_path()
    if not path.exists():
        logger.warning("prompts.yaml not found at %s; using empty prompts.", path)
        _PROMPTS_CACHE = {}
        return _PROMPTS_CACHE

    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            logger.warning("prompts.yaml did not parse as a mapping; got %r", type(data))
            data = {}
        _PROMPTS_CACHE = data
        logger.debug("Loaded prompts.yaml from %s", path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to load prompts.yaml from %s: %s", path, exc)
        _PROMPTS_CACHE = {}

    return _PROMPTS_CACHE


def get_personality_prompt() -> str:
    """
    Return the base personality/system prompt for the agent from prompts.yaml.

    Expected key (but we are tolerant):
        personality_prompt: |-
          Your name is Marvin.
          You are ...
    """
    prompts = load_prompts()
    base = prompts.get("personality_prompt") or ""
    if not isinstance(base, str):
        logger.warning("personality_prompt in prompts.yaml is not a string; ignoring.")
        return ""
    return base.strip()


def build_system_prompt(extra_personality: Optional[str] = None) -> str:
    """
    Build the system prompt used for the LLM.

    - Always starts from the personality_prompt in prompts.yaml.
    - If extra_personality is provided (e.g., from an HTTP body override),
      it is appended, separated by a blank line. This lets you extend or
      specialize the base persona for a given session without replacing
      the config-backed prompt entirely.
    """
    base = get_personality_prompt()
    if extra_personality:
        extra = extra_personality.strip()
        if extra:
            combined = f"{base}\n\n{extra}" if base else extra
            logger.debug(
                "Built system prompt from prompts.yaml with extra personality (%d base chars, %d extra chars).",
                len(base),
                len(extra),
            )
            return combined

    logger.debug("Built system prompt from prompts.yaml (no extra personality).")
    return base


def chat_with_tools(
    model_client: Any,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Thin wrapper around model_client.chat(..., tools=...).

    The AwsModelClient implementation is responsible for:
      - Sending messages + tools to Bedrock (or other backend)
      - Returning a structure compatible with planner.handle_llm_result:
            {
              "messages": [..., {"role": "assistant", "content": "...", "tool_calls": [...]}]
            }
    """
    logger.debug("Invoking model_client.chat with %d messages and %d tools",
                 len(messages), 0 if tools is None else len(tools))
    return model_client.chat(messages=messages, tools=tools)
