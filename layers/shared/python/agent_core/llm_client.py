from __future__ import annotations

from typing import Any, Dict, List

from . import tools as agent_tools


def build_system_prompt(personality_prompt: str) -> str:
    memory_instructions = (
        "You have tools to store and query long-term memory. "
        "Use them to distinguish FACT, SPECULATION, and AI_INSIGHT memories."
    )
    return personality_prompt.strip() + "\n\n" + memory_instructions


def chat_with_tools(
    model_client: Any,
    messages: List[Dict[str, str]],
    tools: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Thin wrapper around a generic tool-using chat model.

    `model_client` must provide a `.chat(messages=..., tools=...)` method that
    returns a dict with at least:
    {
      "messages": [..., { "role": "assistant", "content": "...", "tool_calls": [...] }]
    }

    This avoids hard-coding a specific provider. Adapt to OpenAI/Bedrock/etc.
    """
    if tools is None:
        tools = agent_tools.TOOLS_SPEC

    # This is intentionally abstract; replace with your own client.
    response = model_client.chat(messages=messages, tools=tools)
    return response
