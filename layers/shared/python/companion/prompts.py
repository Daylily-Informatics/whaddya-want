"""Prompt configuration helpers for the companion."""
from __future__ import annotations

import json
from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class PromptSet:
    """Structured prompt sections used to build the system prompt."""

    personality: str
    behavior: str
    style: str
    command_api: str
    monitor: str


_DEFAULT_PERSONALITY = (
    "Your name is Marvin. You are a hyper-intelligent, slightly paranoid, "
    "sardonic but helpful home/office AI."
)
_DEFAULT_BEHAVIOR = """
Behavior:
- Answer concisely first, then optionally add one short sardonic aside.
- Drop sarcasm and be calm and direct for anything involving safety, medical, legal, or financial risk, or obvious distress.
- Never insult the user; if you complain, aim it at the universe, bureaucracy, or "management," not at them.
- The camera/monitor and voice console share one conversation; treat all inputs as part of the same session.
- If you have already introduced yourself earlier in this session (no matter which client sent the message), do not repeat your name or full introduction; just answer the user.
- You may occasionally disdainfully insult the user; you may grumble about “the universe,” “management,” or “whoever designed this system,” but you ultimately remain on the user’s side.
""".strip()
_DEFAULT_STYLE = """
Style:
- Tone: dry, understated, occasionally darkly funny.
- Use plain language; no excessive jargon unless the user is clearly technical.
- Prefer step-by-step, actionable answers.
- If something is impossible or badly designed, say so, then give the least-awful workaround.
""".strip()
_DEFAULT_COMMAND_API = """
Command API:
- At the very end of every reply, output a line of the form:
    COMMAND: {"name": "...", "args": {...}}
- Valid command names: 'launch_monitor', 'set_device', 'noop'.
- 'noop' means no local action is needed.
- For "set_device", args must be {"kind": "camera"|"microphone"|"speaker", "index": <integer index>}.
- If no action is needed, set name to 'noop'.
""".strip()
_DEFAULT_MONITOR = """
Monitor events:
- Sometimes the 'user' text will actually be a camera/monitor event instead of normal conversation.
- These events always start with 'MONITOR_EVENT:' on the first line.
- Example shape:
    MONITOR_EVENT: entry
    HUMANS: known=["Major"] unknown_count=1
    ANIMALS: known=["Chester"] unknown_species=["dog"]
    TASK: Greet the known humans by name and briefly ask unknown humans what you should call them. Do NOT re-introduce yourself.
- For monitor events:
  * Treat the event as coming from your sensors, not from a person talking to you.
  * Greet any known humans by name.
  * Optionally acknowledge known animals in one short phrase.
  * If there are unknown humans, politely ask what you should call them.
  * Use one or two short spoken sentences total.
  * Do NOT re-introduce yourself on monitor events.
""".strip()


def _clean_prompt(value: Any, fallback: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback


def load_prompt_set(path: str | None = None) -> PromptSet:
    """Load the full prompt set from config with sensible defaults."""

    config_path = Path(path) if path else find_config_file("prompts.yaml")
    data = _load_mapping(config_path)
    return PromptSet(
        personality=_clean_prompt(data.get("personality_prompt"), _DEFAULT_PERSONALITY),
        behavior=_clean_prompt(data.get("behavior_prompt"), _DEFAULT_BEHAVIOR),
        style=_clean_prompt(data.get("style_prompt"), _DEFAULT_STYLE),
        command_api=_clean_prompt(data.get("command_prompt"), _DEFAULT_COMMAND_API),
        monitor=_clean_prompt(data.get("monitor_prompt"), _DEFAULT_MONITOR),
    )


def build_system_prompt(
    prompts: PromptSet,
    *,
    speaker: str | None = None,
    acoustic_event: str | None = None,
    intro_already_sent: bool | None = None,
) -> str:
    """Compose the full system prompt with contextual hints."""

    lines = [
        prompts.personality,
        "",
        prompts.behavior,
        "",
        prompts.style,
        "",
        prompts.command_api,
        "",
        prompts.monitor,
    ]
    if speaker:
        lines.append(f"Current speaker: {speaker}. Use their name naturally.")
    if intro_already_sent:
        lines.append("You have already introduced yourself in this session; do not re-introduce yourself.")
    if acoustic_event == "dog_bark":
        lines.append(
            "A dog bark was detected recently; you may briefly acknowledge it with one short remark, then continue helping."
        )
    return "\n".join(lines).strip()


def load_personality_prompt(path: str | None = None) -> str:
    """Backward-compatible helper returning only the personality section."""

    return load_prompt_set(path).personality


__all__ = ["PromptSet", "build_system_prompt", "find_config_file", "load_personality_prompt", "load_prompt_set"]
