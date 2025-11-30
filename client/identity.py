from __future__ import annotations

"""Local identity helpers for CLI utilities.

This module provides a thin wrapper around the shared agent_core voice registry
so that scripts under bin/ (dump_voice_profiles.py, remove_voice_from_registry.py,
unenroll_profiles.py) can work with a simple import:

    from client import identity

Right now we implement *voice* profile helpers backed by the unified
AgentStateTable (DynamoDB). Face profile helpers are stubbed and can be wired
to Rekognition or another store later.
"""

import os
from typing import List

from agent_core import voice_registry


def _agent_id() -> str:
    """Return the logical agent id to use for identity operations.

    Defaults to the AGENT_ID environment variable, falling back to "marvin".
    """
    return os.getenv("AGENT_ID", "marvin")


# ---------------------------------------------------------------------------
# Voice profile helpers
# ---------------------------------------------------------------------------


def list_voice_names() -> List[str]:
    """Return a sorted list of known voice profile names for this agent."""
    return voice_registry.list_voice_names(_agent_id())


def delete_voice(name: str) -> bool:
    """Delete one or more voice profiles for the given name.

    Returns True if at least one profile was removed, False otherwise.
    """
    return voice_registry.delete_voice_profile(_agent_id(), name)


# ---------------------------------------------------------------------------
# Face profile helpers (stubs for now)
# ---------------------------------------------------------------------------


def list_face_names() -> List[str]:
    """Return a list of known face profile names.

    This is currently a stub that returns an empty list. It exists solely so
    that bin/dump_face_profiles.py can import and call it without failing.
    """
    return []


def delete_face(name: str) -> bool:
    """Delete a face profile by name.

    This is currently a stub that always returns False. It exists solely so
    that bin/remove_face_from_registry.py can import and call it without
    failing. Face enrollment and deletion are handled directly via Rekognition
    for now (see bin/entroll_faces.py and bin/unenroll_profiles.py).
    """
    _ = name  # unused
    return False
