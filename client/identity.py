from __future__ import annotations

"""Local identity helpers for CLI utilities.

This module provides a thin wrapper around the shared agent_core voice registry
so that scripts under bin/ (dump_voice_profiles.py, remove_voice_from_registry.py,
unenroll_profiles.py) can work with a simple import:

    from client import identity

Right now we implement *voice* profile helpers backed by the unified
AgentStateTable (DynamoDB). Face profile helpers are backed by an AWS
Rekognition collection (for now).
"""

import os
from typing import List, Optional, Tuple

import boto3

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


def resolve_voice(
    embedding: Optional[List[float]],
    voice_id: Optional[str] = None,
    claimed_name: Optional[str] = None,
    similarity_threshold: float = 0.8,
) -> Tuple[Optional[str], bool]:
    """Resolve a speaker's voice to a stored name via the shared registry.

    Thin wrapper over `voice_registry.resolve_voice` that automatically scopes
    lookups to the current AGENT_ID.

    Returns (name, is_new) where:

      - name is the resolved or newly-registered name, or None if still unknown
      - is_new is True if the voice was not previously known to the registry
        (i.e., caller may wish to ask the user for their name when name is None).
    """
    return voice_registry.resolve_voice(
        _agent_id(),
        voice_id=voice_id,
        claimed_name=claimed_name,
        embedding=embedding,
        similarity_threshold=similarity_threshold,
    )


# ---------------------------------------------------------------------------
# Face profile helpers (Rekognition-backed)
# ---------------------------------------------------------------------------


def _aws_region() -> str:
    """Resolve AWS region for Rekognition calls.

    We look at AWS_REGION, AWS_DEFAULT_REGION, REGION in that order, falling
    back to us-east-1 if none are set.
    """
    return (
        os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
        or os.getenv("REGION")
        or "us-east-1"
    )


def _rekognition_client():
    return boto3.client("rekognition", region_name=_aws_region())


def _rekognition_collection() -> str:
    return os.getenv("REKOGNITION_COLLECTION", "companion-people")


def list_face_names() -> List[str]:
    """Return a sorted list of known face profile names.

    This queries the configured Rekognition collection and returns the set of
    ExternalImageId values (one per logical person).
    """
    client = _rekognition_client()
    collection_id = _rekognition_collection()

    names: set[str] = set()
    token: Optional[str] = None

    while True:
        if token:
            resp = client.list_faces(CollectionId=collection_id, NextToken=token)
        else:
            resp = client.list_faces(CollectionId=collection_id)

        for face in resp.get("Faces", []):
            ext = (face.get("ExternalImageId") or "").strip()
            if ext:
                names.add(ext)

        token = resp.get("NextToken")
        if not token:
            break

    return sorted(names)


def delete_face(name: str) -> bool:
    """Delete Rekognition faces for the given logical name.

    We treat the ExternalImageId in the Rekognition collection as the logical
    "face profile name". All faces with a matching ExternalImageId are deleted.

    Returns True if at least one Rekognition face was removed, False otherwise.
    """
    client = _rekognition_client()
    collection_id = _rekognition_collection()

    name = (name or "").strip()
    if not name:
        return False

    face_ids: List[str] = []
    token: Optional[str] = None

    while True:
        if token:
            resp = client.list_faces(CollectionId=collection_id, NextToken=token)
        else:
            resp = client.list_faces(CollectionId=collection_id)

        for face in resp.get("Faces", []):
            ext = (face.get("ExternalImageId") or "").strip()
            if ext == name:
                fid = face.get("FaceId")
                if fid:
                    face_ids.append(fid)

        token = resp.get("NextToken")
        if not token:
            break

    if not face_ids:
        return False

    client.delete_faces(CollectionId=collection_id, FaceIds=face_ids)
    return True
