#!/usr/bin/env python3
"""Unenroll user voice and/or face profiles by name.

Usage examples:

    ./bin/unenroll_profiles.py --type face --name Major --purge-collection
    ./bin/unenroll_profiles.py --type voice --name "AI Voice"

By default the script updates the local identity registry (and any configured
remote mirrors). When --purge-collection is supplied for faces, it will also
remove matching ExternalImageId entries from a Rekognition collection.
"""

import argparse
import os
from typing import Iterable, List

import boto3

from client import identity


def _delete_from_registry(kind: str, name: str) -> bool:
    if kind == "face":
        return identity.delete_face(name)
    if kind == "voice":
        return identity.delete_voice(name)
    raise ValueError(f"Unsupported profile type: {kind}")


def _iter_faces(client, collection_id: str) -> Iterable[dict]:
    token = None
    while True:
        resp = (
            client.list_faces(CollectionId=collection_id, NextToken=token)
            if token
            else client.list_faces(CollectionId=collection_id)
        )
        for face in resp.get("Faces", []):
            yield face
        token = resp.get("NextToken")
        if not token:
            break


def _purge_rekognition_faces(name: str, collection_id: str, region: str) -> List[str]:
    client = boto3.client("rekognition", region_name=region)
    face_ids = [
        f["FaceId"]
        for f in _iter_faces(client, collection_id)
        if (f.get("ExternalImageId") or "").lower() == name.lower()
    ]
    if not face_ids:
        return []
    client.delete_faces(CollectionId=collection_id, FaceIds=face_ids)
    return face_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Unenroll voice/face profiles by name")
    parser.add_argument("--name", required=True, help="Profile name to remove (case-insensitive)")
    parser.add_argument(
        "--type",
        choices=["voice", "face", "both"],
        default="both",
        help="Which profile type(s) to remove",
    )
    parser.add_argument(
        "--purge-collection",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also delete matching faces from the Rekognition collection",
    )
    parser.add_argument(
        "--collection",
        default=os.getenv("REKOGNITION_COLLECTION", "companion-people"),
        help="Rekognition collection ID (for face removals)",
    )
    parser.add_argument(
        "--region",
        default=os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
        or os.getenv("REGION")
        or "us-east-1",
        help="AWS region for Rekognition calls",
    )

    args = parser.parse_args()
    name = args.name.strip()
    kinds = [args.type] if args.type in {"voice", "face"} else ["voice", "face"]

    changed = []
    for kind in kinds:
        if _delete_from_registry(kind, name):
            changed.append(kind)
            print(f"Removed {kind} profile for '{name}' from registry.")
        else:
            print(f"No {kind} profile found for '{name}' in registry.")

    if args.purge_collection and "face" in kinds:
        deleted_ids = _purge_rekognition_faces(name, args.collection, args.region)
        if deleted_ids:
            print(
                f"Deleted {len(deleted_ids)} Rekognition face(s) for '{name}' from collection "
                f"{args.collection}: {', '.join(deleted_ids)}"
            )
        else:
            print(
                f"No Rekognition faces with ExternalImageId '{name}' found in collection {args.collection}."
            )

    if not changed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
