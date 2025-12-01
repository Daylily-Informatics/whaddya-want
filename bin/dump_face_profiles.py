#!/usr/bin/env python3
"""Print all face profile names from the identity registry / Rekognition."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List

sys.path.append(str(Path(__file__).resolve().parents[1]))
from client import identity


def _collect_face_names() -> List[str]:
    """Return a sorted list of face profile names via the identity helper."""
    return identity.list_face_names()


def _as_json(names: Iterable[str]) -> str:
    return json.dumps({"face_profiles": list(names)}, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json", action=argparse.BooleanOptionalAction, help="Output JSON instead of plain text"
    )
    args = parser.parse_args()

    names = _collect_face_names()
    if args.json:
        print(_as_json(names))
    else:
        for name in names:
            print(name)


if __name__ == "__main__":
    main()
