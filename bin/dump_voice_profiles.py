#!/usr/bin/env python3
"""Print all voice profile names from the identity registry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List

sys.path.append(str(Path(__file__).resolve().parents[1]))
from client import identity


def _collect_voice_names() -> List[str]:
    entries = identity.list_entries()
    names = {
        (entry.get("name") or "").strip()
        for entry in entries
        if entry.get("type") == "person" and entry.get("voice")
    }
    return sorted(n for n in names if n)


def _as_json(names: Iterable[str]) -> str:
    return json.dumps({"voice_profiles": list(names)}, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json", action=argparse.BooleanOptionalAction, help="Output JSON instead of plain text"
    )
    args = parser.parse_args()

    names = _collect_voice_names()
    if args.json:
        print(_as_json(names))
    else:
        for name in names:
            print(name)


if __name__ == "__main__":
    main()
