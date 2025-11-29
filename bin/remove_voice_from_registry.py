#!/usr/bin/env python3
"""Remove a voice profile from the identity registry by name."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from client import identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--name", required=True, help="Voice profile name to remove (case-insensitive)"
    )
    args = parser.parse_args()

    name = args.name.strip()
    if not name:
        parser.error("Name cannot be empty")

    if identity.delete_voice(name):
        print(f"Removed voice profile for '{name}' from registry.")
    else:
        print(f"No voice profile found for '{name}' in registry.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
