#!/usr/bin/env python3
from __future__ import annotations
import sys


def main() -> int:
    print(
        "[monitor] Standalone monitor process deprecated — launch via the main CLI to run in-process.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
