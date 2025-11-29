#!/usr/bin/env python3
from __future__ import annotations
import sys


def main() -> int:
    print(
        "[monitor] Standalone monitor entry removed — start the voice client to run the monitor automatically.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
