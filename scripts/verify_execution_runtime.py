#!/usr/bin/env python3
"""Fail closed when the active Python runtime differs from a frozen manifest."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
from pathlib import Path


def package_versions() -> dict[str, str]:
    return dict(
        sorted(
            (
                distribution.metadata.get("Name", "unknown"),
                distribution.version,
            )
            for distribution in importlib.metadata.distributions()
        )
    )


def main() -> int:
    manifest = Path(sys.argv[1])
    expected = json.loads(manifest.read_text(encoding="utf-8"))["environment"]
    actual = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": package_versions(),
    }
    mismatches = [key for key, value in actual.items() if expected.get(key) != value]
    if mismatches:
        raise SystemExit(
            "Execution runtime verification failed: " + ", ".join(mismatches)
        )
    print(f"Verified exact execution runtime from {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
