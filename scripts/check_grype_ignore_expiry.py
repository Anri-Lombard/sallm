#!/usr/bin/env python3
"""Fail when a Grype ignore is unclassified or an accepted-risk waiver expires."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import Any

import yaml


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--today", type=date.fromisoformat, default=date.today())
    return parser.parse_args()


def _label(entry: dict[str, Any], index: int) -> str:
    vulnerability = entry.get("vulnerability", f"ignore[{index}]")
    package = entry.get("package")
    package_name = package.get("name") if isinstance(package, dict) else None
    return f"{vulnerability} for {package_name or 'unknown package'}"


def main() -> int:
    args = _parse_args()
    document = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    ignores = document.get("ignore", [])
    if not isinstance(ignores, list):
        print("Grype ignore policy: 'ignore' must be a list", file=sys.stderr)
        return 1

    errors: list[str] = []
    for index, entry in enumerate(ignores):
        if not isinstance(entry, dict):
            errors.append(f"ignore[{index}] must be a mapping")
            continue

        label = _label(entry, index)
        reason = entry.get("reason")
        if reason == "false-positive":
            continue
        if reason != "accepted-risk":
            errors.append(
                f"{label} needs reason: false-positive or reason: accepted-risk"
            )
            continue

        raw_expiry = entry.get("expires")
        if not isinstance(raw_expiry, str):
            errors.append(f"{label} needs an ISO expires date")
            continue
        try:
            expiry = date.fromisoformat(raw_expiry)
        except ValueError:
            errors.append(f"{label} has invalid expires date {raw_expiry!r}")
            continue
        if args.today >= expiry:
            errors.append(f"{label} expired on {expiry.isoformat()}")

    if errors:
        for error in errors:
            print(f"Grype ignore policy: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
