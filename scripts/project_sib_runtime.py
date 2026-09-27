#!/usr/bin/env python3
import argparse
import json
import re
from datetime import datetime
from pathlib import Path

TIMESTAMP = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")


def timestamp(line: str) -> datetime | None:
    match = TIMESTAMP.match(line)
    return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S,%f") if match else None


def timing(path: Path) -> dict[str, float]:
    lines = path.read_text(encoding="utf-8").splitlines()
    elapsed = next(
        float(line.split("=", 1)[1].split()[0])
        for line in lines
        if line.startswith("ELAPSED=")
    )
    start = next(timestamp(line) for line in lines if "Running lm-eval" in line)
    end = next(
        timestamp(line)
        for line in reversed(lines)
        if "Saved lm-eval results" in line
    )
    assert start is not None and end is not None
    evaluation = (end - start).total_seconds()
    return {
        "elapsed_seconds": elapsed,
        "evaluation_seconds": evaluation,
        "load_seconds": max(0.0, elapsed - evaluation),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("logs", type=Path, nargs=4)
    args = parser.parse_args()
    rows = {path.stem: timing(path) for path in args.logs}
    for row in rows.values():
        row["projected_validation_and_test_seconds"] = (
            2 * row["load_seconds"] + row["evaluation_seconds"] * (4194 / 40)
        )
    projected = sum(
        row["projected_validation_and_test_seconds"] for row in rows.values()
    )
    payload = {
        "schema": "sallm.retained_sib_runtime_projection/v1",
        "canary_examples_per_model": 40,
        "full_validation_examples_per_model": 2970,
        "selected_test_examples_per_model": 1224,
        "models": rows,
        "projected_seconds": projected,
        "projected_seconds_with_20pct_margin": projected * 1.2,
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(int(projected * 1.2 + 0.999))


if __name__ == "__main__":
    main()
