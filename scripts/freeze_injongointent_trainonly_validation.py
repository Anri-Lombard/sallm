#!/usr/bin/env python3
"""Materialize and freeze an InjongoIntent validation split from train only."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import urlopen

from injongointent_trainonly_validation import (
    DATASET_NAME,
    DATASET_REVISION,
    MANIFEST_SCHEMA,
    VALIDATION_RATIO,
    derive_language_validation,
)

BASE_URL = (
    "https://huggingface.co/datasets/masakhane/InjongoIntent/resolve/"
    f"{DATASET_REVISION}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--languages", nargs="+", default=["eng", "sot", "xho", "zul"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    if args.output.exists() or temp.exists() or args.source_dir.exists():
        raise FileExistsError("Refusing to overwrite frozen validation artifacts")
    if len(set(args.languages)) != len(args.languages):
        raise ValueError("Duplicate language requested")

    payload = {
        "schema": MANIFEST_SCHEMA,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset": DATASET_NAME,
        "revision": DATASET_REVISION,
        "source_split": "train",
        "held_out_data_accessed": False,
        "architecture_blind": True,
        "model_outputs_accessed": False,
        "algorithm": {
            "within_train_deduplication": (
                "keep lowest source index per normalized (intent, text); fail on "
                "one normalized text carrying multiple intents"
            ),
            "validation_split": (
                "stratified by intent; order each label group by deterministic MD5 "
                "of its existing InjongoIntent row key; select round(10%) with at "
                "least one and leave at least one training row"
            ),
            "validation_ratio": VALIDATION_RATIO,
            "source_order_preserved": True,
        },
        "languages": {},
    }
    args.source_dir.mkdir(parents=True, exist_ok=False)
    try:
        for language in args.languages:
            url = f"{BASE_URL}/{language}/train.jsonl"
            with urlopen(url, timeout=60) as response:
                content = response.read()
            language_dir = args.source_dir / language
            language_dir.mkdir(parents=True, exist_ok=False)
            source_path = language_dir / "train.jsonl"
            source_path.write_bytes(content)
            _, evidence = derive_language_validation(language, content)
            evidence["source_path"] = str(source_path.relative_to(args.output.parent))
            payload["languages"][language] = evidence
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temp.replace(args.output)
    except Exception:
        if temp.exists():
            temp.unlink()
        raise
    print(json.dumps({
        "manifest": str(args.output),
        "row_counts": {
            language: values["validation_row_count"]
            for language, values in payload["languages"].items()
        },
        "held_out_data_accessed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
