#!/usr/bin/env python3
"""Materialize frozen News or Intent test inputs without model inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

FAMILIES = {
    "news": ("masakhane/masakhanews", {"eng": 948, "xho": 297}),
    "intent": (
        "masakhane/InjongoIntent",
        {"eng": 622, "sot": 640, "xho": 640, "zul": 640},
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=FAMILIES)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cache_dir = args.cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=False)
    dataset_name, expected_rows = FAMILIES[args.family]
    result: dict[str, object] = {
        "schema": "sallm_news_intent_official_dataset_cache/v1",
        "family": args.family,
        "dataset": dataset_name,
        "split": "test",
        "configs": {},
        "metrics_computed": False,
    }
    configs = result["configs"]
    assert isinstance(configs, dict)
    for config, rows in expected_rows.items():
        dataset = load_dataset(
            dataset_name,
            config,
            split="test",
            cache_dir=str(cache_dir),
        )
        if len(dataset) != rows:
            raise ValueError(f"expected {rows} {config} rows, found {len(dataset)}")
        configs[config] = {"rows": len(dataset), "fingerprint": dataset._fingerprint}

    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Cached {len(expected_rows)} {args.family} test configurations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
