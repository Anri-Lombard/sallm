#!/usr/bin/env python3
"""Materialize the frozen SIB test inputs without running model inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

CONFIGS = ("afr_Latn", "eng_Latn", "nso_Latn", "sot_Latn", "xho_Latn", "zul_Latn")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cache_dir = args.cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=False)
    result: dict[str, object] = {
        "schema": "sallm_sib_official_dataset_cache/v1",
        "dataset": "Davlan/sib200",
        "split": "test",
        "rows_per_config": 204,
        "configs": {},
        "metrics_computed": False,
    }
    configs = result["configs"]
    assert isinstance(configs, dict)
    for config in CONFIGS:
        dataset = load_dataset(
            "Davlan/sib200",
            config,
            split="test",
            cache_dir=str(cache_dir),
        )
        if len(dataset) != 204:
            raise ValueError(f"expected 204 {config} rows, found {len(dataset)}")
        configs[config] = {"rows": len(dataset), "fingerprint": dataset._fingerprint}

    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Cached {len(CONFIGS)} SIB test configurations without model inference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
