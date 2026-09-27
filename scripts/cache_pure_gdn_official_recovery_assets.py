#!/usr/bin/env python3
"""Materialize exact offline assets for the disclosed held-out recovery."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

NEWS_ROWS = {"eng": 948, "xho": 297}
NER_ROWS = {"tn": 996, "xh": 1000, "zu": 1000}
NER_OFFLINE_CONFIGS = {
    "tn": "default-21d7aeb4ecebf694",
    "xh": "default-9278b546d4159f3d",
    "zu": "default-c511ba6223ac8e2c",
}
POS_ROWS = {"tsn": 602, "xho": 601, "zul": 601}
POS_REVISION = "376f4161f0425584d4bd7664122b56fa026926d3"


def _fingerprint(dataset: Any) -> str:
    return str(getattr(dataset, "_fingerprint", ""))


def _count_pos_sentences(lines: list[str]) -> int:
    count = 0
    has_tokens = False
    for raw in lines:
        line = str(raw).strip()
        if not line:
            count += has_tokens
            has_tokens = False
        elif not line.startswith("-DOCSTART-") and len(line.split()) >= 2:
            has_tokens = True
    return count + has_tokens


def _alias_dataset_config(dataset: Any, offline_config: str) -> None:
    cache_files = getattr(dataset, "cache_files", [])
    if not cache_files:
        raise ValueError("dataset has no materialized cache files")
    online_config = Path(cache_files[0]["filename"]).parents[2]
    offline_path = online_config.parent / offline_config
    if offline_path.exists():
        raise FileExistsError(offline_path)
    shutil.copytree(online_config, offline_path)


def materialize(
    cache_root: Path,
    news_cache: Path,
    metric_cache: Path,
    load_dataset: Callable[..., Any],
    load_metric: Callable[..., Any],
) -> dict[str, object]:
    hf_root = cache_root / "hf"
    shutil.copytree(news_cache / "hf", hf_root)
    for directory in (hf_root, *(path for path in hf_root.rglob("*") if path.is_dir())):
        directory.chmod(0o755)
    shutil.copytree(
        metric_cache / "hf" / "modules" / "evaluate_modules",
        hf_root / "modules" / "evaluate_modules",
        dirs_exist_ok=True,
    )

    result: dict[str, object] = {
        "schema": "sallm_pure_gdn_official_recovery_cache/v1",
        "metrics_computed": False,
        "datasets": {},
        "metric_modules": ["bleu", "chrf", "f1"],
    }
    datasets = result["datasets"]
    assert isinstance(datasets, dict)

    dataset_cache = hf_root / "datasets"
    for language, expected in NEWS_ROWS.items():
        dataset = load_dataset(
            "masakhane/masakhanews",
            language,
            split="test",
            cache_dir=str(dataset_cache),
        )
        if len(dataset) != expected:
            raise ValueError(f"expected {expected} News {language} rows")
        datasets[f"news/{language}"] = {
            "rows": len(dataset),
            "fingerprint": _fingerprint(dataset),
        }

    for language, expected in NER_ROWS.items():
        data_files = {
            split: f"data/{language}/{split}.parquet"
            for split in ("train", "validation", "test")
        }
        dataset = load_dataset(
            "anrilombard/masakhaner-x-parquet",
            data_files=data_files,
            split="test",
            cache_dir=str(dataset_cache),
        )
        if len(dataset) != expected:
            raise ValueError(f"expected {expected} NER {language} rows")
        _alias_dataset_config(dataset, NER_OFFLINE_CONFIGS[language])
        datasets[f"ner/{language}"] = {
            "rows": len(dataset),
            "fingerprint": _fingerprint(dataset),
            "offline_config": NER_OFFLINE_CONFIGS[language],
        }

    for language, expected in POS_ROWS.items():
        pinned_url = (
            "https://github.com/masakhane-io/masakhane-pos/raw/"
            f"{POS_REVISION}/data/{language}"
        )
        pinned_files = {
            "train": f"{pinned_url}/train.txt",
            "test": f"{pinned_url}/test.txt",
        }
        pinned = load_dataset(
            "text",
            data_files=pinned_files,
            split="test",
            cache_dir=str(dataset_cache),
        )
        task_url = (
            "https://github.com/masakhane-io/masakhane-pos/raw/"
            f"main/data/{language}"
        )
        dataset = load_dataset(
            "text",
            data_files={
                "train": f"{task_url}/train.txt",
                "test": f"{task_url}/test.txt",
            },
            split="test",
            cache_dir=str(dataset_cache),
        )
        if dataset["text"] != pinned["text"]:
            raise ValueError(f"POS {language} main differs from {POS_REVISION}")
        sentence_rows = _count_pos_sentences(dataset["text"])
        if sentence_rows != expected:
            raise ValueError(f"expected {expected} POS {language} rows")
        datasets[f"pos/{language}"] = {
            "rows": sentence_rows,
            "raw_rows": len(dataset),
            "fingerprint": _fingerprint(dataset),
            "pinned_fingerprint": _fingerprint(pinned),
            "main_matches_revision": POS_REVISION,
        }

    for metric in ("bleu", "chrf", "f1"):
        load_metric(metric, cache_dir=str(hf_root / "evaluate"))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--news-cache", type=Path, required=True)
    parser.add_argument("--metric-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cache_root = args.cache_root.resolve()
    if cache_root.exists():
        raise FileExistsError(cache_root)
    os.environ["HF_HOME"] = str(cache_root / "hf")
    os.environ["HF_DATASETS_CACHE"] = str(cache_root / "hf" / "datasets")
    os.environ["HF_HUB_CACHE"] = str(cache_root / "hf" / "hub")
    os.environ["HF_MODULES_CACHE"] = str(cache_root / "hf" / "modules")
    os.environ["HF_EVALUATE_CACHE"] = str(cache_root / "hf" / "evaluate")

    from datasets import load_dataset
    from evaluate import load as load_metric

    result = materialize(
        cache_root,
        args.news_cache.resolve(),
        args.metric_cache.resolve(),
        load_dataset,
        load_metric,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("Materialized recovery cache without model inference or metrics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
