#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable
from pathlib import Path

DEFAULT_PATTERN = r"\b(nothing|empty|emptied|none|nobody|no one|without)\b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count empty-state language in BabyLM training text."
    )
    parser.add_argument(
        "--dataset-name", default="BabyLM-community/BabyLM-2026-Strict-Small"
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--pattern", default=DEFAULT_PATTERN)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/scratch/alombard/babylm-gdn/diagnostics/empty_state_corpus_audit.json"
        ),
    )
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+", text))


def summarize_texts(texts: Iterable[str], pattern: str, sample_limit: int = 20) -> dict:
    regex = re.compile(pattern, re.IGNORECASE)
    totals = {
        "documents": 0,
        "words": 0,
        "matched_documents": 0,
        "matched_words": 0,
        "matches": 0,
        "examples": [],
    }
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            continue
        wc = word_count(text)
        hits = regex.findall(text)
        totals["documents"] += 1
        totals["words"] += wc
        if hits:
            totals["matched_documents"] += 1
            totals["matched_words"] += wc
            totals["matches"] += len(hits)
            if len(totals["examples"]) < sample_limit:
                totals["examples"].append(text.strip().replace("\n", " ")[:300])
    totals["matched_document_rate"] = (
        totals["matched_documents"] / totals["documents"]
        if totals["documents"]
        else 0.0
    )
    totals["matched_word_rate"] = (
        totals["matched_words"] / totals["words"] if totals["words"] else 0.0
    )
    totals["matches_per_million_words"] = (
        totals["matches"] * 1_000_000 / totals["words"] if totals["words"] else 0.0
    )
    return totals


def self_check() -> None:
    result = summarize_texts(
        ["The box is empty.", "There is nothing here.", "The cup is full."],
        DEFAULT_PATTERN,
    )
    assert result["documents"] == 3
    assert result["matched_documents"] == 2
    assert result["matches"] == 2


def main() -> None:
    args = parse_args()
    if args.self_check:
        self_check()
        print("SELF_CHECK_OK")
        return

    from datasets import load_dataset

    split = (
        args.split if args.max_samples <= 0 else f"{args.split}[:{args.max_samples}]"
    )
    dataset = load_dataset(args.dataset_name, split=split)
    result = summarize_texts(dataset["text"], args.pattern)
    result.update(
        {"dataset_name": args.dataset_name, "split": split, "pattern": args.pattern}
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "examples"}, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
