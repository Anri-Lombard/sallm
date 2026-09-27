#!/usr/bin/env python3
"""Select one validation prompt per News language across all four architectures."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ARCHITECTURES = ("mzansilm", "mamba", "xlstm", "gdn")
LANGUAGE_ROWS = {"eng": 472, "xho": 147}
PROMPTS = tuple(f"p{index}" for index in range(1, 6))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_summaries(specs: list[str]) -> tuple[dict[str, dict], dict[str, str]]:
    if len(specs) != len(ARCHITECTURES):
        raise ValueError("exactly four architecture=summary inputs are required")
    summaries: dict[str, dict] = {}
    hashes: dict[str, str] = {}
    for spec in specs:
        architecture, separator, raw_path = spec.partition("=")
        if not separator or architecture not in ARCHITECTURES:
            raise ValueError(f"invalid architecture=summary input: {spec}")
        if architecture in summaries:
            raise ValueError(f"duplicate architecture input: {architecture}")
        path = Path(raw_path).resolve()
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("arm") != "general":
            raise ValueError(f"{architecture} is not a General-arm summary")
        if report.get("records") != 3095 or report.get("expected_records") != 3095:
            raise ValueError(f"{architecture} summary is incomplete")
        expected_keys = {
            f"{language}/{prompt}" for language in LANGUAGE_ROWS for prompt in PROMPTS
        }
        if set(report.get("subsets", {})) != expected_keys:
            raise ValueError(f"{architecture} summary has unexpected subsets")
        for key, subset in report["subsets"].items():
            language = key.split("/", 1)[0]
            if subset.get("n") != LANGUAGE_ROWS[language]:
                raise ValueError(f"{architecture} {key} row count mismatch")
            metrics = subset.get("first_token", {})
            for metric in ("weighted_f1", "macro_f1", "accuracy"):
                value = metrics.get(metric)
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"{architecture} {key} invalid {metric}")
        summaries[architecture] = report
        hashes[architecture] = sha256(path)
    if set(summaries) != set(ARCHITECTURES):
        raise ValueError("all four architecture summaries are required")
    return summaries, hashes


def select(summaries: dict[str, dict], hashes: dict[str, str]) -> dict:
    candidates: dict[str, dict[str, dict]] = {}
    selections: dict[str, dict] = {}
    for language in LANGUAGE_ROWS:
        candidates[language] = {}
        for prompt in PROMPTS:
            per_architecture = {
                architecture: summaries[architecture]["subsets"][
                    f"{language}/{prompt}"
                ]["first_token"]
                for architecture in ARCHITECTURES
            }
            candidates[language][prompt] = {
                "mean_weighted_f1": sum(
                    metrics["weighted_f1"] for metrics in per_architecture.values()
                )
                / len(ARCHITECTURES),
                "per_architecture": per_architecture,
            }
        selected_prompt = max(
            PROMPTS,
            key=lambda prompt: (
                candidates[language][prompt]["mean_weighted_f1"],
                -int(prompt[1:]),
            ),
        )
        selections[language] = {
            "prompt": selected_prompt,
            **candidates[language][selected_prompt],
        }
    return {
        "schema": "sallm.news_general_shared_prompt_selection/v1",
        "data_boundary": "validation_only",
        "test_accessed": False,
        "selection_metric": (
            "arithmetic mean of validation support-weighted F1 across the four "
            "architectures"
        ),
        "tie_break": "lowest prompt number",
        "architectures": list(ARCHITECTURES),
        "input_summary_sha256": hashes,
        "candidates": candidates,
        "selections": selections,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    summaries, hashes = load_summaries(args.summary)
    output = select(summaries, hashes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"NEWS_GENERAL_SHARED_PROMPTS_SELECTED {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
