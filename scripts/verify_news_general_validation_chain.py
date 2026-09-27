#!/usr/bin/env python3
"""Independently verify the four-model News validation prompt freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from itertools import permutations
from pathlib import Path
from typing import Any

from sklearn.metrics import accuracy_score, f1_score

ARCHITECTURES = ("mzansilm", "mamba", "xlstm", "gdn")
LANGUAGE_ROWS = {"eng": 472, "xho": 147}
PROMPTS = tuple(f"p{index}" for index in range(1, 6))
LABELS = (
    "business",
    "entertainment",
    "health",
    "politics",
    "religion",
    "sports",
    "technology",
)
FROZEN_TIE_ORDER = LABELS
REVERSE_TIE_ORDER = tuple(reversed(FROZEN_TIE_ORDER))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(gold: list[str], predicted: list[str]) -> dict[str, float]:
    observed = list(dict.fromkeys(gold))
    return {
        "accuracy": float(accuracy_score(gold, predicted)),
        "weighted_f1": float(
            f1_score(gold, predicted, average="weighted", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(
                gold,
                predicted,
                labels=observed,
                average="macro",
                zero_division=0,
            )
        ),
    }


def tied_winner(scores: dict[str, float], order: tuple[str, ...]) -> str:
    best = max(float(value) for value in scores.values())
    tied = {
        label for label, value in scores.items() if abs(float(value) - best) <= 1e-12
    }
    return next(label for label in order if label in tied)


def manual_weighted_f1(gold: list[str], predicted: list[str]) -> float:
    supports = Counter(gold)
    predicted_counts = Counter(predicted)
    true_positives = Counter(
        actual for actual, guess in zip(gold, predicted, strict=True) if actual == guess
    )
    total = len(gold)
    weighted = 0.0
    for label, support in supports.items():
        true_positive = true_positives[label]
        false_positive = predicted_counts[label] - true_positive
        false_negative = support - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        label_f1 = 2 * true_positive / denominator if denominator else 0.0
        weighted += support * label_f1
    return weighted / total


def all_common_order_range(rows: list[dict[str, Any]]) -> dict[str, float]:
    gold = [row["gold"] for row in rows]
    values = []
    for order in permutations(LABELS):
        predicted = [
            tied_winner(row["first_token_log_probabilities"], order) for row in rows
        ]
        values.append(manual_weighted_f1(gold, predicted))
    return {
        "minimum_weighted_f1": min(values),
        "maximum_weighted_f1": max(values),
        "maximum_possible_swing": max(values) - min(values),
    }


def verify_architecture(
    *, architecture: str, root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary_path = root / "general_summary.json"
    predictions_path = root / "general_predictions.jsonl"
    manifest_path = root / "manifest.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    predictions = [
        json.loads(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
    ]
    require(manifest.get("data_boundary") == "validation_only", "boundary drift")
    require(manifest.get("test_accessed") is False, "validation touched test")
    selected_architecture = manifest.get("selected_architecture")
    if architecture == "mamba":
        require(
            selected_architecture in (None, "mamba"),
            "Mamba architecture binding drift",
        )
    else:
        require(selected_architecture == architecture, "architecture drift")
    require(manifest.get("limit_per_language_per_prompt") == 0, "not full validation")
    require(set(manifest.get("gates", {}).values()) >= {"passed"}, "missing gate")
    required_gates = {
        "training_token_reconstruction",
        "unique_first_label_tokens",
        "completeness",
        "numerical_finiteness",
        "metric_recomputation",
        "adapter_changes_base_scores",
        "central_first_token_argmax_repeat_stability",
    }
    require(
        all(manifest["gates"].get(gate) == "passed" for gate in required_gates),
        "validation integrity gate failed",
    )
    token_audit = manifest["token_audit"]
    require(
        token_audit["records"] == token_audit["expected_records"] == 3095,
        "token audit incomplete",
    )
    require(
        token_audit["training_token_reconstruction"] == "passed",
        "token reconstruction failed",
    )
    require(
        token_audit["unique_first_label_tokens"] == "passed",
        "label-token uniqueness failed",
    )
    require(summary.get("arm") == "general", "regime drift")
    require(
        summary.get("records") == summary.get("expected_records") == 3095,
        "summary incomplete",
    )
    require(len(predictions) == 3095, "prediction file incomplete")
    require(
        summary["predictions_sha256"] == sha256(predictions_path),
        "prediction hash mismatch",
    )
    arm_report = manifest["arms"][0]
    require(
        arm_report["summary_sha256"] == sha256(summary_path), "summary hash mismatch"
    )
    require(
        arm_report["predictions_sha256"] == sha256(predictions_path),
        "manifest prediction hash mismatch",
    )
    require(summary["maximum_singleton_repeat_score_delta"] == 0, "repeat score drift")

    keys = [
        (row["language"], row["prompt_id"], row["selected_source_index"])
        for row in predictions
    ]
    require(len(keys) == len(set(keys)), "duplicate validation prediction")
    expected = {
        (language, prompt, source_index)
        for language, count in LANGUAGE_ROWS.items()
        for prompt in PROMPTS
        for source_index in range(count)
    }
    require(set(keys) == expected, "validation row coverage mismatch")
    tie_count = 0
    ties_by_subset: dict[str, int] = {}
    recomputed: dict[str, dict[str, float]] = {}
    reverse_metrics: dict[str, dict[str, float]] = {}
    all_order_ranges: dict[str, dict[str, float]] = {}
    for language in LANGUAGE_ROWS:
        for prompt in PROMPTS:
            rows = [
                row
                for row in predictions
                if row["language"] == language and row["prompt_id"] == prompt
            ]
            require(len(rows) == LANGUAGE_ROWS[language], "subset row-count mismatch")
            require(
                all(
                    row["first_token_prediction"]
                    == row["first_token_repeat_prediction"]
                    for row in rows
                ),
                "central argmax repeat instability",
            )
            subset_ties = 0
            for row in rows:
                scores = row["first_token_log_probabilities"]
                require(
                    all(math.isfinite(float(value)) for value in scores.values()),
                    "non-finite central score",
                )
                require(
                    row["first_token_prediction"]
                    == tied_winner(scores, FROZEN_TIE_ORDER),
                    "central argmax drift",
                )
                ranked = sorted(
                    (float(value) for value in scores.values()), reverse=True
                )
                is_tie = int(abs(ranked[0] - ranked[1]) <= 1e-12)
                subset_ties += is_tie
                tie_count += is_tie
            result = metrics(
                [row["gold"] for row in rows],
                [row["first_token_prediction"] for row in rows],
            )
            recorded = summary["subsets"][f"{language}/{prompt}"]["first_token"]
            for metric, value in result.items():
                require(
                    math.isclose(
                        value,
                        float(recorded[metric]),
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    ),
                    f"metric mismatch: {language}/{prompt}/{metric}",
                )
            recomputed[f"{language}/{prompt}"] = result
            reverse_metrics[f"{language}/{prompt}"] = metrics(
                [row["gold"] for row in rows],
                [
                    tied_winner(
                        row["first_token_log_probabilities"],
                        REVERSE_TIE_ORDER,
                    )
                    for row in rows
                ],
            )
            all_order_ranges[f"{language}/{prompt}"] = all_common_order_range(rows)
            ties_by_subset[f"{language}/{prompt}"] = subset_ties
    report = {
        "manifest_sha256": sha256(manifest_path),
        "summary_sha256": sha256(summary_path),
        "predictions_sha256": sha256(predictions_path),
        "binding_sha256": manifest["arm_binding_sha256"],
        "adapter_revision": manifest["selected_adapter_revision"],
        "base_revision": manifest["base_revision"],
        "records": len(predictions),
        "central_top_score_ties": tie_count,
        "central_top_score_ties_by_subset": ties_by_subset,
        "full_first_disagreement_count": summary["full_first_disagreement_count"],
        "batched_first_singleton_disagreement_count": summary[
            "batched_first_singleton_disagreement_count"
        ],
        "metrics": recomputed,
        "reverse_tie_order_metrics": reverse_metrics,
        "all_common_label_order_ranges": all_order_ranges,
    }
    return report, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mamba-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "verification output already exists")

    architecture_roots = {
        "mzansilm": args.root / "mzansilm_full",
        "mamba": args.mamba_root,
        "xlstm": args.root / "xlstm_full",
        "gdn": args.root / "gdn_full",
    }
    reports = {}
    summaries = {}
    for architecture in ARCHITECTURES:
        reports[architecture], summaries[architecture] = verify_architecture(
            architecture=architecture,
            root=architecture_roots[architecture],
        )

    selection_path = args.root / "shared_prompt_selection.json"
    verified_path = args.root / "VERIFIED.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    verified = json.loads(verified_path.read_text(encoding="utf-8"))
    require(verified.get("status") == "VALIDATION_CHAIN_VERIFIED", "chain failed")
    require(verified.get("data_boundary") == "validation_only", "chain boundary drift")
    require(verified.get("test_accessed") is False, "chain touched test")
    require(
        verified["shared_prompt_selection_sha256"] == sha256(selection_path),
        "chain selection hash mismatch",
    )
    require(
        selection["input_summary_sha256"]
        == {
            architecture: reports[architecture]["summary_sha256"]
            for architecture in ARCHITECTURES
        },
        "selection input hash mismatch",
    )

    table: dict[str, dict[str, dict[str, float]]] = {}
    selections = {}
    reverse_selections = {}
    rankings = {}
    for language in LANGUAGE_ROWS:
        table[language] = {}
        for architecture in ARCHITECTURES:
            values = {
                prompt: reports[architecture]["metrics"][f"{language}/{prompt}"][
                    "weighted_f1"
                ]
                for prompt in PROMPTS
            }
            table[language][architecture] = {
                **values,
                "mean": sum(values.values()) / len(values),
            }
        means = {
            prompt: sum(
                table[language][architecture][prompt] for architecture in ARCHITECTURES
            )
            / len(ARCHITECTURES)
            for prompt in PROMPTS
        }
        winner = max(PROMPTS, key=lambda prompt: (means[prompt], -int(prompt[1:])))
        require(
            selection["selections"][language]["prompt"] == winner,
            "selected prompt differs from independent recomputation",
        )
        for prompt, value in means.items():
            require(
                math.isclose(
                    value,
                    selection["candidates"][language][prompt]["mean_weighted_f1"],
                    rel_tol=0.0,
                    abs_tol=1e-15,
                ),
                "selection mean differs",
            )
        selections[language] = {
            "prompt": winner,
            "mean_weighted_f1": means[winner],
            "candidate_means": means,
        }
        reverse_means = {
            prompt: sum(
                reports[architecture]["reverse_tie_order_metrics"][
                    f"{language}/{prompt}"
                ]["weighted_f1"]
                for architecture in ARCHITECTURES
            )
            / len(ARCHITECTURES)
            for prompt in PROMPTS
        }
        reverse_winner = max(
            PROMPTS,
            key=lambda prompt: (reverse_means[prompt], -int(prompt[1:])),
        )
        require(reverse_winner == winner, "reverse tie order changes shared prompt")
        reverse_selections[language] = {
            "prompt": reverse_winner,
            "mean_weighted_f1": reverse_means[reverse_winner],
            "candidate_means": reverse_means,
        }
        frozen_ranking = sorted(
            ARCHITECTURES,
            key=lambda architecture: (
                -table[language][architecture][winner],
                ARCHITECTURES.index(architecture),
            ),
        )
        reverse_ranking = sorted(
            ARCHITECTURES,
            key=lambda architecture: (
                -reports[architecture]["reverse_tie_order_metrics"][
                    f"{language}/{winner}"
                ]["weighted_f1"],
                ARCHITECTURES.index(architecture),
            ),
        )
        require(reverse_ranking == frozen_ranking, "reverse tie order changes ranking")
        rankings[language] = {
            "frozen_order": frozen_ranking,
            "reverse_order": reverse_ranking,
        }

    maximum_possible_swings = {
        architecture: {
            "maximum_across_language_prompt_subsets": max(
                item["maximum_possible_swing"]
                for item in reports[architecture][
                    "all_common_label_order_ranges"
                ].values()
            ),
            "selected_prompt_by_language": {
                language: reports[architecture]["all_common_label_order_ranges"][
                    f"{language}/{selections[language]['prompt']}"
                ]
                for language in LANGUAGE_ROWS
            },
        }
        for architecture in ARCHITECTURES
    }

    output = {
        "schema": "sallm.news_general_validation_chain_independent_audit/v1",
        "status": "VALIDATION_CHAIN_INDEPENDENTLY_VERIFIED",
        "data_boundary": "validation_only",
        "test_accessed": False,
        "chain_verification_sha256": sha256(verified_path),
        "shared_prompt_selection_sha256": sha256(selection_path),
        "reports": reports,
        "selections": selections,
        "reverse_tie_order_selections": reverse_selections,
        "architecture_rankings": rankings,
        "tie_policy": {
            "frozen_order": list(FROZEN_TIE_ORDER),
            "rule": "first label in frozen order among exact top-score ties",
            "reverse_sensitivity_order": list(REVERSE_TIE_ORDER),
            "shared_prompt_changes_under_reverse": False,
            "architecture_ranking_changes_under_reverse": False,
            "maximum_possible_weighted_f1_swing_over_all_5040_common_orders": (
                maximum_possible_swings
            ),
        },
        "validation_weighted_f1": table,
        "gates": {
            "all_four_architectures": "passed",
            "all_12380_predictions": "passed",
            "row_key_uniqueness_and_completeness": "passed",
            "training_token_reconstruction": "passed",
            "unique_first_label_tokens": "passed",
            "finite_central_scores": "passed",
            "central_top_score_ties": "reported_with_frozen_label_order_tie_break",
            "central_argmax_repeat_stability": "passed",
            "independent_sklearn_metric_recomputation": "passed",
            "shared_prompt_mean_and_tie_break_recomputation": "passed",
            "binding_and_artifact_hashes": "passed",
        },
    }
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": output["status"],
                "output": str(args.output),
                "sha256": sha256(args.output),
                "selections": selections,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
