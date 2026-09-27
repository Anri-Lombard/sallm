#!/usr/bin/env python3
"""Audit and analyse the sealed four-model News official result artifacts.

This script deliberately reads prediction, summary, manifest, and protocol
artifacts only. It never opens the materialised source TSV files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from itertools import combinations
from pathlib import Path
from statistics import fmean
from typing import Any

ARCHITECTURES = ("mzansilm", "mamba", "xlstm", "gdn")
DISPLAY_NAMES = {
    "mzansilm": "MzansiLM",
    "mamba": "Mamba",
    "xlstm": "xLSTM",
    "gdn": "GDN",
}
LABELS = (
    "business",
    "entertainment",
    "health",
    "politics",
    "religion",
    "sports",
    "technology",
)
LANGUAGES = ("eng", "xho")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            records.append(value)
    return records


def require_sha(path: Path, expected: str, description: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(
            f"{description} SHA mismatch: expected {expected}, observed {observed}"
        )


def safe_div(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def class_metrics(
    records: list[dict[str, Any]], prediction_field: str
) -> dict[str, dict[str, float | int]]:
    output: dict[str, dict[str, float | int]] = {}
    for label in LABELS:
        true_positive = sum(
            row["gold"] == label and row[prediction_field] == label for row in records
        )
        false_positive = sum(
            row["gold"] != label and row[prediction_field] == label for row in records
        )
        false_negative = sum(
            row["gold"] == label and row[prediction_field] != label for row in records
        )
        support = true_positive + false_negative
        predicted = true_positive + false_positive
        precision = safe_div(true_positive, predicted)
        recall = safe_div(true_positive, support)
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        output[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "predicted": predicted,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
        }
    return output


def aggregate_metrics(
    records: list[dict[str, Any]], prediction_field: str
) -> dict[str, float | int]:
    per_class = class_metrics(records, prediction_field)
    total = len(records)
    correct = sum(row["gold"] == row[prediction_field] for row in records)
    supported = [metrics for metrics in per_class.values() if metrics["support"]]
    macro_f1 = fmean(float(metrics["f1"]) for metrics in supported)
    weighted_f1 = (
        sum(
            float(metrics["f1"]) * int(metrics["support"])
            for metrics in per_class.values()
        )
        / total
    )
    return {
        "n": total,
        "accuracy": correct / total,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def confusion_matrix(records: list[dict[str, Any]]) -> list[list[int]]:
    return [
        [
            sum(
                row["gold"] == gold and row["prediction"] == predicted
                for row in records
            )
            for predicted in LABELS
        ]
        for gold in LABELS
    ]


def top_confusions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    confusions = Counter(
        (row["gold"], row["prediction"])
        for row in records
        if row["gold"] != row["prediction"]
    )
    return [
        {"gold": gold, "prediction": prediction, "count": count}
        for (gold, prediction), count in sorted(
            confusions.items(), key=lambda item: (-item[1], item[0])
        )
    ]


def exact_mcnemar_p(first_only: int, second_only: int) -> float:
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(first_only, second_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def add_holm_adjustment(rows: list[dict[str, Any]]) -> None:
    ordered = sorted(enumerate(rows), key=lambda item: item[1]["mcnemar_exact_p"])
    running_max = 0.0
    count = len(rows)
    for rank, (original_index, row) in enumerate(ordered):
        adjusted = min(1.0, float(row["mcnemar_exact_p"]) * (count - rank))
        running_max = max(running_max, adjusted)
        rows[original_index]["mcnemar_holm_p_across_all_pairs_and_scopes"] = running_max


def quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot compute quantile of an empty list")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def margin_summary(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": fmean(values),
        "minimum": min(values),
        "p05": quantile(values, 0.05),
        "p25": quantile(values, 0.25),
        "median": quantile(values, 0.5),
        "p75": quantile(values, 0.75),
        "p95": quantile(values, 0.95),
        "maximum": max(values),
        "exact_zero_count": sum(value == 0.0 for value in values),
    }


def score_margin(row: dict[str, Any]) -> float:
    scores = row["first_token_log_probabilities"]
    if set(scores) != set(LABELS):
        raise ValueError(
            "score vector does not contain the frozen seven-label vocabulary"
        )
    values = sorted((float(value) for value in scores.values()), reverse=True)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("non-finite first-token score")
    return values[0] - values[1]


def check_summary_metrics(
    summary: dict[str, Any], records: list[dict[str, Any]]
) -> None:
    for language in LANGUAGES:
        subset = [row for row in records if row["language"] == language]
        recomputed = aggregate_metrics(subset, "prediction")
        reported = summary["languages"][language]
        for field in ("n", "accuracy", "macro_f1", "weighted_f1"):
            if not math.isclose(
                float(recomputed[field]),
                float(reported[field]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "summary metric mismatch for "
                    f"{summary['architecture']} {language} {field}"
                )
    recomputed = aggregate_metrics(records, "prediction")
    for field in ("n", "accuracy", "macro_f1", "weighted_f1"):
        if not math.isclose(
            float(recomputed[field]),
            float(summary["overall"][field]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"overall summary metric mismatch for {summary['architecture']} {field}"
            )


def validate_reference(reference: dict[str, Any], description: str) -> Path:
    path = Path(reference["path"])
    require_sha(path, reference["sha256"], description)
    return path


def validate_official_artifacts(
    official_root: Path,
) -> tuple[
    dict[str, Any],
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    verified_path = official_root / "VERIFIED.json"
    verified = load_json(verified_path)
    if "VERIFIED" not in str(verified.get("status")):
        raise ValueError("official aggregate is not VERIFIED")

    protocol_path = official_root / "execution_protocol.json"
    protocol = load_json(protocol_path)
    protocol_sha = sha256_file(protocol_path)
    if protocol["data_boundary"] != "held_out_test" or protocol["split"] != "test":
        raise ValueError("unexpected official data boundary")
    if protocol["selected_prompts"] != {"eng": "p4", "xho": "p4"}:
        raise ValueError("official protocol does not bind P4 for both languages")
    if set(protocol["models"]) != set(ARCHITECTURES):
        raise ValueError("official protocol architecture set changed")

    for gate_name, reference in protocol["validation_gate"].items():
        validate_reference(reference, f"validation gate {gate_name}")
    validate_reference(protocol["dataset_manifest"], "dataset manifest")
    for architecture, model in protocol["models"].items():
        validate_reference(model["binding"], f"{architecture} binding")

    predictions: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    manifests: dict[str, dict[str, Any]] = {}
    reference_keys: set[tuple[str, int]] | None = None
    reference_gold: dict[tuple[str, int], str] | None = None

    for architecture in ARCHITECTURES:
        architecture_root = official_root / architecture
        predictions_path = architecture_root / "predictions.jsonl"
        summary_path = architecture_root / "summary.json"
        manifest_path = architecture_root / "manifest.json"
        manifest = load_json(manifest_path)
        summary = load_json(summary_path)
        rows = load_jsonl(predictions_path)

        if manifest["status"] != "OFFICIAL_MODEL_VERIFIED":
            raise ValueError(f"{architecture} is not OFFICIAL_MODEL_VERIFIED")
        if (
            not manifest["test_accessed"]
            or manifest["data_boundary"] != "held_out_test"
        ):
            raise ValueError(f"{architecture} does not declare held-out test access")
        if manifest["no_score_based_retry"] is not True:
            raise ValueError(f"{architecture} does not bind no-score-based-retry")
        if manifest["protocol_sha256"] != protocol_sha:
            raise ValueError(f"{architecture} protocol SHA mismatch")
        if (
            manifest["binding_sha256"]
            != protocol["models"][architecture]["binding"]["sha256"]
        ):
            raise ValueError(f"{architecture} binding mismatch")
        if manifest["selected_prompts"] != protocol["selected_prompts"]:
            raise ValueError(f"{architecture} selected prompt mismatch")
        require_sha(
            predictions_path,
            manifest["predictions_sha256"],
            f"{architecture} predictions",
        )
        require_sha(summary_path, manifest["summary_sha256"], f"{architecture} summary")
        if summary["status"] != "OFFICIAL_MODEL_VERIFIED":
            raise ValueError(f"{architecture} summary is not verified")

        keys: set[tuple[str, int]] = set()
        gold: dict[tuple[str, int], str] = {}
        for row in rows:
            if row["architecture"] != architecture:
                raise ValueError(f"wrong architecture in {architecture} prediction row")
            language = row["language"]
            key = (language, int(row["source_index"]))
            if (
                language not in LANGUAGES
                or row["prompt_id"] != protocol["selected_prompts"][language]
            ):
                raise ValueError(
                    f"wrong language/prompt in {architecture} prediction row"
                )
            if row["gold"] not in LABELS or row["prediction"] not in LABELS:
                raise ValueError(
                    f"out-of-vocabulary label in {architecture} prediction row"
                )
            if row["first_token_prediction"] != row["prediction"]:
                raise ValueError(
                    f"central prediction alias mismatch for {architecture}"
                )
            score_margin(row)
            if key in keys:
                raise ValueError(f"duplicate row key for {architecture}: {key}")
            keys.add(key)
            gold[key] = row["gold"]
        if len(rows) != 1245 or Counter(row["language"] for row in rows) != {
            "eng": 948,
            "xho": 297,
        }:
            raise ValueError(f"unexpected official row count for {architecture}")
        if reference_keys is None:
            reference_keys = keys
            reference_gold = gold
        elif keys != reference_keys or gold != reference_gold:
            raise ValueError(f"row-key/gold alignment mismatch for {architecture}")

        check_summary_metrics(summary, rows)
        predictions[architecture] = rows
        summaries[architecture] = summary
        manifests[architecture] = manifest

    return protocol, predictions, summaries, manifests


def validate_and_load_validation(
    protocol: dict[str, Any], manifests: dict[str, dict[str, Any]]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    validation: dict[str, list[dict[str, Any]]] = {}
    validation_summaries: dict[str, dict[str, Any]] = {}
    reference_keys: set[tuple[str, int]] | None = None
    reference_gold: dict[tuple[str, int], str] | None = None
    selection_path = validate_reference(
        protocol["validation_gate"]["prompt_selection"], "prompt selection"
    )
    selection = load_json(selection_path)

    for architecture in ARCHITECTURES:
        summary_path = validate_reference(
            protocol["validation_summaries"][architecture],
            f"{architecture} validation summary",
        )
        summary = load_json(summary_path)
        predictions_path = summary_path.with_name("general_predictions.jsonl")
        require_sha(
            predictions_path,
            summary["predictions_sha256"],
            f"{architecture} validation predictions",
        )
        validation_manifest = load_json(summary_path.with_name("manifest.json"))
        if (
            validation_manifest["data_boundary"] != "validation_only"
            or validation_manifest["test_accessed"]
        ):
            raise ValueError(f"{architecture} validation boundary is not clean")
        if (
            validation_manifest["arm_binding_sha256"]
            != manifests[architecture]["binding_sha256"]
        ):
            raise ValueError(f"{architecture} validation/official binding mismatch")
        if (
            validation_manifest["selected_adapter_revision"]
            != manifests[architecture]["adapter_revision"]
        ):
            raise ValueError(
                f"{architecture} validation/official adapter revision mismatch"
            )
        if (
            validation_manifest["protocol_sha256"]
            != manifests[architecture]["validation_protocol_sha256"]
        ):
            raise ValueError(f"{architecture} validation protocol mismatch")

        rows = [
            row
            for row in load_jsonl(predictions_path)
            if row["prompt_id"] == protocol["selected_prompts"][row["language"]]
        ]
        keys: set[tuple[str, int]] = set()
        gold: dict[tuple[str, int], str] = {}
        for row in rows:
            key = (row["language"], int(row["selected_source_index"]))
            if row["first_token_prediction"] not in LABELS or row["gold"] not in LABELS:
                raise ValueError(f"invalid validation label for {architecture}")
            if key in keys:
                raise ValueError(
                    f"duplicate validation row key for {architecture}: {key}"
                )
            keys.add(key)
            gold[key] = row["gold"]
        if len(rows) != 619 or Counter(row["language"] for row in rows) != {
            "eng": 472,
            "xho": 147,
        }:
            raise ValueError(f"unexpected selected validation rows for {architecture}")
        if reference_keys is None:
            reference_keys = keys
            reference_gold = gold
        elif keys != reference_keys or gold != reference_gold:
            raise ValueError(f"validation alignment mismatch for {architecture}")

        for language in LANGUAGES:
            recomputed = aggregate_metrics(
                [row for row in rows if row["language"] == language],
                "first_token_prediction",
            )
            summary_metrics = summary["subsets"][f"{language}/p4"]["first_token"]
            selected_metrics = selection["candidates"][language]["p4"][
                "per_architecture"
            ][architecture]
            for field in ("accuracy", "macro_f1", "weighted_f1"):
                if not math.isclose(
                    float(recomputed[field]),
                    float(summary_metrics[field]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ) or not math.isclose(
                    float(recomputed[field]),
                    float(selected_metrics[field]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        "validation metric mismatch for "
                        f"{architecture} {language} {field}"
                    )
        validation[architecture] = rows
        validation_summaries[architecture] = summary

    return validation, validation_summaries


def scope_records(records: list[dict[str, Any]], language: str) -> list[dict[str, Any]]:
    if language == "all":
        return records
    return [row for row in records if row["language"] == language]


def build_analysis(
    protocol: dict[str, Any],
    predictions: dict[str, list[dict[str, Any]]],
    summaries: dict[str, dict[str, Any]],
    validation: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    indexed = {
        architecture: {(row["language"], int(row["source_index"])): row for row in rows}
        for architecture, rows in predictions.items()
    }
    ordered_keys = sorted(indexed[ARCHITECTURES[0]])

    per_model: dict[str, Any] = {}
    per_class_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    margin_rows: list[dict[str, Any]] = []
    validation_delta_rows: list[dict[str, Any]] = []

    for architecture in ARCHITECTURES:
        architecture_analysis: dict[str, Any] = {}
        for language in (*LANGUAGES, "all"):
            rows = scope_records(predictions[architecture], language)
            metrics = aggregate_metrics(rows, "prediction")
            classes = class_metrics(rows, "prediction")
            matrix = confusion_matrix(rows)
            margins = [score_margin(row) for row in rows]
            correct_margins = [
                score_margin(row) for row in rows if row["prediction"] == row["gold"]
            ]
            incorrect_margins = [
                score_margin(row) for row in rows if row["prediction"] != row["gold"]
            ]
            architecture_analysis[language] = {
                "metrics": metrics,
                "per_class": classes,
                "confusion_matrix": {"labels": LABELS, "matrix": matrix},
                "top_confusions": top_confusions(rows),
                "margins": {
                    "all": margin_summary(margins),
                    "correct": margin_summary(correct_margins),
                    "incorrect": margin_summary(incorrect_margins),
                },
            }
            for label, values in classes.items():
                per_class_rows.append(
                    {
                        "architecture": architecture,
                        "display_name": DISPLAY_NAMES[architecture],
                        "language": language,
                        "label": label,
                        **values,
                    }
                )
            for gold_index, gold in enumerate(LABELS):
                for prediction_index, predicted in enumerate(LABELS):
                    confusion_rows.append(
                        {
                            "architecture": architecture,
                            "display_name": DISPLAY_NAMES[architecture],
                            "language": language,
                            "gold": gold,
                            "prediction": predicted,
                            "count": matrix[gold_index][prediction_index],
                        }
                    )
            for correctness, values in (
                ("all", margins),
                ("correct", correct_margins),
                ("incorrect", incorrect_margins),
            ):
                margin_rows.append(
                    {
                        "architecture": architecture,
                        "display_name": DISPLAY_NAMES[architecture],
                        "language": language,
                        "correctness": correctness,
                        **margin_summary(values),
                    }
                )
        per_model[architecture] = architecture_analysis

        for language in LANGUAGES:
            validation_rows = [
                row for row in validation[architecture] if row["language"] == language
            ]
            validation_metrics = aggregate_metrics(
                validation_rows, "first_token_prediction"
            )
            test_metrics = summaries[architecture]["languages"][language]
            validation_classes = class_metrics(
                validation_rows, "first_token_prediction"
            )
            test_classes = class_metrics(
                [
                    row
                    for row in predictions[architecture]
                    if row["language"] == language
                ],
                "prediction",
            )
            validation_delta_rows.append(
                {
                    "architecture": architecture,
                    "display_name": DISPLAY_NAMES[architecture],
                    "language": language,
                    "label": "__aggregate__",
                    "validation_n": validation_metrics["n"],
                    "test_n": test_metrics["n"],
                    "validation_accuracy": validation_metrics["accuracy"],
                    "test_accuracy": test_metrics["accuracy"],
                    "accuracy_delta": test_metrics["accuracy"]
                    - validation_metrics["accuracy"],
                    "validation_macro_f1": validation_metrics["macro_f1"],
                    "test_macro_f1": test_metrics["macro_f1"],
                    "macro_f1_delta": test_metrics["macro_f1"]
                    - validation_metrics["macro_f1"],
                    "validation_weighted_f1": validation_metrics["weighted_f1"],
                    "test_weighted_f1": test_metrics["weighted_f1"],
                    "weighted_f1_delta": test_metrics["weighted_f1"]
                    - validation_metrics["weighted_f1"],
                }
            )
            for label in LABELS:
                validation_delta_rows.append(
                    {
                        "architecture": architecture,
                        "display_name": DISPLAY_NAMES[architecture],
                        "language": language,
                        "label": label,
                        "validation_n": validation_classes[label]["support"],
                        "test_n": test_classes[label]["support"],
                        "validation_accuracy": "",
                        "test_accuracy": "",
                        "accuracy_delta": "",
                        "validation_macro_f1": "",
                        "test_macro_f1": "",
                        "macro_f1_delta": "",
                        "validation_weighted_f1": validation_classes[label]["f1"],
                        "test_weighted_f1": test_classes[label]["f1"],
                        "weighted_f1_delta": test_classes[label]["f1"]
                        - validation_classes[label]["f1"],
                    }
                )

    pairwise_rows: list[dict[str, Any]] = []
    for language in (*LANGUAGES, "all"):
        keys = [key for key in ordered_keys if language == "all" or key[0] == language]
        for first, second in combinations(ARCHITECTURES, 2):
            first_only = second_only = both_correct = both_wrong = 0
            prediction_disagreements = 0
            for key in keys:
                gold = indexed[first][key]["gold"]
                first_correct = indexed[first][key]["prediction"] == gold
                second_correct = indexed[second][key]["prediction"] == gold
                both_correct += first_correct and second_correct
                both_wrong += not first_correct and not second_correct
                first_only += first_correct and not second_correct
                second_only += second_correct and not first_correct
                prediction_disagreements += (
                    indexed[first][key]["prediction"]
                    != indexed[second][key]["prediction"]
                )
            pairwise_rows.append(
                {
                    "language": language,
                    "first": first,
                    "first_display_name": DISPLAY_NAMES[first],
                    "second": second,
                    "second_display_name": DISPLAY_NAMES[second],
                    "n": len(keys),
                    "both_correct": both_correct,
                    "first_only_correct": first_only,
                    "second_only_correct": second_only,
                    "both_wrong": both_wrong,
                    "prediction_disagreements": prediction_disagreements,
                    "accuracy_delta_first_minus_second": (first_only - second_only)
                    / len(keys),
                    "mcnemar_exact_p": exact_mcnemar_p(first_only, second_only),
                }
            )
    add_holm_adjustment(pairwise_rows)

    consensus: dict[str, Any] = {}
    for language in (*LANGUAGES, "all"):
        keys = [key for key in ordered_keys if language == "all" or key[0] == language]
        correct_count_distribution: Counter[int] = Counter()
        unique_correct = Counter({architecture: 0 for architecture in ARCHITECTURES})
        unanimous_prediction = unanimous_correct = unanimous_wrong = 0
        majority_prediction = majority_correct = 0
        for key in keys:
            gold = indexed[ARCHITECTURES[0]][key]["gold"]
            model_predictions = {
                architecture: indexed[architecture][key]["prediction"]
                for architecture in ARCHITECTURES
            }
            correct_models = [
                architecture
                for architecture, prediction in model_predictions.items()
                if prediction == gold
            ]
            correct_count_distribution[len(correct_models)] += 1
            if len(correct_models) == 1:
                unique_correct[correct_models[0]] += 1
            counts = Counter(model_predictions.values())
            top_label, top_count = counts.most_common(1)[0]
            if top_count == len(ARCHITECTURES):
                unanimous_prediction += 1
                unanimous_correct += top_label == gold
                unanimous_wrong += top_label != gold
            if top_count >= 3:
                majority_prediction += 1
                majority_correct += top_label == gold
        consensus[language] = {
            "n": len(keys),
            "correct_model_count_distribution": {
                str(index): correct_count_distribution[index] for index in range(5)
            },
            "all_models_correct": correct_count_distribution[4],
            "no_model_correct": correct_count_distribution[0],
            "oracle_any_model_accuracy": 1.0
            - correct_count_distribution[0] / len(keys),
            "unique_correct_by_model": dict(unique_correct),
            "unanimous_prediction": unanimous_prediction,
            "unanimous_correct": unanimous_correct,
            "unanimous_wrong": unanimous_wrong,
            "majority_prediction_at_least_three": majority_prediction,
            "majority_correct": majority_correct,
            "majority_accuracy_when_available": safe_div(
                majority_correct, majority_prediction
            ),
        }

    support_shift: dict[str, Any] = {}
    for language in LANGUAGES:
        validation_support = Counter(
            row["gold"]
            for row in validation[ARCHITECTURES[0]]
            if row["language"] == language
        )
        test_support = Counter(
            row["gold"]
            for row in predictions[ARCHITECTURES[0]]
            if row["language"] == language
        )
        support_shift[language] = {
            label: {
                "validation": validation_support[label],
                "test": test_support[label],
                "test_minus_validation": test_support[label]
                - validation_support[label],
            }
            for label in LABELS
        }

    aggregation_rows = []
    for architecture in ARCHITECTURES:
        eng = float(summaries[architecture]["languages"]["eng"]["weighted_f1"])
        xho = float(summaries[architecture]["languages"]["xho"]["weighted_f1"])
        aggregation_rows.append(
            {
                "architecture": architecture,
                "display_name": DISPLAY_NAMES[architecture],
                "eng_weighted_f1": eng,
                "xho_weighted_f1": xho,
                "row_pooled_weighted_f1": float(
                    summaries[architecture]["overall"]["weighted_f1"]
                ),
                "equal_language_mean_weighted_f1": (eng + xho) / 2.0,
            }
        )
    aggregation_rankings = {
        field: [
            row["architecture"]
            for row in sorted(
                aggregation_rows, key=lambda row: float(row[field]), reverse=True
            )
        ]
        for field in (
            "eng_weighted_f1",
            "xho_weighted_f1",
            "row_pooled_weighted_f1",
            "equal_language_mean_weighted_f1",
        )
    }

    return {
        "schema": "sallm.news_general_official_deep_analysis/v2",
        "status": "ANALYSIS_VERIFIED",
        "source_boundary": "sealed_prediction_artifacts_only_no_source_tsv_access",
        "selected_prompts": protocol["selected_prompts"],
        "architectures": ARCHITECTURES,
        "labels": LABELS,
        "integrity": {
            "official_row_alignment": "passed",
            "official_gold_alignment": "passed",
            "official_prediction_summary_hashes": "passed",
            "official_metric_recomputation": "passed",
            "selected_prompt_identity": "passed",
            "validation_reference_hashes": "passed",
            "validation_row_and_gold_alignment": "passed",
            "validation_to_official_binding_identity": "passed",
            "validation_to_official_adapter_revision_identity": "passed",
            "validation_metric_recomputation_and_selection_identity": "passed",
        },
        "per_model": per_model,
        "pairwise": pairwise_rows,
        "consensus": consensus,
        "validation_to_test": validation_delta_rows,
        "class_support_validation_to_test": support_shift,
        "aggregation_sensitivity": {
            "rows": aggregation_rows,
            "rankings": aggregation_rankings,
        },
        "tables": {
            "per_class": per_class_rows,
            "confusion": confusion_rows,
            "margins": margin_rows,
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol, predictions, summaries, manifests = validate_official_artifacts(
        args.official_root
    )
    validation, _ = validate_and_load_validation(protocol, manifests)
    analysis = build_analysis(protocol, predictions, summaries, validation)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    analysis_path = args.output_dir / "analysis.json"
    with analysis_path.open("w", encoding="utf-8") as handle:
        json.dump(analysis, handle, indent=2, sort_keys=True)
        handle.write("\n")
    write_csv(
        args.output_dir / "per_class_metrics.csv", analysis["tables"]["per_class"]
    )
    write_csv(
        args.output_dir / "confusion_matrices.csv", analysis["tables"]["confusion"]
    )
    write_csv(args.output_dir / "pairwise_diagnostics.csv", analysis["pairwise"])
    write_csv(args.output_dir / "margin_summaries.csv", analysis["tables"]["margins"])
    write_csv(
        args.output_dir / "validation_to_test.csv", analysis["validation_to_test"]
    )
    write_csv(
        args.output_dir / "aggregation_sensitivity.csv",
        analysis["aggregation_sensitivity"]["rows"],
    )

    manifest = {
        "schema": "sallm.news_general_official_deep_analysis_manifest/v2",
        "status": "ANALYSIS_VERIFIED",
        "source_boundary": analysis["source_boundary"],
        "files": {
            path.name: sha256_file(path)
            for path in sorted(args.output_dir.iterdir())
            if path.is_file()
        },
    }
    manifest_path = args.output_dir / "VERIFIED.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "output_dir": str(args.output_dir),
                "manifest_sha256": sha256_file(manifest_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
