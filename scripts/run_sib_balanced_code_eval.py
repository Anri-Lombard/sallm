#!/usr/bin/env python3
"""Validation-only SIB-200 scoring with balanced one-token label codes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any

import torch
from sallm.config import ModelEvalConfig
from sallm.evaluation.classification_metrics import ClassificationEvaluator
from sallm.evaluation.constrained_label_scoring import (
    _pad_batch,
    continuation_ids,
    encode,
)
from sallm.evaluation.harness import load_model_and_tokenizer
from transformers import AutoTokenizer

ARCHITECTURES = ("mzansilm", "mamba2", "xlstm", "gdn")
LANGUAGES = ("afr", "eng", "nso", "sot", "xho", "zul")
PROMPTS = tuple(range(1, 6))
LABELS = (
    "science/technology",
    "travel",
    "politics",
    "sports",
    "health",
    "entertainment",
    "geography",
)
CODES = tuple("0123456")
ROTATIONS = tuple(range(len(CODES)))
EXPECTED_ROWS_PER_TASK = 99
REQUIRED_ROW_FIELDS = ("doc_id", "doc_hash", "prompt_hash", "target_hash")
MAX_CONTEXT_TOKENS = 2048
TIE_EPSILON = 1e-8
MIN_UNIQUE_PREDICTIONS = 3
MAX_DOMINANT_SHARE = 0.80
MIN_EFFECTIVE_CLASSES = 2.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_tasks() -> set[str]:
    return {
        f"sallm_sib_{language}_val_prompt_{prompt}"
        for language in LANGUAGES
        for prompt in PROMPTS
    }


def task_parts(task: str) -> tuple[str, int]:
    prefix = "sallm_sib_"
    separator = "_val_prompt_"
    if not task.startswith(prefix) or separator not in task:
        raise ValueError(f"Unexpected validation task {task!r}")
    language, prompt = task.removeprefix(prefix).split(separator, 1)
    if language not in LANGUAGES or int(prompt) not in PROMPTS:
        raise ValueError(f"Unexpected validation task {task!r}")
    return language, int(prompt)


def load_validation_source(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = payload.get("samples")
    configs = payload.get("configs")
    if not isinstance(samples, dict) or set(samples) != expected_tasks():
        raise ValueError("Source must contain the exact 30-task SIB validation grid.")
    if not isinstance(configs, dict):
        raise ValueError("Source is missing resolved task configurations.")

    row_fingerprints: dict[tuple[str, int], list[tuple[Any, str, str, str]]] = {}
    for task in sorted(samples):
        language, prompt = task_parts(task)
        config = configs.get(task, {})
        if config.get("validation_split") != "validation" or "_val_" not in task:
            raise ValueError(f"{task}: only the validation split is permitted.")
        if tuple(config.get("doc_to_choice", ())) != LABELS:
            raise ValueError(f"{task}: label order differs from the sealed contract.")
        rows = samples[task]
        if len(rows) != EXPECTED_ROWS_PER_TASK:
            raise ValueError(
                f"{task}: expected {EXPECTED_ROWS_PER_TASK} rows, got {len(rows)}."
            )
        targets = Counter(str(row.get("target", "")).strip() for row in rows)
        if set(targets) != set(LABELS):
            raise ValueError(f"{task}: gold labels are incomplete: {dict(targets)}")
        fingerprints = []
        for row in rows:
            missing = [
                field
                for field in REQUIRED_ROW_FIELDS
                if field not in row or row[field] is None or row[field] == ""
            ]
            if missing:
                raise ValueError(
                    f"{task}: row {row.get('doc_id')} is missing frozen identity "
                    f"fields: {missing}."
                )
            arguments = row.get("arguments")
            if not isinstance(arguments, list) or len(arguments) != len(LABELS):
                raise ValueError(
                    f"{task}: row {row.get('doc_id')} has invalid choices."
                )
            contexts = {str(argument[0]) for argument in arguments}
            choices = tuple(str(argument[1]).strip() for argument in arguments)
            if len(contexts) != 1 or choices != LABELS:
                raise ValueError(
                    f"{task}: row {row.get('doc_id')} changed prompts/labels."
                )
            fingerprints.append(
                (
                    row["doc_id"],
                    str(row["doc_hash"]),
                    str(row["target_hash"]),
                    str(row["target"]).strip(),
                )
            )
        if len(set(fingerprints)) != EXPECTED_ROWS_PER_TASK:
            raise ValueError(f"{task}: frozen validation rows are not unique.")
        row_fingerprints[(language, prompt)] = fingerprints

    for language in LANGUAGES:
        baseline = row_fingerprints[(language, PROMPTS[0])]
        if any(row_fingerprints[(language, prompt)] != baseline for prompt in PROMPTS):
            raise ValueError(
                f"{language}: prompt variants do not contain identical ordered rows."
            )
    return payload


def codebook(rotation: int) -> tuple[tuple[str, str], ...]:
    if rotation not in ROTATIONS:
        raise ValueError(f"Missing or invalid rotation {rotation}.")
    return tuple(
        (code, LABELS[(code_index - rotation) % len(LABELS)])
        for code_index, code in enumerate(CODES)
    )


def coded_prompt(original_prompt: str, rotation: int) -> str:
    mapping = "\n".join(f"{code} = {label}" for code, label in codebook(rotation))
    return (
        f"{original_prompt.rstrip()}\n\n"
        "Choose using this codebook:\n"
        f"{mapping}\n"
        "Reply with exactly one code.\nCode:"
    )


def tokenizer_audit(tokenizer: Any, source: dict[str, Any]) -> dict[str, Any]:
    observed: dict[str, set[tuple[int, ...]]] = {code: set() for code in CODES}
    maximum_context_tokens = 0
    overflow = []
    collisions = []
    contexts_checked = 0
    for task in sorted(source["samples"]):
        for row in source["samples"][task]:
            original = str(row["arguments"][0][0])
            for rotation in ROTATIONS:
                prompt = coded_prompt(original, rotation)
                prompt_ids = encode(tokenizer, prompt)
                context_tokens = len(prompt_ids) + 1
                contexts_checked += 1
                maximum_context_tokens = max(maximum_context_tokens, context_tokens)
                if context_tokens > MAX_CONTEXT_TOKENS:
                    overflow.append(
                        {
                            "task": task,
                            "doc_id": row["doc_id"],
                            "rotation": rotation,
                            "context_tokens": context_tokens,
                        }
                    )
                ids_by_code = {
                    code: tuple(
                        continuation_ids(tokenizer, prompt, prompt_ids, f" {code}")
                    )
                    for code in CODES
                }
                for code, ids in ids_by_code.items():
                    observed[code].add(ids)
                single_token_ids = [
                    ids[0] for ids in ids_by_code.values() if len(ids) == 1
                ]
                if len(single_token_ids) == len(CODES) and len(
                    set(single_token_ids)
                ) != len(CODES):
                    collisions.append(
                        {
                            "task": task,
                            "doc_id": row["doc_id"],
                            "rotation": rotation,
                            "token_ids": single_token_ids,
                        }
                    )

    failures = {
        code: sorted(map(list, tokenizations))
        for code, tokenizations in observed.items()
        if not tokenizations or any(len(ids) != 1 for ids in tokenizations)
    }
    return {
        "pass": not failures and not overflow and not collisions,
        "rows_checked": sum(len(rows) for rows in source["samples"].values()),
        "contexts_checked": contexts_checked,
        "continuations_checked": contexts_checked * len(CODES),
        "maximum_context_tokens_with_bos": maximum_context_tokens,
        "maximum_context_tokens_allowed": MAX_CONTEXT_TOKENS,
        "context_overflow": overflow,
        "code_token_collisions": collisions,
        "continuation_token_ids": {
            code: sorted(map(list, tokenizations))
            for code, tokenizations in observed.items()
        },
        "failures": failures,
    }


def parse_tokenizers(values: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        architecture, separator, path = value.partition("=")
        if not separator or architecture in parsed:
            raise ValueError("Use each --tokenizer once as ARCHITECTURE=PATH.")
        parsed[architecture] = Path(path)
    if set(parsed) != set(ARCHITECTURES):
        raise ValueError(
            f"Tokenizer audit requires exactly: {', '.join(ARCHITECTURES)}"
        )
    return parsed


def run_tokenizer_audit(args: argparse.Namespace) -> None:
    source = load_validation_source(args.source_results)
    source_hash = sha256(args.source_results)
    bindings_hash = sha256(args.bindings)
    tokenizers = parse_tokenizers(args.tokenizer)
    results = {}
    for architecture in ARCHITECTURES:
        path = tokenizers[architecture]
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        result = tokenizer_audit(tokenizer, source)
        tokenizer_json = path / "tokenizer.json"
        result.update(
            {
                "path": str(path),
                "tokenizer_json_sha256": (
                    sha256(tokenizer_json) if tokenizer_json.is_file() else None
                ),
            }
        )
        results[architecture] = result

    report = {
        "schema": "sallm.sib_balanced_code_token_audit/v1",
        "split": "validation",
        "source": {"path": str(args.source_results), "sha256": source_hash},
        "bindings": {"path": str(args.bindings), "sha256": bindings_hash},
        "codes": list(CODES),
        "required_continuation_tokens_per_code": 1,
        "architectures": results,
        "pass": all(result["pass"] for result in results.values()),
    }
    write_report(args.output, report)
    if not report["pass"]:
        raise SystemExit("Tokenizer audit failed; no model evaluation is admitted.")


def load_required_report(path: Path, schema: str) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != schema:
        raise ValueError(f"{path}: expected schema {schema!r}.")
    return report


def validate_token_audit(
    path: Path,
    *,
    architecture: str,
    source_hash: str,
    bindings_hash: str,
) -> dict[str, Any]:
    report = load_required_report(path, "sallm.sib_balanced_code_token_audit/v1")
    if report.get("split") != "validation" or not report.get("pass"):
        raise ValueError("Tokenizer audit did not pass on validation-only input.")
    if report.get("source", {}).get("sha256") != source_hash:
        raise ValueError("Tokenizer audit source hash differs from this run.")
    if report.get("bindings", {}).get("sha256") != bindings_hash:
        raise ValueError("Tokenizer audit bindings hash differs from this run.")
    architecture_report = report.get("architectures", {}).get(architecture, {})
    if not architecture_report.get("pass"):
        raise ValueError(f"Tokenizer audit failed for {architecture}.")
    return architecture_report


def validate_canary(
    path: Path,
    *,
    args: argparse.Namespace,
    source_hash: str,
    bindings_hash: str,
    require_admission: bool = True,
) -> dict[str, Any]:
    report = load_required_report(path, "sallm.sib_balanced_code_eval/v1")
    expected = {
        "architecture": args.architecture,
        "checkpoint": args.checkpoint,
        "adapter": args.adapter,
        "dtype": args.dtype,
        "device": args.device,
        "merge_lora": args.merge_lora,
        "tie_word_embeddings": args.tie_word_embeddings,
    }
    if report.get("mode") != "canary":
        raise ValueError("Expected a canary report.")
    if require_admission and not report.get("canary_admitted"):
        raise ValueError("Full validation requires a passing canary report.")
    if report.get("run") != expected:
        raise ValueError("Canary model/runtime binding differs from this run.")
    if report.get("source", {}).get("sha256") != source_hash:
        raise ValueError("Canary source hash differs from this run.")
    if report.get("bindings", {}).get("sha256") != bindings_hash:
        raise ValueError("Canary bindings hash differs from this run.")
    if report.get("runtime", {}).get("cyclic_vs_fixed_forward_multiplier") != 7:
        raise ValueError("Canary is missing the sealed seven-rotation projection.")
    return report


def selected_rows(
    source: dict[str, Any], *, canary: bool
) -> dict[str, list[dict[str, Any]]]:
    if not canary:
        return {task: list(rows) for task, rows in source["samples"].items()}

    selected = {}
    for task, rows in source["samples"].items():
        by_label: dict[str, dict[str, Any]] = {}
        for row in rows:
            by_label.setdefault(str(row["target"]).strip(), row)
        if set(by_label) != set(LABELS):
            raise ValueError(f"{task}: canary cannot cover every gold class.")
        selected[task] = [by_label[label] for label in LABELS]
    return selected


def score_rotations(
    *,
    model: Any,
    tokenizer: Any,
    original_prompt: str,
    pad_token_id: int,
    pad_to_multiple_of: int | None,
    device: torch.device,
) -> list[dict[str, float]]:
    contexts: list[list[int]] = []
    code_ids_by_rotation: list[dict[str, int]] = []
    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        raise ValueError("Balanced-code scoring requires a BOS token.")

    for rotation in ROTATIONS:
        prompt = coded_prompt(original_prompt, rotation)
        prompt_ids = encode(tokenizer, prompt)
        ids_by_code = {
            code: continuation_ids(tokenizer, prompt, prompt_ids, f" {code}")
            for code in CODES
        }
        invalid = {code: ids for code, ids in ids_by_code.items() if len(ids) != 1}
        if invalid:
            raise ValueError(
                f"Rotation {rotation} has non-single-token codes: {invalid}"
            )
        token_ids = [ids_by_code[code][0] for code in CODES]
        if len(set(token_ids)) != len(CODES):
            raise ValueError(
                f"Rotation {rotation} maps distinct codes to duplicate token IDs."
            )
        contexts.append([int(bos_id), *prompt_ids])
        code_ids_by_rotation.append(
            {code: token_ids[0] for code, token_ids in ids_by_code.items()}
        )

    if len(contexts) != len(ROTATIONS):
        raise ValueError("All seven cyclic codebook rotations are required.")
    if any(len(context) > MAX_CONTEXT_TOKENS for context in contexts):
        raise ValueError(
            f"Balanced-code prompt exceeds {MAX_CONTEXT_TOKENS} context tokens."
        )
    input_ids, attention_mask = _pad_batch(
        [torch.tensor(context, dtype=torch.long) for context in contexts],
        pad_token_id,
        pad_to_multiple_of,
    )
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    with torch.no_grad():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits

    rotation_scores = []
    for rotation, context in enumerate(contexts):
        log_probs = torch.log_softmax(
            logits[rotation, len(context) - 1].float(), dim=-1
        )
        if log_probs.dtype != torch.float32:
            raise ValueError("Balanced-code log-softmax did not remain float32.")
        score_by_code = {
            code: float(log_probs[token_id].item())
            for code, token_id in code_ids_by_rotation[rotation].items()
        }
        rotation_scores.append(
            {label: score_by_code[code] for code, label in codebook(rotation)}
        )
    return rotation_scores


def prediction(scores: dict[str, float]) -> tuple[str, float, bool]:
    ranked = sorted(LABELS, key=lambda label: scores[label], reverse=True)
    margin = scores[ranked[0]] - scores[ranked[1]]
    return ranked[0], margin, margin <= TIE_EPSILON


def semantic_geometric_mean_scores(
    rotation_scores: list[dict[str, float]],
) -> dict[str, float]:
    """Return log geometric-mean probability for each semantic label."""
    if len(rotation_scores) != len(ROTATIONS):
        raise ValueError("A row is missing cyclic codebook rotations.")
    return {
        label: fmean(scores[label] for scores in rotation_scores)
        for label in LABELS
    }


def summarize(rows: list[dict[str, Any]], prediction_key: str) -> dict[str, Any]:
    gold = [str(row["gold"]) for row in rows]
    predicted = [str(row[prediction_key]) for row in rows]
    evaluator = ClassificationEvaluator.__new__(ClassificationEvaluator)
    metrics = evaluator._compute_classification_metrics(gold, predicted)
    gold_counts = Counter(gold)
    prediction_counts = Counter(predicted)
    confusion = {
        gold_label: {
            predicted_label: sum(
                actual == gold_label and guess == predicted_label
                for actual, guess in zip(gold, predicted, strict=True)
            )
            for predicted_label in LABELS
        }
        for gold_label in LABELS
    }
    if any(sum(confusion[label].values()) != gold_counts[label] for label in LABELS):
        raise ValueError("Confusion row sums do not match gold frequencies.")
    if any(
        sum(confusion[label][predicted_label] for label in LABELS)
        != prediction_counts[predicted_label]
        for predicted_label in LABELS
    ):
        raise ValueError("Confusion column sums do not match prediction frequencies.")

    probabilities = [count / len(rows) for count in prediction_counts.values()]
    entropy = -sum(value * math.log(value) for value in probabilities)
    normalized_entropy = entropy / math.log(len(LABELS))
    tie_key = prediction_key.replace("prediction", "tie")
    summary = {
        "n": len(rows),
        "metrics": metrics,
        "gold_counts": {label: gold_counts[label] for label in LABELS},
        "prediction_counts": {label: prediction_counts[label] for label in LABELS},
        "confusion": confusion,
        "unique_predictions": len(prediction_counts),
        "dominant_prediction_share": max(prediction_counts.values()) / len(rows),
        "normalized_prediction_entropy": normalized_entropy,
        "effective_prediction_classes": math.exp(entropy),
        "top_score_ties": sum(bool(row[tie_key]) for row in rows),
    }
    summary["gate"] = nondegeneracy_gate(summary)
    return summary


def nondegeneracy_gate(summary: dict[str, Any]) -> dict[str, Any]:
    failures = []
    if summary["unique_predictions"] < MIN_UNIQUE_PREDICTIONS:
        failures.append(f"unique_predictions<{MIN_UNIQUE_PREDICTIONS}")
    if summary["dominant_prediction_share"] > MAX_DOMINANT_SHARE:
        failures.append(f"dominant_prediction_share>{MAX_DOMINANT_SHARE}")
    if summary["effective_prediction_classes"] < MIN_EFFECTIVE_CLASSES:
        failures.append(f"effective_prediction_classes<{MIN_EFFECTIVE_CLASSES}")
    if summary["top_score_ties"]:
        failures.append("top_score_ties>0")
    return {"pass": not failures, "failures": failures}


def grouped_summaries(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    tasks = sorted({str(row["task"]) for row in rows})
    return {
        task: summarize([row for row in rows if row["task"] == task], key)
        for task in tasks
    }


def run_score(args: argparse.Namespace) -> None:
    if args.architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture {args.architecture!r}.")
    if args.canary and args.diagnostic_full:
        raise ValueError("Choose either --canary or --diagnostic-full.")
    source = load_validation_source(args.source_results)
    source_hash = sha256(args.source_results)
    bindings_hash = sha256(args.bindings)
    architecture_token_audit = validate_token_audit(
        args.token_audit,
        architecture=args.architecture,
        source_hash=source_hash,
        bindings_hash=bindings_hash,
    )
    if not args.canary:
        if args.admit_from_canary is None:
            raise ValueError("Full validation requires --admit-from-canary.")
        validate_canary(
            args.admit_from_canary,
            args=args,
            source_hash=source_hash,
            bindings_hash=bindings_hash,
            require_admission=not args.diagnostic_full,
        )

    started = time.monotonic()
    model, tokenizer = load_model_and_tokenizer(
        ModelEvalConfig(
            checkpoint=args.checkpoint,
            dtype=args.dtype,
            device=args.device,
            peft_adapter=args.adapter,
            merge_lora=args.merge_lora,
            tie_word_embeddings=args.tie_word_embeddings,
        )
    )
    load_seconds = time.monotonic() - started
    tokenizer_json = Path(args.checkpoint) / "tokenizer.json"
    if not tokenizer_json.is_file():
        raise ValueError("Loaded checkpoint is missing tokenizer.json.")
    if sha256(tokenizer_json) != architecture_token_audit.get(
        "tokenizer_json_sha256"
    ):
        raise ValueError("Loaded model tokenizer differs from the passing token audit.")

    model.eval()
    device = model.device
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Model tokenizer has neither PAD nor EOS token.")
    pad_multiple = ClassificationEvaluator._get_model_chunk_size(model)
    rows_by_task = selected_rows(source, canary=args.canary)

    scoring_started = time.monotonic()
    output_rows = []
    for task in sorted(rows_by_task):
        for source_row in rows_by_task[task]:
            original_prompt = str(source_row["arguments"][0][0])
            rotation_scores = score_rotations(
                model=model,
                tokenizer=tokenizer,
                original_prompt=original_prompt,
                pad_token_id=int(pad_token_id),
                pad_to_multiple_of=pad_multiple,
                device=device,
            )
            fixed_scores = rotation_scores[0]
            cyclic_scores = semantic_geometric_mean_scores(rotation_scores)
            fixed_pred, fixed_margin, fixed_tie = prediction(fixed_scores)
            cyclic_pred, cyclic_margin, cyclic_tie = prediction(cyclic_scores)
            output_rows.append(
                {
                    "task": task,
                    "doc_id": source_row["doc_id"],
                    "doc_hash": source_row.get("doc_hash"),
                    "prompt_hash": source_row.get("prompt_hash"),
                    "target_hash": source_row.get("target_hash"),
                    "gold": str(source_row["target"]).strip(),
                    "fixed_prediction": fixed_pred,
                    "fixed_margin": fixed_margin,
                    "fixed_tie": fixed_tie,
                    "cyclic_prediction": cyclic_pred,
                    "cyclic_margin": cyclic_margin,
                    "cyclic_tie": cyclic_tie,
                    "fixed_scores": fixed_scores,
                    "cyclic_scores": cyclic_scores,
                    "rotation_scores": rotation_scores,
                }
            )
    scoring_seconds = time.monotonic() - scoring_started

    expected_row_keys = {
        (task, row["doc_id"])
        for task, rows in rows_by_task.items()
        for row in rows
    }
    observed_row_keys = {(row["task"], row["doc_id"]) for row in output_rows}
    if (
        len(output_rows) != len(expected_row_keys)
        or observed_row_keys != expected_row_keys
    ):
        raise ValueError("Scoring output failed the row-completeness gate.")

    fixed_tasks = grouped_summaries(output_rows, "fixed_prediction")
    cyclic_tasks = grouped_summaries(output_rows, "cyclic_prediction")
    fixed_all = summarize(output_rows, "fixed_prediction")
    cyclic_all = summarize(output_rows, "cyclic_prediction")
    language_gates = {}
    for language in LANGUAGES:
        language_rows = [
            row for row in output_rows if task_parts(str(row["task"]))[0] == language
        ]
        language_gates[language] = summarize(language_rows, "cyclic_prediction")["gate"]

    full_rows = len(expected_tasks()) * EXPECTED_ROWS_PER_TASK
    projected_scoring_seconds = scoring_seconds * full_rows / len(output_rows)
    report = {
        "schema": "sallm.sib_balanced_code_eval/v1",
        "mode": (
            "canary"
            if args.canary
            else "diagnostic_full"
            if args.diagnostic_full
            else "full"
        ),
        "split": "validation",
        "source": {"path": str(args.source_results), "sha256": source_hash},
        "bindings": {"path": str(args.bindings), "sha256": bindings_hash},
        "token_audit": {
            "path": str(args.token_audit),
            "sha256": sha256(args.token_audit),
        },
        "run": {
            "architecture": args.architecture,
            "checkpoint": args.checkpoint,
            "adapter": args.adapter,
            "dtype": args.dtype,
            "device": args.device,
            "merge_lora": args.merge_lora,
            "tie_word_embeddings": args.tie_word_embeddings,
        },
        "scorer": {
            "labels": list(LABELS),
            "codes": list(CODES),
            "fixed_baseline_rotation": 0,
            "cyclic_rotations": list(ROTATIONS),
            "aggregation": (
                "semantic geometric mean of code probabilities over seven cyclic "
                "codebooks, implemented as mean log probability"
            ),
            "selection_metric": "support_weighted_f1",
            "score_precision": (
                "float32 log-softmax followed by Python float mean-log aggregation"
            ),
            "tie_epsilon": TIE_EPSILON,
            "maximum_context_tokens": MAX_CONTEXT_TOKENS,
            "gate_thresholds": {
                "minimum_unique_predictions": MIN_UNIQUE_PREDICTIONS,
                "maximum_dominant_prediction_share": MAX_DOMINANT_SHARE,
                "minimum_effective_prediction_classes": MIN_EFFECTIVE_CLASSES,
                "maximum_top_score_ties": 0,
            },
        },
        "runtime": {
            "load_seconds": load_seconds,
            "scoring_seconds": scoring_seconds,
            "rows": len(output_rows),
            "rotations_per_row": len(ROTATIONS),
            "fixed_forward_equivalents": len(output_rows),
            "cyclic_forward_equivalents": len(output_rows) * len(ROTATIONS),
            "cyclic_vs_fixed_forward_multiplier": len(ROTATIONS),
            "projected_full_scoring_seconds": projected_scoring_seconds,
            "projected_full_seconds_with_20pct_margin": (
                load_seconds + 1.2 * projected_scoring_seconds
            ),
        },
        "completeness_gate": {
            "pass": True,
            "expected_rows": len(expected_row_keys),
            "observed_rows": len(output_rows),
            "unique_task_doc_ids": len(observed_row_keys),
            "rotations_per_row": len(ROTATIONS),
        },
        "fixed_codebook_baseline": {"all": fixed_all, "tasks": fixed_tasks},
        "cyclic_codebook": {
            "all": cyclic_all,
            "tasks": cyclic_tasks,
            "language_gates": language_gates,
        },
        "rows": output_rows,
        "heldout_action": "none; this scorer accepts validation artifacts only",
    }
    report["canary_admitted"] = bool(
        args.canary
        and cyclic_all["top_score_ties"] == 0
        and all(gate["pass"] for gate in language_gates.values())
    )
    write_report(args.output, report)
    if args.canary and not report["canary_admitted"]:
        raise SystemExit("Canary failed closed; full validation is not admitted.")


def run_select(args: argparse.Namespace) -> None:
    reports = [
        load_required_report(path, "sallm.sib_balanced_code_eval/v1")
        for path in args.report
    ]
    by_architecture = {report["run"]["architecture"]: report for report in reports}
    if set(by_architecture) != set(ARCHITECTURES) or len(reports) != len(ARCHITECTURES):
        raise ValueError("Selection requires one full report for each architecture.")
    source_hashes = {report["source"]["sha256"] for report in reports}
    bindings_hashes = {report["bindings"]["sha256"] for report in reports}
    if len(source_hashes) != 1 or len(bindings_hashes) != 1:
        raise ValueError("Full reports do not share source/binding hashes.")
    for report in reports:
        if report.get("mode") != "full" or report.get("split") != "validation":
            raise ValueError("Selection accepts full validation reports only.")
        if report.get("scorer", {}).get("cyclic_rotations") != list(ROTATIONS):
            raise ValueError("Selection report is missing cyclic rotations.")
        if not report.get("completeness_gate", {}).get("pass"):
            raise ValueError("Selection report failed row completeness.")

    selected = {}
    prompt_diagnostics = {}
    for language in LANGUAGES:
        prompt_diagnostics[language] = {}
        eligible = []
        for prompt in PROMPTS:
            task = f"sallm_sib_{language}_val_prompt_{prompt}"
            cells = {
                architecture: by_architecture[architecture]["cyclic_codebook"]["tasks"][
                    task
                ]
                for architecture in ARCHITECTURES
            }
            passed = all(cell["gate"]["pass"] for cell in cells.values())
            mean_f1 = fmean(
                cell["metrics"]["f1"] for cell in cells.values()
            )
            prompt_diagnostics[language][str(prompt)] = {
                "eligible": passed,
                "mean_support_weighted_f1": mean_f1,
                "architecture_gates": {
                    architecture: cells[architecture]["gate"]
                    for architecture in ARCHITECTURES
                },
                "architecture_metrics": {
                    architecture: cells[architecture]["metrics"]
                    for architecture in ARCHITECTURES
                },
            }
            if passed:
                eligible.append((prompt, mean_f1))
        if eligible:
            winner, score = max(eligible, key=lambda item: (item[1], -item[0]))
            selected[language] = {
                "prompt": winner,
                "validation_mean_support_weighted_f1": score,
                "task": f"sallm_sib_{language}_val_prompt_{winner}",
            }
        else:
            selected[language] = None

    ready = all(selected.values())
    output = {
        "schema": "sallm.sib_balanced_code_selection/v1",
        "split": "validation",
        "selection_rule": (
            "Per language, retain prompts whose cyclic cell passes for all four "
            "architectures; maximize unweighted mean support-weighted F1; break "
            "ties by the lowest prompt number."
        ),
        "source_sha256": source_hashes.pop(),
        "bindings_sha256": bindings_hashes.pop(),
        "reports": [
            {"path": str(path), "sha256": sha256(path)} for path in args.report
        ],
        "selected": selected,
        "prompt_diagnostics": prompt_diagnostics,
        "ready_for_heldout": ready,
        "heldout_action": "none; this tool has no test-split execution path",
    }
    write_report(args.output, output)
    if not ready:
        raise SystemExit(
            "Selection failed closed; held-out evaluation is not admitted."
        )


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(path), "sha256": sha256(path)}, indent=2))


def self_check() -> None:
    for label in LABELS:
        seen = [
            code
            for rotation in ROTATIONS
            for code, mapped_label in codebook(rotation)
            if mapped_label == label
        ]
        assert sorted(seen) == sorted(CODES)

    balanced_rows = [
        {
            "gold": label,
            "cyclic_prediction": label,
            "cyclic_tie": False,
        }
        for label in LABELS
    ]
    balanced = summarize(balanced_rows, "cyclic_prediction")
    assert balanced["gate"]["pass"]
    assert balanced["metrics"]["macro_f1"] == 1.0

    collapsed_rows = [
        {
            "gold": label,
            "cyclic_prediction": LABELS[0],
            "cyclic_tie": False,
        }
        for label in LABELS
    ]
    collapsed = summarize(collapsed_rows, "cyclic_prediction")
    assert not collapsed["gate"]["pass"]
    assert collapsed["prediction_counts"][LABELS[0]] == len(LABELS)

    synthetic = [
        {
            label: math.log((rotation + 1) * (index + 1) / 100)
            for index, label in enumerate(LABELS)
        }
        for rotation in ROTATIONS
    ]
    aggregated = semantic_geometric_mean_scores(synthetic)
    for index, label in enumerate(LABELS):
        direct = math.log(
            math.prod(
                (rotation + 1) * (index + 1) / 100 for rotation in ROTATIONS
            )
            ** (1 / len(ROTATIONS))
        )
        assert math.isclose(aggregated[label], direct, rel_tol=1e-12)
    print("SELF_CHECK_OK")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)

    commands.add_parser("self-check")

    audit = commands.add_parser("audit-tokenizers")
    audit.add_argument("--source-results", type=Path, required=True)
    audit.add_argument("--bindings", type=Path, required=True)
    audit.add_argument("--tokenizer", action="append", required=True)
    audit.add_argument("--output", type=Path, required=True)

    score = commands.add_parser("score")
    score.add_argument("--source-results", type=Path, required=True)
    score.add_argument("--bindings", type=Path, required=True)
    score.add_argument("--token-audit", type=Path, required=True)
    score.add_argument("--architecture", required=True)
    score.add_argument("--checkpoint", required=True)
    score.add_argument("--adapter")
    score.add_argument("--dtype", default="bfloat16")
    score.add_argument("--device", default="cuda:0")
    score.add_argument(
        "--merge-lora", action=argparse.BooleanOptionalAction, default=None
    )
    score.add_argument(
        "--tie-word-embeddings",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    score.add_argument("--canary", action="store_true")
    score.add_argument("--diagnostic-full", action="store_true")
    score.add_argument("--admit-from-canary", type=Path)
    score.add_argument("--output", type=Path, required=True)

    select = commands.add_parser("select")
    select.add_argument("--report", action="append", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "self-check":
        self_check()
    elif args.command == "audit-tokenizers":
        run_tokenizer_audit(args)
    elif args.command == "score":
        run_score(args)
    elif args.command == "select":
        run_select(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
