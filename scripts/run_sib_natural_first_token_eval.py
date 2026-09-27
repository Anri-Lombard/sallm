#!/usr/bin/env python3
"""Validation-only SIB scoring via each natural label's first continuation token."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
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
from sallm.evaluation.constrained_label_scoring import chat_messages_prefix
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
EXPECTED_ROWS_PER_TASK = 99
MAX_CONTEXT_TOKENS = 2048
TIE_EPSILON = 1e-8
MIN_UNIQUE_PREDICTIONS = 3
MAX_DOMINANT_SHARE = 0.80
MIN_EFFECTIVE_CLASSES = 2.0
VALIDATION_CORE_SHA256 = (
    "866e9d82a21a894fb21a2044c05105b6d6eeb7f20dd13a5c07d46e84b47c6ba8"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_core(path: Path) -> Any:
    if sha256(path) != VALIDATION_CORE_SHA256:
        raise ValueError("The sealed validation core changed.")
    spec = importlib.util.spec_from_file_location("sib_validation_core", path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load the sealed validation core.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_report(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(path), "sha256": sha256(path)}, sort_keys=True))


def parse_tokenizers(values: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        architecture, separator, raw_path = value.partition("=")
        if not separator or architecture in parsed:
            raise ValueError("Use each --tokenizer once as ARCHITECTURE=PATH.")
        parsed[architecture] = Path(raw_path)
    if set(parsed) != set(ARCHITECTURES):
        raise ValueError(f"Expected exactly {', '.join(ARCHITECTURES)}.")
    return parsed


def tokenizer_identity_sha256(tokenizer: Any) -> str:
    """Bind the in-memory token map, special IDs, and training template."""
    payload = {
        "vocab": sorted(
            (str(token), int(token_id))
            for token, token_id in tokenizer.get_vocab().items()
        ),
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        ),
        "unk_token_id": tokenizer.unk_token_id,
        "additional_special_tokens_ids": list(
            tokenizer.additional_special_tokens_ids
        ),
        "chat_template": tokenizer.chat_template,
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def training_label_boundary(
    tokenizer: Any, prompt: str, core: Any
) -> tuple[list[int], dict[str, list[int]], dict[str, Any]]:
    """Recover the exact common prefix immediately before the trained labels."""
    training_template = tokenizer.chat_template
    if not isinstance(training_template, str) or not training_template:
        raise ValueError("Effective adapter tokenizer has no saved training template.")
    user_messages = [{"role": "user", "content": prompt}]
    generation_text, generation_ids = chat_messages_prefix(tokenizer, user_messages)
    full_without_eos: dict[str, list[int]] = {}
    for label in LABELS:
        messages = [*user_messages, {"role": "assistant", "content": label}]
        rendered_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        direct_ids = list(
            tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False
            )
        )
        if core.encode(tokenizer, rendered_text) != direct_ids:
            raise ValueError(
                "Rendered training example differs from direct tokenization."
            )
        bos_count = (
            direct_ids.count(tokenizer.bos_token_id)
            if tokenizer.bos_token_id is not None
            else 0
        )
        if bos_count not in (0, 1):
            raise ValueError("Training serialization contains repeated BOS tokens.")
        if tokenizer.eos_token_id is None or direct_ids[-1] != tokenizer.eos_token_id:
            raise ValueError("Training serialization does not end in EOS.")
        full_without_eos[label] = direct_ids[:-1]

    common_length = min(len(ids) for ids in full_without_eos.values())
    for position in range(common_length):
        values = {ids[position] for ids in full_without_eos.values()}
        if len(values) != 1:
            common_length = position
            break
    context_ids = full_without_eos[LABELS[0]][:common_length]
    if context_ids[: len(generation_ids)] != generation_ids:
        raise ValueError(
            "Training prefix does not extend the assistant generation prefix."
        )
    if len(context_ids) <= len(generation_ids):
        raise ValueError("Training prefix is missing assistant-content whitespace.")
    label_ids = {
        label: ids[common_length:] for label, ids in full_without_eos.items()
    }
    empty = [label for label, ids in label_ids.items() if not ids]
    if empty:
        raise ValueError(f"Labels have no continuation tokens: {empty}.")
    first_ids = {label: int(ids[0]) for label, ids in label_ids.items()}
    if len(set(first_ids.values())) != len(LABELS):
        raise ValueError(f"Natural labels collide at the first token: {first_ids}.")
    for label in LABELS:
        if context_ids + label_ids[label] != full_without_eos[label]:
            raise ValueError("Training prefix/label tokens fail exact reconstruction.")
    serialization = {
        "generation_prefix_tokens": len(generation_ids),
        "training_label_prefix_tokens": len(context_ids),
        "static_assistant_content_prefix_tokens": len(context_ids)
        - len(generation_ids),
        "causal_logit_index": len(context_ids) - 1,
        "bos_tokens_in_training_prefix": (
            context_ids.count(tokenizer.bos_token_id)
            if tokenizer.bos_token_id is not None
            else 0
        ),
        "training_chat_template_sha256": hashlib.sha256(
            training_template.encode("utf-8")
        ).hexdigest(),
        "generation_prefix_text_sha256": hashlib.sha256(
            generation_text.encode("utf-8")
        ).hexdigest(),
        "training_prefix_ids_sha256": hashlib.sha256(
            json.dumps(context_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    return context_ids, label_ids, serialization


def tokenizer_audit(
    tokenizer: Any, source: dict[str, Any], core: Any
) -> dict[str, Any]:
    first_ids_seen = {label: set() for label in LABELS}
    lengths_seen = {label: set() for label in LABELS}
    generation_prefix_lengths = set()
    training_prefix_lengths = set()
    static_assistant_prefix_lengths = set()
    failures = []
    maximum_context = 0
    rows = 0
    for task in sorted(source["samples"]):
        for row in source["samples"][task]:
            rows += 1
            prompt = str(row["arguments"][0][0])
            try:
                context_ids, full_ids, serialization = training_label_boundary(
                    tokenizer, prompt, core
                )
            except ValueError as error:
                failures.append(
                    {"task": task, "doc_id": row["doc_id"], "error": str(error)}
                )
                continue
            context_tokens = len(context_ids)
            maximum_context = max(maximum_context, context_tokens)
            generation_prefix_lengths.add(serialization["generation_prefix_tokens"])
            training_prefix_lengths.add(serialization["training_label_prefix_tokens"])
            static_assistant_prefix_lengths.add(
                serialization["static_assistant_content_prefix_tokens"]
            )
            if serialization["causal_logit_index"] != context_tokens - 1:
                failures.append(
                    {
                        "task": task,
                        "doc_id": row["doc_id"],
                        "error": "causal logit position differs from prefix end",
                    }
                )
            if context_tokens > MAX_CONTEXT_TOKENS:
                failures.append(
                    {
                        "task": task,
                        "doc_id": row["doc_id"],
                        "error": f"context_tokens>{MAX_CONTEXT_TOKENS}",
                    }
                )
            for label in LABELS:
                first_ids_seen[label].add(full_ids[label][0])
                lengths_seen[label].add(len(full_ids[label]))
    return {
        "pass": not failures and rows == 30 * EXPECTED_ROWS_PER_TASK,
        "rows_checked": rows,
        "continuations_checked": rows * len(LABELS),
        "maximum_context_tokens_with_bos": maximum_context,
        "maximum_context_tokens_allowed": MAX_CONTEXT_TOKENS,
        "first_token_ids": {
            label: sorted(values) for label, values in first_ids_seen.items()
        },
        "full_label_token_lengths": {
            label: sorted(values) for label, values in lengths_seen.items()
        },
        "serialization": {
            "chat_template_sha256": hashlib.sha256(
                str(tokenizer.chat_template).encode("utf-8")
            ).hexdigest(),
            "generation_prefix_token_lengths": sorted(generation_prefix_lengths),
            "training_label_prefix_token_lengths": sorted(training_prefix_lengths),
            "static_assistant_content_prefix_token_lengths": sorted(
                static_assistant_prefix_lengths
            ),
            "exact_training_prefix_reconstruction": not failures,
            "causal_logit_position": "training_label_prefix_tokens - 1",
        },
        "science_technology_first_token_unique": bool(
            first_ids_seen["science/technology"]
            and all(
                first_ids_seen["science/technology"].isdisjoint(first_ids_seen[label])
                for label in LABELS
                if label != "science/technology"
            )
        ),
        "failures": failures,
    }


def run_audit(args: argparse.Namespace) -> None:
    core = load_core(args.validation_core)
    source = core.load_validation_source(args.source_results)
    tokenizers = parse_tokenizers(args.tokenizer)
    results = {}
    for architecture in ARCHITECTURES:
        path = tokenizers[architecture]
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        result = tokenizer_audit(tokenizer, source, core)
        tokenizer_json = path / "tokenizer.json"
        chat_template_file = path / "chat_template.jinja"
        result.update(
            {
                "path": str(path),
                "tokenizer_json_sha256": sha256(tokenizer_json),
                "in_memory_tokenizer_identity_sha256": tokenizer_identity_sha256(
                    tokenizer
                ),
                "chat_template_file_sha256": (
                    sha256(chat_template_file)
                    if chat_template_file.is_file()
                    else None
                ),
            }
        )
        if result["serialization"]["chat_template_sha256"] != result.get(
            "chat_template_file_sha256"
        ):
            result["pass"] = False
            result["failures"].append(
                {"error": "loaded chat template differs from saved adapter template"}
            )
        results[architecture] = result
    report = {
        "schema": "sallm.sib_natural_first_token_audit/v1",
        "split": "validation",
        "source": {
            "path": str(args.source_results),
            "sha256": sha256(args.source_results),
        },
        "bindings": {"path": str(args.bindings), "sha256": sha256(args.bindings)},
        "validation_core": {
            "path": str(args.validation_core),
            "sha256": VALIDATION_CORE_SHA256,
        },
        "continuation": (
            "exact natural label after the common token prefix of the saved "
            "training chat serialization"
        ),
        "labels": list(LABELS),
        "architectures": results,
        "pass": all(result["pass"] for result in results.values()),
        "heldout_action": "none; validation only",
    }
    write_report(args.output, report)
    if not report["pass"]:
        raise SystemExit("Natural-label first-token audit failed closed.")


def validate_audit(
    path: Path,
    architecture: str,
    source_hash: str,
    bindings_hash: str,
) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("schema") != "sallm.sib_natural_first_token_audit/v1"
        or report.get("split") != "validation"
        or not report.get("pass")
        or report.get("source", {}).get("sha256") != source_hash
        or report.get("bindings", {}).get("sha256") != bindings_hash
        or report.get("validation_core", {}).get("sha256")
        != VALIDATION_CORE_SHA256
    ):
        raise ValueError("Tokenizer audit binding failed.")
    architecture_report = report.get("architectures", {}).get(architecture, {})
    if not architecture_report.get("pass"):
        raise ValueError(f"{architecture}: tokenizer audit did not pass.")
    return architecture_report


def validate_canary(
    path: Path,
    run_binding: dict[str, Any],
    source_hash: str,
    bindings_hash: str,
) -> None:
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("schema") != "sallm.sib_natural_first_token_eval/v1"
        or report.get("mode") != "canary"
        or report.get("run") != run_binding
        or report.get("source", {}).get("sha256") != source_hash
        or report.get("bindings", {}).get("sha256") != bindings_hash
        or not report.get("canary_admitted")
    ):
        raise ValueError("Full validation is not admitted by this canary.")


def score_prompt(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    pad_token_id: int,
    pad_to_multiple_of: int | None,
    device: torch.device,
    core: Any,
) -> tuple[dict[str, float], dict[str, int], dict[str, Any]]:
    context, label_ids, serialization = training_label_boundary(
        tokenizer, prompt, core
    )
    first_ids = {label: ids[0] for label, ids in label_ids.items()}
    if len(context) > MAX_CONTEXT_TOKENS:
        raise ValueError("Natural-label prompt exceeds the context limit.")
    input_ids, attention_mask = core._pad_batch(
        [torch.tensor(context, dtype=torch.long)],
        pad_token_id,
        pad_to_multiple_of,
    )
    expected_mask = torch.zeros_like(attention_mask)
    expected_mask[0, : len(context)] = 1
    if (
        model.training
        or not torch.equal(input_ids[0, : len(context)], torch.tensor(context))
        or not torch.equal(attention_mask, expected_mask)
    ):
        raise ValueError(
            "Single-row evaluation state or right-padding invariant failed."
        )
    autocast_enabled = bool(torch.is_autocast_enabled())
    if autocast_enabled:
        raise ValueError("Autocast must be disabled for natural-label scoring.")
    with torch.no_grad():
        logits = model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            use_cache=False,
        ).logits
    log_probs = torch.log_softmax(logits[0, len(context) - 1].float(), dim=-1)
    if log_probs.dtype != torch.float32:
        raise ValueError("Natural-label log-softmax did not remain float32.")
    if not bool(torch.isfinite(log_probs).all().item()):
        raise ValueError("Natural-label log-softmax produced non-finite values.")
    scores = {
        label: float(log_probs[token_id].item())
        for label, token_id in first_ids.items()
    }
    float_logits = logits[0, len(context) - 1].float()
    log_normalizer = float(torch.logsumexp(float_logits, dim=-1).item())
    manually_computed = {
        label: float(float_logits[token_id].item()) - log_normalizer
        for label, token_id in first_ids.items()
    }
    hand_check = {
        **serialization,
        "model_eval_mode": not model.training,
        "autocast_enabled": autocast_enabled,
        "logits_dtype": str(logits.dtype),
        "log_softmax_dtype": str(log_probs.dtype),
        "finite_float32_reduction": bool(torch.isfinite(log_probs).all().item()),
        "padding": {
            "side": "right",
            "tokenizer_padding_side_ignored": str(tokenizer.padding_side),
            "padded_tokens": int(input_ids.shape[1]) - len(context),
            "attention_mask_exact": True,
            "position_ids_supplied": False,
            "scored_index": len(context) - 1,
        },
        "selected_raw_logits": {
            label: float(float_logits[token_id].item())
            for label, token_id in first_ids.items()
        },
        "vocabulary_logsumexp": log_normalizer,
        "manual_log_probabilities": manually_computed,
        "maximum_absolute_error": max(
            abs(manually_computed[label] - scores[label]) for label in LABELS
        ),
    }
    hand_check["pass"] = hand_check["maximum_absolute_error"] <= 1e-5
    return scores, first_ids, hand_check


def _example_key(example: dict[str, Any]) -> str:
    return f"{example['task']}#{example['doc_id']}"


def select_numeric_gate_examples(
    candidates: list[dict[str, Any]], limit: int = 9
) -> list[dict[str, Any]]:
    """Select deterministic short, long, quantile, and multilingual rows."""
    ordered = sorted(
        candidates,
        key=lambda row: (row["context_tokens"], row["task"], row["doc_id"]),
    )
    selected: dict[str, dict[str, Any]] = {}

    def add(example: dict[str, Any], reason: str) -> None:
        key = _example_key(example)
        if key not in selected:
            selected[key] = {**example, "selection_reasons": []}
        selected[key]["selection_reasons"].append(reason)

    add(ordered[0], "shortest_context")
    add(ordered[-1], "longest_context")
    for quantile in (0.25, 0.50, 0.75):
        index = round((len(ordered) - 1) * quantile)
        add(ordered[index], f"context_length_quantile_{quantile:.2f}")
    for language in LANGUAGES:
        language_rows = [
            row for row in ordered if task_parts(row["task"])[0] == language
        ]
        add(language_rows[len(language_rows) // 2], f"language_{language}")
        if len(selected) >= limit:
            break
    for example in ordered:
        if len(selected) >= limit:
            break
        add(example, "deterministic_fill")
    chosen = list(selected.values())[:limit]
    if len(chosen) < 2 or len({row["context_tokens"] for row in chosen}) < 2:
        raise ValueError("Numeric gate requires at least two distinct context lengths.")
    return chosen


def score_mixed_batch(
    *,
    model: Any,
    tokenizer: Any,
    examples: list[dict[str, Any]],
    pad_token_id: int,
    pad_to_multiple_of: int | None,
    device: torch.device,
    core: Any,
    variant: str,
) -> dict[str, Any]:
    contexts = []
    ids_by_row = []
    for example in examples:
        context, label_ids, _ = training_label_boundary(
            tokenizer, example["prompt"], core
        )
        contexts.append(context)
        ids_by_row.append({label: ids[0] for label, ids in label_ids.items()})
    input_ids, attention_mask = core._pad_batch(
        [torch.tensor(context, dtype=torch.long) for context in contexts],
        pad_token_id,
        pad_to_multiple_of,
    )
    padding_checks = []
    for row_index, context in enumerate(contexts):
        expected_mask = torch.zeros_like(attention_mask[row_index])
        expected_mask[: len(context)] = 1
        padding_checks.append(
            bool(
                torch.equal(
                    input_ids[row_index, : len(context)], torch.tensor(context)
                )
                and torch.equal(attention_mask[row_index], expected_mask)
            )
        )
    if model.training or not all(padding_checks) or torch.is_autocast_enabled():
        raise ValueError(
            "Mixed-batch evaluation state or right-padding invariant failed."
        )
    with torch.no_grad():
        logits = model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            use_cache=False,
        ).logits
    scores_by_key = {}
    reductions_finite = True
    reduction_dtypes = set()
    for row_index, (example, context) in enumerate(
        zip(examples, contexts, strict=True)
    ):
        log_probs = torch.log_softmax(
            logits[row_index, len(context) - 1].float(), dim=-1
        )
        reduction_dtypes.add(str(log_probs.dtype))
        reductions_finite = reductions_finite and bool(
            torch.isfinite(log_probs).all().item()
        )
        scores_by_key[_example_key(example)] = {
            label: float(log_probs[token_id].item())
            for label, token_id in ids_by_row[row_index].items()
        }
    if reduction_dtypes != {"torch.float32"} or not reductions_finite:
        raise ValueError("Mixed-batch float32 reduction gate failed.")
    return {
        "variant": variant,
        "batch_size": len(examples),
        "row_order": [_example_key(example) for example in examples],
        "context_token_lengths": [len(context) for context in contexts],
        "padded_sequence_tokens": int(input_ids.shape[1]),
        "has_actual_trailing_padding": any(
            len(context) < int(input_ids.shape[1]) for context in contexts
        ),
        "pad_to_multiple_of": pad_to_multiple_of,
        "padding_side": "right",
        "attention_masks_exact": all(padding_checks),
        "causal_logit_indices": [len(context) - 1 for context in contexts],
        "position_ids_supplied": False,
        "use_cache": False,
        "autocast_enabled": False,
        "model_eval_mode": not model.training,
        "logits_dtype": str(logits.dtype),
        "log_softmax_dtypes": sorted(reduction_dtypes),
        "finite_float32_reductions": reductions_finite,
        "scores": scores_by_key,
    }


def batch_single_equivalence(
    *,
    model: Any,
    tokenizer: Any,
    candidates: list[dict[str, Any]],
    pad_token_id: int,
    pad_to_multiple_of: int | None,
    device: torch.device,
    core: Any,
) -> dict[str, Any]:
    """Gate score stability by decisions and margins, not exact CUDA equality."""
    examples = select_numeric_gate_examples(candidates)
    ordered = sorted(
        examples,
        key=lambda row: (row["context_tokens"], row["task"], row["doc_id"]),
    )
    variants = [
        score_mixed_batch(
            model=model,
            tokenizer=tokenizer,
            examples=ordered,
            pad_token_id=pad_token_id,
            pad_to_multiple_of=pad_to_multiple_of,
            device=device,
            core=core,
            variant="mixed_length_ascending",
        ),
        score_mixed_batch(
            model=model,
            tokenizer=tokenizer,
            examples=list(reversed(ordered)),
            pad_token_id=pad_token_id,
            pad_to_multiple_of=pad_to_multiple_of,
            device=device,
            core=core,
            variant="mixed_length_reversed",
        ),
    ]
    raw_chunk_size = getattr(getattr(model, "config", None), "chunk_size", None)
    kernel_chunk_size = (
        raw_chunk_size
        if isinstance(raw_chunk_size, int) and not isinstance(raw_chunk_size, bool)
        else None
    )
    if kernel_chunk_size and kernel_chunk_size != pad_to_multiple_of:
        variants.append(
            score_mixed_batch(
                model=model,
                tokenizer=tokenizer,
                examples=ordered,
                pad_token_id=pad_token_id,
                pad_to_multiple_of=kernel_chunk_size,
                device=device,
                core=core,
                variant="kernel_chunk_padded",
            )
        )

    repeated_single_scores = {}
    for example in examples:
        scores, _, _ = score_prompt(
            model=model,
            tokenizer=tokenizer,
            prompt=example["prompt"],
            pad_token_id=pad_token_id,
            pad_to_multiple_of=pad_to_multiple_of,
            device=device,
            core=core,
        )
        repeated_single_scores[_example_key(example)] = scores

    row_diagnostics = []
    maximum_difference = {
        "absolute_error": 0.0,
        "example": None,
        "label": None,
        "variant": None,
        "affects_single_top_two": None,
    }
    argmax_changes = 0
    for example in examples:
        key = _example_key(example)
        single_scores = example["single_scores"]
        ranked = sorted(LABELS, key=single_scores.__getitem__, reverse=True)
        single_prediction = ranked[0]
        single_margin = single_scores[ranked[0]] - single_scores[ranked[1]]
        comparisons = {
            variant["variant"]: variant["scores"][key] for variant in variants
        }
        comparisons["repeat_single_after_batches"] = repeated_single_scores[key]
        epsilon = 0.0
        top_two_epsilon = 0.0
        comparison_predictions = {}
        for variant_name, scores in comparisons.items():
            variant_prediction = prediction(scores)[0]
            comparison_predictions[variant_name] = variant_prediction
            if variant_prediction != single_prediction:
                argmax_changes += 1
            for label in LABELS:
                difference = abs(scores[label] - single_scores[label])
                epsilon = max(epsilon, difference)
                if label in ranked[:2]:
                    top_two_epsilon = max(top_two_epsilon, difference)
                if difference > maximum_difference["absolute_error"]:
                    maximum_difference = {
                        "absolute_error": difference,
                        "example": key,
                        "label": label,
                        "variant": variant_name,
                        "affects_single_top_two": label in ranked[:2],
                    }
        exact_argmax_invariance = all(
            value == single_prediction for value in comparison_predictions.values()
        )
        margin_safe = single_margin > 2 * epsilon + TIE_EPSILON
        row_diagnostics.append(
            {
                "example": key,
                "task": example["task"],
                "doc_id": example["doc_id"],
                "selection_reasons": example["selection_reasons"],
                "context_tokens": example["context_tokens"],
                "single_prediction": single_prediction,
                "single_winning_margin": single_margin,
                "maximum_absolute_score_difference": epsilon,
                "maximum_top_two_score_difference": top_two_epsilon,
                "margin_safety_rule": (
                    "single_winning_margin > 2 * row_epsilon + tie_epsilon"
                ),
                "margin_safe": margin_safe,
                "comparison_predictions": comparison_predictions,
                "exact_argmax_invariance": exact_argmax_invariance,
            }
        )

    all_argmax_invariant = all(
        row["exact_argmax_invariance"] for row in row_diagnostics
    )
    all_margin_safe = all(row["margin_safe"] for row in row_diagnostics)
    batch_structure_pass = all(
        variant["attention_masks_exact"]
        and variant["has_actual_trailing_padding"]
        and variant["finite_float32_reductions"]
        and variant["model_eval_mode"]
        and not variant["autocast_enabled"]
        for variant in variants
    )
    candidate_chunk_buckets = (
        sorted(
            {
                math.ceil(row["context_tokens"] / kernel_chunk_size)
                for row in candidates
            }
        )
        if kernel_chunk_size
        else []
    )
    selected_chunk_buckets = (
        sorted(
            {
                math.ceil(row["context_tokens"] / kernel_chunk_size)
                for row in examples
            }
        )
        if kernel_chunk_size
        else []
    )
    chunk_boundary_available = len(candidate_chunk_buckets) > 1
    chunk_boundary_exercised = (
        not chunk_boundary_available or len(selected_chunk_buckets) > 1
    )
    return {
        "rows": len(examples),
        "comparison": (
            "single-example versus mixed-length, reordered, repeated-single, and "
            "available kernel-chunk-padded float32 first-token scoring"
        ),
        "central_evaluator_batch_size": 1,
        "diagnostic_batch_size": len(examples),
        "single_context_token_range": [
            min(row["context_tokens"] for row in examples),
            max(row["context_tokens"] for row in examples),
        ],
        "kernel_chunk_size": kernel_chunk_size,
        "kernel_chunk_boundary_available": chunk_boundary_available,
        "candidate_kernel_chunk_buckets": candidate_chunk_buckets,
        "selected_kernel_chunk_buckets": selected_chunk_buckets,
        "kernel_chunk_boundary_exercised": chunk_boundary_exercised,
        "padding_contract": (
            "scorer-owned right padding; prefix tokens at indices 0..length-1; "
            "ones over prefix and zeros over trailing pad; score index length-1"
        ),
        "state_reset_check": {
            "use_cache": False,
            "past_cache_or_state_supplied": False,
            "repeated_each_single_after_all_batch_variants": True,
            "repeat_argmax_invariant": all(
                row["comparison_predictions"]["repeat_single_after_batches"]
                == row["single_prediction"]
                for row in row_diagnostics
            ),
        },
        "number_of_argmax_changes": argmax_changes,
        "minimum_single_winning_margin": min(
            row["single_winning_margin"] for row in row_diagnostics
        ),
        "maximum_absolute_score_difference": maximum_difference,
        "all_argmax_invariant": all_argmax_invariant,
        "all_rows_margin_safe": all_margin_safe,
        "batch_structure_pass": batch_structure_pass,
        "variants": [
            {key: value for key, value in variant.items() if key != "scores"}
            for variant in variants
        ],
        "row_diagnostics": row_diagnostics,
        "pass": bool(
            batch_structure_pass
            and chunk_boundary_exercised
            and all_argmax_invariant
            and all_margin_safe
        ),
    }


def prediction(scores: dict[str, float]) -> tuple[str, float, bool]:
    ranked = sorted(LABELS, key=lambda label: scores[label], reverse=True)
    margin = scores[ranked[0]] - scores[ranked[1]]
    return ranked[0], margin, margin <= TIE_EPSILON


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


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gold = [str(row["gold"]) for row in rows]
    predicted = [str(row["prediction"]) for row in rows]
    evaluator = ClassificationEvaluator.__new__(ClassificationEvaluator)
    metrics = evaluator._compute_classification_metrics(gold, predicted)
    gold_counts = Counter(gold)
    prediction_counts = Counter(predicted)
    confusion = {
        actual: {
            guessed: sum(
                target == actual and prediction_value == guessed
                for target, prediction_value in zip(gold, predicted, strict=True)
            )
            for guessed in LABELS
        }
        for actual in LABELS
    }
    probabilities = [count / len(rows) for count in prediction_counts.values()]
    entropy = -sum(value * math.log(value) for value in probabilities)
    summary = {
        "n": len(rows),
        "metrics": metrics,
        "gold_counts": {label: gold_counts[label] for label in LABELS},
        "prediction_counts": {label: prediction_counts[label] for label in LABELS},
        "confusion": confusion,
        "unique_predictions": len(prediction_counts),
        "dominant_prediction_share": max(prediction_counts.values()) / len(rows),
        "normalized_prediction_entropy": entropy / math.log(len(LABELS)),
        "effective_prediction_classes": math.exp(entropy),
        "top_score_ties": sum(bool(row["tie"]) for row in rows),
    }
    summary["gate"] = nondegeneracy_gate(summary)
    return summary


def task_parts(task: str) -> tuple[str, int]:
    prefix, separator = "sallm_sib_", "_val_prompt_"
    language, prompt = task.removeprefix(prefix).split(separator, 1)
    if language not in LANGUAGES or int(prompt) not in PROMPTS:
        raise ValueError(f"Unexpected task {task!r}.")
    return language, int(prompt)


def run_score(args: argparse.Namespace) -> None:
    if args.architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture {args.architecture!r}.")
    core = load_core(args.validation_core)
    source = core.load_validation_source(args.source_results)
    source_hash = sha256(args.source_results)
    bindings_hash = sha256(args.bindings)
    audit = validate_audit(
        args.token_audit, args.architecture, source_hash, bindings_hash
    )
    run_binding = {
        "architecture": args.architecture,
        "checkpoint": args.checkpoint,
        "adapter": args.adapter,
        "dtype": args.dtype,
        "device": args.device,
        "merge_lora": args.merge_lora,
        "tie_word_embeddings": args.tie_word_embeddings,
    }
    if not args.canary:
        if args.admit_from_canary is None:
            raise ValueError("Full validation requires --admit-from-canary.")
        validate_canary(
            args.admit_from_canary, run_binding, source_hash, bindings_hash
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
    tokenizer_json = Path(audit["path"]) / "tokenizer.json"
    if sha256(tokenizer_json) != audit["tokenizer_json_sha256"]:
        raise ValueError("Loaded tokenizer differs from the passing audit.")
    loaded_template_hash = hashlib.sha256(
        str(tokenizer.chat_template).encode("utf-8")
    ).hexdigest()
    if tokenizer_identity_sha256(tokenizer) != audit.get(
        "in_memory_tokenizer_identity_sha256"
    ):
        raise ValueError("Loaded tokenizer identity differs from the passing audit.")
    if loaded_template_hash != audit["serialization"]["chat_template_sha256"]:
        raise ValueError("Loaded chat template differs from the passing audit.")
    model.eval()
    device = model.device
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer has neither PAD nor EOS token.")
    pad_multiple = ClassificationEvaluator._get_model_chunk_size(model)
    rows_by_task = core.selected_rows(source, canary=args.canary)

    scoring_started = time.monotonic()
    output_rows = []
    equivalence_candidates: list[dict[str, Any]] = []
    hand_computed_example = None
    for task in sorted(rows_by_task):
        for row in rows_by_task[task]:
            prompt = str(row["arguments"][0][0])
            scores, first_ids, hand_check = score_prompt(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                pad_token_id=int(pad_token_id),
                pad_to_multiple_of=pad_multiple,
                device=device,
                core=core,
            )
            if hand_computed_example is None:
                hand_computed_example = {
                    "task": task,
                    "doc_id": row["doc_id"],
                    **hand_check,
                }
            guess, margin, tie = prediction(scores)
            equivalence_candidates.append(
                {
                    "task": task,
                    "doc_id": row["doc_id"],
                    "prompt": prompt,
                    "context_tokens": hand_check["training_label_prefix_tokens"],
                    "single_scores": scores,
                }
            )
            output_rows.append(
                {
                    "task": task,
                    "doc_id": row["doc_id"],
                    "doc_hash": row["doc_hash"],
                    "prompt_hash": row["prompt_hash"],
                    "target_hash": row["target_hash"],
                    "gold": str(row["target"]).strip(),
                    "prediction": guess,
                    "margin": margin,
                    "tie": tie,
                    "scores": scores,
                    "first_token_ids": first_ids,
                }
            )
    scoring_seconds = time.monotonic() - scoring_started
    equivalence = (
        batch_single_equivalence(
            model=model,
            tokenizer=tokenizer,
            candidates=equivalence_candidates,
            pad_token_id=int(pad_token_id),
            pad_to_multiple_of=pad_multiple,
            device=device,
            core=core,
        )
        if args.canary
        else {"pass": True, "source": "admitted canary", "rows": 0}
    )
    expected_keys = {
        (task, row["doc_id"])
        for task, rows in rows_by_task.items()
        for row in rows
    }
    observed_keys = {(row["task"], row["doc_id"]) for row in output_rows}
    if len(output_rows) != len(expected_keys) or observed_keys != expected_keys:
        raise ValueError("Natural-label scoring failed row completeness.")

    tasks = {
        task: summarize([row for row in output_rows if row["task"] == task])
        for task in sorted(rows_by_task)
    }
    all_summary = summarize(output_rows)
    language_summaries = {
        language: summarize(
            [row for row in output_rows if task_parts(row["task"])[0] == language]
        )
        for language in LANGUAGES
    }
    language_gates = {
        language: summary["gate"] for language, summary in language_summaries.items()
    }
    report = {
        "schema": "sallm.sib_natural_first_token_eval/v1",
        "mode": "canary" if args.canary else "full",
        "split": "validation",
        "source": {"path": str(args.source_results), "sha256": source_hash},
        "bindings": {"path": str(args.bindings), "sha256": bindings_hash},
        "token_audit": {
            "path": str(args.token_audit),
            "sha256": sha256(args.token_audit),
        },
        "validation_core": {
            "path": str(args.validation_core),
            "sha256": VALIDATION_CORE_SHA256,
        },
        "admission_canary": (
            {
                "path": str(args.admit_from_canary),
                "sha256": sha256(args.admit_from_canary),
            }
            if not args.canary
            else None
        ),
        "run": run_binding,
        "scorer": {
            "labels": list(LABELS),
            "serialization": (
                "adapter-saved training chat template; user task prompt plus "
                "assistant boundary and its static content whitespace"
            ),
            "continuation": (
                "exact natural label at the trained assistant-content boundary"
            ),
            "score": (
                "float32 log probability of the first label-discriminative target "
                "token only"
            ),
            "causal_logit_position": "final token of the exact training-style prefix",
            "tie_epsilon": TIE_EPSILON,
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
            "central_batch_size": 1,
            "model_eval_mode": not model.training,
            "autocast_enabled": bool(torch.is_autocast_enabled()),
            "model_parameter_dtypes": sorted(
                {str(parameter.dtype) for parameter in model.parameters()}
            ),
            "model_config_dtype": str(getattr(model.config, "dtype", None)),
            "model_config_chunk_size": getattr(model.config, "chunk_size", None),
            "tokenizer_padding_side": str(tokenizer.padding_side),
            "position_ids_supplied": False,
            "use_cache": False,
        },
        "completeness_gate": {
            "pass": True,
            "expected_rows": len(expected_keys),
            "observed_rows": len(output_rows),
            "unique_task_doc_ids": len(observed_keys),
        },
        "hand_computed_example": hand_computed_example,
        "batch_single_equivalence": equivalence,
        "natural_first_token": {
            "all": all_summary,
            "tasks": tasks,
            "languages": language_summaries,
            "language_gates": language_gates,
        },
        "rows": output_rows,
        "heldout_action": "none; validation only",
    }
    report["canary_admitted"] = bool(
        args.canary
        and all_summary["top_score_ties"] == 0
        and hand_computed_example is not None
        and hand_computed_example["pass"]
        and equivalence["pass"]
    )
    write_report(args.output, report)
    if args.canary and not report["canary_admitted"]:
        raise SystemExit("Canary failed closed; full validation is not admitted.")


def run_select(args: argparse.Namespace) -> None:
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.report]
    by_architecture = {
        report.get("run", {}).get("architecture"): report for report in reports
    }
    if set(by_architecture) != set(ARCHITECTURES) or len(reports) != len(ARCHITECTURES):
        raise ValueError("Selection requires one full report per architecture.")
    source_hashes = {report.get("source", {}).get("sha256") for report in reports}
    bindings_hashes = {report.get("bindings", {}).get("sha256") for report in reports}
    for report in reports:
        if (
            report.get("schema") != "sallm.sib_natural_first_token_eval/v1"
            or report.get("mode") != "full"
            or report.get("split") != "validation"
            or not report.get("completeness_gate", {}).get("pass")
        ):
            raise ValueError("Selection accepts complete full validation reports only.")
    if len(source_hashes) != 1 or len(bindings_hashes) != 1:
        raise ValueError("Full reports do not share source/binding hashes.")

    selected = {}
    diagnostics = {}
    for language in LANGUAGES:
        diagnostics[language] = {}
        eligible = []
        for prompt in PROMPTS:
            task = f"sallm_sib_{language}_val_prompt_{prompt}"
            cells = {
                architecture: by_architecture[architecture]["natural_first_token"][
                    "tasks"
                ][task]
                for architecture in ARCHITECTURES
            }
            passed = all(cell["top_score_ties"] == 0 for cell in cells.values())
            mean_f1 = fmean(cell["metrics"]["f1"] for cell in cells.values())
            diagnostics[language][str(prompt)] = {
                "eligible": passed,
                "mean_support_weighted_f1": mean_f1,
                "architectures": {
                    architecture: {
                        "metrics": cell["metrics"],
                        "prediction_counts": cell["prediction_counts"],
                        "nondegeneracy_diagnostic": cell["gate"],
                        "top_score_ties": cell["top_score_ties"],
                    }
                    for architecture, cell in cells.items()
                },
            }
            if passed:
                eligible.append((prompt, mean_f1))
        if eligible:
            prompt, score = max(eligible, key=lambda item: (item[1], -item[0]))
            selected[language] = {
                "prompt": prompt,
                "task": f"sallm_sib_{language}_val_prompt_{prompt}",
                "validation_mean_support_weighted_f1": score,
            }
        else:
            selected[language] = None
    output = {
        "schema": "sallm.sib_natural_first_token_selection/v1",
        "split": "validation",
        "selection_rule": (
            "Per language, retain prompts with no numerical top-score ties for all "
            "four architectures; maximize unweighted mean support-weighted F1; "
            "break ties by lowest prompt number. Non-degeneracy is reported but is "
            "not a validity veto."
        ),
        "source_sha256": source_hashes.pop(),
        "bindings_sha256": bindings_hashes.pop(),
        "reports": [
            {"path": str(path), "sha256": sha256(path)} for path in args.report
        ],
        "selected": selected,
        "prompt_diagnostics": diagnostics,
        "ready_for_generation_cross_check": all(selected.values()),
        "heldout_action": "none; validation only",
    }
    write_report(args.output, output)
    if not output["ready_for_generation_cross_check"]:
        raise SystemExit("Prompt selection failed closed.")


def greedy_generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    core: Any,
    max_new_tokens: int,
) -> tuple[str, list[int], list[int]]:
    generated: list[int] = []
    non_eos_special_ids: list[int] = []
    context_ids, _, _ = training_label_boundary(tokenizer, prompt, core)
    sequence = list(context_ids)
    eos_id = tokenizer.eos_token_id
    special_ids = {int(token_id) for token_id in tokenizer.all_special_ids}
    for _ in range(max_new_tokens):
        input_ids = torch.tensor([sequence], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits
        token_id = int(torch.argmax(logits[0, -1].float()).item())
        if eos_id is not None and token_id == eos_id:
            break
        generated.append(token_id)
        if token_id in special_ids:
            non_eos_special_ids.append(token_id)
            break
        sequence.append(token_id)
    return (
        tokenizer.decode(generated, skip_special_tokens=False).strip(),
        generated,
        non_eos_special_ids,
    )


def summarize_generation_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid_rows = [row for row in rows if row["valid_exact_label"]]
    special_rows = [row for row in rows if row["non_eos_special_token_ids"]]
    return {
        "n": len(rows),
        "valid_exact_labels": len(valid_rows),
        "valid_exact_label_rate": len(valid_rows) / len(rows),
        "invalid_outputs": len(rows) - len(valid_rows),
        "outputs_with_non_eos_special_tokens": len(special_rows),
        "strict_accuracy": sum(row["correct"] for row in rows) / len(rows),
        "exact_label_agreement_with_first_token": (
            sum(row["agrees_with_first_token"] for row in rows) / len(rows)
        ),
        "agreement_conditional_on_valid": (
            sum(row["agrees_with_first_token"] for row in valid_rows)
            / len(valid_rows)
            if valid_rows
            else None
        ),
    }


def run_generation(args: argparse.Namespace) -> None:
    core = load_core(args.validation_core)
    source = core.load_validation_source(args.source_results)
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    first_token_report = json.loads(args.first_token_report.read_text(encoding="utf-8"))
    token_audit = json.loads(args.token_audit.read_text(encoding="utf-8"))
    run_binding = {
        "architecture": args.architecture,
        "checkpoint": args.checkpoint,
        "adapter": args.adapter,
        "dtype": args.dtype,
        "device": args.device,
        "merge_lora": args.merge_lora,
        "tie_word_embeddings": args.tie_word_embeddings,
    }
    if (
        selection.get("schema") != "sallm.sib_natural_first_token_selection/v1"
        or not selection.get("ready_for_generation_cross_check")
        or first_token_report.get("schema") != "sallm.sib_natural_first_token_eval/v1"
        or first_token_report.get("mode") != "full"
        or first_token_report.get("split") != "validation"
        or not first_token_report.get("completeness_gate", {}).get("pass")
        or first_token_report.get("run", {}).get("architecture") != args.architecture
        or first_token_report.get("run") != run_binding
        or selection.get("source_sha256")
        != first_token_report.get("source", {}).get("sha256")
        or selection.get("bindings_sha256")
        != first_token_report.get("bindings", {}).get("sha256")
        or selection.get("source_sha256") != sha256(args.source_results)
        or not any(
            report.get("sha256") == sha256(args.first_token_report)
            for report in selection.get("reports", [])
        )
        or token_audit.get("schema") != "sallm.sib_natural_first_token_audit/v1"
        or not token_audit.get("pass")
        or token_audit.get("source", {}).get("sha256")
        != sha256(args.source_results)
        or first_token_report.get("token_audit", {}).get("sha256")
        != sha256(args.token_audit)
    ):
        raise ValueError("Generation cross-check inputs are not admitted.")
    architecture_audit = token_audit["architectures"][args.architecture]
    maximum_label_tokens = max(
        max(lengths)
        for lengths in architecture_audit["full_label_token_lengths"].values()
    )
    if args.max_new_tokens < maximum_label_tokens + 1:
        raise ValueError("Generation cap cannot emit the longest label plus EOS.")
    row_lookup = {
        (row["task"], row["doc_id"]): row for row in first_token_report["rows"]
    }
    selected_rows = []
    for language in LANGUAGES:
        task = selection["selected"][language]["task"]
        by_label = {}
        for row in source["samples"][task]:
            by_label.setdefault(str(row["target"]).strip(), row)
        if set(by_label) != set(LABELS):
            raise ValueError(f"{language}: cannot form label-balanced cross-check.")
        selected_rows.extend((task, by_label[label]) for label in LABELS)

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
    if tokenizer_identity_sha256(tokenizer) != architecture_audit.get(
        "in_memory_tokenizer_identity_sha256"
    ):
        raise ValueError("Generation tokenizer identity differs from the audit.")
    model.eval()
    if model.training or torch.is_autocast_enabled():
        raise ValueError("Generation cross-check requires eval mode without autocast.")
    device = model.device
    rows = []
    scoring_started = time.monotonic()
    for task, source_row in selected_rows:
        prompt = str(source_row["arguments"][0][0])
        generated_text, generated_ids, non_eos_special_ids = greedy_generate(
            model,
            tokenizer,
            prompt,
            device,
            core,
            args.max_new_tokens,
        )
        reference = row_lookup[(task, source_row["doc_id"])]
        exact_label = (
            generated_text
            if not non_eos_special_ids and generated_text in LABELS
            else None
        )
        rows.append(
            {
                "task": task,
                "doc_id": source_row["doc_id"],
                "doc_hash": source_row["doc_hash"],
                "prompt_hash": source_row["prompt_hash"],
                "target_hash": source_row["target_hash"],
                "gold": str(source_row["target"]).strip(),
                "first_token_prediction": reference["prediction"],
                "generated_text": generated_text,
                "generated_token_ids": generated_ids,
                "non_eos_special_token_ids": non_eos_special_ids,
                "parsed_exact_label": exact_label,
                "valid_exact_label": exact_label is not None,
                "correct": exact_label == str(source_row["target"]).strip(),
                "agrees_with_first_token": exact_label == reference["prediction"],
            }
        )
    elapsed = time.monotonic() - scoring_started
    output = {
        "schema": "sallm.sib_natural_label_greedy_cross_check/v1",
        "split": "validation",
        "scope": "one deterministic row per gold class per selected language prompt",
        "strict_parser": "stripped decoded output must exactly equal one natural label",
        "decode_special_tokens": (
            "non-EOS special tokens are retained in decoded text, reported, and "
            "make the output invalid"
        ),
        "invalid_outputs_counted_wrong": True,
        "source": {
            "path": str(args.source_results),
            "sha256": sha256(args.source_results),
        },
        "selection": {"path": str(args.selection), "sha256": sha256(args.selection)},
        "first_token_report": {
            "path": str(args.first_token_report),
            "sha256": sha256(args.first_token_report),
        },
        "token_audit": {
            "path": str(args.token_audit),
            "sha256": sha256(args.token_audit),
        },
        "run": {**run_binding, "max_new_tokens": args.max_new_tokens},
        "runtime": {"load_seconds": load_seconds, "generation_seconds": elapsed},
        "completeness_gate": {
            "pass": len(rows) == len(LANGUAGES) * len(LABELS)
            and len({(row["task"], row["doc_id"]) for row in rows}) == len(rows),
            "expected_rows": len(LANGUAGES) * len(LABELS),
            "observed_rows": len(rows),
            "unique_task_doc_ids": len(
                {(row["task"], row["doc_id"]) for row in rows}
            ),
        },
        "summary": summarize_generation_rows(rows),
        "languages": {
            language: summarize_generation_rows(
                [
                    row
                    for row in rows
                    if task_parts(row["task"])[0] == language
                ]
            )
            for language in LANGUAGES
        },
        "rows": rows,
        "heldout_action": "none; validation only",
        "review_gate": (
            "diagnostic only; valid closed-set scoring is not vetoed by model "
            "generation errors. Serialization, tokenizer identity, completeness, "
            "and special-token handling must pass before held-out admission"
        ),
    }
    if not output["completeness_gate"]["pass"]:
        raise ValueError("Generation cross-check failed row completeness.")
    write_report(args.output, output)


def self_check() -> None:
    balanced = [
        {"gold": label, "prediction": label, "tie": False} for label in LABELS
    ]
    assert summarize(balanced)["gate"]["pass"]
    collapsed = [
        {"gold": label, "prediction": LABELS[0], "tie": False} for label in LABELS
    ]
    assert not summarize(collapsed)["gate"]["pass"]
    winner, margin, tie = prediction(
        {label: float(index) for index, label in enumerate(LABELS)}
    )
    assert winner == LABELS[-1]
    assert margin == 1.0
    assert not tie
    candidates = [
        {
            "task": f"sallm_sib_{LANGUAGES[index % len(LANGUAGES)]}_val_prompt_1",
            "doc_id": index,
            "prompt": f"prompt-{index}",
            "context_tokens": 50 + index * 9,
            "single_scores": {
                label: float(label_index)
                for label_index, label in enumerate(LABELS)
            },
        }
        for index in range(12)
    ]
    selected = select_numeric_gate_examples(candidates)
    selected_lengths = {row["context_tokens"] for row in selected}
    assert min(selected_lengths) == 50
    assert max(selected_lengths) == 149
    assert len(selected) == 9
    generation_rows = [
        {
            "valid_exact_label": True,
            "non_eos_special_token_ids": [],
            "correct": True,
            "agrees_with_first_token": True,
        },
        {
            "valid_exact_label": False,
            "non_eos_special_token_ids": [42],
            "correct": False,
            "agrees_with_first_token": False,
        },
    ]
    generation_summary = summarize_generation_rows(generation_rows)
    assert generation_summary["valid_exact_label_rate"] == 0.5
    assert generation_summary["outputs_with_non_eos_special_tokens"] == 1
    assert generation_summary["agreement_conditional_on_valid"] == 1.0
    print("NATURAL_FIRST_TOKEN_SELF_CHECK_OK")


def add_model_args(command: argparse.ArgumentParser) -> None:
    command.add_argument("--architecture", required=True)
    command.add_argument("--checkpoint", required=True)
    command.add_argument("--adapter", required=True)
    command.add_argument("--dtype", default="float32")
    command.add_argument("--device", default="cuda:0")
    command.add_argument(
        "--merge-lora", action=argparse.BooleanOptionalAction, default=None
    )
    command.add_argument(
        "--tie-word-embeddings", action=argparse.BooleanOptionalAction, default=None
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("self-check")

    audit = commands.add_parser("audit-tokenizers")
    audit.add_argument("--source-results", type=Path, required=True)
    audit.add_argument("--bindings", type=Path, required=True)
    audit.add_argument("--validation-core", type=Path, required=True)
    audit.add_argument("--tokenizer", action="append", required=True)
    audit.add_argument("--output", type=Path, required=True)

    score = commands.add_parser("score")
    score.add_argument("--source-results", type=Path, required=True)
    score.add_argument("--bindings", type=Path, required=True)
    score.add_argument("--validation-core", type=Path, required=True)
    score.add_argument("--token-audit", type=Path, required=True)
    add_model_args(score)
    score.add_argument("--canary", action="store_true")
    score.add_argument("--admit-from-canary", type=Path)
    score.add_argument("--output", type=Path, required=True)

    select = commands.add_parser("select")
    select.add_argument("--report", action="append", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)

    generation = commands.add_parser("generation-cross-check")
    generation.add_argument("--source-results", type=Path, required=True)
    generation.add_argument("--validation-core", type=Path, required=True)
    generation.add_argument("--selection", type=Path, required=True)
    generation.add_argument("--first-token-report", type=Path, required=True)
    generation.add_argument("--token-audit", type=Path, required=True)
    add_model_args(generation)
    generation.add_argument("--max-new-tokens", type=int, default=8)
    generation.add_argument("--output", type=Path, required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "self-check":
        self_check()
    elif args.command == "audit-tokenizers":
        run_audit(args)
    elif args.command == "score":
        run_score(args)
    elif args.command == "select":
        run_select(args)
    elif args.command == "generation-cross-check":
        run_generation(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
