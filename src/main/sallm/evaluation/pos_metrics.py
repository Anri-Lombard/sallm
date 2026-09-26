"""Validation-only constrained MasakhaPOS selection metrics."""

from __future__ import annotations

import ast
import logging
import os
from collections import defaultdict
from typing import Any, cast

import torch
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from sallm.evaluation.constrained_label_scoring import decode_pos_tuple_contract
from sallm.evaluation.pos_batched_scoring import decode_pos_tuple_contract_batch

logger = logging.getLogger(__name__)

UPOS_LABELS = [
    "ADJ",
    "ADP",
    "ADV",
    "AUX",
    "CCONJ",
    "DET",
    "INTJ",
    "NOUN",
    "NUM",
    "PART",
    "PRON",
    "PROPN",
    "PUNCT",
    "SCONJ",
    "SYM",
    "VERB",
    "X",
]
CANONICAL_POS_LANGUAGES = {"tsn", "xho", "zul"}
CANONICAL_POS_TEMPLATES = {
    "masakhane_pos_tagging/lm_eval_p1",
    "masakhane_pos_tagging/lm_eval_p2",
    "masakhane_pos_tagging/lm_eval_p3",
    "masakhane_pos_tagging/lm_eval_p4",
}


def parse_pos_tuple_completion(completion: str) -> tuple[list[str], list[str]]:
    try:
        parsed = ast.literal_eval(completion)
    except (SyntaxError, ValueError) as exc:
        raise ValueError("POS completion is not a valid tuple list.") from exc
    if not isinstance(parsed, list):
        raise ValueError("POS completion must be a list.")

    tokens: list[str] = []
    labels: list[str] = []
    for item in parsed:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError(
                "Every POS completion item must be a (token, label) tuple."
            )
        token, label = str(item[0]), str(item[1]).upper()
        if label not in UPOS_LABELS:
            raise ValueError(f"Illegal UPOS label {label!r} in reference completion.")
        tokens.append(token)
        labels.append(label)
    if not tokens:
        raise ValueError("POS completion contains no token-label tuples.")
    return tokens, labels


def aggregate_pos_cell_accuracies(
    cell_counts: dict[tuple[str, str], tuple[int, int]],
    *,
    strict_contract: bool = True,
    expected_cells: set[tuple[str, str]] | None = None,
) -> tuple[float, dict[str, float]]:
    observed_cells = set(cell_counts)
    if expected_cells is None:
        expected_cells = {
            (language, template)
            for language in CANONICAL_POS_LANGUAGES
            for template in CANONICAL_POS_TEMPLATES
        }
    if strict_contract and observed_cells != expected_cells:
        missing = sorted(expected_cells - observed_cells)
        extra = sorted(observed_cells - expected_cells)
        raise ValueError(f"POS coverage mismatch; missing={missing}, extra={extra}")

    cell_metrics: dict[str, float] = {}
    accuracies: list[float] = []
    for (language, template), (correct, total) in sorted(cell_counts.items()):
        if total <= 0:
            raise ValueError(f"POS cell {language}/{template} has no scored tokens.")
        accuracy = correct / total
        prompt = template.rsplit("/", 1)[-1]
        cell_metrics[f"{language}_{prompt}_token_accuracy"] = accuracy
        accuracies.append(accuracy)
    if not accuracies:
        raise ValueError("POS evaluation produced no language-prompt cells.")
    return sum(accuracies) / len(accuracies), cell_metrics


class PosEvaluator:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        score_mode: str = "mean",
        pad_to_multiple_of: int | None = 64,
        strict_contract: bool = True,
        row_batch_size: int | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.score_mode = score_mode
        self.pad_to_multiple_of = pad_to_multiple_of
        self.strict_contract = strict_contract
        self.row_batch_size = (
            int(os.getenv("SALLM_POS_ROW_BATCH_SIZE", "1"))
            if row_batch_size is None
            else row_batch_size
        )
        if self.row_batch_size < 1:
            raise ValueError("POS row batch size must be at least one.")
        self.last_details: dict[str, Any] = {}

    def evaluate(
        self,
        model: PreTrainedModel,
        dataset: Dataset,
        metric_prefix: str = "eval",
    ) -> dict[str, float]:
        required = {"messages", "lang", "template_id"}
        missing = required - set(dataset.column_names)
        if missing:
            raise ValueError(
                f"POS evaluation dataset is missing columns {sorted(missing)}"
            )

        device = getattr(model, "device", torch.device("cpu"))
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("POS scoring requires pad_token_id or eos_token_id.")

        if hasattr(model, "eval"):
            model.eval()
        counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
        sequence_count = 0
        for batch_start in range(0, len(dataset), self.row_batch_size):
            samples = [
                dataset[index]
                for index in range(
                    batch_start,
                    min(batch_start + self.row_batch_size, len(dataset)),
                )
            ]
            parsed: list[tuple[list[dict[str, str]], list[str], list[str]]] = []
            for offset, sample in enumerate(samples):
                messages = cast(list[dict[str, str]], sample["messages"])
                if len(messages) < 2 or messages[-1].get("role") != "assistant":
                    raise ValueError(
                        f"POS row {batch_start + offset} has an invalid message "
                        "contract."
                    )
                tokens, gold = parse_pos_tuple_completion(messages[-1]["content"])
                parsed.append((messages, tokens, gold))

            if self.row_batch_size == 1:
                messages, tokens, _ = parsed[0]
                results = [
                    decode_pos_tuple_contract(
                        model=model,
                        tokenizer=self.tokenizer,
                        prompt_messages=messages[:-1],
                        tokens=tokens,
                        labels=UPOS_LABELS,
                        score_mode=self.score_mode,
                        pad_token_id=int(pad_token_id),
                        pad_to_multiple_of=self.pad_to_multiple_of,
                        device=device,
                    )
                ]
            else:
                results = decode_pos_tuple_contract_batch(
                    model=model,
                    tokenizer=self.tokenizer,
                    prompt_messages_batch=[messages[:-1] for messages, _, _ in parsed],
                    tokens_batch=[tokens for _, tokens, _ in parsed],
                    labels=UPOS_LABELS,
                    score_mode=self.score_mode,
                    pad_token_id=int(pad_token_id),
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    device=device,
                )

            for offset, ((_, tokens, gold), (predictions, _)) in enumerate(
                zip(parsed, results, strict=True)
            ):
                if len(predictions) != len(tokens) or any(
                    prediction not in UPOS_LABELS for prediction in predictions
                ):
                    raise ValueError(
                        f"POS row {batch_start + offset} did not produce exactly "
                        "one legal label per token."
                    )
                correct = sum(
                    prediction == reference
                    for prediction, reference in zip(predictions, gold, strict=True)
                )
                sample = samples[offset]
                cell = (str(sample["lang"]), str(sample["template_id"]))
                counts[cell][0] += correct
                counts[cell][1] += len(gold)
                sequence_count += 1
                if sequence_count % 50 == 0:
                    logger.info(
                        "Constrained POS validation: %d/%d rows",
                        sequence_count,
                        len(dataset),
                    )

        immutable_counts = {
            cell: (values[0], values[1]) for cell, values in counts.items()
        }
        expected_cells = {
            (str(language), str(template))
            for language in set(dataset["lang"])
            for template in set(dataset["template_id"])
        }
        all_accuracy, cell_metrics = aggregate_pos_cell_accuracies(
            immutable_counts,
            strict_contract=self.strict_contract,
            expected_cells=expected_cells,
        )
        self.last_details = {
            "protocol": "closed_label_tuple_mean_logprob_v1",
            "rows": sequence_count,
            "label_set": list(UPOS_LABELS),
            "score_mode": self.score_mode,
            "scoring_implementation": (
                "row_batch_full_prefix_v1"
                if self.row_batch_size > 1
                else "full_prefix_v1"
            ),
            "row_batch_size": self.row_batch_size,
            "cells": {
                f"{language}/{template}": {"correct": correct, "total": total}
                for (language, template), (correct, total) in sorted(
                    immutable_counts.items()
                )
            },
            "all_token_accuracy": all_accuracy,
        }
        metrics = {
            f"{metric_prefix}/{name}": value for name, value in cell_metrics.items()
        }
        metrics[f"{metric_prefix}/all_token_accuracy"] = all_accuracy
        return metrics
