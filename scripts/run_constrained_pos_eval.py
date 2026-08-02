#!/usr/bin/env python3
"""Constrained MasakhaPOS test-set evaluator for decoder-only models.

The primary POS result here is closed-label token accuracy. The model still
uses its decoder-only prompt, but each output position is forced to choose one
UPOS label from the fixed tag set. This avoids free-form generation artifacts
while preserving the output contract the adapter was trained on.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from constrained_label_scoring import (
    append_text,
    chat_prefix,
    continuation_ids,
    decode_tagseq_contract,
    encode,
    score_labels,
    tokens_repr,
)
from sallm.config import ModelEvalConfig
from sallm.data.formatters.base import safe_format_prompt
from sallm.data.loaders.huggingface import MASAKHAPOS_DATASET, _load_masakhapos_split
from sallm.evaluation.harness import load_model_and_tokenizer
from sallm.templates import registry as tmpl

LOGGER = logging.getLogger("constrained_pos_eval")

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

DEFAULT_TUPLE_PROMPTS = [
    "masakhane_pos_tagging/lm_eval_p1",
    "masakhane_pos_tagging/lm_eval_p2",
    "masakhane_pos_tagging/lm_eval_p3",
    "masakhane_pos_tagging/lm_eval_p4",
]


def _expand_env(value: str | None) -> str | None:
    if value is None:
        return None
    return os.path.expandvars(value)


def _user_prompt(template_id: str, tokens: list[str]) -> str:
    spec = tmpl.get(template_id)
    return safe_format_prompt(spec.prompt, {"tokens": tokens_repr(tokens)})


def _decode_tuple_contract(
    model: Any,
    tokenizer: Any,
    prompt: str,
    tokens: list[str],
    score_mode: str,
    pad_token_id: int,
    pad_to_multiple_of: int | None,
    device: torch.device,
) -> tuple[list[str], float]:
    context_text = chat_prefix(tokenizer, prompt)
    context_ids = encode(tokenizer, context_text)
    predictions: list[str] = []
    total_score = 0.0

    for index, token in enumerate(tokens):
        prefix = ("[" if index == 0 else ", ") + f"({repr(token)}, '"
        context_text, context_ids = append_text(
            tokenizer, context_text, context_ids, prefix
        )
        ids_by_label = {
            label: continuation_ids(tokenizer, context_text, context_ids, label)
            for label in UPOS_LABELS
        }
        label, score, _ = score_labels(
            model=model,
            context_ids=context_ids,
            label_ids=ids_by_label,
            labels=UPOS_LABELS,
            score_mode=score_mode,
            pad_token_id=pad_token_id,
            pad_to_multiple_of=pad_to_multiple_of,
            device=device,
        )
        predictions.append(label)
        total_score += score
        suffix = label + "')"
        if index == len(tokens) - 1:
            suffix += "]"
        context_text, context_ids = append_text(
            tokenizer, context_text, context_ids, suffix
        )

    return predictions, total_score


def _evaluate_language_prompt(
    model: Any,
    tokenizer: Any,
    language: str,
    split: str,
    template_id: str,
    contract: str,
    score_mode: str,
    max_samples: int | None,
    pad_to_multiple_of: int | None,
    device: torch.device,
) -> dict[str, Any]:
    dataset = _load_masakhapos_split(language, split)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id or eos_token_id.")

    token_total = 0
    token_correct = 0
    sentence_accuracies: list[float] = []
    exact_sequence_matches = 0
    prediction_counts: Counter[str] = Counter()
    gold_counts: Counter[str] = Counter()
    confusion: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    error_examples: list[dict[str, Any]] = []

    for index, row in enumerate(dataset):
        tokens = [str(token) for token in row["tokens"]]
        gold = [str(tag).upper() for tag in row["upos"]]
        if len(tokens) != len(gold):
            raise ValueError(
                f"{language}:{row.get('id', index)} token/tag length mismatch: "
                f"{len(tokens)} vs {len(gold)}"
            )
        prompt = _user_prompt(template_id, tokens)
        if contract == "tuple":
            pred, score = _decode_tuple_contract(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                tokens=tokens,
                score_mode=score_mode,
                pad_token_id=int(pad_token_id),
                pad_to_multiple_of=pad_to_multiple_of,
                device=device,
            )
        elif contract == "tagseq":
            pred, score = decode_tagseq_contract(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                tokens=tokens,
                labels=UPOS_LABELS,
                score_mode=score_mode,
                pad_token_id=int(pad_token_id),
                pad_to_multiple_of=pad_to_multiple_of,
                device=device,
            )
        else:
            raise ValueError(f"Unsupported contract: {contract}")
        if len(pred) != len(gold):
            raise ValueError(
                f"{language}:{row.get('id', index)} prediction/tag length mismatch: "
                f"{len(pred)} vs {len(gold)}"
            )

        correct_mask = [
            gold_tag == pred_tag
            for gold_tag, pred_tag in zip(gold, pred[: len(gold)], strict=False)
        ]
        correct = sum(correct_mask)
        token_total += len(gold)
        token_correct += correct
        sentence_acc = correct / len(gold) if gold else 0.0
        sentence_accuracies.append(sentence_acc)
        exact_match = len(pred) == len(gold) and correct == len(gold)
        exact_sequence_matches += int(exact_match)
        prediction_counts.update(pred)
        gold_counts.update(gold)
        for gold_tag, pred_tag in zip(gold, pred[: len(gold)], strict=False):
            confusion[f"{gold_tag}->{pred_tag}"] += 1

        record = {
            "id": row.get("id", str(index)),
            "tokens": tokens,
            "gold": gold,
            "prediction": pred,
            "correct": correct_mask,
            "token_accuracy": sentence_acc,
            "sequence_score": score,
        }
        if len(examples) < 8:
            examples.append(record)
        if not exact_match and len(error_examples) < 12:
            error_examples.append(record)

        if (index + 1) % 50 == 0:
            LOGGER.info(
                "%s %s %s %s: %d/%d global_acc=%.4f",
                language,
                template_id,
                contract,
                score_mode,
                index + 1,
                len(dataset),
                token_correct / max(token_total, 1),
            )

    n_sentences = len(dataset)
    metrics = {
        "n_sentences": n_sentences,
        "n_tokens": token_total,
        "token_accuracy": token_correct / max(token_total, 1),
        "macro_sentence_token_accuracy": (
            sum(sentence_accuracies) / len(sentence_accuracies)
            if sentence_accuracies
            else 0.0
        ),
        "exact_sequence_accuracy": exact_sequence_matches / max(n_sentences, 1),
        "prediction_label_counts": dict(prediction_counts),
        "gold_label_counts": dict(gold_counts),
        "top_confusions": dict(confusion.most_common(40)),
    }
    return {
        "language": language,
        "split": split,
        "template_id": template_id,
        "template_task": tmpl.get(template_id).task,
        "contract": contract,
        "score_mode": score_mode,
        "metrics": metrics,
        "examples": examples,
        "error_examples": error_examples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--peft-adapter")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--merge-lora", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--tie-word-embeddings", action=argparse.BooleanOptionalAction)
    parser.add_argument("--languages", nargs="+", default=["tsn", "xho", "zul"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--prompt-template", action="append")
    parser.add_argument("--contract", choices=["tuple", "tagseq"], default="tuple")
    parser.add_argument("--score-mode", choices=["mean", "sum"], default="mean")
    parser.add_argument("--max-samples-per-lang", type=int)
    parser.add_argument("--pad-to-multiple-of", type=int, default=64)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    prompt_templates = args.prompt_template or DEFAULT_TUPLE_PROMPTS
    model_cfg = ModelEvalConfig(
        checkpoint=_expand_env(args.checkpoint),
        peft_adapter=_expand_env(args.peft_adapter),
        dtype=args.dtype,
        device=args.device,
        merge_lora=args.merge_lora,
        tie_word_embeddings=args.tie_word_embeddings,
    )
    model, tokenizer = load_model_and_tokenizer(model_cfg)
    model.eval()
    device = torch.device(args.device)

    results: list[dict[str, Any]] = []
    for template_id in prompt_templates:
        LOGGER.info("Evaluating template %s", template_id)
        for language in args.languages:
            payload = _evaluate_language_prompt(
                model=model,
                tokenizer=tokenizer,
                language=language,
                split=args.split,
                template_id=template_id,
                contract=args.contract,
                score_mode=args.score_mode,
                max_samples=args.max_samples_per_lang,
                pad_to_multiple_of=args.pad_to_multiple_of,
                device=device,
            )
            results.append(payload)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "config": {
                    "dataset": MASAKHAPOS_DATASET,
                    "split": args.split,
                    "languages": args.languages,
                    "prompt_templates": prompt_templates,
                    "label_set": UPOS_LABELS,
                    "contract": args.contract,
                    "evaluation_protocol": "closed_label_token_logprob",
                    "primary_metric": "token_accuracy",
                    "score_mode": args.score_mode,
                    "max_samples_per_lang": args.max_samples_per_lang,
                    "pad_to_multiple_of": args.pad_to_multiple_of,
                    "checkpoint": args.checkpoint,
                    "peft_adapter": args.peft_adapter,
                    "dtype": args.dtype,
                    "merge_lora": args.merge_lora,
                },
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    LOGGER.info("Wrote %s", args.output)
    return (
        0 if all(math.isfinite(r["metrics"]["token_accuracy"]) for r in results) else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
