#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import fmean, median
from typing import Any, cast

import torch
from datasets import Dataset
from omegaconf import OmegaConf
from sallm.config import ExperimentConfig, ModelEvalConfig
from sallm.data.factory import build_datasets
from sallm.evaluation.classification_metrics import (
    ClassificationEvaluator,
    _gather_target_log_probs,
)
from sallm.evaluation.harness import load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def balanced_rows(dataset: Dataset) -> list[dict[str, Any]]:
    rows = [dict(row) for row in dataset]
    labels = sorted({str(row["messages"][-1]["content"]).strip() for row in rows})
    languages = sorted({str(row["lang"]) for row in rows})
    templates = sorted({str(row["template_id"]) for row in rows})
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row["lang"]),
            str(row["template_id"]),
            str(row["messages"][-1]["content"]).strip(),
        )
        by_key.setdefault(key, row)

    selected = []
    for index, label in enumerate(labels):
        key = (
            languages[index % len(languages)],
            templates[(index // len(languages)) % len(templates)],
            label,
        )
        if key not in by_key:
            raise ValueError(f"Missing balanced validation cell {key!r}")
        selected.append(by_key[key])
    return selected


def score_rows(
    model: Any,
    evaluator: ClassificationEvaluator,
    rows: list[dict[str, Any]],
    *,
    include_scores: bool = True,
) -> list[dict[str, Any]]:
    model.eval()
    device = model.device
    pad_id = evaluator._resolve_pad_id(
        evaluator.tokenizer.pad_token_id,
        evaluator.tokenizer.eos_token_id,
    )
    output = []
    with torch.no_grad():
        for index, row in enumerate(rows):
            messages = row["messages"]
            template_id = str(row["template_id"])
            labels = evaluator._get_label_choices(template_id)
            prompt = evaluator._build_prompt_text(
                prompt_messages=messages[:-1],
                fallback_template=evaluator._get_fallback_template(),
                system_message=row.get("system_message"),
            )
            input_ids, attention, starts = evaluator._build_choice_inputs(
                prompt_text=prompt,
                label_choices=labels,
                model_ctx_limit=evaluator._get_model_ctx_limit(model),
                pad_token_id=pad_id,
                device=device,
                pad_to_multiple_of=evaluator._get_model_chunk_size(model),
            )
            logits = model(
                input_ids=input_ids,
                attention_mask=attention,
                use_cache=False,
            ).logits[:, :-1, :]
            token_log_probs = _gather_target_log_probs(
                logits=logits,
                target_ids=input_ids[:, 1:],
            )
            mask = torch.zeros_like(input_ids, dtype=torch.bool)
            lengths = attention.sum(dim=1)
            for choice_index, start in enumerate(starts):
                mask[choice_index, start : int(lengths[choice_index])] = True
            mask = mask[:, 1:] & attention[:, 1:].bool()
            token_counts = mask.sum(dim=1)
            scores = token_log_probs.masked_fill(~mask, 0).sum(
                dim=1
            ) / token_counts.clamp_min(1)
            ranked = sorted(
                zip(labels, scores.tolist(), token_counts.tolist(), strict=True),
                key=lambda item: item[1],
                reverse=True,
            )
            gold = str(messages[-1]["content"]).strip()
            gold_rank = next(
                rank for rank, (label, _, _) in enumerate(ranked, 1) if label == gold
            )
            result_row = {
                "row": index,
                "lang": row["lang"],
                "template_id": template_id,
                "text": row.get("text") or messages[0]["content"],
                "gold": gold,
                "prediction": ranked[0][0],
                "gold_rank": gold_rank,
                "top_two_margin": ranked[0][1] - ranked[1][1],
            }
            if include_scores:
                result_row["scores"] = [
                    {
                        "label": label,
                        "mean_token_logprob": score,
                        "token_count": token_count,
                        "rank": rank,
                    }
                    for rank, (label, score, token_count) in enumerate(ranked, 1)
                ]
            output.append(result_row)
    return output


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "accuracy": fmean(row["prediction"] == row["gold"] for row in rows),
        "mean_gold_rank": fmean(row["gold_rank"] for row in rows),
        "median_gold_rank": median(row["gold_rank"] for row in rows),
        "mean_top_two_margin": fmean(row["top_two_margin"] for row in rows),
        "winning_label_counts": dict(Counter(row["prediction"] for row in rows)),
    }


def self_check() -> None:
    rows = [
        {
            "prediction": "a",
            "gold": "a",
            "gold_rank": 1,
            "top_two_margin": 0.5,
        },
        {
            "prediction": "a",
            "gold": "b",
            "gold_rank": 2,
            "top_two_margin": 0.25,
        },
    ]
    summary = summarize(rows)
    assert summary["accuracy"] == 0.5
    assert summary["mean_gold_rank"] == 1.5
    assert summary["winning_label_counts"] == {"a": 2}


def main() -> None:
    args = parse_args()
    if args.self_check:
        self_check()
        print("SELF_CHECK_OK")
        return

    config = cast(
        ExperimentConfig,
        OmegaConf.merge(
            OmegaConf.structured(ExperimentConfig),
            OmegaConf.load(args.config),
        ),
    )
    model, tokenizer = load_model_and_tokenizer(
        ModelEvalConfig(
            checkpoint=args.checkpoint,
            peft_adapter=args.adapter,
            merge_lora=False,
        )
    )
    _, validation, _ = build_datasets(config, tokenizer, is_hpo=False)
    rows = balanced_rows(cast(Dataset, validation))
    evaluator = ClassificationEvaluator(tokenizer, max_samples_per_lang=None)

    if args.adapter:
        with model.disable_adapter():
            base_rows = score_rows(model, evaluator, rows)
        adapter_rows = score_rows(model, evaluator, rows)
    else:
        base_rows = score_rows(model, evaluator, rows)
        adapter_rows = None
    result = {
        "architecture": args.architecture,
        "checkpoint": args.checkpoint,
        "adapter": args.adapter,
        "config": str(args.config),
        "split": "deterministic decontaminated validation only",
        "score_mode": "mean_token_logprob",
        "base": {"summary": summarize(base_rows), "rows": base_rows},
    }
    if adapter_rows is not None:
        result["adapter_1epoch"] = {
            "summary": summarize(adapter_rows),
            "rows": adapter_rows,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    summaries = {
        key: value["summary"]
        for key, value in result.items()
        if key in {"base", "adapter_1epoch"}
    }
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
