#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Any, cast

from audit_injongointent_choice_scores import score_rows
from datasets import Dataset, concatenate_datasets
from omegaconf import OmegaConf
from sallm.config import ExperimentConfig, ModelEvalConfig, TemplateChoice
from sallm.data.factory import build_conversation_dataset
from sallm.data.loaders.huggingface import _load_injongointent_split
from sallm.evaluation.classification_metrics import ClassificationEvaluator
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


def resolve_languages(dataset_config: Any) -> list[str]:
    languages = list(dataset_config.languages or [])
    if not languages and dataset_config.subset:
        languages = [str(dataset_config.subset)]
    if not languages:
        raise ValueError(
            "Intent evaluation requires dataset.languages or dataset.subset"
        )
    return languages


def grouped_metrics(
    rows: list[dict[str, Any]],
    evaluator: ClassificationEvaluator,
) -> dict[str, Any]:
    languages = sorted({str(row["lang"]) for row in rows})
    templates = sorted({str(row["template_id"]) for row in rows})

    def metrics(subset: list[dict[str, Any]]) -> dict[str, float]:
        return evaluator._compute_classification_metrics(
            [str(row["gold"]) for row in subset],
            [str(row["prediction"]) for row in subset],
        )

    prompt_metrics = {
        template: metrics([row for row in rows if str(row["template_id"]) == template])
        for template in templates
    }

    def summarize_prompts(
        values_by_prompt: dict[str, dict[str, float]],
    ) -> dict[str, dict[str, Any]]:
        metric_names = sorted(
            {metric for values in values_by_prompt.values() for metric in values}
        )
        summary = {}
        for metric in metric_names:
            values = {
                prompt: prompt_values[metric]
                for prompt, prompt_values in values_by_prompt.items()
                if metric in prompt_values
            }
            winner = max(values, key=lambda prompt: (values[prompt], prompt))
            summary[metric] = {
                "best_prompt": winner,
                "best_prompt_value": values[winner],
                "prompt_mean": fmean(values.values()),
                "prompt_range": [min(values.values()), max(values.values())],
                "prompt_values": values,
            }
        return summary

    language_metrics = {}
    for language in languages:
        lang_rows = [row for row in rows if str(row["lang"]) == language]
        per_prompt = {
            template: metrics(
                [row for row in lang_rows if str(row["template_id"]) == template]
            )
            for template in templates
        }
        metric_prompt_summary = summarize_prompts(per_prompt)
        winner = metric_prompt_summary["f1"]["best_prompt"]
        language_metrics[language] = {
            "prompts": per_prompt,
            "headline_metric": "f1",
            "best_prompt": winner,
            "best_prompt_metrics": per_prompt[winner],
            "metric_prompt_summary": metric_prompt_summary,
        }

    metric_prompt_summary = summarize_prompts(prompt_metrics)
    overall_winner = metric_prompt_summary["f1"]["best_prompt"]
    return {
        "all_rows": metrics(rows),
        "prompts": prompt_metrics,
        "headline_metric": "f1",
        "best_prompt": overall_winner,
        "best_prompt_metrics": prompt_metrics[overall_winner],
        "metric_prompt_summary": metric_prompt_summary,
        "languages": language_metrics,
    }


def self_check() -> None:
    evaluator = ClassificationEvaluator(tokenizer=object())
    rows = [
        {"lang": "a", "template_id": "p1", "gold": "x", "prediction": "x"},
        {"lang": "a", "template_id": "p2", "gold": "x", "prediction": "y"},
    ]
    summary = grouped_metrics(rows, evaluator)
    assert summary["best_prompt"] == "p1"
    assert summary["best_prompt_metrics"]["accuracy"] == 1.0
    assert summary["metric_prompt_summary"]["accuracy"]["prompt_mean"] == 0.5


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
    languages = resolve_languages(config.dataset)
    raw_test = concatenate_datasets(
        [_load_injongointent_split(language, "test") for language in languages]
    )
    test = build_conversation_dataset(
        cast(Dataset, raw_test),
        config,
        template_choice_override=TemplateChoice.ALL,
    )
    evaluator = ClassificationEvaluator(tokenizer, max_samples_per_lang=None)
    rows = score_rows(
        model,
        evaluator,
        [dict(row) for row in test],
        include_scores=False,
    )
    result = {
        "architecture": args.architecture,
        "checkpoint": args.checkpoint,
        "adapter": args.adapter,
        "config": str(args.config),
        "split": "official held-out test only",
        "languages": languages,
        "prompts": sorted({str(row["template_id"]) for row in rows}),
        "score_mode": "mean_token_logprob",
        "headline_policy": (
            "best prompt by F1; per-metric best prompt, mean, range and values retained"
        ),
        "summary": grouped_metrics(rows, evaluator),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
