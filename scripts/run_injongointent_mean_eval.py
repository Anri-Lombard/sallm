#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from math import isfinite
from pathlib import Path
from statistics import fmean
from typing import Any, cast

from audit_injongointent_choice_scores import score_rows
from datasets import Dataset, concatenate_datasets
from injongointent_trainonly_validation import (
    load_frozen_validation_rows,
    sha256_file,
)
from omegaconf import OmegaConf
from sallm.config import ExperimentConfig, ModelEvalConfig, TemplateChoice
from sallm.data.factory import build_conversation_dataset
from sallm.data.formatters.classification import format_classification
from sallm.data.loaders.huggingface import (
    _load_injongointent_split,
)
from sallm.evaluation.classification_metrics import ClassificationEvaluator
from sallm.evaluation.harness import load_model_and_tokenizer
from sallm.templates import registry as template_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--template-id")
    parser.add_argument("--template-selection", type=Path)
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-samples-per-lang", type=int)
    parser.add_argument("--choice-batch-size", type=int)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--merge-lora", action="store_true")
    parser.add_argument("--tie-word-embeddings", choices=("true", "false"))
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def load_template_selection(path: Path, languages: list[str]) -> dict[str, str]:
    selection = json.loads(path.read_text())
    if not isinstance(selection, dict):
        raise ValueError("Template selection must be a JSON object")
    selected = {str(key): str(value) for key, value in selection.items()}
    if set(selected) != set(languages):
        raise ValueError(
            "Template selection languages do not match evaluation languages: "
            f"expected {sorted(languages)}, found {sorted(selected)}"
        )
    if any(not value.strip() for value in selected.values()):
        raise ValueError("Template selection contains an empty template id")
    return selected


def resolve_languages(dataset_config: Any) -> list[str]:
    languages = list(dataset_config.languages or [])
    if not languages and dataset_config.subset:
        languages = [str(dataset_config.subset)]
    if not languages:
        raise ValueError(
            "Intent evaluation requires dataset.languages or dataset.subset"
        )
    return languages


def bind_template_registry(config_path: Path, template_ids: list[str]) -> Path:
    """Bind the source-tree templates named by this frozen evaluation config."""
    template_root = config_path.resolve().parent.parent / "templates"
    if not template_root.is_dir():
        raise FileNotFoundError(f"Template directory does not exist: {template_root}")
    template_registry._TEMPLATE_ROOT = template_root
    template_registry._CACHE.clear()
    template_registry._TASK_INDEX.clear()
    template_registry._load_all()
    missing = sorted(set(template_ids) - set(template_registry._CACHE))
    if missing:
        raise ValueError(f"Evaluation config references missing templates: {missing}")
    return template_root


def attach_validation_identity(
    evaluation: Dataset,
    raw_rows: list[dict[str, Any]],
    config: ExperimentConfig,
) -> Dataset:
    """Reattach frozen source identities removed by template expansion."""
    if config.dataset is None:
        raise ValueError("Missing Intent dataset config")
    label_column = config.dataset.label_column or "label"
    identities: dict[
        tuple[str, str, str], deque[tuple[int, str]]
    ] = defaultdict(deque)
    for raw in raw_rows:
        source_index = int(raw["__validation_source_index"])
        row_digest = str(raw["__validation_row_sha256"])
        language = str(raw["lang"])
        for template in config.dataset.templates:
            weight = max(
                int(template.weight) if isfinite(float(template.weight)) else 1,
                1,
            )
            messages = format_classification(raw, str(template.id), label_column)
            key = (
                language,
                str(template.id),
                json.dumps(messages, ensure_ascii=False, sort_keys=True),
            )
            identities[key].extend([(source_index, row_digest)] * weight)

    annotated = []
    for item in evaluation:
        row = dict(item)
        key = (
            str(row["lang"]),
            str(row["template_id"]),
            json.dumps(row["messages"], ensure_ascii=False, sort_keys=True),
        )
        if not identities[key]:
            raise ValueError("Template expansion could not be bound to frozen source")
        source_index, row_digest = identities[key].popleft()
        row["__validation_source_index"] = source_index
        row["__validation_row_sha256"] = row_digest
        row["example_id"] = f"{row['lang']}:train:{source_index}"
        annotated.append(row)
    leftovers = sum(len(values) for values in identities.values())
    if leftovers:
        raise ValueError(
            f"Template expansion left {leftovers} frozen identities unused"
        )
    return Dataset.from_list(annotated)


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
    temp_output = args.output.with_suffix(args.output.suffix + ".tmp")
    if args.output.exists() or temp_output.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation output: {args.output}")

    config = cast(
        ExperimentConfig,
        OmegaConf.merge(
            OmegaConf.structured(ExperimentConfig),
            OmegaConf.load(args.config),
        ),
    )
    tie_word_embeddings = None
    if args.tie_word_embeddings is not None:
        tie_word_embeddings = args.tie_word_embeddings == "true"
    model, tokenizer = load_model_and_tokenizer(
        ModelEvalConfig(
            checkpoint=args.checkpoint,
            peft_adapter=args.adapter,
            dtype=args.dtype,
            merge_lora=args.merge_lora,
            tie_word_embeddings=tie_word_embeddings,
        )
    )
    languages = resolve_languages(config.dataset)
    template_ids = [str(template.id) for template in config.dataset.templates]
    template_root = bind_template_registry(args.config, template_ids)
    if args.template_id and args.template_selection:
        raise ValueError("Use --template-id or --template-selection, not both")
    if args.split == "test" and not (args.template_id or args.template_selection):
        raise ValueError(
            "Test evaluation requires a validation-selected template id or mapping"
        )
    if args.max_length <= 0:
        raise ValueError("--max-length must be positive")
    if args.choice_batch_size is not None and args.choice_batch_size <= 0:
        raise ValueError("--choice-batch-size must be positive")
    if args.split == "validation":
        if args.validation_manifest is None:
            raise ValueError("Validation requires a frozen train-only manifest")
        raw_rows, validation_data = load_frozen_validation_rows(
            args.validation_manifest,
            languages,
        )
        raw = Dataset.from_list(raw_rows)
    else:
        if args.validation_manifest is not None:
            raise ValueError("The validation manifest cannot be used for test loading")
        validation_data = None
        raw = concatenate_datasets(
            [_load_injongointent_split(language, "test") for language in languages]
        )
    if args.max_samples_per_lang is not None:
        parts = []
        for language in languages:
            part = raw.filter(
                lambda row, language=language: str(row["lang"]) == language
            )
            parts.append(part.select(range(min(args.max_samples_per_lang, len(part)))))
        raw = concatenate_datasets(parts)
    evaluation = build_conversation_dataset(
        cast(Dataset, raw),
        config,
        template_choice_override=TemplateChoice.ALL,
    )
    if args.split == "validation":
        evaluation = attach_validation_identity(
            evaluation,
            [dict(row) for row in raw],
            config,
        )
    if args.template_id:
        evaluation = evaluation.filter(
            lambda row: str(row["template_id"]) == args.template_id
        )
        if not len(evaluation):
            raise ValueError(f"No rows matched template {args.template_id!r}")
    template_selection = None
    if args.template_selection:
        template_selection = load_template_selection(args.template_selection, languages)
        evaluation = evaluation.filter(
            lambda row: template_selection[str(row["lang"])]
            == str(row["template_id"])
        )
        selected_languages = {str(row["lang"]) for row in evaluation}
        if selected_languages != set(languages):
            raise ValueError(
                "No evaluation rows matched one or more selected templates: "
                f"matched {sorted(selected_languages)}"
            )
    evaluator = ClassificationEvaluator(tokenizer, max_samples_per_lang=None)
    rows = score_rows(
        model,
        evaluator,
        [dict(row) for row in evaluation],
        include_scores=False,
        model_ctx_limit=args.max_length,
        choice_batch_size=args.choice_batch_size,
    )
    result = {
        "schema": "sallm.injongointent_mean_eval/v2",
        "architecture": args.architecture,
        "checkpoint": args.checkpoint,
        "adapter": args.adapter,
        "model_interface": {
            "dtype": args.dtype,
            "merge_lora": args.merge_lora,
            "tie_word_embeddings": tie_word_embeddings,
            "choice_batch_size": args.choice_batch_size,
        },
        "config": str(args.config),
        "template_root": str(template_root),
        "split": args.split,
        "data_boundary": (
            "pinned_train_only_validation" if args.split == "validation" else "test"
        ),
        "held_out_data_accessed": args.split == "test",
        "validation_manifest_sha256": (
            sha256_file(args.validation_manifest)
            if args.validation_manifest is not None
            else None
        ),
        "validation_data": validation_data,
        "languages": languages,
        "prompts": sorted({str(row["template_id"]) for row in rows}),
        "score_mode": "mean_token_logprob",
        "maximum_input_tokens": args.max_length,
        "template_selection": template_selection,
        "headline_policy": "validation selection only; test requires a sealed template",
        "summary": grouped_metrics(rows, evaluator),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp_output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    temp_output.replace(args.output)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
