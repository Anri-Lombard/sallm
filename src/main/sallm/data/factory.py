from __future__ import annotations

from math import isfinite
from typing import Any

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from sallm.config import (
    DataConfig,
    ExperimentConfig,
    FinetuneDatasetConfig,
    FinetuneTaskType,
    RunMode,
    TemplateChoice,
    TemplateRef,
)
from sallm.data.formatters.base import TaskFormatter
from sallm.data.formatters.factory import build_formatter
from sallm.data.loaders.factory import get_loader
from sallm.data.loaders.local import LocalDatasetLoader
from sallm.data.utils import resolve_language_column
from sallm.templates import registry as tmpl
from sallm.templates.registry import TemplateSpec


def build_datasets(
    config: ExperimentConfig, tokenizer: AutoTokenizer, is_hpo: bool
) -> tuple[Dataset, Dataset, Dataset | None]:
    if config.mode == RunMode.FINETUNE:
        if config.dataset is None:
            raise ValueError("Finetune mode requires a dataset configuration")
        return _build_finetune_datasets(config.dataset)
    if config.data is None:
        raise ValueError("Training and evaluation modes require a data block")
    return _build_local_datasets(config.data, is_hpo)


def _build_local_datasets(
    data_conf: DataConfig, is_hpo: bool
) -> tuple[Dataset, Dataset, Dataset | None]:
    dataset_dict = LocalDatasetLoader().load(data_conf)
    train_ds = dataset_dict[data_conf.train_split]
    eval_ds = dataset_dict[data_conf.eval_split]
    test_ds = None
    if not is_hpo and data_conf.test_split and data_conf.test_split in dataset_dict:
        test_ds = dataset_dict[data_conf.test_split]
    return train_ds, eval_ds, test_ds


def _build_finetune_datasets(
    dataset_conf: FinetuneDatasetConfig,
) -> tuple[Dataset, Dataset, Dataset | None]:
    if dataset_conf.task is None:
        raise ValueError("dataset.task must be specified for fine-tuning")
    loader = get_loader(dataset_conf)
    dataset_dict = loader.load(dataset_conf)
    train_raw = _resolve_split(dataset_dict, ["train"])
    val_raw = _resolve_split(dataset_dict, ["validation", "val", "dev", "test"])
    template_pairs = _load_template_pairs(dataset_conf)
    train_ds = _format_split(train_raw, dataset_conf, template_pairs)
    val_ds = _format_split(val_raw, dataset_conf, template_pairs)
    return train_ds, val_ds, None


def _load_template_pairs(
    dataset_conf: FinetuneDatasetConfig,
) -> list[tuple[TemplateRef, TemplateSpec]]:
    pairs: list[tuple[TemplateRef, TemplateSpec]] = []
    for ref in dataset_conf.templates:
        spec = tmpl.get(ref.id)
        pairs.append((ref, spec))
    return pairs


def _format_split(
    raw_ds: Dataset,
    dataset_conf: FinetuneDatasetConfig,
    template_pairs: list[tuple[TemplateRef, TemplateSpec]],
) -> Dataset:
    formatter = build_formatter(dataset_conf.task, raw_ds, dataset_conf)
    if dataset_conf.template_choice == TemplateChoice.ALL:
        for _, spec in template_pairs:
            formatter.validate_template(spec, dataset_conf)
        formatted = _format_with_all_templates(
            raw_ds, dataset_conf, formatter, template_pairs
        )
    else:
        first_spec = template_pairs[0][1] if template_pairs else None
        formatter.validate_template(first_spec, dataset_conf)
        formatted = _format_single_template(
            raw_ds, dataset_conf, formatter, first_spec
        )
    return _ensure_lang_column(formatted, raw_ds, dataset_conf)


def _format_with_all_templates(
    raw_ds: Dataset,
    dataset_conf: FinetuneDatasetConfig,
    formatter: TaskFormatter,
    template_pairs: list[tuple[TemplateRef, TemplateSpec]],
) -> Dataset:
    def _expand_batch(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        out_messages: list[list[dict[str, str]]] = []
        out_template_ids: list[str] = []
        out_langs: list[str] = []
        length = len(next(iter(batch.values()))) if batch else 0
        for i in range(length):
            example = {k: v[i] for k, v in batch.items()}
            for ref, spec in template_pairs:
                weight = int(ref.weight) if isfinite(ref.weight) else 1
                weight = max(weight, 1)
                messages = formatter.format(example, spec, dataset_conf)
                for _ in range(weight):
                    out_messages.append(messages)
                    out_template_ids.append(spec.id)
                    if dataset_conf.subset:
                        out_langs.append(dataset_conf.subset)
        result: dict[str, list[Any]] = {
            "messages": out_messages,
            "template_id": out_template_ids,
        }
        if dataset_conf.subset:
            result["lang"] = out_langs
        return result

    formatted = raw_ds.map(
        _expand_batch,
        batched=True,
        remove_columns=raw_ds.column_names,
        desc="Expanding ALL templates",
    )
    if dataset_conf.subset and "lang" not in formatted.column_names:
        formatted = formatted.add_column(
            "lang", [dataset_conf.subset] * len(formatted)
        )
    return formatted.shuffle(seed=42)


def _format_single_template(
    raw_ds: Dataset,
    dataset_conf: FinetuneDatasetConfig,
    formatter: TaskFormatter,
    template_spec: TemplateSpec | None,
) -> Dataset:
    desc = _format_description(dataset_conf.task)

    def _to_messages(example: dict[str, Any]) -> dict[str, Any]:
        messages = formatter.format(example, template_spec, dataset_conf)
        return {"messages": messages}

    formatted = raw_ds.map(
        _to_messages,
        batched=False,
        remove_columns=raw_ds.column_names,
        desc=desc,
    )
    return formatted


def _ensure_lang_column(
    processed: Dataset, raw: Dataset, dataset_conf: FinetuneDatasetConfig
) -> Dataset:
    if "lang" in processed.column_names:
        return processed
    if dataset_conf.subset:
        return processed.add_column(
            "lang", [dataset_conf.subset] * len(processed)
        )
    values = resolve_language_column(raw)
    if values is not None:
        return processed.add_column("lang", values)
    return processed


def _format_description(task: FinetuneTaskType) -> str:
    if task == FinetuneTaskType.INSTRUCTION:
        return "Format instruction data"
    if task == FinetuneTaskType.NAMED_ENTITY_RECOGNITION:
        return "Format NER data"
    if task == FinetuneTaskType.CLASSIFICATION:
        return "Format classification data"
    if task == FinetuneTaskType.POS_TAGGING:
        return "Formatting dataset for POS tagging"
    return "Format dataset"


def _resolve_split(
    dataset_dict: DatasetDict, preferred_keys: list[str]
) -> Dataset:
    for key in preferred_keys:
        if key in dataset_dict:
            return dataset_dict[key]
    first_key = next(iter(dataset_dict.keys()))
    return dataset_dict[first_key]
