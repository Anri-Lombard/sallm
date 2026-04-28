from __future__ import annotations

from math import isfinite
from typing import Any

from datasets import Dataset

from sallm.config import FinetuneDatasetConfig, FinetuneTaskType, TemplateChoice
from sallm.data.formatters.classification import format_classification
from sallm.data.formatters.instruction import format_instruction
from sallm.data.formatters.ner import format_ner
from sallm.data.formatters.pos import format_pos, get_upos_class_names
from sallm.templates import registry as tmpl

FORMATTERS = {
    FinetuneTaskType.INSTRUCTION: lambda ex, t_id, lc, upos: format_instruction(
        ex, t_id
    ),
    FinetuneTaskType.CLASSIFICATION: lambda ex, t_id, lc, upos: format_classification(
        ex, t_id, lc or "label"
    ),
    FinetuneTaskType.NAMED_ENTITY_RECOGNITION: lambda ex, t_id, lc, upos: format_ner(
        ex, t_id
    ),
    FinetuneTaskType.POS_TAGGING: lambda ex, t_id, lc, upos: format_pos(ex, t_id, upos),
}


def _format_example(
    ex: dict[str, Any],
    task: FinetuneTaskType,
    template_id: str | None,
    label_column: str | None,
    upos_class_names: list[str] | None,
) -> list[dict[str, str]]:
    """Format a single example based on task type."""
    formatter = FORMATTERS.get(task)
    if formatter is None:
        raise ValueError(f"Unsupported task type: {task}")
    return formatter(ex, template_id, label_column, upos_class_names)


def _require_task(ds_cfg: FinetuneDatasetConfig) -> FinetuneTaskType:
    if ds_cfg.task is None:
        raise ValueError("A `dataset.task` must be specified for fine-tuning.")
    return ds_cfg.task


def _validate_templates(ds_cfg: FinetuneDatasetConfig, raw_ds: Dataset) -> None:
    """Validate templates for the given task type."""
    template_ids = [t.id for t in ds_cfg.templates]

    if ds_cfg.task == FinetuneTaskType.NAMED_ENTITY_RECOGNITION:
        for t_id in template_ids:
            if not tmpl.get(t_id).ner_tags:
                raise ValueError(
                    f"Template '{t_id}' selected for NER but missing 'ner_tags'."
                )

    if ds_cfg.task == FinetuneTaskType.CLASSIFICATION:
        for t_id in template_ids:
            if not tmpl.get(t_id).label_mapping:
                raise ValueError(
                    f"Template '{t_id}' selected for classification "
                    "but missing 'label_mapping'."
                )
        if ds_cfg.label_column not in raw_ds.column_names:
            raise ValueError(
                f"Classification label column '{ds_cfg.label_column}' "
                "not found in dataset."
            )


def _add_lang_to_output(
    out: dict[str, Any],
    ex: dict[str, Any],
    ds_cfg: FinetuneDatasetConfig,
) -> None:
    """Add language info to output dict if available."""
    lang_val = ex.get("lang")
    if lang_val:
        out["lang"] = str(lang_val)
    elif ds_cfg.subset:
        out["lang"] = ds_cfg.subset


def apply_all_templates(raw_ds: Dataset, ds_cfg: FinetuneDatasetConfig) -> Dataset:
    """Expand each example across all templates."""
    task = _require_task(ds_cfg)
    if not ds_cfg.templates:
        raise ValueError(
            "dataset.template_choice is 'ALL' but no templates were provided."
        )

    _validate_templates(ds_cfg, raw_ds)
    upos_class_names = (
        get_upos_class_names(raw_ds) if task == FinetuneTaskType.POS_TAGGING else None
    )

    def _expand_batch(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        out_messages: list[list[dict[str, str]]] = []
        out_template_ids: list[str] = []
        out_langs: list[str] = []

        n = len(next(iter(batch.values()))) if batch else 0
        for i in range(n):
            ex = {k: v[i] for k, v in batch.items()}

            for tref in ds_cfg.templates:
                t_id = tref.id
                w = max(int(tref.weight) if isfinite(tref.weight) else 1, 1)
                msgs = _format_example(
                    ex, task, t_id, ds_cfg.label_column, upos_class_names
                )

                for _ in range(w):
                    out_messages.append(msgs)
                    out_template_ids.append(t_id)
                    lang_val = ex.get("lang")
                    if lang_val:
                        out_langs.append(str(lang_val))
                    elif ds_cfg.subset:
                        out_langs.append(ds_cfg.subset)

        result: dict[str, list[Any]] = {
            "messages": out_messages,
            "template_id": out_template_ids,
        }
        if out_langs:
            result["lang"] = out_langs
        return result

    processed_ds = raw_ds.map(
        _expand_batch,
        batched=True,
        remove_columns=raw_ds.column_names,
        desc="Expanding ALL templates",
    )

    if ds_cfg.subset and "lang" not in processed_ds.column_names:
        processed_ds = processed_ds.add_column(
            "lang", [ds_cfg.subset] * len(processed_ds)
        )

    return processed_ds.shuffle(seed=42)


def apply_cycle_templates(raw_ds: Dataset, ds_cfg: FinetuneDatasetConfig) -> Dataset:
    """Assign templates in round-robin fashion."""
    task = _require_task(ds_cfg)
    if not ds_cfg.templates:
        raise ValueError(
            "dataset.template_choice is 'CYCLE' but no templates were provided."
        )

    _validate_templates(ds_cfg, raw_ds)
    template_ids = [t.id for t in ds_cfg.templates]
    upos_class_names = (
        get_upos_class_names(raw_ds) if task == FinetuneTaskType.POS_TAGGING else None
    )

    def _cycle_map(ex: dict[str, Any], idx: int) -> dict[str, Any]:
        t_id = template_ids[idx % len(template_ids)]
        msgs = _format_example(ex, task, t_id, ds_cfg.label_column, upos_class_names)
        out: dict[str, Any] = {"messages": msgs, "template_id": t_id}
        _add_lang_to_output(out, ex, ds_cfg)
        return out

    return raw_ds.map(
        _cycle_map,
        batched=False,
        with_indices=True,
        remove_columns=raw_ds.column_names,
        desc=f"Cycling {task.value} templates",
    )


def apply_templates(raw_ds: Dataset, ds_cfg: FinetuneDatasetConfig) -> Dataset:
    """Apply templates based on the template_choice strategy."""
    if "messages" in raw_ds.column_names:
        return raw_ds

    _require_task(ds_cfg)

    if ds_cfg.template_choice == TemplateChoice.ALL:
        return apply_all_templates(raw_ds, ds_cfg)
    else:
        return apply_cycle_templates(raw_ds, ds_cfg)
