from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from datasets import (
    Dataset,
    DatasetDict,
    concatenate_datasets,
    get_dataset_config_names,
    load_dataset,
)
from torch.utils.data import Dataset as TorchDataset

from sallm.config import (
    ExperimentConfig,
    FinetuneDatasetConfig,
    FinetuneTaskType,
    TemplateChoice,
    TemplateRef,
)
from sallm.data.afrihg import load_afrihg_from_github
from sallm.data.loaders.base import load_split_with_fallback
from sallm.data.multitask import TaskComponent, WeightedMultiTaskDataset
from sallm.data.t2x import load_t2x_from_github
from sallm.data.transforms.template_strategies import apply_templates

logger = logging.getLogger(__name__)

MIXES_DIR = Path(__file__).parent.parent / "mixes"

TASK_TYPE_MAP = {
    "classification": FinetuneTaskType.CLASSIFICATION,
    "instruction": FinetuneTaskType.INSTRUCTION,
    "named_entity_recognition": FinetuneTaskType.NAMED_ENTITY_RECOGNITION,
    "pos_tagging": FinetuneTaskType.POS_TAGGING,
}

TEMPLATE_CHOICE_MAP = {
    "all": TemplateChoice.ALL,
    "cycle": TemplateChoice.CYCLE,
}


def load_mix_config(mix_name: str) -> dict[str, Any]:
    """Load mix configuration from YAML file."""
    yaml_path = MIXES_DIR / f"{mix_name}.yaml"

    if not yaml_path.exists():
        for yaml_file in MIXES_DIR.glob("*.yaml"):
            with open(yaml_file) as f:
                config = yaml.safe_load(f)
                aliases = config.get("aliases", [])
                if mix_name in aliases or config.get("name") == mix_name:
                    return config

        raise ValueError(f"Unknown mix configuration: {mix_name}")

    with open(yaml_path) as f:
        return yaml.safe_load(f)


def _component_to_config(
    comp: dict[str, Any],
    base_cfg: FinetuneDatasetConfig,
) -> FinetuneDatasetConfig:
    """Convert a YAML component definition to FinetuneDatasetConfig."""
    templates = [
        TemplateRef(id=t["id"], weight=t.get("weight", 1.0))
        for t in comp.get("templates", [])
    ]

    task_str = comp.get("task", "instruction")
    task = TASK_TYPE_MAP.get(task_str, FinetuneTaskType.INSTRUCTION)

    tc_str = comp.get("template_choice", "cycle")
    template_choice = TEMPLATE_CHOICE_MAP.get(tc_str, TemplateChoice.CYCLE)

    max_seq = comp.get("max_seq_length_override")
    if max_seq is None:
        max_seq = base_cfg.max_seq_length
    elif base_cfg.max_seq_length < max_seq:
        max_seq = base_cfg.max_seq_length

    return FinetuneDatasetConfig(
        hf_name=comp["hf_name"],
        subset=comp.get("subset"),
        languages=comp.get("languages"),
        task=task,
        splits=comp.get("splits", {"train": "train", "val": "validation"}),
        templates=templates,
        template_choice=template_choice,
        label_column=comp.get("label_column"),
        max_seq_length=max_seq,
        packing=base_cfg.packing,
        assistant_only_loss=base_cfg.assistant_only_loss,
    )


def _load_component_raw(comp_cfg: FinetuneDatasetConfig) -> tuple[Dataset, Dataset]:
    """Load raw train/val datasets for a mix component."""
    if isinstance(comp_cfg.hf_name, str) and comp_cfg.hf_name.startswith("github:"):
        gh_ref = comp_cfg.hf_name[len("github:") :]
        if "francois-meyer/t2x" in gh_ref or gh_ref.strip().endswith("/t2x"):
            ds_from_github = load_t2x_from_github()
        else:
            ds_from_github = load_afrihg_from_github(languages=comp_cfg.languages)

        if isinstance(ds_from_github, DatasetDict):
            tr = (
                ds_from_github["train"]
                if "train" in ds_from_github
                else ds_from_github[next(iter(ds_from_github.keys()))]
            )
            if "validation" in ds_from_github:
                va = ds_from_github["validation"]
            elif "dev" in ds_from_github:
                va = ds_from_github["dev"]
            elif "test" in ds_from_github:
                va = ds_from_github["test"]
            else:
                va = tr
        else:
            tr = ds_from_github
            va = ds_from_github
        return tr, va

    try:
        available_configs = get_dataset_config_names(comp_cfg.hf_name)
    except (TypeError, RuntimeError):
        # Datasets 4.0+ rejects scripts; assume parquet files exist
        available_configs = []
    load_name = comp_cfg.subset
    lang_list_cfg = list(comp_cfg.languages or [])

    if lang_list_cfg:
        can_multi_load = all(
            lang_code in available_configs for lang_code in lang_list_cfg
        )
        if can_multi_load:
            train_parts: list[Dataset] = []
            val_parts: list[Dataset] = []
            for lang_code in lang_list_cfg:
                tr = load_dataset(
                    comp_cfg.hf_name,
                    name=lang_code,
                    split=comp_cfg.splits["train"],
                    revision="refs/convert/parquet",
                )
                va = load_split_with_fallback(
                    comp_cfg.hf_name,
                    lang_code,
                    comp_cfg.splits["val"],
                    "refs/convert/parquet",
                )
                if "lang" not in tr.column_names:
                    tr = tr.add_column("lang", [lang_code] * len(tr))
                if "lang" not in va.column_names:
                    va = va.add_column("lang", [lang_code] * len(va))
                train_parts.append(tr)
                val_parts.append(va)
            return concatenate_datasets(train_parts), concatenate_datasets(val_parts)

    try:
        tr = load_dataset(
            comp_cfg.hf_name,
            name=load_name,
            split=comp_cfg.splits["train"],
            revision="refs/convert/parquet",
        )
        va = load_split_with_fallback(
            comp_cfg.hf_name, load_name, comp_cfg.splits["val"], "refs/convert/parquet"
        )
    except Exception:
        # Fallback for datasets without parquet branch
        tr = load_dataset(
            comp_cfg.hf_name,
            name=load_name,
            split=comp_cfg.splits["train"],
        )
        va = load_split_with_fallback(
            comp_cfg.hf_name, load_name, comp_cfg.splits["val"]
        )
    return tr, va


class _CfgWrapper:
    """Minimal wrapper to satisfy apply_templates interface."""

    def __init__(self, ds_cfg: FinetuneDatasetConfig):
        self.dataset = ds_cfg


def _process_component(comp_cfg: FinetuneDatasetConfig) -> tuple[Dataset, Dataset]:
    """Load and process a mix component."""
    tr_raw, va_raw = _load_component_raw(comp_cfg)
    wrapper = _CfgWrapper(comp_cfg)
    tr = apply_templates(tr_raw, wrapper.dataset)
    va = apply_templates(va_raw, wrapper.dataset)
    return tr, va


def load_mix_dataset(
    config: ExperimentConfig,
) -> tuple[TorchDataset, Dataset, None]:
    """Build weighted multi-task dataset from mix configuration.

    Args:
        config: Experiment configuration with dataset.hf_name starting with "mix:"

    Returns:
        Tuple of (WeightedMultiTaskDataset, concatenated_val_ds, None)
    """
    ds_cfg = config.dataset
    mix_name = ds_cfg.hf_name[len("mix:") :].strip().lower()

    mix_config = load_mix_config(mix_name)
    components_yaml = mix_config.get("components", [])

    weight_map = dict(ds_cfg.mix_weights)
    component_names = [c["name"] for c in components_yaml]
    missing_weights = [name for name in component_names if name not in weight_map]
    if missing_weights:
        raise ValueError(
            f"Missing mix weights for components: {sorted(missing_weights)}"
        )

    train_components: list[TaskComponent] = []
    val_parts: list[Dataset] = []

    for comp_yaml in components_yaml:
        comp_cfg = _component_to_config(comp_yaml, ds_cfg)
        train_processed, val_processed = _process_component(comp_cfg)
        train_components.append(
            TaskComponent(
                name=comp_yaml["name"],
                dataset=train_processed,
                weight=weight_map[comp_yaml["name"]],
            )
        )
        val_parts.append(val_processed)

    epoch_size_cfg = ds_cfg.mix_epoch_size
    if isinstance(epoch_size_cfg, str):
        epoch_size_value = None if epoch_size_cfg == "sum" else None
    elif isinstance(epoch_size_cfg, int):
        epoch_size_value = epoch_size_cfg
    else:
        epoch_size_value = None

    seed_value = 0
    training_cfg = config.training
    if training_cfg is not None:
        try:
            seed_candidate = training_cfg.get("seed", 0)
            if seed_candidate is not None:
                seed_value = int(seed_candidate)
        except Exception:
            seed_value = 0

    train_mix = WeightedMultiTaskDataset(
        train_components,
        seed=seed_value,
        temperature=ds_cfg.mix_temperature,
        epoch_size=epoch_size_value,
        min_prob=ds_cfg.mix_min_prob,
        max_prob=ds_cfg.mix_max_prob,
    )

    logger.info("SA General mix distribution | %s", train_mix.describe())

    return (train_mix, concatenate_datasets(val_parts), None)
