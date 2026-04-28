from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml
from datasets import (
    Dataset,
    DatasetDict,
    concatenate_datasets,
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
from sallm.data.loaders.huggingface import apply_language_filters, load_hf_dataset
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
    eval_tc_str = comp.get("eval_template_choice")
    eval_template_choice = (
        TEMPLATE_CHOICE_MAP.get(eval_tc_str, TemplateChoice.CYCLE)
        if eval_tc_str is not None
        else None
    )

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
        eval_template_choice=eval_template_choice,
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

    tr, va, needs_filter = load_hf_dataset(comp_cfg)
    return apply_language_filters(tr, va, comp_cfg, needs_filter)


class _CfgWrapper:
    """Minimal wrapper to satisfy apply_templates interface."""

    def __init__(self, ds_cfg: FinetuneDatasetConfig):
        self.dataset = ds_cfg


def _process_component(comp_cfg: FinetuneDatasetConfig) -> tuple[Dataset, Dataset]:
    """Load and process a mix component."""
    tr_raw, va_raw = _load_component_raw(comp_cfg)
    wrapper = _CfgWrapper(comp_cfg)
    tr = apply_templates(tr_raw, wrapper.dataset)
    val_cfg = comp_cfg
    if (
        comp_cfg.eval_template_choice is not None
        and comp_cfg.eval_template_choice != comp_cfg.template_choice
    ):
        val_cfg = replace(comp_cfg, template_choice=comp_cfg.eval_template_choice)
    elif (
        comp_cfg.task == FinetuneTaskType.CLASSIFICATION
        and len(comp_cfg.templates) > 1
        and comp_cfg.template_choice != TemplateChoice.ALL
    ):
        val_cfg = replace(comp_cfg, template_choice=TemplateChoice.ALL)
    va = apply_templates(va_raw, val_cfg)
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
