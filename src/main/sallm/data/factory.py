from __future__ import annotations

import logging
from dataclasses import is_dataclass, replace
from typing import Protocol, cast

from datasets import Dataset
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset as TorchDataset

from sallm.configs import (
    ExperimentConfig,
    FinetuneDatasetConfig,
    FinetuneTaskType,
    TemplateChoice,
)
from sallm.data.adapters.registry import load_raw_dataset
from sallm.data.loaders.mix import load_mix_dataset
from sallm.data.transforms.template_strategies import apply_templates

logger = logging.getLogger(__name__)


class HasDatasetConfig(Protocol):
    dataset: FinetuneDatasetConfig | None


def resolve_eval_template_choice(ds_cfg: FinetuneDatasetConfig) -> TemplateChoice:
    """Resolve validation template coverage for fine-tuning datasets.

    Classification tasks validate across every prompt template so early stopping
    sees the same prompt coverage as test evaluation.
    """
    if ds_cfg.eval_template_choice is not None:
        return ds_cfg.eval_template_choice
    if ds_cfg.task == FinetuneTaskType.CLASSIFICATION and len(ds_cfg.templates) > 1:
        return TemplateChoice.ALL
    return ds_cfg.template_choice


def _dataset_config_with_template_choice(
    ds_cfg: FinetuneDatasetConfig,
    template_choice: TemplateChoice,
) -> FinetuneDatasetConfig:
    if ds_cfg.template_choice == template_choice:
        return ds_cfg
    if is_dataclass(ds_cfg):
        return replace(ds_cfg, template_choice=template_choice)
    if isinstance(ds_cfg, DictConfig):
        cloned_cfg = OmegaConf.create(OmegaConf.to_container(ds_cfg, resolve=False))
        cloned_cfg.template_choice = template_choice
        return cast(FinetuneDatasetConfig, cloned_cfg)
    raise TypeError(
        "Expected dataset config to be a dataclass or DictConfig, got "
        f"{type(ds_cfg).__name__}"
    )


def build_datasets(
    config: ExperimentConfig,
) -> tuple[Dataset | TorchDataset, Dataset, None]:
    """Build train and validation datasets for fine-tuning."""
    if config.dataset is None:
        raise ValueError("Fine-tuning requires a `dataset` config block.")
    ds_cfg = config.dataset

    if isinstance(ds_cfg.hf_name, str) and ds_cfg.hf_name.startswith("mix:"):
        return load_mix_dataset(config)

    train_raw, val_raw = load_raw_dataset(ds_cfg)
    train_ds = build_conversation_dataset(train_raw, config)
    val_ds = build_conversation_dataset(
        val_raw,
        config,
        template_choice_override=resolve_eval_template_choice(ds_cfg),
    )
    return train_ds, val_ds, None


def build_conversation_dataset(
    raw_ds: Dataset,
    cfg: HasDatasetConfig,
    template_choice_override: TemplateChoice | None = None,
) -> Dataset:
    """Format a raw dataset into conversation format with messages."""
    ds_cfg = cfg.dataset
    if ds_cfg is None:
        raise ValueError("Fine-tuning dataset config is required.")
    if template_choice_override is not None:
        ds_cfg = _dataset_config_with_template_choice(ds_cfg, template_choice_override)
    return apply_templates(raw_ds, ds_cfg)
