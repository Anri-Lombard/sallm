from __future__ import annotations

import logging

from datasets import Dataset
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoTokenizer

from sallm.config import ExperimentConfig, RunMode
from sallm.data.loaders.disk import load_pretrain_datasets
from sallm.data.loaders.github import load_from_github
from sallm.data.loaders.huggingface import apply_language_filters, load_hf_dataset
from sallm.data.loaders.mix import load_mix_dataset
from sallm.data.transforms.template_strategies import apply_templates

logger = logging.getLogger(__name__)


def build_datasets(
    config: ExperimentConfig, tokenizer: AutoTokenizer, is_hpo: bool
) -> tuple[Dataset | TorchDataset, Dataset, Dataset | None]:
    """Build train, validation, and optional test datasets.

    Args:
        config: Experiment configuration
        tokenizer: Tokenizer (unused but kept for API compatibility)
        is_hpo: Whether this is a hyperparameter optimization run

    Returns:
        Tuple of (train_ds, val_ds, test_ds)
    """
    if config.mode == RunMode.FINETUNE:
        return _build_finetune_datasets(config)
    return load_pretrain_datasets(config, is_hpo)


def _build_finetune_datasets(
    config: ExperimentConfig,
) -> tuple[Dataset | TorchDataset, Dataset, None]:
    """Build datasets for fine-tuning mode."""
    assert config.dataset, "Finetune mode requires a `dataset` block."
    ds_cfg = config.dataset

    # Handle mix datasets
    if isinstance(ds_cfg.hf_name, str) and ds_cfg.hf_name.startswith("mix:"):
        return load_mix_dataset(config)

    # Handle GitHub-sourced datasets
    if isinstance(ds_cfg.hf_name, str) and ds_cfg.hf_name.startswith("github:"):
        train_raw, val_raw = load_from_github(ds_cfg)
        train_ds = build_conversation_dataset(train_raw, config)
        val_ds = build_conversation_dataset(val_raw, config)
        return train_ds, val_ds, None

    # Handle HuggingFace datasets
    train_raw, val_raw, needs_filter = load_hf_dataset(ds_cfg)
    train_raw, val_raw = apply_language_filters(
        train_raw, val_raw, ds_cfg, needs_filter
    )
    train_ds = build_conversation_dataset(train_raw, config)
    val_ds = build_conversation_dataset(val_raw, config)
    return train_ds, val_ds, None


def build_conversation_dataset(raw_ds: Dataset, cfg: ExperimentConfig) -> Dataset:
    """Format a raw dataset into conversation format with messages.

    Args:
        raw_ds: Raw dataset to format
        cfg: Experiment configuration

    Returns:
        Dataset with 'messages' column
    """
    ds_cfg = cfg.dataset
    return apply_templates(raw_ds, ds_cfg)
