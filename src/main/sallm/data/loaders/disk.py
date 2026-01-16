from __future__ import annotations

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

from sallm.config import ExperimentConfig


def load_pretrain_datasets(
    config: ExperimentConfig, is_hpo: bool
) -> tuple[Dataset, Dataset, Dataset | None]:
    """Load datasets for pretraining mode.

    Supports loading from local disk (Arrow format) or HuggingFace Hub.

    Args:
        config: Experiment configuration
        is_hpo: Whether this is a hyperparameter optimization run

    Returns:
        Tuple of (train_ds, val_ds, test_ds)
    """
    data_conf = config.data
    if data_conf is None:
        raise ValueError("data config is required for pretraining")

    if data_conf.hf_name:
        dataset_dict = load_dataset(data_conf.hf_name)
    elif data_conf.path:
        dataset_dict = load_from_disk(data_conf.path)
    else:
        raise ValueError("Either hf_name or path must be provided in data config")

    if not isinstance(dataset_dict, DatasetDict):
        source = data_conf.hf_name or data_conf.path
        raise TypeError(
            f"Expected DatasetDict from {source}, found {type(dataset_dict)}"
        )

    train_ds = dataset_dict[data_conf.train_split]
    val_ds = dataset_dict[data_conf.eval_split]

    test_ds = None
    if not is_hpo and data_conf.test_split and data_conf.test_split in dataset_dict:
        test_ds = dataset_dict[data_conf.test_split]

    return train_ds, val_ds, test_ds
