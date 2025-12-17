from __future__ import annotations

from datasets import Dataset, DatasetDict, load_from_disk

from sallm.config import ExperimentConfig


def load_pretrain_datasets(
    config: ExperimentConfig, is_hpo: bool
) -> tuple[Dataset, Dataset, Dataset | None]:
    """Load datasets from disk for pretraining mode.

    Args:
        config: Experiment configuration
        is_hpo: Whether this is a hyperparameter optimization run

    Returns:
        Tuple of (train_ds, val_ds, test_ds)
    """
    data_conf = config.data
    dataset_dict = load_from_disk(data_conf.path)

    if not isinstance(dataset_dict, DatasetDict):
        raise TypeError(
            f"Expected DatasetDict at {data_conf.path}, found {type(dataset_dict)}"
        )

    train_ds = dataset_dict[data_conf.train_split]
    val_ds = dataset_dict[data_conf.eval_split]

    test_ds = None
    if not is_hpo and data_conf.test_split and data_conf.test_split in dataset_dict:
        test_ds = dataset_dict[data_conf.test_split]

    return train_ds, val_ds, test_ds
