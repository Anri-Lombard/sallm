from typing import Optional, Tuple

from datasets import Dataset, DatasetDict, load_from_disk

from sallm.config import ExperimentConfig


def build_datasets(
    config: ExperimentConfig, is_hpo: bool
) -> Tuple[Dataset, Dataset, Optional[Dataset]]:
    """
    Loads a single DatasetDict and extracts the specified splits.
    If in HPO mode, the test set is explicitly not loaded to prevent data leakage.
    """
    data_conf = config.data

    dataset_dict = load_from_disk(data_conf.path)

    if not isinstance(dataset_dict, DatasetDict):
        raise TypeError(
            f"Expected data at {data_conf.path} to be a DatasetDict, "
            f"but found {type(dataset_dict)}"
        )

    train_dataset = dataset_dict[data_conf.train_split]
    eval_dataset = dataset_dict[data_conf.eval_split]

    test_dataset = None
    if not is_hpo and data_conf.test_split and data_conf.test_split in dataset_dict:
        test_dataset = dataset_dict[data_conf.test_split]

    return train_dataset, eval_dataset, test_dataset
