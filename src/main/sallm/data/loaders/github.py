from __future__ import annotations

from datasets import Dataset, DatasetDict

from sallm.config import FinetuneDatasetConfig
from sallm.data.afrihg import load_afrihg_from_github
from sallm.data.t2x import load_t2x_from_github


def _extract_splits(ds: Dataset | DatasetDict) -> tuple[Dataset, Dataset]:
    """Extract train and validation splits from a dataset."""
    if isinstance(ds, DatasetDict):
        if "train" in ds:
            train = ds["train"]
        else:
            train = ds[next(iter(ds.keys()))]

        if "validation" in ds:
            val = ds["validation"]
        elif "dev" in ds:
            val = ds["dev"]
        elif "test" in ds:
            val = ds["test"]
        else:
            val = train
        return train, val

    return ds, ds


def load_from_github(ds_cfg: FinetuneDatasetConfig) -> tuple[Dataset, Dataset]:
    """Load dataset from GitHub source.

    Dispatches to afrihg.py or t2x.py based on the hf_name.

    Args:
        ds_cfg: Dataset configuration with hf_name starting with "github:"

    Returns:
        Tuple of (train_ds, val_ds)
    """
    gh_ref = ds_cfg.hf_name[len("github:") :]

    if "francois-meyer/t2x" in gh_ref or gh_ref.strip().endswith("/t2x"):
        ds_from_github = load_t2x_from_github()
    else:
        langs = [ds_cfg.subset] if ds_cfg.subset else ds_cfg.languages
        ds_from_github = load_afrihg_from_github(languages=langs)

    return _extract_splits(ds_from_github)
