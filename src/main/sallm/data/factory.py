from __future__ import annotations

from typing import Optional, Tuple

from datasets import Dataset, DatasetDict, load_from_disk, load_dataset
from transformers import AutoTokenizer

from sallm.config import ExperimentConfig, RunMode
from sallm.data.utils import make_example_mapper


def build_datasets(
    config: ExperimentConfig, is_hpo: bool
) -> Tuple[Dataset, Dataset, Optional[Dataset]]:
    if config.mode == RunMode.FINETUNE:
        assert config.dataset, "Finetune mode requires a `dataset` block in the config."

        ds_cfg = config.dataset
        split_map = ds_cfg.splits

        # TODO: specify split in config rather
        train_raw = load_dataset(
            ds_cfg.hf_name,
            ds_cfg.subset,
            split=split_map["train"],
            trust_remote_code=True,
        )
        # TODO: specify split in config rather
        val_raw = load_dataset(
            ds_cfg.hf_name,
            ds_cfg.subset,
            split=split_map["val"],
            trust_remote_code=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.path)

        train_ds = _build_finetune_dataset(train_raw, config, tokenizer)
        val_ds = _build_finetune_dataset(val_raw, config, tokenizer)
        return train_ds, val_ds, None

    data_conf = config.data
    dataset_dict = load_from_disk(data_conf.path)

    if not isinstance(dataset_dict, DatasetDict):
        raise TypeError(
            f"Expected data at {data_conf.path} to be a DatasetDict, "
            f"but found {type(dataset_dict)}"
        )

    train_ds = dataset_dict[data_conf.train_split]
    val_ds = dataset_dict[data_conf.eval_split]

    test_ds = None
    if not is_hpo and data_conf.test_split and data_conf.test_split in dataset_dict:
        test_ds = dataset_dict[data_conf.test_split]

    return train_ds, val_ds, test_ds


def _build_finetune_dataset(
    raw_ds: Dataset,
    cfg: ExperimentConfig,
    tokenizer: AutoTokenizer,
) -> Dataset:
    ds_cfg = cfg.dataset
    mapper = make_example_mapper(ds_cfg, tokenizer)
    processed_ds = raw_ds.map(
        mapper,
        batched=False,
        remove_columns=raw_ds.column_names,
        desc="Mapping prompt+label → LM inputs",
    )

    if ds_cfg.subset:
        lang_column = [ds_cfg.subset] * len(processed_ds)
        processed_ds = processed_ds.add_column("lang", lang_column)

    return processed_ds
