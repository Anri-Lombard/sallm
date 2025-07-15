from __future__ import annotations

from typing import Optional, Tuple, Any, Dict, List

from datasets import Dataset, DatasetDict, load_from_disk, load_dataset
from transformers import AutoTokenizer

from sallm.config import ExperimentConfig, RunMode
from sallm.templates import registry as tmpl


def build_datasets(
    config: ExperimentConfig, tokenizer: AutoTokenizer, is_hpo: bool
) -> Tuple[Dataset, Dataset, Optional[Dataset]]:
    if config.mode == RunMode.FINETUNE:
        assert config.dataset, "Finetune mode requires a `dataset` block in the config."

        ds_cfg = config.dataset
        split_map = ds_cfg.splits

        train_raw = load_dataset(
            ds_cfg.hf_name,
            ds_cfg.subset,
            split=split_map["train"],
            trust_remote_code=True,
        )
        val_raw = load_dataset(
            ds_cfg.hf_name,
            ds_cfg.subset,
            split=split_map["val"],
            trust_remote_code=True,
        )

        train_ds = _build_finetune_dataset(train_raw, config)
        val_ds = _build_finetune_dataset(val_raw, config)
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
) -> Dataset:
    ds_cfg = cfg.dataset
    template_spec = tmpl.get(ds_cfg.templates[0].id)
    label_column = getattr(ds_cfg, "label_column", "label")
    numeric_keys = isinstance(next(iter(template_spec.label_mapping.keys())), int)

    def to_messages_format(ex: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]:
        # Automatically detects the text columns we need to use
        user_prompt = template_spec.prompt.format(**ex)

        raw_label = ex[label_column]
        assistant_response = template_spec.label_mapping[
            int(raw_label) if numeric_keys else str(raw_label)
        ]

        return {
            "messages": [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_response},
            ]
        }

    processed_ds = raw_ds.map(
        to_messages_format,
        batched=False,
        remove_columns=raw_ds.column_names,
        desc="Formatting dataset into 'messages' format",
    )

    if ds_cfg.subset:
        processed_ds = processed_ds.add_column(
            "lang", [ds_cfg.subset] * len(processed_ds)
        )

    return processed_ds
