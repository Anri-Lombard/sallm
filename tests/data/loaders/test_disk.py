from datasets import Features, IterableDataset, IterableDatasetDict, List, Value
from sallm.config import (
    DataConfig,
    ExperimentConfig,
    TokenizerConfig,
    WandbConfig,
)
from sallm.data.loaders import disk
from sallm.utils import RunMode


def test_features_from_dict_supports_hub_list_value_metadata() -> None:
    metadata = {
        "input_ids": {
            "_type": "List",
            "feature": {"_type": "Value", "dtype": "int64"},
        }
    }

    assert Features.from_dict(metadata) == Features({"input_ids": List(Value("int64"))})


def test_load_pretrain_datasets_streams_hub_dataset(monkeypatch) -> None:
    def rows():
        return iter([{"input_ids": [1], "lang": "eng"}])

    dataset = IterableDatasetDict(
        {
            "train": IterableDataset.from_generator(rows),
            "validation": IterableDataset.from_generator(rows),
        }
    )

    def load_dataset(name: str, *, streaming: bool):
        assert name == "owner/dataset"
        assert streaming is True
        return dataset

    monkeypatch.setattr(disk, "load_dataset", load_dataset)
    config = ExperimentConfig(
        mode=RunMode.TRAIN,
        wandb=WandbConfig(project="test"),
        data=DataConfig(hf_name="owner/dataset", streaming=True, test_split=None),
        tokenizer=TokenizerConfig(path="tokenizer"),
    )

    train, validation, test = disk.load_pretrain_datasets(config, is_hpo=False)

    assert train is dataset["train"]
    assert validation is dataset["validation"]
    assert test is None
