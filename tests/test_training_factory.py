import json
from types import SimpleNamespace

import pytest
from datasets import IterableDataset
from sallm.config import ExperimentConfig, WandbConfig
from sallm.training import factory
from sallm.utils import RunMode
from trl import SFTConfig


class _FakeSFTConfig:
    def __init__(
        self,
        *,
        output_dir: str,
        max_length: int,
        packing: bool,
        assistant_only_loss: bool,
        packing_strategy: str = "bfd",
        padding_free: bool = False,
    ) -> None:
        self.output_dir = output_dir
        self.max_length = max_length
        self.packing = packing
        self.assistant_only_loss = assistant_only_loss
        self.packing_strategy = packing_strategy
        self.padding_free = padding_free
        self.gradient_checkpointing = False
        self.gradient_checkpointing_kwargs = None
        self.local_rank = 0
        self.world_size = 1


class _FakeTrainer:
    def __init__(self, **kwargs) -> None:
        self.args = kwargs["args"]
        self.processing_class = kwargs["processing_class"]
        self.general_selection = kwargs["general_selection"]


def _build(
    monkeypatch,
    training: dict[str, object],
    train_dataset=None,
    eval_dataset=None,
    sft_config=_FakeSFTConfig,
) -> _FakeTrainer:
    monkeypatch.setattr(factory, "SFTConfig", sft_config)
    monkeypatch.setattr(factory, "CustomSFTTrainer", _FakeTrainer)
    config = ExperimentConfig(
        mode=RunMode.TRAIN,
        wandb=WandbConfig(project="test", name="test"),
        training=training,
    )

    return factory.build_trainer(
        config,
        SimpleNamespace(),
        SimpleNamespace(),
        [] if train_dataset is None else train_dataset,
        [] if eval_dataset is None else eval_dataset,
    )


def test_pretraining_packing_reaches_sft_config(monkeypatch) -> None:
    trainer = _build(
        monkeypatch,
        {"output_dir": "unused", "max_length": 2048, "packing": True},
    )

    assert trainer.args.packing is True
    assert trainer.args.max_length == 2048


def test_pretraining_packing_defaults_to_false(monkeypatch) -> None:
    trainer = _build(monkeypatch, {"output_dir": "unused", "max_length": 2048})

    assert trainer.args.packing is False


def test_selection_options_reach_the_trainer_not_sft_config(monkeypatch) -> None:
    trainer = _build(
        monkeypatch,
        {
            "output_dir": "unused",
            "max_length": 2048,
            "task_metrics": False,
            "general_selection": True,
        },
    )

    assert trainer.general_selection is True
    assert not hasattr(trainer.args, "task_metrics")


def test_pretraining_non_flattening_packing_options_reach_sft_config(
    monkeypatch,
) -> None:
    trainer = _build(
        monkeypatch,
        {
            "output_dir": "unused",
            "max_length": 2048,
            "packing": True,
            "packing_strategy": "wrapped",
            "padding_free": False,
        },
    )

    assert trainer.args.packing_strategy == "wrapped"
    assert trainer.args.padding_free is False


@pytest.mark.parametrize("iterable_dataset", ["train", "eval"])
def test_iterable_dataset_defaults_dispatch_batches_to_false(
    monkeypatch, iterable_dataset: str
) -> None:
    dataset = IterableDataset.from_generator(lambda: iter(()))
    trainer = _build(
        monkeypatch,
        {"output_dir": "unused", "max_length": 2048, "use_cpu": True},
        train_dataset=dataset if iterable_dataset == "train" else None,
        eval_dataset=dataset if iterable_dataset == "eval" else None,
        sft_config=SFTConfig,
    )

    assert trainer.args.accelerator_config.dispatch_batches is False


def test_iterable_dataset_preserves_explicit_dispatch_and_accelerator_options(
    monkeypatch, tmp_path
) -> None:
    dataset = IterableDataset.from_generator(lambda: iter(()))
    explicit_trainer = _build(
        monkeypatch,
        {
            "output_dir": "unused",
            "max_length": 2048,
            "use_cpu": True,
            "accelerator_config": {
                "dispatch_batches": True,
                "split_batches": True,
            },
        },
        train_dataset=dataset,
        sft_config=SFTConfig,
    )

    assert explicit_trainer.args.accelerator_config.dispatch_batches is True
    assert explicit_trainer.args.accelerator_config.split_batches is True

    config_path = tmp_path / "accelerator_config.json"
    config_path.write_text(json.dumps({"split_batches": True}))
    path_trainer = _build(
        monkeypatch,
        {
            "output_dir": "unused",
            "max_length": 2048,
            "use_cpu": True,
            "accelerator_config": str(config_path),
        },
        train_dataset=dataset,
        sft_config=SFTConfig,
    )

    assert path_trainer.args.accelerator_config.dispatch_batches is False
    assert path_trainer.args.accelerator_config.split_batches is True


@pytest.mark.parametrize(("rows", "epochs"), [(4999, 10), (5000, 4)])
def test_auto_epochs_follow_training_set_size(monkeypatch, rows, epochs) -> None:
    trainer = _build(
        monkeypatch,
        {"output_dir": "unused", "max_length": 2048, "num_train_epochs": "auto"},
        train_dataset=[{"text": "x"}] * rows,
        sft_config=SFTConfig,
    )

    assert trainer.args.num_train_epochs == epochs
