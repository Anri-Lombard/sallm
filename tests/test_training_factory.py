from types import SimpleNamespace

import pytest
from datasets import Dataset
from sallm.configs import (
    ExperimentConfig,
    FinetuneDatasetConfig,
    FinetuneTaskType,
    RunMode,
    WandbConfig,
)
from sallm.training import factory
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
        self.callbacks = kwargs["callbacks"]


def _build(
    monkeypatch,
    training: dict[str, object],
    *,
    train_dataset=None,
    eval_dataset=None,
    packing: bool = False,
    max_seq_length: int = 1024,
    assistant_only_loss: bool = True,
    sft_config=_FakeSFTConfig,
) -> _FakeTrainer:
    monkeypatch.setattr(factory, "SFTConfig", sft_config)
    monkeypatch.setattr(factory, "CustomSFTTrainer", _FakeTrainer)
    config = ExperimentConfig(
        mode=RunMode.FINETUNE,
        wandb=WandbConfig(project="test", name="test"),
        dataset=FinetuneDatasetConfig(
            hf_name="dummy/ds",
            task=FinetuneTaskType.INSTRUCTION,
            max_seq_length=max_seq_length,
            packing=packing,
            assistant_only_loss=assistant_only_loss,
        ),
        training=training,
    )
    default_ds = Dataset.from_dict({"messages": [[{"role": "user", "content": "hi"}]]})

    return factory.build_trainer(
        config,
        SimpleNamespace(),
        SimpleNamespace(),
        default_ds if train_dataset is None else train_dataset,
        default_ds if eval_dataset is None else eval_dataset,
    )


def test_dataset_sft_settings_reach_sft_config(monkeypatch) -> None:
    trainer = _build(
        monkeypatch,
        {"output_dir": "unused"},
        max_seq_length=2048,
        packing=True,
        assistant_only_loss=False,
    )

    assert trainer.args.max_length == 2048
    assert trainer.args.packing is True
    assert trainer.args.assistant_only_loss is False


def test_selection_options_reach_the_trainer_not_sft_config(monkeypatch) -> None:
    trainer = _build(
        monkeypatch,
        {
            "output_dir": "unused",
            "task_metrics": False,
            "general_selection": True,
        },
    )

    assert trainer.general_selection is True
    assert not hasattr(trainer.args, "task_metrics")


@pytest.mark.parametrize(("rows", "epochs"), [(4999, 10), (5000, 4)])
def test_auto_epochs_follow_training_set_size(monkeypatch, rows, epochs) -> None:
    trainer = _build(
        monkeypatch,
        {
            "output_dir": "unused",
            "num_train_epochs": "auto",
            "bf16": False,
        },
        train_dataset=[{"text": "x"}] * rows,
        sft_config=SFTConfig,
    )

    assert trainer.args.num_train_epochs == epochs
