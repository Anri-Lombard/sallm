from types import SimpleNamespace

from sallm.config import ExperimentConfig, WandbConfig
from sallm.training import factory
from sallm.utils import RunMode


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


def _build(monkeypatch, training: dict[str, object]) -> _FakeTrainer:
    monkeypatch.setattr(factory, "SFTConfig", _FakeSFTConfig)
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
        [],
        [],
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
