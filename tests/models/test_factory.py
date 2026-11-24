from types import SimpleNamespace

import pytest
import torch
from sallm.configs import ExperimentConfig, ModelConfig, RunMode, WandbConfig
from sallm.models import factory


class FakeCheckpointModel:
    last_from_pretrained_kwargs: dict[str, object] | None = None

    def __init__(self) -> None:
        # The ids transformers' Mamba2Config defaults to: EOS and PAD swapped.
        self.config = SimpleNamespace(bos_token_id=0, eos_token_id=2, pad_token_id=1)
        self.generation_config = SimpleNamespace(
            bos_token_id=0, eos_token_id=2, pad_token_id=1
        )

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.last_from_pretrained_kwargs = kwargs
        return cls()


class FakeTokenizer:
    # The SALLM tokenizer: [BOS]=0, [EOS]=1, [PAD]=2.
    bos_token_id = 0
    eos_token_id = 1
    pad_token_id = 2

    def __len__(self) -> int:
        return 128


def test_build_model_loads_checkpoint_in_float32_and_marks_pure_gdn(
    monkeypatch,
) -> None:
    config = ExperimentConfig(
        mode=RunMode.FINETUNE,
        wandb=WandbConfig(project="test", name="test"),
        model=ModelConfig(
            architecture="gated_deltanet",
            init_checkpoint="checkpoint",
        ),
    )
    monkeypatch.setattr(
        factory,
        "MODEL_CLASS_REGISTRY",
        {"gated_deltanet": FakeCheckpointModel},
    )

    model = factory.build_model(config, FakeTokenizer())

    assert model.accepts_loss_kwargs is False
    assert FakeCheckpointModel.last_from_pretrained_kwargs is not None
    assert FakeCheckpointModel.last_from_pretrained_kwargs["dtype"] is torch.float32


def test_checkpoint_model_takes_special_token_ids_from_tokenizer(monkeypatch) -> None:
    config = ExperimentConfig(
        mode=RunMode.FINETUNE,
        wandb=WandbConfig(project="test", name="test"),
        model=ModelConfig(architecture="mamba2", init_checkpoint="checkpoint"),
    )
    monkeypatch.setattr(
        factory, "MODEL_CLASS_REGISTRY", {"mamba2": FakeCheckpointModel}
    )

    model = factory.build_model(config, FakeTokenizer())

    for target in (model.config, model.generation_config):
        assert (target.bos_token_id, target.eos_token_id, target.pad_token_id) == (
            0,
            1,
            2,
        )


def test_build_model_requires_init_checkpoint(monkeypatch) -> None:
    config = ExperimentConfig(
        mode=RunMode.FINETUNE,
        wandb=WandbConfig(project="test", name="test"),
        model=ModelConfig(architecture="mamba2", init_checkpoint=""),
    )
    monkeypatch.setattr(
        factory, "MODEL_CLASS_REGISTRY", {"mamba2": FakeCheckpointModel}
    )

    with pytest.raises(ValueError, match="init_checkpoint"):
        factory.build_model(config, FakeTokenizer())
