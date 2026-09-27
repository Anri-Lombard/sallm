from types import SimpleNamespace

import torch
from omegaconf import OmegaConf
from sallm.config import ExperimentConfig, ModelConfig, WandbConfig
from sallm.models import factory
from sallm.utils import RunMode
from transformers import Mamba2Config


class FakeModelConfig:
    def __init__(self, layer_types: list[str]):
        self.layer_types = layer_types
        self.vocab_size = 0


class FakeModel:
    last_config: FakeModelConfig | None = None

    def __init__(self, config: FakeModelConfig):
        self.config = config
        FakeModel.last_config = config

    def to(self, dtype):
        self.dtype = dtype
        return self


class FakeCheckpointModel:
    def __init__(self) -> None:
        # The ids transformers' Mamba2Config defaults to: EOS and PAD swapped.
        self.config = SimpleNamespace(bos_token_id=0, eos_token_id=2, pad_token_id=1)
        self.generation_config = SimpleNamespace(
            bos_token_id=0, eos_token_id=2, pad_token_id=1
        )

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()


class FakeTokenizer:
    # The SALLM tokenizer: [BOS]=0, [EOS]=1, [PAD]=2.
    bos_token_id = 0
    eos_token_id = 1
    pad_token_id = 2

    def __len__(self) -> int:
        return 128


def test_build_model_resolves_nested_omegaconf_model_config(monkeypatch) -> None:
    config = ExperimentConfig(
        mode=RunMode.TRAIN,
        wandb=WandbConfig(project="test", name="test"),
        model=ModelConfig(
            architecture="fake",
            config=OmegaConf.create({"layer_types": ["linear_attention"]}),
        ),
        training={},
    )
    monkeypatch.setattr(factory, "MODEL_CONFIG_REGISTRY", {"fake": FakeModelConfig})
    monkeypatch.setattr(factory, "MODEL_CLASS_REGISTRY", {"fake": FakeModel})

    model = factory.build_model(config, FakeTokenizer())

    assert FakeModel.last_config is not None
    assert FakeModel.last_config.layer_types == ["linear_attention"]
    assert type(FakeModel.last_config.layer_types) is list
    assert FakeModel.last_config.vocab_size == 128
    assert FakeModel.last_config.dtype is torch.float32
    assert not hasattr(model, "accepts_loss_kwargs")


def test_build_model_marks_scratch_pure_gdn_as_not_accepting_loss_kwargs(
    monkeypatch,
) -> None:
    config = ExperimentConfig(
        mode=RunMode.TRAIN,
        wandb=WandbConfig(project="test", name="test"),
        model=ModelConfig(
            architecture="gated_deltanet",
            config={"layer_types": ["linear_attention"]},
        ),
    )
    monkeypatch.setattr(
        factory,
        "MODEL_CONFIG_REGISTRY",
        {"gated_deltanet": FakeModelConfig},
    )
    monkeypatch.setattr(factory, "MODEL_CLASS_REGISTRY", {"gated_deltanet": FakeModel})

    model = factory.build_model(config, FakeTokenizer())

    assert model.accepts_loss_kwargs is False


def test_build_model_marks_checkpoint_pure_gdn_as_not_accepting_loss_kwargs(
    monkeypatch,
) -> None:
    config = ExperimentConfig(
        mode=RunMode.TRAIN,
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


def test_scratch_model_takes_special_token_ids_from_tokenizer(monkeypatch) -> None:
    config = ExperimentConfig(
        mode=RunMode.TRAIN,
        wandb=WandbConfig(project="test", name="test"),
        model=ModelConfig(architecture="mamba2", config={"num_hidden_layers": 1}),
    )
    monkeypatch.setattr(factory, "MODEL_CONFIG_REGISTRY", {"mamba2": Mamba2Config})
    monkeypatch.setattr(factory, "MODEL_CLASS_REGISTRY", {"mamba2": FakeModel})

    factory.build_model(config, FakeTokenizer())

    model_config = FakeModel.last_config
    assert (
        model_config.bos_token_id,
        model_config.eos_token_id,
        model_config.pad_token_id,
    ) == (0, 1, 2)


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
