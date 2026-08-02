import torch
from omegaconf import OmegaConf
from sallm.config import ExperimentConfig, ModelConfig, WandbConfig
from sallm.models import factory
from sallm.utils import RunMode


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


class FakeTokenizer:
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

    factory.build_model(config, FakeTokenizer())

    assert FakeModel.last_config is not None
    assert FakeModel.last_config.layer_types == ["linear_attention"]
    assert type(FakeModel.last_config.layer_types) is list
    assert FakeModel.last_config.vocab_size == 128
    assert FakeModel.last_config.dtype is torch.float32
