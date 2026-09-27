from omegaconf import OmegaConf
from sallm.config import HubConfig, ModelEvalConfig, to_resolved_dict


def test_to_resolved_dict_converts_nested_omegaconf_containers() -> None:
    value = {"layers": OmegaConf.create([1, 2]), "nested": OmegaConf.create({"x": 3})}

    resolved = to_resolved_dict(value, name="model config")

    assert resolved == {"layers": [1, 2], "nested": {"x": 3}}


def test_model_eval_config_defaults_lm_eval_model_args() -> None:
    config = ModelEvalConfig(checkpoint="owner/model")

    assert config.lm_eval_model_args == {}


def test_model_eval_config_preserves_lm_eval_model_args() -> None:
    config = ModelEvalConfig(
        checkpoint="owner/model",
        lm_eval_model_args={"logits_cache": False},
    )

    assert config.lm_eval_model_args == {"logits_cache": False}


def test_hub_config_accepts_explicit_repo_id() -> None:
    config = HubConfig(repo_id="owner/experiment")

    assert config.repo_id == "owner/experiment"
