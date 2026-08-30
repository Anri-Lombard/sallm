from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml
from huggingface_hub.errors import StrictDataclassClassValidationError
from peft import LoraConfig, get_peft_model
from sallm.config import ModelEvalConfig
from sallm.evaluation.harness import load_model_and_tokenizer
from sallm.evaluation.lm_eval_runner import _materialize_model_for_lm_eval
from sallm.models.registry import MODEL_CLASS_REGISTRY, MODEL_CONFIG_REGISTRY
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

CONF_ROOT = Path(__file__).resolve().parents[1] / "src" / "conf"
BASE_CONFIG_PATHS = sorted((CONF_ROOT / "base").glob("*.yaml"))


def _checked_in_llama_config(name: str) -> dict[str, object]:
    document = yaml.safe_load((CONF_ROOT / "base" / name).read_text())
    return document["model"]["config"]


def _tiny_legacy_model():
    config_values = _checked_in_llama_config("llama_125m.yaml")
    config_values.update(
        vocab_size=4,
        intermediate_size=32,
        num_hidden_layers=1,
    )
    config = MODEL_CONFIG_REGISTRY["llama"](**config_values)
    return MODEL_CLASS_REGISTRY["llama"](config)


def _save_tokenizer(checkpoint: Path) -> PreTrainedTokenizerFast:
    backend = Tokenizer(
        WordLevel(
            {"[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "hello": 3},
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
    )
    tokenizer.save_pretrained(checkpoint)
    return tokenizer


@pytest.mark.parametrize("config_path", BASE_CONFIG_PATHS, ids=lambda path: path.stem)
def test_checked_in_base_model_config_constructs(config_path: Path) -> None:
    document = yaml.safe_load(config_path.read_text())
    model_values = document["model"]

    config_class = MODEL_CONFIG_REGISTRY[model_values["architecture"]]
    config_class(**model_values["config"])


def test_checked_in_llama_configs_construct_with_frozen_projection_shapes() -> None:
    legacy_values = _checked_in_llama_config("llama_125m.yaml")
    legacy_config = MODEL_CONFIG_REGISTRY["llama"](**legacy_values)

    assert legacy_config.head_dim == 56

    model = _tiny_legacy_model()
    attention = model.model.layers[0].self_attn
    assert attention.q_proj.weight.shape == (504, 512)
    assert attention.k_proj.weight.shape == (168, 512)
    assert attention.v_proj.weight.shape == (168, 512)
    assert attention.o_proj.weight.shape == (512, 504)

    standard_values = _checked_in_llama_config("llama_400m.yaml")
    standard_config = MODEL_CONFIG_REGISTRY["llama"](**standard_values)
    assert standard_config.head_dim == 128


def test_legacy_llama_checkpoint_saves_and_reloads_through_auto_model(
    tmp_path: Path,
) -> None:
    model = _tiny_legacy_model()
    checkpoint = tmp_path / "checkpoint"
    model.save_pretrained(checkpoint)

    saved_config = json.loads((checkpoint / "config.json").read_text())
    assert saved_config["architectures"] == ["LlamaForCausalLM"]
    assert saved_config["hidden_size"] == 512
    assert saved_config["head_dim"] == 56

    reloaded = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        local_files_only=True,
    )
    assert reloaded.model.layers[0].self_attn.q_proj.weight.shape == (504, 512)
    assert torch.equal(
        model.model.layers[0].self_attn.q_proj.weight,
        reloaded.model.layers[0].self_attn.q_proj.weight,
    )


def test_legacy_llama_checkpoint_loads_in_harness_and_lm_eval_materialization(
    tmp_path: Path,
) -> None:
    model = _tiny_legacy_model()
    checkpoint = tmp_path / "checkpoint"
    model.save_pretrained(checkpoint)
    _save_tokenizer(checkpoint)

    model_config = ModelEvalConfig(
        checkpoint=str(checkpoint),
        dtype="float32",
        device="cpu",
    )
    loaded, _ = load_model_and_tokenizer(model_config)
    assert loaded.model.layers[0].self_attn.q_proj.weight.shape == (504, 512)

    adapter = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"],
            r=2,
            lora_alpha=4,
        ),
    )
    adapter_path = tmp_path / "adapter"
    adapter.save_pretrained(adapter_path)
    merged_path, remaining_adapter = _materialize_model_for_lm_eval(
        ModelEvalConfig(
            checkpoint=str(checkpoint),
            peft_adapter=str(adapter_path),
            dtype="float32",
            device="cpu",
        ),
        tmp_path / "lm_eval",
    )

    assert remaining_adapter is None
    merged = AutoModelForCausalLM.from_pretrained(
        merged_path,
        local_files_only=True,
    )
    assert merged.model.layers[0].self_attn.q_proj.weight.shape == (504, 512)


def test_other_non_divisible_llama_configs_still_fail_validation() -> None:
    with pytest.raises(
        StrictDataclassClassValidationError,
        match="not a multiple of the number of attention heads",
    ):
        MODEL_CONFIG_REGISTRY["llama"](
            hidden_size=510,
            num_attention_heads=9,
            num_key_value_heads=3,
            head_dim=56,
        )
