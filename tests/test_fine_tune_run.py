import pytest
import torch
from sallm.config import (
    ExperimentConfig,
    FinetuneDatasetConfig,
    FinetuneTaskType,
    HubConfig,
    ModelConfig,
)
from sallm.fine_tune.run import (
    _apply_peft_if_needed,
    _build_hub_repo_id,
    _configure_task_truncation,
)
from sallm.utils import RunMode
from transformers import LlamaConfig, LlamaForCausalLM, xLSTMConfig, xLSTMForCausalLM


def _experiment(*, repo_id: str) -> ExperimentConfig:
    return ExperimentConfig(
        mode=RunMode.FINETUNE,
        wandb=None,
        model=ModelConfig(architecture="llama", config={}),
        dataset=FinetuneDatasetConfig(
            hf_name="mix:sa_general",
            max_seq_length=1024,
            packing=False,
            assistant_only_loss=True,
            mix_weights={"sib": 1.0},
        ),
        hub=HubConfig(repo_id=repo_id),
    )


def test_build_hub_repo_id_uses_explicit_id() -> None:
    config = _experiment(repo_id="owner/experiment")

    assert _build_hub_repo_id(config) == "owner/experiment"
    assert _build_hub_repo_id(config, merged=True) == "owner/experiment-merged"


def test_build_hub_repo_id_rejects_invalid_explicit_id() -> None:
    config = _experiment(repo_id="experiment")

    with pytest.raises(ValueError, match="owner/model"):
        _build_hub_repo_id(config)


def test_build_hub_repo_id_derives_a_safe_name_from_the_run() -> None:
    config = ExperimentConfig(
        mode=RunMode.FINETUNE,
        wandb=None,
        model=ModelConfig(architecture="mamba2", config={}),
        dataset=FinetuneDatasetConfig(
            hf_name="github:dadelani/AfriHG",
            languages=["xho", "zul"],
            max_seq_length=1024,
            packing=False,
            assistant_only_loss=True,
        ),
        hub=HubConfig(organization="owner"),
    )

    assert (
        _build_hub_repo_id(config)
        == "owner/sallm-mamba2-github-dadelani-afrihg-xho-zul"
    )
    assert _build_hub_repo_id(config, merged=True).endswith("-xho-zul-merged")


@pytest.mark.parametrize(
    ("task_type", "expected"),
    [
        (FinetuneTaskType.CLASSIFICATION, "left"),
        (FinetuneTaskType.INSTRUCTION, "left"),
        (None, "right"),
    ],
)
def test_configure_task_truncation(
    task_type: FinetuneTaskType | None,
    expected: str,
) -> None:
    tokenizer = type("Tokenizer", (), {"truncation_side": "right"})()

    _configure_task_truncation(tokenizer=tokenizer, task_type=task_type)

    assert tokenizer.truncation_side == expected


def test_apply_peft_trains_only_new_token_rows() -> None:
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=16,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            tie_word_embeddings=True,
        )
    )
    model.resize_token_embeddings(18)
    peft_config = type(
        "PeftConfig",
        (),
        {
            "method": "lora",
            "kwargs": {
                "r": 2,
                "lora_alpha": 4,
                "target_modules": ["q_proj", "v_proj"],
            },
        },
    )()

    adapted = _apply_peft_if_needed(
        model=model,
        peft_cfg=peft_config,
        trainable_token_indices=[16, 17],
    )

    assert adapted.peft_config["default"].trainable_token_indices == [16, 17]
    trainable_names = [
        name for name, param in adapted.named_parameters() if param.requires_grad
    ]
    assert any("token_adapter" in name for name in trainable_names)


def test_apply_peft_combines_embedding_lora_with_new_token_rows() -> None:
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=16,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            tie_word_embeddings=True,
        )
    )
    model.resize_token_embeddings(18)
    peft_config = type(
        "PeftConfig",
        (),
        {
            "method": "lora",
            "kwargs": {
                "r": 2,
                "lora_alpha": 4,
                "target_modules": ["q_proj", "v_proj", "embed_tokens"],
            },
        },
    )()

    adapted = _apply_peft_if_needed(
        model=model,
        peft_cfg=peft_config,
        trainable_token_indices=[16, 17],
    )

    config = adapted.peft_config["default"]
    assert set(config.target_modules) == {"q_proj", "v_proj", "embed_tokens"}
    assert config.trainable_token_indices is None
    trainable_embedding_lora = [
        parameter
        for name, parameter in adapted.named_parameters()
        if "embed_tokens" in name and "lora_embedding" in name
    ]
    assert trainable_embedding_lora
    assert all(parameter.requires_grad for parameter in trainable_embedding_lora)

    loss = adapted(
        input_ids=torch.tensor([[16, 17, 16]]),
        labels=torch.tensor([[16, 17, 16]]),
    ).loss
    loss.backward()
    assert any(parameter.grad is not None for parameter in trainable_embedding_lora)


def test_apply_peft_preserves_xlstm_embedding_lora_targets() -> None:
    model = xLSTMForCausalLM(
        xLSTMConfig(
            vocab_size=16,
            hidden_size=16,
            embedding_dim=16,
            num_hidden_layers=1,
            num_heads=2,
            mode="train",
            chunk_size=4,
            tie_word_embeddings=False,
        )
    )
    model.resize_token_embeddings(18)
    target_modules = ["q", "k", "v", "out_proj", "embeddings"]
    peft_config = type(
        "PeftConfig",
        (),
        {
            "method": "lora",
            "kwargs": {
                "r": 2,
                "lora_alpha": 4,
                "target_modules": target_modules,
            },
        },
    )()

    adapted = _apply_peft_if_needed(
        model=model,
        peft_cfg=peft_config,
        trainable_token_indices=[16, 17],
    )

    config = adapted.peft_config["default"]
    assert set(config.target_modules) == set(target_modules)
    assert config.trainable_token_indices is None
    assert any(
        "backbone.embeddings" in name and "lora_embedding" in name
        for name, parameter in adapted.named_parameters()
        if parameter.requires_grad
    )
