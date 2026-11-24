import logging
from typing import Any, cast

import torch
from tokenizers.decoders import ByteLevel
from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from sallm.configs import ExperimentConfig, ModelConfig, TokenizerConfig
from sallm.models.registry import (
    MODEL_CLASS_REGISTRY,
    architecture_profile,
    patch_mamba2_grouped_rmsnorm,
    tie_xlstm_embeddings,
)

logger = logging.getLogger(__name__)


def _use_tokenizer_special_tokens(
    target: Any, tokenizer: PreTrainedTokenizerBase
) -> None:
    """Copy BOS/EOS/PAD ids from the tokenizer; architecture defaults differ."""
    if target is None:
        return
    for name in ("bos", "eos", "pad"):
        token_id = getattr(tokenizer, f"{name}_token_id", None)
        if token_id is None:
            continue
        current = getattr(target, f"{name}_token_id", None)
        if current is not None and current != token_id:
            logger.warning(
                "Replacing %s_token_id %s with the tokenizer's %s.",
                name,
                current,
                token_id,
            )
        setattr(target, f"{name}_token_id", token_id)


def build_tokenizer(config: ExperimentConfig) -> PreTrainedTokenizerBase:
    tokenizer_conf: TokenizerConfig | None = config.tokenizer
    if tokenizer_conf is None:
        raise ValueError("ExperimentConfig.tokenizer is required")
    tokenizer = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(tokenizer_conf.path),
    )
    backend_tokenizer = getattr(tokenizer, "backend_tokenizer", None)
    if backend_tokenizer is not None:
        backend_tokenizer.decoder = ByteLevel()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(
            "tokenizer.pad_token was not set, setting it to eos_token: %s",
            tokenizer.eos_token,
        )

    return tokenizer


def build_model(
    config: ExperimentConfig, tokenizer: PreTrainedTokenizerBase
) -> PreTrainedModel:
    model_conf: ModelConfig | None = config.model
    if model_conf is None:
        raise ValueError("ExperimentConfig.model is required")
    model_class = MODEL_CLASS_REGISTRY.get(model_conf.architecture)
    if not model_class:
        raise ValueError(f"Unsupported model architecture: {model_conf.architecture}")
    if not getattr(model_conf, "init_checkpoint", None):
        raise ValueError("`model.init_checkpoint` is required for fine-tuning.")

    logger.info(
        "Loading model of type %s from checkpoint: %s",
        model_class.__name__,
        model_conf.init_checkpoint,
    )
    attn_impl = getattr(model_conf, "attn_implementation", None)
    # Weights stay float32 when fine-tuning: bf16/fp16 in training config means
    # autocast, whereas pure-bf16 weights underflow small fine-tuning updates.
    model = cast(
        PreTrainedModel,
        model_class.from_pretrained(
            model_conf.init_checkpoint,
            attn_implementation=attn_impl,
            dtype=torch.float32,
        ),
    )
    profile = architecture_profile(model.config, architecture=model_conf.architecture)
    if not profile.accepts_loss_kwargs:
        cast(Any, model).accepts_loss_kwargs = False
    if profile.needs_mamba2_grouped_norm_patch:
        patch_mamba2_grouped_rmsnorm(model)
    tie_xlstm_embeddings(model)
    _use_tokenizer_special_tokens(model.config, tokenizer)
    _use_tokenizer_special_tokens(getattr(model, "generation_config", None), tokenizer)
    return model
