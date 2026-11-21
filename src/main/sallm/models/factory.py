import logging
from typing import Any, cast

import torch
from tokenizers.decoders import ByteLevel
from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from sallm.config import (
    ExperimentConfig,
    ModelConfig,
    TokenizerConfig,
    to_resolved_dict,
)
from sallm.models.registry import (
    MODEL_CLASS_REGISTRY,
    MODEL_CONFIG_REGISTRY,
    architecture_profile,
    patch_mamba2_grouped_rmsnorm,
    tie_xlstm_embeddings,
    use_exact_xlstm_head_dims,
    validate_xlstm_pretrain_kernels,
)
from sallm.utils import RunMode, count_trainable_parameters

logger = logging.getLogger(__name__)


def _get_torch_dtype(config: ExperimentConfig) -> torch.dtype:
    """Determine torch dtype from training config."""
    training = config.training
    if training is None:
        return torch.float32

    if isinstance(training, dict):
        if training.get("bf16"):
            return torch.bfloat16
        if training.get("fp16"):
            return torch.float16
    else:
        if getattr(training, "bf16", False):
            return torch.bfloat16
        if getattr(training, "fp16", False):
            return torch.float16

    return torch.float32


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

    if getattr(model_conf, "init_checkpoint", None):
        logger.info(
            "Loading model of type %s from checkpoint: %s",
            model_class.__name__,
            model_conf.init_checkpoint,
        )
        attn_impl = getattr(config.model, "attn_implementation", None)
        # Weights stay float32 when training from a checkpoint: bf16/fp16 in the
        # training config means autocast. Pure-bf16 weights and AdamW state lose
        # fine-tuning updates (an lr of 1e-5 underflows bf16 precision).
        torch_dtype = torch.float32
        model = cast(
            PreTrainedModel,
            model_class.from_pretrained(
                model_conf.init_checkpoint,
                attn_implementation=attn_impl,
                torch_dtype=torch_dtype,
            ),
        )
        profile = architecture_profile(
            model.config, architecture=model_conf.architecture
        )
        if not profile.accepts_loss_kwargs:
            cast(Any, model).accepts_loss_kwargs = False
        if profile.needs_mamba2_grouped_norm_patch:
            patch_mamba2_grouped_rmsnorm(model)
        tie_xlstm_embeddings(model)
        _use_tokenizer_special_tokens(model.config, tokenizer)
        _use_tokenizer_special_tokens(
            getattr(model, "generation_config", None), tokenizer
        )
        return model

    config_class = MODEL_CONFIG_REGISTRY[model_conf.architecture]
    if model_conf.config is None:
        raise ValueError(
            "`model.config` is required when `init_checkpoint` is not provided."
        )

    model_config = to_resolved_dict(model_conf.config, name="model config")
    # Pop Triton kernel overrides for xLSTM so transformers 5's @strict validator
    # does not reject them before mlstm_kernels is loaded.
    xlstm_kernel_overrides: dict[str, Any] = {}
    if model_conf.architecture == "xlstm":
        for key in ("chunkwise_kernel", "sequence_kernel", "step_kernel"):
            if key in model_config:
                xlstm_kernel_overrides[key] = model_config.pop(key)
    model_config_obj = config_class(**model_config)
    for key, value in xlstm_kernel_overrides.items():
        object.__setattr__(model_config_obj, key, value)
    use_exact_xlstm_head_dims(model_config_obj)
    if (
        config.mode == RunMode.TRAIN
        and architecture_profile(model_config_obj).has_train_and_inference_modes
    ):
        validate_xlstm_pretrain_kernels(
            model_config_obj,
            allow_native=not torch.cuda.is_available(),
        )
    model_config_obj.vocab_size = len(tokenizer)
    _use_tokenizer_special_tokens(model_config_obj, tokenizer)

    torch_dtype = _get_torch_dtype(config)
    model_config_obj.dtype = torch_dtype
    logger.info(f"Creating model with torch_dtype={torch_dtype}")
    model = cast(PreTrainedModel, model_class(model_config_obj).to(torch_dtype))

    profile = architecture_profile(model.config, architecture=model_conf.architecture)
    if not profile.accepts_loss_kwargs:
        cast(Any, model).accepts_loss_kwargs = False
    if profile.needs_mamba2_grouped_norm_patch:
        patch_mamba2_grouped_rmsnorm(model)
    tie_xlstm_embeddings(model)

    if model_conf.param_validation:
        num_params = count_trainable_parameters(model)
        num_params_m = num_params / 1_000_000

        min_p = model_conf.param_validation.min_params_m
        max_p = model_conf.param_validation.max_params_m

        logger.info(f"Validating model size: {num_params_m:.2f}M parameters.")

        if not (min_p <= num_params_m <= max_p):
            raise ValueError(
                f"Model size validation failed! "
                f"Expected between {min_p}M and {max_p}M parameters, "
                f"but got {num_params_m:.2f}M."
            )
        logger.info("Model size validation passed.")

    return model
