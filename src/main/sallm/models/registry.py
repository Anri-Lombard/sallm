from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib import import_module
from typing import Any, cast

import torch
import torch.nn.functional as F

from sallm.models.llama_compatibility import LlamaConfig, LlamaForCausalLM

logger = logging.getLogger(__name__)

type ModuleClassTarget = str | tuple[str, str]

_FLA_INSTALL_MESSAGE = (
    "FLA architectures (`gated_deltanet`, `hybrid`, `fla_mamba2`) require "
    "`flash-linear-attention>=0.5.2`. On Linux, install it with "
    "`uv sync --extra pure-gdn --frozen --inexact`."
)

_MAMBA2_NORM_PATCHED = False


class LazyRegistry(dict):
    """Dict that lazily imports configured classes on first access."""

    def __init__(self, mappings: dict[str, ModuleClassTarget]):
        """Map architecture keys to a class or explicit ``(module, class)`` target."""
        super().__init__()
        self._mappings = mappings

    def __getitem__(self, key: str):
        if key not in dict.keys(self):
            if key not in self._mappings:
                raise KeyError(key)
            target = self._mappings[key]
            module_name, class_name = (
                target if isinstance(target, tuple) else ("transformers", target)
            )
            try:
                module = import_module(module_name)
            except ModuleNotFoundError as exc:
                if module_name.startswith("fla.") and exc.name == "fla":
                    raise ModuleNotFoundError(_FLA_INSTALL_MESSAGE) from exc
                raise
            dict.__setitem__(self, key, getattr(module, class_name))
        return dict.__getitem__(self, key)

    def __contains__(self, key: object) -> bool:
        return key in self._mappings

    def get(self, key: str, default=None):
        if key not in self._mappings:
            return default
        return self[key]


MODEL_CONFIG_REGISTRY = LazyRegistry(
    {
        "llama": "LlamaConfig",
        "mistral": "MistralConfig",
        "gated_deltanet": (
            "fla.models.gated_deltanet",
            "GatedDeltaNetConfig",
        ),
        "hybrid": (
            "fla.models.gated_deltanet",
            "GatedDeltaNetConfig",
        ),
        "fla_mamba2": (
            "fla.models.mamba2",
            "Mamba2Config",
        ),
        "mamba2": "Mamba2Config",
        "xlstm": "xLSTMConfig",
    }
)

MODEL_CLASS_REGISTRY = LazyRegistry(
    {
        "llama": "LlamaForCausalLM",
        "mistral": "MistralForCausalLM",
        "gated_deltanet": (
            "fla.models.gated_deltanet",
            "GatedDeltaNetForCausalLM",
        ),
        "hybrid": (
            "fla.models.gated_deltanet",
            "GatedDeltaNetForCausalLM",
        ),
        "fla_mamba2": (
            "fla.models.mamba2",
            "Mamba2ForCausalLM",
        ),
        "mamba2": "Mamba2ForCausalLM",
        "xlstm": "xLSTMForCausalLM",
    }
)

dict.__setitem__(MODEL_CONFIG_REGISTRY, "llama", LlamaConfig)
dict.__setitem__(MODEL_CLASS_REGISTRY, "llama", LlamaForCausalLM)


@dataclass(frozen=True)
class ArchitectureProfile:
    """Architecture-specific runtime rules, looked up by Hugging Face ``model_type``."""

    accepts_loss_kwargs: bool = True
    disable_cache_for_eval: bool = False
    classification_scores_one_label_at_a_time: bool = False
    honours_label_microbatch: bool = False
    caps_generation_batch: bool = False
    has_train_and_inference_modes: bool = False
    exact_head_dims: bool = False
    pads_to_chunk_size: bool = False
    skips_completions_callback: bool = False
    needs_mamba2_grouped_norm_patch: bool = False


_DEFAULT_PROFILE = ArchitectureProfile()
_MAMBA_PROFILE = ArchitectureProfile(
    disable_cache_for_eval=True,
    classification_scores_one_label_at_a_time=True,
    honours_label_microbatch=True,
    caps_generation_batch=True,
    needs_mamba2_grouped_norm_patch=True,
)
_PROFILES: dict[str, ArchitectureProfile] = {
    # FLA models ignore num_items_in_batch despite accepting **kwargs.
    "gated_deltanet": ArchitectureProfile(accepts_loss_kwargs=False),
    "hybrid": ArchitectureProfile(accepts_loss_kwargs=False),
    "fla_mamba2": ArchitectureProfile(
        accepts_loss_kwargs=False,
        disable_cache_for_eval=True,
        classification_scores_one_label_at_a_time=True,
        honours_label_microbatch=True,
        caps_generation_batch=True,
    ),
    "mamba": _MAMBA_PROFILE,
    "mamba2": _MAMBA_PROFILE,
    "xlstm": ArchitectureProfile(
        has_train_and_inference_modes=True,
        exact_head_dims=True,
        pads_to_chunk_size=True,
        skips_completions_callback=True,
    ),
}


def architecture_profile(
    config: Any, *, architecture: str | None = None
) -> ArchitectureProfile:
    model_type = str(getattr(config, "model_type", "") or "")
    if model_type in _PROFILES:
        return _PROFILES[model_type]
    if architecture and architecture in _PROFILES:
        return _PROFILES[architecture]
    return _DEFAULT_PROFILE


def switch_to_inference_mode(config: Any) -> tuple[Any, Any, Any] | None:
    """Put a train-mode xLSTM config into inference mode; return saved state."""
    if not (
        architecture_profile(config).has_train_and_inference_modes
        and getattr(config, "mode", None) == "train"
    ):
        return None
    saved = (
        config.mode,
        getattr(config, "return_last_states", None),
        getattr(config, "inference_state_dtype", None),
    )
    config.mode = "inference"
    config.return_last_states = True
    config.inference_state_dtype = str(config.dtype).removeprefix("torch.")
    return saved


def restore_mode(config: Any, saved: tuple[Any, Any, Any] | None) -> None:
    if saved is not None:
        config.mode, config.return_last_states, config.inference_state_dtype = saved


def use_exact_xlstm_head_dims(config: Any) -> None:
    """Avoid rounding xLSTM qk_dim/v_dim when hidden_size * factor is non-power-of-2."""
    config_class = type(config)
    if (
        architecture_profile(config).exact_head_dims
        and hasattr(config_class, "qk_dim")
        and hasattr(config_class, "v_dim")
    ):
        config_class.qk_dim = property(  # type: ignore[attr-defined]
            lambda cfg: int(cfg.hidden_size * cfg.qk_dim_factor)
        )
        config_class.v_dim = property(  # type: ignore[attr-defined]
            lambda cfg: int(cfg.hidden_size * cfg.v_dim_factor)
        )


def tie_xlstm_embeddings(model: Any) -> None:
    """Tie xLSTM lm_head to backbone embeddings when tie_word_embeddings=True.

    Transformers 5.15's xLSTMForCausalLM leaves _tied_weights_keys as None,
    causing tie_word_embeddings=True to be silently ignored (~48M extra params).
    """
    config = getattr(model, "config", None)
    if not (
        architecture_profile(config).has_train_and_inference_modes
        and getattr(config, "tie_word_embeddings", False)
    ):
        return
    if getattr(model, "_tied_weights_keys", None) is None:
        model._tied_weights_keys = {"lm_head.weight": "backbone.embeddings.weight"}
    get_in = getattr(model, "get_input_embeddings", None)
    get_out = getattr(model, "get_output_embeddings", None)
    if callable(get_in) and callable(get_out):
        in_emb = get_in()
        out_emb = get_out()
        if (
            in_emb is not None
            and out_emb is not None
            and getattr(in_emb, "weight", None) is not None
            and getattr(out_emb, "weight", None) is not None
            and out_emb.weight is not in_emb.weight
        ):
            out_emb.weight = in_emb.weight


def validate_xlstm_pretrain_kernels(
    config: Any, *, allow_native: bool = False
) -> dict[str, Any]:
    """Log xLSTM kernel selection and reject pure-PyTorch native pretraining on GPU.

    Transformers' xLSTMConfig defaults to ``chunkwise--native_autograd`` when
    kernels are omitted, which silently runs the slow reference path (#143).
    """
    info = {
        "chunkwise_kernel": getattr(config, "chunkwise_kernel", None),
        "sequence_kernel": getattr(config, "sequence_kernel", None),
        "step_kernel": getattr(config, "step_kernel", None),
        "mode": getattr(config, "mode", None),
        "chunk_size": getattr(config, "chunk_size", None),
    }
    logger.info("Resolved xLSTM kernels: %s", info)
    chunkwise = str(info["chunkwise_kernel"] or "")
    sequence = str(info["sequence_kernel"] or "")
    step = str(info["step_kernel"] or "")
    is_native = (
        "native_autograd" in chunkwise
        or "native_custbw" in chunkwise
        or sequence == "native_sequence__native"
        or step == "native"
    )
    if not allow_native and is_native:
        raise ValueError(
            "xLSTM pretraining resolved to a pure-PyTorch native kernel "
            f"({info!r}). Configure Triton kernels explicitly "
            "(chunkwise_kernel='chunkwise--triton_xl_chunk', "
            "sequence_kernel='native_sequence__triton', step_kernel='triton') "
            "or pass allow_native=True for CPU/debug runs."
        )
    return info


def install_mamba2_grouped_rmsnorm_patch() -> None:
    """Fix transformers' MambaRMSNormGated torch fallback to normalize per group (#136).

    Transformers' MambaRMSNormGated.forward averages variance over all channels,
    diverging from FLA and mamba-ssm which normalize per ``n_groups`` group.
    """
    global _MAMBA2_NORM_PATCHED
    if _MAMBA2_NORM_PATCHED:
        return
    try:
        from transformers.models.mamba2.modeling_mamba2 import (
            Mamba2Mixer,
            MambaRMSNormGated,
        )
    except ImportError:
        return

    orig_mixer_init = Mamba2Mixer.__init__

    def _patched_mixer_init(self: Any, *args: Any, **kwargs: Any) -> None:
        orig_mixer_init(self, *args, **kwargs)
        if hasattr(self, "norm") and hasattr(self, "n_groups"):
            self.norm._sallm_n_groups = int(self.n_groups)

    def _grouped_rmsnorm_forward(
        self: Any, hidden_states: torch.Tensor, gate: torch.Tensor | None = None
    ) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        if gate is not None:
            hidden_states = hidden_states * F.silu(gate.to(torch.float32))

        n_groups = int(getattr(self, "_sallm_n_groups", 1))
        if n_groups > 1 and hidden_states.shape[-1] % n_groups == 0:
            orig_shape = hidden_states.shape
            group_size = orig_shape[-1] // n_groups
            grouped = hidden_states.reshape(*orig_shape[:-1], n_groups, group_size)
            variance = grouped.pow(2).mean(-1, keepdim=True)
            hidden_states = (
                grouped * torch.rsqrt(variance + self.variance_epsilon)
            ).reshape(orig_shape)
        else:
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(
                variance + self.variance_epsilon
            )

        return self.weight * hidden_states.to(input_dtype)

    cast(Any, Mamba2Mixer).__init__ = _patched_mixer_init
    cast(Any, MambaRMSNormGated).forward = _grouped_rmsnorm_forward
    _MAMBA2_NORM_PATCHED = True


def patch_mamba2_grouped_rmsnorm(model: Any) -> int:
    """Ensure all transformers Mamba2Mixer norms on ``model`` normalize per group."""
    install_mamba2_grouped_rmsnorm_patch()
    modules_fn = getattr(model, "modules", None)
    if not callable(modules_fn):
        return 0
    patched = 0
    for module in modules_fn():
        if (
            type(module).__name__ == "Mamba2Mixer"
            and hasattr(module, "norm")
            and hasattr(module, "n_groups")
        ):
            module.norm._sallm_n_groups = int(module.n_groups)
            patched += 1
    return patched
