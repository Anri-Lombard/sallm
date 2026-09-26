#!/usr/bin/env python3
"""Verify the optional pure-FLA GatedDeltaNet boundary without import side effects."""

from __future__ import annotations

import argparse
import json
import tempfile
from importlib import import_module
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from sallm.models.optional import register_fla_gated_deltanet
from sallm.models.registry import MODEL_CLASS_REGISTRY, MODEL_CONFIG_REGISTRY
from sallm.utils import count_trainable_parameters
from transformers import AutoConfig, AutoModelForCausalLM

EXPECTED_PURE_GDN_PARAMS = 127_425_448


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("count-only", "preflight", "post-checkpoint"),
        required=True,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--expected-params", type=int, default=EXPECTED_PURE_GDN_PARAMS)
    return parser.parse_args()


def _load_pure_gdn_spec(path: Path) -> tuple[dict[str, Any], int, int]:
    raw = OmegaConf.load(path)
    architecture = OmegaConf.select(raw, "model.architecture")
    if architecture != "gated_deltanet":
        raise ValueError(
            f"{path} must select canonical pure `gated_deltanet`, got {architecture!r}."
        )
    model_config = OmegaConf.to_container(
        OmegaConf.select(raw, "model.config"), resolve=True
    )
    if not isinstance(model_config, dict):
        raise ValueError(f"{path} must contain a mapping at model.config.")
    if model_config.get("attn") is not None:
        raise ValueError("Pure GatedDeltaNet requires model.config.attn: null.")
    if model_config.get("attn_mode") != "chunk":
        raise ValueError("Pure GatedDeltaNet requires model.config.attn_mode: chunk.")
    param_validation = OmegaConf.to_container(
        OmegaConf.select(raw, "model.param_validation"), resolve=True
    )
    if not isinstance(param_validation, dict):
        raise ValueError(f"{path} must define model.param_validation.")
    try:
        min_params_m = float(param_validation["min_params_m"])
        max_params_m = float(param_validation["max_params_m"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "model.param_validation must contain numeric min_params_m and "
            "max_params_m values."
        ) from exc
    return model_config, int(min_params_m * 1_000_000), int(max_params_m * 1_000_000)


def _fla_classes() -> tuple[type[Any], type[Any]]:
    """Import FLA only in modes that explicitly select the pure architecture."""
    module = import_module("fla.models.gated_deltanet")
    return module.GatedDeltaNetConfig, module.GatedDeltaNetForCausalLM


def _assert_pure_fla_model(model: Any) -> None:
    config_class, model_class = _fla_classes()
    if type(model) is not model_class:
        raise TypeError(
            "Expected exact FLA GatedDeltaNetForCausalLM, got "
            f"{type(model).__module__}.{type(model).__name__}."
        )
    if type(model.config) is not config_class:
        raise TypeError(
            "Expected exact FLA GatedDeltaNetConfig, got "
            f"{type(model.config).__module__}.{type(model.config).__name__}."
        )
    if getattr(model.config, "attn", object()) is not None:
        raise ValueError("Pure GatedDeltaNet checkpoint/config has non-null attn.")
    if getattr(model.config, "attn_mode", None) != "chunk":
        raise ValueError("Pure GatedDeltaNet checkpoint/config is not in chunk mode.")


def _assert_parameter_gate(
    model: Any,
    min_params: int,
    max_params: int,
    expected_params: int | None,
) -> int:
    params = count_trainable_parameters(model)
    if not min_params <= params <= max_params:
        raise ValueError(
            "Pure GatedDeltaNet parameter gate failed: expected "
            f"{min_params:,}–{max_params:,}, got {params:,}."
        )
    if expected_params is not None and params != expected_params:
        raise ValueError(
            "Pure GatedDeltaNet exact parameter count failed: expected "
            f"{expected_params:,}, got {params:,}."
        )
    return params


def _build_from_spec(model_config: dict[str, Any]) -> Any:
    config_class = MODEL_CONFIG_REGISTRY["gated_deltanet"]
    model_class = MODEL_CLASS_REGISTRY["gated_deltanet"]
    return model_class(config_class(**model_config))


def _exercise_model_backward(model: Any) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Pure GatedDeltaNet preflight requires CUDA.")
    device = torch.device("cuda")
    model.to(device=device, dtype=torch.bfloat16)
    model.train()
    vocab_size = int(model.config.vocab_size)
    input_ids = torch.randint(0, vocab_size, (1, 16), device=device)
    result = model(input_ids=input_ids, labels=input_ids, use_cache=False)
    loss = getattr(result, "loss", None)
    if loss is None:
        logits = getattr(result, "logits", result[0])
        loss = logits.float().mean()
    if not torch.isfinite(loss.detach()).all().item():
        raise RuntimeError("Pure GatedDeltaNet BF16 forward produced non-finite loss.")
    loss.backward()
    missing = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    if missing:
        raise RuntimeError(
            "Pure GatedDeltaNet BF16 backward produced missing gradients: "
            + ", ".join(missing[:10])
        )
    nonfinite = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and not torch.isfinite(parameter.grad).all().item()
    ]
    if nonfinite:
        raise RuntimeError(
            "Pure GatedDeltaNet BF16 backward produced non-finite gradients: "
            + ", ".join(nonfinite[:10])
        )


def _exercise_chunk_gated_delta_kernel() -> str:
    """Directly execute FLA's CUDA chunk kernel and backpropagate through it."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Pure GatedDeltaNet chunk-kernel verification requires CUDA."
        )
    kernel = import_module("fla.ops.gated_delta_rule").chunk_gated_delta_rule
    device = torch.device("cuda")
    batch, length, query_heads, value_heads, key_dim, value_dim = 1, 64, 2, 4, 32, 32
    q = torch.randn(
        (batch, length, query_heads, key_dim),
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn(
        (batch, length, value_heads, value_dim),
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    g = torch.randn(
        (batch, length, value_heads),
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    beta = torch.randn_like(g, requires_grad=True)
    a_log = torch.zeros(
        (value_heads,),
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    dt_bias = torch.zeros_like(a_log, requires_grad=True)
    output = kernel(
        q,
        k,
        v,
        g,
        beta,
        scale=key_dim**-0.5,
        initial_state=None,
        output_final_state=False,
        A_log=a_log,
        dt_bias=dt_bias,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        allow_neg_eigval=False,
        state_v_first=True,
    )
    primary = output[0] if isinstance(output, tuple) else output
    primary.float().mean().backward()
    if any(tensor.grad is None for tensor in (q, k, v, g, beta, a_log, dt_bias)):
        raise RuntimeError("FLA chunk kernel backward produced no input gradients.")
    return "qk[B,T,H,K];v[B,T,HV,V];g,beta[B,T,HV];A_log,dt_bias[HV]"


def _save_and_reload_integrity(model: Any) -> None:
    with tempfile.TemporaryDirectory(prefix="sallm-pure-gdn-roundtrip-") as raw_dir:
        output_dir = Path(raw_dir)
        try:
            model.save_pretrained(output_dir)
        except RuntimeError as exc:
            if "shared tensors" not in str(exc):
                raise
            model.save_pretrained(output_dir, safe_serialization=False)
        reloaded = AutoModelForCausalLM.from_pretrained(
            output_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        _assert_pure_fla_model(reloaded)
        original_state = model.state_dict()
        reloaded_state = reloaded.state_dict()
        if original_state.keys() != reloaded_state.keys():
            raise RuntimeError("Pure GatedDeltaNet save/load changed state-dict keys.")
        for name, tensor in original_state.items():
            reloaded_tensor = reloaded_state[name]
            if not torch.equal(tensor.detach().cpu(), reloaded_tensor.detach().cpu()):
                raise RuntimeError(
                    f"Pure GatedDeltaNet save/load changed parameter {name!r}."
                )


def _assert_deterministic_greedy_generation(model: Any) -> None:
    model.eval()
    device = next(model.parameters()).device
    prompt = torch.tensor([[1, 2, 3, 4]], device=device)
    with torch.inference_mode():
        first = model.generate(
            input_ids=prompt,
            do_sample=False,
            num_beams=1,
            max_new_tokens=4,
            use_cache=True,
        )
        second = model.generate(
            input_ids=prompt,
            do_sample=False,
            num_beams=1,
            max_new_tokens=4,
            use_cache=True,
        )
    if not torch.equal(first, second):
        raise RuntimeError("Pure GatedDeltaNet greedy generation is not deterministic.")


def _count_only(args: argparse.Namespace) -> None:
    model_config, min_params, max_params = _load_pure_gdn_spec(args.config)
    model = _build_from_spec(model_config)
    _assert_pure_fla_model(model)
    params = _assert_parameter_gate(model, min_params, max_params, args.expected_params)
    print(
        json.dumps(
            {
                "mode": "count-only",
                "params": params,
                "params_m": params / 1_000_000,
                "attn": model.config.attn,
                "attn_mode": model.config.attn_mode,
            },
            sort_keys=True,
        )
    )


def _preflight(args: argparse.Namespace) -> None:
    model_config, min_params, max_params = _load_pure_gdn_spec(args.config)
    model = _build_from_spec(model_config)
    _assert_pure_fla_model(model)
    params = _assert_parameter_gate(model, min_params, max_params, args.expected_params)
    _exercise_model_backward(model)
    layout = _exercise_chunk_gated_delta_kernel()
    print(
        json.dumps(
            {
                "mode": "preflight",
                "params": params,
                "params_m": params / 1_000_000,
                "chunk_kernel_layout": layout,
                "bf16_forward_backward": True,
            },
            sort_keys=True,
        )
    )


def _post_checkpoint(args: argparse.Namespace) -> None:
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required in post-checkpoint mode.")
    if not register_fla_gated_deltanet():
        raise ModuleNotFoundError(
            "Pure GatedDeltaNet checkpoint verification requires the `pure-gdn` "
            "optional dependency."
        )
    _, min_params, max_params = _load_pure_gdn_spec(args.config)
    checkpoint_config = AutoConfig.from_pretrained(
        args.checkpoint,
        trust_remote_code=True,
    )
    config_class, _ = _fla_classes()
    if type(checkpoint_config) is not config_class:
        raise TypeError(
            "Transformers AutoConfig did not resolve the exact FLA GatedDeltaNetConfig."
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    _assert_pure_fla_model(model)
    params = _assert_parameter_gate(model, min_params, max_params, args.expected_params)
    _save_and_reload_integrity(model)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Pure GatedDeltaNet post-checkpoint verification requires CUDA."
        )
    model.to(device=torch.device("cuda"), dtype=torch.bfloat16)
    _exercise_model_backward(model)
    _assert_deterministic_greedy_generation(model)
    print(
        json.dumps(
            {
                "mode": "post-checkpoint",
                "params": params,
                "params_m": params / 1_000_000,
                "save_load_integrity": True,
                "bf16_forward_backward": True,
                "deterministic_greedy_generation": True,
            },
            sort_keys=True,
        )
    )


def main() -> None:
    args = _parse_args()
    if args.mode == "count-only":
        _count_only(args)
    elif args.mode == "preflight":
        _preflight(args)
    else:
        _post_checkpoint(args)


if __name__ == "__main__":
    main()
