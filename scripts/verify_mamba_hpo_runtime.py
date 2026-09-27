#!/usr/bin/env python3
"""Fail-closed, metric-free Mamba-2 HPO activation preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import torch
from datasets import Dataset
from omegaconf import OmegaConf, open_dict
from sallm.chat_template import install_canonical_chat_template
from sallm.config import ExperimentConfig
from sallm.data.factory import build_conversation_dataset
from sallm.data.t2x import _make_dataset_from_files
from sallm.fine_tune.run import _apply_peft_if_needed
from sallm.models.factory import build_model, build_tokenizer

EXPECTED_TARGETS = ("in_proj", "out_proj")
EXPECTED_TARGET_MODULES = 54
EXPECTED_TRAINABLE_PARAMS = 6_663_168
EXPECTED_TRAIN_ROWS = 3_859
MEMORY_LIMIT_BYTES = 11 * 1024**3
TRAIN_ROW_INDEX = 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--train-text", type=Path, required=True)
    return parser.parse_args()


def _validate_contract(
    target_modules: tuple[str, ...],
    *,
    module_count: int,
    trainable_params: int,
) -> None:
    if target_modules != EXPECTED_TARGETS:
        raise ValueError(
            f"Mamba target modules must be {EXPECTED_TARGETS}, got {target_modules}."
        )
    if module_count != EXPECTED_TARGET_MODULES:
        raise ValueError(
            f"Expected {EXPECTED_TARGET_MODULES} Mamba target modules, "
            f"got {module_count}."
        )
    if trainable_params != EXPECTED_TRAINABLE_PARAMS:
        raise ValueError(
            f"Expected {EXPECTED_TRAINABLE_PARAMS:,} trainable parameters, "
            f"got {trainable_params:,}."
        )


def _assert_peak_memory(peak_bytes: int) -> None:
    if peak_bytes >= MEMORY_LIMIT_BYTES:
        raise RuntimeError(
            f"Peak CUDA allocation must be below 11 GiB, got "
            f"{peak_bytes / 1024**3:.3f} GiB."
        )


def _load_config(args: argparse.Namespace) -> ExperimentConfig:
    raw = OmegaConf.load(args.config)
    config = OmegaConf.merge(OmegaConf.structured(ExperimentConfig), raw)
    with open_dict(config):
        config.model.init_checkpoint = str(args.checkpoint)
        config.tokenizer.path = str(args.tokenizer)
        config.peft.kwargs.r = 32
        config.peft.kwargs.lora_alpha = 64
        config.peft.kwargs.lora_dropout = 0.0
        config.peft.kwargs.target_modules = list(EXPECTED_TARGETS)
        config.training.bf16 = True
        config.training.gradient_checkpointing = False
    return cast(ExperimentConfig, config)


def _import_native_extensions() -> dict[str, str]:
    loaded: dict[str, str] = {}
    for name in ("selective_scan_cuda", "causal_conv1d_cuda"):
        module = import_module(name)
        origin = Path(str(getattr(module, "__file__", "")))
        if origin.suffix != ".so":
            raise RuntimeError(f"{name} did not resolve to a native CUDA extension.")
        loaded[name] = str(origin)
    return loaded


def _encode_messages(tokenizer: Any, messages: list[dict[str, str]], *, prompt: bool):
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )
    return encoded["input_ids"] if isinstance(encoded, dict) else encoded


def _assert_finite_gradients(model: Any) -> int:
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    missing = [parameter for parameter in trainable if parameter.grad is None]
    if missing:
        raise RuntimeError(
            f"BF16 backward left {len(missing)} trainable parameter tensors "
            "without gradients."
        )
    if not trainable or any(
        not bool(torch.isfinite(parameter.grad).all()) for parameter in trainable
    ):
        raise RuntimeError("BF16 backward produced missing or non-finite gradients.")
    return len(trainable)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Mamba HPO preflight requires CUDA.")
    extensions = _import_native_extensions()
    config = _load_config(args)
    tokenizer = build_tokenizer(config)
    if tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|system|>", "<|user|>", "<|assistant|>"]}
    ):
        raise RuntimeError("Frozen tokenizer is missing canonical chat special tokens.")
    install_canonical_chat_template(tokenizer)

    model = build_model(config, tokenizer)
    target_count = sum(
        name.rsplit(".", 1)[-1] in EXPECTED_TARGETS
        for name, _module in model.named_modules()
    )
    model = _apply_peft_if_needed(
        model=model,
        peft_cfg=config.peft,
        trainable_token_indices=None,
    )
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    _validate_contract(
        tuple(config.peft.kwargs.target_modules),
        module_count=target_count,
        trainable_params=trainable_params,
    )

    train_dataset = build_conversation_dataset(
        _make_dataset_from_files(str(args.train_data), str(args.train_text)), config
    )
    if (
        not isinstance(train_dataset, Dataset)
        or len(train_dataset) != EXPECTED_TRAIN_ROWS
    ):
        actual_rows = (
            len(train_dataset) if hasattr(train_dataset, "__len__") else "unknown"
        )
        raise RuntimeError(
            f"Expected {EXPECTED_TRAIN_ROWS:,} map-style training rows, "
            f"got {type(train_dataset).__name__} with "
            f"{actual_rows} rows."
        )
    messages = cast(list[dict[str, str]], train_dataset[TRAIN_ROW_INDEX]["messages"])
    row_sha256 = hashlib.sha256(
        json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()

    device = torch.device("cuda")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.cuda.reset_peak_memory_stats(device)
    model.to(device=device, dtype=torch.bfloat16)
    model.train()
    input_ids = _encode_messages(tokenizer, messages, prompt=False).to(device)
    result = model(input_ids=input_ids, labels=input_ids, use_cache=False)
    loss = result.loss
    if loss is None or not bool(torch.isfinite(loss)):
        raise RuntimeError("Mamba BF16 forward produced a missing or non-finite loss.")
    loss.backward()
    gradient_tensors = _assert_finite_gradients(model)
    model.zero_grad(set_to_none=True)

    model.eval()
    prompt_ids = _encode_messages(tokenizer, messages[:-1], prompt=True).to(device)
    with torch.inference_mode():
        generated = model.generate(
            input_ids=prompt_ids,
            do_sample=False,
            num_beams=5,
            max_new_tokens=64,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    del generated
    torch.cuda.synchronize(device)
    peak_bytes = int(torch.cuda.max_memory_allocated(device))
    _assert_peak_memory(peak_bytes)
    return {
        "mode": "metric-free-mamba-hpo-preflight",
        "target_modules": list(EXPECTED_TARGETS),
        "target_module_count": target_count,
        "trainable_params": trainable_params,
        "training_rows": len(train_dataset),
        "training_row_index": TRAIN_ROW_INDEX,
        "training_row_sha256": row_sha256,
        "bf16_forward_backward": True,
        "finite_gradient_tensors": gradient_tensors,
        "beam_generation": {"num_beams": 5, "max_new_tokens": 64},
        "peak_allocated_bytes": peak_bytes,
        "peak_allocated_gib": peak_bytes / 1024**3,
        "native_extensions": extensions,
        "task_metric": None,
    }


def main() -> None:
    print(json.dumps(run(_parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
