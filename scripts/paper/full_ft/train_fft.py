#!/usr/bin/env python3
"""Full fine-tuning entry point: the v9 T2X train/validation-only runner plus two changes.

1. The model loads in float32 even when training.bf16=true. sallm.models.factory otherwise
   loads the weights *in* bfloat16 when bf16=true, which is harmless for LoRA (the adapters
   are the only trainable tensors) but would make full fine-tuning pure-bf16 (bf16 master
   weights and AdamW state; lr 1e-5 updates underflow bf16 precision). With fp32 weights,
   bf16=true is autocast mixed precision with fp32 master weights and optimizer state.
2. Writes $FFT_RUN_INFO (JSON): parameter counts/dtypes after PEFT handling, non-padding token counts of the tokenized
   train/validation sets, GPU name and CUDA peak memory. FFT_COUNT_ONLY=1 stops after the token count (no training).
Everything else (loader, Hydra config, sallm.main) is the v9 runner unchanged.
"""

from __future__ import annotations

import importlib.util
import json
import os
import runpy
import time
from pathlib import Path

import torch

RUNNER = (
    "/scratch/lmbanr001/masters/sallm_snapshots/full-matrix-targeted-recovery-20260916-v9"
    "/.audit/run_train_validation_only_20260914.py"
)
info_path = Path(os.environ["FFT_RUN_INFO"])
info: dict = {"runner": RUNNER}

spec = importlib.util.spec_from_file_location("v9_train_validation_only_runner", RUNNER)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner._install_loader()

import sallm.fine_tune.run as fine_tune_run  # noqa: E402
import sallm.models.factory as model_factory  # noqa: E402

model_factory._get_torch_dtype = lambda config: torch.float32
_apply_peft = fine_tune_run._apply_peft_if_needed


def _apply_peft_and_record(**kwargs):
    model = _apply_peft(**kwargs)
    params = list(model.parameters())  # tied tensors are yielded once
    info["model_class"] = type(model).__name__
    info["n_params"] = sum(p.numel() for p in params)
    info["n_trainable_params"] = sum(p.numel() for p in params if p.requires_grad)
    info["param_dtypes"] = sorted({str(p.dtype) for p in params})
    emb, head = model.get_input_embeddings(), model.get_output_embeddings()
    info["embeddings_tied"] = bool(head is not None and head.weight is emb.weight)
    info["vocab_rows"] = int(emb.weight.shape[0])
    info["config_eos_pad"] = [model.config.eos_token_id, model.config.pad_token_id]
    info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    info_path.write_text(json.dumps(info, indent=1) + "\n")
    if info["n_trainable_params"] != info["n_params"] or info["param_dtypes"] != ["torch.float32"]:
        raise RuntimeError(f"not a full fp32-master fine-tune: {info}")
    return model


fine_tune_run._apply_peft_if_needed = _apply_peft_and_record
_build_trainer = fine_tune_run.build_trainer


def _build_trainer_and_record(*args, **kwargs):
    trainer = _build_trainer(*args, **kwargs)
    try:  # token accounting (non-padding tokens as stored after SFT tokenization; padding is added by the collator)
        for split, ds in (("train", trainer.train_dataset), ("val", trainer.eval_dataset)):
            lengths = [len(ids) for ids in ds["input_ids"]]
            info[f"{split}_rows"] = len(lengths)
            info[f"{split}_tokens_per_epoch"] = sum(lengths)
            info[f"{split}_max_tokens"] = max(lengths)
            if "assistant_masks" in ds.column_names:
                info[f"{split}_loss_tokens_per_epoch"] = sum(sum(m) for m in ds["assistant_masks"])
    except Exception as exc:  # never let bookkeeping break a training run
        info["token_accounting_error"] = repr(exc)
    info_path.write_text(json.dumps(info, indent=1) + "\n")
    if os.environ.get("FFT_COUNT_ONLY") == "1":
        raise SystemExit(0)
    return trainer


fine_tune_run.build_trainer = _build_trainer_and_record

start = time.time()
try:
    runpy.run_module("sallm.main", run_name="__main__")
finally:
    info["wall_seconds"] = round(time.time() - start, 1)
    if torch.cuda.is_available():
        info["peak_mem_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
        info["peak_mem_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 2**30, 3)
    info_path.write_text(json.dumps(info, indent=1) + "\n")
