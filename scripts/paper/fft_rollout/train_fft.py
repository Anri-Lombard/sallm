#!/usr/bin/env python3
"""Full fine-tuning entry point for the rollout (copy of scripts/paper/full_ft/train_fft.py, pilot entry point).

Changes vs the pilot copy: the v9 train/validation-only T2X loader is installed only when FFT_T2X_LOADER=1;
SALLM_FLA_MAMBA2=1 builds Mamba-2 from FLA's class; FFT_EPOCHS=auto sets the epoch count from the tokenized
train rows (10 if fewer than 5000, else 4) before the scheduler is built. Pilot docstring follows.

Full fine-tuning entry point: the v9 T2X train/validation-only runner plus two changes.

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
import sys
import time
from pathlib import Path

import torch

RUNNER = (
    "/scratch/lmbanr001/masters/sallm_snapshots/full-matrix-targeted-recovery-20260916-v9"
    "/.audit/run_train_validation_only_20260914.py"
)
info_path = Path(os.environ["FFT_RUN_INFO"])
info: dict = {"runner": RUNNER}

# Speed settings (27 Sep 2026): TF32 matmuls and fused AdamW (same betas/eps/wd/schedule); for xLSTM also the padded
# TFLA training kernel (tfla.py; the unpadded kernel is unreliable at head dims 92/184, see the RUNBOOK). Runs started
# before this keep the old settings, also when resumed after a lane death: the choice is stored in
# RUN/train_settings.json at first start, and a resumed run without that file is an old-settings run.
# FFT_SPEED=legacy|new forces one.
ARCH = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("finetune.model.architecture=")), None)
KERNEL = "chunkwise--native_autograd (transformers 4.57.3)" if ARCH == "xlstm" else "model default"
TFLA = "tfla_padded128"  # tfla.NAME
RESUMING = any("resume_from_checkpoint=" in a and not a.endswith("=null") for a in sys.argv)
SETTINGS = {
    "legacy": {"settings": "legacy", "optimizer_impl": "adamw_torch", "tf32": False, "train_kernel": KERNEL},
    "new": {"settings": "new-20260927", "optimizer_impl": "adamw_torch_fused", "tf32": True,
            "train_kernel": TFLA if ARCH == "xlstm" else KERNEL},
}


def pick_settings(forced, stored, resuming):
    if forced in SETTINGS:
        return SETTINGS[forced]
    if stored is not None:  # this run's own choice at its first start
        return stored
    return SETTINGS["legacy" if resuming else "new"]  # resumed without a record = started before 27 Sep


_ctrl_run = Path(json.loads(os.environ["FFT_CTRL"])["run"]) if os.environ.get("FFT_CTRL") else None
_marker = _ctrl_run / "train_settings.json" if _ctrl_run else None
settings = pick_settings(os.environ.get("FFT_SPEED"), json.loads(_marker.read_text()) if _marker and _marker.exists() else None, RESUMING)
if _marker and not _marker.exists() and not os.environ.get("FFT_SPEED") and str(info_path) != "/dev/null":  # not the --cfg dry run
    _marker.write_text(json.dumps(settings, indent=1) + "\n")
info["train_settings"] = settings
if settings["tf32"]:  # legacy leaves the torch defaults (matmul off, cuDNN on)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
print(f"FFT_TRAIN_SETTINGS {json.dumps(settings)}", flush=True)

if os.environ.get("FFT_T2X_LOADER") == "1":
    spec = importlib.util.spec_from_file_location("v9_train_validation_only_runner", RUNNER)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner._install_loader()
else:
    info["runner"] = None

import sallm.fine_tune.run as fine_tune_run  # noqa: E402
import sallm.models.factory as model_factory  # noqa: E402
import sallm.models.registry as model_registry  # noqa: E402

model_factory._get_torch_dtype = lambda config: torch.float32
if os.environ.get("SALLM_FLA_MAMBA2") == "1":
    # Same weights layout as transformers' Mamba2 (FLA ports it), but per-group gated RMSNorm on every path.
    for reg, cls in ((model_registry.MODEL_CLASS_REGISTRY, "Mamba2ForCausalLM"), (model_registry.MODEL_CONFIG_REGISTRY, "Mamba2Config")):
        reg._mappings["mamba2"] = ("fla.models", cls)
        dict.pop(reg, "mamba2", None)
    _build_model = model_factory.build_model

    def _build_fla_mamba2(*args, **kwargs):
        model = _build_model(*args, **kwargs)
        assert type(model).__module__.startswith("fla."), type(model)
        model.accepts_loss_kwargs = False  # FLA models ignore num_items_in_batch (as the GDN path)
        return model

    model_factory.build_model = _build_fla_mamba2
    fine_tune_run.build_model = _build_fla_mamba2  # run.py imported it by name
_apply_peft = fine_tune_run._apply_peft_if_needed


def _use_tfla(model) -> None:
    """Padded TFLA for xLSTM training; on any failure (import, self-check, swap) the run trains with the native kernel,
    and its train_settings.json says so (a resume then stays native)."""
    global settings
    try:
        import tfla  # this script's directory is sys.path[0]

        info["tfla_self_check"] = tfla.self_check()
        info["tfla_backends_swapped"] = tfla.use_tfla(model)
    except Exception as exc:  # noqa: BLE001 - fall back, never fail the unit
        settings = {**settings, "train_kernel": KERNEL, "tfla_fallback": f"{type(exc).__name__}: {exc}"[:500]}
        info["train_settings"] = settings
        if _marker and str(info_path) != "/dev/null":
            _marker.write_text(json.dumps(settings, indent=1) + "\n")
    print(f"FFT_XLSTM_TRAIN_KERNEL {settings['train_kernel']} {json.dumps(info.get('tfla_self_check') or settings.get('tfla_fallback'))}",
          flush=True)


def _apply_peft_and_record(**kwargs):
    model = _apply_peft(**kwargs)
    if ARCH == "xlstm" and settings["train_kernel"] == TFLA:
        _use_tfla(model)
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
    from transformers import TrainerCallback
    from transformers.training_args import OptimizerNames

    trainer.args.optim = OptimizerNames(settings["optimizer_impl"])  # the optimizer is created later, in train()

    class _RecordOptimizer(TrainerCallback):
        def on_train_begin(self, args, state, control, optimizer=None, **kw):
            opt = getattr(optimizer, "optimizer", optimizer)  # accelerate wraps it
            d = opt.defaults
            info["optimizer"] = {"class": type(opt).__name__, "fused": d.get("fused"), "foreach": d.get("foreach"),
                                 "betas": list(d["betas"]), "eps": d["eps"], "weight_decay": [g["weight_decay"] for g in opt.param_groups],
                                 "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32}
            print(f"FFT_OPTIMIZER {json.dumps(info['optimizer'])}", flush=True)
            info_path.write_text(json.dumps(info, indent=1) + "\n")
            self.t = []

        def on_step_end(self, args, state, control, **kw):
            self.t.append(time.time())

        def on_train_end(self, args, state, control, **kw):  # optimizer-step time, first 5 steps (compilation) dropped
            d = sorted(b - a for a, b in zip(self.t[5:], self.t[6:]) if b - a < 60)  # skips epoch-end scoring pauses
            if d:
                info["step_s_median"] = round(d[len(d) // 2], 4)
                info_path.write_text(json.dumps(info, indent=1) + "\n")

    trainer.add_callback(_RecordOptimizer())
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
    if os.environ.get("FFT_EPOCHS") == "auto":
        # Protocol: 10 epochs if the training set has fewer than 5000 examples, else 4.
        trainer.args.num_train_epochs = 10 if len(trainer.train_dataset) < 5000 else 4
        info["num_train_epochs"] = trainer.args.num_train_epochs
    info_path.write_text(json.dumps(info, indent=1) + "\n")
    if os.environ.get("FFT_COUNT_ONLY") == "1":
        raise SystemExit(0)
    return trainer


# ---------------------------------------------------------------------------------------------------------------
# Rollout control (env FFT_CTRL = JSON {"out", "run", "unit", "patience"}; absent -> plain training as before).
# Per epoch (on_save, after the epoch checkpoint is written): score validation in this process with the rollout's
# protocol scorer (rollout.val_score, fixed subsample), keep the best epoch's weights in RUN/best (strictly greater
# only; ties keep the earlier epoch), keep the latest full checkpoint in RUN/resume (resume after a lane dies), and stop
# once `patience` epochs pass without a strictly better score. Divergence: a non-finite logged loss, or a loss above
# 3x the first-epoch mean for 200 consecutive steps, writes RUN/DIVERGED.json and aborts the unit (never retried).
CTRL = json.loads(os.environ["FFT_CTRL"]) if os.environ.get("FFT_CTRL") else None
WEIGHT_FILES = ("config.json", "generation_config.json", "model.safetensors", "pytorch_model.bin",
                "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja")


def early_stop_decision(vals: list[float], patience: int) -> tuple[int, bool]:
    """(best epoch, stop?) for per-epoch validation scores vals[0] = epoch 1. Improvement = strictly greater."""
    best = max(range(len(vals)), key=lambda i: (vals[i], -i)) + 1
    return best, patience > 0 and len(vals) - best >= patience


if CTRL:
    import math
    import shutil
    import sys

    from transformers import TrainerCallback

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    RUN = Path(CTRL["run"])

    def _write(path: Path, value) -> None:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(value, indent=1) + "\n")
        os.replace(tmp, path)

    def _replace_dir(src_files: list[Path], dst: Path) -> None:
        tmp = dst.with_name(dst.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        for f in src_files:
            (shutil.copytree if f.is_dir() else shutil.copy2)(f, tmp / f.name)
        old = dst.with_name(dst.name + ".old")
        shutil.rmtree(old, ignore_errors=True)
        if dst.exists():
            os.replace(dst, old)
        os.replace(tmp, dst)
        shutil.rmtree(old, ignore_errors=True)

    class RolloutControl(TrainerCallback):
        def __init__(self):
            ref = RUN / "loss_ref.json"
            self.ref = json.loads(ref.read_text()) if ref.exists() else {"first_epoch": [], "first_mean": None}
            self.high = 0

        def _diverged(self, why: str, step: int):
            _write(RUN / "DIVERGED.json", {"reason": why, "step": step, "time": time.time()})
            raise RuntimeError(f"FAILED_DIVERGED: {why} at step {step}")

        def on_log(self, args, state, control, logs=None, **kw):
            loss = (logs or {}).get("loss")
            if loss is None:
                return
            if not math.isfinite(loss):
                self._diverged(f"non-finite loss {loss}", state.global_step)
            if state.epoch is not None and state.epoch <= 1.0 and self.ref["first_mean"] is None:
                self.ref["first_epoch"].append(loss)
                return
            if self.ref["first_mean"] is None and self.ref["first_epoch"]:
                self.ref["first_mean"] = sum(self.ref["first_epoch"]) / len(self.ref["first_epoch"])
                _write(RUN / "loss_ref.json", self.ref)
            m = self.ref["first_mean"]
            self.high = self.high + args.logging_steps if m is not None and loss > 3 * m else 0
            if self.high >= 200:
                self._diverged(f"loss above 3x the first-epoch mean ({m:.4f}) for {self.high} steps", state.global_step)

        def on_save(self, args, state, control, **kw):
            import rollout

            ckpt = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            epoch = int(round(state.epoch or 0)) or 1
            torch.cuda.empty_cache()  # the scorer runs in a subprocess on this GPU while training waits
            r = rollout.Run(Path(CTRL["out"]))
            t0 = time.time()
            v = rollout.val_score(r, CTRL["unit"], ckpt, RUN / "val" / f"e{epoch}")
            ep_path = RUN / "val_epochs.json"
            epochs = [e for e in (json.loads(ep_path.read_text()) if ep_path.exists() else []) if e["epoch"] < epoch]
            epochs.append({"epoch": epoch, "checkpoint": ckpt.name, "global_step": state.global_step, "val": v["score"], "detail": v,
                           "val_secs": round(time.time() - t0, 1)})
            _write(ep_path, epochs)
            if self.ref["first_mean"] is None and self.ref["first_epoch"]:
                self.ref["first_mean"] = sum(self.ref["first_epoch"]) / len(self.ref["first_epoch"])
                _write(RUN / "loss_ref.json", self.ref)
            best, stop = early_stop_decision([e["val"] for e in epochs], int(CTRL["patience"]))
            if best == epoch:  # weights only
                _replace_dir([ckpt / n for n in WEIGHT_FILES if (ckpt / n).exists()], RUN / "best")
                _write(RUN / "best" / "BEST.json", {"epoch": epoch, "val": v["score"], "checkpoint": ckpt.name})
            planned = int(args.num_train_epochs)
            stopped = stop and epoch < planned
            _write(RUN / "early_stop.json", {"patience": int(CTRL["patience"]), "planned_epochs": planned, "epochs_run": epoch,
                                              "stopped_early": stopped, "stop_epoch": epoch if stopped else None,
                                              "best_epoch": best, "best_val": epochs[best - 1]["val"],
                                              "rule": "stop after `patience` epochs without a strictly greater validation score"})
            if epoch < planned and not stopped:  # full checkpoint to resume from if the lane dies
                _replace_dir(sorted(ckpt.iterdir()), RUN / "resume" / ckpt.name)
                for other in (RUN / "resume").iterdir():
                    if other.name != ckpt.name:
                        shutil.rmtree(other, ignore_errors=True)
            if os.environ.get("FFT_TEST_DIE_AFTER_EPOCH") == str(epoch):  # smoke test of the resume path only
                os._exit(75)
            if stopped:
                control.should_training_stop = True
            shutil.rmtree(ckpt, ignore_errors=True)  # scored and persisted; /dev/shm holds one epoch at a time
            return control


def _build_trainer_and_control(*args, **kwargs):
    trainer = _build_trainer_and_record(*args, **kwargs)
    if CTRL:
        trainer.add_callback(RolloutControl())
    return trainer


fine_tune_run.build_trainer = _build_trainer_and_control

start = time.time()
try:
    runpy.run_module("sallm.main", run_name="__main__")
finally:
    info["wall_seconds"] = round(time.time() - start, 1)
    if torch.cuda.is_available():
        info["peak_mem_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
        info["peak_mem_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 2**30, 3)
    info_path.write_text(json.dumps(info, indent=1) + "\n")
