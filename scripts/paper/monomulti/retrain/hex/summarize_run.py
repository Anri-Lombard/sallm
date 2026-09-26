#!/usr/bin/env python3
"""Summarize one retrain output directory as JSON (stdlib only)."""
import glob
import json
import os
import re
import sys
from pathlib import Path

out = Path(sys.argv[1])
info = {}
for line in (out / "run_info.txt").read_text().splitlines():
    if "=" in line:
        k, v = line.split("=", 1)
        info[k] = v
    elif line.strip():
        info.setdefault("gpu", line.strip())

ckpts = sorted(out.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
state = json.loads((ckpts[-1] / "trainer_state.json").read_text()) if ckpts else {}
hist = state.get("log_history", [])
evals = {int(round(h["epoch"])): {"step": h["step"], "eval_loss": h["eval_loss"]} for h in hist if "eval_loss" in h}
train = next((h for h in hist if "train_runtime" in h), None)

for f in glob.glob(str(out / "debug_generation_examples/*/step-*.jsonl")):
    per = {}
    for line in open(f):
        d = json.loads(line)
        per[d["language"]] = d["metrics"]
        epoch = int(round(d["epoch"]))
    f1 = {k.split("/")[1].rsplit("_f1", 1)[0]: v for m in per.values() for k, v in m.items() if re.fullmatch(r"eval/[a-z]+_f1", k)}
    evals.setdefault(epoch, {})["gen_f1"] = f1
    evals[epoch]["gen_f1_mean"] = sum(f1.values()) / len(f1) if f1 else None
for f in glob.glob(str(out / "validation_artifacts/pos/step-*.json")):
    d = json.loads(open(f).read())
    evals.setdefault(int(round(d["epoch"])), {})["pos_token_accuracy"] = d["all_token_accuracy"]

resolved = (out / "resolved_config.yaml").read_text()
def cfg(key):
    m = re.search(rf"^\s+{key}: (.*)$", resolved, re.M)
    return m.group(1).strip() if m else None
bs, ga = int(cfg("per_device_train_batch_size")), int(cfg("gradient_accumulation_steps"))
summary = {
    "label": out.name,
    "run_info": info,
    "config": {k: cfg(k) for k in ("learning_rate", "num_train_epochs", "per_device_train_batch_size", "gradient_accumulation_steps",
                                   "lr_scheduler_type", "warmup_ratio", "weight_decay", "r", "lora_alpha", "lora_dropout",
                                   "early_stopping_patience", "metric_for_best_model", "load_best_model_at_end", "pad_to_multiple_of", "bf16")},
    "effective_batch": bs * ga,
    "global_step": state.get("global_step"),
    "epochs_run": state.get("epoch"),
    "max_steps": state.get("max_steps"),
    "examples_seen": (state.get("global_step") or 0) * bs * ga,
    "train_runtime_s": train and train.get("train_runtime"),
    "best_model_checkpoint": state.get("best_model_checkpoint"),
    "best_metric": state.get("best_metric"),
    "checkpoints": [p.name for p in ckpts],
    "per_epoch": {str(k): evals[k] for k in sorted(evals)},
}
print(json.dumps(summary, indent=1))
