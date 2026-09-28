#!/usr/bin/env python3
"""Run explicitly bound T2X/AfriHG evaluations with the system prompt kept or dropped."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import cast

import yaml
from omegaconf import OmegaConf
from sallm.config import ExperimentConfig
from sallm.evaluation.run import run

REPAIR = Path("/scratch/lmbanr001/masters/sallm_snapshots/full-matrix-generation-repair-20260919-v1")
BUNDLE = Path("/scratch/lmbanr001/masters/sallm_snapshots/full-matrix-execution-20260916-v1")
sys.path.insert(0, str(BUNDLE))
sys.path.insert(0, str(REPAIR))
from prepare_bindings import tree_sha256  # noqa: E402
from run_generation_unit import sha256, verify_output  # noqa: E402

INTERFACE = {
    "mzansilm": {"dtype": "bfloat16", "merge_lora": False, "tie_word_embeddings": None},
    "mamba2": {"dtype": "float32", "merge_lora": False, "tie_word_embeddings": False},
    "xlstm": {"dtype": "float32", "merge_lora": True, "tie_word_embeddings": False},
    "gdn": {"dtype": "bfloat16", "merge_lora": False, "tie_word_embeddings": None},
}


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from val_subsample import indices as val_subsample_indices  # noqa: E402
from sallm.evaluation.generation_metrics import GenerationEvaluator  # noqa: E402

_evaluate, _cap = GenerationEvaluator.evaluate, GenerationEvaluator._cap_dataset


def _evaluate_with_task(self, model, dataset, *args, **kwargs):
    self._fft_task = str(kwargs.get("metric_prefix", "")).split("/")[-1]  # harness passes eval/<task id>
    return _evaluate(self, model, dataset, *args, **kwargs)


def _cap_with_subsample(self, dataset, world_size, lang_key):
    """Validation only (FFT_VAL_SUBSAMPLE=1): the fixed selection subsample, then the smoke cap if any."""
    task = getattr(self, "_fft_task", "")
    if os.environ.get("FFT_VAL_SPLIT") == "val" and "_" in task:
        family, lang = task.rsplit("_", 1)
        keep = val_subsample_indices(family, lang, len(dataset))
        if keep is not None:
            print(f"VAL_SUBSAMPLE {family}/{lang} {len(dataset)} -> {len(keep)}", flush=True)
            dataset = dataset.select(keep)
    if self.max_samples_per_lang is not None and len(dataset) > self.max_samples_per_lang:
        return dataset.select(range(self.max_samples_per_lang)) if os.environ.get("FFT_VAL_SPLIT") == "val" else _cap(self, dataset, world_size, lang_key)
    return dataset


GenerationEvaluator.evaluate = _evaluate_with_task


import sallm.evaluation.run as eval_run  # noqa: E402

_load_model_and_tokenizer = eval_run.load_model_and_tokenizer


def _wait_until_alone() -> object:
    """Generation sizes its batches from free GPU memory, so it must not share the lane's GPU with another scorer
    (29 Sep 2026: a concurrent AfriHG scorer took 38.7 GB and T2X ran out of memory). Take a per-lane lock, then wait
    until the only other process on this GPU is the paused training process."""
    import fcntl
    import subprocess
    import time

    lock = open(f"/dev/shm/fft_gen_lock_{os.environ.get('SLURM_JOB_ID', 'local')}", "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    for _ in range(720):  # at most 3 h
        pids = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.split()
        others = []
        for pid in pids:
            try:
                cmd = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace")
            except OSError:
                continue
            if int(pid) != os.getpid() and "train_fft" not in cmd:
                others.append(pid)
        if not others:
            break
        time.sleep(15)
    return lock


def _load_maybe_without_cache(model_cfg):
    model, tokenizer = _load_model_and_tokenizer(model_cfg)
    if os.environ.get("FFT_GEN_NO_CACHE") == "1":
        model.config.use_cache = False
    return model, tokenizer


eval_run.load_model_and_tokenizer = _load_maybe_without_cache
GenerationEvaluator._cap_dataset = _cap_with_subsample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--system-prompt", choices=("keep", "drop"), required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--decoding", choices=("config", "greedy"), default="config")
    parser.add_argument("--no-cache", action="store_true", help="decode without the generation cache (beam search on models whose cache cannot be reordered)")
    args = parser.parse_args()
    _gen_lock = _wait_until_alone()  # noqa: F841 - held until the process exits
    if args.output.exists():
        raise FileExistsError(args.output)
    unit = json.loads(args.spec.read_text())[args.index]
    for key in ("base", "adapter"):
        if unit[key] is None:  # full fine-tune: the fine-tuned model is the base, no adapter
            continue
        observed = tree_sha256(Path(unit[key]))
        if observed != unit[f"{key}_tree_sha256"]:
            raise ValueError(f"{unit['unit_id']} {key} tree {observed} != {unit[f'{key}_tree_sha256']}")
    wanted = set(unit["tasks"])
    template = yaml.safe_load((args.source / "src/conf/eval/run_llama_general_full_matrix_r1.yaml").read_text())
    tasks = [task for task in template["evaluation"]["generation_tasks"] if task["id"] in wanted]
    if {task["id"] for task in tasks} != wanted:
        raise ValueError(f"Missing generation tasks: {wanted}")
    for task in tasks:
        task["split"] = args.split
        task["max_samples_per_lang"] = args.max_samples
        if args.system_prompt == "drop":
            task.pop("system_prompt", None)
        if args.decoding == "greedy":
            task["decoding"] = {"strategy": "greedy"}
        elif unit["architecture"] == "mamba2" and not args.no_cache:
            # FLA 0.5.1's cache cannot be reordered across beams; never fall back to greedy under a beam label.
            raise SystemExit("beam search on mamba2 needs --no-cache")
    config_dict = {
        "mode": "EVALUATE",
        "eval_model": {
            "checkpoint": unit["base"],
            "peft_adapter": unit["adapter"],
            "adapter": "hf",
            "device": "cuda:0",
            **INTERFACE[unit["architecture"]],
        },
        "evaluation": {
            "task_packs": [],
            "generation_tasks": tasks,
            "overrides": {},
            "output_dir": str(args.output),
        },
    }
    config = cast(ExperimentConfig, OmegaConf.merge(OmegaConf.structured(ExperimentConfig), OmegaConf.create(config_dict)))
    if args.no_cache:
        os.environ["FFT_GEN_NO_CACHE"] = "1"
    run(config)
    summary = verify_output(args.output, wanted)
    marker = {
        "schema": "sallm.generation_system_prompt_protocol/v1",
        "unit": unit,
        "split": args.split,
        "system_prompt": args.system_prompt,
        "max_samples_per_lang": args.max_samples,
        "decoding": [task["decoding"] for task in tasks],
        "generation_cache": not args.no_cache,
        "evaluation_summary_sha256": sha256(summary),
        "tasks": sorted(wanted),
    }
    (args.output / "UNIT_DONE.json").write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")
    print(f"GENERATION_PROTOCOL_OK unit={unit['unit_id']} split={args.split} system_prompt={args.system_prompt}")


if __name__ == "__main__":
    main()
