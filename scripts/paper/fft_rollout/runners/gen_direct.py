#!/usr/bin/env python3
"""Run explicitly bound T2X/AfriHG evaluations with the system prompt kept or dropped."""

from __future__ import annotations

import argparse
import json
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
    args = parser.parse_args()
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
        # Greedy is the only decoding all four architectures support (Mamba-2 cannot reorder beams).
        if args.decoding == "greedy" or unit["architecture"] == "mamba2":
            task["decoding"] = {"strategy": "greedy"}
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
    run(config)
    summary = verify_output(args.output, wanted)
    marker = {
        "schema": "sallm.generation_system_prompt_protocol/v1",
        "unit": unit,
        "split": args.split,
        "system_prompt": args.system_prompt,
        "max_samples_per_lang": args.max_samples,
        "decoding": [task["decoding"] for task in tasks],
        "evaluation_summary_sha256": sha256(summary),
        "tasks": sorted(wanted),
    }
    (args.output / "UNIT_DONE.json").write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")
    print(f"GENERATION_PROTOCOL_OK unit={unit['unit_id']} split={args.split} system_prompt={args.system_prompt}")


if __name__ == "__main__":
    main()
