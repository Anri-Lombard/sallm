#!/usr/bin/env python3
"""Run one frozen T2X/AfriHG unit with the system prompt kept or dropped, on validation or test."""

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
sys.path.insert(0, str(REPAIR))
from run_generation_unit import ref_id, sha256, verify_output  # noqa: E402

INTERFACE = {
    "mzansilm": {"dtype": "bfloat16", "merge_lora": False, "tie_word_embeddings": None},
    "mamba2": {"dtype": "float32", "merge_lora": False, "tie_word_embeddings": False},
    "xlstm": {"dtype": "float32", "merge_lora": True, "tie_word_embeddings": False},
    "gdn": {"dtype": "bfloat16", "merge_lora": False, "tie_word_embeddings": None},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--system-prompt", choices=("keep", "drop"), required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--decoding", choices=("config", "greedy"), default="config")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    release = json.loads(args.release.read_text())
    unit = json.loads((args.inventory / "official_units.json").read_text())[args.index]
    template = yaml.safe_load((args.source / "src/conf/eval/run_llama_general_full_matrix_r1.yaml").read_text())
    wanted = set()
    if "T2X" in unit["tasks"]:
        wanted.add("t2x_xho")
    if "AfriHG" in unit["tasks"]:
        wanted.update(f"afrihg_{language}" for language in unit["languages"])
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
    entries = json.loads(args.bindings.read_text())["entries"]
    base = entries[ref_id(unit["binding"]["base"])]
    candidates = unit["binding"]["adapter_candidates"]
    selected = release["selected_bindings"].get(unit["unit_id"])
    adapter = entries[ref_id(candidates[int(selected)] if selected is not None else candidates[0])]
    config_dict = {
        "mode": "EVALUATE",
        "eval_model": {
            "checkpoint": base["path"],
            "peft_adapter": adapter["path"],
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
        "base": base,
        "adapter": adapter,
        "split": args.split,
        "system_prompt": args.system_prompt,
        "max_samples_per_lang": args.max_samples,
        "decoding": [task["decoding"] for task in tasks],
        "evaluation_summary_sha256": sha256(summary),
        "tasks": sorted(wanted),
    }
    (args.output / "UNIT_DONE.json").write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")
    print(f"GENERATION_PROTOCOL_OK index={args.index} split={args.split} system_prompt={args.system_prompt}")


if __name__ == "__main__":
    main()
