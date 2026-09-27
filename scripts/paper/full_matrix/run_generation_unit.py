#!/usr/bin/env python3
"""Run one frozen T2X/AfriHG official unit through the retained harness."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import yaml
from omegaconf import OmegaConf
from sallm.config import ExperimentConfig
from sallm.evaluation.run import run


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def ref_id(reference: dict[str, Any]) -> str:
    return hashlib.sha256(canonical(reference).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    release = json.loads(args.release.read_text())
    if not release.get("authorized") or not release.get("no_score_based_retry"):
        raise ValueError("Official release is not valid")
    unit = json.loads((args.inventory / "official_units.json").read_text())[args.index]
    if unit["task_group"] != "generation":
        raise ValueError("Not a generation unit")
    entries = json.loads(args.bindings.read_text())["entries"]
    base = entries[ref_id(unit["binding"]["base"])]
    candidates = unit["binding"]["adapter_candidates"]
    selected = release["selected_bindings"].get(unit["unit_id"])
    adapter = entries[ref_id(candidates[int(selected)] if selected is not None else candidates[0])]
    template_path = args.source / "src/conf/eval/run_llama_general_full_matrix_r1.yaml"
    template = yaml.safe_load(template_path.read_text())
    wanted = set()
    if "T2X" in unit["tasks"]:
        wanted.add("t2x_xho")
    if "AfriHG" in unit["tasks"]:
        wanted.update(f"afrihg_{language}" for language in unit["languages"])
    generation_tasks = [
        task for task in template["evaluation"]["generation_tasks"] if task["id"] in wanted
    ]
    if {task["id"] for task in generation_tasks} != wanted:
        raise ValueError(f"Missing generation tasks: {wanted}")
    interface = {
        "mzansilm": {"dtype": "bfloat16", "merge_lora": False, "tie_word_embeddings": None},
        "mamba2": {"dtype": "float32", "merge_lora": False, "tie_word_embeddings": False},
        "xlstm": {"dtype": "float32", "merge_lora": True, "tie_word_embeddings": False},
        "gdn": {"dtype": "bfloat16", "merge_lora": False, "tie_word_embeddings": None},
    }[unit["architecture"]]
    config_dict = {
        "mode": "EVALUATE",
        "eval_model": {
            "checkpoint": base["path"],
            "peft_adapter": adapter["path"],
            "adapter": "hf",
            "device": "cuda:0",
            **interface,
        },
        "evaluation": {
            "task_packs": [],
            "generation_tasks": generation_tasks,
            "overrides": {},
            "output_dir": str(args.output),
            "wandb": None,
        },
        "wandb": None,
        "model": None,
        "data": None,
        "tokenizer": None,
        "training": None,
        "dataset": None,
        "peft": None,
        "template": None,
    }
    config = cast(
        ExperimentConfig,
        OmegaConf.merge(OmegaConf.structured(ExperimentConfig), OmegaConf.create(config_dict)),
    )
    run(config)
    summary = args.output / "evaluation_summary.json"
    if not summary.is_file():
        raise FileNotFoundError(summary)
    marker = {
        "schema": "sallm.full_matrix_generation_unit/v1",
        "unit": unit,
        "base": base,
        "adapter": adapter,
        "evaluation_summary_sha256": sha256(summary),
        "tasks": sorted(wanted),
    }
    marker_path = args.output / "UNIT_VERIFIED.json"
    marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")
    print(f"FULL_MATRIX_GENERATION_UNIT_OK index={args.index} sha256={sha256(marker_path)}")


if __name__ == "__main__":
    main()
