#!/usr/bin/env python3
"""Score General adapters on InjongoIntent and Belebele test with the answer scored right after <|assistant|> (current) or after the training answer prefix.

Rollout copy (fft_rollout): the model is given by --base/--arch (a fully fine-tuned model, no adapter) instead of a spec file;
--limit caps items per task (smoke tests only). Everything else is prompt/run_prefix_eval.py unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import cast

from lm_eval.models.huggingface import HFLM
from omegaconf import OmegaConf
from sallm.config import ExperimentConfig
from sallm.evaluation.run import run

sys.path.insert(0, "/scratch/lmbanr001/masters/sallm_snapshots/full-matrix-execution-20260916-v1")
from prepare_bindings import tree_sha256  # noqa: E402

HERE = Path(__file__).resolve().parent
TRAIN_PREFIX = "\n        "
# lm-eval drops target_delimiter under apply_chat_template, so the current protocol scores the label directly after <|assistant|>.
CONTEXT_SUFFIX = {"current": "", "train": TRAIN_PREFIX}
PACKS = ["injongointent_all"] + [f"belebele_{lang}" for lang in ("afr", "eng", "sot", "ssw", "tsn", "tso", "xho", "zul")]
INTERFACE = {
    "mzansilm": {"dtype": "bfloat16", "merge_lora": False, "tie_word_embeddings": None},
    "mamba2": {"dtype": "float32", "merge_lora": False, "tie_word_embeddings": False},
    "xlstm": {"dtype": "float32", "merge_lora": True, "tie_word_embeddings": False},
    "gdn": {"dtype": "bfloat16", "merge_lora": False, "tie_word_embeddings": None},
}


def append_training_prefix() -> None:
    original = HFLM.apply_chat_template

    def patched(self, chat_history, add_generation_prompt=True):
        rendered = original(self, chat_history, add_generation_prompt)
        return rendered + TRAIN_PREFIX if add_generation_prompt else rendered

    HFLM.apply_chat_template = patched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--arch", required=True, choices=sorted(INTERFACE))
    parser.add_argument("--mode", required=True, choices=sorted(CONTEXT_SUFFIX))
    parser.add_argument("--packs", nargs="+", default=PACKS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    unit = {"unit_id": Path(args.base).name, "architecture": args.arch, "base": args.base, "adapter": None,
            "base_tree_sha256": tree_sha256(Path(args.base)), "adapter_tree_sha256": None}
    if args.mode == "train":
        append_training_prefix()
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
            "task_packs": list(args.packs),
            "generation_tasks": [],
            "overrides": {p: {"limit": args.limit} for p in args.packs} if args.limit else {},
            "output_dir": str(args.output),
        },
    }
    config = cast(ExperimentConfig, OmegaConf.merge(OmegaConf.structured(ExperimentConfig), OmegaConf.create(config_dict)))
    run(config)
    suffix = "<|assistant|>" + CONTEXT_SUFFIX[args.mode]
    counts = {}
    for pack in args.packs:
        result = json.loads((args.output / pack / "results.json").read_text())
        for task, samples in result["samples"].items():
            cfg = result["configs"][task]
            assert cfg["num_fewshot"] == 0 and cfg["test_split"] == "test", task
            for sample in samples:
                for request in sample["arguments"]:
                    assert request[0].endswith(suffix), request[0][-40:]
                    assert request[1] in cfg["doc_to_choice"], request[1]
            counts[task] = len(samples)
    marker = {
        "schema": "sallm.general_prefix_fix/v1",
        "unit": unit,
        "mode": args.mode,
        "context_suffix_after_assistant_tag": CONTEXT_SUFFIX[args.mode],
        "interface": INTERFACE[unit["architecture"]],
        "packs": list(args.packs),
        "samples_per_task": counts,
    }
    (args.output / "UNIT_DONE.json").write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")
    print(f"PREFIX_EVAL_OK arch={args.arch} mode={args.mode}")


if __name__ == "__main__":
    main()
