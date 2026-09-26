#!/usr/bin/env python3
"""InjongoIntent General protocol (run_prefix_eval.py --mode train, sha d0066af2) restricted to the protocol-prompt tasks.

Unchanged from run_prefix_eval.py: sallm EVALUATE config (downstream-generation-20260914-v8), pack injongointent_all
(0-shot, apply_chat_template, batch auto:4, max 64), per-arch INTERFACE, the "\\n        " suffix after <|assistant|>,
and the context/continuation asserts. Changes: only the protocol-prompt task of each requested language is run
(pack override `tasks`), and --split validation swaps the test docs of those tasks for the train-held-out rows
built by intent_val_split.py (--val-source carve) or the upstream dev rows from intent_dev_rows.py
(--val-source upstream_dev); same doc_to_text/doc_to_choice/metrics.
Rollout copy (fft_rollout): --adapter is optional (a fully fine-tuned model is passed as --base), --limit caps rows per language (smoke tests only), frozen kit files are read from the HEX reselect kit.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import cast

from datasets import Dataset
from lm_eval.api.task import ConfigurableTask
from lm_eval.models.huggingface import HFLM
from omegaconf import OmegaConf
from sallm.config import ExperimentConfig
from sallm.evaluation.run import run
from sklearn.metrics import f1_score

TRAIN_PREFIX = "\n        "
PROMPTS = {"eng": 4, "sot": 1, "xho": 2, "zul": 2}
INTERFACE = {
    "mzansilm": {"dtype": "bfloat16", "merge_lora": False, "tie_word_embeddings": None},
    "mamba2": {"dtype": "float32", "merge_lora": False, "tie_word_embeddings": False},
    "xlstm": {"dtype": "float32", "merge_lora": True, "tie_word_embeddings": False},
    "gdn": {"dtype": "bfloat16", "merge_lora": False, "tie_word_embeddings": None},
}
VAL_DIRS = {"carve": Path("/scratch/lmbanr001/masters/sallm/results/monomulti_reselect_20260925/jobs/kit/intent_val"), "upstream_dev": Path("/scratch/lmbanr001/masters/sallm/results/monomulti_reselect_20260925/jobs/kit/intent_val_upstream_dev")}


def append_training_prefix() -> None:
    original = HFLM.apply_chat_template

    def patched(self, chat_history, add_generation_prompt=True):
        rendered = original(self, chat_history, add_generation_prompt)
        return rendered + TRAIN_PREFIX if add_generation_prompt else rendered

    HFLM.apply_chat_template = patched


def use_validation_docs(tasks: dict[str, str], val_dir: Path) -> None:
    docs = {task: Dataset.from_list([json.loads(x) for x in (val_dir / f"{lang}.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()])
            for task, lang in tasks.items()}
    original = ConfigurableTask.test_docs

    def test_docs(self):
        return docs[self.config.task] if self.config.task in docs else original(self)

    ConfigurableTask.test_docs = test_docs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=sorted(INTERFACE))
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--split", required=True, choices=("validation", "test"))
    ap.add_argument("--langs", default="eng,sot,xho,zul")
    ap.add_argument("--val-source", choices=sorted(VAL_DIRS), default="carve")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    langs = args.langs.split(",")
    tasks = {f"injongointent_{lang}_prompt_{PROMPTS[lang]}": lang for lang in langs}
    append_training_prefix()
    if args.split == "validation":
        use_validation_docs(tasks, VAL_DIRS[args.val_source])
    config_dict = {
        "mode": "EVALUATE",
        "eval_model": {"checkpoint": args.base, "peft_adapter": args.adapter, "adapter": "hf", "device": "cuda:0", **INTERFACE[args.arch]},
        "evaluation": {"task_packs": ["injongointent_all"], "generation_tasks": [], "overrides": {"injongointent_all": {"tasks": list(tasks), **({"limit": args.limit} if args.limit else {})}},
                       "output_dir": str(args.output)},
    }
    run(cast(ExperimentConfig, OmegaConf.merge(OmegaConf.structured(ExperimentConfig), OmegaConf.create(config_dict))))
    result = json.loads((args.output / "injongointent_all" / "results.json").read_text())
    assert set(result["samples"]) == set(tasks), sorted(result["samples"])
    suffix = "<|assistant|>" + TRAIN_PREFIX
    per_lang = {}
    for task, samples in result["samples"].items():
        cfg = result["configs"][task]
        choices = cfg["doc_to_choice"]
        assert cfg["num_fewshot"] == 0 and cfg["test_split"] == "test", task
        gold, pred = [], []
        for sample in samples:
            for request in sample["arguments"]:
                assert request[0].endswith(suffix), request[0][-40:]
                assert request[1] in choices, request[1]
            lls = [float(r[0]) for r in sample["filtered_resps"]]
            pred.append(choices[max(range(len(lls)), key=lls.__getitem__)])
            gold.append(sample["doc"]["intent"])
        f1 = float(result["results"][task]["f1,none"])
        check = f1_score(gold, pred, average="weighted")
        assert abs(f1 - check) < 1e-6, (task, f1, check)
        top = Counter(pred).most_common(1)[0]
        per_lang[tasks[task]] = {"task": task, "prompt": f"p{PROMPTS[tasks[task]]}", "weighted_f1": f1, "acc": float(result["results"][task]["acc,none"]),
                                 "n_items": len(samples), "distinct_predicted_labels": len(set(pred)), "top_prediction": top[0],
                                 "top_prediction_share": top[1] / len(pred)}
    summary = {"schema": "reselect.intent_eval/v1", "split": args.split, "arch": args.arch, "base": args.base, "adapter": args.adapter,
               "interface": INTERFACE[args.arch], "context_suffix_after_assistant_tag": TRAIN_PREFIX, "languages": per_lang,
               "val_source": args.val_source if args.split == "validation" else None,
               "validation_manifest": str(VAL_DIRS[args.val_source] / "MANIFEST.json") if args.split == "validation" else None}
    (args.output / "SUMMARY.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("INTENT_EVAL_OK", json.dumps({k: round(v["weighted_f1"] * 100, 4) for k, v in per_lang.items()}))


if __name__ == "__main__":
    main()
