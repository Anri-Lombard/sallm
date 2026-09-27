#!/usr/bin/env python3
"""SIB-200 General protocol: the Phase-2 driver score_sib_trainpos.py / HEX score_sib_split_hex.py with a --split switch.
Frozen scorer run_sib_natural_first_token_eval.py (b247505a) + core run_sib_balanced_code_eval.py (866e9d82) from kit/sib,
fp32, batch 1 right-padded to the model chunk size, xLSTM merge_lora=True tie=False, v10 prompts, support-weighted F1.
Validation = Davlan/sib200 validation split (99 rows per language) at the test split's cached revision.
Rollout copy (fft_rollout): --adapter is optional (a fully fine-tuned model is passed as --base), --limit caps rows per language (smoke tests only), frozen kit files are read from the HEX reselect kit.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

import torch
import yaml
from datasets import Dataset

KIT = Path("/scratch/lmbanr001/masters/sallm/results/monomulti_reselect_20260925/jobs/kit/sib")
PROTOCOL = KIT / "sib_protocol_hex.json"
PROMPTS = {"afr": 4, "eng": 3, "nso": 4, "sot": 3, "xho": 5, "zul": 5}
ROWS = {"test": 204, "validation": 99}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_sha256(root: Path) -> str:
    root = root.resolve()
    files = sorted(p for p in root.rglob("*") if p.is_file() and ".cache" not in p.relative_to(root).parts)
    return hashlib.sha256("".join(f"{sha(p)}  ./{p.relative_to(root).as_posix()}\n" for p in files).encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=("mzansilm", "mamba2", "xlstm", "gdn"))
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--adapter", type=Path)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--split", required=True, choices=("validation", "test"))
    ap.add_argument("--langs", default=",".join(PROMPTS))
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists")
    protocol = json.loads(PROTOCOL.read_text())
    runtime, test = protocol["runtime"], protocol["test"]
    scorer_path, core_path = KIT / "run_sib_natural_first_token_eval.py", KIT / "run_sib_balanced_code_eval.py"
    assert sha(scorer_path) == runtime["validation_scorer_sha256"] and sha(core_path) == runtime["validation_core_sha256"]
    scorer = load_module(scorer_path, "sealed_sib_validation")
    core = scorer.load_core(core_path)
    xlstm = args.arch == "xlstm"
    started = time.monotonic()
    model, tokenizer = scorer.load_model_and_tokenizer(scorer.ModelEvalConfig(
        checkpoint=str(args.base), dtype="float32", device="cuda:0", peft_adapter=str(args.adapter) if args.adapter else None,
        merge_lora=True if xlstm else None, tie_word_embeddings=False if xlstm else None))
    model.eval()
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    mult = scorer.ClassificationEvaluator._get_model_chunk_size(model)
    rows, per_lang, data_files = [], {}, {}
    with torch.inference_mode():
        for lang in args.langs.split(","):
            entry = test["languages"][lang]
            if entry["prompt"] != PROMPTS[lang]:
                raise SystemExit(f"protocol prompt for {lang} differs")
            prompt_yaml = KIT / "src/conf/eval/lm_eval_tasks/sib_validation" / f"sallm_sib_{lang}_val_prompt_{PROMPTS[lang]}.yaml"
            assert sha(prompt_yaml) == entry["prompt_yaml_sha256"], prompt_yaml
            template = yaml.safe_load(prompt_yaml.read_text())["doc_to_text"]
            arrow = Path(os.environ["HF_DATASETS_CACHE"]) / "Davlan___sib200" / entry["subset"] / "0.0.0" / test["revision"] / f"sib200-{args.split}.arrow"
            if args.split == "test":
                assert sha(arrow) == entry["arrow_sha256"], arrow
            data_files[lang] = {"path": str(arrow), "sha256": sha(arrow)}
            data = Dataset.from_file(str(arrow))
            if len(data) != ROWS[args.split]:
                raise SystemExit(f"{lang} rows differ")
            lang_rows = []
            if args.limit:
                data = data.select(range(min(args.limit, len(data))))
            for item in data:
                scores, first_ids, hand = scorer.score_prompt(model=model, tokenizer=tokenizer, prompt=template.replace("{{text}}", str(item["text"])),
                                                              pad_token_id=int(pad), pad_to_multiple_of=mult, device=model.device, core=core)
                guess, margin, tie = scorer.prediction(scores)
                row = {"language": lang, "prompt": PROMPTS[lang], "doc_id": int(item["index_id"]), "gold": str(item["category"]),
                       "prediction": guess, "scores": scores, "tie": tie, "context_tokens": hand["training_label_prefix_tokens"]}
                rows.append(row)
                lang_rows.append(row)
            per_lang[lang] = scorer.summarize(lang_rows)["metrics"]
            per_lang[lang]["n_items"] = len(lang_rows)
    payload = {"schema": "reselect.sib_eval/v1", "split": args.split, "arch": args.arch, "base": str(args.base), "adapter": str(args.adapter),
               "adapter_tree_sha256": tree_sha256(args.adapter) if args.adapter else None, "model_tree_sha256": tree_sha256(args.adapter or args.base), "limit": args.limit, "protocol": str(PROTOCOL), "prompts": {k: PROMPTS[k] for k in per_lang},
               "data_files": data_files, "chat_template_sha256": hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest(),
               "languages": per_lang, "ties": sum(r["tie"] for r in rows), "runtime_seconds": time.monotonic() - started, "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload))
    print("SIB_EVAL_OK", json.dumps({k: round(v["f1"] * 100, 4) for k, v in per_lang.items()}))


if __name__ == "__main__":
    main()
