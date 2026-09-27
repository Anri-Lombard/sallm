#!/usr/bin/env python3
"""Score LoRA adapters on a SIB-200 split (validation for checkpoint selection) with the frozen General natural first-token scorer (v10 prompts)."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import torch
import yaml
from datasets import Dataset

PROTOCOL = Path("/scratch/lmbanr001/masters/sallm/results/monomulti_retrain_20260924/sibkit/sib_protocol_hex.json")
PROMPTS = {"afr": 4, "eng": 3, "nso": 4, "sot": 3, "xho": 5, "zul": 5}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tree_sha256(root: Path) -> str:
    files = sorted(p for p in root.resolve().rglob("*") if p.is_file() and ".cache" not in p.relative_to(root.resolve()).parts)
    return hashlib.sha256("".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  ./{p.relative_to(root.resolve()).as_posix()}\n" for p in files).encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=("mzansilm", "mamba2", "xlstm", "gdn"))
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--adapter", required=True, type=Path, nargs="+")
    ap.add_argument("--split", required=True, choices=("validation", "test"))
    ap.add_argument("--langs", default=",".join(PROMPTS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--train-template", type=Path, default=None, help="training template YAML; its prompt gets text=text.strip()")
    ap.add_argument("--tag", default="shared")
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    for adapter in args.adapter:
        score_one(args, adapter, args.out_dir / f"{adapter.name}.{args.split}.{args.tag}.json")


def score_one(args, adapter: Path, out: Path) -> None:
    if out.exists():
        raise SystemExit(f"{out} exists")
    protocol = json.loads(PROTOCOL.read_text())
    runtime, test = protocol["runtime"], protocol["test"]
    scorer = load_module(Path(runtime["validation_scorer"]), "sealed_sib_validation")
    core = scorer.load_core(Path(runtime["validation_core"]))
    xlstm = args.arch == "xlstm"
    started = time.monotonic()
    model, tokenizer = scorer.load_model_and_tokenizer(scorer.ModelEvalConfig(
        checkpoint=str(args.base), dtype="float32", device="cuda:0", peft_adapter=str(adapter),
        merge_lora=True if xlstm else None, tie_word_embeddings=False if xlstm else None))
    model.eval()
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    mult = scorer.ClassificationEvaluator._get_model_chunk_size(model)
    rows, per_lang = [], {}
    with torch.inference_mode():
        for lang in args.langs.split(","):
            entry = test["languages"][lang]
            if entry["prompt"] != PROMPTS[lang]:
                raise SystemExit(f"protocol prompt for {lang} differs")
            train_prompt = None
            if args.train_template is not None:
                spec_t = yaml.safe_load(args.train_template.read_text())
                if sorted(spec_t["label_mapping"].values()) != sorted(core.LABELS):
                    raise SystemExit("training template labels differ")
                train_prompt = str(spec_t["prompt"])
            template = yaml.safe_load((Path(runtime["snapshot"]) / "src/conf/eval/lm_eval_tasks/sib_validation" / f"sallm_sib_{lang}_val_prompt_{PROMPTS[lang]}.yaml").read_text())["doc_to_text"]
            data = Dataset.from_file(str(Path(test["cache_root"]) / entry["subset"] / "0.0.0" / test["revision"] / f"sib200-{args.split}.arrow"))
            if len(data) != {"test": test["rows_per_language"], "validation": 99}[args.split]:
                raise SystemExit(f"{lang} rows differ")
            lang_rows = []
            for i, item in enumerate(data):
                if args.limit is not None and i >= args.limit:
                    break
                scores, first_ids, hand = scorer.score_prompt(model=model, tokenizer=tokenizer, prompt=template.replace("{{text}}", str(item["text"])) if train_prompt is None else train_prompt.format(text=str(item["text"]).strip()),
                                                              pad_token_id=int(pad), pad_to_multiple_of=mult, device=model.device, core=core)
                guess, margin, tie = scorer.prediction(scores)
                row = {"language": lang, "prompt": PROMPTS[lang], "doc_id": int(item["index_id"]), "gold": str(item["category"]),
                       "prediction": guess, "scores": scores, "tie": tie, "context_tokens": hand["training_label_prefix_tokens"]}
                rows.append(row)
                lang_rows.append(row)
            per_lang[lang] = scorer.summarize(lang_rows)
    payload = {"arch": args.arch, "base": str(args.base), "adapter": str(adapter), "adapter_tree_sha256": tree_sha256(adapter), "split": args.split,
               "protocol": str(PROTOCOL), "prompts": {k: PROMPTS[k] for k in per_lang}, "limit": args.limit,
               "train_template": str(args.train_template) if args.train_template else None,
               "train_template_sha256": hashlib.sha256(args.train_template.read_bytes()).hexdigest() if args.train_template else None,
               "chat_template_sha256": hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest(),
               "languages": {k: v["metrics"] for k, v in per_lang.items()}, "ties": sum(r["tie"] for r in rows),
               "runtime_seconds": time.monotonic() - started, "rows": rows}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload))
    print(adapter.name, json.dumps({k: round(v["f1"], 4) for k, v in payload["languages"].items()}), flush=True)
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
