#!/usr/bin/env python3
"""MasakhaNEWS General protocol: the Phase-2 driver score_news_trainpos.py (Kombuys monomulti_rescore_inventory_20260924)
with HEX data paths and a --split switch. Frozen v15 scoring code (kit/news_v15, byte-identical to the Kombuys
v15 snapshot files it contains), bf16 + merge_lora=True + tie=False, normalized template with the BOS line when the
adapter's own template has one ("auto", as Phase 2), natural first-label-token scorer, support-weighted F1.
Validation = MasakhaNEWS dev.tsv at the test TSVs' revision fa3b5fff.
Rollout copy (fft_rollout): --adapter is optional (a fully fine-tuned model is passed as --base), --limit caps rows per language (smoke tests only), frozen kit files are read from the HEX reselect kit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
import time
from pathlib import Path

REPO = Path("/scratch/lmbanr001/masters/sallm/results/monomulti_reselect_20260925/jobs/kit/news_v15")
DATA = Path(os.environ["HF_HUB_CACHE"]) / "datasets--masakhane--masakhanews/snapshots/fa3b5fff8a91d187bf0c5900a39c4271d08cf7fe/data"
FILES = {
    ("eng", "test"): "d7e4fdf4f71a8ec5b3cb4a183030a64501a2cbc9d1ee011eb430f4fdd465800e",
    ("xho", "test"): "5687dc04056e19403253e28ee52b921351c5cb0257aec21de0a23cb4f9f0ee9a",
    ("eng", "validation"): "0b524e10acf3e6b1ac8d112dda399386d81a8dd084cd361162654be2e23d4b0f",
    ("xho", "validation"): "d0f20df11816eb0281493548819ef2cdf126b0e10c78d0a23d16a1e97c250aa7",
}
PROMPTS = {"eng": "p2", "xho": "p4"}
LABELS = ["business", "entertainment", "health", "politics", "religion", "sports", "technology"]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tree_sha256(root: Path) -> str:
    root = root.resolve()
    files = sorted(p for p in root.rglob("*") if p.is_file() and ".cache" not in p.relative_to(root).parts)
    return hashlib.sha256("".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  ./{p.relative_to(root).as_posix()}\n" for p in files).encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=("mzansilm", "mamba2", "xlstm", "gdn"))
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--adapter", type=Path)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--split", required=True, choices=("validation", "test"))
    ap.add_argument("--langs", default="eng,xho")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists")
    common = load_module(REPO / "scripts/run_mamba_news_common_validation.py", "news_common")
    diag = load_module(REPO / "scripts/run_mamba_news_interface_diagnostic.py", "news_diag")
    diag.common = common
    from sallm.config import ModelEvalConfig
    from sallm.evaluation.classification_metrics import ClassificationEvaluator
    from sallm.evaluation.harness import load_model_and_tokenizer

    started = time.monotonic()
    model, tok = load_model_and_tokenizer(ModelEvalConfig(checkpoint=str(args.base), peft_adapter=str(args.adapter) if args.adapter else None, dtype="bfloat16",
                                                          device="cuda:0", merge_lora=True, tie_word_embeddings=False))
    model.eval()
    own_dir = args.adapter or args.base
    own = (own_dir / "chat_template.jinja").read_text() if (own_dir / "chat_template.jinja").exists() else str(tok.chat_template)
    template = ("{{- bos_token -}}\n" if "bos_token" in own else "") + common.NORMALIZED_CHAT_TEMPLATE
    tok.chat_template = template
    ev = ClassificationEvaluator(tok, max_samples_per_lang=None)
    prompts = {k: str(common.load_yaml(REPO / f"src/conf/templates/masakhane_news_classification/lm_eval_{v}.yaml")["prompt"]) for k, v in PROMPTS.items()}
    rows, per_lang, data_files = [], {}, {}
    for lang in args.langs.split(","):
        path = DATA / lang / ("dev.tsv" if args.split == "validation" else "test.tsv")
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != FILES[(lang, args.split)]:
            raise SystemExit(f"{path} hash mismatch")
        data_files[lang] = {"path": str(path), "sha256": FILES[(lang, args.split)]}
        data = list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig")), delimiter="\t"))[: args.limit]
        lang_rows = []
        for i, row in enumerate(data):
            scored = diag._score_interfaces(model=model, tokenizer=tok, evaluator=ev,
                                            prompt=prompts[lang].format(headline=row["headline"], text=row["text"]), labels=LABELS)
            scored.pop("root_text", None)
            r = {"language": lang, "prompt_id": PROMPTS[lang], "source_index": i, "gold": row["category"].strip(),
                 "prediction": scored["first_token_prediction"], **scored}
            rows.append(r)
            lang_rows.append(r)
        per_lang[lang] = common.manual_classification_metrics([r["gold"] for r in lang_rows], [r["prediction"] for r in lang_rows])
        per_lang[lang]["n_items"] = len(lang_rows)
    payload = {"schema": "reselect.news_eval/v1", "split": args.split, "arch": args.arch, "base": str(args.base), "adapter": str(args.adapter),
               "adapter_tree_sha256": tree_sha256(args.adapter) if args.adapter else None, "model_tree_sha256": tree_sha256(args.adapter or args.base), "limit": args.limit, "prompts": {k: PROMPTS[k] for k in per_lang}, "data_files": data_files,
               "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(), "languages": per_lang,
               "runtime_seconds": time.monotonic() - started, "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, default=str))
    print("NEWS_EVAL_OK", json.dumps({k: round(v["weighted_f1"] * 100, 4) for k, v in per_lang.items()}))


if __name__ == "__main__":
    main()
