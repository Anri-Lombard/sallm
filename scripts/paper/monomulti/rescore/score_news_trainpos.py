#!/usr/bin/env python3
"""Score any LoRA adapter on MasakhaNEWS test with the frozen General natural first-token scorer (v10 prompts)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import time
from pathlib import Path

REPO = Path("/scratch/alombard/sallm_snapshots/news-general-corrected-official-20260914-v15")
MANIFEST = Path("/scratch/alombard/sallm/results/mamba_sixfamily_priority_replacement_20260915_v10/official/protocols/news_dataset_manifest.json")
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
    ap.add_argument("--adapter", required=True, type=Path)
    ap.add_argument("--langs", default="eng,xho")
    ap.add_argument("--template", default="auto", help="auto | normalized | normalized_bos | path to a chat_template.jinja")
    ap.add_argument("--limit", type=int, default=None)
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
    model, tok = load_model_and_tokenizer(ModelEvalConfig(checkpoint=str(args.base), peft_adapter=str(args.adapter), dtype="bfloat16",
                                                          device="cuda:0", merge_lora=True, tie_word_embeddings=False))
    model.eval()
    own = (args.adapter / "chat_template.jinja").read_text() if (args.adapter / "chat_template.jinja").exists() else str(tok.chat_template)
    bos_line = "{{- bos_token -}}\n"
    if args.template == "auto":
        template = (bos_line if "bos_token" in own else "") + common.NORMALIZED_CHAT_TEMPLATE
    elif args.template == "normalized":
        template = common.NORMALIZED_CHAT_TEMPLATE
    elif args.template == "normalized_bos":
        template = bos_line + common.NORMALIZED_CHAT_TEMPLATE
    else:
        template = Path(args.template).read_text()
    tok.chat_template = template
    ev = ClassificationEvaluator(tok, max_samples_per_lang=None)
    manifest = json.loads(MANIFEST.read_text())
    prompts = {k: str(common.load_yaml(REPO / f"src/conf/templates/masakhane_news_classification/lm_eval_{v}.yaml")["prompt"]) for k, v in PROMPTS.items()}
    rows, per_lang = [], {}
    for lang in args.langs.split(","):
        spec = manifest["files"][lang]
        payload = Path(spec["path"]).read_bytes()
        if hashlib.sha256(payload).hexdigest() != spec["sha256"]:
            raise SystemExit(f"{lang} test hash mismatch")
        data = list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig")), delimiter="\t"))
        lang_rows = []
        for i, row in enumerate(data):
            if args.limit is not None and i >= args.limit:
                break
            scored = diag._score_interfaces(model=model, tokenizer=tok, evaluator=ev,
                                            prompt=prompts[lang].format(headline=row["headline"], text=row["text"]), labels=LABELS)
            scored.pop("root_text", None)
            r = {"language": lang, "prompt_id": PROMPTS[lang], "source_index": i, "gold": row["category"].strip(),
                 "prediction": scored["first_token_prediction"], **scored}
            rows.append(r)
            lang_rows.append(r)
        per_lang[lang] = common.manual_classification_metrics([r["gold"] for r in lang_rows], [r["prediction"] for r in lang_rows])
    payload = {"arch": args.arch, "base": str(args.base), "adapter": str(args.adapter), "adapter_tree_sha256": tree_sha256(args.adapter),
               "prompts": {k: PROMPTS[k] for k in per_lang}, "limit": args.limit,
               "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(), "languages": per_lang,
               "runtime_seconds": time.monotonic() - started, "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, default=str))
    print(json.dumps({k: round(v["weighted_f1"], 4) for k, v in per_lang.items()}))


if __name__ == "__main__":
    main()
