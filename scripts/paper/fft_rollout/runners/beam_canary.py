#!/usr/bin/env python3
"""Beam-search support check for one architecture (no test data): 5-beam generation with and without the cache.

For each short prompt, runs beam search (5 beams, length penalty 0.7, early stopping, as the v1 AfriHG setting) and
greedy, each with use_cache True and False, at batch 1. Records exceptions, whether the cached beam output equals the
cache-free one (a cache that is not reordered across beams gives different, wrong beams), and seconds per prompt.
  python beam_canary.py --checkpoint DIR --dtype bfloat16|float32 --out OUT.json
"""

import argparse
import json
import time
import traceback

import torch
from sallm.config import ModelEvalConfig
from sallm.evaluation.harness import load_model_and_tokenizer

PROMPTS = [
    "[BOS]<|user|>\nBhala isihloko esifutshane sale ndaba: Urhulumente uvule isikolo esitsha eMthatha namhlanje.[EOS]<|assistant|>\n        ",
    "[BOS]<|user|>\nBhala isihloko: Iqembu lebhola lezinyawo linqobe umdlalo wamanqamu ngamagoli amabili.[EOS]<|assistant|>\n        ",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dtype", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=40)
    args = ap.parse_args()
    model, tok = load_model_and_tokenizer(ModelEvalConfig(checkpoint=args.checkpoint, dtype=args.dtype, device="cuda:0"))
    model.eval()
    res = {"class": f"{type(model).__module__}.{type(model).__name__}", "config_use_cache": model.config.use_cache, "runs": {}}
    for mode, kw in (("beam", dict(num_beams=5, length_penalty=0.7, early_stopping=True)), ("greedy", dict(num_beams=1))):
        for cache in (True, False):
            key, outs, secs = f"{mode}_cache{int(cache)}", [], 0.0
            try:
                for p in PROMPTS:
                    ids = tok(p, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda:0")
                    torch.cuda.synchronize()
                    t0 = time.time()
                    with torch.no_grad():
                        g = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False, use_cache=cache,
                                           max_new_tokens=args.max_new_tokens, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id, **kw)
                    torch.cuda.synchronize()
                    secs += time.time() - t0
                    outs.append(g[0, ids.shape[1]:].tolist())
                res["runs"][key] = {"ok": True, "tokens": outs, "s_per_prompt": round(secs / len(PROMPTS), 3)}
            except Exception as exc:  # noqa: BLE001
                res["runs"][key] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300], "trace": traceback.format_exc()[-800:]}
    runs = res["runs"]
    for mode in ("beam", "greedy"):
        a, b = runs[f"{mode}_cache1"], runs[f"{mode}_cache0"]
        res[f"{mode}_cache_matches_nocache"] = a["ok"] and b["ok"] and a["tokens"] == b["tokens"]
    json.dump(res, open(args.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "runs"} | {k: (v["ok"], v.get("s_per_prompt"), v.get("error")) for k, v in runs.items()}))


if __name__ == "__main__":
    main()
