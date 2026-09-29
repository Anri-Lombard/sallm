#!/usr/bin/env python3
"""Controlled efficiency benchmark of one ~125M decoder (training step, prefill, decode) on an otherwise idle GPU.

Loads the checkpoint with the same loader and per-architecture interface settings as the scorers (gen_direct.INTERFACE),
random token ids, median and IQR over timed iterations after warmup. See RUNBOOK.md "Efficiency benchmark".
Training uses fp32 master weights with bf16 autocast (the rollout's fine-tuning setup) when --dtype bfloat16, and plain
fp32 (TF32 off) when --dtype float32. Prefill and decode load the weights in --dtype. --dry-run needs no torch.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ARCHS = ("mzansilm", "mamba2", "xlstm", "gdn")
TIE = {"mzansilm": None, "mamba2": False, "xlstm": False, "gdn": None}  # gen_direct.INTERFACE tie_word_embeddings
TRAIN = {"batch": (4, 8), "seq": (512, 2048), "warmup": 5, "iters": 20}
PREFILL = {"batch": (1, 8), "seq": (512, 2048), "warmup": 5, "iters": 20}
DECODE = {"batch": (1, 8), "prompt": 512, "new": 256, "warmup": 1, "iters": 5}
LR = 1e-5


def grid() -> list[str]:
    g = [f"train  batch={b} seq={s}  ({TRAIN['warmup']} warmup + {TRAIN['iters']} timed)" for b in TRAIN["batch"] for s in TRAIN["seq"]]
    g += [f"prefill batch={b} seq={s}  ({PREFILL['warmup']} warmup + {PREFILL['iters']} timed)" for b in PREFILL["batch"] for s in PREFILL["seq"]]
    g += [f"decode batch={b} prompt={DECODE['prompt']} new={DECODE['new']} greedy  ({DECODE['warmup']} warmup + {DECODE['iters']} timed)" for b in DECODE["batch"]]
    return g


def summarize(times: list[float]) -> dict:
    q = statistics.quantiles(times, n=4, method="inclusive")
    return {"median_s": statistics.median(times), "iqr_s": q[2] - q[0], "n": len(times)}


def gpu_users() -> str:
    r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                       capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else f"nvidia-smi unavailable: {r.stderr.strip()}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", choices=ARCHS, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    ap.add_argument("--device", default="cuda:0", help="cpu is for smoke tests only")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.dry_run:
        print(f"efficiency_bench arch={args.arch} dtype={args.dtype} checkpoint={args.checkpoint} output={args.output}")
        print("\n".join(grid()))
        return
    if args.output.exists():
        raise FileExistsError(args.output)

    import gc

    import torch
    import transformers
    from sallm.config import ModelEvalConfig
    from sallm.evaluation.harness import load_model_and_tokenizer

    dev = torch.device(args.device)
    cuda = dev.type == "cuda"
    result: dict = {"arch": args.arch, "dtype": args.dtype, "checkpoint": str(args.checkpoint),
                    "timestamp": datetime.now(timezone.utc).isoformat(), "hostname": platform.node(),
                    "gpu": torch.cuda.get_device_name(dev) if cuda else "cpu", "gpu_users_at_start": gpu_users() if cuda else "",
                    "torch": torch.__version__, "transformers": transformers.__version__, "python": sys.version.split()[0],
                    "protocol": {"train": TRAIN, "prefill": PREFILL, "decode": DECODE, "lr": LR}}
    try:
        import fla
        result["fla"] = fla.__version__
    except ImportError:
        result["fla"] = None
    if args.arch == "xlstm":  # same as gen_direct_bs1.py: recurrent states stay fp32 whatever the weight dtype
        import xlstm_cache_fp32  # noqa: F401
        result["xlstm_cache_fp32_patch"] = True

    def sync() -> None:
        if cuda:
            torch.cuda.synchronize(dev)

    def cleanup() -> None:
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()

    def measure(fn, warmup: int, iters: int) -> dict:
        """fn() runs one iteration. Returns time summary + peak memory (includes resident weights), or 'oom'."""
        try:
            for _ in range(warmup):
                fn()
            sync()
            if cuda:
                torch.cuda.reset_peak_memory_stats(dev)
            times = []
            for _ in range(iters):
                sync()
                t = time.perf_counter()
                fn()
                sync()
                times.append(time.perf_counter() - t)
            out = summarize(times)
            out["peak_mem_gib"] = torch.cuda.max_memory_allocated(dev) / 2**30 if cuda else None
            return out
        except torch.cuda.OutOfMemoryError:
            return {"status": "oom"}
        finally:
            cleanup()

    def load(dtype: str):
        return load_model_and_tokenizer(ModelEvalConfig(checkpoint=str(args.checkpoint), dtype=dtype, device=args.device,
                                                        tie_word_embeddings=TIE[args.arch]))

    def rand_ids(model, tok, b: int, s: int):
        vocab = min(len(tok), int(model.get_input_embeddings().weight.shape[0]))
        g = torch.Generator().manual_seed(0)
        return torch.randint(3, vocab, (b, s), generator=g).to(dev)

    # ---- training: forward + backward + AdamW step
    bf16 = args.dtype == "bfloat16"
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = bf16
    model, tok = load("float32")
    result["n_params"] = sum(p.numel() for p in model.parameters())
    result["train_autocast_bf16"] = bf16
    if args.arch == "xlstm" and cuda:  # padded TFLA training kernel, as train_fft.py; native kernel if it cannot load
        try:
            import tfla
            result["tfla_backends_swapped"] = tfla.use_tfla(model)
            result["train_kernel"] = tfla.NAME
        except Exception as exc:  # noqa: BLE001
            result["train_kernel"] = f"native (tfla failed: {type(exc).__name__}: {str(exc)[:200]})"
    model.requires_grad_(True)
    model.train()
    model.config.use_cache = False
    result["train"] = {}
    for b in TRAIN["batch"]:
        for s in TRAIN["seq"]:
            ids = rand_ids(model, tok, b, s)
            opt = torch.optim.AdamW(model.parameters(), lr=LR, fused=cuda)

            def step() -> None:
                with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=bf16):
                    loss = model(input_ids=ids, labels=ids, use_cache=False).loss
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)

            r = measure(step, TRAIN["warmup"], TRAIN["iters"])
            if "median_s" in r:
                r["tokens_per_s"] = b * s / r["median_s"]
            result["train"][f"b{b}_s{s}"] = r
            print(f"train b={b} s={s}: {r}", flush=True)
            opt.zero_grad(set_to_none=True)
            del opt, ids
            cleanup()
    del model
    cleanup()

    # ---- inference: weights in --dtype
    model, tok = load(args.dtype)
    model.eval()
    result["prefill"], result["decode"] = {}, {}
    for b in PREFILL["batch"]:
        for s in PREFILL["seq"]:
            ids = rand_ids(model, tok, b, s)

            def fwd() -> None:
                with torch.no_grad():
                    model(input_ids=ids, use_cache=False)

            r = measure(fwd, PREFILL["warmup"], PREFILL["iters"])
            if "median_s" in r:
                r["tokens_per_s"] = b * s / r["median_s"]
            result["prefill"][f"b{b}_s{s}"] = r
            print(f"prefill b={b} s={s}: {r}", flush=True)
    n = DECODE["new"]
    for b in DECODE["batch"]:
        ids = rand_ids(model, tok, b, DECODE["prompt"])
        made = []

        def gen() -> None:
            with torch.no_grad():  # min == max forces exactly n tokens (EOS is suppressed), so every architecture decodes n steps
                out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False, num_beams=1,
                                     max_new_tokens=n, min_new_tokens=n, use_cache=True, pad_token_id=tok.pad_token_id)
            made.append(out.shape[1] - ids.shape[1])

        r = measure(gen, DECODE["warmup"], DECODE["iters"])
        if "median_s" in r:  # generate() = prefill of the prompt + n cached steps
            assert set(made) == {n}, made
            r.update(new_tokens=n, tokens_per_s=b * n / r["median_s"], per_token_ms=1000 * r["median_s"] / n)
        result["decode"][f"b{b}"] = r
        print(f"decode b={b}: {r}", flush=True)
    result["gpu_users_at_end"] = gpu_users() if cuda else ""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    print(f"EFFICIENCY_BENCH_OK {args.output}")


if __name__ == "__main__":
    main()
