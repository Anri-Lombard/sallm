"""Aggregate bench runs -> throughput.csv (CPU only). Usage: python aggregate.py RUNS_DIR OUT_CSV"""
import csv, glob, json, sys
from pathlib import Path

import numpy as np

WARM = 20  # steps excluded (Triton autotune / compile)
L40S_BF16_DENSE = 362e12  # NVIDIA datasheet dense BF16 TFLOPS (with sparsity 733); MFU reference only

rows = []
for d in sorted(glob.glob(f"{sys.argv[1]}/bench_*")):
    d = Path(d)
    if not (d / "run_meta.json").exists() or not (d / "train_log.jsonl").exists():
        continue
    meta = json.loads((d / "run_meta.json").read_text())
    tl = [json.loads(x) for x in open(d / "train_log.jsonl")]
    ev = [json.loads(x) for x in open(d / "event_log.jsonl")] if (d / "event_log.jsonl").exists() else []
    if len(tl) <= WARM + 5:
        continue
    st = tl[WARM:]
    steps = len(st)
    tok = sum(meta["tokens_per_step"] for _ in st)
    t = sum(r["step_time"] for r in st)
    g = meta["world"]
    fpt = meta["flops"]["train_total"]
    evs = [e for e in ev if e["kind"] == "eval"]
    sw = [e["seconds"] for e in ev if e["kind"] == "save_weights"]
    sr = [e["seconds"] for e in ev if e["kind"] == "save_resume"]
    pw = np.nan
    if (d / "power.csv").exists():
        vals = []
        for line in open(d / "power.csv"):
            parts = [p.strip() for p in line.split(",")]
            try:
                vals.append(float(parts[3].split()[0]))
            except (IndexError, ValueError):
                pass
        # mean board power over the whole job per sampled GPU (includes startup/eval; conservative)
        pw = float(np.mean(vals)) if vals else np.nan
    tps = tok / t
    rows.append({
        "run": d.name, "arch": meta["cfg"]["arch"], "stack": meta.get("stack"), "compile": meta.get("compile"),
        "gpus": g, "gpu": meta["env"]["gpu"], "micro_batch": meta["micro_batch"], "grad_accum": meta["grad_accum"],
        "global_batch_tokens": meta["tokens_per_step"], "timed_steps": steps,
        "tokens_per_s": round(tps), "tokens_per_s_per_gpu": round(tps / g),
        "sec_per_step_median": float(np.median([r["step_time"] for r in st])),
        "data_time_frac": sum(r["data_time"] for r in st) / t,
        "peak_mem_gb": max(r["peak_mem_gb"] for r in st),
        "first_step_s": tl[0]["step_time"], "startup_to_step1_s": tl[0]["wall"],
        "loss_first": tl[0]["loss"], "loss_last": tl[-1]["loss"],
        "eval_seconds": evs[0]["seconds"] if evs else None, "eval_ALL_bpb": evs[0].get("ALL/bpb") if evs else None,
        "save_weights_s": max(sw) if sw else None, "save_resume_s": max(sr) if sr else None,
        "flops_per_token_train": fpt, "achieved_tflops_per_gpu": tps * fpt / g / 1e12,
        "mfu_vs_362tflops": tps * fpt / g / L40S_BF16_DENSE,
        "mean_power_w_per_gpu": pw, "joules_per_token": pw * g / tps if pw == pw else None,
        "kernels": json.dumps(meta.get("kernels")),
    })
    print(rows[-1]["run"], rows[-1]["tokens_per_s"], rows[-1]["peak_mem_gb"])
if rows:
    with open(sys.argv[2], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
