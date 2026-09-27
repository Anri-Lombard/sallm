#!/usr/bin/env python3
"""xLSTM TFLA training-kernel checks (27 Sep 2026; results in RUNBOOK "Speed settings").

  tfla_check.py parity BASE OUT.json      one batch per shape, transformers' native kernel vs TFLA (tfla.use_tfla) on the
                                         fp32 model under bf16 autocast: loss, grad norm/cosine, fwd+bwd time
  tfla_check.py kernel OUT.json           kernel level: repeat calls on identical inputs, head dims 92/184 vs 64/128
  tfla_check.py loop BASE SPEED OUT.json [MICRO SEQ STEPS]   20-step AdamW loop with the rollout recipe;
                                         SPEED legacy (native, foreach AdamW) | opt (fused AdamW + TF32) | new (opt + TFLA)
Text for the batches: $FFT_CHECK_TEXT (plain text or a parquet with a `text` column).
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


TEXT = os.environ.get("FFT_CHECK_TEXT", "/scratch/lmbanr001/masters/sallm/results/fft_rollout_20260926/assets/t2x_train_validation_only/train.text")


def token_ids(tok):
    if TEXT.endswith(".parquet"):
        import pyarrow.parquet as pq
        text = "\n".join(pq.read_table(TEXT, columns=["text"]).column("text").to_pylist()[:2000])
    else:
        text = Path(TEXT).read_text()
    return tok(text[:2_000_000], return_tensors="pt").input_ids[0]


def loop(base: str, speed: str, out: str, micro: str = "16", seq: str = "256", steps: str = "20") -> None:
    """Plain AdamW loop with the rollout recipe (lr 1e-4, betas 0.9/0.95, eps 1e-8, wd 0.01, cosine, 10% warmup, clip 1.0,
    effective batch 16, fp32 master + bf16 autocast): legacy = native kernel, foreach AdamW, TF32 off; new = TFLA, fused, TF32."""
    import math

    import torch
    from transformers import AutoTokenizer, xLSTMForCausalLM, get_cosine_schedule_with_warmup

    from tfla import use_tfla

    micro, seq, steps = int(micro), int(seq), int(steps)
    torch.manual_seed(42)
    tok = AutoTokenizer.from_pretrained(base)
    ids = token_ids(tok)
    model = xLSTMForCausalLM.from_pretrained(base, torch_dtype=torch.float32).cuda().train()
    new = speed in ("new", "opt")  # opt = fused AdamW + TF32, native kernel
    if new:
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    if speed == "new":
        use_tfla(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01, fused=True if new else None)
    sched = get_cosine_schedule_with_warmup(opt, math.ceil(0.1 * steps), steps)
    accum = 16 // micro
    batches = ids[: steps * 16 * seq].view(steps * accum, micro, seq).cuda()
    losses, times = [], []
    for step in range(steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        tot = 0.0
        for a in range(accum):
            x = batches[step * accum + a]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=x, labels=x).loss / accum
            loss.backward()
            tot += float(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        losses.append(tot)
    res = {"speed": speed, "micro": micro, "seq": seq, "accum": accum, "steps": steps, "losses": losses,
           "step_s_median": statistics.median(times[3:]), "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
           "gpu": torch.cuda.get_device_name(0), "optimizer_defaults": {k: str(v) for k, v in opt.defaults.items()}}
    Path(out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res))


def kernel(out: str) -> None:
    """Kernel level, random inputs, head dims 92/184 (this xLSTM) and 64/128: TFLA (xl_chunk), limit_chunk and the
    custom-backward native kernel vs mlstm_kernels' native autograd kernel in fp32. Each Triton kernel is called
    three times on the same inputs (third after filling the allocator with NaN) to expose nondeterministic results."""
    import torch
    import torch.nn.functional as F

    import tfla

    sys.path.insert(0, tfla.PYDEPS)
    from mlstm_kernels.torch import get_mlstm_kernel

    K = {n: get_mlstm_kernel(f"chunkwise--{n}") for n in ("native_autograd", "native_custbw", "triton_xl_chunk", "triton_limit_chunk")}
    rows = []
    for dq, dv in [(92, 184), (64, 128)]:
        for B, S in [(16, 256), (16, 256), (16, 192), (4, 1024)]:
            g = torch.Generator(device="cuda").manual_seed(B * 7 + S)
            base = [torch.randn(B, 4, S, d, device="cuda", generator=g) * 0.5 for d in (dq, dq, dv)]
            base += [torch.randn(B, 4, S, device="cuda", generator=g), torch.randn(B, 4, S, device="cuda", generator=g) + 3]
            dh = torch.randn(B, 4, S, dv, device="cuda", generator=g)

            def run(fn, dtype, junk=False):
                if junk:
                    x = torch.full((1 << 28,), float("nan"), device="cuda")
                    del x
                ts = [t.detach().clone().to(dtype).requires_grad_() for t in base]
                h = fn(*ts, eps=1e-6, chunk_size=64, autocast_kernel_dtype=dtype)
                (h.float() * dh).sum().backward()
                return h.detach().float(), [t.grad.float().flatten() for t in ts]

            href, gref = run(K["native_autograd"], torch.float32)
            row = {"dq": dq, "dv": dv, "B": B, "S": S}
            calls = [("native_autograd_bf16", "native_autograd", torch.bfloat16, False), ("native_custbw_fp32", "native_custbw", torch.float32, False)]
            for name in ("triton_xl_chunk", "triton_limit_chunk"):
                calls += [(f"{name}_bf16_{j}", name, torch.bfloat16, j == 3) for j in (1, 2, 3)]
            for label, name, dt, junk in calls:
                h, gs = run(K[name], dt, junk)
                row[label] = {"h_rel": round(float((h - href).norm() / href.norm()), 5),
                              "grad_cos_qkvif": [round(float(F.cosine_similarity(a, b, dim=0)), 4) for a, b in zip(gs, gref)]}
            rows.append(row)
            print(json.dumps(row), flush=True)
    Path(out).write_text(json.dumps(rows, indent=1) + "\n")


def parity(base: str, out: str) -> None:
    import torch
    from transformers import AutoTokenizer, xLSTMForCausalLM

    from tfla import use_tfla

    tok = AutoTokenizer.from_pretrained(base)
    model = xLSTMForCausalLM.from_pretrained(base, torch_dtype=torch.float32).cuda().train()
    ids = token_ids(tok)
    shapes = [(16, 256), (4, 1024), (4, 2048)]  # SIB/NER-like micro 16; AfriHG/News micro 4

    def run(x):
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_ids=x, labels=x).loss
        loss.backward()
        g = torch.cat([p.grad.flatten().float() for p in model.parameters() if p.grad is not None])
        return float(loss), g

    def bench(x, n=10):
        for _ in range(3):
            run(x)
        ts = []
        for _ in range(n):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            run(x)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        return statistics.median(ts)

    batches = {f"{b}x{s}": ids[: b * s].view(b, s).cuda() for b, s in shapes}
    ref = {k: run(x) for k, x in batches.items()}
    t_native = {k: bench(x) for k, x in batches.items()}
    peak_native = torch.cuda.max_memory_allocated() / 2**30
    torch.cuda.reset_peak_memory_stats()
    cfg_before = {k: getattr(model.config, k) for k in ("chunkwise_kernel", "sequence_kernel", "step_kernel", "mode")}
    n = use_tfla(model)
    res = {"base": base, "backends_swapped": n, "config_after_swap": {k: getattr(model.config, k) for k in cfg_before},
           "config_unchanged": cfg_before == {k: getattr(model.config, k) for k in cfg_before}, "gpu": torch.cuda.get_device_name(0),
           "torch": torch.__version__, "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "batches": {}}
    for k, x in batches.items():
        loss, g = run(x)
        loss_r, g_r = ref[k]
        t = bench(x)
        res["batches"][k] = {"loss_native": loss_r, "loss_tfla": loss, "loss_rel_diff": abs(loss - loss_r) / abs(loss_r),
                             "gradnorm_native": float(g_r.norm()), "gradnorm_tfla": float(g.norm()),
                             "gradnorm_rel_diff": float((g.norm() - g_r.norm()).abs() / g_r.norm()),
                             "grad_cosine": float(torch.nn.functional.cosine_similarity(g, g_r, dim=0)),
                             "fwdbwd_s_native": round(t_native[k], 4), "fwdbwd_s_tfla": round(t, 4),
                             "speedup": round(t_native[k] / t, 2)}
    res["peak_gb_native"], res["peak_gb_tfla"] = round(peak_native, 2), round(torch.cuda.max_memory_allocated() / 2**30, 2)
    try:  # does TFLA itself still need chunk multiples? (the rollout pads to 64 either way)
        run(ids[:200].view(1, 200).cuda())
        res["tfla_accepts_len_200"] = True
    except Exception as exc:  # noqa: BLE001
        res["tfla_accepts_len_200"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    res["pass"] = all(b["loss_rel_diff"] <= 1e-3 for b in res["batches"].values()) and res["config_unchanged"]
    Path(out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"parity": parity, "kernel": kernel, "loop": loop}[sys.argv[1]](*sys.argv[2:])
