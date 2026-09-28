"""Output parity of every fast-stack change vs the baseline path, plus a micro-batch memory probe (1 GPU).

Same init (seed) and the same real batch (first blocks of the bench train.bin) for every variant.
  A baseline : eager, model logits -> fp32 CE          (reference)
  B flce     : eager, shared Liger fused-linear-CE
  C flce+compile
  D xLSTM only: native chunkwise vs --xlstm-kernel (logits, loss, grads), both baseline loss
  E fused AdamW vs foreach AdamW: one step from identical grads -> max |param diff|
  M memory/speed probe of the fast stack (eager FLCE) at several micro-batches
Usage: python parity.py --config cfg.json --data DIR --out out.json [--xlstm-kernel K]
"""
import argparse, copy, json, time
from pathlib import Path

import numpy as np
import torch

import pretrain as P

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--data", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--xlstm-kernel", default=None)
ap.add_argument("--mbs", default="8,12,16,24,48")
a = ap.parse_args()
cfg = json.loads(Path(a.config).read_text())
dev = torch.device("cuda")
data = P.Blocks(a.data, cfg["seq_len"], cfg["seed"])
x = data.get(0, 4).to(dev)
torch.manual_seed(cfg["seed"])
base = P.build_model(cfg).to(dev)
res = {"arch": cfg["arch"], "env": P.env_info(), "kernels": P.kernel_report(base, cfg["arch"]), "batch": list(x.shape)}


def run(model, fused, compile_=False, xin=x):
    m = copy.deepcopy(model)
    w = P.LMLoss(m, fused, compile_)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = w(xin)
    loss.backward()
    g = torch.cat([p.grad.float().flatten() for p in m.parameters() if p.grad is not None])
    return loss.item(), g, m


def cmp(name, ref, other):
    (l0, g0), (l1, g1) = ref, other
    res[name] = {"loss": l1, "loss_ref": l0, "loss_rel_diff": abs(l1 - l0) / abs(l0),
                 "gradnorm": g1.norm().item(), "gradnorm_ref": g0.norm().item(),
                 "gradnorm_rel_diff": abs(g1.norm().item() - g0.norm().item()) / g0.norm().item(),
                 "grad_cosine": torch.nn.functional.cosine_similarity(g0, g1, dim=0).item()}
    print(name, res[name], flush=True)


lA, gA, mA = run(base, False)
lB, gB, _ = run(base, True)
cmp("B_flce_vs_baseline", (lA, gA), (lB, gB))
try:
    lC, gC, _ = run(base, True, compile_=True)
    cmp("C_flce_compile_vs_baseline", (lA, gA), (lC, gC))
except Exception as e:  # noqa: BLE001
    res["C_flce_compile_vs_baseline"] = {"error": repr(e)[:2000]}
    print("compile failed", repr(e)[:500], flush=True)

if cfg["arch"] == "xlstm" and a.xlstm_kernel:
    try:
        cfg2 = dict(cfg, xlstm_chunkwise_kernel=a.xlstm_kernel)
        torch.manual_seed(cfg["seed"])
        fastk = P.build_model(cfg2).to(dev)
        fastk.load_state_dict(base.state_dict())
        res["kernels_fast"] = P.kernel_report(fastk, "xlstm")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            z0 = base(input_ids=x, use_cache=False).logits.float()
            z1 = fastk(input_ids=x, use_cache=False).logits.float()
        lD, gD, _ = run(fastk, False)
        cmp("D_xlstm_kernel_vs_native", (lA, gA), (lD, gD))
        res["D_xlstm_kernel_vs_native"].update(kernel=a.xlstm_kernel, logits_max_abs_diff=(z0 - z1).abs().max().item(),
                                               logits_mean_abs_diff=(z0 - z1).abs().mean().item(),
                                               logits_ref_mean_abs=z0.abs().mean().item())
        lE, gE, _ = run(fastk, True)
        cmp("D2_xlstm_kernel_flce_vs_baseline", (lA, gA), (lE, gE))
        del fastk
    except Exception as e:  # noqa: BLE001
        res["D_xlstm_kernel_vs_native"] = {"error": repr(e)[:2000], "kernel": a.xlstm_kernel}
        print("xlstm kernel failed", repr(e)[:500], flush=True)

# E: fused vs foreach AdamW, one step from identical grads
o = cfg["optim"]
diffs = []
ma, mb_ = copy.deepcopy(mA), copy.deepcopy(mA)
for mm, fused in ((ma, False), (mb_, True)):
    opt = torch.optim.AdamW(P.param_groups(mm, o["weight_decay"]), lr=o["peak_lr"], betas=tuple(o["betas"]), eps=o["eps"], fused=fused)
    opt.step()
res["E_fused_adamw_max_abs_param_diff"] = max((p.detach() - q.detach()).abs().max().item() for p, q in zip(ma.parameters(), mb_.parameters()))
res["E_adamw_update_scale"] = o["peak_lr"]
del ma, mb_, mA

# M: memory / speed probe, fast stack eager FLCE, plus baseline at mb 8
probe = []
for fused, mbs in ((False, [8]), (True, [int(v) for v in a.mbs.split(",")])):
    for mb in mbs:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        m = copy.deepcopy(base)
        w = P.LMLoss(m, fused)
        xb = data.get(0, mb).to(dev)
        try:
            ts = []
            for i in range(4):
                torch.cuda.synchronize(); t = time.time()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = w(xb)
                loss.backward()
                m.zero_grad(set_to_none=True)
                torch.cuda.synchronize(); ts.append(time.time() - t)
            probe.append({"fused": fused, "mb": mb, "peak_gb": torch.cuda.max_memory_allocated() / 2**30,
                          "tok_per_s_fwdbwd": mb * cfg["seq_len"] / float(np.median(ts[1:]))})
        except torch.OutOfMemoryError:
            probe.append({"fused": fused, "mb": mb, "oom": True})
        print(probe[-1], flush=True)
        del m, w
res["M_probe"] = probe
a.out.write_text(json.dumps(res, indent=2, default=str))
print("DONE", a.out)
