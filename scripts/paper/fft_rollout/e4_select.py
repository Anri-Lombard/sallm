"""Epoch-4 fine-tuning LR re-selection with exactly the epoch-1 rule (rollout.py do_select / next_edge_lr / transferred_lr),
pooled over the epoch-4 run dirs (<arch>_e4, _e4_lowlr, _e4_lrcheck, _e4_sweepA/B; each trains one fixed rate per family).
Base grid 3e-5/1e-4/3e-4 on Multi validation (T2X: isiXhosa Mono); ties -> smaller LR; while the best is at a grid edge add
the next x3 ladder point beyond it (at most 2). Rates trained outside the grid count only if the edge rule asks for them.
Multitask: most frequent selected rate over news/sib/intent/ner/pos/afrihg, ties -> lower. Prints JSON (--json) or a table.
"""
import json, subprocess, sys
from collections import Counter

LRS = ("3e-5", "1e-4", "3e-4")
LADDER = ("3e-6", "1e-5", "3e-5", "1e-4", "3e-4", "1e-3", "3e-3")
MAX_EXT = 2
FAMS = ("news", "sib", "intent", "ner", "pos", "afrihg", "nchlt_ner", "t2x")
MT_FAMS = ("news", "sib", "intent", "ner", "pos", "afrihg")
ARCHS = ("mzansilm", "mamba2", "xlstm", "gdn")


def next_edge_lr(grid, best_lr, n_ext):  # copied from rollout.py
    if n_ext >= MAX_EXT or len(grid) < 2 or best_lr not in (grid[0], grid[-1]):
        return None
    i = LADDER.index(best_lr) + (1 if best_lr == grid[-1] else -1)
    return LADDER[i] if 0 <= i < len(LADDER) and LADDER[i] not in grid else None


def select(val):  # val: {lr: best validation score}
    grid = sorted(LRS, key=float)
    if any(lr not in val for lr in grid):
        return {"state": "base", "missing": [lr for lr in grid if lr not in val]}
    ext = []
    while True:
        best = max(grid, key=lambda lr: (val[lr], -float(lr)))
        nxt = next_edge_lr(grid, best, len(ext))
        if nxt is None:
            return {"state": "final", "lr": best, "val": val[best], "grid": grid}
        if nxt not in val:
            return {"state": "extend", "need": nxt, "best_so_far": best, "grid": grid}
        ext.append(nxt)
        grid = sorted(grid + [nxt], key=float)


REMOTE = r'''F=/scratch/lmbanr001/masters/sallm/results/fft_rollout_20260926/runs
for r in $(ls $F | grep -E "^(mzansilm|mamba2|xlstm|gdn)_e4"); do for d in $F/$r/runs/*-multi-lr*-s42 $F/$r/runs/t2x-mono-xho-lr*-s42; do
  [ -f $d/RUN_DONE.json ] && jq -c --arg r $r --arg n $(basename $d) '{dir:$r, run:$n, lr, val:.best_val}' $d/RUN_DONE.json; done
  [ -f $F/$r/keep/general/LR_TRANSFER.json ] && jq -c --arg r $r '{dir:$r, mt_lr:.lr}' $F/$r/keep/general/LR_TRANSFER.json; done'''
rows = [json.loads(l) for l in subprocess.run(["ssh", "-o", "BatchMode=yes", "-q", "hex", REMOTE], capture_output=True, text=True).stdout.splitlines() if l.startswith("{")]
val, mt_used, reused = {}, {}, {}
for r in rows:
    a = r["dir"].split("_e4")[0]
    if "mt_lr" in r:
        if r["dir"] == f"{a}_e4":
            mt_used[a] = r["mt_lr"]
        continue
    fam = r["run"].split("-")[0]
    val.setdefault((a, fam), {})[r["lr"]] = r["val"]
    if r["dir"] == f"{a}_e4":
        reused[(a, fam)] = r["lr"]
out = {}
for a in ARCHS:
    o = out[a] = {}
    for f in FAMS:
        s = select(val.get((a, f), {}))
        s["reused"] = reused.get((a, f))
        s["mono_rerun"] = s["state"] == "final" and s["reused"] is not None and s["lr"] != s["reused"]
        o[f] = s
    if all(o[f]["state"] == "final" for f in MT_FAMS):
        c = Counter(o[f]["lr"] for f in MT_FAMS)
        lr = max(c, key=lambda x: (c[x], -float(x)))
        o["general"] = {"state": "final", "lr": lr, "used": mt_used.get(a), "retrain": mt_used.get(a) not in (None, lr)}
    else:
        o["general"] = {"state": "waiting", "used": mt_used.get(a)}
if "--json" in sys.argv:
    print(json.dumps(out))
elif "--summary" in sys.argv:  # for the live pane
    fam = [(a, f, s) for a, o in out.items() for f, s in o.items() if f != "general"]
    print(json.dumps({"final": sum(s["state"] == "final" for _, _, s in fam), "total": len(fam),
                      "extend": [f"{a} {f} {s['need']}" for a, f, s in fam if s["state"] == "extend"],
                      "mono": [f"{a} {f} {s['lr']}" for a, f, s in fam if s.get("mono_rerun")],
                      "mt": [f"{a} {o['general']['lr']}" for a, o in out.items() if o["general"].get("retrain")]}))
else:
    for a in ARCHS:
        for f, s in out[a].items():
            print(f"{a:9} {f:10} {s['state']:7} " + {"base": lambda: f"missing {s['missing']}", "extend": lambda: f"needs {s['need']} (best so far {s['best_so_far']})",
                  "final": lambda: f"lr {s['lr']}" + (f" val {s['val']:.1f} reused {s['reused']}" + (" -> MONO RERUN" if s['mono_rerun'] else "") if f != "general" else f" used {s['used']}" + (" -> RETRAIN" if s['retrain'] else "")),
                  "waiting": lambda: f"used {s['used']}"}[s["state"]]())
