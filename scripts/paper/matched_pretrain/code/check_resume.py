"""Verify resume/WSD test logs. Usage: python check_resume.py OUT"""
import json, math, sys
from pathlib import Path
o = Path(sys.argv[1])
p1 = [json.loads(l) for l in open(o / "phase1_train_log.jsonl")]
allm = [json.loads(l) for l in open(o / "main/train_log.jsonl")]
p2 = allm[len(p1):]
br = [json.loads(l) for l in open(o / "branch/train_log.jsonl")]
first = {r["step"]: r for r in p1}
print("phase1 last step", p1[-1]["step"], "| phase2 steps", p2[0]["step"], "->", p2[-1]["step"], "| branch", br[0]["step"], "->", br[-1]["step"])
ov = [(first[r["step"]], r) for r in p2 if r["step"] in first]
res = {"overlap_steps": [a["step"] for a, _ in ov],
       "data_ck_identical": all(a["data_ck"] == b["data_ck"] for a, b in ov),
       "lr_identical": all(a["lr"] == b["lr"] for a, b in ov),
       "loss_max_abs_diff": max(abs(a["loss"] - b["loss"]) for a, b in ov),
       "gradnorm_max_rel_diff": max(abs(a["grad_norm"] - b["grad_norm"]) / a["grad_norm"] for a, b in ov),
       "phase2_starts_at": p2[0]["step"], "main_ends_at": allm[-1]["step"],
       "stable_ckpt_exists": (o / "main/stable_step000122/state.pt").exists(),
       "branch_starts_at": br[0]["step"], "branch_ends_at": br[-1]["step"],
       "branch_continuous_with_main": br[0]["step"] == allm[-1]["step"] + 1}
# expected WSD lr: total 153, decay 31 steps from 122, floor 0.1*4e-4
exp = lambda s: 4e-5 + (4e-4 - 4e-5) * (1 - math.sqrt(min(1, (s - 122) / 31)))
res["branch_lr_max_abs_err"] = max(abs(r["lr"] - exp(r["step"] - 1)) for r in br)
res["stable_lr_constant_peak"] = all(abs(r["lr"] - 4e-4) < 1e-12 for r in allm if 21 <= r["step"] <= 122)
res["branch_final_lr"] = br[-1]["lr"]
ev = [json.loads(l) for l in open(o / "branch/event_log.jsonl") if '"eval"' in l]
res["branch_eval_ALL_ce"] = ev[-1]["ALL/ce"] if ev else None
res["final_weights"] = [str(p.name) for p in (o / "branch/weights").glob("*")]
print(json.dumps(res, indent=2))
(o / "resume_check.json").write_text(json.dumps(res, indent=2))
