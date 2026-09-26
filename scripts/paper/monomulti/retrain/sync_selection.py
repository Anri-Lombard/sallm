#!/usr/bin/env python3
"""Add selection_summary.json entries for finished Phase 3 units from their SELECTION.json (HEX or Kombuys), then the caller runs build_results.py."""
import json, subprocess
from pathlib import Path
D = Path(__file__).parent
HEX, KOM = "/scratch/lmbanr001/masters/sallm/results/monomulti_retrain_20260924", "/scratch/alombard/sallm/results/monomulti_retrain_20260924"
UNITS = {"mamba2_sib_mono_afr_eb64": ("jbuys", KOM, "max validation support-weighted F1, General SIB scorer, protocol prompt; all epochs scored"),
         "mamba2_sib_multi": ("hex", HEX, "max unweighted mean over 6 langs of validation support-weighted F1, General SIB scorer, protocol prompts; all epochs scored"),
         **{f"mamba2_ner_mono_{l}": ("hex", HEX, "max validation span F1, General NER v2 runner at the protocol prompt, EOS-fixed base; all epochs scored") for l in ("tsn", "xho", "zul")},
         "mamba2_pos_multi": ("hex", HEX, "max unweighted mean over 3 langs of validation token accuracy, General POS v2 runner at P3; all epochs scored")}
NOTES = {"mamba2_sib_mono_afr_eb64": ["<10 steps/epoch rule: frozen mamba_sib_afr recipe (eff 128, 6 steps/epoch) rerun at effective batch 64; frozen-recipe result 50.33 superseded"],
         "mamba2_ner_mono_tsn": ["<10 steps/epoch rule: frozen recipe eff 256 (6 steps/epoch) trained at effective batch 64"],
         "mamba2_ner_mono_xho": ["<10 steps/epoch rule: frozen recipe eff 256 (6 steps/epoch) trained at effective batch 64"],
         "mamba2_ner_mono_zul": ["frozen recipe eff 128 (12 steps/epoch); trained at effective batch 64 under the approved Mamba Mono NER deviation"]}
p = D / "selection_summary.json"
summ = json.loads(p.read_text())
for unit, (host, root, rule) in UNITS.items():
    if unit in summ:
        continue
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "LogLevel=ERROR", host, f"cat {root}/selection/{unit}/SELECTION.json"], capture_output=True, text=True)
    if r.returncode or not r.stdout.strip().startswith("{"):
        continue
    s = json.loads(r.stdout[r.stdout.index("{"):])
    steps = sorted(c["step"] for c in s["per_checkpoint"])
    per_ep = steps[0]
    n_ep = len(steps)
    if steps != [per_ep * (i + 1) for i in range(n_ep)]:
        print("WARN: not every epoch scored", unit, steps)
    mean = s.get("selected_mean", s.get("selected_mean_f1"))
    summ[unit] = {"selected_checkpoint": f"checkpoint-{s['selected_step']} (epoch {s['selected_step'] // per_ep} of {n_ep})",
                  "rule": rule, "validation_score": f"{100 * mean:.2f}", "notes": NOTES.get(unit, [])}
    print("added", unit, summ[unit]["selected_checkpoint"], summ[unit]["validation_score"])
p.write_text(json.dumps(summ, indent=1))
