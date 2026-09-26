#!/bin/bash
# Rewrite $S/monomulti/reselect_scope.csv from HEX state.
S=/private/tmp/claude-501/-Users-anrilombard-Desktop-sa-architecture-comparison-paper/8c159bcd-58cf-4f3d-afb4-5e7b76e82930/scratchpad
state=$(ssh -o BatchMode=yes -o LogLevel=ERROR hex 'R=/scratch/lmbanr001/masters/sallm/results/monomulti_reselect_20260925; for d in $R/train/*/; do l=$(basename $d); f=0; [ -e $d/final_adapter/adapter_config.json ] && f=1; echo "T $l $f"; done; for f in $R/val/*/selection.json; do echo "S $(basename $(dirname $f))"; done; for f in $R/test/*.json; do echo "D $(basename $f .json)"; done; for f in $R/logs/train:*.log; do b=$(basename $f .log); b=${b#train:}; echo "L ${b%%.*} $(grep -c -E "Traceback|OutOfMemory" $f) $(stat -c %Y $f)"; done' 2>/dev/null)
[ -n "$state" ] || { echo "scope: HEX unreachable, file left unchanged"; exit 0; }
python3 - "$S/monomulti/reselect_scope.csv" <<PY
import sys, csv
state = """$state""".splitlines()
tr, sel, done, errs = {}, set(), set(), {}
for l in state:
    p = l.split()
    if len(p) < 2: continue
    if p[0] == "T": tr[p[1]] = p[2] == "1"
    elif p[0] == "S": sel.add(p[1])
    elif p[0] == "D": done.add(p[1])
    elif p[0] == "L" and len(p) >= 4:
        errs.setdefault(p[1], []).append((int(p[3]), int(p[2])))
A = {"gdn": "GDN", "xlstm": "xLSTM", "mamba2": "Mamba", "mzansilm": "MzansiLM"}
T = {"news": "News", "sib": "SIB-200", "intent": "Intent", "ner": "NER"}
L = {"news": ["eng", "xho"], "sib": ["afr", "eng", "nso", "sot", "xho", "zul"], "intent": ["eng", "sot", "xho", "zul"], "ner": ["tsn", "xho", "zul"]}
labels = []
labels += [f"gdn_news_mono_{x}" for x in L["news"]] + ["gdn_news_multi"] + [f"gdn_sib_mono_{x}" for x in L["sib"]] + ["gdn_sib_multi"] + [f"gdn_intent_mono_{x}" for x in L["intent"]] + ["gdn_intent_multi"]
labels += [f"xlstm_news_mono_{x}" for x in L["news"]] + ["xlstm_news_multi"] + [f"xlstm_intent_mono_{x}" for x in L["intent"]] + ["xlstm_intent_multi"]
labels += [f"mamba2_news_mono_{x}" for x in L["news"]] + ["mamba2_news_multi"] + [f"mamba2_intent_mono_{x}" for x in L["intent"]] + ["mamba2_intent_multi"]
labels += [f"mzansilm_intent_mono_{x}" for x in L["intent"]] + ["mzansilm_intent_multi", "mzansilm_ner_multi_monorecipe"]
rows = []
for lab in labels:
    arch, task, regime = lab.split("_")[:3]
    langs = [lab.split("_")[3]] if regime == "mono" else L[task]
    if lab in done: st = "done"
    elif lab in sel: st = "validating"
    elif lab in tr:
        last = max(errs.get(lab, [(0, 0)]))
        st = "validating" if tr[lab] else ("failed" if last[1] else "training")
    else: st = "queued"
    for g in langs:
        v = "eff_batch_64 (<10 steps/epoch rule)" if lab in ("mamba2_intent_mono_eng", "mamba2_intent_mono_sot", "mamba2_news_mono_eng") else ("mzansilm_mono_ner_recipe (author decision 3)" if lab == "mzansilm_ner_multi_monorecipe" else "own_recipe")
        rows.append([A[arch], T[task], g, regime.capitalize(), lab, st, v])
with open(sys.argv[1], "w", newline="") as f:
    w = csv.writer(f); w.writerow(["model", "task", "language", "regime", "adapter_id", "status", "variant"]); w.writerows(rows)
from collections import Counter
print(len(rows), "rows;", dict(Counter(r[5] for r in rows)), "| adapters", len(labels))
PY
