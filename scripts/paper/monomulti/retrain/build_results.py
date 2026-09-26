#!/usr/bin/env python3
"""Fetch retrain test outputs and write monomulti/retrain_results.csv (same columns as rescore_results.csv + retrain columns)."""
import collections
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path

S = Path("/private/tmp/claude-501/-Users-anrilombard-Desktop-sa-architecture-comparison-paper/8c159bcd-58cf-4f3d-afb4-5e7b76e82930/scratchpad/monomulti")
LOCAL = S / "retrain/fetched"
HEX = "/scratch/lmbanr001/masters/sallm/results/monomulti_retrain_20260924"
KOM = "/scratch/alombard/sallm/results/monomulti_retrain_20260924"
SHOW = {"mzansilm": "MzansiLM", "mamba2": "Mamba", "xlstm": "xLSTM", "gdn": "GDN"}
NER = {"tsn": "sallm_masakhaner_tn_prompt_2_test", "xho": "sallm_masakhaner_xh_prompt_5_test", "zul": "sallm_masakhaner_zu_prompt_5_test"}
TASKK = {"NER": "ner", "POS": "pos", "SIB-200": "sib"}

# unit: (arch, task, regime, langs, host, output relative path, selection note, gpu note, xlstm batch)
UNITS = [
    ("xlstm", "SIB-200", "Multi", ["afr", "eng", "nso", "sot", "xho", "zul"], "kombuys", "test/sib/xlstm_sib_multi.json", "Kombuys GPU1 RTX 3080 Ti fp32", "1 (right-padded to chunk size)"),
    *[("mamba2", "SIB-200", "Mono", [l], "kombuys", f"test/sib/mamba2_sib_mono_{l}.json", "Kombuys GPU1 RTX 3080 Ti fp32", "") for l in ("nso", "sot", "xho", "zul")],
    *[("mamba2", "SIB-200", "Mono", [l], "kombuys", f"test/sib/mamba2_sib_mono_{l}_eb64.json", "Kombuys GPU1 RTX 3080 Ti fp32", "") for l in ("afr", "eng")],
    ("mamba2", "SIB-200", "Multi", ["afr", "eng", "nso", "sot", "xho", "zul"], "hex", "test/sib/mamba2_sib_multi.test.json", "HEX L40S fp32 (Kombuys GPUs down; sib scorer kit copied byte-identical)", ""),
    ("mzansilm", "POS", "Mono", ["tsn"], "hex", "test/pos/mzansilm_pos_mono_tsn.json", "HEX L40S bf16, bounded single-language runner", ""),
    ("xlstm", "NER", "Multi", ["tsn", "xho", "zul"], "hex", "test/ner/xlstm_ner_multi.json", "HEX L40S fp32 merge_lora, v2 runner", "1"),
    *[("mamba2", "NER", "Mono", [l], "hex", f"test/ner/mamba2_ner_mono_{l}.json", "HEX L40S fp32, EOS-fixed base, v2 runner", "") for l in ("tsn", "xho", "zul")],
    ("mamba2", "POS", "Multi", ["tsn", "xho", "zul"], "hex", "test/pos/mamba2_pos_multi.json", "HEX L40S fp32, v2 runner", ""),
]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def looped(text):
    segs = [s.strip() for s in re.split(r"\$\$|\n", text) if s.strip()]
    return bool(segs) and collections.Counter(segs).most_common(1)[0][1] >= 3


def fetch(host, rel):
    dest = LOCAL / host / rel
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        src = f"{'hex' if host == 'hex' else 'jbuys'}:{HEX if host == 'hex' else KOM}/{rel}"
        if subprocess.run(["scp", "-q", src, str(dest)], capture_output=True).returncode != 0:
            return None
    return dest


man = {(r["architecture"], r["task"], r["regime"], r["language"]): r for r in csv.DictReader(open(S / "manifest.csv"))}
sel = json.loads((S / "retrain/selection_summary.json").read_text())
out, missing = [], []
for arch, task, regime, langs, host, rel, gpu, xb in UNITS:
    path = fetch(host, rel)
    if path is None:
        missing.append(rel)
        continue
    d = json.loads(path.read_text())
    h = sha(path)
    unit = Path(rel).name.split(".")[0]
    s = sel.get(unit, {})
    for lang in langs:
        row = {"model": SHOW[arch], "task": task, "language": lang, "regime": regime}
        if task == "SIB-200":
            m = d["languages"][lang]
            rs = [r for r in d["rows"] if r["language"] == lang]
            preds = [r["prediction"] for r in rs]
            row.update(metric="support_weighted_f1", score_points=100 * m["f1"], n_items=len(rs), prompt=f"p{d['prompts'][lang]}",
                       distinct_predicted_labels=len(set(preds)), pct_rows_loop="", pct_blank="",
                       top_prediction_share=collections.Counter(preds).most_common(1)[0][1] / len(rs),
                       adapter_path=d["adapter"], adapter_tree_sha256=d["adapter_tree_sha256"])
        elif task == "NER":
            rs = [r for r in d["rows"] if r["task"] == NER[lang]]
            row.update(metric="entity_span_f1", score_points=100 * d["reported_metrics"][NER[lang]], n_items=len(rs), prompt="P" + NER[lang].split("_")[-2],
                       distinct_predicted_labels="", pct_rows_loop=100 * sum(looped(r["raw_response"]) for r in rs) / len(rs),
                       pct_blank=100 * sum(not r["prediction"].strip() for r in rs) / len(rs), top_prediction_share="",
                       adapter_path=d["binding"]["adapter_path"], adapter_tree_sha256=d["binding"]["adapter_tree_sha256"])
        else:
            m = d["reported_metrics"][f"{lang}/P3"]
            rs = [r for r in d["rows"] if r["language"] == lang]
            row.update(metric="token_accuracy", score_points=100 * m["token_accuracy"], n_items=m["total"], prompt="P3",
                       distinct_predicted_labels=len({t for r in rs for t in r["prediction"]}), pct_rows_loop="", pct_blank="", top_prediction_share="",
                       adapter_path=d["binding"]["adapter_path"], adapter_tree_sha256=d["binding"]["adapter_tree_sha256"])
        mrow = man[(arch, TASKK[task], regime, lang)]
        prev = round(100 * float(mrow["sheet_score"]), 2) if mrow["sheet_score"] else ""
        adj = d.get("wrapper_adjustments")
        notes = [gpu, f"retrained ({mrow['status']} in Phase 1 manifest)"]
        if adj:
            notes.append(f"runner wrapper adjustments {json.dumps(adj, sort_keys=True)}")
        notes += s.get("notes", [])
        row["variant"] = ("eff_batch_64 (approved Mamba Mono NER deviation; frozen recipe had 12 steps/epoch)" if unit == "mamba2_ner_mono_zul"
                          else "eff_batch_64 (<10 steps/epoch rule)" if (unit.endswith("eb64") or unit.startswith("mamba2_ner_mono")) else "frozen_recipe")
        row.update(output_path=f"{host}:{(HEX if host == 'hex' else KOM)}/{rel}", output_sha256=h,
                   previous_score_points=prev, xlstm_batch_size=xb, tree_matches_manifest="n/a (new adapter)",
                   selected_checkpoint=s.get("selected_checkpoint", ""), selection_rule=s.get("rule", ""),
                   selection_validation_score=s.get("validation_score", ""), notes="; ".join(notes))
        for k in ("score_points", "top_prediction_share", "pct_rows_loop", "pct_blank"):
            if isinstance(row[k], float):
                row[k] = f"{row[k]:.4f}" if k in ("score_points", "top_prediction_share") else f"{row[k]:.2f}"
        out.append(row)

order = {"MzansiLM": 0, "Mamba": 1, "xLSTM": 2, "GDN": 3}
tord = ["News", "SIB-200", "Intent", "NER", "POS"]
out.sort(key=lambda x: (tord.index(x["task"]), order[x["model"]], x["regime"], x["language"], x["variant"]))
with open(S / "retrain_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)
print(len(out), "rows; missing:", missing)
for r in out:
    print(r["model"], r["task"], r["regime"], r["language"], r["score_points"], "prev", r["previous_score_points"], r["selected_checkpoint"])
