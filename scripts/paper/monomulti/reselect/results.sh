#!/bin/bash
# Rebuild $S/monomulti/reselect_results.csv from HEX selections + tests.
S=/private/tmp/claude-501/-Users-anrilombard-Desktop-sa-architecture-comparison-paper/8c159bcd-58cf-4f3d-afb4-5e7b76e82930/scratchpad/monomulti
scp -q $S/reselect/hex/collect.py hex:/scratch/lmbanr001/masters/sallm/results/monomulti_reselect_20260925/jobs/collect.py 2>/dev/null
# no head-node python (cluster rule): run the collector as a step inside one of my running allocations
ssh -o BatchMode=yes -o LogLevel=ERROR hex 'j=$(squeue -u lmbanr001 -h -t R -o "%i %j" | grep -E " mrs-" | head -1 | cut -d" " -f1); [ -n "$j" ] || exit 3; srun --jobid=$j --overlap --ntasks=1 env PYTHONDONTWRITEBYTECODE=1 /home/lmbanr001/masters/sallm/.venv/bin/python /scratch/lmbanr001/masters/sallm/results/monomulti_reselect_20260925/jobs/collect.py' 2>/dev/null > $S/reselect/collected.new && mv $S/reselect/collected.new $S/reselect/collected.jsonl
python3 - "$S" <<'PY'
import csv, json, sys
S = sys.argv[1]
old = list(csv.DictReader(open(f"{S}/rescore_results.csv")))
cols = list(old[0].keys()) + ["selected_epoch", "selected_step", "val_scores_by_epoch", "val_scores_by_epoch_lang", "val_source", "examples_seen", "examples_seen_method", "reselect_label", "variant"]
prev = {(r["model"], r["task"], r["language"], r["regime"]): r["score_points"] for r in old}
rows = []
for line in open(f"{S}/reselect/collected.jsonl"):
    j = json.loads(line)
    key = (j["model"], j["task"], j["language"], j["regime"])
    notes = [f"reselect: retrained with own recipe, every epoch saved, no early stopping; epoch picked on {j['val_source']} validation with the protocol scorer (ties->earlier); HEX {j['gpu']}"]
    if j["label"] == "mzansilm_ner_multi_monorecipe":
        notes.append("DEVIATION: MzansiLM Multi NER trained with the MzansiLM Mono NER recipe (r128+embed/lm_head, v5 template, 20 ep)")
    if j["label"] in ("mamba2_intent_mono_eng", "mzansilm_intent_mono_eng"):
        notes.append("DEVIATION: Feb recipe on the clean July eng split (Feb loader validated on test)")
    eb64 = j["label"] in ("mamba2_intent_mono_eng", "mamba2_intent_mono_sot", "mamba2_news_mono_eng")
    if eb64:
        notes.append("RULE: <10 optimizer steps/epoch in the original recipe -> trained at effective batch 64 (same lr, epochs, schedule, LoRA)")
    if j.get("merge_lora_override"):
        notes.append("merge_lora=true (modules_to_save adapter), as Phase 2")
    rows.append({
        "model": j["model"], "task": j["task"], "language": j["language"], "regime": j["regime"], "metric": j["metric"],
        "score_points": j["score_points"], "n_items": j["n_items"], "prompt": j["prompt"], "adapter_path": j["adapter_path"],
        "adapter_tree_sha256": j["adapter_tree_sha256"], "output_path": j["output_path"], "output_sha256": "",
        "distinct_predicted_labels": j["distinct_predicted_labels"], "pct_rows_loop": j["pct_rows_loop"], "pct_blank": j["pct_blank"],
        "top_prediction_share": j["top_prediction_share"], "previous_score_points": prev.get(key, ""), "xlstm_batch_size": "",
        "tree_matches_manifest": "", "notes": "; ".join(notes),
        "selected_epoch": j["selected_epoch"], "selected_step": j["selected_step"],
        "val_scores_by_epoch": json.dumps(j["val_scores_by_epoch"]), "val_scores_by_epoch_lang": json.dumps(j["val_scores_by_epoch_lang"]),
        "val_source": j["val_source"], "examples_seen": j["examples_seen"], "examples_seen_method": j["examples_seen_method"],
        "reselect_label": j["label"],
        "variant": "eff_batch_64 (<10 steps/epoch rule)" if eb64 else ("mzansilm_mono_ner_recipe (author decision 3)" if j["label"] == "mzansilm_ner_multi_monorecipe" else "own_recipe"),
    })
with open(f"{S}/reselect_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
print(len(rows), "result rows")
for r in rows:
    print(f"  {r['model']:8} {r['task']:6} {r['language']} {r['regime']:5} new={r['score_points']:.2f} old={r['previous_score_points']} epoch={r['selected_epoch']}")
PY
