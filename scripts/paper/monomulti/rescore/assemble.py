import csv, json, sys
S = "/private/tmp/claude-501/-Users-anrilombard-Desktop-sa-architecture-comparison-paper/8c159bcd-58cf-4f3d-afb4-5e7b76e82930/scratchpad/monomulti"
L = "/private/tmp/claude-501/-Users-anrilombard-Desktop-sa-architecture-comparison-paper/8c159bcd-58cf-4f3d-afb4-5e7b76e82930/scratchpad/p2"
SHOW = {"mzansilm": "MzansiLM", "mamba2": "Mamba", "xlstm": "xLSTM", "gdn": "GDN"}
TASKK = {"News": "news", "SIB-200": "sib", "Intent": "intent", "NER": "ner", "POS": "pos"}
OLD_GENERAL_NEWS = {("mzansilm", "eng"): 81.62, ("mzansilm", "xho"): 91.48, ("mamba2", "eng"): 86.80, ("mamba2", "xho"): 94.31,
                    ("xlstm", "eng"): 84.80, ("xlstm", "xho"): 87.87, ("gdn", "eng"): 89.63, ("gdn", "xho"): 96.22}
BOS_TEMPLATE = "fb0b8a9e"  # filled below from outputs
man = {(r["architecture"], r["task"], r["regime"], r["language"]): r for r in csv.DictReader(open(f"{S}/manifest.csv"))}
rows = json.load(open(f"{L}/parsed_hex.json")) + json.load(open(f"{L}/parsed_kombuys.json"))
out, missing = [], []
for r in rows:
    if "validation" in r or str(r.get("unit", "")).startswith("VALIDATION"): continue
    if "missing" in r: missing.append(r); continue
    arch, task = r["arch"], r["task"]
    notes = []
    if r["regime"] == "General":
        prev = OLD_GENERAL_NEWS[(arch, r["language"])]; tree_ok = ""
        notes.append(f"General News rescored with training chat template (sha {r['chat_template_sha256'][:8]}); paper v10 value in previous_score_points")
    else:
        m = man[(arch, TASKK[task], r["regime"], r["language"])]
        prev = round(100 * float(m["sheet_score"]), 2) if m["sheet_score"] else ""
        tree_ok = r["adapter_tree_sha256"] in [t.strip() for t in m["tree_sha256"].split("|")]
    if "reused_unit" in r: notes.append(f"reused full-matrix official unit {r['reused_unit']} (General protocol), not rerun")
    if task == "News": notes.append(f"Kombuys GPU1 RTX 3080 Ti bf16; chat_template {r['chat_template_sha256'][:8]}")
    if task == "SIB-200": notes.append("Kombuys GPU1 RTX 3080 Ti fp32")
    if task == "Intent": notes.append("Kombuys lm-eval loglik, --mode train, injongointent_all")
    if task == "POS" and "reused_unit" not in r: notes.append("HEX A100 (matches General POS GPU)" if arch == "mzansilm" or r.get("gpu") == "A100" else ("HEX L40S (fp32; General Mamba POS xho on L40S: same score, 2/601 rows differ; A100 rerun of tsn differed by 0.006 pts)" if arch == "mamba2" else "HEX L40S (fp32; General xLSTM POS xho reproduced exactly on L40S)"))
    if "l40s_score_points" in r: notes.append(f"L40S run gave {r['l40s_score_points']:.4f}")
    if task == "NER" and "reused_unit" not in r: notes.append("HEX L40S (as General NER v2)")
    if task == "NER" and arch == "mzansilm" and r["regime"] == "Mono": notes.append("merge_lora=true (modules_to_save embed/lm_head adapter cannot load unmerged in lm-eval)")
    xb = ""
    if arch == "xlstm":
        xb = {"News": "1 (singleton scorer)", "SIB-200": "1 (right-padded to chunk size)", "Intent": "lm-eval auto:4 max 64, right-padded loglik (pads after the scored tokens)",
              "NER": "1", "POS": "17 label continuations per step, right-padded"}[task]
    if "auto4_score_points" in r: notes.append(f"lm-eval batch auto:4 (left-padded, xLSTM pad leak) gave {r['auto4_score_points']:.4f}")
    out.append({"model": SHOW[arch], "task": task, "language": r["language"], "regime": r["regime"], "metric": r["metric"],
                "score_points": f"{r['score_points']:.4f}", "n_items": r["n_items"], "prompt": r["prompt"],
                "adapter_path": r["adapter_path"], "adapter_tree_sha256": r["adapter_tree_sha256"], "output_path": r["output_path"], "output_sha256": r["output_sha256"],
                "distinct_predicted_labels": r.get("distinct_predicted_labels", ""),
                "pct_rows_loop": f"{r['pct_rows_loop']:.2f}" if r.get("pct_rows_loop") not in ("", None) else "",
                "pct_blank": f"{r['pct_blank']:.2f}" if r.get("pct_blank") not in ("", None) else "",
                "top_prediction_share": f"{r['top_prediction_share']:.4f}" if "top_prediction_share" in r else "",
                "previous_score_points": prev, "xlstm_batch_size": xb, "tree_matches_manifest": tree_ok, "notes": "; ".join(notes)})
order = {"MzansiLM": 0, "Mamba": 1, "xLSTM": 2, "GDN": 3}; tord = list(TASKK); rord = {"General": 0, "Mono": 1, "Multi": 2}
out.sort(key=lambda x: (tord.index(x["task"]), order[x["model"]], rord[x["regime"]], x["language"]))
with open(f"{S}/rescore_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
print(len(out), "rows;", len(missing), "missing units:", [m.get("unit") or m.get("missing") for m in missing])
