import csv
import json
import re
import sys

SHOW = {"mzansilm": "Transformer", "mamba2": "Mamba", "xlstm": "xLSTM", "gdn": "GDN"}
PROTOCOL = {"injongointent_eng_prompt_4", "injongointent_sot_prompt_1", "injongointent_xho_prompt_2", "injongointent_zul_prompt_2"} | {
    f"belebele_{lang}_prompt_1" for lang in ("afr", "eng", "sot", "ssw", "tsn", "tso", "xho", "zul")}
RESULTS_ROOT = sys.argv[3]
summary = json.load(open(sys.argv[1]))
rows = []
for run, value in summary.items():
    marker = value["marker"]
    unit = marker["unit"]
    for task, r in value["tasks"].items():
        m = re.fullmatch(r"(injongointent|belebele)_(\w{3})_prompt_(\d)", task)
        rows.append({
            "model": SHOW[unit["architecture"]] if not unit["unit_id"].startswith("control") else unit["unit_id"],
            "task": "Intent" if m[1] == "injongointent" else "Belebele",
            "language": m[2],
            "metric": r["metric"],
            "score_points": f"{r['score_points']:.4f}",
            "n_items": r["n_items"],
            "prompt_task_name": task,
            "protocol_prompt": task in PROTOCOL,
            "mode": marker["mode"],
            "prefix_used": json.dumps("<|assistant|>" + marker["context_suffix_after_assistant_tag"]),
            "distinct_predicted_labels": r["distinct_pred"],
            "top_prediction": r["top_pred"],
            "top_prediction_share": f"{r['top_pred_share']:.4f}",
            "acc_points": f"{r['acc_points']:.4f}",
            "adapter_tree_sha256": unit["adapter_tree_sha256"],
            "adapter_path": unit["adapter"],
            "base_tree_sha256": unit["base_tree_sha256"],
            "base_path": unit["base"],
            "kombuys_output": f"{RESULTS_ROOT}/{run}",
        })
order = {"Transformer": 0, "Mamba": 1, "xLSTM": 2, "GDN": 3}
rows.sort(key=lambda r: (order.get(r["model"], 9), r["task"] != "Intent", r["language"], r["prompt_task_name"], r["mode"]))
with open(sys.argv[2], "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(len(rows), "rows")
