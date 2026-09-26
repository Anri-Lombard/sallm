"""Collect selections + test scores into reselect_results rows (JSON lines to stdout)."""
import glob, json, os, re, yaml
R = "/scratch/lmbanr001/masters/sallm/results/monomulti_reselect_20260925"
ARCH = {"gdn": "GDN", "xlstm": "xLSTM", "mamba2": "Mamba", "mzansilm": "MzansiLM"}
TASK = {"news": "News", "sib": "SIB-200", "intent": "Intent", "ner": "NER"}
METRIC = {"news": "support_weighted_f1", "sib": "support_weighted_f1", "intent": "support_weighted_f1", "ner": "entity_span_f1"}


def train_rows(label):
    for log in sorted(glob.glob(f"{R}/logs/train:{label}.*.log"), key=os.path.getmtime, reverse=True):
        txt = open(log, errors="ignore").read()
        m = re.search(r"Samples: train=(\d+)", txt) or re.search(r"train=(\d+),? val", txt)
        if m:
            return int(m.group(1)), log
    return None, None


def eff_batch(label):
    p = f"{R}/train/{label}/resolved_config.yaml"
    if not os.path.exists(p):
        return None
    try:
        cfg = yaml.safe_load(open(p))
        t = (cfg.get("finetune") or cfg)["training"]
        return int(t["per_device_train_batch_size"]) * int(t.get("gradient_accumulation_steps") or 1)
    except Exception:
        return None


for sel_path in sorted(glob.glob(f"{R}/val/*/selection.json")):
    label = sel_path.split("/")[-2]
    test_path = f"{R}/test/{label}.json"
    if not os.path.exists(test_path):
        continue
    sel, test = json.load(open(sel_path)), json.load(open(test_path))
    arch, task, regime = label.split("_")[:3]
    s = sel["selected"]
    n_rows, _ = train_rows(label)
    eb = eff_batch(label)
    epoch = s.get("epoch")
    if n_rows and epoch:
        seen, how = int(round(n_rows * float(epoch))), f"train_rows({n_rows})*epoch"
    elif eb:
        seen, how = s["step"] * eb, f"step*eff_batch({eb})"
    else:
        seen, how = None, "unknown"
    curve = {f"e{int(e['epoch']) if e.get('epoch') else e['checkpoint']}": round(e["score"], 4) for e in sel["scores_by_epoch"]}
    curve_lang = {f"e{int(e['epoch']) if e.get('epoch') else e['checkpoint']}": {k: round(v, 4) for k, v in e["per_lang"].items()} for e in sel["scores_by_epoch"]}
    for lang, r in test["languages"].items():
        print(json.dumps({
            "label": label, "model": ARCH[arch], "task": TASK[task], "language": lang, "regime": regime.capitalize(),
            "metric": METRIC[task], "score_points": round(r["score"], 4), "n_items": r.get("n_items"), "prompt": r.get("prompt"),
            "adapter_path": "hex:" + test["adapter"], "adapter_tree_sha256": test.get("adapter_tree_sha256"),
            "output_path": "hex:" + (test.get("raw_output") or test_path), "test_json": "hex:" + test_path,
            "distinct_predicted_labels": r.get("distinct_predicted_labels"), "pct_rows_loop": r.get("pct_rows_loop"),
            "pct_blank": r.get("pct_blank"), "top_prediction_share": r.get("top_prediction_share"),
            "selected_epoch": epoch, "selected_step": s["step"], "n_epochs_scored": len(sel["scores_by_epoch"]),
            "val_scores_by_epoch": curve, "val_scores_by_epoch_lang": curve_lang, "val_source": sel.get("val_source"),
            "examples_seen": seen, "examples_seen_method": how, "gpu": test.get("gpu"), "merge_lora_override": test.get("merge_lora_override"),
        }))
