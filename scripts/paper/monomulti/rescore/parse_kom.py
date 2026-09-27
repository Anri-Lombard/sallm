import collections, hashlib, json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
D = Path("/scratch/alombard/sallm/results/monomulti_rescore_20260924")
INTENT = {"eng": 4, "sot": 1, "xho": 2, "zul": 2}
NEWSP = {"eng": "p2", "xho": "p4"}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def intent_rows(outdir, langs):
    res = outdir / "injongointent_all/results.json"; d = json.loads(res.read_text()); out = []
    marker = json.loads((outdir / "UNIT_DONE.json").read_text())
    for lang in langs:
        task = f"injongointent_{lang}_prompt_{INTENT[lang]}"
        samples = [s for s in d["samples"][task] if s.get("filter", "none") == "none"]
        pred = np.array([[r[0] for r in s["filtered_resps"]] for s in samples]).argmax(1)
        gold = np.array([s["f1"][0] for s in samples])
        assert (pred == np.array([s["f1"][1] for s in samples])).all()
        v = d["results"][task]["f1,none"]
        assert abs(f1_score(gold, pred, average="weighted") - v) < 1e-9
        out.append({"task": "Intent", "language": lang, "metric": "support_weighted_f1", "score_points": 100 * v, "n_items": len(samples),
                    "prompt": f"p{INTENT[lang]}", "distinct_predicted_labels": len(set(pred.tolist())),
                    "top_prediction_share": collections.Counter(pred.tolist()).most_common(1)[0][1] / len(samples),
                    "adapter_path": marker["unit"]["adapter"], "adapter_tree_sha256": marker["unit"]["adapter_tree_sha256"],
                    "output_path": f"kombuys:{res}", "output_sha256": sha(res)})
    return out
def cls_rows(path, task):
    d = json.loads(path.read_text()); h = sha(path); out = []
    for lang, m in d["languages"].items():
        rs = [r for r in d["rows"] if r["language"] == lang]
        preds = [r["prediction"] for r in rs]
        out.append({"task": task, "language": lang, "metric": "support_weighted_f1",
                    "score_points": 100 * (m["weighted_f1"] if task == "News" else m["f1"]), "n_items": len(rs),
                    "prompt": NEWSP[lang] if task == "News" else f"p{d['prompts'][lang]}", "distinct_predicted_labels": len(set(preds)),
                    "top_prediction_share": collections.Counter(preds).most_common(1)[0][1] / len(rs),
                    "adapter_path": d["adapter"], "adapter_tree_sha256": d["adapter_tree_sha256"], "output_path": f"kombuys:{path}", "output_sha256": h,
                    "chat_template_sha256": d.get("chat_template_sha256"), "runtime_seconds": d["runtime_seconds"]})
    return out
if __name__ == "__main__":
    units = json.loads((D / "scripts/units.json").read_text())
    rows = []
    for u in units:
        if u["host"] != "kombuys": continue
        try:
            if u["task"] == "intent":
                rs = intent_rows(D / f"raw/intent/{u['id']}-train", u["langs"])
            else:
                rs = cls_rows(D / f"raw/{u['task']}/{u['id']}.json", "News" if u["task"] == "news" else "SIB-200")
                assert {r["language"] for r in rs} == set(u["langs"])
            rows += [dict(r, arch=u["arch"], regime=u["regime"], unit=u["id"]) for r in rs]
        except FileNotFoundError as e:
            rows.append({"unit": u["id"], "missing": str(e)})
    for a in ("mzansilm", "mamba2", "xlstm", "gdn"):
        p = D / f"raw/news/general_{a}.json"
        if p.exists(): rows += [dict(r, arch=a, regime="General", unit=f"general_{a}") for r in cls_rows(p, "News")]
    v = D / "validation/intent_general_gdn-train"
    if (v / "UNIT_DONE.json").exists():
        rows += [dict(r, arch="gdn", regime="General", unit="VALIDATION_intent_general_gdn") for r in intent_rows(v, list(INTENT))]
    (D / "parsed_kombuys.json").write_text(json.dumps(rows, indent=1))
    for r in rows: print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k in ("unit", "language", "score_points", "n_items", "distinct_predicted_labels", "top_prediction_share", "missing")})
