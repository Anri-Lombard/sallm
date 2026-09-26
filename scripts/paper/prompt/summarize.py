import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

root = Path(sys.argv[1])
out = {}
for run in sorted(p for p in root.iterdir() if (p / "UNIT_DONE.json").exists() and "smoke" not in p.name):
    marker = json.loads((run / "UNIT_DONE.json").read_text())
    rows = {}
    for pack in marker["packs"]:
        d = json.loads((run / pack / "results.json").read_text())
        for task, samples in d["samples"].items():
            choices = d["configs"][task]["doc_to_choice"]
            samples = [s for s in samples if s.get("filter", "none") == "none"]
            lls = np.array([[r[0] for r in s["filtered_resps"]] for s in samples])
            pred = lls.argmax(1)
            res = d["results"][task]
            if task.startswith("injongointent"):
                gold = np.array([s["f1"][0] for s in samples])
                assert (pred == np.array([s["f1"][1] for s in samples])).all(), task
                metric, value = "support_weighted_f1", res["f1,none"]
                assert abs(100 * f1_score(gold, pred, average="weighted") - 100 * value) < 1e-6, task
            else:
                gold = np.array([s["target"] for s in samples])
                metric, value = "acc_norm", res["acc_norm,none"]
                assert abs((pred == gold).mean() - value) < 1e-9, task
            counts = Counter(pred.tolist())
            rows[task] = {
                "metric": metric,
                "score_points": 100 * value,
                "acc_points": 100 * res["acc,none"],
                "n_items": len(samples),
                "distinct_pred": len(counts),
                "top_pred": choices[counts.most_common(1)[0][0]],
                "top_pred_share": counts.most_common(1)[0][1] / len(samples),
                "distinct_gold": len(set(gold.tolist())),
                "context_tail": samples[0]["arguments"][0][0][-30:],
            }
    out[run.name] = {"marker": marker, "tasks": rows}
json.dump(out, open(sys.argv[2], "w"), indent=1)
print("\n".join(f"{k}: {len(v['tasks'])} tasks" for k, v in out.items()))
