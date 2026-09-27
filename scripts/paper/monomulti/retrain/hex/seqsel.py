#!/usr/bin/env python3
"""Validation checkpoint selection for NER/POS retrains with the General sequence runner.

  seqsel.py units LABEL [--top K]   write jobs/units_<LABEL>_val.tsv (+ protocols) for the epoch checkpoints
  seqsel.py select LABEL            read validation outputs, write selection/<LABEL>/SELECTION.json,
                                    build the test protocol and jobs/units_<LABEL>_test.tsv
  seqsel.py prune LABEL             delete non-selected checkpoints (keeps trainer_state_final.json)
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

R = Path("/scratch/lmbanr001/masters/sallm/results/monomulti_retrain_20260924")
PROMPT = {"ner": {"tsn": 2, "xho": 5, "zul": 5}, "pos": {"tsn": 3, "xho": 3, "zul": 3}}
NERKEY = {"tsn": "tn", "xho": "xh", "zul": "zu"}


def info(label):
    arch = label.split("_")[0]
    task = label.split("_")[1]
    langs = [label.split("_")[-1]] if "_mono_" in label else ["tsn", "xho", "zul"]
    return arch, task, langs


def ckpts(label):
    return sorted((R / "train" / label).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))


def protocol(unit, arch, task, adapter):
    p = R / "protocols" / f"{unit}.json"
    if not p.exists():
        subprocess.run([sys.executable, str(R / "jobs/make_protocol.py"), unit, arch, task, str(adapter)], check=True, capture_output=True)
    return p, json.loads(p.read_text())["models"][arch]["base_path"]


def line(unit, arch, task, adapter, out, phase, langs):
    p, base = protocol(unit, arch, task, adapter)
    runner = "bounded" if (task == "pos" and phase == "test" and len(langs) == 1) else "v2"
    poslang = langs[0] if runner == "bounded" else "-"
    batch1 = "1" if arch == "xlstm" else "0"
    vp = "protocol" if phase == "validation" else "-"
    sl = ",".join(langs) if phase == "validation" else "-"
    return "\t".join([unit, arch, task, runner, poslang, base, str(adapter), str(p), str(out), phase, batch1, vp, sl])


def val_score(path, task, langs):
    d = json.loads(Path(path).read_text())
    m = d["reported_metrics"]
    if task == "ner":
        vals = {l: m[f"sallm_masakhaner_{NERKEY[l]}_prompt_{PROMPT['ner'][l]}_val"] for l in langs}
    else:
        vals = {l: m[f"{l}/P3"]["token_accuracy"] for l in langs}
    return sum(vals.values()) / len(vals), vals


def in_training_best(label, langs):
    """Step with the highest mean in-training generation span F1 (training chat template, cycled prompts)."""
    best = None
    for f in (R / "train" / label / "debug_generation_examples").glob("*/step-*.jsonl"):
        per = {}
        for raw in f.read_text().splitlines():
            d = json.loads(raw)
            per[d["language"]] = d["metrics"].get(f"eval/{d['language']}_f1")
        vals = [v for v in per.values() if v is not None]
        if vals:
            step = int(f.name.split("-")[1].split(".")[0])
            score = sum(vals) / len(vals)
            if best is None or score > best[1]:
                best = (step, score)
    return None if best is None else best[0]


def main():
    cmd, label = sys.argv[1], sys.argv[2]
    arch, task, langs = info(label)
    sel = R / "selection" / label
    if cmd == "units":
        cs = ckpts(label)
        if "--top" in sys.argv:
            k = int(sys.argv[sys.argv.index("--top") + 1])
            arts = {}
            for f in (R / "train" / label / "validation_artifacts/pos").glob("step-*.json"):
                arts[int(f.name.split("-")[1].split(".")[0])] = json.loads(f.read_text())["all_token_accuracy"]
            keep = sorted(arts, key=lambda s: -arts[s])[:k]
            cs = [c for c in cs if int(c.name.split("-")[1]) in keep]
        lines = [line(f"{label}_{c.name}_val", arch, task, c, sel / f"{c.name}.validation.json", "validation", langs) for c in cs]
        out = R / "jobs" / f"units_{label}_val.tsv"
        out.write_text("\n".join(lines) + "\n")
        print(out, len(lines))
    elif cmd == "select":
        rows = []
        for f in sorted(sel.glob("checkpoint-*.validation.json")):
            step = int(f.name.split(".")[0].split("-")[1])
            mean, vals = val_score(f, task, langs)
            rows.append({"step": step, "mean": mean, "per_language": vals})
        rows.sort(key=lambda r: r["step"])
        best = max(rows, key=lambda r: (r["mean"], -r["step"]))
        adapter = R / "train" / label / f"checkpoint-{best['step']}"
        rec = {"label": label, "rule": f"max unweighted mean validation {'span F1' if task == 'ner' else 'token accuracy'} over {langs}, General {task.upper()} v2 runner at protocol prompts; earliest step on ties",
               "selected_step": best["step"], "selected_adapter": str(adapter), "selected_mean": best["mean"], "per_checkpoint": rows}
        (sel / "SELECTION.json").write_text(json.dumps(rec, indent=1))
        test_langs = langs if task == "pos" else ["tsn", "xho", "zul"]
        tl = line(label, arch, task, adapter, R / "test" / task / f"{label}.json", "test", test_langs)
        (R / "jobs" / f"units_{label}_test.tsv").write_text(tl + "\n")
        print(json.dumps({k: rec[k] for k in ("selected_step", "selected_mean")}), [(r["step"], round(r["mean"], 4)) for r in rows])
    elif cmd == "prune":
        keep = {json.loads((sel / "SELECTION.json").read_text())["selected_step"]}
        best = in_training_best(label, langs)
        if best is not None:
            pass  # prune rule: keep only the validation-selected checkpoint
        cs = ckpts(label)
        shutil.copy(cs[-1] / "trainer_state.json", R / "train" / label / "trainer_state_final.json")
        for c in cs:
            if int(c.name.split("-")[1]) not in keep:
                shutil.rmtree(c)
        fa = R / "train" / label / "final_adapter"
        if fa.exists():
            shutil.rmtree(fa)
        print("kept", keep)


if __name__ == "__main__":
    main()
