import collections, hashlib, json, re
from pathlib import Path
R = Path("/scratch/lmbanr001/masters/sallm/results")
M = R / "monomulti_rescore_20260924"
NER = {"tsn": "sallm_masakhaner_tn_prompt_2_test", "xho": "sallm_masakhaner_xh_prompt_5_test", "zul": "sallm_masakhaner_zu_prompt_5_test"}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def looped(text):
    segs = [s.strip() for s in re.split(r"\$\$|\n", text) if s.strip()]
    return bool(segs) and collections.Counter(segs).most_common(1)[0][1] >= 3
def rows_for(path, arch, task, regime, langs, prompt_note=None):
    d = json.loads(path.read_text()); h = sha(path); b = d["binding"]; out = []
    for lang in langs:
        if task == "ner":
            rs = [r for r in d["rows"] if r["task"] == NER[lang]]
            out.append({"arch": arch, "task": "NER", "language": lang, "regime": regime, "metric": "entity_span_f1",
                        "score_points": 100 * d["reported_metrics"][NER[lang]], "n_items": len(rs), "prompt": NER[lang].split("_")[-2].join(["P", ""]) if False else "P" + NER[lang].split("_")[-2],
                        "pct_rows_loop": 100 * sum(looped(r["raw_response"]) for r in rs) / len(rs),
                        "pct_blank": 100 * sum(not r["prediction"].strip() for r in rs) / len(rs),
                        "distinct_predicted_labels": ""})
        else:
            m = d["reported_metrics"][f"{lang}/P3"]
            rs = [r for r in d["rows"] if r["language"] == lang]
            out.append({"arch": arch, "task": "POS", "language": lang, "regime": regime, "metric": "token_accuracy",
                        "score_points": 100 * m["token_accuracy"], "n_items": m["total"], "prompt": "P3",
                        "distinct_predicted_labels": len({t for r in rs for t in r["prediction"]}), "pct_rows_loop": "", "pct_blank": ""})
        out[-1].update({"adapter_path": b["adapter_path"], "adapter_tree_sha256": b["adapter_tree_sha256"], "output_path": f"hex:{path}", "output_sha256": h,
                        "elapsed_seconds": d["elapsed_seconds"]})
    return out
rows = []
for u in json.loads((M / "hex_units.json").read_text()):
    p = M / "raw" / u["task"] / f"{u['id']}.json"
    a100 = M / "raw/pos_a100" / f"{u['id']}.json"
    bs1 = M / "raw/ner_bs1" / f"{u['id']}.json"
    if bs1.exists():
        auto = {r["language"]: r["score_points"] for r in rows_for(p, u["arch"], u["task"], u["regime"], u["langs"])}
        rows += [dict(r, batch="1", auto4_score_points=auto[r["language"]]) for r in rows_for(bs1, u["arch"], u["task"], u["regime"], u["langs"])]
    elif u["arch"] == "xlstm" and u["task"] == "ner":
        rows.append({"arch": u["arch"], "task": "NER", "missing": str(bs1), "langs": u["langs"], "regime": u["regime"]})
    elif a100.exists():
        l40 = {r["language"]: r["score_points"] for r in rows_for(p, u["arch"], u["task"], u["regime"], u["langs"])}
        rows += [dict(r, gpu="A100", l40s_score_points=l40[r["language"]]) for r in rows_for(a100, u["arch"], u["task"], u["regime"], u["langs"])]
    elif p.exists():
        rows += rows_for(p, u["arch"], u["task"], u["regime"], u["langs"])
    else:
        rows.append({"arch": u["arch"], "task": u["task"].upper(), "missing": str(p), "langs": u["langs"], "regime": u["regime"]})
F = R / "full_matrix_execution_20260916_v1/official/sequence"
for unit, task, regime, langs in ((22, "ner", "Mono", ["tsn"]), (23, "ner", "Mono", ["xho"]), (24, "ner", "Mono", ["zul"]), (78, "ner", "Multi", ["tsn", "xho", "zul"]),
                                  (27, "pos", "Mono", ["tsn"]), (28, "pos", "Mono", ["xho"]), (29, "pos", "Mono", ["zul"]), (80, "pos", "Multi", ["tsn", "xho", "zul"])):
    rows += [dict(r, reused_unit=unit) for r in rows_for(F / f"{unit}.json", "gdn", task, regime, langs)]
for f in sorted((M / "validation").glob("*.json")):
    d = json.loads(f.read_text())
    rows.append({"validation": f.name, "metrics": d["reported_metrics"], "elapsed_seconds": d["elapsed_seconds"]})
(M / "parsed_hex.json").write_text(json.dumps(rows, indent=1))
for r in rows: print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k not in ("adapter_path", "output_path", "output_sha256", "adapter_tree_sha256")})
