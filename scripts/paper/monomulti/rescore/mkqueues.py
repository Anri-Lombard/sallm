import json
D = "/scratch/alombard/sallm/results/monomulti_rescore_20260924"
PY = "/scratch/alombard/sallm/.venv/bin/python"
V15 = "/scratch/alombard/sallm_snapshots/news-general-corrected-official-20260914-v15/src/main"
SIBRT = "/scratch/alombard/sallm/snapshots/retained-sib-evaluator-20260913-v6/src/main"
RB = "/scratch/alombard/sallm/results/retained_standardized_20260913_v1/bases"
BASES = {"mzansilm": f"{RB}/mzansilm", "mamba2": f"{RB}/mamba2", "xlstm": f"{RB}/xlstm",
         "gdn": "/scratch/alombard/sallm/checkpoints/sallm-pure-gdn-125m/a10080-3epoch-20260802/final_model"}
PF = "/scratch/alombard/sallm_snapshots/general-prefix-fix-20260924"
INTENT_ENV = f"PYTHONPATH=/scratch/alombard/sallm_snapshots/downstream-generation-20260914-v8/src/main:/scratch/alombard/sallm_snapshots/full-matrix-execution-20260916-v1"
units = json.load(open("units.json"))
q = {"news": [], "sib": [], "intent": []}
order = {"mamba2": 0, "xlstm": 1, "gdn": 2, "mzansilm": 3}
for u in sorted(units, key=lambda u: (order[u["arch"]], u["id"])):
    if u["host"] != "kombuys": continue
    ad = u["local_path"] or f"{D}/adapters/{u['id']}"
    langs = ",".join(u["langs"])
    if u["task"] == "news":
        out = f"{D}/raw/news/{u['id']}.json"
        q["news"].append((u["id"], out, f"PYTHONPATH={V15} {PY} {D}/scripts/score_news_trainpos.py --arch {u['arch']} --base {BASES[u['arch']]} --adapter {ad} --langs {langs} --out {out}"))
    elif u["task"] == "sib":
        out = f"{D}/raw/sib/{u['id']}.json"
        q["sib"].append((u["id"], out, f"PYTHONPATH={SIBRT} {PY} {D}/scripts/score_sib_trainpos.py --arch {u['arch']} --base {BASES[u['arch']]} --adapter {ad} --langs {langs} --out {out}"))
    else:
        out = f"{D}/raw/intent/{u['id']}-train"
        q["intent"].append((u["id"], f"{out}/UNIT_DONE.json", f"cd {PF} && {INTENT_ENV} {PY} run_prefix_eval.py --spec {D}/scripts/intent_spec.json --unit-id {u['id']} --arch {u['arch']} --mode train --packs injongointent_all --output {out}"))
for k, v in q.items():
    open(f"q_{k}.tsv", "w").write("".join("\t".join(x) + "\n" for x in v))
    print(k, len(v))
