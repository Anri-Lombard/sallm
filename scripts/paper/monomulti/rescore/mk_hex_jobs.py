import json
R = "/scratch/lmbanr001/masters/sallm/results/monomulti_rescore_20260924"
units = json.load(open(f"{R}/hex_units.json"))
cost = {("pos", "xlstm", "multi"): 0, ("pos", "mzansilm", "multi"): 1}
def key(u):
    multi = u["unit_lang"] == "multi"
    return (cost.get((u["task"], u["arch"], "multi" if multi else "mono"), 2 if u["task"] == "pos" else 3), u["id"])
lines = []
for i, u in enumerate(sorted(units, key=key)):
    bounded = u["task"] == "pos" and u["unit_lang"] != "multi"
    lines.append([str(i), u["id"], u["arch"], u["task"], "bounded" if bounded else "v2", u["unit_lang"] if bounded else "-",
                  u["base_path"], u["adapter"], u["protocol"], f"{R}/raw/{u['task']}/{u['id']}.json"])
open(f"{R}/scripts/hex_jobs.tsv", "w").write("".join("\t".join(l) + "\n" for l in lines))
for l in lines: print(l[0], l[1], l[4], l[5])
