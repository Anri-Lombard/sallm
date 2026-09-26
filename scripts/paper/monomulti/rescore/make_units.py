import csv, json, sys
S, L = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(f"{S}/manifest.csv")))
host_for = {"news": "kombuys", "sib": "kombuys", "intent": "kombuys", "ner": "hex", "pos": "hex"}
units = {}
for r in rows:
    if r["status"] != "OK": continue
    if r["architecture"] == "gdn" and r["task"] in ("ner", "pos"): continue
    key = (r["architecture"], r["task"], r["regime"], r["language"] if r["regime"] == "Mono" else "multi")
    paths = [p.strip() for p in r["adapter_paths"].split("|")]
    target = host_for[r["task"]]
    local = [p.split(":",1)[1] for p in paths if p.startswith(target + ":")]
    u = units.setdefault(key, {"id": "_".join(key).lower(), "arch": key[0], "task": key[1], "regime": key[2],
        "unit_lang": key[3], "langs": [], "host": target, "tree_sha256": r["tree_sha256"],
        "source": paths[0], "local_path": local[0] if local else None})
    u["langs"].append(r["language"])
    assert u["tree_sha256"] == r["tree_sha256"]
out = list(units.values())
json.dump(out, open(f"{L}/units.json", "w"), indent=1)
for u in out: print(u["id"], u["host"], u["langs"], "LOCAL" if u["local_path"] else "COPY:" + u["source"])
print(len(out))
