import json, sys
from pathlib import Path
sys.path.insert(0, "/scratch/alombard/sallm_snapshots/full-matrix-execution-20260916-v1")
from prepare_bindings import tree_sha256
D = Path("/scratch/alombard/sallm/results/monomulti_rescore_20260924")
general = {u["architecture"]: u for u in json.load(open("/scratch/alombard/sallm_snapshots/general-prefix-fix-20260924/units.json"))}
units = json.load(open(D / "scripts/units.json"))
spec, missing = [], []
for u in units:
    if u["task"] != "intent": continue
    adapter = Path(u["local_path"] or D / "adapters" / u["id"])
    if not adapter.exists():
        missing.append(u["id"]); continue
    t = tree_sha256(adapter)
    if t not in [x.strip() for x in u["tree_sha256"].split("|")]:
        print("TREE_MISMATCH", u["id"], t, u["tree_sha256"]); continue
    g = general[u["arch"]]
    spec.append({"unit_id": u["id"], "architecture": u["arch"], "base": g["base"], "base_tree_sha256": g["base_tree_sha256"],
                 "adapter": str(adapter), "adapter_tree_sha256": t, "languages": u["langs"]})
    print("OK", u["id"])
(D / "scripts/intent_spec.json").write_text(json.dumps(spec, indent=1) + "\n")
print("missing", missing)
