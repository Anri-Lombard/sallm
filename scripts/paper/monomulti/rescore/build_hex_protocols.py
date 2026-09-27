import copy, hashlib, json, sys
from pathlib import Path
R = Path("/scratch/lmbanr001/masters/sallm/results/monomulti_rescore_20260924")
BASE = Path("/scratch/lmbanr001/masters/sallm_snapshots/general-sequence-official-test-20260916-v1/general_sequence_validation_hex_protocol_20260915_v3.json")
EOSFIX = Path("/scratch/lmbanr001/masters/sallm/results/mamba_eos_fix_20260924/protocols/general_ner_protocol_eosfix.json")
FILES = ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin", "chat_template.jinja", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def tree(root):
    root = root.resolve()
    files = sorted(p for p in root.rglob("*") if p.is_file() and ".cache" not in p.relative_to(root).parts)
    return hashlib.sha256("".join(f"{sha(p)}  ./{p.relative_to(root).as_posix()}\n" for p in files).encode()).hexdigest()

units = json.load(open(sys.argv[1]))
nonmodel = lambda d: json.dumps({k: v for k, v in d.items() if k != "models"}, sort_keys=True)
base_v3, base_eos = json.loads(BASE.read_text()), json.loads(EOSFIX.read_text())
assert nonmodel(base_v3) == nonmodel(base_eos)
out = []
for u in units:
    if u["host"] != "hex": continue
    adapter = Path(u["local_path"] or f"{R}/adapters/{u['id']}")
    t = tree(adapter)
    if t not in [x.strip() for x in u["tree_sha256"].split("|")]:
        print("TREE_MISMATCH", u["id"], t, u["tree_sha256"]); continue
    src = base_eos if (u["arch"] == "mamba2" and u["task"] == "ner") else base_v3
    p = copy.deepcopy(src)
    m = p["models"][u["arch"]]
    m["adapter_path"] = str(adapter)
    m["adapter_files"] = {f: sha(adapter / f) for f in FILES if (adapter / f).is_file()}
    m["adapter_tree_sha256"] = t
    assert nonmodel(p) == nonmodel(src)
    for a in p["models"]:
        if a != u["arch"]: assert p["models"][a] == src["models"][a]
    path = R / "protocols" / f"{u['id']}.json"
    if path.exists():
        assert json.loads(path.read_text()) == p, path
    else:
        path.write_text(json.dumps(p, indent=2) + "\n")
    out.append({**u, "adapter": str(adapter), "protocol": str(path), "base_path": m["base_path"], "protocol_sha256": sha(path)})
    print("OK", u["id"], m["base_path"].split("/")[-1], sorted(m["adapter_files"]))
(R / "hex_units.json").write_text(json.dumps(out, indent=1))
