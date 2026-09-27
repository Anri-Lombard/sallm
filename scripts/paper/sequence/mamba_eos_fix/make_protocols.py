import json, hashlib, copy
from pathlib import Path
R = Path("/scratch/lmbanr001/masters/sallm/results/mamba_eos_fix_20260924")
FIX = R / "bases/mamba2_eosfix"
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
files = dict(line.split()[::-1] for line in (R / "bases/mamba2_eosfix.FILES.sha256").read_text().splitlines())
tree = hashlib.sha256((R / "bases/mamba2_eosfix.FILES.sha256").read_text().splitlines().__iter__().__class__ and "".encode()).hexdigest()
out = R / "protocols"; out.mkdir(exist_ok=True)
def patch(src, dst):
    p = json.loads(Path(src).read_text())
    m = p["models"]["mamba2"]
    old = copy.deepcopy(m)
    m["base_path"] = str(FIX)
    m["base_files"] = {k: files[k] for k in m["base_files"]}
    if "generation_config.json" not in m["base_files"]:
        m["base_files"]["generation_config.json"] = files["generation_config.json"]
    m["base_tree_sha256"] = "eosfix:" + hashlib.sha256(json.dumps(m["base_files"], sort_keys=True).encode()).hexdigest()
    m["eos_fix_note"] = {"source_base_path": old["base_path"], "source_base_tree_sha256": old["base_tree_sha256"],
                         "change": "config.json and generation_config.json: eos_token_id 2->1 ([EOS]), pad_token_id 1->2 ([PAD]); all other files byte-identical"}
    Path(dst).write_text(json.dumps(p, indent=2, sort_keys=True) + "\n")
    diff = {k: (old.get(k), m.get(k)) for k in set(old) | set(m) if old.get(k) != m.get(k)}
    print(dst, sha(dst)); print("  changed keys:", sorted(diff))
    others = [a for a in p["models"] if a != "mamba2"]
    print("  untouched models:", others)
patch("/scratch/lmbanr001/masters/sallm_snapshots/general-sequence-official-test-20260916-v1/general_sequence_validation_hex_protocol_20260915_v3.json", out / "general_ner_protocol_eosfix.json")
patch("/scratch/lmbanr001/masters/sallm/results/full_matrix_execution_20260916_v1/base-mamba-xlstm-20260924/sequence_protocols/u005-base-mamba2-ner.json", out / "u005-base-mamba2-ner-eosfix.json")
# unit 7: bindings + inventory with AfriMGSM-only tasks
B = "/scratch/lmbanr001/masters/sallm/results/full_matrix_execution_20260916_v1/base-mamba-xlstm-20260924/bindings/BINDINGS.json"
b = json.loads(Path(B).read_text())
key = "5a9dcaaab4f04013d770b91a8701d8e06d6acd2e6e4d53a658681d98786bfd34"
assert b["entries"][key]["path"].endswith("full-matrix-retained-bindings-20260916-v1/bases/mamba2")
b["entries"][key]["path"] = str(FIX)
b["entries"][key]["files"] = {k: files[k] for k in b["entries"][key]["files"]}
b["entries"][key]["eos_fix_note"] = "path redirected to EOS-fixed copy; eos 2->1, pad 1->2"
(out / "u007_bindings").mkdir(exist_ok=True)
(out / "u007_bindings/BINDINGS.json").write_text(json.dumps(b, indent=2, sort_keys=True) + "\n")
inv = Path("/scratch/lmbanr001/masters/sallm_snapshots/full-matrix-execution-20260916-v1/frozen-inventory/official_units.json")
units = json.loads(inv.read_text())
u = units[7]; assert u["unit_id"] == "u007-base-mamba2-prompt"
u["task_names"] = [t for t in u["task_names"] if t.startswith("afrimgsm_")]
u["cell_ids"] = [c for c in u["cell_ids"] if ":afrimgsm:" in c]
u["cells"] = len(u["cell_ids"]); u["tasks"] = ["AfriMGSM"]
u["eos_fix_note"] = "AfriMGSM-only subset of u007 for EOS-fixed rerun"
(out / "u007_inventory").mkdir(exist_ok=True)
(out / "u007_inventory/official_units.json").write_text(json.dumps(units, ensure_ascii=False, indent=2) + "\n")
print("u007 tasks:", u["task_names"])
