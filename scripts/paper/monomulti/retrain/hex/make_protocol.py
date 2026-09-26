#!/usr/bin/env python3
"""Write a General NER/POS protocol copy that binds one retrained adapter (only models.<arch> changes)."""
import copy, hashlib, json, sys
from pathlib import Path

R = Path("/scratch/lmbanr001/masters/sallm/results/monomulti_retrain_20260924")
BASE = Path("/scratch/lmbanr001/masters/sallm_snapshots/general-sequence-official-test-20260916-v1/general_sequence_validation_hex_protocol_20260915_v3.json")
EOSFIX = Path("/scratch/lmbanr001/masters/sallm/results/mamba_eos_fix_20260924/protocols/general_ner_protocol_eosfix.json")
FILES = ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin", "chat_template.jinja", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def tree(root):
    root = root.resolve()
    files = sorted(p for p in root.rglob("*") if p.is_file() and ".cache" not in p.relative_to(root).parts)
    return hashlib.sha256("".join(f"{sha(p)}  ./{p.relative_to(root).as_posix()}\n" for p in files).encode()).hexdigest()


unit, arch, task, adapter = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4])
nonmodel = lambda d: json.dumps({k: v for k, v in d.items() if k != "models"}, sort_keys=True)
src = json.loads((EOSFIX if (arch == "mamba2" and task == "ner") else BASE).read_text())
p = copy.deepcopy(src)
m = p["models"][arch]
m["adapter_path"] = str(adapter)
m["adapter_files"] = {f: sha(adapter / f) for f in FILES if (adapter / f).is_file()}
m["adapter_tree_sha256"] = tree(adapter)
assert nonmodel(p) == nonmodel(src)
assert all(p["models"][a] == src["models"][a] for a in p["models"] if a != arch)
out = R / "protocols" / f"{unit}.json"
out.parent.mkdir(parents=True, exist_ok=True)
assert not out.exists(), out
out.write_text(json.dumps(p, indent=2) + "\n")
print(out, m["base_path"], m["adapter_tree_sha256"], sorted(m["adapter_files"]))
