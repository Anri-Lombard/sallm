#!/usr/bin/env bash
#SBATCH --account=maths
#SBATCH --partition=ada
#SBATCH --qos=normal
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --chdir=/home/lmbanr001/masters/sallm
#SBATCH --output=/scratch/lmbanr001/masters/sallm/logs/jobs/%x-%j.out

set -euo pipefail

: "${OUTPUT_ROOT:?set OUTPUT_ROOT}"
: "${RETAINED_STEP:?set RETAINED_STEP}"

[[ "$OUTPUT_ROOT" = /scratch/lmbanr001/masters/sallm/checkpoints/* ]]
[[ "$RETAINED_STEP" =~ ^[0-9]+$ ]]

/home/lmbanr001/masters/sallm/.venv/bin/python - <<'PY'
import hashlib
import os
from pathlib import Path

import torch
from safetensors.torch import load_file

root = Path(os.environ["OUTPUT_ROOT"])
retained = root / f"checkpoint-{os.environ['RETAINED_STEP']}" / "adapter_model.bin"
final = root / "final_adapter" / "adapter_model.safetensors"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


retained_state = torch.load(retained, map_location="cpu", weights_only=True)
final_state = load_file(final, device="cpu")
assert retained_state.keys() == final_state.keys()
assert all(torch.equal(retained_state[key], final_state[key]) for key in retained_state)

print(f"output_root={root}")
print(f"retained_step={os.environ['RETAINED_STEP']}")
print(f"keys={len(retained_state)}")
print(f"values={sum(t.numel() for t in retained_state.values())}")
print(f"retained_sha256={sha256(retained)}")
print(f"final_sha256={sha256(final)}")
print("exact_roundtrip=true")
PY
