#!/usr/bin/env bash
# usage: select_and_test_sib.sh LABEL GPU   (Mamba SIB retrains: validation selection over every epoch checkpoint, then TEST on the selected one)
set -euo pipefail
label="$1"; gpu="$2"
R=/scratch/alombard/sallm/results/monomulti_retrain_20260924
run="$R/train/$label"
grep -q '^final_adapter_tree_sha256=' "$run/run_info.txt"
case "$label" in
  mamba2_sib_multi) langs=afr,eng,nso,sot,xho,zul ;;
  mamba2_sib_mono_*_eb64) langs="${label#mamba2_sib_mono_}"; langs="${langs%_eb64}" ;;
  mamba2_sib_mono_*) langs="${label##*_}" ;;
  *) exit 4 ;;
esac
base=/scratch/alombard/sallm/results/retained_standardized_20260913_v1/bases/mamba2
export HF_HOME=/scratch/alombard/sallm/hf HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 \
  WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=42 PYTHONDONTWRITEBYTECODE=1 \
  FLA_DISABLE_BACKEND_DISPATCH=1 SALLM_SKIP_MAMBA_KERNEL_CHECK=1 MAMBA_SCAN_IMPL=cuda CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu
export PYTHONPATH=/scratch/alombard/sallm/snapshots/retained-sib-evaluator-20260913-v6/src/main
py=/scratch/alombard/sallm/.venv/bin/python
mapfile -t ckpts < <(ls -d "$run"/checkpoint-* | sort -t- -k2 -n)
sel="$R/selection/$label"
mkdir -p "$sel"
for c in "${ckpts[@]}"; do
  [[ -e "$sel/$(basename "$c").validation.json" ]] && continue
  "$py" "$R/scripts/score_sib_split.py" --arch mamba2 --base "$base" --split validation --langs "$langs" --out-dir "$sel" --adapter "$c"
done
choice=$("$py" - "$sel" "$langs" <<'PY'
import json, sys
from pathlib import Path
sel, langs = Path(sys.argv[1]), sys.argv[2].split(",")
rows = []
for f in sel.glob("checkpoint-*.validation.json"):
    d = json.loads(f.read_text())
    step = int(f.name.split(".")[0].split("-")[1])
    rows.append((step, sum(d["languages"][l]["f1"] for l in langs) / len(langs), {l: d["languages"][l]["f1"] for l in langs}, d["adapter"]))
rows.sort()
best = max(rows, key=lambda r: (r[1], -r[0]))
json.dump({"rule": "max unweighted mean validation weighted-F1 over languages; earliest step on ties", "selected_step": best[0],
           "selected_adapter": best[3], "selected_mean_f1": best[1],
           "per_checkpoint": [{"step": s, "mean_f1": m, "f1": f} for s, m, f, _ in rows]}, open(sel / "SELECTION.json", "w"), indent=1)
print(best[3])
PY
)
echo "SELECTED $choice"
nvidia-smi -i "$gpu" --query-gpu=name --format=csv,noheader > "$R/test/sib/$label.gpu.txt"
"$py" /scratch/alombard/sallm/results/monomulti_rescore_inventory_20260924/score_sib_trainpos.py --arch mamba2 --base "$base" \
  --adapter "$choice" --langs "$langs" --out "$R/test/sib/$label.json"
echo SIB_SELECT_TEST_DONE
