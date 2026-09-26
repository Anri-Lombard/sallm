#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --qos=l40sfree
#SBATCH --partition=l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=srvrocgpu009
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
# Validation selection -> TEST -> prune, for each label in $LABELS (training must be finished).
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
R=/scratch/lmbanr001/masters/sallm/results/monomulti_retrain_20260924
PAR=${PAR:-3}
for label in ${LABELS:?}; do
  grep -q '^final_adapter_tree_sha256=' "$R/train/$label/run_info.txt"
  mkdir -p "$R/selection/$label"
  echo "$(date -Is) PIPELINE_START $label"
  case "$label" in
    *_sib_*)
      base=/scratch/lmbanr001/masters/sallm_snapshots/full-matrix-retained-bindings-20260916-v1/bases/mamba2
      [[ "$label" == xlstm_* ]] && base=/scratch/lmbanr001/masters/sallm_snapshots/full-matrix-retained-bindings-20260916-v1/bases/xlstm
      arch=${label%%_*}
      mapfile -t cks < <(ls -d "$R/train/$label"/checkpoint-* | sort -t- -k2 -n)
      for c in "${cks[@]}"; do
        [[ -e "$R/selection/$label/$(basename "$c").validation.json" ]] && continue
        ARCH=$arch BASE=$base SPLIT=validation OUTDIR=$R/selection/$label ADAPTERS="$c" bash "$R/jobs/sib_eval.sbatch"
      done
      choice=$(python3 - "$R/selection/$label" <<'PY'
import json, sys
from pathlib import Path
sel = Path(sys.argv[1]); rows = []
for f in sel.glob("checkpoint-*.validation.json"):
    d = json.loads(f.read_text()); step = int(f.name.split(".")[0].split("-")[1])
    langs = list(d["languages"])
    rows.append((step, sum(d["languages"][l]["f1"] for l in langs) / len(langs), {l: d["languages"][l]["f1"] for l in langs}, d["adapter"]))
rows.sort(); best = max(rows, key=lambda r: (r[1], -r[0]))
json.dump({"rule": "max unweighted mean validation support-weighted F1 over languages, General SIB scorer (HEX copy), protocol prompts; earliest step on ties",
           "selected_step": best[0], "selected_adapter": best[3], "selected_mean": best[1],
           "per_checkpoint": [{"step": s, "mean": m, "per_language": f} for s, m, f, _ in rows]}, open(sel / "SELECTION.json", "w"), indent=1)
print(best[3])
PY
)
      mkdir -p "$R/test/sib"
      ARCH=$arch BASE=$base SPLIT=test OUTDIR=$R/test/sib ADAPTERS="$choice" bash "$R/jobs/sib_eval.sbatch"
      mv "$R/test/sib/$(basename "$choice").test.json" "$R/test/sib/$label.test.json"
      keep=$(basename "$choice")
      cp "$(ls -d "$R/train/$label"/checkpoint-* | sort -t- -k2 -n | tail -1)/trainer_state.json" "$R/train/$label/trainer_state_final.json"
      for c in "$R/train/$label"/checkpoint-*; do [[ "$(basename "$c")" == "$keep" ]] || rm -rf "$c"; done
      rm -rf "$R/train/$label/final_adapter"
      ;;
    *)
      python3 "$R/jobs/seqsel.py" units "$label" ${TOP:+--top $TOP}
      UNITS=$R/jobs/units_${label}_val.tsv PAR=$PAR bash "$R/jobs/eval_multi.sbatch"
      python3 "$R/jobs/seqsel.py" select "$label"
      UNITS=$R/jobs/units_${label}_test.tsv PAR=1 bash "$R/jobs/eval_multi.sbatch"
      python3 "$R/jobs/seqsel.py" prune "$label"
      ;;
  esac
  echo "$(date -Is) PIPELINE_DONE $label"
done
