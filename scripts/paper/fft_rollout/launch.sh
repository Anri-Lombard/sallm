#!/bin/bash
# Fire one architecture's whole post-pretraining pipeline on HEX (head node: bash + sbatch only).
#   bash launch.sh ARCH BASE_PATH [CAP] [NAME]
# ARCH       mzansilm | mamba2 | xlstm | gdn
# BASE_PATH  the pretrained weights dir (config.json + pytorch_model.bin), or the matched-pretraining run dir
#            (runs/full_<arch>_wsd_lr<LR>; prep then binds weights/step<total_steps>_tok*, the final weights)
# CAP        concurrent L40S lanes for this architecture (default 4; the per-user QOS limit is 10 GPUs in total)
# NAME       run directory name under runs/ (default ARCH)
# env AFTER=<jobid>        hold the lanes until that Slurm job ends (afterany), e.g. the pretraining job
# env FIX_SPECIAL_IDS=1    stand-in bases only: rewrite config bos/eos/pad to the tokenizer's 0/1/2
# env SMOKE=1              smoke matrix (tiny cells, ~20 scored items); SMOKE_FAMILIES="t2x" restricts it
# env LANE_TIME=HH:MM:SS   lane wall time (default 48:00:00); lanes resubmit themselves before it runs out
set -euo pipefail
ARCH="${1:?ARCH}"; BASE="${2:?BASE_PATH}"; CAP="${3:-4}"; NAME="${4:-$ARCH}"
R=/scratch/lmbanr001/masters/sallm/results/fft_rollout_20260926
OUT="$R/runs/$NAME"
case "$ARCH" in mzansilm|mamba2|xlstm|gdn) ;; *) echo "ERROR: unknown ARCH $ARCH" >&2; exit 2 ;; esac
[[ "$BASE" == /* ]] || { echo "ERROR: BASE_PATH must be absolute" >&2; exit 2; }
if [[ -z "${AFTER:-}" ]]; then
  if [[ -f "$BASE/run_meta.json" ]]; then
    ls -d "$BASE"/weights/step*_tok* >/dev/null || { echo "ERROR: no weights under $BASE/weights" >&2; exit 2; }
  else
    [[ -f "$BASE/config.json" && ( -f "$BASE/pytorch_model.bin" || -f "$BASE/model.safetensors" ) ]] \
      || { echo "ERROR: no config.json + weights in $BASE" >&2; exit 2; }
    if [[ "${FIX_SPECIAL_IDS:-0}" != 1 ]]; then
      for kv in '"bos_token_id": 0' '"eos_token_id": 1' '"pad_token_id": 2'; do
        grep -q "$kv" "$BASE/config.json" || { echo "ERROR: $BASE/config.json lacks $kv" >&2; exit 3; }
      done
    fi
  fi
fi
if [[ -e "$OUT/config.json" ]]; then
  echo "ERROR: $OUT exists. To resume it: sbatch --export=ALL,OUT=$OUT,LANE=<i> $R/code/fft_rollout/lane.sbatch" >&2; exit 4
fi
(cd "$R/code" && sha256sum -c CODE.sha256 --quiet)
mkdir -p "$OUT" "$R/logs"
smoke=false; [[ "${SMOKE:-0}" == 1 ]] && smoke=true
fix=false; [[ "${FIX_SPECIAL_IDS:-0}" == 1 ]] && fix=true
sf=null; [[ -n "${SMOKE_FAMILIES:-}" ]] && sf="[\"${SMOKE_FAMILIES// /\", \"}\"]"
printf '{"arch": "%s", "base_src": "%s", "smoke": %s, "fix_special_ids": %s, "cap": %s, "smoke_families": %s, "lane_time": "%s", "launched": "%s", "code_manifest_sha256": "%s"}\n' \
  "$ARCH" "$BASE" "$smoke" "$fix" "$CAP" "$sf" "${LANE_TIME:-48:00:00}" "$(date -Is)" "$(sha256sum "$R/code/CODE.sha256" | cut -d' ' -f1)" > "$OUT/config.json"
echo "$CAP" > "$OUT/MAX_LANES"
dep=(); [[ -n "${AFTER:-}" ]] && dep=(--dependency="afterany:$AFTER")
lt="${LANE_TIME:-48:00:00}"; IFS=: read -r h m _ <<< "$lt"; lane_hours=$(( 10#$h * 60 + 10#$m - 10 ))
for ((i = 0; i < CAP; i++)); do
  j=$(sbatch --parsable "${dep[@]}" --time="$lt" --job-name="fft-$NAME-$i" --export="ALL,OUT=$OUT,LANE=$i,LANE_HOURS=$(( lane_hours / 60 )).$(( (lane_hours % 60) * 100 / 60 ))" "$R/code/fft_rollout/lane.sbatch")
  echo "$(date -Is) lane $i -> job $j" | tee -a "$OUT/LAUNCH.log"
done
echo "status: cat $OUT/STATUS.txt   lanes: squeue -u \$USER -n $(for ((i = 0; i < CAP; i++)); do printf 'fft-%s-%s,' "$NAME" "$i"; done | sed 's/,$//')"
