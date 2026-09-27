#!/bin/bash
#SBATCH --partition=l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-28%4
#SBATCH --mail-type=FAIL,END

set -u

configs=(
  eval/run_mamba_sib_all
  eval/run_mamba_afrihg_all
  eval/run_mamba_afrihg_xho
  eval/run_mamba_afrihg_zul
  eval/run_mamba_injongointent_all
  eval/run_mamba_injongointent_eng
  eval/run_mamba_injongointent_sot
  eval/run_mamba_injongointent_xho
  eval/run_mamba_injongointent_zul
  eval/run_mamba_masakhaner_all
  eval/run_mamba_masakhaner_tsn
  eval/run_mamba_masakhaner_xho
  eval/run_mamba_masakhaner_zul
  eval/run_mamba_masakhanews_all
  eval/run_mamba_masakhanews_eng
  eval/run_mamba_masakhanews_xho
  eval/run_mamba_masakhapos_all
  eval/run_mamba_masakhapos_tsn
  eval/run_mamba_masakhapos_xho
  eval/run_mamba_masakhapos_zul
  eval/run_mamba_sa_general_all
  eval/run_mamba_afrisenti_tso
  eval/run_mamba_sib_afr
  eval/run_mamba_sib_eng
  eval/run_mamba_sib_nso
  eval/run_mamba_sib_sot
  eval/run_mamba_sib_xho
  eval/run_mamba_sib_zul
  eval/run_mamba_t2x_xho
)

repos=(
  anrilombard/sallm-gated_deltanet-davlan-sib200-afr_latn-eng_latn-nso_latn-sot_latn-xho_latn-zul_latn
  anrilombard/sallm-gated_deltanet-github-dadelani-afrihg-all
  anrilombard/sallm-gated_deltanet-github-dadelani-afrihg-xho
  anrilombard/sallm-gated_deltanet-github-dadelani-afrihg-zul
  anrilombard/sallm-gated_deltanet-masakhane-injongointent-eng-sot-xho-zul
  anrilombard/sallm-gated_deltanet-masakhane-injongointent-eng
  anrilombard/sallm-gated_deltanet-masakhane-injongointent-sot
  anrilombard/sallm-gated_deltanet-masakhane-injongointent-xho
  anrilombard/sallm-gated_deltanet-masakhane-injongointent-zul
  anrilombard/sallm-gated_deltanet-masakhane-masakhaner2-tsn-xho-zul
  anrilombard/sallm-gated_deltanet-masakhane-masakhaner2-tsn
  anrilombard/sallm-gated_deltanet-masakhane-masakhaner2-xho
  anrilombard/sallm-gated_deltanet-masakhane-masakhaner2-zul
  anrilombard/sallm-gated_deltanet-masakhane-masakhanews-eng-xho
  anrilombard/sallm-gated_deltanet-masakhane-masakhanews-eng
  anrilombard/sallm-gated_deltanet-masakhane-masakhanews-xho
  anrilombard/sallm-gated_deltanet-masakhane-masakhapos-tsn-xho-zul
  anrilombard/sallm-gated_deltanet-masakhane-masakhapos-tsn
  anrilombard/sallm-gated_deltanet-masakhane-masakhapos-xho
  anrilombard/sallm-gated_deltanet-masakhane-masakhapos-zul
  anrilombard/sallm-gated_deltanet-sa_general-all
  anrilombard/sallm-gated_deltanet-masakhane-afrisenti-tso
  anrilombard/sallm-gated_deltanet-davlan-sib200-afr_latn
  anrilombard/sallm-gated_deltanet-davlan-sib200-eng_latn
  anrilombard/sallm-gated_deltanet-davlan-sib200-nso_latn
  anrilombard/sallm-gated_deltanet-davlan-sib200-sot_latn
  anrilombard/sallm-gated_deltanet-davlan-sib200-xho_latn
  anrilombard/sallm-gated_deltanet-davlan-sib200-zul_latn
  anrilombard/sallm-gated_deltanet-github-francois-meyer-t2x-xho
)

fine_tune_jobs=(
  1020298 1020313 1020314 1020315 1020316 1020317 1020318 1020319 1020320
  1020328 1020330 1020337 1020339 1020341 1020342 1020345 1020349 1020350
  1020355 1020356 1020357 1029325 1020359 1020360 1020361 1020365 1020366
  1020367 1020368
)

index="${SLURM_ARRAY_TASK_ID:-}"
if [[ ! "$index" =~ ^[0-9]+$ || "$index" -ge "${#configs[@]}" ]]; then
  echo "ERROR: invalid array index '$index'." >&2
  exit 1
fi

if [[ "${SKIP_FINETUNE_STATE_CHECK:-0}" != 1 ]]; then
  fine_tune_state="$(sacct -X -n -P -j "${fine_tune_jobs[$index]}" --format=State | tr -d ' ' | tail -n 1)"
  if [[ "$fine_tune_state" != COMPLETED ]]; then
    echo "Skipping ${configs[$index]} because fine-tune ${fine_tune_jobs[$index]} ended in $fine_tune_state."
    exit 1
  fi
fi

suffix="${configs[$index]##*/}"
suffix="${suffix#run_mamba_}"
output_root="${SALLM_EVAL_OUTPUT_ROOT:-/scratch/lmbanr001/masters/sallm/results/eval}"

if [[ "$index" =~ ^(1|2|3)$ ]] && \
   [[ -f "${output_root}/gdn_125m_adapter_${suffix}_r2/evaluation_summary.json" ]]; then
  echo "Skipping unaffected native-generation result for ${suffix}."
  exit 0
fi

output_dir="${output_root}/gdn_125m_adapter_${suffix}_r3"
wandb_name="eval-gdn-125m-adapter-${suffix}-r3"

if [[ -f "$output_dir/evaluation_summary.json" ]]; then
  echo "Skipping completed result $output_dir/evaluation_summary.json"
  exit 0
fi

export FLA_DISABLE_BACKEND_DISPATCH=1
export SALLM_SKIP_MAMBA_KERNEL_CHECK=1
exec bash "${SALLM_REPO_DIR:-$HOME/masters/sallm}/scripts/launch_evaluation.sh" \
  "${configs[$index]}" \
  "eval.eval_model.checkpoint=anrilombard/sallm-gated-deltanet-125m-shallowwide-4x40-20260707" \
  "++eval.eval_model.tie_word_embeddings=true" \
  "++eval.eval_model.peft_adapter=${repos[$index]}" \
  "eval.evaluation.output_dir=${output_dir}" \
  "eval.wandb.name=${wandb_name}"
