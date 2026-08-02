#!/bin/bash

set -uo pipefail

repo=/scratch/alombard/sallm
log="$repo/logs/post_gdn_architecture_controls_20260716.log"
marker="$repo/results/diagnostics/post_gdn_architecture_controls_20260716"

mkdir -p "$marker"
exec >>"$log" 2>&1

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

echo "[$(timestamp)] Coordinator started."

while tmux has-session -t gdn-base-fewshot 2>/dev/null ||
      tmux has-session -t gdn-base-tail 2>/dev/null; do
  echo "[$(timestamp)] Waiting for active GDN base queues."
  sleep 300
done

while pgrep -u "$USER" -x python >/dev/null; do
  echo "[$(timestamp)] Waiting for SALLM GPU processes to clear."
  sleep 300
done

available_kb="$(df --output=avail /scratch | tail -n 1 | tr -d ' ')"
if [[ "$available_kb" -lt 104857600 ]]; then
  echo "[$(timestamp)] BLOCKED: less than 100 GiB free on /scratch."
  exit 1
fi

echo "[$(timestamp)] Kombuys clear; starting remaining corrected GDN adapter evaluations."
cd "$repo" || exit 1

if ! bash scripts/run_gdn_adapter_evals_kombuys.sh; then
  echo "[$(timestamp)] BLOCKED: corrected GDN adapter matrix failed."
  exit 1
fi

echo "[$(timestamp)] Corrected GDN adapter matrix complete."

source "$repo/.venv/bin/activate"
export CUDA_VISIBLE_DEVICES=0
export SALLM_HOME_DIR=/home/alombard
export SALLM_SCRATCH_DIR=/scratch/alombard
export SALLM_REPO_DIR="$repo"
export SALLM_EVAL_OUTPUT_ROOT="$repo/results/eval"
export HF_TOKEN_FILE="$repo/hf/token"
export FLA_DISABLE_BACKEND_DISPATCH=1
export SALLM_SKIP_MAMBA_KERNEL_CHECK=1

mamba_output="$repo/results/eval/diagnostics/matched_protocol_20260716/mamba_injongointent_all"
if [[ -f "$mamba_output/evaluation_summary.json" ]]; then
  echo "[$(timestamp)] Skipping completed matched Mamba InjongoIntent control."
else
  echo "[$(timestamp)] Starting matched Mamba InjongoIntent control."
  if ! bash scripts/launch_evaluation.sh \
    eval/run_mamba_injongointent_all \
    "eval.evaluation.output_dir=$mamba_output" \
    "eval.wandb.name=matched-mamba-injongointent-all-20260716"; then
    echo "[$(timestamp)] BLOCKED: matched Mamba InjongoIntent control failed."
    exit 1
  fi
fi

llama_checkpoint=
for candidate in \
  "$repo/checkpoints/ft_llama_125m_injongointent_all/final_merged_model" \
  "/scratch/alombard/checkpoints/ft_llama_125m_injongointent_all/final_merged_model" \
  "/scratch/alombard/masters/sallm/checkpoints/ft_llama_125m_injongointent_all/final_merged_model"; do
  if [[ -f "$candidate/config.json" ]]; then
    llama_checkpoint="$candidate"
    break
  fi
done

if [[ -z "$llama_checkpoint" ]]; then
  echo "[$(timestamp)] BLOCKED: matched LLaMA InjongoIntent checkpoint not found; provenance audit required."
  touch "$marker/llama_checkpoint_missing"
else
  llama_output="$repo/results/eval/diagnostics/matched_protocol_20260716/llama_injongointent_all"
  if [[ -f "$llama_output/evaluation_summary.json" ]]; then
    echo "[$(timestamp)] Skipping completed matched LLaMA InjongoIntent control."
  else
    echo "[$(timestamp)] Starting matched LLaMA InjongoIntent control from $llama_checkpoint."
    if ! bash scripts/launch_evaluation.sh \
      eval/run_llama_injongointent_all \
      "eval.eval_model.checkpoint=$llama_checkpoint" \
      "eval.evaluation.output_dir=$llama_output" \
      "eval.wandb.name=matched-llama-injongointent-all-20260716"; then
      echo "[$(timestamp)] BLOCKED: matched LLaMA InjongoIntent control failed."
      exit 1
    fi
  fi
fi

echo "[$(timestamp)] Starting zero-shot GDN base evaluations on GPU 0."
if ! bash scripts/run_gdn_base_fewshot_kombuys.sh core 0 0; then
  echo "[$(timestamp)] BLOCKED: zero-shot GDN core evaluations failed."
  exit 1
fi
if ! bash scripts/run_gdn_base_fewshot_kombuys.sh tail 0 0; then
  echo "[$(timestamp)] BLOCKED: zero-shot GDN tail evaluations failed."
  exit 1
fi
echo "[$(timestamp)] Zero-shot GDN base evaluations complete."

touch "$marker/coordinator_complete"
echo "[$(timestamp)] Coordinator complete."
