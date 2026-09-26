#!/bin/bash
# Uniform validation-only HPO wrapper for SALLM adapter-capable architectures.

set -euo pipefail

model_id="${1:?Usage: $0 <pure_gdn|llama125|mamba125|xlstm125|qwen_gdn_hybrid|custom> <family> <stage_a|stage_b|confirm> <candidate> <seed>}"
family="${2:?missing family}"
stage="${3:?missing stage}"
candidate="${4:?missing candidate}"
seed="${5:?missing seed}"
repo="${SALLM_REPO_DIR:-$HOME/masters/sallm}"
scratch="${SCRATCH:-/scratch/lmbanr001}"
registry="${SALLM_HPO_REGISTRY:-$repo/src/conf/hpo/pure_gdn_enhanced_v1.json}"
python_bin="${SALLM_RUNTIME_REPO:-$repo}/.venv/bin/python"
[[ -x "$python_bin" ]] || python_bin=python3

case "$stage" in
  stage_a) [[ "$candidate" =~ ^a[0-2]$ && "$seed" == 42 ]] ;;
  stage_b) [[ "$candidate" =~ ^b[0-7]$ && "$seed" == 42 ]] ;;
  confirm) [[ "$candidate" =~ ^[ab][0-9]+$ && "$seed" =~ ^(13|87)$ ]] ;;
  *) false ;;
esac || { echo "ERROR: invalid stage/candidate/seed combination." >&2; exit 1; }
if [[ "$stage" == confirm ]]; then
  confirm_candidates="${SALLM_HPO_CONFIRM_CANDIDATES:?confirmation requires the frozen comma-separated top two in SALLM_HPO_CONFIRM_CANDIDATES}"
  [[ ",$confirm_candidates," == *",$candidate,"* ]] || {
    echo "ERROR: '$candidate' is not a frozen confirmation candidate." >&2
    exit 1
  }
fi

config_suffix="$family"
case "$family" in
  news) config_suffix=news_all ;;
  ner|pos|sib|afrihg) config_suffix="${family}_all" ;;
  intent) config_suffix=injongointent_all ;;
  t2x) config_suffix=t2x_xho ;;
  general) config_suffix=sa_general_all ;;
  *) echo "ERROR: unsupported family '$family'." >&2; exit 1 ;;
esac

case "$model_id" in
  pure_gdn)
    export SALLM_HPO_MODEL_ARCHITECTURE=gated_deltanet
    export SALLM_HPO_MODEL="${SALLM_HPO_MODEL:-$scratch/masters/sallm/checkpoints/sallm-pure-gdn-125m/a10080-3epoch-20260802/final_model}"
    export SALLM_HPO_TOKENIZER="${SALLM_HPO_TOKENIZER:-$SALLM_HPO_MODEL}"
    export SALLM_HPO_TARGET_MODULES="${SALLM_HPO_TARGET_MODULES:-[q_proj,k_proj,v_proj,a_proj,b_proj,g_proj,o_proj,gate_proj,up_proj,down_proj]}"
    ;;
  llama125)
    export SALLM_HPO_CONFIG="llama_${config_suffix}"
    export SALLM_HPO_MODEL_ARCHITECTURE=llama
    export SALLM_HPO_MODEL="${SALLM_HPO_MODEL:-$scratch/masters/sallm/checkpoints/sallm-llama-125m/final_model}"
    export SALLM_HPO_TARGET_MODULES="${SALLM_HPO_TARGET_MODULES:-[q_proj,k_proj,v_proj,o_proj]}"
    ;;
  mamba125)
    export SALLM_HPO_CONFIG="mamba_${config_suffix}"
    export SALLM_HPO_MODEL_ARCHITECTURE=mamba2
    export SALLM_HPO_MODEL="${SALLM_HPO_MODEL:-anrilombard/sallm-mamba-125m}"
    export SALLM_HPO_TARGET_MODULES="${SALLM_HPO_TARGET_MODULES:-[in_proj,out_proj]}"
    export SALLM_HPO_TRAIN_BATCH_SIZE="${SALLM_HPO_TRAIN_BATCH_SIZE:-1}"
    export SALLM_HPO_EVAL_BATCH_SIZE="${SALLM_HPO_EVAL_BATCH_SIZE:-1}"
    export SALLM_HPO_GRADIENT_ACCUMULATION_STEPS="${SALLM_HPO_GRADIENT_ACCUMULATION_STEPS:-8}"
    export SALLM_HPO_MAX_LENGTH="${SALLM_HPO_MAX_LENGTH:-1024}"
    export SALLM_HPO_GRADIENT_CHECKPOINTING="${SALLM_HPO_GRADIENT_CHECKPOINTING:-false}"
    ;;
  xlstm125)
    export SALLM_HPO_CONFIG="xlstm_${config_suffix}"
    export SALLM_HPO_MODEL_ARCHITECTURE=xlstm
    export SALLM_HPO_MODEL="${SALLM_HPO_MODEL:-anrilombard/sallm-xlstm-125m-native-3epoch-20260531}"
    export SALLM_HPO_TARGET_MODULES="${SALLM_HPO_TARGET_MODULES:-[q,k,v,out_proj]}"
    ;;
  qwen_gdn_hybrid)
    export SALLM_HPO_CONFIG="llama_${config_suffix}"
    export SALLM_HPO_MODEL_ARCHITECTURE=qwen3next_gdn_hybrid
    export SALLM_HPO_MODEL="${SALLM_HPO_MODEL:-anrilombard/sallm-gated-deltanet-125m-shallowwide-4x40-20260707}"
    export SALLM_HPO_TARGET_MODULES="${SALLM_HPO_TARGET_MODULES:-[in_proj_qkvz,in_proj_ba,out_proj,q_proj,k_proj,v_proj,o_proj]}"
    ;;
  custom)
    : "${SALLM_HPO_CONFIG:?custom profile requires SALLM_HPO_CONFIG}"
    : "${SALLM_HPO_MODEL_ARCHITECTURE:?custom profile requires SALLM_HPO_MODEL_ARCHITECTURE}"
    : "${SALLM_HPO_MODEL:?custom profile requires SALLM_HPO_MODEL}"
    ;;
  *) echo "ERROR: unsupported model profile '$model_id'." >&2; exit 1 ;;
esac

if [[ "$model_id" != pure_gdn ]]; then
  export SALLM_HPO_TOKENIZER="${SALLM_HPO_TOKENIZER:-$HOME/masters/sallm/tokenizer/sallm_bpe_tokenizer}"
fi

IFS=$'\t' read -r lr rank alpha dropout warmup < <(
  "$python_bin" "$repo/scripts/hpo_protocol.py" resolve \
    --registry "$registry" --candidate "$candidate" --tsv
)

root="$scratch/masters/sallm"
export SALLM_HPO_REGISTRY="$registry"
export SALLM_HPO_MODEL_ID="$model_id"
export SALLM_HPO_STAGE="$stage"
export SALLM_HPO_CANDIDATE="$candidate"
export SALLM_HPO_SEED="$seed"
export SALLM_HPO_DATA_SEED="$seed"
export SALLM_HPO_LEARNING_RATE="$lr"
export SALLM_HPO_LORA_RANK="$rank"
export SALLM_HPO_LORA_ALPHA="$alpha"
export SALLM_HPO_LORA_DROPOUT="$dropout"
export SALLM_HPO_WARMUP_RATIO="$warmup"
export SALLM_HPO_RUN_ID="${SALLM_HPO_RUN_ID_OVERRIDE:-hpo-${model_id}-${family}-${stage}-${candidate}-seed${seed}}"
export SALLM_HPO_OUTPUT_DIR="${SALLM_HPO_OUTPUT_DIR_OVERRIDE:-$root/checkpoints/adapter_hpo_v3/$model_id/$family/$stage/$candidate/seed_${seed}}"
export SALLM_HPO_LOGGING_DIR="${SALLM_HPO_LOGGING_DIR_OVERRIDE:-$root/logs/adapter_hpo_v3/$model_id/$family/$stage/$candidate/seed_${seed}}"
export SALLM_HPO_ENTRYPOINT=scripts/run_validation_hpo_trial.sh

trial=0
[[ "$candidate" =~ ^a([0-2])$ ]] && trial="${BASH_REMATCH[1]}"
exec bash "$repo/scripts/run_pure_gdn_validation_trial.sh" "$family" "$trial"
