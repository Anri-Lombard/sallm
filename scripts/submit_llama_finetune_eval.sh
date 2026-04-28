#!/bin/bash
# Submit llama LoRA and full-finetune jobs, then dependent eval jobs.

set -u
set -o pipefail

cd ~/masters/sallm

SCRATCH_ROOT="${SCRATCH:-/scratch/lmbanr001}"

# Mapping of finetune task names to eval task names.
declare -A TASK_MAP=(
    ["ner"]="masakhaner"
    ["news"]="masakhanews"
    ["pos"]="masakhapos"
    ["sib"]="sib"
    ["injongointent"]="injongointent"
    ["t2x"]="t2x"
    ["afrihg"]="afrihg"
)

extract_top_level_value() {
    local key="$1"
    local path="$2"
    grep -E "^  ${key}:" "$path" | head -n 1 | cut -d: -f2- | xargs
}

sanitize_scalar() {
    local value="$1"
    value="${value//\"/}"
    value="${value//\$\{oc.env:SCRATCH\}/$SCRATCH_ROOT}"
    echo "$value"
}

submit_eval_if_present() {
    local dep_job="$1"
    local eval_cfg="$2"
    local checkpoint="$3"
    local eval_output="$4"
    local eval_name="$5"

    local eval_cfg_path="src/conf/eval/${eval_cfg}.yaml"
    if [ ! -f "$eval_cfg_path" ]; then
        echo "  -> No eval config found at $eval_cfg_path"
        return
    fi

    local result
    result=$(sbatch \
        --dependency=afterok:"$dep_job" \
        scripts/launch_evaluation.sh \
        "eval/${eval_cfg}" \
        "eval_model.checkpoint=${checkpoint}" \
        "evaluation.output_dir=${eval_output}" \
        "wandb.name=${eval_name}" 2>&1)
    local job_id
    job_id=$(echo "$result" | grep -oP '(?<=Submitted batch job )\d+')
    if [ -n "$job_id" ]; then
        echo "  -> Eval job: $job_id (depends on $dep_job)"
    else
        echo "  -> WARNING: Failed to submit eval job: $result"
    fi
}

mapfile -t CONFIGS < <(
    find src/conf/finetune -maxdepth 1 -name 'llama_*.yaml' -type f \
        -exec basename {} .yaml \; | sort
)

echo "Submitting llama LoRA + full-finetune jobs with dependent evals..."
echo "==============================================================="
echo "Found ${#CONFIGS[@]} llama finetune configs."

for cfg in "${CONFIGS[@]}"; do
    task=""
    lang=""

    if [[ "$cfg" =~ ^llama_sa_general_(.+)$ ]]; then
        task="sa_general"
        lang="${BASH_REMATCH[1]}"
    elif [[ "$cfg" =~ ^llama_([^_]+)_(.+)$ ]]; then
        task="${BASH_REMATCH[1]}"
        lang="${BASH_REMATCH[2]}"
    else
        echo "WARNING: Could not parse config name '$cfg', skipping"
        continue
    fi

    ft_cfg_path="src/conf/finetune/${cfg}.yaml"
    output_dir_raw=$(extract_top_level_value "output_dir" "$ft_cfg_path")
    logging_dir_raw=$(extract_top_level_value "logging_dir" "$ft_cfg_path")
    run_name_raw=$(extract_top_level_value "run_name" "$ft_cfg_path")
    wandb_name_raw=$(extract_top_level_value "name" "$ft_cfg_path")

    output_dir=$(sanitize_scalar "$output_dir_raw")
    logging_dir=$(sanitize_scalar "$logging_dir_raw")
    run_name=$(sanitize_scalar "$run_name_raw")
    wandb_name=$(sanitize_scalar "$wandb_name_raw")

    if [ -z "$output_dir" ] || [ -z "$logging_dir" ]; then
        echo "WARNING: Missing output/logging dir in $ft_cfg_path, skipping"
        continue
    fi

    eval_task="${TASK_MAP[$task]:-}"
    eval_cfg=""
    if [ -n "$eval_task" ]; then
        eval_cfg="run_llama_${eval_task}_${lang}"
    fi

    # LoRA run
    lora_result=$(sbatch scripts/launch_finetune.sh "finetune/${cfg}" 2>&1)
    lora_job_id=$(echo "$lora_result" | grep -oP '(?<=Submitted batch job )\d+')
    if [ -z "$lora_job_id" ]; then
        echo "ERROR: Failed to submit LoRA finetune for $cfg"
        echo "  Result: $lora_result"
        continue
    fi
    echo "[$cfg][lora] Finetune job: $lora_job_id"

    if [ -n "$eval_cfg" ]; then
        submit_eval_if_present \
            "$lora_job_id" \
            "$eval_cfg" \
            "${output_dir}/final_merged_model" \
            "${SCRATCH_ROOT}/masters/sallm/results/eval/${cfg}" \
            "eval-${cfg//_/-}"
    else
        echo "  -> No eval mapping for task '$task', skipping eval"
    fi

    # Full-finetune run
    full_output_dir="${output_dir}_fullft"
    full_logging_dir="${logging_dir}_fullft"
    full_run_name="${run_name}-fullft"
    full_wandb_name="${wandb_name}-fullft"

    full_result=$(sbatch \
        scripts/launch_finetune.sh \
        "finetune/${cfg}" \
        "peft.method=none" \
        "training.output_dir=${full_output_dir}" \
        "training.logging_dir=${full_logging_dir}" \
        "training.run_name=${full_run_name}" \
        "wandb.name=${full_wandb_name}" 2>&1)
    full_job_id=$(echo "$full_result" | grep -oP '(?<=Submitted batch job )\d+')
    if [ -z "$full_job_id" ]; then
        echo "ERROR: Failed to submit full-finetune for $cfg"
        echo "  Result: $full_result"
        continue
    fi
    echo "[$cfg][full] Finetune job: $full_job_id"

    if [ -n "$eval_cfg" ]; then
        submit_eval_if_present \
            "$full_job_id" \
            "$eval_cfg" \
            "${full_output_dir}/final_merged_model" \
            "${SCRATCH_ROOT}/masters/sallm/results/eval/${cfg}_fullft" \
            "eval-${cfg//_/-}-fullft"
    else
        echo "  -> No eval mapping for task '$task', skipping eval"
    fi
done

echo ""
echo "Done! Check jobs with: squeue -u \$USER"
