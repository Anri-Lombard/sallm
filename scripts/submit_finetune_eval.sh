#!/bin/bash
# Submit finetune jobs then eval jobs with dependencies

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/env.sh"
set_sallm_cluster_env
set_sallm_sbatch_options

cd "$SALLM_REPO_DIR"

# Mapping of finetune task names to eval task names
declare -A TASK_MAP=(
    ["ner"]="masakhaner"
    ["news"]="masakhanews"
    ["pos"]="masakhapos"
    ["sib"]="sib"
    ["injongointent"]="injongointent"
    ["t2x"]="t2x"
    ["afrihg"]="afrihg"
)

# Discover all Mamba finetune configs automatically.
mapfile -t CONFIGS < <(
    find src/conf/finetune -maxdepth 1 -name 'mamba_*.yaml' -type f \
        -exec basename {} .yaml \; | sort
)

echo "Submitting finetune and eval jobs..."
echo "======================================="
echo "Found ${#CONFIGS[@]} Mamba finetune configs."

for cfg in "${CONFIGS[@]}"; do
    task=""
    lang=""
    if [[ "$cfg" =~ ^mamba_sa_general_(.+)$ ]]; then
        task="sa_general"
        lang="${BASH_REMATCH[1]}"
    elif [[ "$cfg" =~ ^mamba_([^_]+)_(.+)$ ]]; then
        task="${BASH_REMATCH[1]}"
        lang="${BASH_REMATCH[2]}"
    else
        echo "WARNING: Could not parse config name '$cfg', skipping"
        continue
    fi

    # Get eval task name
    eval_task="${TASK_MAP[$task]}"
    if [ -z "$eval_task" ]; then
        echo "Warning: No eval mapping for task $task, skipping eval"
        eval_task=""
    fi

    eval_cfg="run_mamba_${eval_task}_${lang}"

    # Check if eval config exists
    eval_config_path="src/conf/eval/${eval_cfg}.yaml"

    # Submit finetune job
    ft_result=$(sbatch "${SALLM_SBATCH_OPTIONS[@]}" scripts/launch_finetune.sh "finetune/$cfg" 2>&1)
    ft_job_id=$(echo "$ft_result" | grep -oP '(?<=Submitted batch job )\d+')

    if [ -z "$ft_job_id" ]; then
        echo "ERROR: Failed to submit finetune job for $cfg"
        echo "  Result: $ft_result"
        continue
    fi

    echo "[$cfg] Finetune job: $ft_job_id"

    # Submit eval job with dependency if eval config exists
    if [ -n "$eval_task" ] && [ -f "$eval_config_path" ]; then
        eval_result=$(sbatch "${SALLM_SBATCH_OPTIONS[@]}" --dependency=afterok:$ft_job_id scripts/launch_evaluation.sh "eval/$eval_cfg" 2>&1)
        eval_job_id=$(echo "$eval_result" | grep -oP '(?<=Submitted batch job )\d+')

        if [ -n "$eval_job_id" ]; then
            echo "  -> Eval job: $eval_job_id (depends on $ft_job_id)"
        else
            echo "  -> WARNING: Failed to submit eval job: $eval_result"
        fi
    else
        echo "  -> No eval config found at $eval_config_path"
    fi
done

echo ""
echo "Done! Check jobs with: squeue -u \$USER"
