#!/bin/bash
# Submit finetune jobs then eval jobs with dependencies

cd ~/masters/sallm

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

# Configs to submit (those we updated with HPO parameters)
CONFIGS=(
    # news
    "mamba_news_all"
    "mamba_news_eng"
    "mamba_news_xho"
    # sib
    "mamba_sib_all"
    "mamba_sib_afr"
    "mamba_sib_eng"
    "mamba_sib_nso"
    "mamba_sib_sot"
    "mamba_sib_xho"
    "mamba_sib_zul"
    # injongointent
    "mamba_injongointent_all"
    "mamba_injongointent_eng"
    "mamba_injongointent_sot"
    "mamba_injongointent_xho"
    "mamba_injongointent_zul"
    # ner
    "mamba_ner_all"
    "mamba_ner_tsn"
    "mamba_ner_xho"
    "mamba_ner_zul"
    # t2x
    "mamba_t2x_xho"
    # afrihg
    "mamba_afrihg_all"
    "mamba_afrihg_xho"
    "mamba_afrihg_zul"
)

echo "Submitting finetune and eval jobs..."
echo "======================================="

for cfg in "${CONFIGS[@]}"; do
    # Extract task and lang from config name (mamba_<task>_<lang>)
    task=$(echo "$cfg" | sed 's/mamba_\([^_]*\)_.*/\1/')
    lang=$(echo "$cfg" | sed 's/mamba_[^_]*_//')

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
    ft_result=$(sbatch scripts/launch_finetune.sh "finetune/$cfg" 2>&1)
    ft_job_id=$(echo "$ft_result" | grep -oP '(?<=Submitted batch job )\d+')

    if [ -z "$ft_job_id" ]; then
        echo "ERROR: Failed to submit finetune job for $cfg"
        echo "  Result: $ft_result"
        continue
    fi

    echo "[$cfg] Finetune job: $ft_job_id"

    # Submit eval job with dependency if eval config exists
    if [ -n "$eval_task" ] && [ -f "$eval_config_path" ]; then
        eval_result=$(sbatch --dependency=afterok:$ft_job_id scripts/launch_evaluation.sh "eval/$eval_cfg" 2>&1)
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
