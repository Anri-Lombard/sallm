#!/bin/bash
# Launch all xLSTM baseline evaluations
# Usage: ./scripts/launch_xlstm_baseline_eval.sh [--dry-run]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/lib/env.sh"
set_sallm_cluster_env
set_sallm_sbatch_options

EVAL_SCRIPT="$SCRIPT_DIR/launch_evaluation.sh"

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE ==="
fi

# All xLSTM baseline eval configs
CONFIGS=(
    # SIB (6)
    "run_xlstm_sib_afr"
    "run_xlstm_sib_eng"
    "run_xlstm_sib_nso"
    "run_xlstm_sib_sot"
    "run_xlstm_sib_xho"
    "run_xlstm_sib_zul"
    # MasakhaNews (2)
    "run_xlstm_masakhanews_eng"
    "run_xlstm_masakhanews_xho"
    # MasakhaNER (3)
    "run_xlstm_masakhaner_tsn"
    "run_xlstm_masakhaner_xho"
    "run_xlstm_masakhaner_zul"
    # MasakhaPOS (3)
    "run_xlstm_masakhapos_tsn"
    "run_xlstm_masakhapos_xho"
    "run_xlstm_masakhapos_zul"
    # AfriHG (2)
    "run_xlstm_afrihg_xho"
    "run_xlstm_afrihg_zul"
    # T2X (1)
    "run_xlstm_t2x_xho"
    # InjongoIntent (4)
    "run_xlstm_injongointent_eng"
    "run_xlstm_injongointent_sot"
    "run_xlstm_injongointent_xho"
    "run_xlstm_injongointent_zul"
    # Belebele (8)
    "run_xlstm_belebele_afr"
    "run_xlstm_belebele_eng"
    "run_xlstm_belebele_sot"
    "run_xlstm_belebele_ssw"
    "run_xlstm_belebele_tsn"
    "run_xlstm_belebele_tso"
    "run_xlstm_belebele_xho"
    "run_xlstm_belebele_zul"
    # Benchmark SA (3)
    "run_xlstm_afrixnli_sa"
    "run_xlstm_afrimmlu_sa"
    "run_xlstm_afrimgsm_sa"
)

echo "Submitting ${#CONFIGS[@]} xLSTM baseline evaluation jobs..."
echo ""

submitted=0
for config in "${CONFIGS[@]}"; do
    if $DRY_RUN; then
        printf "[DRY RUN] sbatch"
        printf " %q" "${SALLM_SBATCH_OPTIONS[@]}" "$EVAL_SCRIPT" "$config"
        printf "\n"
    else
        echo "Submitting: $config"
        sbatch "${SALLM_SBATCH_OPTIONS[@]}" "$EVAL_SCRIPT" "$config"
        ((submitted++))
        # Small delay to avoid overwhelming scheduler
        sleep 0.5
    fi
done

echo ""
if $DRY_RUN; then
    echo "Would submit ${#CONFIGS[@]} jobs (dry run)"
else
    echo "Submitted $submitted jobs"
    echo "Monitor with: squeue -u \$USER"
fi
