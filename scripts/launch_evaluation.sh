#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=4
#SBATCH --job-name="sallm-eval"
#SBATCH --mail-user=LMBANR001@myuct.ac.za
#SBATCH --mail-type=FAIL,END

# ----------------------------------------------------------------------------
# Launch evaluation on a SLURM cluster
#
# Purpose
#   Dispatch an evaluation job that selects an evaluation YAML from
#   src/conf/eval and runs the project's evaluation entrypoint.
#
# Usage
#   sbatch launch_evaluation.sh <eval_config_name>
#
# Example
#   sbatch launch_evaluation.sh run_mamba_belebele_zul
#
# Notes
#   - The script attempts to detect the repository root using SALLM_REPO_ROOT,
#     SLURM_SUBMIT_DIR, or `git rev-parse`. Set SALLM_REPO_ROOT on the compute
#     node if the repo is not checked out under the same path.
#   - Requires a conda env named `sallm-ner` or adjust the activation logic.
# ----------------------------------------------------------------------------

CONFIG_NAME="$1"

# Detect repository root robustly to locate eval configs
if [ -n "$SALLM_REPO_ROOT" ]; then
    REPO_ROOT="$SALLM_REPO_ROOT"
elif [ -n "$SLURM_SUBMIT_DIR" ] && [ -d "$SLURM_SUBMIT_DIR/src/conf/eval" ]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    if command -v git >/dev/null 2>&1 && git rev-parse --show-toplevel >/dev/null 2>&1; then
        REPO_ROOT=$(git rev-parse --show-toplevel)
    else
        REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
    fi
fi

CONF_DIR="$REPO_ROOT/src/conf/eval"

if [ -z "$CONFIG_NAME" ]; then
    echo "Usage: sbatch $0 <eval_config_name>"
    echo "Examples:"
    echo "  sbatch $0 run_mamba_belebele_zul"
    echo "  sbatch $0 mamba_belebele_zul"
    echo ""
    echo "Repository root (detected): $REPO_ROOT"
    echo "Looking for eval configs in: $CONF_DIR"
    echo "Available evaluation configs:"
    if [ -d "$CONF_DIR" ]; then
        for f in "$CONF_DIR"/*.yaml; do
            [ -e "$f" ] || continue
            echo "  - $(basename "$f" .yaml)"
        done
    else
        echo "  (no eval configs found at $CONF_DIR)"
        if [ -n "$SLURM_SUBMIT_DIR" ]; then
            echo "  SLURM_SUBMIT_DIR is set to: $SLURM_SUBMIT_DIR"
        fi
        echo "  When running under Slurm, set SALLM_REPO_ROOT to the repo path on the compute node or ensure SLURM_SUBMIT_DIR points to it."
    fi
    exit 1
fi

export HYDRA_FULL_ERROR=1

export SCRATCH="/scratch/lmbanr001"
export HOME="/home/lmbanr001"

module load python/miniconda3-py3.12
CONDA_BASE=$(conda info --base 2>/dev/null || echo "")
if [ -n "$CONDA_BASE" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate sallm-ner
else
    echo "Warning: conda not found in PATH. Activate the correct environment if Python imports fail."
fi

CAND="$CONFIG_NAME"
if [[ "$CAND" != run_* ]]; then
    CAND_RUN="run_$CAND"
else
    CAND_RUN="$CAND"
fi

# Fallback detection for CONF_DIR
if [ ! -d "$CONF_DIR" ]; then
    if [ -n "$SLURM_SUBMIT_DIR" ] && [ -d "$SLURM_SUBMIT_DIR/src/conf/eval" ]; then
        CONF_DIR="$SLURM_SUBMIT_DIR/src/conf/eval"
        REPO_ROOT="$SLURM_SUBMIT_DIR"
    elif command -v git >/dev/null 2>&1 && git rev-parse --show-toplevel >/dev/null 2>&1; then
        REPO_ROOT=$(git rev-parse --show-toplevel)
        CONF_DIR="$REPO_ROOT/src/conf/eval"
    fi
fi

if [ -f "$CONF_DIR/$CAND_RUN.yaml" ]; then
    EVAL_CONFIG="$CAND_RUN"
elif [ -f "$CONF_DIR/$CONFIG_NAME.yaml" ]; then
    EVAL_CONFIG="$CONFIG_NAME"
else
    echo "Could not find evaluation config matching '$CONFIG_NAME'. Repository root: $REPO_ROOT"
    echo "Looking for eval configs in: $CONF_DIR"
    echo "Available evaluation configs:"
    if [ -d "$CONF_DIR" ]; then
        for f in "$CONF_DIR"/*.yaml; do
            [ -e "$f" ] || continue
            echo "  - $(basename "$f" .yaml)"
        done
    else
        echo "  (no eval configs found at $CONF_DIR)"
        echo "Suggestions:"
        echo "  - Ensure the repository is available on the compute node and SLURM_SUBMIT_DIR points to it."
        echo "  - Use: sbatch --chdir=/path/to/repo $0 <config> or set SALLM_REPO_ROOT to the repo path."
    fi
    exit 2
fi

echo "Starting evaluation using evaluation=$EVAL_CONFIG (repo root: $REPO_ROOT)"
python -m sallm.main --config-path "$REPO_ROOT/src/conf/eval" --config-name "$EVAL_CONFIG"
