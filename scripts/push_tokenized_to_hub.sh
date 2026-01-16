#!/bin/bash
#SBATCH --account=maths
#SBATCH --partition=ada
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-user=LMBANR001@myuct.ac.za
#SBATCH --mail-type=FAIL,END
#SBATCH --job-name=push-tokenized-hub

export SCRATCH="/scratch/lmbanr001"
export HOME="/home/lmbanr001"
export HF_HOME="$SCRATCH/hf"
export HF_TOKEN="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"

module load python/miniconda3-py3.12
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate sallm-uv
set -u

export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/masters/sallm"
source .venv/bin/activate

echo "Starting tokenized dataset push at $(date)"
python scripts/push_datasets_to_hub.py \
    --mode tokenized \
    --data-dir "$SCRATCH/masters/sallm/data/sallm_dataset_tokenized" \
    --repo-id "anrilombard/mzansi-text-tokenized"

echo "Completed at $(date)"
