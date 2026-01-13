#!/bin/bash
#SBATCH --account=maths
#SBATCH --partition=ada
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --mail-user=LMBANR001@myuct.ac.za
#SBATCH --mail-type=FAIL,END
#SBATCH --job-name=push-raw-hub

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

echo "Starting raw dataset push at $(date)"
python scripts/push_datasets_to_hub.py \
    --mode raw \
    --data-dir "$SCRATCH/masters/sallm/data/filtered_data" \
    --repo-id "anrilombard/mzansi-text"

echo "Completed at $(date)"
