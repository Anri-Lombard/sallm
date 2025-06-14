#!/bin/bash
#SBATCH --job-name=sallm-final-train
#SBATCH --gpus-per-node=8
#...

# This script runs a single, final training job.
# The YAML file should contain the BEST hyperparameters found from the sweep.
# The --resume_from_checkpoint flag ensures we pick up where we left off if the job is preempted.
accelerate launch src/main/sallm/main.py \
    --config_path=configs/llama_125m_final.yaml \
    --resume_from_checkpoint