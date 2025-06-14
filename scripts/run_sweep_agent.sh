#!/bin/bash
#SBATCH --job-name=sallm-sweep-agent
#SBATCH --gpus-per-node=8
#...

# 1. Initialize sweep: wandb sweep configs/sweeps/bayesian_sweep.yaml
# 2. Get SWEEP_ID from output and place it here
export WANDB_SWEEP_ID="<REPLACE_WITH_YOUR_SWEEP_ID>"

# This command will be controlled by the W&B agent
accelerate launch $(which wandb) agent $WANDB_SWEEP_ID