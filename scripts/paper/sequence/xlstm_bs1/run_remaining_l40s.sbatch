#!/bin/bash
#SBATCH --job-name=xlstm-bs1-rest-l40s
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --qos=l40sfree
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=srvrocgpu009
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/lmbanr001/masters/sallm/results/xlstm_bs1_other_20260925/logs/%x-%j.out
#SBATCH --error=/scratch/lmbanr001/masters/sallm/results/xlstm_bs1_other_20260925/logs/%x-%j.err
# Runs the remaining padding-free xLSTM units concurrently on one GPU (each process is batch size 1).
nvidia-smi -L
bash /scratch/lmbanr001/masters/sallm_snapshots/generation-protocol-v4-xlstm-bs1-20260925/run_xlstm_bs1.sbatch 6 &
for w in general_ner base_ner base_afrimgsm general_afrimgsm; do
  bash /scratch/lmbanr001/masters/sallm_snapshots/xlstm-bs1-other-20260925/run_other_bs1.sh $w &
done
wait
