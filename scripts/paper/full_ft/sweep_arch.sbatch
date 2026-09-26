#!/bin/bash
#SBATCH --job-name=fft-t2x
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --qos=l40sfree
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/scratch/lmbanr001/masters/sallm/results/equal_recipe_fullft_t2x_pilot_20260925/logs/%x-%j.out
# usage: sbatch --job-name=fft-<arch> --export=ALL,ARCH=<mzansilm|mamba2|xlstm|gdn> sweep_arch.sbatch
exec bash /scratch/lmbanr001/masters/sallm/results/equal_recipe_fullft_t2x_pilot_20260925/scripts/sweep_arch.sh
