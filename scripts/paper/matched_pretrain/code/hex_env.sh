# sourced by the HEX sbatch scripts
R=/scratch/lmbanr001/masters/sallm/results/matched_pretrain_20260926
PY=/home/lmbanr001/masters/sallm/.venv/bin   # one venv for all four (torch 2.9.1, transformers 4.57.3, fla 0.5.1, mamba-ssm 2.3.2.post1, xlstm 2.0.5)
export HF_HOME=/scratch/lmbanr001/hf HF_HUB_DISABLE_XET=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=42
export PYTHONPATH=$R/pydeps OMP_NUM_THREADS=4 PYTORCH_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=${TMPDIR:-/tmp}/mp-triton-$SLURM_JOB_ID TORCHINDUCTOR_CACHE_DIR=${TMPDIR:-/tmp}/mp-inductor-$SLURM_JOB_ID
export WANDB_DIR=$R/runs WANDB_CACHE_DIR=$R/.wandb_cache WANDB_DATA_DIR=$R/.wandb_data
export MP_VAL_PARQUET=$R/data/mzansi-text-validation-340265320.parquet
export MP_TOKENIZER=/home/lmbanr001/masters/sallm/tokenizer/sallm_bpe_tokenizer
