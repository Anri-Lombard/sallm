R=/scratch/lmbanr001/masters/sallm/results/mamba_eos_fix_20260924
FIX=$R/bases/mamba2_eosfix
PY=/home/lmbanr001/masters/sallm/.venv/bin/python
(cd $FIX && sha256sum -c $R/bases/mamba2_eosfix.FILES.sha256 --quiet)
export HF_HOME=/scratch/lmbanr001/hf-cache HF_DATASETS_CACHE=/scratch/lmbanr001/hf-cache/datasets HF_HUB_CACHE=/scratch/lmbanr001/hf-cache/hub
export HF_HUB_DISABLE_XET=1 WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=42
export FLA_DISABLE_BACKEND_DISPATCH=1 MAMBA_SCAN_IMPL=cuda
export PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/mamba-eosfix-pycache-${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name --format=csv,noheader
date -Is
