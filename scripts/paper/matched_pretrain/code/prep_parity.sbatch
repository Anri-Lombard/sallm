#!/bin/bash
#SBATCH --job-name=mp-prep-parity
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --qos=l40sfree
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --exclude=srvrocgpu009
#SBATCH --output=/scratch/lmbanr001/masters/sallm/results/matched_pretrain_20260926/logs/%x-%j.out
# Bench data + Liger (target dir, no deps; shared venvs untouched) + parity/memory probe for all four on one L40S.
set -eo pipefail
source /scratch/lmbanr001/masters/sallm/results/matched_pretrain_20260926/code/hex_env.sh
echo "job=$SLURM_JOB_ID node=$(hostname) start=$(date -Is)"; nvidia-smi
cd $R/code && sha256sum -c CODE.sha256
[ -d $R/pydeps/liger_kernel ] || UV_CACHE_DIR=/scratch/lmbanr001/.cache/uv $HOME/.local/bin/uv pip install --python $PY/python --target $R/pydeps --no-deps liger-kernel==0.8.3
[ -f $R/data/bench3/train.bin ] || $PY/python pack.py --repo uctnlp/mzansi-text-tokenized --rev 7dab1ebf1e1f00ea967a3845aea0acde6cfc0792 --files 3 --out $R/data/bench3
[ -f $MP_VAL_PARQUET ] || cp "$($PY/python -c "from huggingface_hub import hf_hub_download as d; print(d('uctnlp/mzansi-text','data/validation-00000-of-00001.parquet',repo_type='dataset',revision='340265320aa4282c0ffc4f2a1b0bc0fc5ba3e83b'))")" $MP_VAL_PARQUET
sha256sum $R/data/bench3/train.bin $MP_VAL_PARQUET
export HF_HUB_OFFLINE=1
mkdir -p $R/parity
for a in ${ARCHS:-mzansilm mamba2 gdn xlstm}; do
  echo "== parity $a $(date -Is)"
  extra=(); [ $a = xlstm ] && extra=(--xlstm-kernel ${XLSTM_KERNEL:-chunkwise--triton_xl_chunk})
  $PY/python parity.py --config ../configs/${a}_cosine.json --data $R/data/bench3 --out $R/parity/${a}_$SLURM_JOB_ID.json "${extra[@]}" 2>&1 | grep -v -i warn || echo "FAILED $a"
done
echo "end=$(date -Is)"
