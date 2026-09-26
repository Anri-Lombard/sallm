#!/bin/bash

set -euo pipefail

runtime="${SALLM_RUNTIME_REPO:?HPO runtime repository is required}"
repo="${SALLM_REPO_DIR:?immutable gate source snapshot is required}"

module load python/miniconda3-py3.12
export UV_CACHE_DIR="${SCRATCH:-/scratch/lmbanr001}/.cache/uv"
cd "$runtime"
uv sync --frozen --inexact

exec bash "$repo/scripts/run_pos_runtime_equivalence_gate.sh"
