#!/bin/bash

setup_sallm_cluster_env() {
  local submit_dir="${SLURM_SUBMIT_DIR:-$(pwd)}"
  local scratch_default=""
  if [[ -n "${USER:-}" && -d "/scratch/${USER}" ]]; then
    scratch_default="/scratch/${USER}"
  else
    scratch_default="$HOME/scratch"
  fi
  export PROJECT_ROOT="${PROJECT_ROOT:-$submit_dir}"
  export SCRATCH="${SCRATCH:-${PROJECT_SCRATCH:-$scratch_default}}"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH/.cache/uv}"
  export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$SCRATCH/.cache/pip}"
}
