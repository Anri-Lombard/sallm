#!/bin/bash

setup_sallm_cluster_env() {
  export PROJECT_ROOT="${PROJECT_ROOT:-$HOME/masters/sallm}"
  export SCRATCH="${SCRATCH:-${PROJECT_SCRATCH:-$HOME/scratch}}"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH/.cache/uv}"
  export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$SCRATCH/.cache/pip}"
}
