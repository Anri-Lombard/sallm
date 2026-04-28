#!/bin/bash

set_sallm_cluster_env() {
  local user_name="${USER:-$(id -un)}"

  : "${SALLM_HOME_DIR:=${HOME}}"
  : "${SALLM_SCRATCH_DIR:=${SCRATCH:-/scratch/${user_name}}}"
  : "${SALLM_REPO_DIR:=${SALLM_HOME_DIR}/masters/sallm}"
  : "${SALLM_SLURM_USER:=${user_name}}"

  export SALLM_HOME_DIR
  export SALLM_SCRATCH_DIR
  export SALLM_REPO_DIR
  export SALLM_SLURM_USER

  export SCRATCH="${SALLM_SCRATCH_DIR}"
}

set_sallm_sbatch_options() {
  SALLM_SBATCH_OPTIONS=()

  if [[ -n "${SALLM_SLURM_ACCOUNT:-}" ]]; then
    SALLM_SBATCH_OPTIONS+=(--account="$SALLM_SLURM_ACCOUNT")
  fi

  if [[ -n "${SALLM_SLURM_MAIL_USER:-}" ]]; then
    SALLM_SBATCH_OPTIONS+=(--mail-user="$SALLM_SLURM_MAIL_USER")
  fi
}
