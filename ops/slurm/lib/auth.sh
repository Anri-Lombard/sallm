#!/bin/bash

load_hf_token() {
  if [[ -n "${HF_TOKEN:-}" ]]; then
    export HF_TOKEN
    return 0
  fi

  local token_home="${SALLM_HOME_DIR:-$HOME}"
  local token_file="${HF_TOKEN_FILE:-$token_home/.huggingface/token}"
  if [[ -f "$token_file" ]]; then
    HF_TOKEN="$(<"$token_file")"
    export HF_TOKEN
    return 0
  fi

  return 1
}
