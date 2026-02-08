#!/bin/bash

load_hf_token() {
  if [[ -n "${HF_TOKEN:-}" ]]; then
    export HF_TOKEN
    return 0
  fi

  local token_file="${HF_TOKEN_FILE:-$HOME/.huggingface/token}"
  if [[ -f "$token_file" ]]; then
    HF_TOKEN="$(<"$token_file")"
    export HF_TOKEN
    return 0
  fi

  return 1
}

require_hf_token() {
  if load_hf_token; then
    return 0
  fi

  echo "ERROR: Hugging Face token not found." >&2
  echo "Set HF_TOKEN or create \$HOME/.huggingface/token." >&2
  return 1
}
