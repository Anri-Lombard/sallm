#!/bin/bash
# Local (Mac): stage this directory plus a frozen copy of the sallm library (src/ + tokenizer/) and rsync it to
# HEX $R/code. Writes code/sallm/SNAPSHOT_MANIFEST.sha256 and code/CODE.sha256, which every lane re-checks.
set -euo pipefail
repo="$(cd "$(dirname "$0")/../../.." && pwd)"
stage="${STAGE:-$(mktemp -d)}"
R=/scratch/lmbanr001/masters/sallm/results/fft_rollout_20260926
rm -rf "$stage/code"; mkdir -p "$stage/code/sallm"
rsync -a --exclude __pycache__ --exclude '*.pyc' "$repo/scripts/paper/fft_rollout/" "$stage/code/fft_rollout/"
rsync -a --exclude __pycache__ --exclude '*.pyc' --exclude '.DS_Store' "$repo/src" "$repo/tokenizer" "$stage/code/sallm/"
git -C "$repo" rev-parse HEAD > "$stage/code/sallm/SOURCE_COMMIT.txt"
git -C "$repo" status --porcelain -- src tokenizer > "$stage/code/sallm/SOURCE_DIRTY.txt"
(cd "$stage/code/sallm" && find . -type f ! -name SNAPSHOT_MANIFEST.sha256 | LC_ALL=C sort | xargs shasum -a 256 > SNAPSHOT_MANIFEST.sha256)
(cd "$stage/code" && find fft_rollout -type f | LC_ALL=C sort | xargs shasum -a 256 > CODE.sha256 && shasum -a 256 sallm/SNAPSHOT_MANIFEST.sha256 >> CODE.sha256)
rsync -a --delete "$stage/code/" "hex:$R/code/"
echo "synced $(wc -l < "$stage/code/CODE.sha256") files; CODE.sha256 $(shasum -a 256 "$stage/code/CODE.sha256" | cut -c1-12)"
