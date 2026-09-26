#!/bin/bash
# Mac: archive every selected checkpoint of a rollout run from HEX to Kombuys and verify it there.
#   bash archive_to_kombuys.sh NAME [--delete-hex]
# Selected = keep/<family>/<run>_e<epoch> with a <run>_e<epoch>.selected marker (written in-job, with the
# <run>_e<epoch>.sha256 manifest). Kombuys copy: /scratch/alombard/sallm/results/fft_rollout_20260926/<NAME>/keep/.
# A checkpoint counts as archived once `sha256sum -c` passes on Kombuys (marker <run>_e<epoch>.archived there).
# --delete-hex then removes the verified HEX copy (skip it while the optional cross_eval stage still needs them).
# The stream passes through the HEX login node (the only route from the Mac); ~0.5 GB per checkpoint.
set -euo pipefail
NAME="${1:?NAME}"; DELETE="${2:-}"
R=/scratch/lmbanr001/masters/sallm/results/fft_rollout_20260926/runs/$NAME/keep
K=/scratch/alombard/sallm/results/fft_rollout_20260926/$NAME/keep
q() { grep -v -E '^\*\*|^ *\*|AUP|agree|^$' || true; }
for sel in $(ssh hex "cd $R && ls */*.selected 2>/dev/null" 2> /dev/null | q); do
  d="${sel%.selected}"; fam="${d%%/*}"; base="${d#*/}"
  if ! ssh jbuys "test -f $K/$d.archived"; then
    ssh hex "cd $R && tar cf - $d $d.sha256 $d.selected" 2> /dev/null \
      | ssh jbuys "mkdir -p $K && cd $K && tar xf - && cd $fam && nice sha256sum -c --quiet $base.sha256 && touch $base.archived"
    echo "archived $NAME/$d"
  fi
  if [[ "$DELETE" == --delete-hex ]] && ssh jbuys "test -f $K/$d.archived"; then
    ssh hex "rm -rf $R/$d && touch $R/$d.on_kombuys" 2> /dev/null && echo "deleted HEX copy $d"
  fi
done
ssh jbuys "du -sh $K 2>/dev/null; df -h /scratch | tail -1"
