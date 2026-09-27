# Pure-GDN A100-80GB runtime-correction amendment, 2026-08-25

Preregistered at 21:40 SAST before any corrected hardware-gate metric or new
A100-80GB HPO run. The user authorized moving future work to the currently idle
A100-80GB pool. No held-out split may be loaded, inspected, or scored.

## Why one correction is admissible

The 25 August gate failed only because its two jobs did not use the current HPO
runtime. Both scientific outputs matched exactly, including predictions,
selected scores, and metrics. The A100-40GB manifest captured `pip 23.3.1` and
the later A100-80GB manifest captured `pip 26.2.1` from the shared scratch venv.
Current HPO jobs instead use `/home/lmbanr001/masters/sallm/.venv/bin/python`.
This is confirmed temporal environment drift, not score-driven evidence for a
new recipe, threshold, or hardware result.

This amendment supersedes only the earlier no-rerun decision for that
environment mismatch. The failed gate, jobs, logs, manifests, comparison
artifact, and SHA-256 provenance remain preserved and quarantined.

## Frozen corrected gate

Run the same validation-only POS full-prefix gate once on A100-40GB and then
once on A100-80GB. Both jobs must:

- use the immutable source snapshot
  `/home/lmbanr001/masters/sallm_snapshots/pure-gdn-a100-equivalence-20260822-38d367bf`;
- use the current HPO runtime at `/home/lmbanr001/masters/sallm`, synchronized
  immediately before execution with `uv sync --frozen --inexact`;
- use the canonical pure-GDN model and frozen POS b0 seed-42 checkpoint named
  in the 22 August amendment;
- keep BF16, no adapter merge, the same 12 validation cells, scorer, subset,
  thresholds, equality checks, and fail-closed comparator unchanged;
- request the exact account/QOS/GRES for each GPU family, 24 hours, eight CPUs,
  and `--chdir=$HOME/masters/sallm`.

The A100-40GB reference may start only after running jobs `1267877` and
`1267878` are terminal. The A100-80GB candidate may start only after the
reference exits successfully. There may be no cross-family overlap before the
comparison passes. Pending, never-started job `1269234` may be cancelled and
replaced unchanged after the gate to preserve the three-owned-job cap.

## Decision rule

If every original comparison check passes, future unchanged pure-GDN
validation HPO and confirmations may run on either A100 family within the
existing three-job cap. Pending and future work should prefer the idle
A100-80GB pool. If any check fails, preserve the corrected attempt and continue
on A100-40GB. Do not rerun again, relax a threshold, or consult held-out data.

Temporary permission for four concurrent jobs, if later granted, is a separate
capacity change and requires a prospective cap amendment before use.
