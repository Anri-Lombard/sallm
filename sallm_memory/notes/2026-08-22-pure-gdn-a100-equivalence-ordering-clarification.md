# Pure-GDN A100 equivalence ordering clarification, 2026-08-22

Preregistered at 20:41 SAST before the dependency-gated A100-80GB replacement
canary starts and without inspecting any canary prediction, score, or metric.

The user's authorization permits A100-40GB and A100-80GB jobs to overlap only
after the prospective validation-only hardware-equivalence gate passes. The
paired canaries must therefore run sequentially while any owned A100-40GB job
remains active.

A100-80GB canary `1257518` started automatically while AfriHG b4 `1253374`
was still running on A100-40GB and completed before cancellation took effect.
Its result and manifest are preserved but quarantined from the equivalence
decision. This quarantine is based only on the observed Slurm overlap; the
result payload was not inspected.

The valid sequence is:

1. Preserve b4 `1253374` unchanged and run A100-40GB reference `1257517`.
2. Start a fresh A100-80GB canary only after b4 has ended and reference
   `1257517` has completed successfully.
3. Compare only those two sequential artifacts with the frozen thresholds in
   the original amendment.

Replacement job `1257520` enforces step 2 with Slurm dependencies
`afterany:1253374,afterok:1257517` and writes to a new output prefix. No
held-out split may be loaded or inspected. All recipes, candidates,
confirmations, metrics, and selection rules remain unchanged.
