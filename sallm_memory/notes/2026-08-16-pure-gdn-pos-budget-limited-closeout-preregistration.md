# Pure-GDN POS budget-limited close-out preregistration — 2026-08-16

Preregistered at 22:18 SAST after the user approved the deadline-safe scope
change and before cancelling any live or pending POS job. Thursday 20 August
is the fixed delivery deadline. This amendment is based on measured runtime
and queue capacity, not held-out data or a comparison among Stage-B recipes.
No held-out split has been loaded or inspected.

## Stage-B quarantine

- Stop the remaining enhanced Stage-B POS program. Pending jobs
  `1241580/1241581` have not started and may be cancelled immediately.
- Running b2 job `1241578` may finish only its already-active checkpoint-283
  exact callback, then must be stopped at the first safe boundary. It may not
  begin a scientifically eligible later checkpoint under this amendment.
- Quarantine every POS Stage-B artifact uniformly: b0/b1/b2 partial or complete
  validation metrics, cancelled/recovery job records, checkpoints, logs, and
  failure provenance remain preserved but may not affect recipe selection,
  confirmation selection, reporting as trusted HPO, or held-out evaluation.
- Never launch b3--b7. No Stage-B recipe is promoted regardless of its metric.

## Reduced Stage-A close-out

- The eligible selection pool is restricted to the three terminal-valid,
  preregistered Stage-A seed-42 candidates completed before Stage B:
  a0 `0.8211477429884227`, a1 `0.8474682174611892`, and a2
  `0.8594919779162872` exact full-validation token accuracy.
- The fixed finalists are a2 and a1, the top two Stage-A candidates under the
  existing validation metric. Their recipes, data, prompts, evaluator,
  checkpoint cadence, early stopping, and seed-42 artifacts remain unchanged.
- To fit the deadline while retaining an independent-seed check, run exactly
  one additional validation-only seed for each finalist: seed 13, the first
  non-selection seed in the pre-existing `[13, 42, 87]` registry. Seed 87 is
  omitted for both finalists uniformly and may not be run later to break a
  disappointing or ambiguous outcome.
- Freeze the POS winner by the arithmetic mean of each finalist's terminal
  exact full-validation token accuracy across seeds 42 and 13. If the means
  tie exactly, apply the existing deterministic lower-learning-rate then
  earlier-checkpoint tie-break. Held-out data remain untouched until the
  global winner-freeze gate.

This produces a transparent budget-limited coarse POS search, not the planned
11-candidate enhanced search. The paper and artifact manifest must disclose
that limitation. Original artifacts are never deleted or overwritten.

## Execution — 22:19--22:53 SAST

- Pending exact-resume jobs `1241580/1241581` were cancelled at `22:19:17`
  with elapsed `00:00:00`; neither acquired a node nor loaded model/data.
- B2 `1241578` completed its already-active step-283 exact callback. An
  automated fail-closed guard required the validation JSON, its SHA-256 file,
  and complete checkpoint adapter/optimizer/scheduler/RNG/trainer state. The
  JSON hash verified before cancellation at `22:52:12`; Slurm records
  `CANCELLED by 733329384`, elapsed `02:36:12`, exit `0:0`. The artifact and
  checkpoint are preserved but quarantined under this protocol.
- Dry runs resolved the two eligible reduced confirmations exactly: a2 seed 13
  at LR `1.5e-4`, rank/alpha `16/32`, dropout `0.05`, warmup `0.03`; and a1
  seed 13 at LR `8e-5` with the same remaining recipe. Both use full exact POS
  validation, A100-40GB `gpu:ampere:1`, 24 hours, eight CPUs, and isolated
  confirmation output paths.
- Confirmation jobs are a2 seed 13 `1241719` and a1 seed 13 `1241720`.
  Job `1241719` acquired the A100 released by b2 at `22:52:14`, two seconds
  after cancellation, verified all `695` immutable source/config files, passed
  the GatedDeltaNet fast-kernel gate, and wrote execution manifest SHA-256
  `b00cc6252d5e5ecf9b9bef84e8dc1557b8729b77aa36b30bad83f084b0dc144e`.
  Job `1241720` is Resources-pending with current Slurm projection
  `2026-08-17 11:58:10 SAST`.
