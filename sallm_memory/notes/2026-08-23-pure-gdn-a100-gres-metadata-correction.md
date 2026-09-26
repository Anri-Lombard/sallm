# Pure-GDN A100 gate GRES-metadata correction, 2026-08-23

## Final strictly serial chain submitted — 09:47 SAST

- Final isolated A100-80GB candidate `1258898` is dependency-pending on
  successful completion of A100-40GB b5 `1258485`. It requests exactly
  `nlpgroup80/a100/nlpgroup80`, `gpu:ampere80:1`, `24:00:00`, eight CPUs,
  and `--chdir=/home/lmbanr001/masters/sallm`. Its explicit process metadata
  is `SLURM_JOB_GRES=gpu:ampere80:1`, and its result prefix is the frozen
  isolated suffix `pos-a10080-gres-correction-r3`.
- CPU-only comparator `1258899` depends on successful candidate completion
  and uses the unchanged immutable verifier against corrected A100-40GB
  reference `1258452`. Mandatory A100-40GB b6 `1258900` depends on successful
  comparator completion. Slurm records the exact dependency chain
  `1258485 -> 1258898 -> 1258899 -> 1258900` and the expected requested GRES
  on both GPU jobs.
- B5 `1258485` is the only runnable owned GPU job. Candidate, verification,
  and b6 result paths were absent before submission. The hardware gate remains
  scientifically unpassed until `1258899` exits successfully; no A100-80GB
  HPO is eligible before then.

## Second race stopped; final replacement ordered after b5 — 09:45 SAST

- Replacement `1258494` and dependent verifier `1258495` were submitted as
  preregistered, but b4 released A100-40GB b5 `1258485` in the same scheduler
  cycle that released `1258494`. The two GPU jobs began at `09:40:45/46`.
  `1258494` was cancelled after exactly `00:00:30`; `1258495` never started.
  No replacement result content was inspected. Preserve both jobs and any
  partial files as an ordering-failure attempt.
- B5 `1258485` is now the only owned GPU job and is scientifically clean on
  A100-40GB. Freeze one final isolated A100-80GB replacement suffix `-r3`
  dependency-gated after successful b5 completion, followed by the unchanged
  fail-closed CPU comparator against reference `1258452`. This removes the
  scheduler race rather than relaxing any gate. No A100-80GB job may run
  before b5 ends.
- Mandatory A100-40GB b6 may be submitted dependency-gated after comparator
  success. This preserves the three-GPU-job cap and ensures b6 cannot start
  unless hardware equivalence passes.

## Scheduler-race quarantine and replacement — 09:40 SAST

- Corrected candidate `1258453` unexpectedly became runnable at `09:36:18`
  while b4 `1253374` was still running and completed `0:0` at `09:37:21`.
  An attempted dependency update occurred only after completion. No result,
  metric, prediction, or checkpoint content was inspected; only Slurm state
  and artifact paths/sizes were observed.
- Because this overlaps owned A100-40GB work before the gate passes,
  `1258453` is quarantined as an ordering failure and must not be compared or
  used. This decision is independent of its result.
- Freeze one replacement of only the A100-80GB candidate with a new isolated
  output suffix `-r2`, unchanged immutable snapshot, validation subset,
  scorer, model, adapter, BF16/runtime settings, explicit
  `SLURM_JOB_GRES=gpu:ampere80:1`, and unchanged comparator. It depends on
  successful b4 completion. A dependency-gated CPU comparator must compare
  reference `1258452` only with this replacement and exit nonzero on failure.
  Pending A100-40GB b5 job `1258485` must depend on successful comparator
  completion, so no cross-variant scientific work can overlap before a pass.

Preregistered at 05:35 SAST after the original paired comparator failed and
before any correction rerun. No held-out split was loaded, inspected, or
scored.

## Observed failure

- A100-40GB reference `1257517` completed `0:0` at `05:29:41` SAST. The
  original pair `1257517/1257520` was compared once with the frozen fail-closed
  verifier (SHA-256
  `64c8e42aeba1e32fed011e901cd2ebecf677190e75d628f984b046c979f81a89`).
- Verification artifact SHA-256 is
  `e0aa48c710dde2a52d6651c1365db4656c95f0420ab17e80e0c837cd3f42edc6`.
  It failed only `hardware_pair_match`: both benchmark artifacts recorded
  `slurm_job_gres=null` because HEX does not populate the assumed
  `SLURM_JOB_GRES` environment variable. Every other check passed, including
  exact predictions, counts, metrics, manifests, model/source/runtime
  identity, and zero maximum/mean selected-score difference. Runtime is not
  an acceptance condition.
- This is an instrumentation failure, not a passing hardware gate. A100-80GB
  remains unratified. Running b5 `1257792` must be cancelled and preserved as
  quarantined partial provenance rather than allowed to consume more compute
  for an ineligible result.

## Frozen correction

- Rerun the same A100-40GB benchmark once and the same A100-80GB benchmark
  once, sequentially, with new isolated result prefixes. Keep the immutable
  source snapshot, verifier, checkpoint, adapter, validation subset, BF16
  mode, scorer, thresholds, resources, and all other settings unchanged.
- The only correction is to set `SLURM_JOB_GRES` explicitly in each benchmark
  process wrapper to the already requested and Slurm-recorded allocation:
  `gpu:ampere:1` for the reference and `gpu:ampere80:1` for the candidate.
  Verify both submitted jobs' requested GRES from Slurm before comparison.
- Compare only the new correction pair with the unchanged verifier. Do not
  reuse the original failed pair as a passing gate, relax any check, inspect
  held-out data, or use HPO metrics for this decision.
- If the correction pair passes, A100-80GB may be ratified under the existing
  amendment. If it fails, exclude A100-80GB and rerun affected HPO on
  A100-40GB. Keep at most three owned GPU jobs and do not use L40S.

## Execution state, 05:42 SAST

- Sealed b5 `1257792` was cancelled at `05:33:46` after `06:38:26` runtime.
  Its `16` files (`123,445,307` bytes) remain untouched as quarantined partial
  provenance; no result contents were inspected.
- Corrected A100-40GB reference `1258452` and dependency-gated A100-80GB
  candidate `1258453` were submitted once with new isolated result prefixes.
  Slurm records requested GRES `gpu:ampere:1` and `gpu:ampere80:1`
  respectively. Reference `1258452` is Priority-pending; candidate `1258453`
  is Dependency-pending on its success. B4 `1253374` is the only running
  owned job, so the three-job cap is exactly full. No L40S is in use.
