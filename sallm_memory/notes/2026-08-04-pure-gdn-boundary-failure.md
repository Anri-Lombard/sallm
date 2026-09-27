# Pure-GDN streaming-boundary failure — 2026-08-04

## Confirmed state

- Primary job `1167989` failed `1:0` after `1-00:17:13`; its sole `afterany`
  continuation `1167990` resumed `checkpoint-29000` and failed `1:0` after
  `01:30:09`. No owned Slurm job remains active or schedulable.
- Both segments reproduced the same failure immediately after training step
  `29,954`. The last retained checkpoints are `checkpoint-28000` and
  `checkpoint-29000`, each about `735M`; `checkpoint-29000/trainer_state.json`
  confirms `global_step=29000` and `max_steps=67498`.
- Latest confirmed validation is step 29,000,
  `eval_loss=3.5169744492`. Training remained numerically healthy near the
  failure (`loss=3.4456` at step 29,950, finite gradient norm, about
  `2.64--2.70 s/step`).
- Scratch is `90/300 GB` (`30.1%`). A100-80GB node `srvrocgpu011` is mixed;
  the owned queue is empty. No A100-40GB or L40S work was submitted.

## Root-cause evidence

- This is a deterministic distributed collective-order divergence at the
  finite streaming boundary, not a numerical blow-up and not currently
  evidence of a GPU failure.
- On the first failure, collective sequence `1836708` disagreed across ranks:
  rank 0 entered `_ALLGATHER_BASE` with `NumelIn=1, NumelOut=2`, while rank 1
  entered gradient `ALLREDUCE` with `NumelIn=NumelOut=786944`. The watchdog
  timed out after `1,800,048 ms`.
- The continuation reproduced the same mismatch at sequence `53643` and the
  same step: rank 0 `_ALLGATHER_BASE(1 -> 2)`, rank 1
  `ALLREDUCE(786944 -> 786944)`, timing out after approximately 1,800 seconds.
  Rank 1 had enqueued through sequence `53650` while rank 0 stopped at `53643`.
- The max-step Trainer treats the iterable as one step-bounded epoch. The Hub
  stream is finite and independently sharded after `dispatch_batches=False`;
  one rank reaches exhaustion while the other is still reducing gradients.
  A blind continuation would therefore fail again at the same boundary.

## Smallest local fix and verification

- `load_pretrain_datasets` now applies Hugging Face
  `IterableDataset.repeat(None)` only to the training split when an explicit
  positive `training.max_steps` governs the run. Validation remains finite and
  non-repeated. Runs without positive `max_steps` are unchanged.
- Added one focused regression test proving that a one-row streaming training
  split yields beyond its physical end under a max-step contract.
- Focused verification: `18 passed`; Ruff and `git diff --check` pass.
- At the diagnosis stage the change remained local and uncommitted, with no
  HEX mutation. The subsequently authorized recovery launch is recorded below.

## Authorized recovery launch

- The fix, focused test, and isolated boundary-canary launcher were frozen in
  local signed commit `f377285` and synced to HEX with exact matching SHA256
  values. Nothing was pushed and `main` remains untouched.
- With explicit user authorization, two-A100-80GB diagnostic job `1179842`
  was submitted on `srvrocgpu011`. It reads canonical `checkpoint-29000` but
  writes only to
  `checkpoints/sallm-pure-gdn-125m-diagnostics/stream-boundary-20260804`.
  Preflight reconfirmed BF16, exact `127,425,448` parameters, and real
  2,048-token batches. The job resumes to diagnostic step 30,010 so it must
  cross the reproducible step-29,954 boundary before post-checkpoint reload,
  exact roundtrip, and deterministic generation verification can return 0.
- Canonical continuation job `1179847` is submitted with strict Slurm
  dependency `afterok:1179842`. It cannot start unless the complete diagnostic
  job, including post-checkpoint verification, exits successfully. If released,
  it resumes the unchanged canonical output from `checkpoint-29000` under the
  frozen 67,498-step contract. No second continuation was submitted.

## Boundary gate passed and canonical recovery released

- Canary `1179842` completed `0:0` in `01:41:27`. It crossed the former failure
  boundary continuously (`29,954 -> 29,955`), reached step `30,010`, and saved
  its isolated `final_model`. Training stayed finite at ordinary scale near the
  boundary, with about `2.7 s/step` after replay completed and no NCCL timeout,
  traceback, or CUDA OOM.
- Post-checkpoint verification returned exact model identity and parameter count
  (`127,425,448`), `save_load_integrity=true`, and
  `deterministic_greedy_generation=true`. This validates the finite-stream
  repeat fix against the exact reproducible two-rank failure position.
- Because the complete canary exited successfully, Slurm automatically released
  canonical continuation `1179847` at `13:04:11 SAST`. It is running on two
  `ampere80` GPUs and has reconfirmed the canonical `checkpoint-29000`, full
  `67,498` max-step contract, real 2,048-token batch, and original optimizer
  schedule state. It is currently replay-skipping to reconstruct the resume
  position; no new optimizer step has yet been logged.
- The boundary canary's overridden `max_steps=30,010` made its diagnostic LR
  approach zero. Its losses must not be used for scientific comparison; the
  gate establishes execution correctness only. Canonical validation-loss
  monitoring resumes with job `1179847`.
- At `14:08 SAST`, canonical job `1179847` had completed replay and resumed
  genuine optimizer progress through step `29,317` at approximately
  `2.69--2.71 s/step`. No current-run traceback, NCCL timeout, or CUDA OOM was
  present. The first recovered canonical validation and checkpoint are due at
  step `30,000`; retained canonical checkpoints remain 28,000 and 29,000 until
  that gate completes.
- Canonical recovery then passed the complete step-30,000 gate. Job `1179847`
  crossed 29,954 without a collective mismatch, saved intact
  `checkpoint-30000`, and continued through step `30,551` by `15:09 SAST`.
  Checkpoint rotation now retains 29,000 and 30,000, each about `735M`.
- Validation loss improved from `3.5169744492` at step 29,000 to
  `3.5101182461` at step 30,000. Step-30,000 training loss was `3.4373`,
  gradient norm `0.310546875`, and learning rate `0.0002451732898`; all are
  finite and consistent with the pre-failure trajectory. Steady throughput is
  about `2.71 s/step`. The recovery is scientifically accepted and no further
  boundary-specific action is required.
- Hourly follow-up at `16:10 SAST`: job `1179847` reached step `31,778` and
  retained checkpoints 30,000/31,000. Step-31,000 validation loss improved
  again to `3.5026390553`; training loss was `3.3992`, gradient norm
  `0.3046875`, and learning rate `0.0002357798754`. No current-run error marker
  appeared and steady throughput remained about `2.71--2.73 s/step`.
- Hourly follow-up at `17:09 SAST`: job `1179847` reached step `33,000` and
  entered its validation/checkpoint boundary; retained checkpoints were
  31,000/32,000 at inspection time. Step-32,000 validation loss improved to
  `3.4943051338`. Slurm still reported `RUNNING` on two `ampere80` GPUs, with
  no evidence of a current-run traceback, NCCL timeout, or CUDA OOM; error
  markers elsewhere in the append-only log are stale failures from jobs
  `1167989/1167990`. Scratch remained `90/300 GB` (`30.2%`), and no owned
  A100-40GB or L40S job was active or schedulable. ETA remained approximately
  `2026-08-05 21:00 SAST`, safely within job `1179847`'s wall-time allocation.
- Hourly follow-up at `18:19 SAST`: job `1179847` reached step `34,389` and
  retained checkpoints 33,000/34,000. Validation loss improved at both new
  gates, to `3.4876956940` at step 33,000 and `3.4813935757` at step 34,000.
  Slurm reported `RUNNING` on two `ampere80` GPUs; targeted log inspection
  found no error marker after the stale failures from jobs `1167989/1167990`.
  Scratch remained `90/300 GB` (`30.2%`), with no owned A100-40GB or L40S
  work. ETA remained approximately `2026-08-05 21:00 SAST`, safely within the
  current allocation.
- Hourly follow-up at `20:20 SAST`: job `1179847` reached step `36,878` and
  retained checkpoints 35,000/36,000. Validation loss continued improving to
  `3.4756999016` at step 35,000 and `3.4700014591` at step 36,000. Slurm
  remained `RUNNING` on two `ampere80` GPUs; the last error marker in the
  append-only log was still the stale traceback from the failed predecessor,
  before the current recovery launch. Scratch remained `90/300 GB` (`30.2%`),
  no owned A100-40GB or L40S job was active or schedulable, and the ETA
  remained approximately `2026-08-05 21:00 SAST` within wall time.
- Hourly follow-up at `21:21 SAST`: job `1179847` reached step `38,069` and
  retained checkpoints 37,000/38,000. Validation loss improved to
  `3.4642486572` at step 37,000 and `3.4596061707` at step 38,000. Slurm
  remained `RUNNING` on two `ampere80` GPUs; no error marker followed the
  stale predecessor traceback. Scratch remained `90/300 GB` (`30.2%`), no
  owned A100-40GB or L40S work existed, and ETA remained approximately
  `2026-08-05 21:00 SAST`, safely inside the allocation.
- Hourly follow-up at `22:20 SAST`: job `1179847` reached step `39,293` and
  retained checkpoints 38,000/39,000. Step-39,000 validation loss improved to
  `3.4549090862`. Slurm remained `RUNNING` on two `ampere80` GPUs, with no new
  traceback, NCCL timeout, or CUDA OOM after the stale predecessor error.
  Scratch remained `90/300 GB` (`30.2%`), no owned A100-40GB or L40S work
  existed, and ETA remained approximately `2026-08-05 21:00 SAST` within wall
  time.
- Hourly follow-up at `23:21 SAST`: job `1179847` reached step `40,523` and
  retained checkpoints 39,000/40,000. Step-40,000 validation loss improved to
  `3.4511444569`. Slurm remained `RUNNING` on two `ampere80` GPUs; no new
  traceback, NCCL timeout, or CUDA OOM followed the stale predecessor error.
  Scratch remained `90/300 GB` (`30.2%`), no owned A100-40GB or L40S work
  existed, and ETA remained approximately `2026-08-05 21:00 SAST`, safely
  within wall time.

## Kombuys readiness

- MasakhaNews classification remains successful with its saved adapter; it was
  not repeated.
- The prior AfriHG Xhosa generation smoke failed only because Kombuys could not
  resolve `raw.githubusercontent.com`; no CUDA OOM occurred. DNS now resolves,
  so only that generation arm was relaunched in tmux
  `pure-gdn-downstream-readiness` on RTX 3080 Ti GPU 1. GPU 0 was not touched.
- The generation-only retry then completed cleanly and saved
  `afrihg_generation/final_adapter/{adapter_config.json,adapter_model.safetensors}`.
  TileLang backward ran and no traceback or CUDA OOM occurred. The one-step
  loss and gradient norm were both zero because this truncated diagnostic batch
  exposed no trainable target tokens; therefore the smoke verifies execution,
  generation, and adapter persistence only, not learning quality or a useful
  LoRA recipe. No metric is selected or reported.
