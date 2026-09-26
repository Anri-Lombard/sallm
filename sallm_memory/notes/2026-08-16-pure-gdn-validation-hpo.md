# Pure-GDN corrected validation-only HPO — 2026-08-16

## POS remains the deadline-critical bottleneck — 22:13 SAST

- POS b2 `1241578` remains healthy on `srvrocgpu010` A100-40GB. Its first
  exact checkpoint-283 callback advanced to `1,300/1,800` rows at `22:13`
  with a fresh fault-free log; completion is projected around `22:50--23:00`
  SAST. The callback consumes roughly 2.25 hours versus roughly 20 minutes of
  training per epoch, so exact validation, not model training, dominates POS
  wall time.
- Exact checkpoint-849 resumes b0/b1 `1241580/1241581` remain pending for
  Resources/Priority. Slurm currently projects b0 at
  `2026-08-17 11:58:10 SAST` and provides no b1 estimate. All four A100-40GB
  devices on `srvrocgpu010` are allocated; our b2 occupies one.
- The already-gated cache variants delivered only `1.41x/1.04x`, and
  cross-row batching was rejected before training after unsafe
  `38,942/40,960 MiB` use. Exact checkpoint resume preserved about 7.5 hours
  each for b0/b1, but it does not reduce future exact callbacks. Therefore the
  full remaining POS Stage-B plus confirmation plan is not credibly compatible
  with the Thursday deadline without a prospectively documented scientific
  scope change. No job, recipe, metric, held-out split, or artifact was changed
  by this status assessment.

## B2 healthy; b0 queue estimate regresses — 21:54 SAST

- POS b2 `1241578` remains healthy on `srvrocgpu010` A100-40GB. Its frozen
  checkpoint-283 exact callback reached `1,000/1,800` rows at `21:50`, with
  a fresh fault-free log. Sustained throughput still projects the complete
  validation-only artifact around `22:45--23:00 SAST`; partial rows remain
  operational only.
- Slurm's dynamic estimate for exact b0 checkpoint-849 resume `1241580`
  regressed from `2026-08-16 22:43:34` to `2026-08-17 11:58:10 SAST` while
  it remains Resources-pending. B1 resume `1241581` remains Priority-pending
  without an estimate. No recipe, checkpoint, metric, or held-out action was
  changed in response. Owned state is one running plus two pending
  A100-40GB jobs; quota is home `88.6%`, scratch `39.0%`; Kombuys, Sheet,
  quarantine, and publication state are unchanged.

## Serial b2 callback reaches 650/1,800 — 21:24 SAST

- POS b2 `1241578` remains healthy on `srvrocgpu010` A100-40GB. Its frozen
  checkpoint-283 exact callback advanced from `200` to `650/1,800` rows with
  a fresh log and no fault marker; sustained throughput keeps the first
  complete artifact around `22:45--23:00 SAST`. Partial rows remain
  operational only and cannot affect retention or ranking.
- Exact b0/b1 checkpoint-849 resumes `1241580/1241581` remain
  Resources/Priority-pending. Slurm still projects b0 at
  `2026-08-16 22:43:34 SAST` and gives b1 no estimate. Owned state is one
  running plus two pending A100-40GB jobs, with no A100-80GB or L40S work.
  Quota is home `88.6%`, scratch `39.0%`; Kombuys remains read-only and idle,
  Sheet E/F/G remain blank, and trusted counts and held-out/publication gates
  are unchanged.

## Live cross-facility GPU capacity audit — 21:22 SAST

- HEX has no immediately free modern GPU. All `12/12` A100 devices are
  allocated: four `amperemk` on `srvrocgpu009`, four A100-40GB `ampere` on
  `srvrocgpu010`, and four A100-80GB `ampere80` on `srvrocgpu011`. Our job
  `1241578` occupies one A100-40GB; resumed b0/b1 `1241580/1241581` remain
  Resources/Priority-pending. Slurm projects b0 at `2026-08-16 22:43:34 SAST`
  and provides no b1 start estimate.
- All `20/20` L40S devices across `srvrocgpu012--016` are allocated, with a
  substantial pending queue. The current correction protocol also excludes
  L40S, so it offers neither immediate nor scientifically admissible capacity.
  The `16` Pascal GPUs on `srvrocgpu005--008` are physically idle, but this
  user has no `gpumk` association and they are not an appropriate replacement
  for the frozen A100-40GB grid.
- Kombuys is live and idle: RTX 5090 `10/32607 MiB`, RTX 3080 Ti `1/12288 MiB`,
  both `0%` utilization/P8; `/scratch` is `60%` used. It remains read-only and
  outside the A100-only correction protocol. It may support non-selection
  diagnostics only after separate authorization; moving HPO there now would
  require a prospective cross-hardware equivalence gate and would not be a
  clean immediate speedup.
- HEX quota remains home `88.6%` and scratch `39.0%`. No jobs or protocols were
  changed by this read-only audit.

## Serial b2 first callback healthy; b0 queue improves — 20:50 SAST

- POS b2 `1241578` is healthy on `srvrocgpu010` A100-40GB. Epoch-1 training
  reached step `283/4245`, declared evaluation covered all `1,800/1,800` rows
  at health-only loss `1.2360939534505209`, and the frozen exact callback
  reached `200/1,800` rows at `20:50`. Current throughput projects its first
  complete exact artifact around `22:45--23:00 SAST`; partial output cannot
  affect retention or ranking.
- Exact b0/b1 checkpoint-849 resumes remain queued as `1241580/1241581`.
  Slurm's dynamic b0 projection improved materially to `2026-08-16 22:43:34
  SAST`; b1 still has no projection. Owned state remains one running plus two
  pending A100-40GB jobs, no other GPU family. Quota is home `88.6%`, scratch
  `39.0%`; trusted scientific counts, Sheet, Kombuys, held-out, and publication
  state are unchanged.

## Deadline acceleration deployed — 20:19 SAST

- B1 `1238990` and b0 recovery `1238989` were cancelled at elapsed
  `09:28:20` and `09:34:19` after the metric-blind preregistrations above.
  Both preserve complete checkpoint-849 adapter/optimizer/scheduler/RNG/trainer
  states. B0 hashes are `f8296fb4...4508`, `2574da68...752`,
  `75f870b8...455`, `8940a8ae...1af`, `5c5dcb9f...f65`; b1 hashes are
  `cbde3e88...553`, `5c6d2f7d...522`, `46c7b8a5...549`,
  `6e691909...fa5`, `c7f1d1a1...704`. Their incomplete checkpoint-1132
  callbacks are excluded from every scientific decision.
- Initial corrected row-batch job `1239042` failed before training at `00:01:01`
  on the real-tokenizer multi-token-label assertion. The narrowly corrected
  local regression covers one- and multi-token labels; eight targeted tests,
  Ruff, formatting, and whitespace checks pass. Immutable v3m manifest/scorer/
  test hashes are `61044e3f...09b`, `a37e1d27...e4b`, and
  `32de3554...cc3`. Replacement gate `1241563` was stopped before training at
  `00:11:40` after its 12-row benchmark consumed `38,942/40,960 MiB` without
  producing the required equivalence/speed artifact. Cross-row batching is not
  authorized and no more deadline-path optimization attempts are planned.
- Exact checkpoint resume support was added with complete-state and same-output
  path assertions plus resume-specific manifests. Startup jobs `1241576/1241577`
  failed before model/data at `00:00:34/00:00:33` because Hydra required an
  append override. The one-character correction passed shell syntax, local dry
  run, and remote Hydra composition. Immutable resume-v2 snapshot
  `/home/lmbanr001/masters/sallm_snapshots/pure-gdn-hpo-pos-resume-v2-20260816-45fec06b`
  verifies `695/695` files; manifest/launcher hashes are
  `bdfd7f2f...34e2` and `45fec06b...eab`.
- Unchanged serial b2 job `1241578` is healthy on `srvrocgpu010` A100-40GB,
  loaded exact `2,259/1,800` train/validation rows, uses the frozen candidate
  and has begun training. Exact checkpoint-849 resumes b0/b1 are queued as
  `1241580/1241581`; Slurm currently projects b0 at
  `2026-08-17 11:58:10 SAST` and gives b1 no start. This is one running plus
  two pending owned A100-40GB jobs, the three-job cap, with no A100-80GB/L40S
  overlap. Quota is home `88.6%`, scratch `39.0%`.
- Operational work was accelerated by preserving roughly 7.5 hours of completed
  training/validation state per resumed candidate, not by changing scientific
  selection. Trusted counts remain base `16/16`, NER `11/11` plus `2/4`
  confirmations, T2X `11/11` plus `4/4`, POS `3/11`, winners `0/8`, held-out
  `0`, and Mono not started. Kombuys remains read-only; Sheet E/F/G remain
  blank, quarantined rows unpublished, and publication/held-out gates closed.

## Resume launcher Hydra correction — preregistered 20:16 SAST

- Resume job `1241576` failed before model/data loading or training because the
  new optional key used a normal Hydra override; the structured config requires
  an append override. Correct only `training.resume_from_checkpoint=` to
  `+training.resume_from_checkpoint=` and redeploy an immutable launcher
  snapshot. Candidate state, checkpoint, evaluator, recipe, and all scientific
  gates remain unchanged. Preserve job `1241576` and any identically affected
  startup as infrastructure-only failure provenance.

## Row-batch v3m rejected operationally before training — 20:12 SAST

- Corrected gate `1241563` remained inside the validation-only benchmark with
  no equivalence artifact after more than ten minutes for 12 rows. A targeted
  allocation-local GPU check measured `38,942 MiB` used of `40,960 MiB` while
  scoring. This cannot satisfy a production-safe >=3x authorization: the
  frozen validation contains longer rows and training-time callbacks retain
  optimizer state. Stop the gate before b2 training and retain all artifacts as
  failure provenance. No score, held-out data, or candidate result informed
  this operational rejection.
- Do not continue speculative POS scorer optimization on the deadline path.
  The minimum safe acceleration is exact resume from the complete, hashed
  checkpoint-849 states for b0/b1, avoiding roughly 7.5 hours of repeated work
  per candidate, while queuing unchanged serial b2 concurrently. The evaluator,
  metric, patience, recipes, seed, and data remain frozen.

## Deadline acceleration: metric-blind b0 pause — preregistered 20:00 SAST

- A foreign job occupied the A100 released by failed job `1239042` before the
  corrected replacement `1241563` could start; Slurm therefore projects the
  replacement only at `2026-08-17 10:26:08`. To prevent losing the night, pause
  b0 recovery `1238989` at its last complete `checkpoint-849` and release that
  A100 to the already-preregistered corrected gate. This choice is operational
  and deadline-driven; neither held-out data nor the incomplete checkpoint-1132
  callback score is consulted.
- Preserve and hash b0 checkpoint-849's adapter, optimizer, scheduler, RNG, and
  trainer state. Discard its incomplete checkpoint-1132 callback in full. B0
  may resume only from checkpoint 849 with its frozen recipe and metric protocol
  under the same batched-evaluator authorization rule already stated for b1.

## Row-batch v3 real-tokenizer correction — preregistered 19:56 SAST

- Job `1239042` started on the released A100-40GB at `19:54:38` and failed
  closed before b2 training. Its real-tokenizer gate exposed an implementation
  assumption missed by the synthetic unit test: at least one UPOS label is a
  multi-token continuation in context, while the batched scorer required every
  label to be one token. No equivalence/speed result or training update exists.
- The only authorized correction is to preserve the one-token fast path and,
  when any contextual label continuation is multi-token, batch the same
  full-prefix row-label sequences already used by the serial reference scorer.
  Add a regression test covering both one- and multi-token labels. The frozen
  batch size, validation subset, predictions/counts/metrics, score tolerances,
  >=3x speed requirement, b2 recipe, and fail-closed behavior remain unchanged.
- A corrected immutable snapshot and complete source/launcher/config hash
  manifest are required before exactly one replacement gate/b2 submission. No
  held-out data or partial checkpoint-1132 output may be consulted.

## Deadline acceleration: metric-blind b1 pause — preregistered 19:54 SAST

Thursday 20 August is now the fixed delivery deadline. Before any cancellation
or new optimized measurement, the following operational rule is frozen. It was
chosen to reduce wall-clock time and did not consult held-out data or the
incomplete checkpoint-1132 callback scores.

- Keep the lower-index recovery candidate b0 (`1238989`) running. Pause the
  higher-index candidate b1 (`1238990`) solely by candidate order, freeing its
  A100-40GB for the already-preregistered row-batch v3 equivalence gate/b2 job
  `1239042`. No completed b0/b1 validation score is compared for this choice.
- B1's last complete restart point is `checkpoint-849`, containing adapter,
  optimizer, scheduler, RNG, and trainer state. Preserve and hash it. The
  partially evaluated checkpoint-1132 callback is discarded in full and may
  not enter retention, ranking, early stopping, or any later decision.
- Job `1239042` remains unchanged and must pass its frozen validation-only
  prediction/count/metric/score-drift and >=3x speed gates before b2 training.
  Failure remains fail-closed before b2 training.
- Resuming b1 is permitted only from the preserved checkpoint-849 state under
  the same frozen candidate, seed, data, optimizer, checkpoint cadence,
  patience, and metric protocol. Batched evaluation may be used for future POS
  work only if job `1239042` passes its prospective equivalence gate; otherwise
  b1 resumes with the original serial evaluator. No held-out data may be loaded.

## POS checkpoint-1132 callbacks reach 1050/1800 — 19:46 SAST

- Full-prefix recovery jobs `1238989/1238990` remain healthy on
  `srvrocgpu010` A100-40GB after about 9h19m. Both checkpoint-1132 frozen
  exact callbacks reached `1,050/1,800` rows by 19:45, with fresh fault-free
  logs. Observed serial throughput keeps complete-artifact ETA at
  20:35--20:50 SAST. Partial rows remain operational only and cannot affect
  retention or ranking.
- Fused row-batch v3 gate/b2 `1239042` remains Resources-pending on
  `gpu:ampere`, conservatively scheduled for
  `2026-08-17 10:26:08 SAST` but eligible when a recovery GPU releases.
  Owned state is two running plus one pending A100-40GB job, no A100-80GB or
  L40S execution. HEX quota remains home `70.3%`, scratch `38.9%`.
- Terminal trusted counts remain base `16/16`, NER `11/11` plus `2/4`
  confirmations, T2X `11/11` plus `4/4`, POS `3/11`, winners `0/8`,
  held-out `0`, and Mono not started. Kombuys was not accessed; its latest
  verified state remains read-only and idle. Sheet E/F/G remain blank,
  quarantined rows unpublished, Hugging Face blocked, and remaining
  validation/winner-freeze gates still block held-out work.

## POS checkpoint-1132 callbacks reach 650/1800 — 19:16 SAST

- Full-prefix recovery jobs `1238989/1238990` remain healthy on
  `srvrocgpu010` A100-40GB after about 8h49m. Both checkpoint-1132 frozen
  exact callbacks reached `650/1,800` rows by 19:15, with fresh fault-free
  logs. Observed serial throughput keeps complete-artifact ETA at
  20:35--20:50 SAST. Partial rows remain operational only and cannot affect
  retention or ranking.
- Fused row-batch v3 gate/b2 `1239042` remains Resources-pending on
  `gpu:ampere`, conservatively scheduled for
  `2026-08-17 10:26:08 SAST` but eligible when a recovery GPU releases.
  Owned state is two running plus one pending A100-40GB job, no A100-80GB or
  L40S execution. HEX quota remains home `70.3%`, scratch `38.9%`.
- Terminal trusted counts remain base `16/16`, NER `11/11` plus `2/4`
  confirmations, T2X `11/11` plus `4/4`, POS `3/11`, winners `0/8`,
  held-out `0`, and Mono not started. Kombuys was not accessed; its latest
  verified state remains read-only and idle. Sheet E/F/G remain blank,
  quarantined rows unpublished, Hugging Face blocked, and remaining
  validation/winner-freeze gates still block held-out work.

## POS checkpoint-1132 callbacks active — 18:46 SAST

- Full-prefix recovery jobs `1238989/1238990` remain healthy on
  `srvrocgpu010` A100-40GB after about 8h19m. B0 reached `300/1,800` rows at
  18:45 and b1 reached `250/1,800` at 18:44 in their checkpoint-1132 frozen
  exact callbacks. Logs are fresh and fault-free; observed serial throughput
  puts both complete artifacts around 20:35--20:50 SAST. Partial rows remain
  operational only and cannot affect retention or ranking.
- Fused row-batch v3 gate/b2 `1239042` remains Resources-pending on
  `gpu:ampere`, conservatively scheduled for
  `2026-08-17 10:26:08 SAST` but eligible when a recovery GPU releases.
  Owned state is two running plus one pending A100-40GB job, no A100-80GB or
  L40S execution. HEX quota remains home `70.3%`, scratch `38.9%`.
- Terminal trusted counts remain base `16/16`, NER `11/11` plus `2/4`
  confirmations, T2X `11/11` plus `4/4`, POS `3/11`, winners `0/8`,
  held-out `0`, and Mono not started. Kombuys was not accessed; its latest
  verified state remains read-only and idle. Sheet E/F/G remain blank,
  quarantined rows unpublished, Hugging Face blocked, and remaining
  validation/winner-freeze gates still block held-out work.

## POS checkpoint-849 artifacts verified and retained — 18:17 SAST

- Full-prefix recovery b0 `1238989` and b1 `1238990` completed their
  checkpoint-849 frozen exact callbacks over all `1,800/1,800` rows, 12
  cells, and 17 labels under `closed_label_tuple_mean_logprob_v1`. B0 exact
  accuracy is `0.830626634400054`; b1 is `0.8016046208083835`. Both improve
  on their own checkpoint-566 scores and retain checkpoint 849 with patience
  `0/2`. Artifact SHA-256 values are
  `9574e4c17f3804c7b43adc6dca127999e17f0e41bf85034fa5b90fe8cb1bca7a`
  and `6c217453c5e4318897eba4f8fceba8ea1087a512f89991fbb665a2d7b03f2d2c`.
  Trainer-state/adapter SHA-256 values are respectively
  `5c5dcb9f168a6c553d42653d68ec57975061d7eb05f48d88feec773177e55f65` /
  `f8296fb4d1ab57953cf2fa12031c45ba13a3c733c36e20a85fb35432d1ea4508`
  and
  `c7f1d1a16b2c227d844c01660bae9255ef0527e5d1a8ecb3dc39af9a63483704` /
  `cbde3e88c09207c77e8db5a454473c6c8e3bc3d3c99151bf661f3979f344a553`.
  Cancelled b0 `1238876` remains provenance-only and is not used in recovery
  retention.
- Both jobs resumed healthy training on `srvrocgpu010` A100-40GB, reaching
  steps 1022/975 by 18:15 with no fault marker. Their next callbacks are due
  at checkpoint 1132; at current serial timing complete exact artifacts are
  expected roughly 20:30--20:50 SAST. These are valid within-candidate
  validation artifacts, not terminal candidate results or cross-candidate
  rankings; terminal trusted POS remains `3/11`.
- Fused row-batch v3 gate/b2 `1239042` remains Resources-pending on
  `gpu:ampere`, conservatively scheduled for
  `2026-08-17 10:26:08 SAST` but eligible when a recovery GPU releases.
  Owned state is two running plus one pending A100-40GB job, no A100-80GB or
  L40S execution. HEX quota remains home `70.3%`, scratch `38.9%`.
- Operationally b0/b1 now each have three verified recovery artifacts, but
  terminal trusted counts remain base `16/16`, NER `11/11` plus `2/4`
  confirmations, T2X `11/11` plus `4/4`, POS `3/11`, winners `0/8`,
  held-out `0`, and Mono not started. Kombuys was not accessed; its latest
  verified state remains read-only and idle. Sheet E/F/G remain blank,
  quarantined rows unpublished, Hugging Face blocked, and remaining
  validation/winner-freeze gates still block held-out work.

## POS checkpoint-849 callbacks reach 1500/1450 — 17:46 SAST

- Full-prefix recovery jobs `1238989/1238990` remain healthy on
  `srvrocgpu010` A100-40GB after about 7h19m. B0 reached `1,500/1,800` rows
  at 17:44 and b1 reached `1,450/1,800` at 17:43 in their checkpoint-849
  frozen exact callbacks. Logs are fresh and fault-free; remaining rows put
  both complete artifacts around 18:05--18:15 SAST. Partial rows remain
  operational only and cannot affect retention or ranking.
- Fused row-batch v3 gate/b2 `1239042` remains Resources-pending on
  `gpu:ampere`, conservatively scheduled for
  `2026-08-17 10:26:08 SAST` but eligible when a recovery GPU releases.
  Owned state is two running plus one pending A100-40GB job, no A100-80GB or
  L40S execution. HEX quota remains home `70.3%`, scratch `38.9%`.
- Terminal trusted counts remain base `16/16`, NER `11/11` plus `2/4`
  confirmations, T2X `11/11` plus `4/4`, POS `3/11`, winners `0/8`,
  held-out `0`, and Mono not started. Kombuys was not accessed; its latest
  verified state remains read-only and idle. Sheet E/F/G remain blank,
  quarantined rows unpublished, Hugging Face blocked, and remaining
  validation/winner-freeze gates still block held-out work.

## POS checkpoint-849 callbacks pass 1050/1800 — 17:14 SAST

- Full-prefix recovery jobs `1238989/1238990` remain healthy on
  `srvrocgpu010` A100-40GB after about 6h47m. B0 reached `1,100/1,800` rows
  at 17:12 and b1 reached `1,050/1,800` at 17:11 in their checkpoint-849
  frozen exact callbacks. Logs are fresh and fault-free; observed serial
  throughput puts both complete artifacts around 18:05--18:15 SAST.
  Partial rows remain operational only and cannot affect retention or
  ranking.
- Fused row-batch v3 gate/b2 `1239042` remains Resources-pending on
  `gpu:ampere`, conservatively scheduled for
  `2026-08-17 10:26:08 SAST` but eligible when a recovery GPU releases.
  Owned state is two running plus one pending A100-40GB job, no A100-80GB or
  L40S execution. HEX quota remains home `70.3%`, scratch `38.9%`.
- Terminal trusted counts remain base `16/16`, NER `11/11` plus `2/4`
  confirmations, T2X `11/11` plus `4/4`, POS `3/11`, winners `0/8`,
  held-out `0`, and Mono not started. Kombuys was not accessed; its latest
  verified state remains read-only and idle. Sheet E/F/G remain blank,
  quarantined rows unpublished, Hugging Face blocked, and remaining
  validation/winner-freeze gates still block held-out work.

## POS checkpoint-566 artifacts verified and retained — 15:42 SAST

- Full-prefix recovery b0 `1238989` and b1 `1238990` completed their
  checkpoint-566 frozen exact callbacks over all `1,800/1,800` rows, 12
  cells, and 17 labels under `closed_label_tuple_mean_logprob_v1`. B0 exact
  accuracy is `0.7890084636039543`; b1 is `0.7616054032322993`. Both improve
  on their own recovery checkpoint-283 scores and retain checkpoint 566 with
  patience `0/2`. Artifact SHA-256 values are
  `ea7de0a378816d5cf35435d1dca440c6044e2f30f3e54c547cf5105893c8daf4`
  and `9acee96952db70a1dbf2432449cd441c75d8da67a54d41f67370abf96ff52a48`.
  Trainer-state/adapter SHA-256 values are respectively
  `f53b85cb6362f11b43c086831fa9ab32e09067a8520c6be9e92a293372d9df7b` /
  `9c633b32927d15152fef63c78804082d3bd9a1b8d40371c5bb6d2086381c7424`
  and
  `1db8b308dd0f61b8966518245124c526dff8cf677928d7caa1da815c3a023ab2` /
  `6aee35baa118333af88626d9ab919a1cd2a75218b525e1acb720347cf02aadc7`.
  Cancelled b0 `1238876` remains provenance-only and is not used in recovery
  retention.
- Both jobs resumed healthy training on `srvrocgpu010` A100-40GB, reaching
  steps 738/686 by 15:41 with no fault marker. Their next exact callbacks are
  due at checkpoint 849 and, at current serial timing, complete artifacts
  are expected roughly 17:50--18:10 SAST. These are valid within-candidate
  validation artifacts, not terminal candidate results or cross-candidate
  rankings; terminal trusted POS remains `3/11`.
- Fused row-batch v3 gate/b2 `1239042` remains Resources-pending on
  `gpu:ampere`, conservatively scheduled for
  `2026-08-17 10:26:08 SAST` but eligible when a recovery GPU releases.
  Owned state is two running plus one pending A100-40GB job, no A100-80GB or
  L40S execution. HEX quota remains home `70.3%`, scratch `38.9%`.
- Operationally b0/b1 now each have two verified recovery artifacts, but
  terminal trusted counts remain base `16/16`, NER `11/11` plus `2/4`
  confirmations, T2X `11/11` plus `4/4`, POS `3/11`, winners `0/8`,
  held-out `0`, and Mono not started. Kombuys was not accessed; its latest
  verified state remains read-only and idle. Sheet E/F/G remain blank,
  quarantined rows unpublished, Hugging Face blocked, and remaining
  validation/winner-freeze gates still block held-out work.

## POS checkpoint-566 callbacks reach 1450/1800 — 15:11 SAST

- Full-prefix recovery jobs `1238989/1238990` remain healthy on
  `srvrocgpu010` A100-40GB after about 4h45m. Both checkpoint-566 frozen
  exact callbacks reached `1,450/1,800` rows by 15:09, with fresh fault-free
  logs. Only 350 rows remain per job; observed throughput puts complete
  artifacts around 15:35--15:45 SAST. Partial rows remain operational only
  and cannot affect retention or ranking.
- Fused row-batch v3 gate/b2 `1239042` remains Resources-pending on
  `gpu:ampere`, conservatively scheduled for
  `2026-08-17 10:26:08 SAST` but eligible when a recovery GPU releases.
  Owned state is two running plus one pending A100-40GB job, no A100-80GB or
  L40S execution. HEX quota remains home `70.3%`, scratch `38.9%`.
- Terminal trusted counts remain base `16/16`, NER `11/11` plus `2/4`
  confirmations, T2X `11/11` plus `4/4`, POS `3/11`, winners `0/8`,
  held-out `0`, and Mono not started. Kombuys was not accessed; its latest
  verified state remains read-only and idle. Sheet E/F/G remain blank,
  quarantined rows unpublished, Hugging Face blocked, and remaining
  validation/winner-freeze gates still block held-out work.

## POS checkpoint-566 callbacks pass 1050/1800 — 14:42 SAST

- Full-prefix recovery jobs `1238989/1238990` remain healthy on
  `srvrocgpu010` A100-40GB after about 4h16m. B0 reached `1,100/1,800` rows
  at 14:39 and b1 reached `1,050/1,800` at 14:38 in their checkpoint-566
  frozen exact callbacks. Logs are fresh and fault-free; observed throughput
  keeps both complete artifacts around 15:30--15:40 SAST. Partial rows stay
  operational only and cannot affect retention or ranking.
- Fused row-batch v3 gate/b2 `1239042` remains Resources-pending on
  `gpu:ampere`, conservatively scheduled for
  `2026-08-17 10:26:08 SAST` but eligible when a recovery GPU releases.
  Owned state is two running plus one pending A100-40GB job, no A100-80GB or
  L40S execution. HEX quota remains home `70.3%`, scratch `38.9%`.
- Terminal trusted counts remain base `16/16`, NER `11/11` plus `2/4`
  confirmations, T2X `11/11` plus `4/4`, POS `3/11`, winners `0/8`,
  held-out `0`, and Mono not started. Kombuys was not accessed; its latest
  verified state remains read-only and idle. Sheet E/F/G remain blank,
  quarantined rows unpublished, Hugging Face blocked, and remaining
  validation/winner-freeze gates still block held-out work.

## POS checkpoint-566 callbacks reach 500/1800 — 13:59 SAST

- Full-prefix recovery b0 `1238989` and b1 `1238990` remain healthy on
  `srvrocgpu010` A100-40GB after about 3h33m. Both checkpoint-566 frozen
  exact callbacks reached `500/1,800` rows by 13:57, with fresh logs and no
  fault marker. Sustained serial throughput puts both complete artifacts
  around 15:30--15:40 SAST. Partial callback output is operational only and
  cannot affect retention or candidate ranking.
- Fused row-batch v3 gate/b2 `1239042` remains Resources-pending on
  `gpu:ampere`; Slurm's conservative start is
  `2026-08-17 10:26:08 SAST`, though it may start when a recovery GPU
  releases. Owned state remains two running plus one pending A100-40GB job,
  with no A100-80GB or L40S execution. HEX quota remains home `70.3%`,
  scratch `38.9%`.
- Terminal trusted counts remain base `16/16`, NER `11/11` plus `2/4`
  confirmations, T2X `11/11` plus `4/4`, POS `3/11`, winners `0/8`,
  held-out `0`, and Mono not started. Kombuys was not accessed; its latest
  verified state remains read-only and idle. Sheet E/F/G remain blank,
  quarantined rows unpublished, Hugging Face blocked, and the remaining
  validation/winner-freeze gates still block held-out work.

## POS recovery first artifacts verified; second callbacks active — 13:32 SAST

- Full-prefix recovery b0 `1238989` and b1 `1238990` completed their first
  frozen exact callbacks at checkpoint 283 over all `1,800/1,800` rows, 12
  cells, and 17 labels under `closed_label_tuple_mean_logprob_v1`. B0 exact
  accuracy is `0.7144075908238836`; b1 is `0.621204427395874`. Artifact
  SHA-256 values are
  `477026720accb5060d8fdbc68247c0c6fa93b5d4cb6eb0207587dd02b8b9984f`
  and `b7a8ddb64bd89e7beaa222ae4c37301f76ec6e0ef01c67251dc1bf754596e4d0`.
  Trainer-state/adapter SHA-256 values are respectively
  `f0926851464ecfcd73920a81173bf1e6f34c51b0d82811fc675d566560eb537f` /
  `214f621990b1bbf192562b17c00014c03b14aed050f6dd01bf7ed74daf82d80f`
  and
  `7f0b4314d12f570644b47cf228db87900faf8b1accd6402d3771e2c47bd5a113` /
  `18dfc369c58c6a3fedaeeb41d82ed49c84a8a722190f74ed91b827e5fb4a1154`.
  Each recovery run retained its own checkpoint 283. The cancelled b0
  `1238876` metric remains provenance-only and is not mixed into recovery
  selection.
- Both jobs resumed healthily and reached checkpoint 566. Their second exact
  callbacks are active: b0 reached `150/1,800` at 13:28 and b1 reached
  `100/1,800` at 13:27, with no fault marker. Sustained serial throughput
  puts complete second artifacts around 15:25--15:40 SAST. These are valid
  within-candidate validation artifacts, not terminal candidate results or
  cross-candidate rankings; trusted terminal POS therefore remains `3/11`.
- Fused row-batch v3 gate/b2 `1239042` remains Resources-pending on
  `gpu:ampere`, with Slurm's conservative start at
  `2026-08-17 10:26:16 SAST`; it may start when a recovery GPU releases.
  Owned state is two running plus one pending A100-40GB job, with no
  A100-80GB or L40S execution. HEX quota is home `70.3%`, scratch `38.9%`.
- Operationally both recovery candidates now have one verified exact
  artifact, but terminal trusted counts remain base `16/16`, NER `11/11`
  plus `2/4` confirmations, T2X `11/11` plus `4/4`, POS `3/11`, winners
  `0/8`, held-out `0`, and Mono not started. Kombuys was not accessed in
  this pass; its latest verified state remains read-only and idle. Sheet
  E/F/G remain blank, quarantined rows unpublished, Hugging Face blocked,
  and the remaining validation/winner-freeze gates still block held-out
  evaluation.

## POS recovery callbacks reach 1750/1800 — 13:00 SAST

- Full-prefix recovery jobs `1238989/1238990` remain healthy on
  `srvrocgpu010` A100-40GB after about 2h34m. Both first exact callbacks had
  reached `1,750/1,800` rows by 12:59, with fresh logs and no fault marker.
  Only 50 rows remain per job, putting complete artifacts around 13:03--13:08
  SAST. Partial output remains operational only until each artifact closes
  and verifies.
- Fused row-batch gate/b2 `1239042` remains Resources-pending on
  `gpu:ampere`; Slurm still displays `2026-08-17 10:26:16`, but it may start
  when either recovery job releases a GPU. Owned state is two running plus
  one pending A100-40GB job, no A100-80GB/L40S execution. Quota remains home
  `70.3%`, scratch `38.8%`.
- Kombuys remains read-only with no execution in this pass. Trusted counts,
  Sheet E/F/G, held-out status, and publication gates are unchanged. The next
  material checks are recovery-artifact verification and the v3 real-model
  equivalence/speed gate.

## POS recovery callbacks reach 1350/1800 — 12:29 SAST

- Full-prefix recovery jobs `1238989/1238990` remain healthy on
  `srvrocgpu010` A100-40GB after about 2h03m. At 12:26/12:28 their first
  exact validation callbacks had each reached `1,350/1,800` rows; logs were
  fresh and targeted scans remained free of traceback, OOM, CUDA/NCCL, and
  other fault markers. The remaining 450 rows keep complete-artifact ETA at
  about 13:00--13:10 SAST. Partial output is operational only.
- Fused row-batch gate/b2 `1239042` remains Resources-pending on
  `gpu:ampere`; Slurm still shows the conservative
  `2026-08-17 10:26:16` projection, although it may start when either running
  job releases a GPU. Owned state is two running plus one pending A100-40GB
  job, with no A100-80GB/L40S execution. Quota remains home `70.3%`, scratch
  `38.8%`.
- Kombuys remains read-only with no execution in this pass. Trusted counts,
  held-out state, Sheet E/F/G, and publication gates remain unchanged; the
  immediate blockers are terminal recovery artifacts and the unratified v3
  equivalence/speed gate.

## POS recovery callbacks reach 950/1800 — 12:00 SAST

- Full-prefix recovery jobs `1238989/1238990` remain healthy on
  `srvrocgpu010` A100-40GB after about 1h33m. At 11:56/11:58 their first
  exact validation callbacks had each reached `950/1,800` rows; logs were
  fresh and targeted scans showed no traceback, OOM, CUDA/NCCL, or other
  fault marker. Sustained throughput still projects both complete artifacts
  around 13:00--13:10 SAST. Partial output is operational only and cannot
  affect retention or ranking.
- Fused row-batch gate/b2 `1239042` remains Resources-pending on
  `gpu:ampere`, with Slurm's conservative projection still
  `2026-08-17 10:26:16`; it may start earlier when a recovery job releases a
  GPU. Owned state is two running plus one pending A100-40GB job, no
  A100-80GB/L40S execution. Quota is home `70.3%`, scratch `38.8%`.
- The latest read-only Kombuys state remains idle and no access or execution
  occurred in this pass. Trusted counts remain base `16/16`, NER `11/11`
  plus `2/4` confirmations, T2X `11/11` plus `4/4`, POS `3/11`, winners
  `0/8`, held-out `0`, Mono not started. Sheet E/F/G remain blank; held-out,
  publication, remaining-validation, and winner-freeze gates remain closed.

## POS recovery callbacks reach 550/1800 — 11:30 SAST

- Full-prefix recovery jobs `1238989/1238990` remain healthy on
  `srvrocgpu010` A100-40GB. At 11:29 their first exact validation callbacks
  had each reached `550/1,800` rows, with fresh logs and no fault marker.
  Sustained callback throughput projects both complete artifacts around
  13:00--13:10 SAST. Partial output remains operational only and cannot alter
  checkpoint retention or candidate ranking.
- Fused row-batch gate/b2 `1239042` remains Resources-pending on the same
  `gpu:ampere` family; Slurm's conservative projection remains
  `2026-08-17 10:26:16`. Owned state is two running plus one pending
  A100-40GB job, with no A100-80GB or L40S execution. HEX quota is unchanged
  at home 70.3% and scratch 38.8%.
- Kombuys was checked read-only and remains idle: RTX 5090 `10 MiB/0%`, RTX
  3080 Ti `1 MiB/0%`, scratch `60%`, and only the `tailscale-kombuys` tmux.
  Scientifically trusted counts, held-out state, Sheet E/F/G, and publication
  gates are unchanged.

## POS recovery callbacks reach 150/1800 — 10:59 SAST

- Full-prefix recovery jobs `1238989/1238990` remain healthy on
  `srvrocgpu010` A100-40GB. Their first exact validation callbacks reached
  `150/1,800` rows at 10:56:03 and 10:56:32 respectively, with fresh logs and
  no fault marker. Sustained callback throughput projects both complete
  artifacts around 12:58--13:10 SAST. Partial output is operational only and
  cannot affect checkpoint retention or candidate ranking.
- Fused row-batch gate/b2 `1239042` remains Resources-pending on the same
  `gpu:ampere` family. Slurm's conservative projection is
  `2026-08-17 10:26:16`, but it may start when either recovery job releases a
  GPU. Owned state is two running plus one pending A100-40GB job; no
  A100-80GB/L40S or Kombuys execution exists. Quota remains home 70.3% and
  scratch 38.8%. Scientifically trusted counts, Sheet E/F/G, held-out state,
  and publication gates are unchanged.

## Cross-row batching v3 deployed; fused b2 gate queued — 10:48 SAST

- The prospective v3 amendment was written at 10:40 before implementation or
  optimized measurement. It fixes row batch size 8 and retains full-prefix
  `use_cache=False` inference. Acceptance remains exact row predictions/order,
  token and cell counts, cell metrics, and aggregate accuracy; score drift is
  bounded at maximum 0.05 and mean 0.01, and elapsed time must be at most one
  third of serial. No held-out split may be loaded. The two failed cache paths
  remain prohibited.
- The minimal implementation adds one cross-row scorer and routes
  `PosEvaluator` through it only when `SALLM_POS_ROW_BATCH_SIZE=8`. Sixteen
  targeted tests pass, including serial/batched prediction-score equivalence
  with fewer forward calls and the fail-closed verifier. Ruff lint/format,
  shell syntax, and whitespace checks also pass.
- Immutable snapshot
  `/home/lmbanr001/masters/sallm_snapshots/pure-gdn-hpo-pos-rowbatch-v3p-20260816-ad16dfe5`
  is based on the clean full-prefix recovery snapshot, is read-only, verifies
  698 source/config hashes, and has independent replacement/manifest inodes.
  Deployment manifest SHA-256 is
  `4ab77afa26415706beff8524bb6a5970512db9b76b91d0734b04b04ab324a930`;
  batched scorer, evaluator, benchmark, verifier, and wrapper hashes are
  `f9bd4b9f...646a`, `49ff23f8...6344`, `cc323b3f...d16b`,
  `1ac7217f...16f`, and corrected wrapper `ad16dfe5...ea48`.
- Full-prefix b2 `1238991` was confirmed pending with no canonical checkpoint
  or logging output directories, then cancelled at 10:46:59 with exactly
  `00:00:00` elapsed. Fused gate/b2 replacement `1239041` was submitted at
  10:47:00 under the required `nlpgroup/a100/nlpgroup`, `gpu:ampere:1`,
  24-hour, eight-CPU, home-chdir envelope. It is Resources-pending and Slurm
  initially projected `2026-08-17 10:26:16`. A final pre-start audit found its
  batch/output overrides were present in immutable launcher source but would
  be exported after execution-manifest creation. Job `1239041` was therefore
  cancelled pending-only at `00:00:00`. Corrected fused job `1239042` was
  submitted at 10:49:29 from the snapshot/hash above; it exports batch size 8
  and isolated run/output/logging overrides before creating the manifest. It
  remains pending without a reliable start estimate. The gate will run before
  b2 loads training data or updates weights and will abort on any failed check.
- Running recovery jobs `1238989/1238990` were not interrupted. Both remain
  healthy on `srvrocgpu010`, have reached step 283, and completed declared
  health-only validation over 1,800 rows. Home/scratch quota is 70.3%/38.8%.
  `srvrocgpu009` remains idle but its `gpu:amperemk` association is unavailable;
  no L40S/A100-80GB, Kombuys, held-out, Sheet E/F/G, or Hugging Face action
  occurred. V3 is deployed and queued, but no real-model speedup is claimed
  until job `1239042` passes the frozen gate.

## POS gate fused into b1 allocation — 10:09 SAST

- Because `gpu:amperemk` is unavailable and a standalone gate would consume a
  complete additional queue turn, the preregistration was prospectively
  amended to run the unchanged full-prefix-versus-cache gate at the start of
  b1's original `gpu:ampere` allocation. The gate still uses frozen b0
  checkpoint 566 and the fixed 12 validation-only cells, before b1 loads
  training data or updates weights. B1 can proceed only if exact prediction,
  cell, aggregate, score-drift, speed, and provenance gates pass.
- Eighteen targeted local tests pass, including the executable gate verifier;
  Ruff, shell syntax, and whitespace checks pass. Immutable fused snapshot is
  `/home/lmbanr001/masters/sallm_snapshots/pure-gdn-hpo-poscache-fused-20260816-5ed8c304`;
  verifier and wrapper hashes are
  `ff88a86e767dd66bbf449189d36a7a23703d5ba0f0b543dd7eb457caeee666c2`
  and `2938482c0d0ef1436c7517f9e57a2e0ae30eda296293e86dacbe057d0502a709`.
- Standalone gate `1238975` was cancelled pending-only and replaced by fused
  b1 job `1238979`. It has the required `nlpgroup/a100/nlpgroup`,
  `gpu:ampere:1`, 24-hour, eight-CPU, home-chdir envelope and is currently
  Resources-pending without a reliable start estimate. Running b0 `1238876`
  remains unchanged and healthy; its checkpoint-849 legacy callback reached
  `1,100/1,800` rows at 10:03. The cache speedup is therefore implemented and
  queued, but not yet active or ratified. It affects POS exact validation
  only; training and other task-family evaluators are unchanged.

## POS cache optimization implemented; equivalence gate queued — 10:00 SAST

- A prospective implementation-only optimization was preregistered at 09:49
  in `2026-08-16-pure-gdn-pos-runtime-optimization-preregistration.md` before
  any optimized run. The new POS path reuses FLA's recurrent cache instead of
  recomputing the growing prefix for every token; tokenization, legal labels,
  selected-label log-probability rule, aggregation, and early stopping are
  unchanged. The legacy path remains available as the reference. Seventeen
  targeted local tests pass, including a cache-aware regression proving
  identical predictions/scores with fewer processed tokens; Ruff, shell
  syntax, and whitespace checks pass.
- The byte-verified, read-only gate snapshot is
  `/home/lmbanr001/masters/sallm_snapshots/pure-gdn-hpo-poscache-20260816-34a3f7d5`.
  Key source hashes are constrained scorer
  `f2410699c86158933c39967709d4b7ef1b0106815bd038d39871d87a928d7a0f`,
  POS evaluator
  `3273f3a88c284e11d237a4eee9885074325f7ffdf375b90769519c155086f834`,
  benchmark
  `e0aab85d3a0377a7858820923016a250851b9bcd1055f5463060434bb802dd0c`,
  and gate launcher
  `aac78d35a22480b46a59c06f5a7c88050bb1e831f8f02d596a6e889b0469b956`.
  Unchanged snapshot files are hard-linked to avoid another 1.9 GB home
  charge; the quota refresh fell from the transient 88.6% to 83.4%, with
  scratch still 38.8%.
- Pending-only b1/b2 jobs `1238877/1238878` were cancelled at 09:54 before
  either started or accessed model/data. Their records remain provenance.
  Full-prefix-versus-cache validation-only gate job `1238975` is queued on
  `gpu:ampere` against frozen b0 checkpoint 566 and the deterministic first
  row from each of 12 language/template cells. It is pending Resources with
  Slurm's conservative projection `2026-08-17 03:20:32`; the node's other
  three GPUs are occupied by other users, and running b0 `1238876` is
  untouched on the fourth.
- The planned `gpu:amperemk` gate cannot be submitted: live Slurm association
  policy reports `nlpgroup` `GrpTRES gpu:amperemk=0` and rejects the request
  with `AssocGrpGRES`, despite `srvrocgpu009` being idle with four A100-40GB
  GPUs. No workaround was attempted. Hardware expansion and b1/b2
  resubmission remain blocked pending an allowed `amperemk` association and
  the ordinary implementation gate. No held-out data, Sheet E/F/G, Kombuys,
  A100-80GB, or L40S was touched.

## POS b0 checkpoint 849 callback reaches 450/1800 — 09:17 SAST

- Stage-B b0 job `1238876` remains healthy on `srvrocgpu010` A100-40GB. Its
  third frozen constrained callback reached `450/1,800` rows at `09:15:52`
  with a fresh log and no fault marker. Sustained throughput narrows the
  complete-artifact ETA to `10:55--11:05 SAST`. Partial output remains
  operational only and cannot influence retention or ranking; checkpoint 566
  remains retained at exact accuracy `0.7845538768361983`, patience `0/2`.
- Jobs `1238877/1238878` remain Resources/Priority-pending without model or
  data access. B1 projects `2026-08-17 03:20:32 SAST`; b2 has no projection.
  Owned state is one running plus two pending A100-40GB jobs at cap, with no
  A100-80GB/L40S. HEX quota remains home `70.3%`, scratch `38.8%`.
- Terminal trusted counts remain base `16/16`, NER `11/11` plus `2/4`
  confirmations, T2X `11/11` plus `4/4` confirmations, POS `3/11`, winners
  `0/8`, held-out `0`, Mono not started. Kombuys remains read-only and idle
  at RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `60%`, only
  `tailscale-kombuys` tmux. Sheet E/F/G remain blank, quarantined rows
  unpublished, and remaining validation and winner-freeze gates still block
  held-out work and a fixed full-program ETA.

## POS b0 checkpoint 566 retained; checkpoint 849 callback active — 08:48 SAST

- Stage-B b0 job `1238876` completed checkpoint 566's frozen exact artifact
  over all `1,800` rows, 12 cells, and 17 labels at accuracy
  `0.7845538768361983` under `closed_label_tuple_mean_logprob_v1`. This
  exceeds checkpoint 283's `0.7099399341886862`, so checkpoint 566 is now
  retained with patience `0/2`. Artifact, trainer-state, and adapter SHA-256
  values are
  `027c336592042f65dad6f5e0f1378f9854f91db16ebae32bd86be49c127822ed`,
  `297547262d9e59ea3538d663ae320b24014aa21666ec47ad90eb5c1d44b8fb04`,
  and `5449ef4b1985295a6ea002d2951650266b45651c8cd444820fd70ecc8274df6c`.
  This is valid within-candidate validation evidence, not a terminal b0
  result or cross-candidate ranking.
- B0 remains healthy on `srvrocgpu010` A100-40GB. It reached checkpoint
  `849/4245`, evaluated all `1,800/1,800` declared rows at health-only loss
  `0.14097159915500218`, and began its third frozen callback, reaching
  `50/1,800` rows at `08:46:18`. The complete step-849 exact artifact is
  estimated around `10:55--11:10 SAST`.
- Jobs `1238877/1238878` remain Resources/Priority-pending without model or
  data access. B1 projects `2026-08-17 03:20:32 SAST`; b2 has no projection.
  Owned state is one running plus two pending A100-40GB jobs at cap, with no
  A100-80GB/L40S. HEX quota remains home `70.3%`, scratch `38.8%`.
- Terminal trusted counts remain base `16/16`, NER `11/11` plus `2/4`
  confirmations, T2X `11/11` plus `4/4` confirmations, POS `3/11`, winners
  `0/8`, held-out `0`, Mono not started. Kombuys remains read-only and idle
  at RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `60%`, only
  `tailscale-kombuys` tmux. Sheet E/F/G remain blank, quarantined rows
  unpublished, and remaining validation and winner-freeze gates still block
  held-out work and a fixed full-program ETA.

## POS b0 checkpoint 566 callback reaches 1650/1800 — 08:18 SAST

- Stage-B b0 job `1238876` remains healthy on `srvrocgpu010` A100-40GB. Its
  second frozen constrained callback reached `1,650/1,800` rows at
  `08:15:40` with a fresh log and no fault marker. Only 150 rows remain;
  sustained throughput narrows the complete-artifact ETA to
  `08:25--08:30 SAST`. Partial output remains operational only and cannot
  influence retention or ranking; checkpoint 283 is provisionally retained
  at exact accuracy `0.7099399341886862`, patience `0/2`.
- Jobs `1238877/1238878` remain Resources/Priority-pending without model or
  data access. B1 projects `2026-08-17 03:20:32 SAST`; b2 has no projection.
  Owned state is one running plus two pending A100-40GB jobs at cap, with no
  A100-80GB/L40S. HEX quota remains home `70.3%`, scratch `38.8%`.
- Terminal trusted counts remain base `16/16`, NER `11/11` plus `2/4`
  confirmations, T2X `11/11` plus `4/4` confirmations, POS `3/11`, winners
  `0/8`, held-out `0`, Mono not started. Kombuys remains read-only and idle
  at RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `60%`, only
  `tailscale-kombuys` tmux. Sheet E/F/G remain blank, quarantined rows
  unpublished, and remaining validation and winner-freeze gates still block
  held-out work and a fixed full-program ETA.

## POS b0 checkpoint 566 callback reaches 1250/1800 — 07:48 SAST

- Stage-B b0 job `1238876` remains healthy on `srvrocgpu010` A100-40GB. Its
  second frozen constrained callback reached `1,250/1,800` rows at
  `07:45:20` with a fresh log and no fault marker. Sustained throughput
  narrows the complete-artifact ETA to `08:25--08:35 SAST`. Partial output
  remains operational only and cannot influence retention or ranking;
  checkpoint 283 is provisionally retained at exact accuracy
  `0.7099399341886862`, patience `0/2`.
- Jobs `1238877/1238878` remain Resources/Priority-pending without model or
  data access. B1 projects `2026-08-17 03:20:32 SAST`; b2 has no projection.
  Owned state is one running plus two pending A100-40GB jobs at cap, with no
  A100-80GB/L40S. HEX quota remains home `70.3%`, scratch `38.8%`.
- Terminal trusted counts remain base `16/16`, NER `11/11` plus `2/4`
  confirmations, T2X `11/11` plus `4/4` confirmations, POS `3/11`, winners
  `0/8`, held-out `0`, Mono not started. Kombuys remains read-only and idle
  at RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `60%`, only
  `tailscale-kombuys` tmux. Sheet E/F/G remain blank, quarantined rows
  unpublished, and remaining validation and winner-freeze gates still block
  held-out work and a fixed full-program ETA.

## POS b0 checkpoint 566 callback reaches 850/1800 — 07:18 SAST

- Stage-B b0 job `1238876` remains healthy on `srvrocgpu010` A100-40GB. Its
  second frozen constrained callback reached `850/1,800` rows at `07:15:05`
  with a fresh log and no fault marker. Sustained throughput keeps the
  complete-artifact ETA at `08:25--08:40 SAST`. Partial output remains
  operational only and cannot influence retention or ranking; checkpoint 283
  is provisionally retained at exact accuracy `0.7099399341886862`, patience
  `0/2`.
- Jobs `1238877/1238878` remain Resources/Priority-pending without model or
  data access. B1 projects `2026-08-17 03:20:32 SAST`; b2 has no projection.
  Owned state is one running plus two pending A100-40GB jobs at cap, with no
  A100-80GB/L40S. HEX quota remains home `70.3%`, scratch `38.8%`.
- Terminal trusted counts remain base `16/16`, NER `11/11` plus `2/4`
  confirmations, T2X `11/11` plus `4/4` confirmations, POS `3/11`, winners
  `0/8`, held-out `0`, Mono not started. Kombuys remains read-only and idle
  at RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `60%`, only
  `tailscale-kombuys` tmux. Sheet E/F/G remain blank, quarantined rows
  unpublished, and remaining validation and winner-freeze gates still block
  held-out work and a fixed full-program ETA.

## POS b0 checkpoint 566 callback reaches 450/1800 — 06:48 SAST

- Stage-B b0 job `1238876` remains healthy on `srvrocgpu010` A100-40GB. Its
  second frozen constrained callback reached `450/1,800` rows at `06:46:04`
  with a fresh log and no fault marker. Observed throughput supports the
  existing `08:25--08:40 SAST` complete-artifact window. Partial callback
  output is operational only and is not used for checkpoint retention or
  candidate ranking; checkpoint 283 remains provisionally retained at exact
  accuracy `0.7099399341886862`, patience `0/2`.
- Jobs `1238877/1238878` remain Resources/Priority-pending without model or
  data access. B1 projects `2026-08-17 03:20:32 SAST`; b2 has no projection.
  Owned state is one running plus two pending A100-40GB jobs at cap, with no
  A100-80GB/L40S. HEX quota remains home `70.3%`, scratch `38.8%`.
- Terminal trusted counts remain base `16/16`, NER `11/11` plus `2/4`
  confirmations, T2X `11/11` plus `4/4` confirmations, POS `3/11`, winners
  `0/8`, held-out `0`, Mono not started. Kombuys remains read-only and idle
  at RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `60%`, only
  `tailscale-kombuys` tmux. Sheet E/F/G remain blank, quarantined rows
  unpublished, and remaining validation and winner-freeze gates still block
  held-out work and a fixed full-program ETA.

## POS b0 checkpoint 566 exact callback active — 06:18 SAST

- Stage-B b0 job `1238876` remains healthy on `srvrocgpu010` A100-40GB. It
  reached checkpoint `566/4245`, completed all `1,800/1,800` declared rows
  at health-only loss `0.18320615980360244`, and began the second frozen
  constrained callback, reaching `50/1,800` rows at `06:16:24`. No step-566
  exact artifact or selection decision exists yet. Observed callback timing
  keeps the completion estimate at `08:25--08:40 SAST`; checkpoint 283
  remains provisionally retained at exact accuracy `0.7099399341886862` and
  patience `0/2` until the complete artifact is compared.
- B1/b2 jobs `1238877/1238878` remain Resources/Priority-pending without
  model or data access. Slurm projects b1 for
  `2026-08-17 03:20:32 SAST`; b2 has no projection. Owned state remains one
  running plus two pending A100-40GB jobs at cap, with no A100-80GB/L40S.
  HEX quota is home `70.3%`, scratch `38.8%`.
- This is operational progress only. Terminal trusted counts remain base
  `16/16`, NER `11/11` plus `2/4` confirmations, T2X `11/11` plus `4/4`
  confirmations, POS `3/11`, winners `0/8`, held-out `0`, and Mono not
  started. Kombuys remains read-only and idle at RTX 5090 `10 MiB/0%`, RTX
  3080 Ti `1 MiB/0%`, scratch `60%`, only `tailscale-kombuys` tmux. Sheet
  E/F/G remain blank, quarantined rows unpublished, and remaining validation
  grids and winner freezes block held-out work and a fixed full-program ETA.

## POS b0 first exact validation artifact — 05:59 SAST

- Stage-B b0 job `1238876` completed its first frozen validation callback at
  checkpoint `283`: exact accuracy `0.7099399341886862` over all `1,800`
  rows, 12 cells, and 17 labels under
  `closed_label_tuple_mean_logprob_v1`. The artifact SHA-256 is
  `394219b33d242dad5339d1d67e2571268e5296049a9ce76fba127fffc82f140b`;
  checkpoint trainer-state and adapter hashes are
  `8dbdb7f7f4f32992579e9d1a14e0bdd100929ab25062ec5f973a6a2827ccbb27`
  and `fa518b2e999c014fbf8ac8521bac89d87e0372cf22466e52439a03761940c548`.
  Checkpoint 283 is retained with early-stopping counter `0/2`. This is valid
  within-candidate validation evidence, not a terminal b0 result or a
  cross-candidate ranking; b0 resumed training healthily on
  `srvrocgpu010` A100-40GB. Its next exact artifact, at step 566, is
  estimated around `08:25--08:40 SAST`.
- B1/b2 jobs `1238877/1238878` remain Resources/Priority-pending without
  model or data access. Slurm projects b1 for
  `2026-08-17 03:20:32 SAST`; b2 has no projection. Owned state remains one
  running plus two pending A100-40GB jobs at the three-job cap, with no
  A100-80GB/L40S. HEX quota is home `70.3%`, scratch `38.8%`.
- Terminal scientifically trusted counts remain base `16/16`, NER `11/11`
  plus `2/4` confirmations, T2X `11/11` plus `4/4` confirmations, POS
  `3/11`, winners `0/8`, held-out `0`, and Mono not started. Kombuys remains
  read-only and idle at RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`,
  scratch `60%`, only `tailscale-kombuys` tmux. Sheet E/F/G remain blank,
  quarantined rows unpublished, and the remaining validation grids and
  winner freezes continue to block held-out work and a fixed full-program
  ETA.

## POS b0 constrained validation reaches 1250/1800 — 05:18 SAST

- Stage-B b0 `1238876` remains healthy on `srvrocgpu010` A100-40GB and
  reached `1,250/1,800` frozen constrained rows at `05:15:01`. The log is
  fresh with no fault marker and no premature selection artifact. Sustained
  throughput keeps the first complete exact accuracy near
  `05:55--06:05 SAST`; no checkpoint or candidate comparison is permitted
  before that artifact closes.
- B1/b2 `1238877/1238878` remain Resources/Priority-pending without model or
  data access. B1's dynamic start remains `2026-08-17 03:20:32 SAST`; b2 has
  no projection. Owned state is one running plus two pending A100-40GB jobs
  at cap, no A100-80GB/L40S; HEX quota remains home `70.3%`, scratch `38.8%`.
- Trusted progress is unchanged: base `16/16`, NER `11/11` plus `2/4`
  confirmations, T2X `11/11` plus `4/4` confirmations, POS `3/11`, winners
  `0/8`, held-out `0`, Mono not started. Kombuys remains read-only and idle
  at RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `60%`, only
  `tailscale-kombuys` tmux. Sheet E/F/G remain blank, quarantined rows
  unpublished, and publication and held-out gates remain closed.

## POS b0 constrained validation reaches 850/1800 — 04:48 SAST

- Stage-B b0 `1238876` remains healthy on `srvrocgpu010` A100-40GB and
  reached `850/1,800` frozen constrained rows at `04:44:23`. The log remains
  fresh with no fault marker and no complete selection artifact. Sustained
  callback throughput projects the first exact accuracy around
  `05:55--06:05 SAST`; partial progress is operational only and has not
  changed retention or ranking.
- B1/b2 `1238877/1238878` remain Resources/Priority-pending without model or
  data access. B1's dynamic start remains `2026-08-17 03:20:32 SAST`; b2 has
  no projection. Owned state is one running plus two pending A100-40GB jobs
  at cap, with no A100-80GB/L40S. HEX quota remains home `70.3%`, scratch
  `38.8%`.
- Trusted counts remain base `16/16`, NER `11/11` plus `2/4` confirmations,
  T2X `11/11` plus `4/4` confirmations, POS `3/11`, winners `0/8`, held-out
  `0`, Mono not started. Kombuys remains read-only and idle at RTX 5090
  `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `60%`, only
  `tailscale-kombuys` tmux. Sheet E/F/G stay blank, quarantined rows remain
  unpublished, and publication and held-out gates remain closed.

## POS b0 constrained validation reaches 450/1800 — 04:18 SAST

- Stage-B b0 job `1238876` remains healthy on `srvrocgpu010` A100-40GB and
  reached `450/1,800` frozen constrained rows at `04:14:54`. Its log is fresh,
  no runtime fault marker or complete selection artifact exists, and the
  epoch-1 health-only loss remains `0.34238827175564235`. Observed callback
  throughput keeps the first exact accuracy estimate at about
  `05:55--06:10 SAST`; this is operational progress only and cannot yet alter
  checkpoint retention or ranking.
- B1/b2 jobs `1238877/1238878` remain Resources/Priority-pending without
  model or data access. B1's dynamic projection remains
  `2026-08-17 03:20:32 SAST`; b2 has no projection. Owned state remains one
  running plus two pending A100-40GB jobs at the three-job cap, with no
  A100-80GB/L40S work. HEX quota is home `70.3%`, scratch `38.8%`.
- Scientifically trusted counts remain base `16/16`, NER seed-42 `11/11`,
  NER confirmations `2/4`, T2X seed-42 `11/11`, T2X confirmations `4/4`,
  POS seed-42 `3/11`, winners `0/8`, held-out `0`, and Mono not started.
  Kombuys remains read-only and idle at GPU0 RTX 5090 `10 MiB/0%`, GPU1 RTX
  3080 Ti `1 MiB/0%`, scratch `60%`, only `tailscale-kombuys` tmux. Sheet
  E/F/G remain blank, quarantined rows unpublished, and publication blocked;
  remaining validation grids and winner freezes still block held-out work.

## POS b0 first constrained validation active — 03:48 SAST

- Stage-B b0 job `1238876` remains healthy on `srvrocgpu010` A100-40GB.
  It completed epoch-1 training at step `283/4245`, then evaluated all
  `1,800/1,800` declared rows at health-only loss
  `0.34238827175564235`. The frozen
  `closed_label_tuple_mean_logprob_v1` callback reached `100/1,800` rows at
  `03:46:41`; no complete selection artifact or accuracy exists yet, so no
  checkpoint comparison or ranking has occurred. Current throughput puts the
  first exact artifact roughly around `05:50--06:10 SAST`, subject to callback
  variance.
- Jobs `1238877/1238878` remain pending for Resources/Priority without log,
  model, or data access. Slurm still projects b1 for
  `2026-08-17 03:20:32 SAST` and gives b2 no current projection. Owned state
  is one running plus two pending A100-40GB jobs, exactly the three-job cap;
  no A100-80GB or L40S work is owned. HEX quota remains home `70.3%`, scratch
  `38.8%`.
- This is operational validation-only progress; scientifically trusted
  terminal counts remain base `16/16`, NER seed-42 `11/11`, NER
  confirmations `2/4`, T2X seed-42 `11/11`, T2X confirmations `4/4`, POS
  seed-42 `3/11`, winners `0/8`, held-out `0`, and Mono not started. Kombuys
  remains read-only and idle at GPU0 RTX 5090 `10 MiB/0%`, GPU1 RTX 3080 Ti
  `1 MiB/0%`, scratch `60%`, with only `tailscale-kombuys` tmux. Sheet E/F/G
  remain blank, quarantined rows remain unpublished, and Hugging Face
  publication remains blocked. The remaining validation grids and winner
  freezes still prevent a fixed full-program ETA.

## POS Stage A terminal-valid; Stage B released — 03:21 SAST

- POS a2 job `1238228` completed cleanly `0:0` at `03:04:33 SAST` after
  `22:53:12`. Its step-2547 exact validation accuracy is
  `0.8509148607619882` over all `1,800` declared rows, 12
  language/template cells, and 17 closed labels under
  `closed_label_tuple_mean_logprob_v1`. This is the frozen second
  consecutive threshold miss after step 2264, so checkpoint 1981 remains
  retained at exact accuracy `0.8594919779162872` and the run terminated
  normally. The final metric artifact SHA-256 is
  `913219f05adb5d3c8264cdd59b750cbc0e49dcde4942852b22a1afc2fb7455c5`;
  retained trainer-state and adapter hashes are
  `829494f8d734a64cefb4720e2aa72298634c02921d0c7eea6d17e07507a1b82d`
  and
  `9683f8022fe52267b74d2d2f550abb0f197d36743c3128ef22092c8a96a591c0`.
- Final-adapter weight/config SHA-256 values are
  `2a7501771db946db6ba56460ff3c4750923e0b04149de40f03663c4062404099`
  and
  `b7033604b2d431c88b2c146b284b575d32cc7b4d3add70b32392c3226d6a4ade`.
  A targeted read-only tensor comparison proved identical key sets and exact
  equality for all `424/424` tensors (`71,762,560` values) between retained
  checkpoint 1981 and `final_adapter`. All three POS Stage-A candidates are
  therefore terminal-valid; a2 is only the provisional Stage-A leader and
  no POS recipe has been frozen.
- After confirming zero owned jobs, no A100-80GB/L40S work, absent canonical
  b0/b1/b2 output paths, and the immutable POS-source snapshot, fixed Stage-B
  jobs b0/b1/b2 were submitted as `1238876/1238877/1238878`. All use
  `nlpgroup/a100/nlpgroup`, one `gpu:ampere`, 24 hours, eight CPUs, one node,
  `/home/lmbanr001/masters/sallm`, seed 42, and snapshot
  `/home/lmbanr001/masters/sallm_snapshots/pure-gdn-hpo-possourcefix-20260814-d9501087`.
  Job `1238876` started on `srvrocgpu010` at `03:20:32`; it wrote execution
  manifest SHA-256
  `06b825ae37f4eea1d97dd799193f9e445f76bed5cfede53faeb2e18c1f5c22b1`,
  verified `694/694` immutable files, passed the pure-GDN fast-path gate, and
  loaded exactly `2,259` train and `1,800` validation rows with `2,325,312`
  trainable parameters. It has no fault marker while tokenizing the declared
  data. Its trial artifact SHA-256 is
  `e3cb975276810bebfc6c487e2d8f86240f6edac4b96338f168b3f8c0afb9babf`.
  Jobs `1238877/1238878` remain pending for Resources/Priority with no log,
  model, or data access; Slurm currently projects b1 for
  `2026-08-17 03:20:32 SAST` and provides no projection for b2. Queue
  projections are dynamic.
- Operationally, all three POS Stage-A jobs are complete and one Stage-B job
  is running. Scientifically trusted terminal progress is base `16/16`, NER
  seed-42 `11/11`, NER confirmations `2/4`, T2X seed-42 `11/11`, T2X
  confirmations `4/4`, and POS seed-42 `3/11`; global winners remain `0/8`,
  held-out remains `0`, and Mono has not started. Owned GPU-family state is
  one running plus two pending A100-40GB jobs, exactly the three-job cap,
  with no A100-80GB or L40S work. HEX quota is home `70.3%`, scratch `38.8%`.
  Kombuys remains read-only and idle at GPU0 RTX 5090 `10 MiB/0%`, GPU1 RTX
  3080 Ti `1 MiB/0%`, scratch `60%`, with only `tailscale-kombuys` tmux.
  Sheet E/F/G remain blank, quarantined rows remain unpublished, and Hugging
  Face publication remains blocked. POS Stage B, its confirmations, the
  remaining task grids, and winner freezes still block held-out evaluation;
  no fixed full-program ETA is scientifically supportable.

## POS a2 first miss at step 2264 — 00:49 SAST

- POS a2 `1238228` step 2264 scored exact validation accuracy
  `0.8594007031274665` over all `1,800` declared rows, 12 language/template
  cells, and 17 closed labels under
  `closed_label_tuple_mean_logprob_v1`. This is `0.0000912747888207` below
  retained checkpoint 1981 accuracy `0.8594919779162872`, so it is the first
  frozen threshold miss and EarlyStopping patience is `1/2`. Artifact,
  trainer-state, and current-step adapter SHA-256 values are
  `e6c755c85a98a442066928dd7a183b6bc3d3a75b631dc0d24ebbfad5267f5647`,
  `4330749c6f43ecfc855ad68bc094aa47f3d8c0ebb80cfad8375ae62261d5978d`,
  and
  `c0efc576b1f654a34a41b041c50a8a9147a351056ebce51625a554840b5b2ff5`.
  This remains clean validation-only operational evidence, not terminal
  trusted evidence.
- The job remains healthy on `srvrocgpu010` A100-40GB and has reached step
  2547, whose operational loss is `0.12763774447970921`; its frozen
  constrained callback began at `00:47`. The exact artifact and possible
  terminal decision are estimated around `02:55--03:15 SAST`, before the
  `04:11` wall-time limit. A second miss would terminate normally with
  checkpoint 1981 retained; an improvement of at least `0.001` would reset
  patience and expose the already documented wall-time provenance risk.
  Stage B remains blocked and no continuation or additional job was
  submitted.
- Owned state remains exactly one running A100-40GB `gpu:ampere` job,
  `1238228`, with no pending owned job and no A100-80GB or L40S work. HEX
  quota is home `70.3%`, scratch `38.8%`. Kombuys remains read-only and idle:
  foreign GPU 0 RTX 5090 is `10 MiB/0%`, assigned GPU 1 RTX 3080 Ti is
  `1 MiB/0%`, scratch is `60%` used, and only `tailscale-kombuys` tmux exists.
  Trusted terminal counts remain base `16/16`, NER seed-42 `11/11`, NER
  confirmations `2/4`, T2X seed-42 `11/11`, T2X confirmations `4/4`, POS
  seed-42 `2/11`, winners `0/8`, held-out `0`, and Mono not started. Sheet
  E/F/G remain blank, quarantined rows unpublished, and publication blocked.
  Remaining validation grids prevent a fixed full-results ETA.

## POS Stage-B speed gate stopped safely — 10:13 SAST

- User-authorized cancellation stopped b0 `1238876` at `10:09:55 SAST`
  after `06:49:23`; completed step-283/566 validation artifacts and
  checkpoint 566 remain intact, but the interrupted candidate is
  provenance-only and must not enter selection.
- B1 fused gate job `1238979` started immediately on the freed
  `srvrocgpu010` A100-40GB at `10:09:56`. The cache implementation was
  exactly equivalent on all frozen validation-only checks, but delivered
  only `1.413645836445053x` speedup (`48.3900309689343 s` versus
  `34.23066069406923 s`), below the preregistered 3x requirement. It failed
  closed at `00:01:49`, before training. No held-out data was touched.
- There are now zero owned jobs and zero active owned GPUs. Trusted POS
  seed-42 progress remains `3/11`; winners remain `0/8`, held-out `0`, and
  Mono not started. HEX quota remains home `70.3%`, scratch `38.8%`.
  Sheet E/F/G remain blank, quarantined rows remain unpublished, Kombuys
  remains read-only, and Hugging Face publication remains blocked.

## POS cached last-logit v2 also rejected — 10:21 SAST

- Job `1238982` ran the same frozen 12-cell validation-only gate from
  immutable 698-file snapshot
  `pure-gdn-hpo-poscache-lastlogit-20260816-61ce254f`. Predictions, counts,
  metrics, aggregate accuracy, and selected scores matched exactly, but
  runtime was `35.0182068439899 s` full-prefix versus
  `33.77654465707019 s` cached last-logit, only
  `1.0367610778285992x`. It failed closed after `00:01:33` before b1
  training; no held-out or training data was loaded.
- There are zero owned jobs and no active owned GPU family. The runtime
  optimization is not ratified and trusted POS remains `3/11`; Sheet E/F/G,
  Kombuys, quarantined artifacts, and publication state are unchanged.

## Full-prefix Stage-B recovery preregistration — 10:27 SAST

After both prospective cache gates failed, the user explicitly authorized
continuing the original POS Stage-B work. No held-out result informed this
decision. The scientific recipe, validation data, 12-cell aggregation,
checkpoint cadence, patience/threshold, seed 42, and candidate registry stay
frozen; `SALLM_POS_INCREMENTAL_CACHE` must be absent or `0`.

- Interrupted b0 `1238876` will restart from initialization, not resume from
  checkpoint 566, and will write only to new path
  `/scratch/lmbanr001/masters/sallm/checkpoints/adapter_hpo_v3/pure_gdn/pos/stage_b/b0_recovery_fullprefix_20260816/seed_42`.
  The cancelled b0 directory remains untouched and provenance-only.
- B1 and b2 may use their original absent canonical output paths. Up to all
  three fixed jobs may be queued concurrently, but only A100-40GB
  `gpu:ampere`, one GPU per job, `nlpgroup/a100/nlpgroup`, 24 hours, and
  eight CPUs are allowed.
- A launcher-only environment override may redirect b0's output and logging
  roots. It may not change any model, data, optimizer, evaluator, metric, or
  selection setting. A dry run must prove b0 isolation and unchanged b1/b2
  paths before submission.
- During the rejected v2 deployment, hard-linking caused regeneration of the
  shared `deployment_manifest.json` inode to replace the historical
  `possourcefix` deployment manifest (original recorded SHA-256
  `d996a19bb3441471f4a13e70c14acc2615a16d502cea3f0ee2a5f04cc6308c0a`).
  The source itself is intact: cancelled b0 execution manifest
  `06b825ae37f4eea1d97dd799193f9e445f76bed5cfede53faeb2e18c1f5c22b1`
  still verifies all `694/694` frozen source/config hashes. None of the three
  shared-manifest snapshots may be used for recovery. A new snapshot must
  unlink its manifest and launcher before modification, generate independent
  inodes, verify its complete manifest, and record hashes before submission.

## Full-prefix recovery jobs active — 10:28 SAST

- Launcher-only output/run/log overrides passed shell syntax, whitespace, and
  three dry runs. B0 resolves to its isolated recovery path with the frozen
  b0 recipe; b1/b2 resolve to their unchanged canonical paths and registry
  values. Launcher SHA-256 is
  `38aba8e35f74efc3827769aeb261759b6751d73eb8d4575e75b7c2a0f99e52a1`.
- Clean immutable snapshot
  `/home/lmbanr001/masters/sallm_snapshots/pure-gdn-hpo-pos-fullprefix-recovery-20260816-38aba8e3`
  verifies `694/694` source/config files, contains no cache references, and
  has deployment-manifest SHA-256
  `819a4663b9a90ba65170c73c9dd10b30a16ec14d743c160c55b9c6b89a6df1dc`.
  Its deployment-manifest inode is independent (`10133099172914987`, link
  count 1). Original full-prefix constrained scorer and POS evaluator hashes
  are `571a09a857ef57913f56ed10f73e659d366d78526c47fd310f9810f61580449c`
  and `a93327c31b37477ed01b8ec919ffb51d627658f0dd49f9d8c85318ab5648fb15`.
- Fixed jobs b0 recovery/b1/b2 were submitted as
  `1238989/1238990/1238991`. Every job has account/partition/QOS
  `nlpgroup/a100/nlpgroup`, one `gpu:ampere`, 24 hours, one node, eight CPUs,
  and `/home/lmbanr001/masters/sallm` working directory. B0 and b1 started in
  parallel on two A100-40GB devices on `srvrocgpu010` at
  `10:26:08/10:26:16 SAST`; b2 is Resources-pending with a dynamic scheduler
  projection of `2026-08-17 10:26:16 SAST`.
- B0 execution-manifest SHA-256 is
  `916850c0d5a95cc31420811fd9511237eec41f049e92cc0b4d6f05e09eae68f7`;
  b1 is `5096054826ec1ebdb12e0a0c2592d0fa3655789ff3de33f3fa9d26dd5c6d1d52`.
  Both verified all 694 immutable hashes, passed the fast-GDN gate, loaded
  exactly 2,259 train and 1,800 validation rows, and began fine-tuning without
  fault markers. Trainable parameters are exactly `2,325,312` for b0 and
  `9,296,640` for b1. B0 writes only to the new recovery path; the cancelled
  b0 artifacts remain untouched.
- Operationally two of the maximum three owned A100-40GB slots are running
  and the third job is queued; no A100-80GB or L40S work exists. HEX quota is
  home `70.3%`, scratch `38.8%`. Trusted POS remains `3/11` until terminal
  artifacts verify. Sheet E/F/G remain blank, Kombuys remains read-only,
  quarantined results stay unpublished, and held-out/Hugging Face work stays
  blocked.
