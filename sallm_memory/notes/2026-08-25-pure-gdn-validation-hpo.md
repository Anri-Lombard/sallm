# Pure-GDN validation HPO — 2026-08-25

## Seed-13 confirmation health — 23:33 SAST

- B7 seed-13 `1267877` completed its step-9246 exact callback at `22:40:11`
  and resumed healthy training at `10254/15410`. The artifact SHA-256 is
  `4b1319c0fa35ff512d070e6bd0a0ff803f554c1fa3c83bbb0cba45fb23ed2bf9`.
  It has exactly 128 rows, `64/64` Xho/Zul coverage, `64/64` unique
  predictions per language, no empty predictions, no malformed prompt
  boundary, and no generated EOS marker. Validation chrF is
  `24.076995914285224` Xho and `25.855008400755747` Zul; this remains interim
  validation-only evidence and does not select or freeze a winner.
- A2 seed-13 `1267878` remains healthy at `12003/15410`. Both jobs are the
  only owned GPUs and remain on A100-40GB. Gate jobs `1269243 -> 1269245 ->
  1269246` remain dependency-held as intended; all four A100-80GB devices are
  idle. Quota is home `88.6%`, scratch `43.0%`; held-out access remains zero
  and Sheet E/F/G remain blank.

## Full downstream scope and ETA audit — 22:05 SAST

- Completion is not AfriHG or Multilingual HPO alone. After all eight family
  winners freeze, the frozen protocol still requires 21 Monolingual adapters:
  News 2, NER 3, POS 3, SIB 6, Intent 4, T2X 1, and AfriHG 2. Each uses the
  already-frozen family LR and selects only its own checkpoint on validation.
- The final one-time official test phase covers all 21 applicable Mono arms,
  the six applicable Multi families across their 20 language rows, T2X with no
  duplicate Multi arm, and the General adapter across the full 16-lane matrix.
  No Mono/Multi arm is forced for Belebele or SA-general.
- With four continuously available A100-80GB GPUs, the current best case is:
  global eight-family freeze around 30 August, Monolingual freeze around
  31 August, and the complete verified Mono/Multi/General held-out table late
  31 August or 1 September. The working ETA remains 1 September; 2 September
  is the failure/queue buffer. The earlier late-31-August statement is a best
  case, not a promise.

## Four-GPU ETA refresh — 22:00 SAST

- B7 seed-13 `1267877` is at its exact step-9246 callback; a2 seed-13
  `1267878` has resumed at `10223/15410` after a clean step-9246 artifact.
  Both remain healthy after 12 hours on A100-40GB.
- Terminal confirmation remains expected around `05:00--08:30 SAST` on
  26 August. The automatic `1269243 -> 1269245 -> 1269246` gate should then
  finish within roughly one hour if its short jobs allocate normally.
- A passing gate allows the first four-job A100-80GB wave around
  `09:00--10:00 SAST` on 26 August. Four-way capacity reduces the corrected
  General critical path from about 108 hours to about 72 hours.
- Best-case global Multilingual freeze is now `29--30 August`. Best-case full
  downstream artifacts are late `31 August`; the defensible working ETA is
  `1 September`, with `2 September` retained for queue or run failure.

## Corrected gate chain fully queued — 21:53 SAST

- A100-80GB candidate `1269245` is dependency-held after terminal seed-13 jobs
  `1267877/1267878` and successful A100-40GB reference `1269243`. It requests
  `nlpgroup80/a100/nlpgroup80`, `gpu:ampere80:1`, 24 hours, and eight CPUs.
- CPU comparator `1269246` is dependency-held after candidate success. It
  verifies all four SHA-256 sidecars and runs the unchanged immutable verifier
  at SHA-256 `38d367bfb0580e735fa6ba31e54b5c53c6e7799f2fdb4f57fef63604a6b4b93f`.
- The full gate is now automatic: active confirmations, reference, candidate,
  then fail-closed comparison. No A100-family overlap can occur, and no hourly
  handoff is required before the comparison artifact.

## Four A100-80GB jobs are already permitted — 21:49 SAST

- Live Slurm accounting reports `MaxJobs=4` for the user's `nlpgroup80`
  association and per-user QOS capacity of four `gpu:ampere80` devices. All
  four devices were idle. A dummy GPU job was unnecessary.
- The user authorized a prospective capacity-only amendment. It permits up to
  four real, preregistered A100-80GB validation jobs only after the corrected
  equivalence gate passes. It changes no recipe, candidate, seed, metric,
  checkpoint, confirmation, or held-out rule.

## A100-80GB move preregistered and gate queued — 21:45 SAST

- The user authorized moving pending and future work to the idle A100-80GB
  pool. The prospective runtime-correction amendment was frozen before any new
  metric at SHA-256 `14b2d2dd99ee974073bfe5f4797e2a401f7830db7c89dbb5d6b4bd89e225cf54`;
  its pinned-runtime wrapper has SHA-256
  `d043dc512da6c98c53a9723a2f9e1dda3aecc1e2aac177a76b774afb7cd7c941`.
- Never-started b7 seed-87 job `1269234` was cancelled with its output root
  still absent, solely to preserve the three-owned-job cap. It will be
  resubmitted unchanged after the corrected gate.
- Corrected A100-40GB reference `1269243` is dependency-pending after both
  untouched seed-13 confirmations `1267877/1267878`. It requests
  `nlpgroup/a100/nlpgroup`, `gpu:ampere:1`, 24 hours, and eight CPUs. Its fresh
  result directory was empty at submission.
- After `1269243` exits successfully, one sequential A100-80GB candidate will
  use the same immutable source, model, POS adapter, frozen runtime, and
  unchanged comparator. There is no A100-family overlap before a pass. If the
  gate passes, b7 seed-87, a2 seed-87, and corrected General a0 become the
  first A100-80GB work, within the unchanged three-job cap.
- All four A100-80GB devices were idle at the decision point. Quota remained
  home `88.6%`, scratch `43.0%`; held-out access remained zero, Sheet E/F/G
  blank, Kombuys read-only, and L40S unused.

## A100-40GB ownership and queue count — 21:31 SAST

- Owned work currently uses two of four `gpu:ampere` A100-40GB devices:
  `1267877` and `1267878`. A third owned job, `1269234`, is pending.
- The other two devices run `a100free` jobs `1263877_2` for `chkkar002` and
  `1267954_2` for `bxxjin001`. The latter user has one task running and 45
  collapsed array tasks queued: seven source tasks, 24 dependency-held target
  tasks, and 14 dependency-held evaluation tasks.
- This is a large queued array, but it is not using 46 GPUs. Only one of that
  user's tasks is running. The user's pending source priority is `7795` and
  dependency-held priorities are `7795/7784`, below owned pending job
  `1269234` at `7847`. On current evidence the queue is crowded, but their
  pending array should not start ahead of `1269234` when a device frees.

## Access clarification and A100-40GB contention — 21:28 SAST

- Jan confirmed `gpu:amperemk` is reserved for Michelle's group, while
  `gpu:ampere80` is accessible to this workstream. He also identified another
  student's use of the shared `ampere` capacity as a possible priority issue
  and offered to ask them to back off.
- Live Slurm evidence shows `1269234` pending with `Reason=Priority` and
  unknown start time. The other two devices on `srvrocgpu010` are occupied by
  `a100free` jobs `1263877_2` and `1267954_2`; owned confirmations
  `1267877/1267878` occupy the remaining two. Asking Jan to resolve the
  priority contention is the fastest capacity increase that changes no
  scientific protocol.

## Capacity alternatives checked — 21:26 SAST

- Live read-only checks found all four `gpu:ampere80` devices on
  `srvrocgpu011` idle. The earlier prospective gate remains failed closed:
  scientific outputs matched exactly, but the frozen environment check failed
  solely on `pip` `23.3.1` versus `26.2.1`. Using A100-80GB for HPO would
  require an explicit prospective protocol amendment, an identical pinned
  runtime, and a fresh untouched validation-only equivalence gate. No such
  amendment or job has been authorized.
- Kombuys is idle with RTX 5090 32GB and RTX 3080 Ti 12GB. It remains
  read-only and the RTX 5090 remains untouched. The 3080 Ti is not suitable
  for unchanged full General/AfriHG training; either consumer GPU would need
  a new equivalence gate before contributing selection evidence. Kombuys may
  still absorb non-selection checks, hashes, and bounded diagnostics.
- Live HEX state remains AfriHG confirmations `1267877/1267878` running on
  A100-40GB and seed-87 b7 `1269234` Priority-pending. Quota is home `88.6%`
  and scratch `43.0%`; held-out access remains zero and Sheet E/F/G blank.

## Validity-preserving deadline acceleration — 21:09 SAST

- The monitor cadence is tightened from hourly to every 15 minutes through 31 August so a completed job does not create an avoidable hand-off gap.
- The critical path now overlaps independent families without changing science: after either seed-13 confirmation exits, submit corrected General Stage-A a0 seed-42 at the first free owned-job slot while keeping queued b7 seed-87 `1269234`; use the next free slot for a2 seed-87. Continue at no more than three A100-40GB jobs with absent-output/no-duplicate preflights and immutable provenance.
- No candidate, checkpoint, prompt, seed, metric, coverage assertion, held-out gate, GPU-family rule, or three-job cap changes. The prepared `gpu:amperemk` support request remains unsent pending explicit user approval.

## Third AfriHG confirmation queued; full-table deadline audited — 21:06 SAST

- Frozen b7 seed-87 confirmation was submitted once as job `1269234` after its canonical output root and prior job history were confirmed absent. It requests the unchanged immutable snapshot, `nlpgroup/a100/nlpgroup`, one `gpu:ampere`, 24 hours, eight CPUs, and the canonical home working directory. It is `Priority`-pending because all four A100-40GB devices are allocated; jobs `1267877/1267878` remain the two running owned jobs. The three-job cap is now full, with no A100-80GB or L40S use.
- The end-to-end critical path is not compatible with a defensible 31 August full downstream table under current rules. After the four AfriHG confirmations, corrected General still requires 11 seed-42 trials plus four confirmations, previously budgeted near 18 GPU-hours each. Even with three continuously available A100-40GB devices and no failures, its dependency waves require about 108 wall-clock hours before the global eight-family freeze. Monolingual validation training and the one-time Mono/Multi/General held-out pass then remain.
- Current best-case planning target is global Multilingual freeze around `31 August--1 September`, full downstream artifacts around `2--3 September`, and a queue/failure-aware delivery window of `3--5 September`. A scientifically clean pre-month-end commitment would require new eligible capacity and/or a prospective resource-cap amendment; it cannot be achieved by skipping candidates, confirmations, Monolingual validation, or using held-out results for selection.

## A2 enters third exact callback; b7 healthy — 20:57 SAST

- A2 seed-13 `1267878` reached exactly `9246/15410`, completed full declared validation at health-only loss `2.2032243356079655`, and entered its frozen third exact-generation callback at `20:16:38 SAST`. It remained active at `20:56` with normal generation progress and no artifact or fault marker; neither incomplete output nor health-only loss is used for selection.
- B7 seed-13 `1267877` remained healthy at `8693/15410`, about 553 steps from its third boundary, at roughly `3.12--3.16 s/step`. Both remained `RUNNING` after `10:58:47` on separate A100-40GB `gpu:ampere:1` allocations on `srvrocgpu010`; targeted fault scans were empty.
- A2's third exact artifact remains expected around `21:10--21:30 SAST`; b7 should reach step 9246 around `21:24--21:30`, with its artifact callback-dependent around `22:50--23:20`. The conditional terminal window remains `04:45--08:15 SAST` on 26 August.
- These remain the only owned jobs, with no owned A100-80GB or L40S work. Combined optimizer progress is about `58%`; scientifically AfriHG remains seed-42 `11/11`, confirmations `0/4`, global freeze `0/8`. Quota is home `88.6%`, scratch `43.0%`; Kombuys remains read-only, held-out access is `0`, Sheet E/F/G remain blank, and terminal confirmation compute is the only blocker.

## Confirmations healthy before third boundaries — 20:04 SAST

- A2 seed-13 `1267878` is healthy at `9056/15410`; b7 seed-13 `1267877` is healthy at `7680/15410`. Both have run for `10:05:40` on separate A100-40GB `gpu:ampere:1` allocations on `srvrocgpu010`, sustaining about `3.09--3.15 s/step`; targeted fault scans are empty.
- A2 should reach step 9246 around `20:13--20:16 SAST` and produce its third exact artifact around `21:15--21:40`. B7 should reach step 9246 around `21:22--21:30`, with its artifact callback-dependent around `22:50--23:20`. The conditional terminal window remains `04:45--08:15 SAST` on 26 August.
- These remain the only owned jobs, both A100-40GB, with no owned A100-80GB or L40S work. Combined optimizer progress is about `54%`; scientifically no new artifact exists since the clean second boundaries, so AfriHG remains seed-42 `11/11`, confirmations `0/4`, global freeze `0/8`. Quota remains home `88.6%`, scratch `43.0%`; Kombuys remains read-only, held-out access is `0`, Sheet E/F/G remain blank, and terminal confirmation compute is the only blocker.

## Both second exact artifacts improve; training resumes — 19:04 SAST

- B7 seed-13 `1267877` completed its frozen step-6164 exact callback at `18:43:21 SAST` and resumed healthy training near `6527/15410`. Its immutable artifact has exactly 128 rows, `64/64` Xho/Zul coverage, `64/64` unique predictions per language, zero empty predictions, zero bad `[BOS]` starts, zero bad `[EOS]<|assistant|>` boundaries, and zero generated EOS markers.
- B7 Xho/Zul chrF is `23.23858364622246/24.275093281531426`; preregistered mean chrF is `23.756838463876943`, improving step 3082 mean `23.153769509422958`. Checkpoint 6164 therefore becomes its current validation-only within-run retained checkpoint. Artifact/trainer-state/adapter SHA-256 values are `dd46c013926bd93bf227884b32d884d52843317b9ee5a55a59e71b2a36489ff3`/`2aaeb8fc30c7ad2c98ae26b50c3c173f6bb2341f1192cc52010b956becdcdec1`/`a2d3665db9bd5abea1e635a23f1a1581f78781869afb8a5e5c3357d602c2cd89`.
- A2 seed-13 `1267878` remained healthy near `7895/15410`. At the second clean boundary, b7 mean chrF `23.756838463876943` is slightly above a2 `23.683208682030707`; this is interim validation-only evidence and does not freeze a winner. Both runs must reach terminal artifacts and exact adapter roundtrip verification before confirmation evidence advances.
- Both remain `RUNNING` after `9:05:08` on separate A100-40GB `gpu:ampere:1` allocations on `srvrocgpu010`, with empty targeted fault scans. A2 should reach step 9246 around `20:10--20:20 SAST`; b7 around `21:20--21:35`, with exact artifacts roughly one to two hours later. The conditional terminal window remains `04:45--08:15 SAST` on 26 August. Combined optimizer progress is about `47%`; scientifically AfriHG remains seed-42 `11/11`, confirmations `0/4`, global freeze `0/8`. These are the only owned jobs, no A100-80GB or L40S work exists, quota is home `88.6%` and scratch `43.0%`, Kombuys remains read-only, held-out access is `0`, Sheet E/F/G remain blank, and terminal confirmation compute is the only blocker.

## A2 second exact artifact improves; b7 callback active — 18:03 SAST

- A2 seed-13 `1267878` completed its frozen step-6164 exact callback at `17:32:30 SAST` and resumed healthy training near `6735/15410`. Its immutable artifact has exactly 128 rows, `64/64` Xho/Zul coverage, `64/64` unique predictions per language, zero empty predictions, zero bad `[BOS]` starts, zero bad `[EOS]<|assistant|>` boundaries, and zero generated EOS markers.
- A2 Xho/Zul chrF is `22.871254941246914/24.4951624228145`; preregistered mean chrF is `23.683208682030707`, improving step 3082 mean `22.872278601860273`. Checkpoint 6164 therefore becomes the current validation-only within-run retained checkpoint. Artifact/trainer-state/adapter SHA-256 values are `0edaa940cd44b75e7630d79c9ff3bc8ccb01509fe50cbacab3bb3cc3d07cd84e`/`6a9ef5e80b5c8a95e940871e026ea42d958858811d6a0e62910d3b041e8e51b6`/`c010c2812f8d16274693a6f60f30175d32f924236990dd1f27c9eba6c8c6436b`.
- B7 seed-13 `1267877` reached exactly `6164/15410`, completed full declared validation at health-only loss `2.172864157684742`, and entered its frozen second exact callback at `17:16:24 SAST`. It remained active at `18:02` with normal generation progress, no exact artifact yet, and no fault marker; neither incomplete callback output nor health-only loss is used for selection.
- Both remain `RUNNING` after `8:04:34` on separate A100-40GB `gpu:ampere:1` allocations on `srvrocgpu010`. B7's artifact is expected around `18:20--18:40 SAST`; the conditional terminal window remains `04:45--08:15 SAST` on 26 August. Combined optimizer progress is about `42%`; scientifically AfriHG remains seed-42 `11/11`, confirmations `0/4`, global freeze `0/8`. These are the only owned jobs, no A100-80GB or L40S work exists, quota is home `88.6%` and scratch `43.0%`, Kombuys remains read-only, held-out access is `0`, Sheet E/F/G remain blank, and terminal confirmation compute is the only blocker.

## A2 enters second exact callback; b7 approaches boundary — 17:04 SAST

- A2 seed-13 `1267878` reached exactly `6164/15410`, completed full declared validation at health-only loss `2.18823875907487`, and entered its frozen second exact-generation callback at `16:33:56 SAST`. The callback remained active at `17:03`, with normal automatic batch-size and context-truncation messages and no step-6164 artifact yet; no incomplete output or health-only loss is used for selection.
- B7 seed-13 `1267877` remained healthy at `6062/15410`, about 102 optimizer steps from the same boundary, at roughly `3.14--3.18 s/step`. Both jobs remained `RUNNING` after `7:05:20` on separate A100-40GB `gpu:ampere:1` allocations on `srvrocgpu010`; targeted fault scans contained only the expected `logging_nan_inf_filter=True` configuration line and no runtime fault.
- A2's exact artifact remains expected around `17:25--17:45 SAST`; b7 should enter step 6164 around `17:08--17:12`, with its artifact callback-dependent around `18:35--19:00`. The conditional terminal window remains `04:45--08:15 SAST` on 26 August.
- These remain the only owned jobs, with no owned A100-80GB or L40S work. Combined optimizer progress is about `40%`; scientifically AfriHG remains seed-42 `11/11`, confirmations `0/4`, global freeze `0/8`. Quota is home `88.6%`, scratch `43.0%`; Kombuys remains read-only, held-out access is `0`, Sheet E/F/G remain blank, and terminal confirmation compute is the only blocker.

## Seed-13 confirmations healthy between first and second boundaries — 16:02 SAST

- B7 seed-13 `1267877` is healthy at `4892/15410`; a2 seed-13 `1267878` is healthy at `5651/15410`. Both have run for `6:04:05` on separate A100-40GB `gpu:ampere:1` allocations on `srvrocgpu010`, sustaining about `3.14--3.21 s/step`. Targeted fault scans remain empty.
- A2 is projected to reach step 6164 around `16:29--16:33 SAST`, with its second exact artifact around `17:25--17:45`. B7 is projected to reach step 6164 around `17:08--17:15`, with its artifact callback-dependent around `18:35--19:00`. The conditional terminal window remains `04:45--08:15 SAST` on 26 August.
- These remain the only owned jobs, both A100-40GB, with no owned A100-80GB or L40S work. Combined optimizer progress is about `34%`; scientifically no new artifact exists since the clean first boundaries, so AfriHG remains seed-42 `11/11`, confirmations `0/4`, global freeze `0/8`. Quota remains home `88.6%`, scratch `43.0%`; Kombuys remains read-only, held-out access is `0`, Sheet E/F/G remain blank, and terminal confirmation compute is the only blocker.

## Both seed-13 first callbacks are clean; training resumes — 15:01 SAST

- B7 seed-13 `1267877` completed its frozen step-3082 exact callback at `14:25:53 SAST` and resumed healthy training near `3734/15410`. Its immutable artifact has exactly 128 rows, `64/64` Xho/Zul coverage, `64/64` unique predictions per language, zero empty predictions, zero bad `[BOS]` starts, zero bad `[EOS]<|assistant|>` boundaries, and zero generated EOS markers.
- B7 Xho/Zul chrF is `22.957096156670843/23.350442862175072`; preregistered mean chrF is `23.153769509422958`. Checkpoint 3082 becomes the current validation-only within-run retained checkpoint. Artifact/trainer-state/adapter SHA-256 values are `a9db8e2c...975034`/`c7c9ed89...1eaa6e`/`b293626f...113640`.
- A2 seed-13 `1267878` remains healthy near `4493/15410`. At the first clean boundary, b7 mean chrF `23.153769509422958` is above a2 `22.872278601860273`; this is interim validation-only evidence and does not freeze a winner. Both runs must reach terminal artifacts and exact adapter roundtrip verification before confirmation evidence can advance.
- Both are the only owned jobs on A100-40GB, with no owned A100-80GB or L40S work. Operational optimizer progress across the pair is about `27%`; targeted fault scans remain empty. Based on observed callback runtimes, the conditional terminal window is widened to approximately `04:45--08:15 SAST` on 26 August. Scientifically AfriHG remains seed-42 `11/11`, confirmations `0/4`, global freeze `0/8`. Quota is home `88.6%`, scratch `43.0%`; Kombuys remains read-only, held-out access is `0`, Sheet E/F/G remain blank, and terminal confirmation compute remains the only blocker.

## A2 seed-13 first exact artifact is clean; b7 callback active — 13:59 SAST

- A2 seed-13 `1267878` completed its frozen step-3082 exact callback at `13:45:58 SAST` and resumed healthy training near `3324/15410`. Its immutable artifact has exactly 128 rows, `64/64` Xho/Zul coverage, `64/64` unique predictions per language, zero empty predictions, zero bad `[BOS]` starts, zero bad `[EOS]<|assistant|>` boundaries, and zero generated EOS markers.
- A2 Xho/Zul chrF is `22.690663935823533/23.053893267897013`; preregistered mean chrF is `22.872278601860273`. Checkpoint 3082 becomes the current validation-only within-run retained checkpoint. Artifact/trainer-state/adapter SHA-256 values are `c38bfc67...eb6fc9`/`15bf56b4...a261e`/`8e76fd66...c2c0c8`.
- B7 seed-13 `1267877` remains healthy in its step-3082 exact callback. Its log advanced through `13:28:36 SAST` with normal context-truncation warnings and no fault marker; no exact artifact exists yet, so no b7 metric or checkpoint decision is available.
- Both jobs remain the only owned jobs on A100-40GB; no owned A100-80GB or L40S work exists. B7's artifact is expected around `14:00--14:20 SAST`; terminal artifacts remain approximately `05:30--07:00 SAST` on 26 August if uninterrupted. Scientifically a2 is valid within-run evidence but not a terminal confirmation, so AfriHG remains seed-42 `11/11`, confirmations `0/4`, global freeze `0/8`. Quota is home `88.6%`, scratch `42.9%`; Kombuys remains read-only, held-out access is `0`, Sheet E/F/G remain blank, and b7 callback plus terminal confirmation compute remain the blockers.

## Both seed-13 runs enter first exact callback — 12:57 SAST

- A2 seed-13 `1267878` reached exactly `3082/15410`, completed full declared validation at health-only loss `2.232720357113887`, and entered the frozen exact-generation callback at `12:42:37 SAST` with automatic generation batch size 64. B7 seed-13 `1267877` reached exactly `3082/15410`, completed full declared validation at health-only loss `2.2235934426458064`, and entered the same callback at `12:44:00 SAST` with batch size 64.
- Both jobs remain `RUNNING` on separate A100-40GB `gpu:ampere:1` allocations on `srvrocgpu010` after `2:59:38`. Targeted scans are empty for traceback, OOM, CUDA, NCCL, killed-process, runtime-error, and non-finite markers. No step-3082 exact artifact exists yet, so neither health-only loss nor any incomplete callback output is used for selection.
- These are the only owned jobs; no owned A100-80GB or L40S work exists. First exact artifacts remain due approximately `13:40--14:10 SAST`, with terminal artifacts approximately `05:30--07:00 SAST` on 26 August if uninterrupted. Operational optimizer progress is `20%`; scientifically AfriHG remains seed-42 `11/11`, confirmations `0/4`, global freeze `0/8`. Quota is home `88.6%`, scratch `42.9%`; Kombuys remains read-only, held-out access is `0`, Sheet E/F/G remain blank, and the active callback is the only blocker.

## Seed-13 confirmations healthy at 15% — 11:57 SAST

- B7 seed-13 `1267877` is healthy at `2257/15410`; a2 seed-13 `1267878` is healthy at `2272/15410`. Both have run for `1:59:07` on separate A100-40GB `gpu:ampere:1` allocations on `srvrocgpu010`, sustaining about `3.06--3.12 s/step`. Targeted scans remain empty for traceback, OOM, CUDA, NCCL, killed-process, runtime-error, and non-finite markers.
- They remain the only owned jobs, so the two-of-three cap and single-family rule are satisfied with no owned A100-80GB or L40S work. Both first `3082` boundaries remain on schedule around `12:38--12:43 SAST`; exact callback artifacts remain approximately `13:45--14:45 SAST`, and terminal artifacts approximately `05:30--07:00 SAST` on 26 August if uninterrupted.
- Operational seed-13 training is about `15%` complete. Scientifically no new callback artifact exists: AfriHG remains seed-42 `11/11`, confirmations `0/4`, global freeze `0/8`. Quota is home `88.6%`, scratch `42.9%`; Kombuys remains read-only, held-out access is `0`, Sheet E/F/G remain blank, and confirmation compute remains the only blocker.

## Seed-13 confirmations healthy at 7% — 10:56 SAST

- B7 seed-13 `1267877` is healthy at `1070/15410`; a2 seed-13 `1267878` is healthy at `1083/15410`. Both have run for `58:16` on separate `gpu:ampere:1` A100-40GB allocations on `srvrocgpu010`, sustaining about `3.05--3.10 s/step` with no traceback, OOM, CUDA, NCCL, killed-process, runtime-error, or non-finite marker.
- These are the only owned jobs. There is no owned A100-80GB or L40S job, so GPU-family isolation and the two-of-three job cap remain satisfied. The first `3082` validation boundaries are projected around `12:35--12:45 SAST`; full exact validation artifacts remain callback-dependent around `13:45--14:45 SAST`. Terminal artifacts remain approximately `05:30--07:00 SAST` on 26 August if uninterrupted.
- Operationally the live seed-13 training is about `7%` complete. Scientifically no new callback or terminal artifact exists: AfriHG remains seed-42 `11/11`, confirmations `0/4`, and global freeze `0/8`. Quota remains home `88.6%`, scratch `42.9%`; Kombuys remains read-only, held-out access is `0`, Sheet E/F/G remain blank, and confirmation compute is the only blocker.

## A100-80GB gate fails closed; seed-13 confirmations start — 10:00 SAST

- A100-80GB candidate `1258898` completed `0:0` at `08:21:15 SAST` on `srvrocgpu011`. Its preregistered validation-only result and execution-manifest sidecars pass their recorded SHA-256 checks. Predictions, metrics, selected score, dataset identity, model identity, and requested/allocated `gpu:ampere80:1` evidence are intact.
- Original CPU comparator `1258899` failed `2:0` in zero seconds because its launcher selected a stale verifier that did not accept the frozen manifest arguments. It stopped in `argparse`, before loading either result or manifest, and produced no comparison artifact. This is preserved as execution-failure provenance.
- Execution-only comparator retry `1267937` used the already frozen verifier at SHA-256 `64c8e42a...1a89` and the same four sidecar-validated inputs. It exited `1:0` with fail-closed comparison artifact SHA-256 `8142fb8b...cfe73`. All scientific equality checks passed, including exact predictions and zero selected-score difference, but `runtime_environment_match=false`: the only package difference is `pip` `23.3.1` on the A100-40GB reference versus `26.2.1` on the A100-80GB candidate. Under the preregistered any-check-fails rule, the prospective hardware-equivalence gate therefore fails. A100-80GB is excluded; the threshold will not be relaxed and the gate will not be rerun.
- The failed comparator left seed-13 confirmations b7 `1267877` and a2 `1267878` in `DependencyNeverSatisfied`. After the gate was resolved fail-closed and no owned GPU was active, their obsolete dependencies were cleared without changing either frozen candidate. Both began at `09:57:54 SAST` on separate A100-40GB devices on `srvrocgpu010`, with exact `nlpgroup/a100/nlpgroup`, `gpu:ampere:1`, 24 hours, eight CPUs, and immutable 694-file source manifests. Their logs reached fine-tuning startup at `09:59 SAST`; no fault marker is present.
- B7 seed-13 uses manifest SHA-256 `966088ea...12fa`; a2 seed-13 uses `8b829180...a99`. Both declare `selection_split=validation`, `seed=13`, and the frozen registry SHA-256 `8fdd6ea5...bb726`. No held-out artifact was read or written.
- Operationally two A100-40GB GPUs are active; no other owned job, A100-80GB, or L40S job is active. At `10:01 SAST`, both confirmations were healthy at step `27/15410` near `3.15 s/step`, with no fault marker. The first full validation-only callbacks are expected around `13:30--14:30 SAST`, with terminal artifacts around `05:30--07:00 SAST` on 26 August if uninterrupted. Scientifically AfriHG remains seed-42 `11/11`, confirmations `0/4`, and global freeze `0/8`. Quota is `88.6%/42.9%`; Kombuys remains read-only, held-out access is `0`, Sheet E/F/G remain blank, and the current blocker is confirmation compute completion rather than artificial scheduling delay.

## AfriHG seed-42 grid closes; Stage-C top two frozen — 07:59 SAST

- B7 `1262543` completed `0:0` at `07:50:50 SAST` after exactly `15410/15410`. Its immutable terminal artifact has 128 rows (`64/64` Xho/Zul), `64/64` unique predictions per language, zero empty predictions, correct `[BOS]` through `[EOS]<|assistant|>` boundaries, and zero generated EOS markers.
- Terminal Xho/Zul chrF is `23.814926201664736/26.355581419154543`; mean `25.08525381040964` improves step 12328, so terminal checkpoint 15410 is the validation-selected retained checkpoint. Artifact/trainer-state SHA-256 values are `845187a0...c857d77`/`6779f1ef...1adb4e1`.
- CPU verifier `1267874` completed `0:0` and proved exact retained-to-final equality across all 424 tensor keys and `76,410,112` values. Retained/final SHA-256 values are `9dfc09f8...af236bf`/`450e4163...7dbfbc`. AfriHG seed-42 therefore closes scientifically at `11/11`.
- CPU ranking job `1267875` completed `0:0`. The preregistered validation-only rank over all 11 candidates freezes b7 and a2 as the Stage-C candidates at `25.08525381040964/24.910865619073284`. The local and HEX ranking artifact is byte-identical at SHA-256 `b252ddcc...5f70df`; it records the frozen registry hash, all candidate configurations and metrics, `selection_split=validation`, and no held-out evidence.
- Seed-13 confirmations b7 `1267877` and a2 `1267878` are dependency-pending after successful comparator `1258899`. Together with A100-80GB gate `1258898`, this is exactly the three-owned-GPU-job cap. The gate is allocation-pending on `AssocGrpGRES` with Slurm estimate `2026-08-25 21:47:49 SAST`; all four A100-80GB devices are occupied by other-user jobs. The dependency chain prevents A100-family overlap.
- Operationally no GPU is active now. Scientifically AfriHG is seed-42 `11/11`, confirmations `0/4`, and global freeze `0/8`. Quota is `88.6%/42.5%`; Kombuys remains read-only, held-out access is `0`, and Sheet E/F/G remain blank.

## B6 is scientifically terminal-valid; b7 terminal callback active — 06:58 SAST

- B6 `1262542` completed `0:0` after exactly `15410/15410`. Its immutable terminal artifact has 128 rows (`64/64` Xho/Zul), `64/64` unique predictions per language, zero empty predictions, correct `[BOS]` through `[EOS]<|assistant|>` boundaries, and zero generated EOS markers.
- Terminal Xho/Zul chrF is `21.906310485665017/22.74221693861099`; mean `22.324263712138006` improves step 12328, so terminal checkpoint 15410 is the validation-selected retained checkpoint. Artifact/trainer-state SHA-256 values are `76be85b2...adf4f5a`/`1a957eff...0a7ea7`.
- CPU verifier `1267864` completed `0:0` and proved exact retained-to-final equality across all 424 tensor keys and `71,762,560` values. Retained/final file SHA-256 values are `fed9fd82...8e19522`/`bdae26c8...84c40a`. The first verifier attempt `1267863` is preserved as a zero-second `1:0` launcher failure caused solely by a nonexistent venv activation path; corrected script SHA-256 is `4d6629d9...3feb3a8`.
- AfriHG advances to `10/11` scientifically terminal-valid candidates. B7 `1262543` has completed `15410/15410`, full declared terminal validation at health-only loss `2.317962587073126`, and is healthy in its frozen terminal exact-generation callback. Gate `1258898` remains dependency-blocked until b7 completes, followed by comparator `1258899`; the b7 terminal artifact is tentatively due around `07:15--07:45 SAST`.
- Quota is `88.6%/42.5%`; only A100-40GB is active, no A100-80GB or L40S overlap exists, Kombuys remains read-only, held-out access is `0`, Sheet E/F/G remain blank, and global freeze is `0/8`.

## B6 terminal callback active; b7 at 94% — 05:52 SAST

- B6 `1262542` remains active in its frozen terminal exact-generation callback after `15410/15410`. The log advanced through `05:26:51 SAST`, including normal context-truncation warnings and automatic generation batch-size selection; no terminal artifact exists yet and there is no fault marker. This is active generation, not a stall.
- B7 `1262543` remains healthy near `14425/15410` (`93.6%`) with about 52 minutes of displayed training remaining before terminal validation and its frozen terminal callback. Gate `1258898` remains dependency-blocked on successful completion of both runs, followed by comparator `1258899`.
- The callback-dependent terminal window is now roughly `06:30--08:30 SAST`. Quota is `88.6%/42.4%`; only A100-40GB is active, no A100-80GB or L40S overlap exists, Kombuys remains read-only, held-out access is `0`, Sheet E/F/G remain blank, AfriHG is `9/11` terminal-valid, and global freeze is `0/8`.

## B7 step 12328 improves; b6 terminal callback active — 04:52 SAST

- B7 `1262543` completed its frozen step-12328 exact-generation callback at `04:01:47 SAST` and resumed healthy A100-40GB training near `13283/15410`. Its immutable artifact has exactly 128 rows (`64/64` Xho/Zul), `64/64` unique predictions per language, zero empty predictions, correct `[BOS]` through `[EOS]<|assistant|>` boundaries, and zero generated EOS markers.
- Xho/Zul chrF is `23.826584265027677/26.270991275007738`; registered mean chrF is `25.048787770017707`, improving step 9246 mean `25.00603066274296`. Checkpoint 12328 therefore becomes retained using validation evidence only. Artifact/trainer-state SHA-256 values are `f5c261d0...689a3e`/`98153bf3...00ebb2`.
- B6 `1262542` reached exactly `15410/15410`, completed full declared terminal validation at health-only loss `2.276732130997526`, and entered its frozen terminal exact-generation callback. No terminal artifact exists yet, so b6 remains scientifically non-terminal and AfriHG remains `9/11` terminal-valid.
- B7 is about `86.2%` complete with roughly 1h51m displayed training remaining before terminal validation/callback. Targeted fault scans are empty. Gate `1258898` remains dependency-blocked on successful completion of both runs, followed by comparator `1258899`. The callback-dependent terminal window remains roughly `06:00--08:00 SAST`.
- Quota is `88.6%/42.4%`; only A100-40GB is active, no A100-80GB or L40S overlap exists, Kombuys remains read-only, held-out access is `0`, Sheet E/F/G remain blank, and global freeze is `0/8`.

## B7 enters fourth exact callback; b6 at 93% — 03:52 SAST

- B7 `1262543` reached exactly `12328/15410`, completed full declared validation at health-only loss `2.2675734513150956`, and entered its frozen step-12328 exact-generation callback. No step-12328 exact artifact exists yet, so no new b7 metric or checkpoint decision is available.
- Concurrent b6 `1262542` remains healthy near `14399/15410` (`93.4%`) with about 53 minutes of displayed training remaining before terminal validation and its frozen terminal callback. Targeted fault scans are empty for both jobs. Gate `1258898` remains dependency-blocked on both, followed by comparator `1258899`.
- The scheduler- and callback-dependent terminal window is now roughly `06:00--08:00 SAST`. Quota is `88.6%/42.4%`; only A100-40GB is active, no A100-80GB or L40S overlap exists, Kombuys remains read-only, held-out access is `0`, Sheet E/F/G remain blank, AfriHG is `9/11` terminal-valid, and global freeze is `0/8`.

## B6 step 12328 is clean and becomes retained — 02:51 SAST

- B6 `1262542` completed its frozen step-12328 exact-generation callback at `02:03:58 SAST` and resumed healthy A100-40GB training near `13219/15410`. Its immutable artifact has exactly 128 rows (`64/64` Xho/Zul), `64/64` unique predictions per language, zero empty predictions, correct `[BOS]` through `[EOS]<|assistant|>` boundaries, and zero generated EOS markers.
- Xho/Zul chrF is `21.830004875620137/22.782335134481`; registered mean chrF is `22.306170005050568`, improving step 9246 mean `22.078007172731617`. Checkpoint 12328 therefore becomes retained using validation evidence only. Artifact/trainer-state SHA-256 values are `e915cd82...242c50`/`4c4660ea...56cbf`.
- Concurrent b7 `1262543` remains healthy near `12240/15410`, approaching its frozen step-12328 boundary. Targeted fault scans are empty for both jobs. Gate `1258898` remains dependency-blocked on both, followed by comparator `1258899`.
- Quota is `88.6%/42.4%`; only A100-40GB is active, no A100-80GB or L40S overlap exists, Kombuys remains read-only, held-out access is `0`, Sheet E/F/G remain blank, AfriHG is `9/11` terminal-valid, and global freeze is `0/8`.

## B6 enters fourth exact callback; b7 healthy — 01:50 SAST

- B6 `1262542` reached exactly `12328/15410`, completed full declared validation at health-only loss `2.277246473981992`, and entered its frozen step-12328 exact-generation callback. No step-12328 exact artifact exists yet, so no new b6 metric or checkpoint decision is available.
- Concurrent b7 `1262543` remains healthy near `11078/15410` at about `3.14 s/step`. Targeted fault scans are empty for both jobs. Gate `1258898` remains dependency-blocked on both, followed by comparator `1258899`.
- Quota is `88.6%/42.4%`; only A100-40GB is active, no A100-80GB or L40S overlap exists, Kombuys remains read-only, held-out access is `0`, Sheet E/F/G remain blank, AfriHG is `9/11` terminal-valid, and global freeze is `0/8`.

## B7 step 9246 is clean and becomes retained — 00:49 SAST

- B7 `1262543` completed its frozen step-9246 exact-generation callback at `00:13:33 SAST` and resumed healthy A100-40GB training near `9918/15410`. Its immutable artifact has exactly 128 rows (`64/64` Xho/Zul), `64/64` unique predictions per language, zero empty predictions, correct `[BOS]` through `[EOS]<|assistant|>` boundaries, and zero generated EOS markers.
- Xho/Zul chrF is `24.154545975888247/25.85751534959767`; registered mean chrF is `25.00603066274296`, improving step 6164 mean `24.25932370479508`. Checkpoint 9246 therefore becomes retained using validation evidence only. Artifact/trainer-state SHA-256 values are `bb3a7f7f...1b342`/`78c213ca...bdd11`.
- Concurrent b6 `1262542` remains healthy near `12296/15410`, immediately before its frozen step-12328 boundary. Targeted fault scans are empty for both jobs. Gate `1258898` remains dependency-blocked on both, followed by comparator `1258899`.
- Quota is `88.6%/42.4%`; only A100-40GB is active, no A100-80GB or L40S overlap exists, Kombuys remains read-only, held-out access is `0`, Sheet E/F/G remain blank, AfriHG is `9/11` terminal-valid, and global freeze is `0/8`.
