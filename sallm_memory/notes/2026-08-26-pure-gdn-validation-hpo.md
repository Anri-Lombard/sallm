# Pure-GDN validation HPO — 2026-08-26

## Corrected General first boundaries verify fully — 22:57 SAST

- General Stage-A a0/a1 `1271042/1271043` completed their first frozen
  step-2728 evaluations and resumed healthy training. Their sidecar hashes
  exactly match artifact SHA-256 values
  `0e41f378af2e129b6b9e27224a70be61a078dbc9eb8dd24a84f86c64f684a5ce`
  and
  `ab5a8b492640bf46e778f687e71a125f68d153f99f82e44d64df2dc97288fd0d`.
- Both artifacts persist the preregistered
  `equal_family_assistant_token_nll_v1` inputs and aggregation: exact 22,167
  processed rows across SIB `2970`, News `3095`, NER `10760`, POS `1800`,
  AfriHG `3082`, and T2X `460`; AfriHG language counts `1305/1777` Xho/Zul;
  summed assistant-token NLL; valid-token counts; per-family NLL; and the
  equal-family macro NLL. No family or row is missing.
- A0 macro NLL is `1.1208475241267701`; a1 macro NLL is
  `0.988150101648468`. These are interim validation-only checkpoint metrics,
  not candidate winners. AfriHG confirmations also remain healthy after their
  clean first artifacts. All four A100-80GB jobs remain active; held-out
  access is zero and Sheet E/F/G remain blank.

## B7 seed-87 first exact artifact clean — 21:03 SAST

- Isolated b7 seed-87 `1271354` completed its frozen step-3082 callback and
  resumed healthy training. The immutable artifact SHA-256 is
  `c6b829e1a523986ddaa414bee6fbfd064df684c4da8c577d3f0d1e63baadf46a`.
- It contains exactly 128 rows with `64/64` Xho/Zul coverage and unique
  predictions, zero empty predictions, zero bad `[BOS]` starts, zero bad
  `[EOS]<|assistant|>` boundaries, and zero generated EOS markers. Validation
  Xho/Zul chrF is `22.79339990657292/23.77130044788626` (mean
  `23.28235017722959`). This remains interim validation-only evidence and does
  not freeze or select a winner.
- A2 seed-87 remains healthy beyond its first clean callback. General a0/a1
  have reached their first frozen full-coverage validation work without a
  fault marker. All four A100-80GB cards remain occupied; held-out access is
  zero and Sheet E/F/G remain blank.

## A2 seed-87 first exact artifact clean — 20:50 SAST

- A2 seed-87 `1271041` completed its frozen step-3082 callback and resumed
  healthy training. The immutable artifact SHA-256 is
  `5fb9227ab88afe15a52705bbddceb5a8e9db509e6fc8afe808d4f5203dc18d0d`.
- It contains exactly 128 rows with `64/64` Xho/Zul coverage and unique
  predictions, zero empty predictions, zero bad `[BOS]` starts, zero bad
  `[EOS]<|assistant|>` boundaries, and zero generated EOS markers. Validation
  Xho/Zul chrF is `23.232538742524287/23.51599415139435` (mean
  `23.37426644695932`). This is interim validation-only evidence and does not
  freeze or select a winner.
- B7 seed-87 `1271354` remains inside the same first frozen callback. General
  a0/a1 `1271042/1271043` remain healthy. All four A100-80GB cards stay
  occupied; held-out access remains zero and Sheet E/F/G remain blank.

## Four-card A100-80GB wave remains healthy — 16:51 SAST

- All four ratified A100-80GB jobs remain `RUNNING` on `srvrocgpu011`: a2
  seed-87 `1271041`, General a0/a1 `1271042/1271043`, and isolated b7 seed-87
  `1271354`. Live optimizer positions are respectively `144/15410`,
  `61/13640`, `62/13640`, and `88/15410`, with no fault marker.
- A five-sample compute-side utilization check confirms that the four-card
  concurrency ceiling is fully occupied, while each frozen small-model job
  uses only about `11.6--17.6 GiB` and has bursty SM utilization. This is the
  same known launch/validation-bound behavior documented prospectively on
  A100-40GB; it is not a slow fallback. Changing microbatch inside the active
  AfriHG or General grids would require matched reruns, so the live jobs remain
  unchanged.
- Quota is home `88.6%`, scratch `43.8%`; held-out access remains zero and
  Sheet E/F/G remain blank. The working complete-table ETA remains 1 September,
  with late 31 August best case and 2 September buffer.

## A100-80GB gate passes; four-job wave active — 16:45 SAST

- Reference `1271037` and A100-80GB candidate `1271038` completed `0:0` in
  `60/62` seconds without overlap. Stale-verifier comparator `1271039` failed
  in `argparse` before payload loading; execution-only corrected comparator
  `1271352` used frozen verifier SHA-256 `64c8e42a...1a89` and completed `0:0`.
- The sidecar-verified comparison artifact is
  `9f699ff2fe0377d6d149c25863500b3c252565b0d477fa2546253f25bd1fa377`.
  Every frozen check passes, including exact predictions, runtime environment,
  hardware labels, and zero maximum/mean score difference. A100-80GB is
  scientifically ratified.
- A2 seed-87 `1271041` and General a0/a1 `1271042/1271043` are healthy on
  `srvrocgpu011`. Initial b7 `1271040` was cancelled after 39 seconds when its
  old wrapper ignored the isolated-output override; the canonical partial path
  is quarantined. Corrected b7 `1271354` is healthy in the intended isolated
  path with manifest SHA-256 `2b2738fa...e8b40`.
- All four A100-80GB devices are now productively occupied. AfriHG remains
  confirmations `2/4` and global freeze `0/8` until terminal artifacts and
  roundtrips complete. Quota is home `88.6%`, scratch `43.5%`; held-out access
  remains zero and Sheet E/F/G blank. Working complete-table ETA is 1 September,
  with late 31 August best case and 2 September buffer.

## User-prioritised A100-80GB launcher correction queued — 15:45 SAST

- A prospective correction now permits one replacement gate because failed
  reference `1269243` never reached model, data, evaluator, or result. The only
  wrapper change invokes the unchanged read-only nested launcher through
  `bash`; amendment/wrapper hashes are `73988566...3f4a` and
  `36022115...3c81`.
- To prioritise the gate, running b7 seed-87 `1270629` was cancelled and
  preserved after `02:48:42`; pending `1270630/1270631` never started and were
  cancelled. No validation metric informed this scheduling change.
- The final strict chain is A100-40GB reference `1271037` -> A100-80GB
  candidate `1271038` -> CPU comparator `1271039`. Requested and exported GRES
  values are exact. Slurm estimates reference start at `17:41:17 SAST`.
- After comparator success, b7 seed-87 `1271040`, a2 seed-87 `1271041`, and
  corrected General a0/a1 `1271042/1271043` automatically become eligible on
  all four A100-80GB devices. Until then no A100-80GB science can run. Held-out
  access remains zero and Sheet E/F/G remain blank. Conditional on a gate pass,
  the complete-table ETA returns to 1 September, with late 31 August best case
  and 2 September queue/failure buffer.

## B7 seed-87 starts; remaining work queue-bound — 13:49 SAST

- AfriHG b7 seed-87 confirmation `1270629` started on A100-40GB at `12:52:33`
  and is healthy at `1113/15410` (about `7%`, `3.02` seconds per optimizer
  step) with no fault marker. Its immutable execution manifest re-verifies at
  SHA-256 `8888487e2d650175a44ca2e2eb52d0a31aab323f2af942ffa46ba93f8ccbf584`.
- A2 seed-87 `1270630` remains `Resources`-pending with Slurm reservation
  `2026-08-27 12:52:33`; corrected General a0 `1270631` remains
  `Priority`-pending without a start estimate. Three other users' A100-40GB
  jobs occupy the remaining devices, so b7 is the only active owned job.
- AfriHG confirmations remain `2/4` scientifically terminal-valid and the
  global freeze remains `0/8`. Held-out access is zero and Sheet E/F/G remain
  blank. Quota is home `88.6%`, scratch `43.3%`; the A100-40GB queue remains
  the blocker. Complete-table ETA remains 2--3 September best case and 3--5
  September with queue/failure buffer.

## Corrected A100 gate fails before science; A100-40GB work resumes — 12:47 SAST

- Corrected A100-40GB reference `1269243` finally allocated, then failed
  `126:0` after one second. Its pinned wrapper completed `uv sync --frozen
  --inexact` (`117` packages audited), then could not execute immutable nested
  launcher `run_pos_runtime_equivalence_gate.sh`: the file is mode `0444`, so
  Slurm reported `Permission denied`. No gate result artifact exists. Launcher
  and failure-log SHA-256 values are
  `aac78d35a22480b46a59c06f5a7c88050bb1e831f8f02d596a6e889b0469b956`
  and `0e615685862cce3161c57099fd8633474dde079028efed5767309dd75589f20d`.
- This is a pre-science launcher failure, but the frozen corrected gate is not
  rerun or relaxed. A100-80GB is excluded. Never-started dependency jobs
  `1269245/1269246` were cancelled after exact state checks; all failure
  provenance remains preserved.
- After absent-output and no-active-duplicate checks, three unchanged required
  A100-40GB jobs were submitted under the exact `nlpgroup/a100/nlpgroup`,
  `gpu:ampere:1`, 24-hour, eight-CPU protocol: b7 seed-87 `1270629`, a2 seed-87
  `1270630`, and corrected General Stage-A a0 seed-42 `1270631`. All are queue
  pending; no owned GPU is active yet.
- AfriHG remains confirmations `2/4`; global freeze remains `0/8`. Held-out
  access remains zero and Sheet E/F/G remain blank. With A100-80GB excluded,
  the defensible best-case complete-table ETA moves to 2--3 September and the
  queue/failure-aware ETA to 3--5 September. Quota is home `88.6%`, scratch
  `43.1%`; the A100-40GB queue is the blocker.

## A100-40GB gate queue slips again — 09:42 SAST

- Gate reference `1269243` remains `Priority`-pending and Slurm's estimate has
  slipped from `13:19` to `19:22 SAST`. No owned GPU is active. Two
  `bxxjin001` array tasks and two `chkkar002` jobs occupy all four A100-40GB
  devices; two `alsilo001` jobs are also queued ahead of the reference.
- Candidate `1269245` and comparator `1269246` remain dependency-safe; all four
  A100-80GB devices remain idle. Releasing only a `bxxjin001` slot may not be
  sufficient because the higher-priority `alsilo001` jobs are now ahead. The
  useful intervention is priority or immediate allocation for the short gate
  reference itself.
- The A100-40GB queue remains the only blocker. Working complete-table ETA is
  still 1 September, with late 31 August best case and 2 September
  failure/queue buffer. Quota is home `88.6%`, scratch `43.1%`; held-out access
  remains zero and Sheet E/F/G remain blank.

## Both seed-13 confirmations close; gate queue slips — 07:39 SAST

- B7 exact roundtrip verifier `1269961` completed `0:0` and proved exact
  retained-to-final equality across 424 keys and 76,410,112 values. Retained
  and final adapter SHA-256 values are
  `375c0c71aa53eb4253d99ef4a7eec17f2fd60713df95d9aa2ff28a786465ff61`
  and `0825e0dc7db65209614474c7d792fd7f64fa712b92efdc009d8ba59a85f25a73`;
  verifier output SHA-256 is
  `6a96a7194744e5b263184350d1d6103ed939fc3e0ee700f416b245a7ecce60ff`.
  B7 seed-13 is scientifically terminal-valid, so AfriHG confirmations advance
  to `2/4`.
- A100-40GB reference `1269243` remains allocation-pending and changed from
  `Resources` to `Priority`; Slurm's current estimate slipped from `11:19:14`
  to `13:19:00 SAST`. Three A100-40GB devices remain occupied by `bxxjin001`
  and one by `chkkar002`; all four A100-80GB devices remain idle. Candidate
  `1269245` and comparator `1269246` remain dependency-safe. No owned GPU is
  active, and the A100-40GB gate queue is the only blocker.
- Working complete-table ETA remains 1 September, with late 31 August best case
  and 2 September failure/queue buffer. Quota is home `88.6%`, scratch `43.1%`;
  held-out access remains zero and Sheet E/F/G remain blank.

## B7 terminal artifact clean; gate reference queued — 06:39 SAST

- B7 seed-13 `1267877` completed `0:0` after exactly `15410/15410`. Its
  immutable terminal artifact has 128 rows, exact `64/64` Xho/Zul coverage and
  unique predictions, no empty prediction, no malformed prompt boundary, and
  no generated EOS marker. Terminal Xho/Zul chrF is
  `24.27694839782136/26.475773492688447`, mean `25.376360945254902`;
  checkpoint 15410 is the validation-selected checkpoint. Artifact and
  trainer-state SHA-256 values are
  `e2cf0bac32f1466413ce007733649478d0e6ac10f7cdb27509a30bab0382275c`
  and `567b0fe194beb7882d5ed24e43d93c30b8a182f1764aa2c22cba0c0d48dd155c`.
- Exact roundtrip verifier `1269961` was submitted after an absent-output and
  no-duplicate preflight using the immutable confirmation verifier. It is
  `Priority`-pending on CPU, so b7 is not yet counted scientifically and
  AfriHG confirmations remain `1/4`.
- A100-40GB gate reference `1269243` is now allocation-pending on `Resources`,
  with Slurm's current estimate `11:19:14 SAST`. Three A100-40GB devices are
  occupied by `bxxjin001` array tasks and one by `chkkar002`; no owned GPU is
  active. Candidate `1269245` and comparator `1269246` remain dependency-safe.
  All four A100-80GB devices are idle. This A100-40GB queue is now the critical
  blocker; the working complete-table ETA remains 1 September, with late
  31 August best case and 2 September failure/queue buffer.
- Quota is home `88.6%`, scratch `43.1%`; held-out access remains zero and
  Sheet E/F/G remain blank.

## B7 terminal callback — 05:39 SAST

- B7 seed-13 `1267877` reached exactly `15410/15410` and entered its frozen
  terminal exact-generation callback after complete declared validation. Its
  health-only loss is `2.312071922148767`; neither that loss nor the incomplete
  callback is used for selection. No terminal artifact or roundtrip result
  exists yet.
- B7 is the only active owned GPU job on A100-40GB and has no fault marker.
  Gate jobs `1269243 -> 1269245 -> 1269246` remain dependency-held as intended;
  all four A100-80GB devices remain idle. Quota is home `88.6%`, scratch
  `43.0%`; held-out access remains zero and Sheet E/F/G remain blank.

## A2 seed-13 confirmation closes scientifically — 04:39 SAST

- A2 seed-13 `1267878` completed `0:0` after exactly `15410/15410`. Its
  immutable terminal artifact has 128 rows, exact `64/64` Xho/Zul coverage and
  unique predictions, no empty prediction, no malformed prompt boundary, and
  no generated EOS marker. Terminal Xho/Zul chrF is
  `23.94949905652421/25.8407130264596`, mean `24.895106041491907`; checkpoint
  15410 is the validation-selected checkpoint. Artifact and trainer-state
  SHA-256 values are `4c4302ced16f2150958a7d76c19659a39977df4b4988b7b16e46c0f5007063c6`
  and `82db5037f80ce8a98ae11a2bff6dea4b4c4e3c5cdd54ce467285543d54c638d0`.
- CPU verifier `1269959` completed `0:0` and proved exact retained-to-final
  equality across 424 keys and 71,762,560 values. Retained/final adapter hashes
  are `3d26daf2d40cec952c468e4812cfef776b3a53304078f1ac758e75ce4fd0eae2`
  and `2bc3295f7fe234a27f6c2f45ed744700c7edbe7154cf45b8022119bfe3c7a8d4`.
  The narrow confirmation verifier is immutable at SHA-256
  `a14c8e27162ce6c2e50343750006d8118499c7bd60cbc4c6bddf985ec5821003`;
  it changes no metric or selection rule. AfriHG confirmations advance to
  `1/4` scientifically terminal-valid.
- B7 seed-13 `1267877` remains healthy at `15035/15410`, about 20 minutes from
  its terminal boundary. It is the only active owned GPU job. Gate jobs
  `1269243 -> 1269245 -> 1269246` remain dependency-held as intended; all four
  A100-80GB devices are idle. Quota is home `88.6%`, scratch `43.0%`; held-out
  access remains zero and Sheet E/F/G remain blank.

## A2 terminal callback — 03:37 SAST

- A2 seed-13 `1267878` reached exactly `15410/15410` and entered its frozen
  terminal exact-generation callback after complete declared validation. Its
  health-only loss is `2.2644056435299724`; neither that loss nor the
  incomplete callback is used for selection. No terminal artifact or adapter
  roundtrip result exists yet.
- B7 seed-13 `1267877` remains healthy at `13869/15410`, about 1541 optimizer
  steps from its terminal boundary. Both remain the only owned GPU jobs on
  A100-40GB without a fault marker. The gate chain is correctly dependency-held
  and all four A100-80GB devices remain idle. Quota is home `88.6%`, scratch
  `43.0%`; held-out access remains zero and Sheet E/F/G remain blank.

## B7 fourth artifact and final stretch — 02:37 SAST

- B7 seed-13 `1267877` completed its step-12328 exact callback at `02:15:21`
  and resumed healthy training at `12750/15410`. The 128-row artifact SHA-256
  is `6b33cac0310dffa26e83adec4563f3deb4afe723c87321ad631f63cda1c4eb62`.
  It has `64/64` Xho/Zul coverage and unique predictions, no empty prediction,
  no malformed prompt boundary, and no generated EOS marker. Validation chrF
  is `24.255220269384274` Xho and `26.249710081242622` Zul; this remains
  interim validation-only evidence and does not select or freeze a winner.
- A2 seed-13 `1267878` remains healthy at `14509/15410`, about 901 optimizer
  steps from its terminal boundary. Both jobs remain on A100-40GB without a
  fault marker. The gate chain remains correctly dependency-held and all four
  A100-80GB devices remain idle. Quota is home `88.6%`, scratch `43.0%`;
  held-out access remains zero and Sheet E/F/G remain blank.

## Fourth exact-boundary artifacts — 01:34 SAST

- A2 seed-13 `1267878` completed its step-12328 exact callback at `00:41:35`
  and resumed healthy training at `13313/15410`. The 128-row artifact SHA-256
  is `5aea656ea4eac5de9db793e76dc484972ab57c82ac48d9db41bfacacd4720d47`.
  It has `64/64` Xho/Zul coverage and unique predictions, no empty prediction,
  no malformed prompt boundary, and no generated EOS marker. Validation chrF
  is `23.47910667520067` Xho and `25.548802315907626` Zul; this remains interim
  validation-only evidence and does not select or freeze a winner.
- B7 seed-13 `1267877` reached step 12328 and entered its fourth exact callback
  after complete declared validation. Its health-only loss is
  `2.265365966955305`; neither that loss nor the incomplete callback is used
  for selection. Both jobs remain healthy on A100-40GB with no fault marker.
- Gate jobs `1269243 -> 1269245 -> 1269246` remain dependency-held as intended;
  all four A100-80GB devices remain idle. Quota is home `88.6%`, scratch
  `43.0%`; held-out access remains zero and Sheet E/F/G remain blank.

## Seed-13 confirmations — 00:33 SAST

- A2 seed-13 `1267878` reached the frozen step-12328 validation boundary and
  entered its fourth exact-generation callback after complete declared
  validation. Its health-only loss is `2.2400511507087524`; neither that loss
  nor the incomplete callback is used for selection.
- B7 seed-13 `1267877` remains healthy at `11404/15410`. Both confirmations
  remain the only owned GPU jobs on A100-40GB; targeted fault scans are empty.
- The automatic `1269243 -> 1269245 -> 1269246` A100 equivalence gate remains
  dependency-held as intended. All four A100-80GB devices are idle. Quota is
  home `88.6%` and scratch `43.0%`; held-out access remains zero and Sheet
  E/F/G remain blank.
