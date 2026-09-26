# Pure-GDN validation-only HPO — 2026-08-17

## POS a1 step 1981 reaches 1550 rows — 23:52 SAST

- A1 seed-13 job `1241720` remains healthy and fault-free on
  `srvrocgpu010` A100-40GB. Its step-1981 exact callback reached
  `1,550/1,800` rows; the remaining `250` rows project a complete artifact
  near `00:10--00:15 SAST` on 18 August.
- This is operational progress only. A1 remains incomplete, POS remains
  unfrozen, and no interim metric has been used to select or prune it.
- It is the sole owned job and no A100-80GB/L40S work overlaps it. Quota is
  home `88.6%`, scratch `39.2%`; held-out remains `0`, Sheet E/F/G remain
  blank, last verified Kombuys state remains read-only/idle, and quarantine
  and publication gates are unchanged.

## POS a1 step 1981 reaches 1150 rows — 23:22 SAST

- A1 seed-13 job `1241720` remains healthy and fault-free on
  `srvrocgpu010` A100-40GB. Its step-1981 exact callback reached
  `1,150/1,800` rows; observed throughput keeps the complete-artifact ETA
  near `00:05--00:15 SAST` on 18 August.
- This is operational progress only. A1 remains incomplete, POS remains
  unfrozen, and no interim metric has been used to select or prune it.
- It is the sole owned job and no A100-80GB/L40S work overlaps it. Quota is
  home `88.6%`, scratch `39.2%`; held-out remains `0`, Sheet E/F/G remain
  blank, last verified Kombuys state remains read-only/idle, and quarantine
  and publication gates are unchanged.

## POS a1 step 1981 reaches 750 rows — 22:52 SAST

- A1 seed-13 job `1241720` remains healthy and fault-free on
  `srvrocgpu010` A100-40GB. Its step-1981 exact callback reached
  `750/1,800` rows; observed throughput keeps the complete-artifact ETA near
  `00:05--00:15 SAST` on 18 August.
- This is operational progress only. A1 remains incomplete, POS remains
  unfrozen, and no interim metric has been used to select or prune it.
- It is the sole owned job and no A100-80GB/L40S work overlaps it. Quota is
  home `88.6%`, scratch `39.2%`; held-out remains `0`, Sheet E/F/G remain
  blank, last verified Kombuys state remains read-only/idle, and quarantine
  and publication gates are unchanged.

## POS a1 step 1981 monitoring — 22:22 SAST

- A1 seed-13 job `1241720` remains healthy and fault-free on
  `srvrocgpu010` A100-40GB. Its step-1981 exact callback reached
  `350/1,800` rows; observed throughput projects a complete artifact near
  `00:05--00:15 SAST` on 18 August.
- This is operational progress only. A1 remains incomplete, POS remains
  unfrozen, and no interim metric has been used to select or prune it.
- It is the sole owned job and no A100-80GB/L40S work overlaps it. Quota is
  home `88.6%`, scratch `39.2%`; held-out remains `0`, Sheet E/F/G remain
  blank, last verified Kombuys state remains read-only/idle, and quarantine
  and publication gates are unchanged.

## POS a1 step 1698 complete — 21:52 SAST

- A1 seed-13 job `1241720` independently hash-verified its complete
  step-1698 exact artifact over all `1,800` rows, `12` cells, and `17`
  labels. Token accuracy is `0.843005859282623`; artifact SHA-256 is
  `ea0ac77781f762cb871143a2bfd951bc1c8b87399de35b2fbf5efe62a332c5f0`.
- The retained best remains checkpoint 1415 at `0.8452102586327794`, leaving
  the provisional a1 two-seed mean unchanged at `0.8463392380469843`. The
  job remains healthy and fault-free and has entered its step-1981 declared
  validation; the exact callback is expected next, with a complete artifact
  projected near `00:05--00:20 SAST` on 18 August.
- This remains operational progress only. A1 is not terminal-valid, POS is
  not frozen, and no partial metric has informed selection or pruning. It is
  the sole owned A100-40GB job, with no A100-80GB/L40S overlap. Quota is home
  `88.6%`, scratch `39.2%`; held-out remains `0`, Sheet E/F/G remain blank,
  Kombuys remains read-only, and quarantine/publication gates are unchanged.

## POS a1 step 1698 reaches 1600 rows — 21:22 SAST

- A1 seed-13 job `1241720` remains healthy on `srvrocgpu010` A100-40GB at
  `1,600/1,800` rows in the step-1698 exact callback. Targeted traceback,
  OOM, runtime, assertion, NCCL, and killed-process scans are empty. Observed
  throughput projects the complete artifact near `21:37--21:42 SAST`.
- This remains operational progress only. A1 is incomplete, POS remains
  unfrozen, and no partial metric has informed selection or pruning.
- It is the sole owned job and no A100-80GB/L40S work overlaps it.
  `srvrocgpu009` is idle but exposes the excluded `gpu:amperemk` GRES. Quota
  is home `88.6%`, scratch `39.2%`; held-out remains `0`, Sheet E/F/G remain
  blank, Kombuys remains read-only, and quarantine/publication gates are
  unchanged.

## POS a1 step 1698 monitoring — 20:52 SAST

- A1 seed-13 job `1241720` remains healthy and fault-free on
  `srvrocgpu010` A100-40GB. Its step-1698 exact callback reached
  `1,200/1,800` rows; observed throughput projects a complete artifact near
  `21:35--21:40 SAST`.
- This is operational progress only. A1 remains incomplete, POS remains
  unfrozen, and no interim metric has been used to select or prune it.
- The sole owned job is A100-40GB; no A100-80GB/L40S work is active. Quota is
  home `88.6%`, scratch `39.2%`; held-out remains `0`, Sheet E/F/G remain
  blank, Kombuys remains read-only, and quarantine/publication gates are
  unchanged.

## POS a1 step 1415 — 19:22 SAST

- A1 seed-13 job `1241720` completed exact step-1415 validation over all
  `1,800` rows, `12` cells, and `17` labels with token accuracy
  `0.8452102586327794`.
- Artifact SHA-256 is
  `4406ed9b0d44c8c386904744d912a5ac6e7e61b64c94503daeb49320ad70eb1f`;
  `sha256sum -c` passes from the artifact directory.
- Its provisional mean with seed-42 is `0.8463392380469843`, still below
  terminal-valid a2's `0.8606543221600175`. A1 remains incomplete and may
  not be pruned or selected from this interim comparison; it reached the
  step-1698 boundary healthy.
- One A100-40GB job is owned and no A100-80GB/L40S work is active. Quota is
  home `88.6%`, scratch `39.2%`; held-out remains untouched, Sheet E/F/G
  remain blank, Kombuys remains read-only, and quarantine/publication gates
  are unchanged.

## POS a1 step 1132 — 16:46 SAST

- A1 seed-13 job `1241720` completed exact step-1132 validation over all
  `1,800` rows, `12` cells, and `17` labels with token accuracy
  `0.8363515921069141`.
- Artifact SHA-256 is
  `e76d7a1ba485336c597e2243db0e01c5fd9cb6169403d3ba81db169b3c85c2c1`;
  `sha256sum -c` passes from the artifact directory.
- Its provisional mean with fixed seed-42 score `0.8474682174611892` is
  `0.8419099047840517`, below terminal-valid a2's `0.8606543221600175`.
  This interim comparison cannot prune or select a1; the preregistered run
  remains incomplete and resumed training toward step 1415.
- One A100-40GB job is owned and no A100-80GB/L40S work is active. Quota is
  home `88.6%`, scratch `39.2%`; held-out remains untouched, Sheet E/F/G
  remain blank, Kombuys remains read-only, and quarantine/publication gates
  are unchanged.

## POS a2 terminal and a1 step 849 — 14:16 SAST

- A2 seed-13 job `1241719` completed `0:0` at `14:10:13 SAST` after
  `15:17:59`. Its terminal step-1698 exact validation artifact covers all
  `1,800` rows, `12` cells, and `17` labels and scores
  `0.8538223932360355`. SHA-256
  `be14c27a3ea8c882eb49ecdb2822e13c68ca538381acee2606b6456e4d780879`
  passes `sha256sum -c`.
- The seed-13 retained winner is step 1132 at `0.8618166664037479`; with the
  fixed seed-42 score `0.8594919779162872`, a2's final preregistered two-seed
  arithmetic mean is `0.8606543221600175`.
- A1 seed-13 job `1241720` completed exact step-849 validation with the same
  full coverage and improved to `0.8326089557196429`. SHA-256
  `ef6ec65f0406f8e35b31e9404f19abca287d421079850a72889bc4a4db5b401a`
  passes `sha256sum -c`; step-1132 validation then began.
- A2 is terminal-valid, but POS is not frozen until a1 completes. Owned GPU
  state is one running A100-40GB job and no A100-80GB/L40S work. Quota is
  home `88.6%`, scratch `39.2%`; held-out remains untouched, Sheet E/F/G
  remain blank, Kombuys remains read-only, and quarantine/publication gates
  are unchanged.

## POS a2 step 1415 and a1 step 566 — 11:46 SAST

- A2 seed-13 job `1241719` completed exact step-1415 validation over all
  `1,800` rows, `12` cells, and `17` labels with token accuracy
  `0.8601064478765359`. Artifact SHA-256 is
  `22315c34915b1e38b665394f54e166bc08cad6a32714a435e8bba14ac1439bea`;
  `sha256sum -c` passes. This does not exceed its eligible step-1132 best
  `0.8618166664037479`.
- A1 seed-13 job `1241720` completed exact step-566 validation with the same
  complete coverage and improved to `0.8065868856875102`. Artifact SHA-256
  is `2ef415d2dea2592b635428a176b9e09ccb387b8bfdbed94a8dba636b50e5a960`;
  `sha256sum -c` passes. Its step-849 callback began and reached `50/1,800`.
- Both are eligible interim checkpoints only, not terminal seed results or a
  frozen POS winner. Jobs `1241719/1241720` remain healthy on separate
  A100-40GB GPUs. Quota is home `88.6%`, scratch `39.1%`; no A100-80GB or
  L40S work is owned. Held-out remains untouched, Sheet E/F/G remain blank,
  Kombuys remains read-only, and quarantine/publication gates are unchanged.

## POS a2 step 1132 — 09:24 SAST

- A2 seed-13 job `1241719` completed exact step-1132 validation over all
  `1,800` rows, `12` cells, and `17` labels with token accuracy
  `0.8618166664037479`.
- Artifact SHA-256 is
  `bf82daa0fb5f7261e130985ff6ee35cc5e40cb102830025c1ddd9ec5bfdd8b94`;
  `sha256sum -c` passes from the artifact directory.
- This is an eligible within-run improvement over step 849, not the terminal
  seed result or winner. The job reached its next step-1415 evaluation
  boundary.
- A1 seed-13 job `1241720` remained healthy at `250/1,800` rows in its
  step-566 exact callback. Both jobs remain on separate A100-40GB GPUs;
  quota is home `88.6%`, scratch `39.1%`. Held-out, Sheet E/F/G, Kombuys
  read-only, quarantine, and publication gates remain unchanged.

## POS a1 step 283 — 08:54 SAST

- A1 seed-13 job `1241720` completed exact step-283 validation over all
  `1,800` rows, `12` cells, and `17` labels with token accuracy
  `0.726226396953919`.
- Artifact SHA-256 is
  `4057cf4e88a2f1a7d07abab8a69017b2e6e0e367b43e407285118798ab6bc75c`;
  `sha256sum -c` passes from the artifact directory.
- This is an eligible first within-run checkpoint, not the terminal seed
  result or winner. It resumed healthy training at step `423`.
- A2 seed-13 job `1241719` remained fault-free at `1,600/1,800` rows in its
  step-1132 exact callback, projecting a complete artifact near `09:10`.
  Both jobs remain on separate A100-40GB GPUs; quota is home `88.6%`, scratch
  `39.1%`. Held-out, Sheet E/F/G, Kombuys read-only, quarantine, and
  publication gates remain unchanged.

## POS a2 step 849 — 06:54 SAST

- A2 seed-13 job `1241719` completed exact step-849 validation over all
  `1,800` rows, `12` cells, and `17` labels with token accuracy
  `0.8535822546623176`.
- Artifact SHA-256 is
  `f28d700a95045225f8513a13749a119875d658eae7b1c4b4e8a453971ffc3b23`;
  `sha256sum -c` passes from the artifact directory.
- This is an eligible within-run improvement over step 566, not the terminal
  seed result or winner. Its step-1132 exact callback began and reached
  `50/1,800` rows.
- A1 seed-13 job `1241720` remained healthy in its first step-283 exact
  callback at `300/1,800` rows. Both jobs are fault-free on separate
  A100-40GB GPUs; no A100-80GB/L40S work is owned. Quota remains home
  `88.6%`, scratch `39.1%`; held-out, Sheet E/F/G, Kombuys read-only,
  quarantine, and publication gates are unchanged.

## Both POS confirmations active — 06:24 SAST

- A1 seed-13 job `1241720` started at `06:09:27 SAST` on a separate
  A100-40GB on `srvrocgpu010`, more than five hours ahead of its earlier
  scheduler estimate. It verified all `695` immutable source/config files;
  execution-manifest SHA-256 is
  `b723f33b64a6e9fb895a9b370dffed357fca9ea31356877ad8f1070c314e46f2`.
  It loaded the canonical pure-GDN checkpoint and reached step `276` with no
  fault marker.
- A2 seed-13 job `1241719` remained healthy at `1,650/1,800` rows in its
  step-849 exact callback, still projecting a complete artifact near
  `06:35 SAST`.
- Owned state is now two running A100-40GB jobs with no pending,
  A100-80GB, or L40S work. Quota remains home `88.6%`, scratch `39.1%`.
  Held-out, Sheet E/F/G, Kombuys read-only, quarantine, and publication gates
  remain unchanged.

## POS queue acceleration — 05:54 SAST

- A2 seed-13 job `1241719` remains healthy and fault-free on one A100-40GB;
  its step-849 exact callback reached `1,250/1,800` rows and projects around
  `06:35 SAST` for a complete artifact.
- Slurm advanced pending a1 seed-13 job `1241720` from `11:58:10` to
  `06:38:15 SAST` (`Resources`), so it may start shortly after a2 releases
  the GPU. This is a scheduler estimate, not a guaranteed reservation.
- Quota is home `88.6%`, scratch `39.1%`; no A100-80GB/L40S work is owned.
  Held-out, Sheet E/F/G, Kombuys read-only, quarantine, and publication gates
  remain unchanged.

## POS a2 step 566 — 04:24 SAST

- Job `1241719` completed exact step-566 validation over `1,800` rows, `12`
  cells, and `17` labels with token accuracy `0.8276377533234429`.
- Artifact SHA-256 is
  `db177f15b2486d540757883eb082936d8d1a46385971c35113bf253bed3e5fca`;
  `sha256sum -c` passes from the artifact directory.
- This is an eligible within-run improvement over step 283, not the terminal
  seed result or winner. The step-849 callback is active at `100/1,800` rows.
- A1 seed-13 `1241720` remains Resources-pending for `11:58:10 SAST`.
  Quota is home `88.6%`, scratch `39.1%`; held-out, Sheet, GPU-family,
  Kombuys, quarantine, and publication gates are unchanged.

## POS reduced confirmation — 01:54 SAST

- A2 seed-13 job `1241719` is healthy on `srvrocgpu010`, one A100-40GB
  `gpu:ampere`, under the immutable 695-file snapshot and the preregistered
  budget-limited close-out.
- Step 283 completed exact constrained POS validation over all `1,800` rows,
  `12` language/prompt cells, and `17` labels. Token accuracy is
  `0.7823488792993896`; artifact SHA-256 is
  `d605b4289e770194da5849e1ce85c03828372e42c67db0671ef60775199f8f7c`
  and `sha256sum -c` passes from the artifact directory.
- This is an eligible within-run checkpoint only. It is not the terminal seed
  result and cannot freeze POS. The step-566 callback is active at
  `150/1,800` rows.
- A1 seed-13 `1241720` remains Resources-pending with Slurm estimate
  `2026-08-17 11:58:10 SAST`. Quota is home `88.6%`, scratch `39.1%`.
  There is no owned A100-80GB/L40S work. Held-out remains untouched, Sheet
  E/F/G remain blank, and Kombuys remains read-only.
