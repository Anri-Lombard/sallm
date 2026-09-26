# Pure-GDN validation HPO — 2026-08-23

## No avoidable approved-capacity idle — 21:46 SAST

- B5 `1258485` is actively training at step 9985/15410 on one A100-40GB at
  about 3.06 seconds/step. Jobs `1258898 -> 1258899 -> 1258900` are already
  dependency-queued, so the hardware gate and b6 have no manual hand-off gap.
- Live A100 allocation accounts for all approved devices: four one-GPU jobs
  occupy all four `gpu:ampere` devices on `srvrocgpu010`, and four one-GPU
  jobs occupy all four `gpu:ampere80` devices on `srvrocgpu011`. There is no
  approved idle A100-40GB or A100-80GB slot to use now.
- `srvrocgpu009` has four idle `gpu:amperemk` devices, but the current
  association cannot allocate them. The prepared support request is the only
  plausible capacity-side acceleration and remains unsent pending explicit
  approval. L40S and Kombuys remain intentionally excluded by the frozen
  hardware/scientific rules.
- The workload itself is not throughput-optimal: the earlier live audit found
  about 15.9% mean SM utilization, and full declared validation plus exact
  beam-generation callbacks add substantial wall time. Changing microbatch,
  evaluator, generation parallelism, or hardware mid-grid would invalidate
  comparability, so this is frozen protocol cost rather than orchestration
  idle. After the A100 equivalence gate passes, unchanged eligible work may
  fan out across both A100 variants up to the three-job cap.

## B5 clean through step 9246; checkpoint 9246 leads — 21:18 SAST

- Isolated b5 `1258485` completed clean exact callbacks at steps 6164 and
  9246, then resumed healthy A100-40GB training beyond step 9405/15410 at
  about 3.08 seconds/step. Each immutable artifact has exactly 128 rows
  (`64/64` Xho/Zul), zero empty predictions, `64/64` unique predictions per
  language, and clean `[BOS]` through `[EOS]<|assistant|>` boundaries.
- Step 6164 Xho/Zul chrF is
  `21.256444749325954/22.283805332395133` (mean
  `21.770125040860544`); artifact SHA-256 is
  `f788723151dc1005e2338589f3c32cb91bd31d1e5dda7b2c9b450571840e10f8`.
  Step 9246 Xho/Zul chrF is
  `23.09641266750219/23.682730256153707` (mean
  `23.38957146182795`); artifact SHA-256 is
  `dba75b13a189f1c9fe20fd116826caccd8dccf0aded5b76d13d39581c0036c78`.
  Trainer state retains checkpoint 9246 at that registered mean; its SHA-256
  is `ee5e84b3bc51e88282a2ebc00dd20c15a03baccb1b69495151eb16f1ade71a05`.
- This remains validation-only within-run evidence, not a terminal result.
  B5 is tentatively due around `05:00--06:00 SAST` on 24 August after its
  remaining frozen callbacks. Jobs `1258898 -> 1258899 -> 1258900` remain
  dependency-serial behind it. Quota is home `88.6%`, scratch `41.9%`; no
  L40S is used, Kombuys remains read-only, held-out access is `0`, Sheet
  E/F/G are blank, AfriHG remains `8/11` terminal-valid, and global freeze
  remains `0/8`.
- Live Sheet metadata identifies target sheetId `202608060` as `GDN Results`,
  not `Pure GatedDeltaNet Results`, and shows multiple other visible tabs.
  The sheetId itself still matches the frozen target. A read-only `E1:G43`
  check confirms only the three headers are populated and E2:G43 are blank;
  A39:G43 confirms quarantined T2X/AfriHG base values remain only in column D.
  No Sheet cell was changed.

## B5 step-3082 exact callback is clean; training resumes — 14:01 SAST

- Isolated b5 `1258485` completed its first exact validation callback at
  step 3082 and resumed healthy A100-40GB training beyond step 3498. The
  immutable artifact has `128` rows (`64/64` Xho/Zul), zero empty predictions,
  `64/64` unique predictions per language, and clean `[BOS]` through
  `[EOS]<|assistant|>` boundaries.
- Xho/Zul chrF is `21.145336441680488/22.225575303667362`; registered mean is
  `21.685455872673925`. Artifact SHA-256 is
  `38f0591c0ea93189a60b7a7c2c00c15c9f11c7650aae36c4c4b89d8628718850`.
  This is clean validation-only within-run evidence, not terminal evidence:
  b5 has no retained winner or scientific terminal status yet.

## B4 verified; isolated b5 starts cleanly — 09:47 SAST

- AfriHG b4 `1253374` completed `0:0` at `09:40:45` after `18:46:46`.
  Its terminal artifact has exactly `128` rows (`64/64` Xho/Zul), zero empty
  predictions, `64/64` unique predictions per language, clean prompt
  boundaries, and Xho/Zul chrF `23.17451792788935/24.664075832542633`
  (mean `23.919296880215992`). Terminal mean is below retained step-12328
  mean `23.9692081245192`, so checkpoint 12328 remains the validation-only
  winner. Terminal artifact SHA-256 is
  `bd52036d3b1f59185b26fd00f29f1d49da400793478ccde5eeb12ab930573f24`.
- CPU verifier `1258498` completed `0:0` and proved exact equality across all
  `424` retained/final tensor keys and `69,438,784` values, with no missing,
  shape, dtype, or value mismatches. Verifier-log SHA-256 is
  `0d7a3c19edf70fdb192fa8afbf712828b64c2a5e261977c0bd833c2143f09b3b`.
  B4 and candidate b4 are therefore scientifically terminal-valid; AfriHG
  advances from `7/11` to `8/11`.
- Isolated A100-40GB b5 `1258485` started at `09:40:45` on `srvrocgpu010`.
  It loaded the immutable 694-file snapshot, exact registry candidate b5, and
  isolated `b5-a10040-r1` output. At `09:47` it had completed train/eval
  tokenization and was training beyond step 80/15410 at about 3.1 seconds per
  step with no fault marker. Training-only ETA is about 13 hours; frozen
  validation callbacks make terminal verification tentatively due early on
  24 August.
- Strictly serial jobs are now queued: A100-80GB equivalence replacement
  `1258898` after b5, CPU comparator `1258899` after the replacement, and
  mandatory A100-40GB b6 `1258900` after comparator success. Quota is
  home `88.6%`, scratch `41.9%`; no L40S is used, Kombuys remains read-only,
  held-out adapter access is `0`, Sheet E/F/G are blank, and global freeze
  remains `0/8`.

## B5 isolation preflight fails closed — 09:35 SAST

- No b5 job was submitted. The immutable wrapper predates local support for
  run/output/logging override variables and therefore ignored the isolated
  names during a dry run. It rewrote exactly three metadata files in the
  quarantined default b5 root at `09:34`: `execution_manifest.json`, its
  `.sha256`, and `hpo_trial.json`. Their new SHA-256 values are
  `992daf97...03523`, `ec761832...0880`, and `00ee2a34...32f`.
  The preserved partial checkpoint files were not modified.
- This is new failure provenance and the affected default root remains
  quarantined. Do not restore, delete, or treat its metadata as the original
  cancelled-run manifest. The preflight correctly stopped before `sbatch`.
- For the isolated b5 execution, use the immutable snapshot's existing
  `run_pure_gdn_validation_trial.sh` directly with every scientific variable
  resolved from the hashed registry and exported explicitly. This bypasses
  only the wrapper lines that hard-code default run/output/logging paths; the
  runner, source snapshot, model, data, metric, candidate, seed, and training
  behavior remain unchanged. Require an isolated dry-run manifest and exact
  scientific-field reconciliation before submission.

## Preregistered isolated A100-40GB b5 execution — 09:34 SAST

- Before submission and without inspecting the quarantined A100-80GB b5
  partial result, freeze one full A100-40GB execution of mandatory AfriHG
  Stage-B candidate b5. Candidate, seed, registry, model, data, scorer,
  checkpointing, batch construction, source snapshot, and all scientific
  settings remain unchanged.
- The default b5 output root contains the preserved partial files from
  cancelled job `1257792`, so the eligible A100-40GB execution must use new
  isolated run/output/logging names ending `b5-a10040-r1`. It must not resume,
  read, delete, or overwrite the partial directory.
- To preserve the pre-pass hardware rule, b5 must depend on successful b4
  completion and corrected A100-80GB gate candidate `1258453` must in turn
  depend on successful b5 completion. This keeps all pre-pass scientific work
  on A100-40GB while using the currently free A100-40GB slot; it does not
  change any selection rule or consult held-out data.

## AfriHG b4 enters terminal exact generation — 09:01 SAST

- A100-40GB job `1253374` reached exactly `15410/15410` at `08:36 SAST`
  after `17:40:10` of training. It completed full declared terminal
  validation at health-only loss `2.2331547721649905` and entered the frozen
  exact-generation callback at `08:40:08`; the log remained active through
  `08:53:22` with no traceback, CUDA, NCCL, OOM, or fault marker.
- No terminal artifact exists yet. Validation-only checkpoint 12328 remains
  the audited within-run winner at mean chrF `23.9692081245192`; b4 remains
  non-terminal and AfriHG remains `7/11` terminal-valid. Based on prior exact
  callbacks, terminal output is tentatively due around `09:30--09:50 SAST`.
- Corrected A100-80GB gate candidate `1258453` remains
  `AssocGrpGRES`-pending behind four other-user jobs, with Slurm's current
  start estimate unchanged at `2026-08-24 22:17:48 SAST`. Quota remains home
  `88.6%`, scratch `41.7%`; no L40S is used, Kombuys remains read-only,
  global freeze is `0/8`, held-out adapter access is `0`, and Sheet E/F/G are
  blank.

## Corrected A100-40 reference completes — 07:31 SAST

- Corrected gate reference `1258452` completed `0:0` in `00:01:00` on
  `srvrocgpu010`. Its result records the prospectively corrected
  `slurm_job_gres="gpu:ampere:1"`; result and execution-manifest SHA-256
  values are `bb9fbf652c39de7c076cccd59ae551ac33b018f6ca93193a58e432e07c276b47`
  and `0b11f97073258f02fad3ea6088b5d527a16151e463dbd7cd24bcc77a927707e0`.
  This is operational completion only: the hardware gate cannot pass until
  candidate `1258453` completes and the frozen comparator passes.
- Candidate `1258453` is eligible but `AssocGrpGRES`-pending. All four
  A100-80GB devices are currently occupied by another user's jobs
  `1257780/1257781/1257782/1257800`; Slurm's current start estimate is
  `2026-08-24 22:17:48 SAST`. No owned job or other user's job was altered.
- AfriHG b4 `1253374` remains healthy on A100-40GB beyond
  `14195/15410`; training is tentatively due to finish around `08:35 SAST`,
  followed by its frozen terminal validation and exact-generation path.
  Terminal verification remains tentatively due around `10:00--10:30 SAST`.
  Quota is home `88.6%`, scratch `41.7%`. No L40S is used; Kombuys was not
  accessed and remains read-only. AfriHG remains `7/11` terminal-valid,
  global freeze `0/8`, held-out adapter access `0`, and Sheet E/F/G blank.

## AfriHG b4 improves at checkpoint 12328 — 06:08 SAST

- A100-40GB job `1253374` completed its preregistered step-12328 exact
  validation callback and resumed healthy training beyond step `12451/15410`.
  Its artifact has exactly `128` rows (`64/64` Xho/Zul), zero whitespace or
  debug-empty predictions, `64/64` unique predictions per language, and clean
  `[BOS]` through `[EOS]<|assistant|>` prompt boundaries.
- Xho/Zul chrF is `23.322881784975422/24.61553446406298`; registered mean
  chrF is `23.9692081245192`. This improves checkpoint 9246 and exactly
  matches `trainer_state.json`, so checkpoint 12328 is the current within-run
  best. Artifact and trainer-state SHA-256 values are
  `4823dd05a44dcd9f6c870fc3b1b4b71fb159fca36c7c943d1591782d270b7f26`
  and `867ecef52627629a274b22b0648fb68fbf92395aca61eb446065cab9753b2384`.
  This remains non-terminal validation-only evidence.
- GRES-metadata correction reference `1258452` is Priority-pending with a
  provisional `17:54` SAST start; candidate `1258453` is dependency-pending.
  B4 is the only running owned job. A100-80GB remains unratified, no L40S is
  used, AfriHG remains `7/11` terminal-valid, global freeze remains `0/8`,
  held-out adapter access remains `0`, and Sheet E/F/G remain blank.

## AfriHG b4 improves at checkpoint 9246 — 02:33 SAST

- A100-40GB job `1253374` completed its preregistered step-9246 exact
  validation callback and resumed healthy training beyond step `9591/15410`.
  Its artifact has exactly `128` rows (`64/64` Xho/Zul), zero whitespace or
  debug-empty predictions, `64/64` unique predictions per language, and clean
  prompt boundaries: all prompts begin `[BOS]` and end
  `[EOS]<|assistant|>`.
- Xho/Zul chrF is `23.331670222361144/23.887552144959805`; registered mean
  chrF is `23.609611183660476`. This matches `trainer_state.json` and retains
  checkpoint 9246 as the within-run best. Artifact and trainer-state SHA-256
  values are `d47722612685d3f77502d79c51619529630548d567be9c08a4b5f40318ab941d`
  and `7baaf968c737ada2a5fee6e5d1c2bfaf87d28010153bb10d710ea8f9465dbc3c`.
  This is eligible validation-only within-run selector evidence, not a
  terminal-valid candidate result.
- Sealed A100-80GB b5 `1257792` remains running on `srvrocgpu011`; none of
  its logs, metrics, checkpoints, or result contents were inspected.
  A100-40GB equivalence reference `1257517` remains Priority-pending. Owned
  work is exactly these three jobs, with no L40S. AfriHG remains `7/11`
  terminal-valid, global freeze `0/8`, and held-out adapter access `0`.
  Quota is home `88.6%`, scratch `41.6%`; Kombuys remains read-only with RTX
  5090 untouched, and Sheet E/F/G remain blank.
