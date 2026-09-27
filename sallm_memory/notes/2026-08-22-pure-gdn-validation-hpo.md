# Pure-GDN validation HPO — 2026-08-22

## Sealed A100-80GB execution begins — 22:55 SAST

- Under the user-approved pre-gate sealed-execution amendment, dependency
  gating was removed from A100-80GB equivalence replacement `1257520` only
  after b4 `1253374`, reference `1257517`, the three-job cap, and the absent
  replacement output were rechecked. Job `1257520` ran on `srvrocgpu011`
  from `22:52:50` to `22:53:54` SAST and completed `0:0` in `00:01:04`.
  Its result contents were not opened; only existence and SHA-256
  `d90f1a645716a46eb82eb5972bb7422d4fc1e9e8042ac09c270858b9dd38ae98`
  were observed. It remains sealed and is not equivalence evidence until
  A100-40GB reference `1257517` completes and the fail-closed comparator
  passes.
- The canonical AfriHG b5 output was absent, no duplicate b5 job existed,
  and the immutable registry again matched SHA-256
  `8fdd6ea5f192a1d2ef2267353162b9a640ccd09ae0d889cb5a999645198bb726`.
  The snapshot launcher is intentionally read-only (`-r--r--r--`) and is
  invoked explicitly through `bash`, so execute permission is not required.
  Exact frozen b5 was submitted once as job `1257792`, changing only the
  approved hardware association to `nlpgroup80/nlpgroup80` and GRES to one
  `gpu:ampere80` while retaining 24 hours, eight CPUs, canonical working
  directory, immutable source, mutable runtime, seed 42, and recipe `b5`.
- Although its pre-submission scheduler probe estimated `2026-08-23
  23:25:03` SAST, job `1257792` started at `22:55:20` SAST on
  `srvrocgpu011`. Owned work is exactly b4 `1253374` running on A100-40GB,
  reference `1257517` pending on A100-40GB, and sealed b5 `1257792` running
  on A100-80GB. No L40S is in use. AfriHG remains `7/11`
  terminal-valid, global freeze `0/8`, and held-out adapter access `0`;
  Sheet E/F/G remain blank. Quota is home `88.6%`, scratch `41.5%`, and
  Kombuys remains read-only with RTX 5090 untouched.

## A100-40GB queue self-unblock audit — 21:00 SAST

- No job was changed. A live read-only check found b4 `1253374` healthy at
  roughly `5733/15410` after `06:05`, while A100-40GB reference `1257517`
  remains Priority-pending and sequential A100-80GB replacement `1257520`
  remains dependency-pending.
- Exact scheduler-only probes kept the reference launcher, account/partition/
  QOS, GRES, eight CPUs, `73,136 MB` memory, and working directory unchanged.
  Wall-time requests of `00:10`, `00:30`, `02:00`, and `24:00` all returned
  the same projected start, `2026-08-24 08:23:59 SAST`. The 24-hour envelope
  is therefore not the current queue blocker; shortening it would not unblock
  the equivalence gate.
- `srvrocgpu009` remains idle with four `gpu:amperemk` A100s, but test-only
  requests through both owned A100 associations (`nlpgroup` and
  `nlpgroup80`) fail `AssocGrpGRES`. There is no self-service association
  route to those devices. `srvrocgpu011` remains idle with four allowed
  `gpu:ampere80` devices, but they cannot be used for HPO until the paired
  A100-40GB reference passes under the prospective amendment.
- The scientifically clean unblock is external scheduler/access relief: send
  the already-prepared `amperemk` support request or wait for normal
  `gpu:ampere` priority. Reusing the quarantined overlapping canary, changing
  the preregistered gate after seeing a candidate artifact, or co-locating the
  reference on b4's running GPU remain prohibited. Quota is home `88.6%` and
  scratch `41.5%`; held-out access remains `0`, global freeze remains `0/8`,
  and Sheet E/F/G remain blank.

## Sequential A100 equivalence gate queued — 20:41 SAST

- The user approved a prospective validation-only A100-40GB/A100-80GB
  equivalence gate, concurrent A100 variants only after it passes, the full
  unchanged AfriHG and General grids, and preparation—but not sending—of an
  `amperemk` access request. Held-out access remains `0`.
- Local gate verification is clean: the focused test is `1 passed`, Ruff is
  clean, Python compilation passes, and the launcher is `bash -n` clean.
  Benchmark, launcher, current verifier, current test, and original amendment
  SHA-256 values are `ba17f775...`, `aac78d35...`, `64c8e42a...`,
  `2a474b48...`, and `4de45d89...` respectively.
- The new read-only source snapshot is
  `/home/lmbanr001/masters/sallm_snapshots/pure-gdn-a100-equivalence-20260822-38d367bf`.
  It contains `702` files, hashes `698` source/script files, occupies `1.7M`,
  and its complete file-manifest SHA-256 is
  `ccef0a5174067d3db485ec5bcc4912e82df1c42f41a613f893f4d6068e7b31d4`.
- B5/b6 `1257468/1257469` were rechecked as Priority-pending, never-started,
  and absent from both output and log roots, then cancelled unchanged to free
  the two gate slots. B4 `1253374` remained healthy beyond `5347/15410`.
- A100-40GB reference `1257517` is Priority-pending. The first A100-80GB
  canary `1257518` started automatically on `srvrocgpu011` and completed
  `0:0` in `00:01:02` while b4 was still active. Its result contents were not
  inspected; the artifact is quarantined because pre-pass cross-variant
  overlap was not authorized. Its result and execution-manifest SHA-256
  values are `0538cc18...` and `fccc663f...`.
- Sequential replacement `1257520` is Dependency-pending on
  `afterany:1253374,afterok:1257517`, with a new output prefix. The gate will
  compare only `1257517/1257520`; no A100-80GB HPO may start before it passes.
  The ordering clarification was copied read-only beside the gate artifacts at
  SHA-256
  `8f87242ec2c41e3f09c1397e6dc2fb43a72ffa72a8fb58770ae4b5280c5d0c47`.
  Independent review found that the first comparator revision did not enforce
  execution-manifest identity. The current fail-closed comparator now requires
  matching manifest schema, immutable snapshot, complete source hashes, model
  hashes, Python/platform/package environment, and stable kernel variables. It
  and its passing test are deployed read-only beside the gate artifacts at the
  full hashes above; the GPU benchmark does not import the comparator.
  Owned work is exactly b4 plus the two gate jobs. Quota is home `88.6%` and
  scratch `41.5%`; no L40S work exists, Kombuys remains read-only with RTX
  5090 untouched, Sheet E/F/G remain blank, and global freeze remains `0/8`.

## B4 epoch-one artifact verified — 18:55 SAST

- B4 `1253374` completed its first preregistered exact-generation callback and
  resumed training beyond step `3322/15410`. The step-3082 artifact has exactly
  `128` rows with complete `64/64` Xho/Zul coverage, zero empty predictions,
  and `128/128` unique predictions. Xho/Zul chrF is
  `21.277088395946436/22.53969039418806`, mean `21.908389395067248`; this is
  validation-only within-run evidence, not a terminal candidate result. The
  artifact SHA-256 is
  `72999ffbce1405585374c8aa54698ebd31694dd65a640254728e653d48443e96`.
- B5/b6 `1257468/1257469` remain Priority-pending, so owned work remains
  exactly three A100-40GB `gpu:ampere` jobs. Quota is home `88.6%`, scratch
  `41.5%`; Kombuys is read-only, held-out access remains `0`, global freeze
  remains `0/8`, and Sheet E/F/G remain blank.

## Exact proofs complete; AfriHG queue filled — 17:52 SAST

- CPU-only exact verifier `1257467` completed `0:0` in four seconds on
  `srvrochpc100`. B2 and b3 retained BIN and final safetensors are exactly
  equal in keys, shapes, dtypes, and tensor values. B2 has `424` keys and
  `69,438,784` values with retained/final SHA-256
  `567f63f405236b0e12894ddfbc26ad4b5683d73a5fa8183daa0ce94f2607d1b5` /
  `36017276eec12f1a30da5a49d83dd0a7ba46a8f9717898d9fbe732e5efd2c15a`.
  B3 has `424` keys and `71,762,560` values with retained/final SHA-256
  `4e7fa2497781f545659791a45fe27e60da5e570f16dae4d9de47c4ae8c013b5e` /
  `ead9ba99c7833cc1aec3999f5f90741639e0d1ebce5fea0dbc4c0ca6b99aa333`.
  The verifier log SHA-256 is
  `6cee58f46d85859c4eee30a0704532469b54a6f5dda15f29a6e12d84bd1d626d`.
  AfriHG scientific progress therefore advances from `5/11` to `7/11`.
- GPU verifier jobs `1253806/1256680` were cancelled while still never-started
  and replaced by the equivalent CPU job because both scripts explicitly load
  and compare all tensors on CPU. Original scripts, jobs, and provenance are
  preserved. This frees accelerator capacity without changing any model,
  metric, artifact, or selection decision.
- B4 `1253374` remains healthy on `srvrocgpu010` and completed epoch-one loss
  evaluation at step `3082/15410`, `eval_loss=2.3000728302695084`. It is in
  the preregistered exact-generation callback with no fault marker.
- With two owned slots free, absent b5/b6 output and logging roots, no duplicate
  jobs, immutable snapshot permissions verified, registry validation passing
  at SHA-256
  `8fdd6ea5f192a1d2ef2267353162b9a640ccd09ae0d889cb5a999645198bb726`,
  and both frozen dry runs passing, b5 and b6 were submitted once as jobs
  `1257468/1257469`. Both request the required `nlpgroup/a100/nlpgroup`, one
  `gpu:ampere`, 24 hours, and eight CPUs. Both are Priority-pending without a
  scheduler ETA. Owned work is exactly three jobs: b4 running and b5/b6
  pending.
- A fresh `sbatch --test-only` request for idle `gpu:amperemk` still fails
  `AssocGrpGRES`; the existing prospective equivalence gate therefore cannot
  run. Idle A100-80GB remains excluded. Quota is home `88.6%`, scratch
  `41.4%`; Kombuys remains read-only, held-out access remains `0`, global
  freeze remains `0/8`, and Sheet E/F/G remain blank.

## Full-table distance and live capacity — 17:31 SAST

- AfriHG b4 job `1253374` remains healthy near its first epoch boundary at
  step `3075/15410` after `02:34:55`. B2/b3 tensor verifiers
  `1253806/1256680` remain Priority-pending. Scientific AfriHG progress is
  still `5/11`; operationally b2/b3 have completed and await exact proof.
- After b4, the frozen AfriHG program still requires b5--b7 and four Stage-C
  confirmation runs. At the measured roughly 19 hours per run, this is about
  `133` additional GPU-hours after b4. Corrected General then requires eleven
  seed-42 candidates plus four confirmations. The invalid historical General
  jobs ran about `14.6--15.2` hours without the omitted AfriHG portion, so a
  corrected run should be budgeted near `18` hours until measured otherwise,
  or roughly `270` General GPU-hours. These counts precede Monolingual
  validation selection and every one-time held-out adapter evaluation.
- The live A100 partition has `srvrocgpu009` idle with excluded
  `gpu:amperemk:4`, `srvrocgpu010` mixed with all four `gpu:ampere` devices
  occupied, and `srvrocgpu011` idle with excluded `gpu:ampere80:4`. B4 owns
  one eligible A100-40GB; three foreign array tasks own the other three.
  The current protocol permits only `gpu:ampere`, caps owned jobs at three,
  and forbids A100-80GB/L40S overlap. Kombuys remains read-only. There is
  therefore no additional scientifically admissible GPU available now.
- Conditional projection: one continuously available eligible GPU needs
  roughly `403` GPU-hours after b4 merely to close AfriHG and General, about
  `17` days before queue and verification delays. Three continuously
  available eligible GPUs reduce that compute floor to roughly six days.
  A full downstream table remains later because Monolingual selection has
  not started and held-out access remains `0`; no defensible fixed full-table
  date exists until the Monolingual execution plan and observed runtimes are
  known.

## AfriHG throughput audit — 17:15 SAST

- B4 job `1253374` remains healthy on `srvrocgpu010` A100-40GB, beyond
  step `2700/15410` at about `2.98 s/step`. The log proves that the intended
  FLA GatedDeltaNet fast path is available, `attn_mode=chunk` is frozen in
  the pure-GDN configuration, and the launcher aborts rather than permitting
  the slow torch fallback. This is therefore not an accidental fallback run.
- A 15-second live `nvidia-smi dmon` sample showed bursty SM utilization
  `3,34,2,3,49,3,3,39,4,3,37,4,3,38,4` (mean `15.9%`), power mostly
  `39--60 W` with two `97--104 W` bursts, and only `13,147 MiB` of the
  `40,960 MiB` framebuffer in use. The fast kernel is loaded, but the full
  training pipeline is not saturating the A100.
- The frozen run uses per-device batch `4`, gradient accumulation `2`, four
  data-loader workers, no packing, no padding-free mode, no gradient
  checkpointing, BF16, and `torch_compile=False`. The small 125M LoRA
  workload therefore launches short GPU bursts and leaves material capacity
  unused. Increasing the microbatch while reducing accumulation is a
  plausible throughput optimization, but it has not been benchmarked and
  cannot be introduced mid-grid because the preregistration freezes effective
  batch construction.
- Wall time is mainly intrinsic to the registered AfriHG protocol: five
  epochs produce `15,410` optimizer steps; each epoch also evaluates all
  `3,082` declared validation rows and runs the exact 128-example,
  two-language beam-search generation callback. Prior b2/b3 wall times were
  about 19 hours, consistent with roughly 13 hours of optimizer steps plus
  roughly 5--6 hours of validation/generation and checkpoint overhead.
  Scientifically fair acceleration therefore requires a separate
  validation-blind performance canary and, if adopted, matched reruns rather
  than changing b4 or later candidates in place.

## AfriHG b4 starts healthy — 15:13 SAST

- B4 job `1253374` started at `14:53:59 SAST` on `srvrocgpu010` A100-40GB,
  earlier than its prior scheduler estimate. The live Slurm envelope is the
  required account/partition/qos `nlpgroup/a100/nlpgroup`, one
  `gpu:ampere`, `24:00:00`, eight CPUs, and repository working directory.
  Its execution manifest SHA-256 is
  `cd32f4e1a65d8046c78b2cf3f63ae3b1a14303206d0564f97c8c830a7ff1bf85`;
  all `694` source/config files verified before training.
- The frozen registry recipe is seed `42`, LR `8.981661817441004e-05`,
  validation-only `eval_all_chrf` selection, and epoch-aligned evaluation and
  retention. The job is fault-free beyond step `347/15410` at about
  `3.1 s/step`. Its first exact validation artifact is tentatively expected
  around `18:35--19:00 SAST`; terminal output is roughly due around
  `2026-08-23 10:00 SAST`. These are operational estimates, not selection
  evidence.
- Exact b2/b3 verifier jobs `1253806/1256680` remain Priority-pending without
  scheduler estimates. Owned work is exactly b4 running plus two pending
  verifier jobs, all A100-40GB, with no A100-80GB or L40S work. AfriHG remains
  `5/11` terminal-valid and global freeze remains `0/8`. Quota is home
  `88.6%`, scratch `41.4%`; Kombuys remains read-only with RTX 5090
  untouched, held-out access is `0`, Sheet E/F/G remain blank, and
  General/Monolingual/publication remain blocked.

## AfriHG b3 completes; exact tensor proof queued — 12:43 SAST

- B3 job `1250970` completed `0:0` at about `12:20:51 SAST` after
  `19:05:12` on `srvrocgpu010` A100-40GB. Its terminal step-15410 artifact
  has `128` rows (`64/64` Xho/Zul), zero exact, whitespace, or debug-empty
  predictions, `64/64` unique predictions per language, and clean prompt
  boundaries. All prompts start `[BOS]`, end `[EOS]<|assistant|>`, and
  contain zero EOS tokens after the assistant marker.
- Terminal Xho/Zul chrF is `23.614333434018096/24.862300215885675`; mean
  `24.238316824951886`. This does not improve validation-only retained
  checkpoint 12328 at mean `24.344455545302008`. Terminal-artifact,
  retained-trainer-state, retained-BIN, and final-safetensors SHA-256 values
  are `27ad84c93ec691143f80bc4ee1b15ab6f2bb2a139612617a95e540f998393c32`,
  `d3935936721e6a2a6b20b674e15ee70df153ea3e48f1e2890d700f49f94a4c29`,
  `4e7fa2497781f545659791a45fe27e60da5e570f16dae4d9de47c4ae8c013b5e`,
  and `ead9ba99c7833cc1aec3999f5f90741639e0d1ebce5fea0dbc4c0ca6b99aa333`.
  Retained/final adapter configs are byte-identical at SHA-256
  `3097a0c6627b28ce8b57d4e1ebb11856e9d15f10c2efe048341898c781d9f703`.
- With exactly two owned A100-40GB jobs pending and no duplicate, the freed
  slot was used once for read-only exact tensor verifier job `1256680`. Its
  `bash -n` clean standalone script is stored at
  `sallm_memory/artifacts/2026-08-22/verify_afrihg_b3.sbatch`, SHA-256
  `774a5955e044a3eb651251d15bd431265b3b5b3b134e4a800c2ef1ee6d193dc1`.
  It requires exact key, tensor-count, shape, dtype, and value equality between
  checkpoint 12328 and the final adapter. The first deployment target used an
  absent `$HOME/masters/sallm/slurm` directory and failed before submission;
  the same hashed bytes were then deployed to the repository root and
  submitted once. The verifier is Priority-pending.
- B2 verifier `1253806` is also Priority-pending without an ETA. B4 job
  `1253374` is Resources-pending with a provisional `2026-08-22 19:20:51
  SAST` start. Owned work is exactly three A100-40GB jobs, with no A100-80GB
  or L40S work. B3 is operationally complete but not yet scientifically
  terminal-valid, so AfriHG remains `5/11` and global freeze remains `0/8`.
  Quota is home `88.6%`, scratch `41.3%`; Kombuys remains read-only with RTX
  5090 untouched, held-out access is `0`, Sheet E/F/G remain blank, and
  General/Monolingual/publication remain blocked.

## AfriHG b3 enters terminal validation — 11:13 SAST

- B3 job `1250970` reached exactly `15410/15410` at about `11:11 SAST`
  after `17:55:32` of training and entered the frozen terminal validation path
  on `srvrocgpu010` A100-40GB. No terminal exact-generation artifact exists
  yet and no fault marker is present. Validation-only checkpoint 12328 remains
  the within-run winner at mean chrF `24.344455545302008`.
- Based on prior b3 callbacks, terminal exact-generation output is tentatively
  due around `12:15--12:45 SAST`. This is an operational estimate, not
  selection evidence. Exact b2 verifier `1253806` remains Priority-pending
  without an ETA; b4 `1253374` remains Priority-pending with provisional
  start `2026-08-23 00:15 SAST`.
- Owned work remains exactly three A100-40GB jobs, with no A100-80GB or L40S
  work. AfriHG remains `5/11` terminal-valid and global freeze remains `0/8`.
  Quota is home `88.6%`, scratch `41.2%`; Kombuys remains read-only with RTX
  5090 untouched, held-out access is `0`, Sheet E/F/G remain blank, and
  General/Monolingual/publication remain blocked.

## AfriHG b3 improves at checkpoint 12328 — 08:44 SAST

- B3 job `1250970` completed its frozen step-12328 callback and resumed
  fault-free training beyond step `12550/15410` on `srvrocgpu010`
  A100-40GB. Its exact artifact has `128` rows (`64/64` Xho/Zul), zero exact,
  whitespace, or debug-empty predictions, `64/64` unique predictions per
  language, and clean prompt boundaries. All prompts start `[BOS]`, end
  `[EOS]<|assistant|>`, and contain zero EOS tokens after the assistant
  marker.
- Xho/Zul chrF is `23.541671781341/25.147239309263014`; registered mean chrF
  is `24.344455545302008`. This exactly matches `trainer_state.json`, improves
  b3's step-9246 score, and moves its validation-only retained checkpoint to
  12328. Artifact and trainer-state SHA-256 values are
  `6290e2088ad381225e53c5401e1245e79abd9564eea8dcd59915c23cf4a3ff78`
  and `d3935936721e6a2a6b20b674e15ee70df153ea3e48f1e2890d700f49f94a4c29`.
  This is eligible within-run selector evidence, not terminal-valid evidence.
- B3 remains healthy and is tentatively due to enter its terminal callback
  around `11:20 SAST`, with terminal output around `12:30--13:00 SAST`.
  Exact b2 verifier `1253806` remains Priority-pending without a scheduler
  ETA; b4 `1253374` remains Priority-pending with provisional start
  `2026-08-23 00:15 SAST`. Owned work remains exactly three A100-40GB jobs,
  with no A100-80GB or L40S work. AfriHG remains `5/11` terminal-valid and
  global freeze remains `0/8`. Quota is home `88.6%`, scratch `41.2%`;
  Kombuys remains read-only with RTX 5090 untouched, held-out access is `0`,
  Sheet E/F/G remain blank, and General/Monolingual/publication remain
  blocked.

## AfriHG b3 enters checkpoint-12328 callback — 07:44 SAST

- B3 job `1250970` reached step `12328/15410`, evaluated all `3,082`
  declared AfriHG validation rows, and entered the frozen exact-generation
  callback on `srvrocgpu010` A100-40GB. Health-only validation loss is
  `2.2188514744129217`; the exact generation artifact does not exist yet, so
  checkpoint 9246 remains the audited within-run winner at mean chrF
  `24.12720516406921`.
- Automatic generation batch size 64 was selected for the first segment at
  `07:32:17 SAST`. Based on the prior b3 callbacks, the step-12328 artifact is
  tentatively expected around `08:40--09:00 SAST`; the terminal artifact is
  roughly due around `12:30--13:00 SAST`. These are operational estimates,
  not selection evidence.
- Exact b2 verifier `1253806` and b4 `1253374` remain Priority-pending. B4's
  current provisional start is `2026-08-23 00:15 SAST`; the verifier has no
  scheduler ETA. Owned work remains exactly three A100-40GB jobs, with no
  A100-80GB or L40S work. AfriHG remains `5/11` terminal-valid and global
  freeze remains `0/8`. Quota is home `88.6%`, scratch `41.2%`; Kombuys
  remains read-only with RTX 5090 untouched, held-out access is `0`, Sheet
  E/F/G remain blank, and General/Monolingual/publication remain blocked.

## AfriHG b2 completes; exact tensor proof queued — 06:15 SAST

- B2 job `1248577` completed `0:0` at `06:00:47 SAST` after `19:09:46` on
  `srvrocgpu010` A100-40GB. Its terminal step-15410 artifact has `128` rows
  (`64/64` Xho/Zul), zero exact, whitespace, or debug-empty predictions,
  `64/64` unique predictions per language, and clean prompt boundaries. All
  prompts start `[BOS]`, end `[EOS]<|assistant|>`, and contain zero EOS
  tokens after the assistant marker.
- Terminal Xho/Zul chrF is `21.745224131354718/22.99258739484475`; mean
  `22.368905763099733`. This does not improve the validation-only retained
  checkpoint 12328 at mean `22.382494324095585`. Terminal-artifact,
  retained-trainer-state, retained-BIN, and final-safetensors SHA-256 values
  are `7b3802ebb13659e5c7d7eca619e662bcf827cf5ad97b723b4e5e74d7d1054091`,
  `87c0a37744edab36842293eb2f7878212d41a7a8bf8cca0f2227903e8c1f658a`,
  `567f63f405236b0e12894ddfbc26ad4b5683d73a5fa8183daa0ce94f2607d1b5`,
  and `36017276eec12f1a30da5a49d83dd0a7ba46a8f9717898d9fbe732e5efd2c15a`.
  Retained/final adapter configs are byte-identical at SHA-256
  `a8f5181af649d9953f6b3e146367fd3153506562946041f92abc1ea318aa610f`.
- The freed slot was used once for read-only exact tensor verifier job
  `1253806`. Its `bash -n` clean standalone script is stored at
  `sallm_memory/artifacts/2026-08-22/verify_afrihg_b2.sbatch`, SHA-256
  `be6f3739dc7d4abe3724a905de242c1fda6dce6be443edf472aaa13ce16734b9`.
  It requires exact key, tensor-count, shape, dtype, and value equality between
  checkpoint 12328 and the final adapter. The verifier is Priority-pending;
  b2 is operationally complete but not yet scientifically terminal-valid.
- B3 `1250970` remains healthy beyond step `10887/15410`, retaining
  checkpoint 9246 at validation-only mean chrF `24.12720516406921`; b4
  `1253374` remains Priority-pending. Owned work is b3 running plus b4 and
  verifier `1253806` pending, all A100-40GB, with no A100-80GB or L40S work.
  AfriHG remains `5/11` terminal-valid and global freeze remains `0/8`.
  Quota is home `88.6%`, scratch `41.2%`; Kombuys remains read-only with RTX
  5090 untouched, held-out access is `0`, Sheet E/F/G remain blank, and
  General/Monolingual/publication remain blocked.

## AfriHG b3 improves at checkpoint 9246; b2 enters terminal callback — 05:13 SAST

- B3 job `1250970` completed its frozen step-9246 callback and resumed
  fault-free training beyond step 9730 on `srvrocgpu010` A100-40GB. Its exact
  artifact has `128` rows (`64/64` Xho/Zul), zero exact, whitespace, or
  debug-empty predictions, `64/64` unique predictions per language, and clean
  prompt boundaries. All prompts start `[BOS]`, end
  `[EOS]<|assistant|>`, and contain zero EOS tokens after the assistant
  marker.
- Xho/Zul chrF is `23.597619040378838/24.656791287759578`; registered mean
  chrF is `24.12720516406921`. This exactly matches `trainer_state.json`,
  improves b3's step-6164 score, and moves its validation-only retained
  checkpoint to 9246. Artifact and trainer-state SHA-256 values are
  `2847bbd52d82ec4a1f8d53d6ad18a316f02c4873747eef5d1588c201d5488e5c`
  and `6619a7336c797836bbe285ceaa9e250df90c79e8b43133443492f6d2a05715c8`.
  This is eligible within-run selector evidence, not terminal-valid evidence.
- B2 `1248577` reached `15410/15410`, completed all `3,082` declared
  validation rows at health-only loss `2.272301319278269`, and entered its
  frozen terminal exact-generation callback at `04:57:08 SAST`. No terminal
  artifact or retained/final reconciliation exists yet; the artifact remains
  tentatively due around `06:00--06:30 SAST`. B4 `1253374` remains
  Priority-pending without a scheduler estimate.
- Operationally b2 is in its terminal callback, b3 is training, and b4 is
  pending, all on A100-40GB; no A100-80GB or L40S work is owned.
  Scientifically AfriHG remains `5/11` terminal-valid and global freeze
  remains `0/8`. Quota is home `88.6%`, scratch `41.2%`; Kombuys remains
  read-only with RTX 5090 untouched, held-out access is `0`, Sheet E/F/G
  remain blank, and General/Monolingual/publication remain blocked.

## AfriHG b2 improves at checkpoint 12328 — 02:14 SAST

- B2 job `1248577` completed its frozen step-12328 callback and resumed
  fault-free training beyond step 12356 on `srvrocgpu010` A100-40GB. Its
  exact artifact has `128` rows (`64/64` Xho/Zul), zero exact, whitespace,
  or debug-empty predictions, `64/64` unique predictions per language, and
  clean prompt boundaries. All prompts start `[BOS]`, end
  `[EOS]<|assistant|>`, and contain zero EOS tokens after the assistant
  marker.
- Xho/Zul chrF is `21.913107235865905/22.851881412325266`; registered mean
  chrF is `22.382494324095585`. This exactly matches `trainer_state.json`,
  improves b2's step-9246 score, and moves its validation-only retained
  checkpoint to 12328. Artifact and trainer-state SHA-256 values are
  `3760fd2074244b1fb7d90066d3a0c3c2a3795595e0dc129145904576f268262b`
  and `87c0a37744edab36842293eb2f7878212d41a7a8bf8cca0f2227903e8c1f658a`.
  This is eligible within-run selector evidence, not terminal-valid evidence.
- B3 `1250970` remains healthy beyond step `7579/15410`; b4 `1253374`
  remains Priority-pending without a scheduler estimate. B2's rough terminal
  artifact ETA remains `06:00--06:30 SAST`. Operationally b2 and b3 are
  running and b4 is pending, all on A100-40GB; no A100-80GB or L40S work is
  owned. Scientifically AfriHG remains `5/11` terminal-valid and global freeze
  remains `0/8`. Quota is home `88.6%`, scratch `41.2%`; Kombuys remains
  read-only with RTX 5090 untouched, held-out access is `0`, Sheet E/F/G
  remain blank, and General/Monolingual/publication remain blocked.

## AfriHG b3 improves at checkpoint 6164; b2 enters callback — 01:14 SAST

- B3 job `1250970` completed its frozen step-6164 callback and resumed
  fault-free training beyond step 6420 on `srvrocgpu010` A100-40GB. Its exact
  artifact has `128` rows (`64/64` Xho/Zul), zero exact, whitespace, or
  debug-empty predictions, `64/64` unique predictions per language, and clean
  prompt boundaries. All prompts start `[BOS]`, end
  `[EOS]<|assistant|>`, and contain zero EOS tokens after the assistant
  marker.
- Xho/Zul chrF is `22.305296547403668/23.369240354846387`; registered mean
  chrF is `22.837268451125027`. This exactly matches `trainer_state.json`,
  improves b3's step-3082 score, and moves its validation-only retained
  checkpoint to 6164. Artifact and trainer-state SHA-256 values are
  `de331d569cce31014f80045689514c372c090d4def9dc77d8654051ff000f369`
  and `993bd7de2f85cfd1b737f4482abe81279bcee7cda8986521457824523f29ddb9`.
  This is eligible within-run selector evidence, not terminal-valid evidence.
- B2 `1248577` reached step `12328/15410`, completed all `3,082` declared
  validation rows at health-only loss `2.2725518763181376` in `140.0467 s`,
  and entered frozen exact generation at `01:04:49 SAST`. No step-12328 exact
  artifact exists yet, so checkpoint 9246 remains the audited within-run best.
  The callback artifact is tentatively expected around `02:10--02:25 SAST`;
  the rough terminal-artifact ETA is now `06:00--06:30 SAST`.
- B4 `1253374` remains Priority-pending without a scheduler estimate.
  Operationally b2 and b3 are running and b4 is pending, all on A100-40GB;
  no A100-80GB or L40S work is owned. Scientifically AfriHG remains `5/11`
  terminal-valid and global freeze remains `0/8`. Quota is home `88.6%`,
  scratch `41.2%`; Kombuys remains read-only with RTX 5090 untouched,
  held-out access is `0`, Sheet E/F/G remain blank, and
  General/Monolingual/publication remain blocked.

## AfriHG b0/b1 pass tensor verification; b4 queued — 00:17 SAST

- Exact read-only verifier retry `1252840` completed `0:0` at
  `23:51:09 SAST` on 21 August after four seconds. Its output SHA-256 is
  `98fd80db0522250e91ec7d42074c0aa2d3f59d56fcbf8a9c8c9f2a8857194eef`.
  The earlier quoting failure `1252793` remains preserved as operational
  failure provenance and produced no scientific result.
- B0 retained BIN and final safetensors each contain `424` keys and
  `69,438,784` tensor values, with zero missing keys and zero shape, dtype, or
  value mismatches. Their SHA-256 values are
  `43e5a923320415a7545f10d7d32054150a973513dfa473a90447e4a4614a8efb`
  and `36283250957cac594ddf20926a95e8110b11b12cffce4502827d87032265adcd`.
- B1 retained BIN and final safetensors each contain `424` keys and
  `76,410,112` tensor values, again with zero missing keys and zero shape,
  dtype, or value mismatches. Their SHA-256 values are
  `32c3378f8a1d7418e5f277e97274fdc82121f8e97d50c5074d84ae0a87ce36ef`
  and `0228762dff097e00bc7654cb9406bfb6d0a47ee3245eb281cb519132cbd3cb16`.
  B0 and b1 are therefore scientifically terminal-valid. AfriHG advances
  from `3/11` to `5/11`; the global family freeze remains `0/8`.
- B2 `1248577` remains fault-free beyond step `11487/15410`, retaining
  checkpoint 9246 at validation-only mean chrF `22.230781529182863`. Its rough
  terminal-artifact ETA is `04:40--05:10 SAST`. B3 `1250970` remains healthy
  in the frozen step-6164 exact-generation callback; no artifact exists yet,
  and its tentative artifact ETA remains `00:55--01:10 SAST`. These are
  operational estimates, not selection evidence.
- With exactly two active owned jobs, no b4 duplicate, an absent b4 output
  root, and registry SHA-256
  `8fdd6ea5f192a1d2ef2267353162b9a640ccd09ae0d889cb5a999645198bb726`
  verified, the next preregistered seed-42 candidate was submitted once as
  job `1253374`. B4 uses LR `8.981661817441004e-05`, rank/alpha `8/16`,
  dropout `0.0971165418624878`, and warmup `0.04982422709465027` from the
  frozen registry. It requests `nlpgroup/a100/nlpgroup`, one A100-40GB
  `gpu:ampere`, 24 hours, eight CPUs, and the canonical working directory.
  It is Priority-pending with provisional start `13:51 SAST`.
- Operationally, jobs `1248577/1250970` are running and `1253374` is pending;
  all are A100-40GB, with no owned A100-80GB or L40S work. Quota is home
  `88.6%`, scratch `41.2%`. Kombuys remains read-only with RTX 5090 untouched.
  Held-out access remains `0`, Sheet E/F/G remain blank, and
  General/Monolingual/publication remain blocked until the correction gates
  close.
