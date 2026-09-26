# Pure-GDN validation HPO — 2026-08-24

## B7 enters third exact callback; b6 healthy — 23:46 SAST

- B7 `1262543` reached exactly `9246/15410`, completed full declared validation at health-only loss `2.2039773533825437`, and entered its frozen step-9246 exact-generation callback. No step-9246 exact artifact exists yet, so no new b7 metric or checkpoint decision is available.
- Concurrent b6 `1262542` remains healthy near `11080/15410` at about `3.10 s/step`. Targeted fault scans are empty for both jobs. Gate `1258898` remains dependency-blocked on both, followed by comparator `1258899`.
- Quota is `88.6%/42.4%`; only A100-40GB is active, no A100-80GB or L40S overlap exists, Kombuys remains read-only, held-out access is `0`, Sheet E/F/G remain blank, AfriHG is `9/11` terminal-valid, and global freeze is `0/8`.

## HPO method remains frozen — 22:50 SAST

- User approved continuing the preregistered pure-GDN anchor-plus-Sobol validation HPO and explicitly declined a midstream switch to the historical xLSTM W&B Bayesian process. Promising GDN validation behavior is not treated as evidence that Sobol is universally superior.
- No candidate, search-space, checkpoint, prompt, retry, seed, or hardware decision changes. The paper-level claim should remain best observed performance under each architecture's documented validation-selected tuning process, with method and budget disclosed, rather than a strict equal-HPO-budget causal comparison.
- B6 `1262542` and b7 `1262543` remain healthy on two A100-40GB devices near `10025/15410` and `8986/15410`; gate `1258898` and comparator `1258899` remain dependency-blocked. Held-out access remains `0`, Sheet E/F/G remain blank, AfriHG remains `9/11` terminal-valid, and global freeze remains `0/8`.

## B6 step 9246 is clean and becomes retained — 22:46 SAST

- B6 `1262542` completed its frozen step-9246 exact-generation callback at `22:09:03 SAST` and resumed healthy A100-40GB training near `9948/15410`. Its immutable artifact has exactly 128 rows (`64/64` Xho/Zul), `64/64` unique predictions per language, zero empty predictions, correct `[BOS]` through `[EOS]<|assistant|>` boundaries, and zero generated EOS markers.
- Xho/Zul chrF is `21.828465231626993/22.32754911383624`; registered mean chrF is `22.078007172731617`, improving step 3082 mean `21.270276801059854`. Checkpoint 9246 therefore becomes retained using validation evidence only. Artifact/trainer-state SHA-256 values are `3c42fa22...3b9ef`/`efc063f9...534e5`.
- Concurrent b7 `1262543` remains healthy near `8900/15410`. Neither run has a targeted fault marker. Gate `1258898` remains dependency-blocked on both runs, followed by comparator `1258899`. Quota is `88.6%/42.4%`; only A100-40GB is active, no A100-80GB or L40S overlap exists, Kombuys remains read-only, held-out access is `0`, Sheet E/F/G remain blank, AfriHG is `9/11` terminal-valid, and global freeze is `0/8`.

## Live progress — 20:59 SAST

- B6 `1262542` reached `9246/15410` (`60.0%`) and entered its third frozen
  validation boundary; b7 `1262543` is healthy near `6877/15410` (`44.6%`).
  Together the two remaining seed-42 trials have completed `52.3%` of their
  optimizer steps. Targeted fault scans remain empty.
- AfriHG is scientifically terminal-valid at `9/11`; b6 and b7 are the last
  two seed-42 candidates, after which the dependency-gated A100-80GB hardware
  check `1258898`, CPU comparator `1258899`, and four preregistered
  confirmations remain before an AfriHG winner can be frozen.
- Quota is `88.6%/42.5%`; two A100-40GB GPUs are active with the third owned
  GPU slot occupied only by the dependency-pending gate. No A100-family or
  L40S overlap exists. Kombuys read-only, held-out `0`, Sheet E/F/G blank,
  and global freeze `0/8`.

## B7 step 6164 is clean and becomes retained — 20:45 SAST

- B7 `1262543` completed its frozen step-6164 exact-generation callback at
  `20:22:01 SAST` and resumed healthy A100-40GB training near
  `6603/15410`. The immutable artifact has exactly 128 rows (`64/64`
  Xho/Zul), `64/64` unique predictions per language, zero empty predictions,
  correct `[BOS]` through `[EOS]<|assistant|>` prompt boundaries, and zero
  EOS markers after the assistant marker or in generated output.
- Xho/Zul chrF is `23.435519836035333/25.083127573554826`; registered mean
  chrF is `24.25932370479508`, improving step 3082 mean
  `23.188471389318273`. Checkpoint 6164 therefore becomes retained using only
  frozen validation evidence. Artifact/trainer-state SHA-256 values are
  `d09e81b6...f10048`/`3630238e...c9667d4`.
- Concurrent b6 `1262542` remains healthy near `8996/15410`. Gate `1258898`
  remains dependency-blocked on both runs, followed by comparator `1258899`.
  Quota is `88.6%/42.5%`; only A100-40GB is active, no A100-80GB or L40S
  overlap, Kombuys read-only, held-out `0`, Sheet E/F/G blank, AfriHG `9/11`,
  and global freeze `0/8`.

## B7 enters second exact callback; b6 healthy — 19:42 SAST

- B7 `1262543` reached exactly `6164/15410`, completed full declared
  validation at health-only loss `2.1724466154282935`, and entered its frozen
  step-6164 exact-generation callback. The exact artifact is not yet present,
  so no new validation metric is available or used; it is tentatively due
  around `20:15--20:35 SAST`.
- Concurrent b6 `1262542` remains healthy near `7787/15410` at about
  `3.12 s/step`. Neither run has a traceback, OOM, CUDA, NCCL, killed,
  exception, or Slurm fault marker. Gate `1258898` remains dependency-blocked
  on both runs, followed by comparator `1258899`.
- Quota is `88.6%/42.5%`; only A100-40GB is active, no A100-80GB or L40S
  overlap, Kombuys read-only, held-out `0`, Sheet E/F/G blank, AfriHG `9/11`,
  and global freeze `0/8`.

## B6 step 6164 is clean; checkpoint 3082 remains retained — 18:42 SAST

- B6 `1262542` completed its frozen step-6164 exact-generation callback at
  `18:16:16 SAST` and resumed healthy A100-40GB training near
  `6654/15410`. The immutable artifact has exactly 128 rows (`64/64`
  Xho/Zul), `64/64` unique predictions per language, zero empty predictions,
  correct `[BOS]` through `[EOS]<|assistant|>` prompt boundaries, and zero
  EOS markers after the assistant marker or in generated output.
- Xho/Zul chrF is `20.629001232654016/21.55695707470458`; registered mean
  chrF is `21.0929791536793`, below step 3082 mean `21.270276801059854`.
  Checkpoint 3082 therefore remains retained without consulting held-out
  data. Artifact/trainer-state SHA-256 values are
  `83d4c9e8...09b53`/`36a56215...9cf46`.
- Concurrent b7 `1262543` is healthy near `5568/15410`. Gate `1258898`
  remains dependency-blocked on both runs, followed by comparator `1258899`.
  Quota is `88.6%/42.5%`; only A100-40GB is active, no A100-80GB or L40S
  overlap, Kombuys read-only, held-out `0`, Sheet E/F/G blank, AfriHG `9/11`,
  and global freeze `0/8`.

## B6 enters second exact callback; b7 healthy — 17:28 SAST

- B6 `1262542` reached exactly `6164/15410`, completed full declared
  validation at health-only loss `2.306739406102977`, and entered its frozen
  step-6164 exact-generation callback. The callback remained active through
  `17:24:44 SAST` with normal context-truncation warnings and no traceback,
  OOM, CUDA, NCCL, killed, exception, or Slurm fault marker. The exact
  `step-00006164.jsonl` artifact was still absent at `17:28`, so no new
  validation metric is available or used.
- Concurrent b7 `1262543` remains healthy near `4166/15410` at about
  `3.15 s/step`, without a runtime fault marker. A100-80GB gate `1258898`
  remains dependency-blocked on successful completion of both b6 and b7;
  CPU comparator `1258899` remains after it.
- Quota is home `88.6%`, scratch `42.4%`; only A100-40GB is active, with no
  A100-80GB or L40S overlap. Kombuys remains read-only, held-out access `0`,
  Sheet E/F/G blank, AfriHG scientifically trusted `9/11`, and global freeze
  `0/8`.

## B7 step 3082 is clean and retained — 16:40 SAST

- B7 `1262543` completed its first frozen exact-generation callback and
  resumed healthy A100-40GB training near `3250/15410`. Its immutable
  step-3082 artifact has exactly 128 rows (`64/64` Xho/Zul), `64/64` unique
  predictions per language, zero empty predictions, correct `[BOS]` through
  `[EOS]<|assistant|>` prompt boundaries, and zero EOS markers in generated
  output.
- Xho/Zul chrF is `22.39658842179247/23.98035435684408`; registered mean
  chrF is `23.188471389318273`, exactly matching retained checkpoint 3082.
  Artifact/trainer-state SHA-256 values are
  `c2d9a125...b58f2f`/`4c2be62b...f7d510`. This is eligible validation-only
  within-run evidence, not a terminal-valid candidate result.
- Concurrent b6 `1262542` is healthy near `5698/15410`; its next exact
  artifact remains tentatively due around `18:00--18:20`. Gate `1258898`
  remains blocked on both jobs and comparator `1258899` remains after it.
  Quota `88.6%/42.4%`; only A100-40GB active, no L40S, Kombuys read-only,
  held-out `0`, Sheet E/F/G blank, AfriHG `9/11`, global freeze `0/8`.

## B7 enters first exact callback; b6 continues — 15:37 SAST

- B7 `1262543` reached exactly `3082/15410`, completed full declared
  validation at health-only loss `2.223826493171033`, and entered its frozen
  exact-generation callback at `15:19:01 SAST`. The job remains active with
  no runtime fault marker and no step-3082 exact artifact yet; the artifact is
  tentatively due around `16:20--16:35 SAST`.
- B6 `1262542` remains healthy near `4497/15410` at about `3.15 s/step`, with
  its next step-6164 exact artifact tentatively due around `18:00--18:20`.
  Gate `1258898` remains dependency-blocked on both jobs and comparator
  `1258899` remains after the gate.
- Quota is `88.6%/42.4%`; only two A100-40GB jobs are active, no A100-80GB or
  L40S overlap, Kombuys read-only, held-out `0`, Sheet E/F/G blank, AfriHG
  `9/11`, and global freeze `0/8`.

## B6 step 3082 is clean and retained — 14:36 SAST

- B6 `1262542` completed its first frozen exact-generation callback and
  resumed healthy A100-40GB training near `3342/15410`. The immutable
  step-3082 artifact has exactly 128 rows (`64/64` Xho/Zul), `64/64` unique
  predictions per language, zero empty predictions, correct `[BOS]` through
  `[EOS]<|assistant|>` prompt boundaries, and zero EOS markers in generated
  output.
- Xho/Zul chrF is `21.216082571156228/21.32447103096348`; registered mean
  chrF is `21.270276801059854`, exactly matching the retained checkpoint-3082
  trainer state. Artifact/trainer-state SHA-256 values are
  `6d5cf744...4ef0e5` and `df2b9fc4...134871`. This is eligible validation-
  only within-run evidence, not a terminal-valid candidate result.
- Concurrent b7 `1262543` remains healthy near `2291/15410`. Gate `1258898`
  remains dependency-blocked on both jobs and comparator `1258899` remains
  after it. Quota is `88.6%/42.4%`; only A100-40GB is active, no L40S,
  Kombuys read-only, held-out `0`, Sheet E/F/G blank, AfriHG `9/11`, freeze
  `0/8`. B7's first exact artifact remains tentatively due around
  `16:15--16:45 SAST`.

## Concurrent b6/b7 healthy — 13:36 SAST

- B6 `1262542` reached exactly `3082/15410`, completed full declared
  validation at health-only loss `2.4353304596244323`, and entered its frozen
  exact-generation callback at `13:18:39 SAST`. It remains active without a
  fault marker; no step-3082 exact artifact exists yet, so no metric decision
  is available or made. The artifact is tentatively due around
  `14:10--14:30 SAST`.
- B7 `1262543` remains healthy near `1109/15410` at about `3.11 s/step`, with
  its first exact artifact tentatively due around `16:15--16:45 SAST`.
  Gate `1258898` remains dependency-blocked on both runs and comparator
  `1258899` remains after the gate.
- Operationally two A100-40GB jobs run concurrently; scientifically AfriHG
  remains `9/11` and global freeze `0/8`. Quota is `88.6%/42.3%`; there is no
  A100-80GB or L40S overlap, Kombuys remains read-only, held-out access is
  `0`, and Sheet E/F/G remain blank.

## B6 and b7 now use available same-family capacity — 12:36 SAST

- A prospective execution-only amendment was frozen at SHA-256
  `5c4490b9...25c7af` before dependency edits. B7 output/logging roots were
  absent, only two of four A100-40GB devices were allocated, and no b6 metric
  or held-out result was inspected.
- B6 `1262542` remains healthy near `2266/15410`. B7 `1262543` started at
  `12:35:59 SAST` on a second `srvrocgpu010` A100-40GB, wrote execution-
  manifest SHA-256 `ab0a2d96...c45c79`, and verified all `694/694` immutable
  files. It then passed fast-GDN and loaded the canonical pure-GDN model with
  the exact b7 trainable parameter count `9,296,640`. This removes an
  otherwise artificial same-family serial delay.
- A100-80GB gate `1258898` now depends on successful b6 and b7 completion;
  CPU comparator `1258899` remains after it. Thus the gate cannot overlap
  either A100-40GB run. Owned GPU jobs remain exactly three, no L40S is used,
  quota is `88.6%/42.2%`, Kombuys remains read-only, Sheet E/F/G blank,
  held-out access `0`, AfriHG scientifically trusted `9/11`, and global
  freeze `0/8`.

## B6 healthy — 11:34 SAST

- Corrected AfriHG b6 `1262542` is healthy on A100-40GB `srvrocgpu010` after
  `00:55:52`, advancing near `1053/15410` at about `2.97--3.00 s/step` with
  no traceback, OOM, CUDA, NCCL, killed, or Slurm fault marker. The first
  frozen step-3082 exact artifact is tentatively due around `14:00 SAST`;
  terminal verification remains roughly early `25 August`.
- B7 `1262543`, A100-80GB gate `1258898`, and CPU comparator `1258899`
  remain dependency-pending in exact strict order. Only A100-40GB is active;
  there is no A100-80GB or L40S overlap. Quota is home `88.6%`, scratch
  `42.2%`; Kombuys remains read-only, Sheet E/F/G blank, held-out access `0`,
  AfriHG scientifically trusted `9/11`, and global freeze `0/8`.

## B6 launcher failed before science; corrected b6/b7 chain queued — 10:40 SAST

- B6 job `1258900` failed `1:0` in zero seconds at `10:12:03 SAST` because
  its stale launcher referenced absent mutable-repository file
  `scripts/hpo_protocol.py`. Its only output is a 157-byte log with SHA-256
  `76ca9c0c...de94f`; model, data, evaluator, and candidate metrics were never
  accessed, and both candidate output/logging roots remained absent.
- A prospective correction was frozen at SHA-256 `e5c08c40...4ed63b` before
  submission. The immutable snapshot sidecar and all `694/694` source hashes
  passed, registry SHA-256 remained `8fdd6ea5...bb726`, and the b6/b7 roots
  were absent. No b6/b7 or held-out metric informed the correction.
- Corrected unchanged b6 `1262542` started at `10:37:56 SAST` on A100-40GB
  `srvrocgpu010`, verified all `694/694` immutable files, wrote execution-
  manifest SHA-256 `94a1587e...a3bb6b`, passed fast-GDN, and loaded the
  canonical model cleanly. Unchanged b7 `1262543` is dependency-pending after
  b6; gate `1258898` is dependency-pending after b7, then CPU comparator
  `1258899`. This exact serial order uses all three allowed owned GPU slots
  without A100-family overlap or a manual b6-to-b7 gap.
- Scientifically trusted progress is unchanged: AfriHG `9/11`, global freeze
  `0/8`, held-out access `0`, Sheet E/F/G blank. Quota is home `88.6%`,
  scratch `42.0%`; no L40S is used and Kombuys remains read-only.

## B6 released ahead of capacity-only gate — 09:37 SAST

- Slurm moved A100-80GB gate candidate `1258898` from an estimated
  `2026-08-25 21:47:49` start to `2026-09-01 17:31:43 SAST`. A prospective
  execution-only amendment was frozen before dependency edits at SHA-256
  `e8dbf98e...5c83d`; it uses no held-out or candidate metric.
- Existing unchanged A100-40GB b6 job `1258900` is now eligible and
  `Resources`-pending with an absent output root. Gate candidate `1258898` is
  safely dependency-pending after b6, and comparator `1258899` remains after
  the gate. This preserves zero A100-40GB/A100-80GB overlap while removing an
  avoidable capacity-only delay.
- Slurm rejected the first attempt to point b6 at already-complete b5; the
  gate was held, no dependency changed, and no job started. The safe recovery
  cleared b6's dependency, installed `afterok:1258900` on the held gate, then
  released it. All approved A100-40GB devices remain occupied by other users,
  so b6 has no current scheduler start estimate. Quota is home `88.6%`,
  scratch `42.0%`; no L40S is used, Kombuys remains read-only, held-out
  access is `0`, global freeze is `0/8`, and Sheet E/F/G remain blank.

## B5 is scientifically terminal-valid; AfriHG advances to 9/11 — 05:32 SAST

- Isolated A100-40GB b5 job `1258485` completed `0:0` at `04:43:25` after
  `19:02:40`. Its terminal exact artifact has exactly 128 rows (`64/64`
  Xho/Zul), zero empty predictions, `64/64` unique predictions per language,
  clean `[BOS]` through `[EOS]<|assistant|>` prompt boundaries, and zero EOS
  tokens after the assistant marker.
- Terminal Xho/Zul chrF is `23.320999482462366/23.993243166470638`; mean
  chrF is `23.657121324466502`. This does not improve the validation-only
  retained checkpoint-12328 mean `23.750084550710223`. Terminal artifact,
  retained trainer-state, retained BIN, final safetensors, and byte-identical
  retained/final config SHA-256 values are
  `880bf0b4...3b5ba`/`d45e3874...0a8b16`/`2ccaf25d...850a5`/
  `2b69ab43...15a7b`/`3a461e2b...18474`.
- CPU verifier `1261967` completed `0:0` in six seconds and proved exact
  equality across all 424 retained/final tensor keys and 71,762,560 values,
  with no missing, shape, dtype, or value mismatch. Verifier-script/log
  SHA-256 values are `0faf2c44...28133` and `28eefe65...d133`. B5 is therefore
  scientifically terminal-valid and AfriHG advances from `8/11` to `9/11`.
- The next strict chain is A100-80GB gate candidate `1258898`, CPU comparator
  `1258899`, then A100-40GB b6 `1258900`. Candidate `1258898` is
  `AssocGrpGRES`-pending with Slurm estimate `2026-08-25 21:47:49 SAST`:
  all four A100-80GB devices are occupied by other-user jobs. All four
  approved A100-40GB devices are also occupied by other-user jobs, while four
  idle `gpu:amperemk` devices remain inaccessible to the current association.
  No approved capacity is being left idle and no job was altered. Quota is
  home `88.6%`, scratch `42.0%`; no L40S is used, Kombuys remains read-only,
  held-out access is `0`, global freeze is `0/8`, and Sheet E/F/G remain
  blank.

## B5 training complete; terminal exact callback active — 04:30 SAST

- Isolated AfriHG b5 `1258485` reached exactly `15410/15410`, completed the
  full declared terminal validation at health-only loss
  `2.2310394148483126`, and entered the frozen terminal exact-generation
  callback at `03:36:52` SAST. The job remains healthy on A100-40GB
  `srvrocgpu010`; its log was active through `04:15:27` with no traceback,
  OOM, CUDA, NCCL, Slurm fault, or killed marker.
- No terminal `step-00015410.jsonl` artifact exists yet, so b5 remains
  non-terminal and AfriHG remains `8/11` terminal-valid. Jobs
  `1258898 -> 1258899 -> 1258900` remain strictly dependency-pending. Quota
  is home `88.6%`, scratch `41.9%`; no L40S is used, Kombuys remains
  read-only, held-out access is `0`, global freeze is `0/8`, and Sheet E/F/G
  remain blank.

## B5 step 12328 is clean and becomes within-run best — 01:26 SAST

- Isolated AfriHG b5 `1258485` completed its frozen step-12328 exact callback
  and resumed healthy A100-40GB training beyond step 12929/15410 at about
  3.08 seconds/step. The immutable artifact has exactly 128 rows (`64/64`
  Xho/Zul), zero empty predictions, `64/64` unique predictions per language,
  and clean `[BOS]` through `[EOS]<|assistant|>` boundaries.
- Xho/Zul chrF is `23.192972608788125/24.307196492632322`; registered mean
  chrF is `23.750084550710223`, improving step 9246 and matching the retained
  checkpoint-12328 trainer state. Artifact SHA-256 is
  `ca46edf371d8862a6fa0d3e70fd5c466134e52ec0130c8f86b513852777c2bde`;
  trainer-state SHA-256 is
  `d45e3874c1e9a360879a3274a7c6ce73d365dfb716d2b10c6be239fd3b0a8b16`.
- This is validation-only within-run evidence, not terminal evidence. AfriHG
  remains `8/11` terminal-valid and global freeze remains `0/8`. Terminal
  verification is tentatively due around `04:30--05:15 SAST`; jobs
  `1258898 -> 1258899 -> 1258900` remain dependency-serial behind b5. Quota
  is home `88.6%`, scratch `41.9%`; no L40S is used, Kombuys remains
  read-only, held-out access is `0`, and Sheet E/F/G remain blank.
