# SALLM Progress

- 13 September 10:37 SAST: the user authorized moving the three missing Base
  POS scores to Kombuys after all HEX A100/L40S nodes were drained with no
  start estimate. HEX job `1334564` is held `JobHeldUser` before payload with
  no claim or output. The exact checkpoint hashes, 3.3 MB source snapshot and
  157 MB offline cache were transferred; the resolved config matches after
  host-path substitution only. An isolated `datasets==4.8.5` overlay aligns
  the sole evaluation-package mismatch. Binding `a0f0be1b...562e` covers
  1,970 files. The 12-task/7,216-row run is active in tmux
  `gdn-base-pos-completeness` on Kombuys GPU1, RTX 3080 Ti, with metrics
  unopened. This is a disclosed cross-host post-hoc completeness amendment.

- 10 September 18:52 SAST: canonical POS `1331984` failed before model
  loading or output creation because its Hydra overrides targeted
  `evaluation...` instead of the config-group path `eval.evaluation...`.
  CPU seal `1332060` verified the two-key correction and exact equality to
  the prior resolved config apart from batch1/max1; binding
  `73db8b0c...a175`. Isolated replacement `1332061` is running on
  `gpu:ampere` and reached all `7,216` generation requests. AfriHG Zulu
  `1331985` completed `0:0` in 34 minutes; its sidecar verifies all `1,776`
  rows with `metric_values_included=false`, taking General to 18/19. No
  metric value was opened.

- 10 September16:37 SAST: canonical General recovery bundle sealed in CPU
  job `1331982` with binding `8e10d466...f230`. POS `1331984` (batch1/max1,
  48h) and AfriHG Zulu `1331985` (cache-only, 4h) were submitted once on
  `gpu:ampere`. Both are pending; Slurm projects POS start at 01:00 on
  11 September. Prior exact throughput implies about 26–32h after start,
  placing likely full structural completion on 12 September morning.

- 10 September16:29 SAST: General validation gate `1331858` failed closed.
  Batch 8 was about 4x faster on 12 synthetic rows but changed raw responses
  on four rows and filtered responses on two; matching aggregate synthetic
  scores do not establish equivalence. Batch 8 is rejected. The prospective
  fallback restores exact original batch1/max1 semantics with a 48-hour
  walltime; AfriHG Zulu remains cache-only. No official recovery has run.

- 10 September14:55 SAST: validation-only General batch gate `1327451`
  failed in one second before model/data loading because its CPU-created
  manifest required Ada-node management packages on the A100 compute node.
  No output or runtime directory exists. A versioned runtime-v2 wrapper now
  separates immutable source/artifact hashes from an exact A100-created
  runtime manifest; replacement gate `1331858` was submitted once and is
  resource-pending. Official POS/AfriHG Zulu remain unstarted.

- 8 September11:34 SAST: General14/19 units have structural verification.
  g0 completed0:0 in3:01:29 and g1 remainder1321101 completed0:0 in21:11.
  Only g2/POS remains running (615/7216), followed by its four unstarted
  units. POS24h allocation risk remains; administrator extension requested.
  Base14-lane preparation remains outstanding; no Sheet update.

- 8 September11:07 SAST: General10/19 units structurally verified. AfriMGSM
  failed only the original verifier's one-record-per-document assumption;
  CPU1321090 verified all5000documents under both frozen filters with corrected
  versioned verifier. No inference repeated. g1 remainder1321101 submitted
  for never-started BelebeleEng/Tso/AfriHGXho only, A10080/48hours.
  g0/g2 preserved; POS extension risk and Base14-lane preparation outstanding.
  See notes/2026-09-08-general-filter-verifier-correction.md.

- 8 September10:33 SAST: General9/19 execution units structurally verified.
  Newly completed NER15tasks/14980rows, Intent20/12710, BelebeleAfr5/4500
  and BelebeleTsn5/4500. g0 now T2X, g1 AfriMGSM, g2 POS409/7216;
  g3 completed0:0. Scores withheld until whole General matrix verifies.
  POS walltime risk and Base correction outstanding; no rerun or Sheet edit.

- 8 September 09:33 SAST: General g3 completed0:0; five execution packs
  are structurally verified (News, SIB, AfriXNLI, BelebeleSsw, BelebeleZul).
  g0/g1/g2 remain healthy and running. POS walltime risk remains:207/7216
  requests with approximately37hours remaining; requested administrator
  extension after in-place48hour change was denied. No restart or metric
  release. Base14-lane amendment re-read; preparation remains outstanding.

- General official evaluation is RUNNING on all four A10080 cards:
  g0 1319956, g1 1319958, g2 1319959, g3 1319960. Preparation is complete;
  actual-loader and seal jobs passed and all launch bindings verified.
  Four fixed groups cover19 execution units/full16 logical lanes without
  pending-array gaps. No duplicate or failed official arm may be restarted.
  See notes/2026-09-08-general-official-execution.md. Results and Base14-lane
  correction remain to finish; no Sheet update yet.

- General offline coverage is complete: data audit1319869 and actual-loader
  preflight1319896 passed. Nineteen execution units cover the full16 logical
  lanes. Seal job1319932 is running; four fixed A10080 execution groups launch
  only after READY. No official arm has run yet. Base14-lane correction remains
  required. See notes/2026-09-08-general-official-execution.md.

- The user approved routine execution repairs without further permission.
  General exporter now preserves YAML function tag/value explicitly and
  rejects unknown objects. CPU audit1319707 is submitted once with a built-in
  regression check and new v2 paths; failed1318956 is preserved. Continuation
  is ACTIVE again. Official test and scientific acceptance rules are unchanged.
  See notes/2026-09-08-general-audit-export-correction.md.

- General audit1318956 FAILED1:0 during JSON export: lm-eval simple YAML
  resolution yields a ScalarNode for !function. No data/model/scoring ran.
  Failed artifacts are preserved; the partial contract is invalid. Under the
  no-failed-gate-rerun rule, correction and a new metric-free audit await user
  approval. General tests remain unstarted. See the official-preflight note.

- General task-definition CPU audit1318956 is running. It inventories the
  frozen187 prompt tasks and3 generation tasks without loading data or a
  model. Preserve the first artifact and await terminal verification; this
  is not an official test. See notes/2026-09-08-general-official-preflight.md.

- General configuration-only CPU preflight1318551 completed0:0 on8September.
  The frozen base and B7 adapter resolve with merging disabled and the full
  prescribed matrix. No General test ran; result root remains absent. Offline
  data coverage and per-lane structural verification are the next gates. See
  notes/2026-09-08-general-official-preflight.md. Configuration stdout hash
  40ed7aab...936c73; no workbook or training changes.

- On 8 September, verifier 1317663 and ranking 1318546 completed 0:0.
  All four General confirmations are terminal-verified. General freezes to
  B7 seed42 checkpoint5456: three-seed validation NLL 0.9156026 (sample SD
  0.0019475), versus A2 0.9252626. Ranking hash 1d51c1e3...076cdf;
  exact adapter hashes and recipe are in
  notes/2026-09-08-general-confirmation-ranking.md. General official tests
  remain unrun pending their frozen execution bundle and metric-free preflight.
  No Sheet change. Base 14-lane versus16/16 provenance remains unresolved.

- On 7 September at 22:21 SAST, all four General confirmations were
  COMPLETED 0:0 after approximately 21h40m, ending at step10912 with
  retained checkpoint5456. All four step10912 sidecars and exact coverage
  passed. CPU roundtrip verifier 1317663 was submitted once for all four.
  General is not yet frozen; require verifier success and frozen ranking
  before official testing. See notes/2026-09-07-general-confirmations-running.md.

- General Stage-B is 8/8 verified: b3 1312957 completed 0:0; verifier
  1314418 proved exact retained-to-final equality. Frozen all-eleven ranking
  1314420 selected b7 and a2, artifact SHA-256
  `1773058c448a619c3046fec56ae80347e152d95d65585e3af05df9870b54d87a`.
  Four confirmations started on A100-80GB at 00:26 SAST on 7 September:
  b7 seed13 1314428, a2 seed13 1314429, b7 seed87 1314430, a2 seed87 1314431.
  All scheduler limits verify as 36 hours. See
  `notes/2026-09-07-general-confirmations-running.md`. No duplicate, General
  official test or Sheet change; freeze only after all confirmations verify.

- General Stage-B is now 7/8 terminal-verified. B5 1312084 and b6 1312231
  completed 0:0; all scheduled coverage/sidecar and execution-manifest checks
  passed. CPU verifier 1313549 completed 0:0 and proved exact retained-to-final
  equality at checkpoints 8184 and 13640. B3 1312957 remains running, last
  seen at 12780/13640. No ranking, confirmation or General test is eligible
  until b3 verifies. See `notes/2026-09-06-general-b56-terminal-verification.md`.
  No held-out evaluation or Sheet change occurred.

- B3 infrastructure recovery `1312957` was submitted once after preflight
  `1312488` completed 0:0 and its full check output, log hash, immutable
  wrapper/amendment and fresh no-duplicate/absent-final/prior-resume checks
  verified. This consumes the single disclosed 6 September exception;
  do not submit another replacement. B5/b6 remain running on A100-80GB.
  See `notes/2026-09-06-general-b3-launch.md`; trainer resume is verified
  directly from checkpoint 10912 to step 10913 and through 10925.
  No held-out evaluation or Sheet change occurred.

- The user authorizes availability-aware A100-40GB/80GB switching when
  viable, optimizing net finish time while retaining one-family-at-a-time
  and scientific gates. Immediate switching would discard roughly two hours
  each of b5/b6 progress beyond checkpoint 10912, while an 80GB card is free
  for b3. No job was interrupted. See
  `notes/2026-09-06-gpu-availability-switch-policy.md` for scheduling authority
  and target-hardware checks. B3 preflight 1312488 has now completed 0:0.

- B3 no-launch offline preflight `1312488` is submitted once and pending
  on `AssocGrpGRES`: b5/b6 and two other-user jobs occupy the four-card
  association. Both running continuations are healthy. B3's disclosed
  infrastructure amendment is frozen at SHA-256
  `2f12c3cf67b37eb5d1bf8a87c0af4170ad696b33dcf3807da452c87b4c112c5b`.
  Its failed recovery provenance and unchanged complete state are archived.
  No scientific b3 replacement has been submitted; require preflight 0:0
  and fresh submission checks first. See `notes/2026-09-06-general-b3-preflight.md`.

- On 6 September, b6's complete checkpoint-10912 state matched all five
  hashes recorded on 31 August. Its original root is preserved read-only
  in archive `14ab572d07184607ec3d4281a30424c241019259e747e30fa96fbb52351b38d5`.
  No-launch preflight `1312227` passed 0:0, including exact runtime and all
  twelve frozen train/validation digests. Continuation `1312231` was
  submitted once and is RUNNING alongside b5 `1312084` on A100-80GB.
  B6 trainer continuation is proved: it jumped to step 10913 and advanced
  through 10920/13640; b5 reached 11063/13640. The prospective
  amendment is `notes/2026-09-06-general-b6-offline-continuation-amendment.md`,
  SHA-256 `a119efbad4b90d9d60c13846095cb483f815acf8f5ef5df99afada251deeedc6`.
  The half-hour monitor now tracks both jobs and the remaining b3 boundary.

- On 6 September, the cache-only General b5 preflight `1312049` passed 0:0.
  It verified unchanged checkpoint/archive/runtime, exact AfriHG records and
  all six raw validation component counts offline. The isolated v2 amendment
  is frozen at SHA-256 `4a435cce1cb717612750a59b98e75aae9a4165f5e3cf6cd8e6e50c9ee515c1f9`.
  User-authorized continuation `1312084` was submitted once and is RUNNING
  on A100-80GB `srvrocgpu011`. Trainer-level resume is verified: the loop
  advanced directly to step 10913/13640 after loading checkpoint 10912.
  Preflight-only failure `1312019` and its first snapshot are preserved.
  General Stage-B is 5/8 terminal-verified; b3/b5/b6 remain. No held-out
  evaluation or Sheet edit was performed. See
  `notes/2026-09-06-general-b5-offline-continuation-amendment.md`.

- On 6 September, CPU verifier `1311939` completed 0:0 and proved exact
  retained-to-final equality for General b1/b2/b4 at steps 8184/13640/8184.
  All scheduled artifact sidecars and exact six-family 22,167-row coverage
  checks passed. B5 no-launch preflight `1311940` also completed 0:0,
  verifying its frozen source/runtime/archive/checkpoint bindings. Training
  remains unlaunched because the frozen AfriHG loader still contacts GitHub
  before checking cached files; offline-data preflight is not yet satisfied.
  Details are in `notes/2026-09-06-pure-gdn-general-verification.md`.

- The 6 September user-authorized VPN switch restored the normal relay route:
  Cisco is disconnected, Tailscale is connected, and `ssh hex` returned
  purequota and `srvrochpc001` with exit 0 after one Tailscale reconnect.
  The restored relay now works end to end without Cisco. Its tmux process
  still has no reboot-autostart configuration. No experiment was launched
  during this connectivity verification.

- On 6 September, Cisco login restored direct SSH to HEX and Kombuys. The
  user-owned Kombuys relay daemon was absent after a host reboot and was
  restarted with its existing identity, state, and original arguments in
  tmux `sallm-hex-relay`. It reached Running with its existing port-2222
  forwarding intact. End-to-end relay SSH is not yet verified because the
  Mac's Tailscale is offline while Cisco is connected; `hex-direct` works.
  The owned HEX queue is empty. Accounting confirms b1/b2/b4 continuations
  completed 0:0 and b3 failed 1:0. No new research job was submitted. See
  `/Users/anrilombard/Desktop/Masters/Notes/sallm_relay_restoration_2026-09-06.md`.

- The 4 September SSH diagnosis found the Mac's Tailscale connected but
  `kombuys-hex-relay` offline for about 15 hours. SSH stops at that peer's
  TCP port 2222 before reaching HEX; both direct campus SSH addresses also
  time out. General launch is blocked by relay/campus connectivity, not a
  scheduler queue. Restore the relay host/service or a working campus route.

- On 4 September, the user authorized General resumption. All three configured
  HEX access routes timed out, so no new job was submitted. The next steps and
  b3 infrastructure-recovery boundary are recorded in
  `notes/2026-09-04-pure-gdn-general-resumption.md`. Live Sheet coverage is
  Mono 21/21 and Multi 20/20, with 41 populated Base cells and General blank.
  Base's later 16/16 acceptance claim conflicts with the 30 August 14-lane
  correction requirement and needs artifact reconciliation before paper use.

- On 3 September at the 19:08 UTC follow-up, quota-first HEX readback found
  no owned running or queued jobs. The half-hour `gdn-hpo-workstream-monitor`
  automation was deleted because all authorized non-General test work and
  Sheet promotion are complete, while General remains paused. This is not
  completion of the entire program. General requires an explicit protocol
  decision before further execution; a monitor can be recreated when work resumes.

- On 3 September at the 18:38 UTC follow-up, all 28 POS/AfriHG result files
  listed in the structural verifiers passed direct hash checks. Four AfriHG
  Sheet notes were corrected to preserve the failed `1291770` versus
  successful continuation `1291874` history. Exact readback confirms scores,
  formatting, and General blanks are unchanged; no inference was rerun.

- On 3 September, POS recovery `1291776` and AfriHG continuation `1291874`
  completed `0:0`. All ten official-test arms have exact structural coverage,
  verified artifact and manifest sidecars, and family tree hashes
  `cc26ce67...13b4` (POS) and `323bdbab...5feb` (AfriHG). Their verified
  Mono/Multi test headlines are now in the canonical Sheet, completing all
  planned non-General GDN test rows.

- The same readback exposed a pre-existing chart-source defect: all 160 GDN
  formulas in `Language Charts` referenced Qwen columns `P:S`. They now
  reference GDN columns `T:W`; exact formula/value readback found zero
  remaining mixed sources or broken references. General stays blank.

- On 3 September, 154 `Not applicable: no task-specific adapter` placeholders
  were cleared from the Transformer, Mamba, xLSTM, and Qwen result tabs. GDN
  was already blank. Exact workbook-wide readback found no remaining
  occurrence or broken reference; scores were unchanged.

- On 3 September, the nonexistent AfriHG English placeholder row was deleted
  from the Transformer, xLSTM, Qwen, GDN, and hidden GDN-template result tabs.
  Mamba and all comparison/chart tabs already used only AfriHG Xhosa and Zulu.
  Exact post-delete readback found no broken references and preserved the
  AfriHG variant chart plus its two language charts.

- On 3 September, 354 result cells across the Transformer, Mamba, xLSTM, Qwen,
  and GDN Sheet tabs were simplified to show only the reported scores. Prompt
  summaries, means, and ranges remain in provenance notes where applicable;
  no metric or Variant Chart value changed.

- On 3 September, the obsolete `Architecture Comparison` tab was removed from
  the canonical Sheet after confirming no surviving formula depended on it.
  Exact post-delete readback preserved all 11 `Variant Charts`; their GDN
  series still use only columns T:W, with the latest verified News, NER, and
  T2X results. POS and AfriHG remain blank while `1291776/1291874` run.

- A 3 September post-hoc interpretation audit of the frozen News and NER
  official artifacts found that their Multi-over-Mono gaps are prompt-stable
  and survive paired document bootstraps. News weakness is a repeatable partial
  label-space collapse (especially zero sports and near-zero technology recall),
  while NER Multi improves every entity type and reduces both false positives
  and false negatives across all three languages. These are not single-prompt
  flukes, but Multi also receives concatenated multilingual data and more total
  updates, so the gap does not isolate architecture-level multilingual transfer.
  The News official metric is support-weighted F1, not macro-F1. Live Sheet
  readback confirms News/NER/T2X in the separate GDN columns and GDN General
  plus current POS Mono/Multi still blank.

- On 3 September, the independently terminal-valid News, NER, and T2X
  official held-out recovery results were promoted to the canonical Sheet
  rather than waiting for POS and AfriHG. Exact readback verified `GDN
  Results!C2:F6` and `C41:F41` plus propagation to `Comparison Data` and
  `Variant Comparison`; Qwen and GDN remain separate. Sheet coverage is now
  Mono `16/21` and Multi `15/20`. POS and AfriHG remain blank pending their
  own terminal verification.

- At 11:34 SAST on 3 September, News recovery `1291841` completed `0:0` and
  all four Mono/Multi English/Xhosa bundles passed exact row, task, sidecar,
  source-equivalence, and result-tree hash verification. Five-prompt held-out
  mean F1 is Multi `0.3841577509141925/0.43592257295697856` and Mono
  `0.2591946070862873/0.33770995918666835` for English/Xhosa. The sealed
  artifact-tree manifest SHA-256 is `1b34d001...bf5f`.

- AfriHG `1291770` completed its first Multi Xhosa inference but stopped in
  structural verification because `str.splitlines()` treated a valid U+2028
  inside one JSON prediction as a JSONL boundary. The partial arm remains
  unreportable. Focused regression and actual-artifact verification prove the
  one-line file-iteration correction. Immutable overlay manifest SHA-256 is
  `6073a94f...9630`; continuation `1291874` reverified all `1,305` existing
  rows without rerunning them and is evaluating only the three absent arms on
  A100-80GB. POS `1291776` remains healthy. Sheet E/F/G is unchanged until
  both remaining families complete and verify.

- At 10:50 SAST on 3 September, the disclosed missing-family held-out
  recovery produced terminal-valid T2X and NER results. T2X job `1291767`
  completed `0:0` over exact `378` Xhosa rows with chrF
  `51.73983564531445`; its sealed artifact-tree manifest SHA-256 is
  `4d008493...67536`. NER job `1291768` completed `0:0` for all six
  Mono/Multi arms with exact `4,980/5,000/5,000` five-prompt rows per mode.
  Its sealed tree SHA-256 is `d1b2808b...5e253`; a separate source
  equivalence artifact (`046826b1...758f6`) proves all 32 selected task files
  match the immutable snapshot exactly. Five-prompt mean F1 is Multi
  `0.78012/0.70086/0.73668` and Mono `0.67681/0.52601/0.49877` for
  Tswana/Xhosa/Zulu. These are official recovery test results, not validation
  values and not successful-original-job claims.

- POS first recovery `1291769` failed before dataset loading or any result
  because the offline cache held the expected data only under pinned-revision
  URL keys while the frozen task requests `main` URL keys. Prospective
  amendment `971aaf96...d427` requires byte-exact `main` versus pinned input
  equality. Corrected cache build/seal `1291774/1291775` completed `0:0`, and
  replacement POS `1291776` is healthy on A100-80GB. AfriHG `1291770` is
  healthy; News `1291841` is queued for the next association card. Sheet
  E/F/G remains unchanged pending complete recovery verification.

- At 17:21 SAST on 1 September, NER Zulu Mono `1282231` completed `0:0`
  after retaining checkpoint `1629` at validation-only F1
  `0.5095029239765582`. CPU verifier `1283371` proved exact
  retained-to-final equality across 424 keys and 71,762,560 values, so all
  three NER Mono adapters are terminal-valid and frozen before official
  testing.

- The released A100-80GB card was filled by AfriHG Xhosa Mono `1283372`
  using the frozen AfriHG winner LR `0.0001223079850011719` and fixed Mono
  LoRA recipe. Its canonical execution-manifest SHA-256 is
  `0e3e4ce929636d6a5317f2c3027adb9cb69f3b6dbc4187304b4fcbfd913f731a`;
  it verified all 715 immutable files, loaded exact `10,440/1,305`
  train/validation rows, and entered the fixed 6,525-step training loop on
  A100-80GB.
  A login-node dry-run had written only a manifest before failing on absent
  `nvidia-smi`; that root is quarantined and excluded. POS Tsn/Xho/Zul and
  AfriHG Xho now use all four cards. The official NER family bundle is next
  at the next eligible release; AfriHG Zulu follows. Held-out and Sheet E/F/G
  state are unchanged.
- SIB a0 seed-13 confirmation `1286171` is submitted once as the next frozen
  candidate-major entry after News a0. Exact absence, duplicate, registry,
  account/QoS/GRES, and four-active-owned checks passed. It is
  `AssocGrpGRES`-pending; the queue now contains three running plus one pending
  owned A100-80GB job. Held-out and Sheet state are unchanged.

- At 16:17 SAST on 1 September, NER Xhosa Mono `1282017` completed `0:0`
  after the fixed early-stop at retained checkpoint `1991`, with
  validation-only F1 `0.53664556282858`. CPU verifier `1283303` proved exact
  retained-to-final equality across 424 keys and 71,762,560 values. NER
  Tswana and Xhosa are now terminal-valid; NER Zulu `1282231` remains
  healthy.

- The released A100-80GB card was filled immediately by first-run POS Zulu
  Mono `1283304` after absent-output, no-duplicate, immutable-snapshot,
  unchanged-recipe, and active-cap checks. Its execution-manifest SHA-256 is
  `b31ac58a1505686a5e4b658ed4e7f42efffa26f68c8d8bfe26a7ab4dc761e200`.
  It verified the immutable snapshot, loaded exact `753/750` train/eval rows,
  and entered the fixed 1,425-step training loop. POS Tswana/Xhosa/Zulu and
  NER Zulu now use all four A100-80GB cards. No official held-out result or
  Sheet E/F/G cell changed.

- At 15:22 SAST on 1 September, all nine reduced News/SIB/Intent seed-42
  candidates were terminal-valid and exact validation-only rankings froze the
  seed-13 finalists: News a0/a1, SIB a0/a1, and Intent a2/a1. Ranking SHA-256
  values are `ffb124a7...ca048`, `c9ef792c...7a49e`, and
  `b63f78be...3b36`. NER Tswana Mono job `1282016` retained checkpoint 2172
  at validation-only F1 `0.5834360027378009`; verifier `1282784` proved exact
  equality across 424 keys and 71,762,560 values.

- Initial POS Tswana Mono job `1282672` failed after its first complete
  750-row validation because the strict evaluator hardcoded the Multilingual
  three-language, four-prompt grid. It produced no checkpoint, final adapter,
  or selection artifact. Prospective correction
  `9a9336bb...584e4` derives the strict language by template grid from the
  declared validation dataset. The focused suite passes 8/8. Immutable
  715-file snapshot deployment manifest `439f6bb9...6530` backs isolated
  corrected Tswana `1282788` and first-run Xhosa `1282789`. They run with NER
  Xhosa `1282017` and Zulu `1282231` on all four A100-80GB cards. The failed
  POS output is excluded; held-out and Sheet state are unchanged.

- At 23:15 SAST on 31 August, the user prospectively authorized a 36-hour
  Slurm ceiling for only the four not-yet-submitted General Stage-C
  confirmations. Amendment SHA-256 is `4a07b2a2...d0bf5`. The native
  `sbatch --time=36:00:00` override will leave the immutable scientific
  launcher and all model/data/optimizer/validation rules unchanged. The
  runtime-only basis is the fixed 13,640-step schedule at about 6.8 seconds
  per step, or 25.8 hours before exact validation and finalization. Current
  b1/b2/b4 jobs `1279470/1279471/1279652` remain on their original 24-hour
  requests; b5 preflight `1279656` remains pending. The change does not
  authorize another b3 attempt or touch held-out data. A quota-first HEX
  `sbatch --test-only` accepted the exact 36-hour A100-80GB/eight-CPU
  envelope, and queue readback confirmed that no probe job was created.

- At 21:47 SAST on 31 August, exact General b3 continuation `1279472`
  terminally failed after model load and partial data processing on a transient
  SSL error fetching the unchanged AfriHG Zulu validation file. It produced no
  resumed training, validation artifact, or final adapter, but the frozen
  payload terminality rule forbids another attempt. The strict all-11 General
  ranking is therefore blocked pending an explicit disclosed protocol choice.
  B1/b2 remain healthy near steps `11486/11492`. Candidate-order b4 then passed
  immutable compute-node preflight `1279651`, started as `1279652`, and
  advanced directly from checkpoint 10912 to step 10913. Candidate-specific
  b5 preflight `1279656` is
  capacity-pending as the fourth active owned submission. General Stage-B
  remains `2/8`, global freeze `4/8`, adapter held-out access zero, and Sheet
  E/F/G blank.

- At 20:42 SAST on 31 August, General b7 `1277552` became terminal-valid:
  checkpoint 5456 retained validation-only macro NLL `0.9166715052168936`,
  its terminal artifact had exact `22,167`-row coverage, and verifier
  `1279479` proved exact retained-to-final equality across 424 keys and
  76,410,112 values. General Stage-B advances to `2/8`. B6 timed out with no
  final adapter and a complete hash-preserved checkpoint 10912. Initial b1/b2
  recovery jobs `1278130/1278993` failed before payload loading due to a
  Slurm spool-relative bundle path; the prospective path-only correction is
  frozen under SHA-256 `5248bf2d...db54`. Corrected b1/b2 jobs
  `1279470/1279471` passed immutable provenance and resumed exactly at step
  10913; b3 recovery `1279472` is capacity-pending. Global family freeze
  remains `4/8`, adapter held-out access is zero, and Sheet E/F/G remain
  blank.

- At 18:35 SAST on 31 August, General b6 `1277424` reached a complete
  step-10912 boundary with exact `22,167`-row coverage, sidecar-verified
  artifact SHA-256 `b35cc3f0...18f19`, and validation-only macro NLL
  `1.0334865102881885`. It remains healthy and retains step 10912 within-run;
  this changed no cross-candidate decision. B6/b7 remain running and exact
  b1/b2 recoveries remain capacity-pending. Freeze is `4/8`, adapter held-out
  access is zero, and Sheet E/F/G remain blank.

- At 14:31 SAST on 31 August, pure-GDN General b6/b7 remained healthy after
  `18:20/16:40`; exact b1/b2 recoveries `1278130/1278993` remained
  capacity-pending behind two other-user jobs. Freeze remains `4/8`, adapter
  held-out access remains zero, and Sheet E/F/G remain blank. Separately, the
  post-hoc LLaMA T2X lane completed all `4/4` confirmations and froze b7 by
  validation-only three-seed mean: `49.89486922190678` versus a2
  `48.57093807884933`. B7 seed-42 checkpoint 1932 is representative. The
  read-only confirmation-ranking artifact SHA-256 is `6e224598...329a2`;
  this post-hoc result cannot affect pure-GDN.

- At 13:32 SAST on 31 August, General b5 `1277423` reached its fixed wall and
  ended `TIMEOUT`/`0:0` with no final adapter; complete checkpoint 10912 and
  its five state hashes are preserved. The released owned position activated
  exact b2 recovery: frozen candidate-specific preflight passed and job
  `1278993` was submitted once behind b1 `1278130`. B6/b7 remain healthy;
  two other-user jobs keep both recoveries capacity-pending. Pure-GDN freeze
  remains `4/8`, adapter held-out access remains zero, and Sheet E/F/G remain
  blank. Separately, post-hoc LLaMA T2X a2 seed 13 became terminal-valid with
  retained validation chrF `48.084019151368054` at checkpoint 1932 and exact
  retained-to-final equality across 124 keys and 67,929,088 values.
  Confirmation progress is `3/4`; a2 seed 87 started cleanly with exact
  `3,859/460` rows and execution-manifest SHA-256 `6be8bfd8...68d3`.

- At 12:31 SAST on 31 August, pure-GDN General b5/b6/b7 remained healthy after
  `23:21/16:20/14:39`; exact b1 recovery `1278130` remained queued to take
  b5's released A100-80GB slot. Freeze remains `4/8`, adapter held-out access
  remains zero, and Sheet E/F/G remain blank. Separately, post-hoc LLaMA T2X
  b7 seed 87 became terminal-valid with retained validation chrF
  `49.52933730149589` at checkpoint 1449 and exact retained-to-final equality
  across 124 keys and 68,743,168 values. Confirmation progress is `2/4`.
  Frozen a2 seed 13 then started cleanly with exact `3,859/460` rows and
  execution-manifest SHA-256 `a6d62d02...d81b`.

- At 11:32 SAST on 31 August, pure-GDN General b5/b6/b7 remained healthy after
  `22:20/15:19/13:39`; exact b1 recovery `1278130` remained
  capacity-pending. B5 crossed complete checkpoint 10912 and continues
  unchanged toward its fixed wall. Freeze remains `4/8`, adapter held-out
  access remains zero, and Sheet E/F/G remain blank. Separately, Kombuys GPU 0
  became isolated and frozen LLaMA T2X b7 seed 87 passed every absence,
  provenance, ranking, and isolation check before starting with exact
  `3,859/460` rows. Its execution-manifest SHA-256 is `108d532d...e465`;
  confirmation progress remains `1/4` until terminal verification.

- At 10:30 SAST on 31 August, pure-GDN General b5/b6/b7 remained healthy after
  `21:22/14:21/12:40`; exact b1 recovery `1278130` remained
  capacity-pending. Freeze remains `4/8`, adapter held-out access remains zero,
  and Sheet E/F/G remain blank. Separately, post-hoc LLaMA T2X b7 seed 13
  became terminal-valid with retained validation chrF `50.25952890593673` at
  checkpoint 1932 and exact retained-to-final equality across 124 keys and
  68,743,168 values. Confirmation progress is `1/4`. B7 seed 87 remains absent
  because another user's process currently owns frozen serial GPU 0; no
  foreign process was changed.

- At 09:32 SAST on 31 August, pure-GDN General b5/b6/b7 remained healthy near
  steps `10259/6708/5808`; exact b1 recovery `1278130` remained
  capacity-pending. Freeze remains `4/8`, adapter held-out access remains zero,
  and Sheet E/F/G remain blank. Separately, post-hoc LLaMA T2X b7 became
  terminal-valid with retained validation chrF `49.89574145828771` at
  checkpoint 1932 and exact retained-to-final equality across 124 keys and
  68,743,168 values. The complete seed-42 grid is `11/11`; hashed ranking
  artifact `deb603fa...9fc8` freezes b7 and a2 as the top two. B7 seed 13 is
  now running serially on isolated Kombuys GPU 0 with exact `3,859/460` rows
  and execution-manifest SHA-256 `44a95846...6d37`.

- At 08:47 SAST on 31 August, pure-GDN General b5/b6/b7 remained healthy near
  steps `9885/6336/5456`; exact b1 recovery `1278130` remained
  capacity-pending. B6 step 5456 and b7 step 2728 each have exact 22,167-row
  coverage and verified sidecars; their within-run macro NLL values are
  `1.079590829544548` and `0.9391455045913958`. Freeze remains `4/8`, adapter
  held-out access remains zero, and Sheet E/F/G remain blank. Separately,
  post-hoc LLaMA T2X b6 became terminal-valid with retained validation chrF
  `39.990824045303356` at checkpoint 1932 and exact retained-to-final equality
  across 124 keys and 67,929,088 values. Unchanged b7 then started serially on
  isolated Kombuys GPU 0 with exact `3,859/460` rows and execution-manifest
  SHA-256 `e143827b...697c`; that isolated grid is now `10/11`.

- At 07:26 SAST on 31 August, pure-GDN General b5/b6/b7 remained healthy near
  steps `9174/5629/4881`; exact b1 recovery `1278130` remained
  capacity-pending. Freeze remains `4/8`, adapter held-out access remains zero,
  and Sheet E/F/G remain blank. Separately, post-hoc LLaMA T2X b5 became
  terminal-valid with retained validation chrF `43.51713717457478` at
  checkpoint 1932 and exact retained-to-final equality across 124 keys and
  67,929,088 values. Unchanged b6 then started serially on isolated Kombuys
  GPU 0 with exact `3,859/460` rows and execution-manifest SHA-256
  `b98c410b...fd56`; that isolated grid is now `9/11`.

- At 06:27 SAST on 31 August, pure-GDN General b5 `1277423` passed its frozen
  step-8184 boundary with exact 22,167-row coverage, macro NLL
  `0.9589405712228745`, and matching artifact/sidecar SHA-256
  `b7ddf007...6039`; this changed only its within-run retained checkpoint.
  B5/B6/B7 remain healthy near steps `8591/5131/4331`; exact b1 recovery
  `1278130` remains capacity-pending. Pure-GDN freeze remains `4/8`, adapter
  held-out access remains zero, and Sheet E/F/G remain blank. Separately,
  post-hoc LLaMA T2X b4 became terminal-valid with retained validation chrF
  `44.57380920361152` at checkpoint 1449 and exact retained-to-final equality
  across 124 keys and 67,522,048 values. Unchanged b5 then started serially on
  isolated Kombuys GPU 0 with exact `3,859/460` rows and execution-manifest
  SHA-256 `50f22236...2633`; that isolated grid is now `8/11`.

- At 05:31 SAST on 31 August, post-hoc LLaMA-125M T2X b3 became terminal-valid
  with validation chrF
  `40.69645308307086/44.201515387271336/44.865800837994094/44.89238882495095`;
  checkpoint 1932 is retained and exactly equals the final adapter across 124
  keys and 67,929,088 values. All four 64-row artifacts have exact indices,
  beam 5, zero empty predictions, and 64 unique predictions. Retained/final
  SHA-256 values are `c216a948...e115` and `fa7b87bf...c9c0`. The unchanged
  b4 candidate then passed all frozen absence, provenance, and isolation checks
  and started serially on Kombuys GPU 0 with exact `3,859/460` rows. Its
  execution-manifest SHA-256 is `27e60edb...78e1`. The grid is `7/11`; this
  lane remains isolated from pure-GDN.

- At 04:25 SAST on 31 August, General b4 `1277379` reached its fixed wall at
  step `12115/13640` and ended `TIMEOUT`/`0:0` after `1-00:00:10`, with no
  final adapter. Its complete checkpoint-10912 adapter, optimizer, scheduler,
  RNG, and trainer-state hashes were preserved. The released owned position
  activated the frozen b1 same-trial recovery: exact wrapper
  `b3cd62c36dd368e4ee90f32324651f7de04dbcce8bc6914a260e1b48c7a730fa`
  passed all absence, archive, runtime, and checkpoint checks and was submitted
  once as `1278130`. It is `AssocGrpGRES`-pending while b5/b6/b7 continue
  healthily near steps `7730/4204/3282`. B4 continuation remains later than
  the preregistered b1/b2/b3 recovery sequence. Quota is `88.6%/45.0%`;
  adapter held-out access remains zero and Sheet E/F/G remain blank.

- At 03:18 SAST on 31 August, post-hoc LLaMA-125M T2X b2 became the sixth of
  eleven terminal-valid seed-42 candidates. Its four validation chrF values
  are `34.25156415347831/39.492568075259186/40.59239731926782/39.96438880908078`;
  checkpoint 1449 is retained and exactly equals the final adapter across 124
  keys and 67,522,048 values. The first boundary contains one scored empty
  prediction; retained and terminal boundaries contain none. Unchanged b3 then
  passed all frozen absence/provenance/isolation checks and started serially on
  Kombuys GPU 0. B3 execution-manifest SHA-256 is
  `a0d1f5bf5fc1d87acf42d1ec59a5381ababe9f1d630599a6eaca0457ee3e4fda`.
  This post-hoc lane remains isolated from pure-GDN.

- At 02:18 SAST on 31 August, frozen post-hoc LLaMA-125M T2X b2 started on an
  isolated Kombuys RTX 5090 after absent-output/no-duplicate checks and exact
  verification of 694 source/config files, six model artifacts, tokenizer,
  registry recipe, and GPU isolation. It loaded exact `3,859/460`
  train/validation rows and entered the 1,932-step schedule. Execution-manifest
  SHA-256 is
  `651e627bd83ff23887bd30e867fb66dc9d7f270bfd30ccfa68717790736a8abd`.
  The lane is post-hoc and cannot affect pure-GDN selection, scheduling,
  held-out access, or ETA.

- At 02:16 SAST on 31 August, all four General jobs remained healthy near
  steps `11187/6606/3075/2310`, with no recovery slot released. B4 step-10912
  and b6 step-2728 artifacts both have exact `22,167`-row coverage and matching
  sidecars. B4's macro NLL `0.9606701593321049` is a within-run
  non-improvement, so it retains step 8184; b6's first-boundary macro NLL is
  `1.4567470667356914`. Artifact SHA-256 values are
  `a1d589a017078ce78286279006009138d60834e0b8ece07ba069eb15549e4e46` and
  `6c24d2f8b77ea7c6f5abdb2d7739ff03a809d4159c4235d81109cacb00e4c518`.
  These are validation-only within-run facts and changed no schedule or budget.

- At 00:16 SAST on 31 August, General b4/b5/b6/b7 remained healthy on all four
  A100-80GB cards near steps `10284/5552/2151/1252`, with no targeted fault
  markers and no released recovery slot. B5's sidecar-verified step-5456
  validation artifact has exact `22,167`-row coverage, SHA-256
  `6239224a015003aef3f3a7cd1548bf9c06ab176cb70ae74f419e79fe5af0da82`,
  and within-run macro NLL `0.9722614411299868`. It changes no candidate,
  schedule, retry, or budget. Adapter held-out access remains zero and Sheet
  E/F/G remain blank.

- At 22:12 SAST on 30 August, General b7 `1277552` left the association queue
  and started cleanly on A100-80GB. B4/B5/B6/B7 are now all running near
  steps `9183/4619/1056/162`, with fresh logs and zero targeted fault markers.
  Home/scratch quota is `88.6%/44.9%`. All four owned positions remain
  occupied, so the frozen b1 checkpoint-10912 recovery is still gated. On
  Kombuys, another user's process continues to occupy frozen LLaMA GPU 0 and
  b2 remains absent and unlaunched. No held-out artifact, Sheet E/F/G cell,
  job, or foreign process was changed.

- At 21:29 SAST on 30 August, the user prospectively reduced the corrected
  post-BOS News/SIB/Intent HPO budget before any run or metric existed. Frozen
  amendment SHA-256
  `23818b9fbaf064d0e0f131f23678db39701eaf8f6379ffcd203241ebf4e13872`
  requires `a0/a1/a2` at seed 42 plus seed 13 for each top-two finalist, with
  two-seed mean selection. Stage-B and seed 87 are uniformly omitted. This
  reduces those families from 45 to 15 GPU runs and the post-General
  training/HPO tail from 69 to 39 launches. Pure-GDN remains A100-80GB-only;
  A100 families may never overlap. Current General jobs were unchanged,
  adapter held-out access remains zero, and Sheet E/F/G remain blank.

- At 21:23 SAST on 30 August, the full-results critical path was corrected and
  accelerated without changing science. The heartbeat's 15 September expiry
  is a safety cutoff, not an ETA; monitoring now runs every 15 minutes to
  reduce released-slot idle time. General Stage-B has an optimistic lower
  bound of late 1 September, but the full table still requires four General
  confirmations, 45 missing-family runs, 20 new Mono trainings, freeze
  verification, one-time adapter evaluation, the real-FLA CUDA canary, and 14
  corrected raw base lanes. A calendar ETA remains withheld until the three
  post-BOS News/SIB/Intent a0 runtimes exist. The
  monitor now explicitly overlaps General confirmations with those canaries
  and avoids completion-serial barriers while preserving frozen order and the
  four-submission cap.

- At 21:10 SAST on 30 August, final independent review of the local strict
  producer port found and fixed one scientific-record integrity defect:
  General validation artifacts and sidecars now fail closed instead of
  replacing different bytes at the same step. The ordinary-classification
  concern was disproved against merged `main`; duplication-only suggestions
  were not pursued. Amended signed commit `633a993` passes 131 CPU tests,
  Ruff, formatting, ty, diff-check, and signature verification. It remains
  local and unpublished.

- At 21:00 SAST on 30 August, the reviewed strict General/POS/News/SIB/Intent
  producer commit was proven against the actual merged `main` (`94815d5`) in
  a disposable local branch. Signed commit `633a993` applies cleanly and passes
  131 CPU tests, all-file pre-commit, ignore-expiry, diff-check, signature, and
  high-severity Grype gates. Nothing was pushed and no PR was created. The
  deliberately stopped overbuilt HPO slice remains excluded.

- At 20:55 SAST on 30 August, the authorized cleanup stack merged in the
  required order: `#126`, `#127`, `#129`, `#130`, `#134`, then `#131`.
  Rebased heads passed local CPU, pre-commit, policy, signature, and Grype
  gates plus fresh required GitHub checks. PR `#118` is closed as superseded
  by `#127`; issue `#125` correctly remains open because the merged dependency
  work does not eliminate its upstream sqlitedict advisory. The dirty research
  checkout was not rebased or cleaned in place.

- The latest quota-first HEX readback is home `88.6%` and scratch `44.9%`.
  General b4/b5/b6 are running and b7 remains association-pending. Four owned
  submissions are still active, so no b1 recovery slot exists. The defensible
  family freeze remains `4/8`; adapter held-out access remains zero and Sheet
  E/F/G remain blank.

- At 20:00 SAST on 30 August, the last known producer-validity gap was closed
  locally without launching science. The disposable integration stack now has
  strict post-BOS News, SIB, and Intent contracts. SIB requires exactly 2,970
  rows in 30 language/prompt cells; Intent requires 4,155 rows in 20 cells;
  both require constrained validation-only mean-language macro-F1. Fifty-four
  focused tests plus Ruff, format, and ty pass. The configs and code remain
  uncommitted and no held-out artifact was inspected.

- The first issue-43 implementation was intentionally stopped at the user's
  requested clean boundary. Its focused tests pass, but the current
  frozen-trial implementation is too large to merge as cleanup. It remains
  disposable evidence for a later smaller slice; no protocol JSON, live job,
  remote branch, or production run uses it.

- At 19:47 SAST on 30 August, the result audit and fresh-stack producer port
  converge on a defensible `4/8` pure-GDN family freeze. T2X, NER, POS, and
  AfriHG are frozen under their disclosed authorities; News, SIB, Intent, and
  General remain unfinished. The post-BOS News/SIB/Intent protocols have no
  candidate runtimes yet, so the previous 5--6 September complete-table claim
  is retired rather than pushed out again. A new completion estimate will be
  made only from live first-candidate runtimes.

- The disposable six-PR integration stack now contains behavior-complete
  strict validation producers for General, serial full-prefix POS, and
  post-BOS News. News fails closed unless all `3,095` validation rows, exact
  Eng/Xho and P1--P5 cells, constrained scoring routes, and finite macro-F1
  verify before artifact publication. Callback errors are broadcast across
  distributed ranks instead of hanging peers. Ordinary classification and
  Mono POS remain backward compatible. The focused combined suite is `58
  passed`; Ruff, format, and ty checks pass. No commit, push, live launch, or
  held-out access occurred.

- Issue `#43` is the only required provenance follow-up for those producers.
  Its production slice is intentionally narrow: immutable candidate/protocol
  pins, non-sweep trial execution, terminal markers written only after hashed
  validation artifacts and exact retained-to-final equality, and ranking that
  rejects partial or timed-out runs. Four obsolete dirty-root launcher/protocol
  scripts remain retirement candidates and will not be revived.

- At 19:34 SAST, HEX quota remained `88.6%/44.9%`; General b4/b5
  `1277379/1277423` were healthy near steps `7947/3226`, b6/b7
  `1277424/1277552` remained association-pending, and all four owned submission
  positions were occupied. The exact b1 checkpoint-10912 recovery therefore
  remained administratively ineligible. On Kombuys, another user's process
  held frozen LLaMA GPU 0, so b2 remained absent and unlaunched. No other-user
  job or process was changed.

- The post-hoc architecture audit confirms five terminal-valid LLaMA-125M T2X
  seed-42 candidates and no legitimate winner yet. The frozen xLSTM checkpoint
  is already present and hash-valid, making its metric-free GPU-1 structural
  gate the highest-value separate breadth action once explicitly authorized.
  The frozen Mamba target protocol is terminally incompatible with PEFT; any
  replacement must be a new prospectively frozen protocol. None of this may
  influence pure-GDN selection.

- The six cleanup PRs remain a safe ordered stack locally: `#126`, `#127`,
  `#129`, `#130`, `#134`, then `#131`, with one known `#126/#134` test conflict
  already resolved in disposable integration evidence. Remote mutation is
  paused because the app requires explicit approval naming this exact stack;
  the user's repository-cleanup objective otherwise remains active. PR `#118`
  and issue `#125` may close only after their replacements merge.

- At 19:00 SAST on 30 August, the two live launch gates remained correctly
  closed. On HEX, quota was `88.6%/44.9%`; General b4/b5
  `1277379/1277423` were healthy at steps `7629/2910`, while b6/b7
  `1277424/1277552` remained `AssocGrpGRES`-pending. All four owned
  submission positions were occupied and other-user jobs `1276835/1276836`
  were untouched, so no b1 recovery preflight or submission was allowed. On
  Kombuys, another user's ASR process held frozen LLaMA GPU 0; the b2 root,
  log, manifest, tmux session, and process were absent. GPU 1 remained idle
  but outside the frozen LLaMA protocol. No held-out metric was accessed.

- The General b1/b2 recovery handoff is now fully audited for the next
  eligible HEX slot. Only wrapper SHA-256
  `b3cd62c36dd368e4ee90f32324651f7de04dbcce8bc6914a260e1b48c7a730fa`
  may be used, in frozen b1-then-b2 order, from each exact
  `checkpoint-10912`; the first resumed training step must be `10913`.
  Before either one-time submission, rerun the quota-first queue/history,
  absent-output, wrapper/registry-hash, and `--preflight-only` checks, then
  record the parsable job ID immediately. The three superseded recovery
  wrappers remain forbidden.

- At 18:57 SAST on 30 August, the General/POS fresh-stack port became
  behavior-complete locally in `/private/tmp/sallm-stack-py5FEn/repo` without
  changing any live PR. Public red-to-green tracers now cover General dispatch,
  assistant-token-weighted equal-family NLL, exact artifact/sidecar bytes,
  real serial full-prefix POS scoring, constrained POS callback artifacts, and
  factory composition. The frozen four-prompt GDN POS config is included.
  A critical review correction makes constrained POS opt-in through the exact
  `SALLM_POS_SELECTION_PROTOCOL`; ordinary Mono and five-prompt POS runs retain
  their existing generation evaluator. The port also fails closed on the three
  earlier review defects: incomplete NER expansion, Transformers 5
  `BatchEncoding`, and undercovered POS cells. It deliberately omits rejected
  POS batch/cache code, the obsolete chat helper, and unrelated dirty paths.
  All 110 CPU tests, all-file pre-commit, diff-check, and the cached Grype high
  gate pass; independent review found no substantial issue. This remains
  uncommitted integration evidence. Issue #43 must add and record the explicit
  POS protocol in the prospective production launcher/manifest before live use.

- At 18:40 SAST on 30 August, quota remained `88.6%/44.9%`. General b4/b5
  `1277379/1277423` were healthy at steps `7467/2750`; b6/b7
  `1277424/1277552` remained `AssocGrpGRES`-pending with absent logs. Other-user
  jobs `1276835/1276836` still held the other two A100-80GB cards and were
  untouched. All four owned submission positions remain occupied, so no b1
  recovery preflight or submission was allowed. Held-out state is unchanged.

- At 18:40 SAST on 30 August, the frozen post-hoc LLaMA-125M T2X b2 launch
  remained correctly gated. Kombuys GPU 0 was occupied by `csikasote` ASR PID
  `1983501` using `19,452 MiB`; b2 output, log, tmux session, and process were
  absent. GPU 1 was idle but remains outside the frozen LLaMA protocol. No
  process or artifact changed and no held-out metric was accessed.

- At 18:08 SAST on 30 August, stale open PR #118 was reproduced read-only and
  should not be rebased or merged. Its pytest-CI work is superseded by #127;
  the exact head still fails Ruff and two of six CLI tests, while its Oryx
  file, one-off `SCRATCH` fallback, and undocumented in-process CLI contract
  have no current requirement. After explicit authorization and #127's merge,
  close #118 as superseded and transplant none of its files. No new issue is
  warranted unless a current caller later requires in-process execution or a
  repository-wide no-`SCRATCH` policy.

- At 18:03 SAST on 30 August, independent review closed the dirty-root
  classification's provenance gap. The report now enumerates all 20 exact-use
  evidence scripts/tests, the separate causal-label-shift investigation test,
  and all 15 PR-represented source/test paths; pins the six current draft
  heads; separates the News code port from its future live canary; and forbids
  merging draft or blocked heads. Each authorized merge
  must start from refreshed main, rebase the remaining draft, rerun its checks,
  and preserve both sides of the known #126/#134 test conflict. The corrected
  cleanup report, now also carrying the PR #118 triage, has SHA-256
  `aeca1c7a1049949bde8ecf0d9d79088935e803429390241ebe3e40118a3c428a`.

- At 18:01 SAST on 30 August, quota remained `88.6%/44.9%`. General b4/b5
  `1277379/1277423` were healthy near steps `7110/2553`; b6/b7
  `1277424/1277552` remained `AssocGrpGRES`-pending with absent roots and
  logs. B4's manifest and both validation sidecars, plus b5's manifest
  sidecar, verified exactly. No terminal adapter or roundtrip exists yet.
  Other-user jobs `1276835/1276836` were untouched. All four owned submission
  positions remain occupied, so frozen b1 is eligible but cannot yet launch;
  no recovery was preflighted or submitted and held-out state is unchanged.

- At 18:00 SAST on 30 August, the frozen post-hoc LLaMA-125M T2X b2 launch
  remained correctly gated. Its output, log, tmux session, and process are
  all absent, but Kombuys GPU 0 is occupied by `csikasote` process `1966755`
  using `28,044 MiB` for ASR work. No other-user process was changed and b2
  was not launched. GPU 1 remains idle and outside the LLaMA protocol.

- At 17:58 SAST on 30 August, the dirty-root cleanup was classified without
  deleting user work: 168 files at the pre-report snapshot split into 107
  frozen evidence files, 15 files whose behavior is already represented by
  focused PRs, 14 production files requiring selective fresh-main ports, 20
  superseded implementations to archive then retire, and 12 local `.audit`
  files. Issues #124/#43 cover the remaining work; no new ticket is needed.
  The existing snapshot predates later evidence, so a new manifest-bound final
  archive is required immediately before retiring the research checkout.

- At 17:53 SAST on 30 August, read-only provenance corrected the xLSTM
  transfer premise. The exact private revision was already in Kombuys'
  protected HF cache from 29 June, when the user manually authenticated there
  and ran the exact xLSTM smoke; every required file hash matches. No current-
  task transfer occurred. The new post-hoc T2X activation remains blocked on
  explicit start approval and its absent isolated runtime/structural gate, not
  on a new copy.

- At 17:48 SAST on 30 August, an exact disposable integration of PRs
  #126/#127/#129/#130/#134/#131 found one deterministic cross-PR test failure:
  #131's CPU-only factory test inherited TRL 0.26.2 `bf16=True` after #126.
  The production path is correct; a three-line test-fixture fix explicitly
  sets `use_cpu=True`, `bf16=False`, and `fp16=False`. With that disposable
  fix, all 89 CPU tests, all-file pre-commit, diff-check, Grype ignore policy,
  and exact high-severity Grype gate pass. The live PR remains unchanged and
  not merge-ready until its branch receives this specifically authorized fix.
  Independent review confirmed all production finetune configs explicitly set
  precision, the fixture patch cannot mask a production defect, and the known
  #126/#134 conflict resolution preserves both parents' complete test groups.

- At 17:35 SAST on 30 August, the original Monolingual recipe authority was
  made explicit before any Mono launch: only the learning-rate component of
  the final hashed family winner transfers. Rank 16, alpha 32, dropout 0.05,
  warmup 0.03, and the original architecture-complete targets and optimizer
  recipe remain fixed. Train 20 new language adapters; the frozen T2X HPO
  winner is already the one T2X Xhosa Mono arm and must not be retrained.
  General b1--b3 continuation authority followed their timeouts and interim
  artifacts; the b4--b7 rule froze after start but before terminal outcomes
  and activates only after timeout. Both require metric-independent amended-
  protocol disclosure, not an unamended preregistration claim.

- Hash correction: the current sidecar-verified SHA-256 of
  `2026-08-30-full-results-and-valid-improvement-audit.md` is
  `ca897d77e5ab42555908a93012d097252b662bdf2bac06a2ac196f8574a90254`.
  Earlier hashes identify prior versions of the same evolving audit note.

- The reviewed replacement for closed PR #128 will extend the existing
  `src/main/sallm/hpo/trial.py` seam after PRs #126, #130, #134, and #131 land.
  The dirty shell graph and obsolete `scripts/` launch paths will not be
  copied. Issue #43 freezes a protocol-owned trial interface with no caller
  override of config, checkpoint, tokenizer, metric, direction, or seed;
  ranking remains disabled until the same seam consumes full-coverage,
  final-adapter, retained-state, exact-roundtrip, terminal-sidecar, and
  complete-seed evidence. Existing General recoveries stay on their frozen
  wrapper.

- The smaller PR-128 replacement prototype is also deliberately uncommitted
  and undeployed. It proved a 303-line bind seam can be tested, but origin/main
  does not yet contain the corrected News/POS/General configs or the real
  launch graph, so its wrapper fails closed before live execution. Issue #43
  now requires those real inputs to be ported first and terminal/ranking checks
  to land in the same production-called vertical slice; do not merge the
  disconnected prototype.

- Draft PR #128 is closed and preserved, not merged. Independent acceptance
  review found zero production callers, acceptance of partial/TIMEOUT metrics,
  caller-controlled ranking direction, and no end-to-end binding to the actual
  config, architecture, checkpoint, tokenizer, or selection metric. A tested
  hardening prototype added another 615 lines without closing that integration
  gap, so it remains uncommitted. Replace it later with one small vertical
  launch-to-rank slice through the real wrapper; issue #43 records the exact
  terminality, roundtrip, coverage, direction, and complete-seed criteria.

- Recovery safety gate: wrappers `b4984293...`, `1e9d3e6c...`, and
  `39bc3ebc...` are superseded and forbidden. The only eligible b1--b3 wrapper
  is `b3cd62c36dd368e4ee90f32324651f7de04dbcce8bc6914a260e1b48c7a730fa`.
  No wrapper is yet eligible for b4--b7.

- At 17:16 SAST on 30 August, quota-first live state remained healthy and
  capacity-bound: General b4/b5 run near steps `6717/2157` on A100-80GB,
  b6/b7 remain `AssocGrpGRES`-pending, and all four owned submissions are
  occupied. B1 is scientifically eligible for its frozen recovery but cannot
  be submitted until a slot releases. Kombuys GPU 0 remains occupied by
  another user's process, so LLaMA b2 stays gated; GPU 1 is idle behind the
  explicit private xLSTM transfer approval. No job or process was changed.
  Adapter held-out access remains zero and Sheet E/F/G remain blank.

- At 16:42 SAST on 30 August, draft PR #131 was corrected at signed head
  `8a4336dcf4aacb5885f523cacc17a41d60d0bc45`. The shared classification
  training fallback now selects `eval_classification/all_macro_f1` when early
  stopping has no explicit metric, while `all_f1` remains reported. Fourteen
  focused and 57 full tests, pre-commit, diff-check, and independent review
  pass. Live lint is green; Grype remains inherited-red until #126 lands. The
  PR remains draft/unmerged.

- A disposable post-#126 rehearsal proved PR #127 rebases without conflict and
  passes frozen sync, 57 tests, pre-commit, and the current vulnerability
  policy. PR #134 has one predictable test-file conflict with #126; preserve
  both compatibility and BOS-boundary cases when rebasing. PR #131 combines
  cleanly after #134. No remote branch or dirty research file changed.

- At 16:30 SAST on 30 August, two prospective evidence corrections were frozen
  before any new terminal result. First, any b4--b7 24-hour timeout may receive
  one metric-independent same-trial continuation from its latest complete scheduled
  training-state checkpoint, after b1--b3 and in candidate order, with the same
  archive, manifest, runtime, source, and checkpoint bindings. Second, exactly
  14 invalid raw base lanes will be rerun once after the eight-family and
  21-Monolingual freeze using the merged evaluator corrections; unaffected T2X
  and AfriHG generation will not be repeated. Historical base held-out outputs
  already exist, so the accurate current boundary is adapter held-out access
  zero. Sheet E/F/G remain blank.

- Independent review tightened both amendments before use. A scientific
  continuation or corrected base artifact is terminal after any model, data,
  training, evaluator, prediction, or metric payload; only a pre-payload
  implementation failure can receive a separately hashed execution correction.
  Corrected base artifacts can update only Sheet column D, while E/F/G remain
  adapter-only. The exact frozen pure-GDN checkpoint must also pass a metric-
  free real Transformers 5/FLA CUDA load, forward/backward, and generation
  canary on A100-80GB before any corrected base held-out lane runs. The final
  General-timeout amendment SHA-256 is
  `f7f3d51b704ff3d0fdd303de97a854703d53ca40855e02ac262f3d23f5fa9779`;
  the corrected-base amendment SHA-256 is
  `9de3d72729f27036e044c53525e25cd2e428beac4c38851f97ec190be1c4c4d2`.

- At 16:19 SAST on 30 August, draft PR #126 resolved its Transformers 5
  blocker without changing the frozen LLaMA-125M architecture. A local strict
  compatibility class accepts only the exact `512/9/3/56` tuple, preserves
  the established `504/168/168/504` attention projection widths, and retains
  upstream validation for every other LLaMA. Ten focused cases cover all six
  checked-in base configs; 57 full tests, pre-commit, lock/sync, live lint,
  live Grype, and independent review pass. It remains unmerged pending the
  repository-required explicit confirmation to mark the draft ready.

- At 16:18 SAST on 30 August, issue #132's BOS/raw root fix reached reviewed
  draft PR #134. It prevents duplicate BOS and terminal EOS in the actual
  evaluator paths, gives raw lm-eval a BOS-only tokenizer, and fails closed
  if that corrected tokenizer cannot load or save. Fourteen focused and 51
  full CPU tests plus pre-commit and live lint pass; independent review found
  no issue. Its Grype check inherits current main's dependency failures and
  will rerun after PR #126 lands. The 14 corrected raw lm-eval base lanes stay
  quarantined for prospective rerun. Frozen adapter families, General HPO,
  LLaMA T2X HPO, and the held-out boundary remain unaffected.

- At 15:51 SAST on 30 August, focused pre-commit hooks passed for the final
  manifest-bound General recovery files. A quota-first HEX readback still has
  b4/b5 `1277379/1277423` running and b6/b7 `1277424/1277552` pending, so no
  recovery is eligible. Another user's ASR process `1928383` owns frozen
  Kombuys GPU 0, so post-hoc LLaMA b2 remains unlaunched. No job or process
  was changed; held-out access remains zero and Sheet E/F/G remain blank.

- Final independent review of recovery wrapper `b3cd62c3...`, its immutable
  bundle, durable b1/b2/b3 preflights, regression test, and decision trail
  found no remaining issue. It is implementation-ready only after the live
  external b1--b2--b3 queue gate releases.

- At 15:44 SAST on 30 August, the final General recovery review found that
  the runtime verifier's input manifest was not hash-bound. A red regression
  reproduced that gap. The only eligible wrapper is now SHA-256
  `b3cd62c36dd368e4ee90f32324651f7de04dbcce8bc6914a260e1b48c7a730fa`;
  it binds each original execution manifest before exact runtime comparison.
  Candidate-specific durable HEX preflights pass, and 128 full CPU tests pass.
  No recovery job has been submitted.

- At 15:39 SAST on 30 August, independent review blocked the General
  recovery path before submission because in-place checkpoint pruning could
  erase original evidence and the job did not enforce runtime equality. The
  final bundle `pure-gdn-general-stage-b-recovery-20260830-39bc3ebc` now
  requires candidate-specific read-only archives of all three interrupted
  roots and exact Python, platform, executable, and 179-package equality.
  Durable no-launch HEX preflights pass for b1/b2/b3; 128 full local CPU tests
  pass. No recovery job was submitted, and only wrapper SHA-256
  `39bc3ebc1b50f17d62c28484c7a2a40f337ae3cee9c1dcca18bc844e1a769e8c`
  is eligible after the external queue gate releases.

- At 15:37 SAST on 30 August, General b4/b5 `1277379/1277423` remained
  healthy near steps `5772/1222`; b6/b7 `1277424/1277552` remained
  association-pending. All four owned A100-80GB submission positions are
  occupied, so the exact b1, b2, then b3 recoveries remain correctly gated.
  Kombuys GPU 0 is still occupied by another user's 16.1 GB ASR process, so
  post-hoc LLaMA b2 remains unlaunched. No process or job was changed.

- At 15:35 SAST on 30 August, draft PR #131 corrected multilingual
  classification selection to the arithmetic mean of per-language label-macro
  F1 while preserving the existing support-weighted `all_f1` metric. All nine
  multilingual classification sweeps and three Mamba checkpoint selectors now
  use the intended key. Thirteen focused and 56 full tests, pre-commit, and
  GitHub lint pass; no result, held-out artifact, or frozen experiment file
  changed. The full local research-tooling suite also passes 127 tests.

- PR #126's Transformers 5 evidence is deliberately narrower than previously
  worded: source inspection and a tiny CPU stand-in verify the causal-loss,
  tokenizer, PEFT, and loader seams, but they do not run the real FLA
  GatedDeltaNet CUDA path. Real FLA/CUDA compatibility remains an unverified
  pre-deployment gate, not a claimed result.

- At 15:25 SAST on 30 August, independent review rejected the first General
  recovery wrapper before submission because it did not verify the full
  launcher/helper chain. The preserved replacement, SHA-256
  `1e9d3e6cb5b7b4afe1cd98bf30c58bafc03add7b4becfbb63fad0326a6212ab2`,
  now verifies the deployment manifest and all 695 frozen source/config
  files before every checkpoint-state check. Corrected focused tests pass,
  and actual HEX dry runs pass for b1/b2/b3. Scheduler eligibility and the
  b1--b2--b3 submission order remain live external gates; no recovery job has
  been submitted.

- Draft PR #130 pins MasakhaPOS to frozen upstream revision
  `376f4161f0425584d4bd7664122b56fa026926d3` and adds a bounded three-attempt
  retry only for transient HTTP and transport failures while preserving the
  existing filename fallback. Fourteen focused and 50 full tests plus all
  pre-commit hooks pass; independent implementation and trail reviews found
  no issue. It remains draft and changes no experiment result or held-out
  state.

- At 15:15 SAST on 30 August, the exact General b1/b2/b3 wall-time recovery
  path became implementation-ready without submitting a job. A single tested
  wrapper, SHA-256
  `b4984293b709c4e7cec6f98e288cf73923f362303f5ad2b2513b801f1ba9cdc8`,
  is deployed immutably on HEX. Actual dry runs verified every preserved
  checkpoint-10912 state hash and bound each candidate to its unchanged
  recipe, output root, and resume path. B4--b7 still occupy all four owned
  positions, so b1 remains correctly gated ahead of b2 and b3.

- At 15:12 SAST on 30 August, post-hoc LLaMA-125M T2X Stage-B b1 completed
  cleanly and retained terminal checkpoint 1932 at validation chrF
  `42.81002752398118`. All four artifacts have exact 64-row/index coverage,
  no empty predictions, and beam 5. The retained adapter exactly matches the
  final adapter across 124 keys and 68,743,168 values; retained/final SHA-256
  values are `4db781e6e301b7ae731cbeb34cba1ba8abc13c688d63a970204b07d591edd8ab`
  and `17ac9c9593e5e9bb20f6b3c2a091884f9a0d6362f50fbbbaa2834e2a69d65e92`.
  B2 remains gated because another user's processes occupy the assigned RTX
  5090; no process was modified and no held-out result was opened.

- Draft PR #129 ports the corrected NER entity parser and POS
  overprediction denominator from the immutable research snapshot to fresh
  main. Focused and full CPU tests plus pre-commit pass, and independent
  review found no substantial issue. The frozen NER manifests already bind
  the exact evaluator hash, while frozen POS used its separate constrained
  scorer, so this cleanup changes no scientific result. It remains draft and
  inherits the base dependency-security failure pending PR #126.

- At 15:01 SAST on 30 August, General b4 `1277379` completed its second
  frozen validation boundary. The step-5456 artifact and sidecar agree at
  SHA-256 `0ff8cbfabe4e48b64f6cebd1e2d81ed1dcf3c6441ea1e6f5f1bd50eefa84808e`,
  with exact 22,167-row coverage and validation-only macro NLL
  `0.9677870068460876`. B4 resumed training; b5 `1277423` remains healthy,
  and b6/b7 `1277424/1277552` remain association-pending. This is within-run
  retention evidence only; no cross-candidate selection or held-out access
  occurred.

- Draft PR #126 now fails closed at its new Grype-ignore policy step because
  the `sqlitedict` waiver expired on 16 April. Grype's unsupported `expires`
  key can no longer silently pass CI; obsolete pip/protobuf ignores were
  removed. The PR stays blocked until the user explicitly approves the
  minimum proposed renewal through 15 September or the dependency path is
  removed. Draft PR #128 separately ports the HPO registry/provenance tooling
  from fresh main; independent review found additional manifest/trial/ranking
  bindings that must be fixed before merge.

- At 14:58 SAST on 30 August, the dirty research checkout was classified as
  a provenance source, not a merge candidate. It predates the current data-
  adapter and `ops/slurm` refactors. Only focused ports from fresh main are
  allowed. The first port fixes NER entity parsing and POS overprediction
  scoring; rejected POS acceleration paths, terminal hardware gates, exact
  recovery wrappers, upload code, `.audit/`, and bulk `sallm_memory/` remain
  outside public main.

- At 14:51 SAST on 30 August, independent review blocked draft PR #126 from
  merge. Grype does not support the `expires` field used by the repository's
  ignore rules, so the expired sqlitedict High waiver is still silently
  active. The obsolete pip and protobuf ignores can be removed, but any
  renewed sqlitedict exception needs explicit approval and an expiry check
  that Grype actually enforces. The earlier green-security claim is withdrawn.

- The same review confirmed PR #127's CPU pytest job is not yet a protected
  required check. After #126 is corrected and merged, #127 must be updated,
  pass all checks, merge, and then `CPU pytest` must be added to main branch
  protection. Neither PR is ready to merge yet.

- At 14:48 SAST on 30 August, General b4 `1277379` reached step
  `5456/13640` and entered its second frozen validation pass; b5 `1277423`
  remained healthy near step `856/13640`. B6/b7 `1277424/1277552` remain
  association-pending, so b4--b7 still occupy the four owned submission
  positions and the frozen b1, b2, then b3 exact checkpoint recoveries are
  not yet eligible. The reconciled global freeze remains `4/8`; held-out
  access remains zero and Sheet E/F/G remain blank.

- At 14:48 SAST on 30 August, post-hoc LLaMA-125M T2X b1 remained healthy
  on Kombuys GPU 0 and had written checkpoint 966; no final adapter exists,
  so b2 remains correctly gated. The frozen xLSTM lane remains paused before
  private checkpoint transfer. An attempted public issue update was rejected
  because it would disclose private checkpoint identifiers; no workaround was
  attempted.

- At 14:45 SAST on 30 August, draft PR #127 added an independent CPU pytest
  CI job from fresh `origin/main`. Local verification is 43 tests, all
  pre-commit hooks, and `git diff --check`; GitHub CPU pytest and lint pass.
  Its Grype job sees only the vulnerable lock still present on `main`, so the
  explicit merge order is green dependency PR #126 first, then update and
  rerun #127. Both remain draft pending explicit ready/merge approval.

- At 14:43 SAST on 30 August, the frozen xLSTM gate paused before model
  transfer. Moving the private xLSTM checkpoint to Kombuys requires explicit
  user approval for that exact sensitive payload and destination; no indirect
  route is permitted. The destination remains empty and no package, data,
  metric, or candidate was created. Pure-GDN and LLaMA work are unaffected.

- At 14:42 SAST on 30 August, the idle Kombuys GPU 1 was assigned a
  prospective, separately labelled xLSTM T2X structural gate. The exact
  3-epoch xLSTM base revision, file hashes, runtime versions, architecture-
  native LoRA targets, fixed rank range, effective-batch-eight fit ladder,
  chunk-64 checks, candidate order, and no-retry boundary were frozen before
  any new xLSTM validation result. No task data or metric has been loaded.
  The activation is
  `2026-08-30-cross-architecture-kombuys-xlstm-t2x-activation.md`, SHA-256
  `c62a44b9fba66f4111fb91f5767ffcae4ec3aadfa54704def67d3e4e87f7efc5`.

- At 14:36 SAST on 30 August, a validation-only full-results audit ranked the
  current evidence without opening held-out data. T2X is the clearest
  full-protocol win; POS has the largest observed classification margin but is
  explicitly budget-limited; NER and AfriHG are valid narrower wins. The
  highest-payoff admissible improvements are completing General, running the
  missing post-BOS News/SIB/Intent grids, freezing all eight families, and
  then training the 21 Monolingual adapters. The audit is
  `2026-08-30-full-results-and-valid-improvement-audit.md`, SHA-256
  `bb8b2dbb2bfde6663efcde50ba0f01abc6ea9351268cf907a337305acd8d87bb`.

- At 14:34 SAST on 30 August, General b4/b5 `1277379/1277423` remained
  healthy near steps `5420/719`; b4's first frozen artifact is sidecar-valid
  with exact 22,167-row six-family coverage. B6/b7 `1277424/1277552` remain
  association-pending, so b4--b7 still fill the four owned submission
  positions and the preregistered b1, b2, then b3 checkpoint-10912 recoveries
  are not yet eligible. Post-hoc LLaMA-125M T2X b1 is also healthy near step
  `842/1932` on Kombuys GPU 0; its first 64-row artifact is structurally
  complete, with no final adapter yet. No held-out result was opened and no
  job was changed.

- At 14:30 SAST on 30 August, fresh-main draft PR #126 completed both required
  GitHub checks after a paired Transformers 5/lm-eval migration. Local
  verification is 45 tests plus all pre-commit hooks; the CI-configured Grype
  scan has zero unsuppressed High/Critical findings. The sole no-ignore High
  is unfixed `sqlitedict` 2.1.0 inherited from lm-eval. Its prior waiver was
  not silently renewed; follow-up is issue #125. PR #126 remains draft until
  explicit ready/merge approval.

- At 14:18 SAST on 30 August, post-hoc LLaMA-125M T2X Stage-B b0 completed
  cleanly and exactly roundtripped terminal checkpoint 1932 across 124 keys
  and 67,522,048 values; its validation chrF is `46.98977851830069`.
  Unchanged b1 then passed absent-output, registry, manifest, exact-data, and
  GPU-isolation checks and entered training on Kombuys GPU 0. The separately
  frozen Mamba-2 activation stopped before data or metrics: after correcting a
  metric-free tokenizer-verifier assumption, a preserved CPU-only structural
  probe proved PEFT `0.18.1` rejects the required Mamba-2 `out_proj` target.
  Changing targets or bypassing PEFT would be a new protocol, so Mamba is
  terminally blocked under this preregistration and GPU 1 remains idle.

- At 13:50 SAST on 30 August, a repository-wide and remote HEX evidence audit
  corrected the pure-GDN global freeze from the unsupported `7/8` shorthand to
  a defensible `4/8`. T2X and NER have full enhanced rankings, POS is frozen
  under its prospective budget-limited close-out, and a new three-seed-mean
  reconciliation confirms AfriHG b7 without changing any result. No post-BOS
  News, SIB, or Intent adapter-HPO job, output, manifest, or ranking exists;
  each remains `0/11 + 0/4`. The missing News wrapper path is now covered by a
  red-green dry-run test. Current General and Kombuys jobs remain unchanged,
  held-out access remains zero, and Sheet E/F/G remain blank. Mono and official
  tests are blocked until News/SIB/Intent and General complete and one hashed
  eight-family freeze manifest verifies. The superseding audit is
  `2026-08-30-pure-gdn-global-freeze-evidence-reconciliation.md`.

- At 13:30 SAST on 30 August, General b3 `1276434` ended `TIMEOUT` at exact
  step `11991/13640` with no final adapter; its complete checkpoint-10912
  state is preserved and the recovery preregistration now freezes b1, b2,
  then b3 exact same-trial resumes (`48d8881e...304b6`). B5 `1277423`
  started cleanly, b4 remained
  healthy near `4872`, b6 stayed pending, and unchanged b7 was submitted once
  as `1277552` after fail-closed preflight. Separately, post-hoc LLaMA T2X a2
  completed and exactly roundtripped retained terminal checkpoint 1932 at
  validation chrF `48.52835434271314`; b0 then started cleanly on Kombuys GPU
  0 with manifest `ea1a72d...b8c0f`. Pure-GDN Stage-B remains `1/8`, global
  freeze `7/8`, quota `88.6%/44.7%`, held-out access zero, and Sheet E/F/G
  blank.

- At 11:00 SAST on 30 August, post-hoc LLaMA-125M T2X a1 completed cleanly
  and exactly roundtripped its retained step-1449 adapter across 124 keys and
  67,929,088 values; retained validation chrF is `45.53530032884151`. The
  first a2 launcher omitted `WANDB_MODE=offline` and failed before model/data
  loading, so its metadata-only root is preserved. A prospective one-line
  correction was frozen at SHA-256 `346a023b...1f398`; corrected a2 now runs
  in isolated tmux `sallm-llama125-t2x-a2-corrected`, with exact 3,859/460
  data counts and manifest `77d2de4b...194be2`. On HEX, General b3 remained
  active after 21:51, b4 was healthy near `3539/13640`, and b5/b6 remained
  `AssocGrpGRES`-pending. Pure-GDN Stage-B stays `1/8`, global freeze
  `7/8`, quota `88.6%/44.6%`, held-out access zero, and Sheet E/F/G blank.

- At 09:52 SAST on 30 August, post-hoc LLaMA-125M T2X Stage-A a0 completed
  cleanly on Kombuys GPU 0. Frozen-boundary validation chrF peaked at
  `42.29958392023858` at retained checkpoint 1449; exact retained-to-final
  equality was proven across 124 keys and 67,929,088 values. After
  absent-output/no-duplicate and GPU-isolation checks, unchanged a1 started
  serially in `sallm-llama125-t2x-a1`; its execution-manifest SHA-256 is
  `0b77f549...195`. GPU 1 remains idle. This post-hoc lane remains isolated
  from pure-GDN and has not accessed held-out data.

- At 08:48 SAST on 30 August, the user activated a separate post-hoc uniform
  HPO reproduction for older architectures on otherwise idle Kombuys GPUs;
  pure-GDN remains unchanged on HEX. The first frozen lane is LLaMA-125M T2X
  on GPU 0 RTX 5090, with architecture-native `q_proj`/`v_proj` targets and
  the fixed 11-candidate validation-only budget. Stage-A a0 is running in
  `sallm-llama125-t2x-a0`, verified the immutable 694-file manifest
  `203c1816...5c944`, loaded exact `3,859/460` train/validation rows, and
  entered its 1,932-step run. The activation preregistration SHA-256 is
  `d0450b2f...a9d8`. GPU 1 remains idle pending a separate xLSTM or Mamba
  protocol. This does not authorize held-out access or Sheet changes.

- At 06:44 SAST on 30 August, b3/b4 remained healthy near `8778/1419` while
  b5/b6 stayed `AssocGrpGRES`-pending behind two other-user jobs. A
  prospective wall-time recovery protocol now freezes one exact
  checkpoint-10912 same-trial resume each for b1 then b2, using only the
  already-proven corrected resume propagation after b7 is submitted and later
  slots release. No recovery was submitted and no held-out or cross-candidate
  evidence informed the amendment. Stage-B remains `1/8`, global freeze
  `7/8`, quota `88.6%/44.6%`, held-out access zero, and Sheet E/F/G blank.

- At 05:43 SAST on 30 August, General Stage-B b3 `1276434` produced a clean,
  sidecar-matching step-8184 artifact at SHA-256 `d11f0e9d...f308`, with
  exact 22,167-row coverage and validation-only macro NLL
  `0.9508656889374341`. It improves its own step-5456 value and retains step
  8184. B3/b4 remained healthy near `8251/887`; b5/b6 stayed
  `AssocGrpGRES`-pending behind two other-user jobs. Stage-B remains `1/8`,
  global freeze `7/8`, quota `88.6%/44.6%`, held-out access zero, and Sheet
  E/F/G blank.

- At 04:44 SAST on 30 August, General Stage-B b1/b2 `1276432/1276433` ended
  as `TIMEOUT` at the fixed 24-hour limit with their complete step-10912
  adapter, optimizer, scheduler, RNG, and trainer states preserved and hashed;
  neither wrote a final adapter. B4 `1277379` started at 04:00 and was healthy
  near step 352; b3 remained healthy near step 7881. Immutable-registry,
  absent-output, and no-duplicate checks passed before unchanged b5/b6 were
  submitted once in order as `1277423/1277424`; both are pending behind the
  shared four-card cap. Stage-B remains `1/8`, global freeze `7/8`, held-out
  access zero, and Sheet E/F/G blank. Exact b1/b2 same-trial recoveries remain
  required; the complete-table plan now uses the 5--6 September failure
  buffer.

- At 02:43 SAST on 30 August, CPU verifier `1277380` completed `0:0` and
  proved exact equality between General b0's retained step-5456 adapter and
  final adapter across 424 keys and 69,438,784 values. General Stage-B
  advances to `1/8` terminal-valid. B1/B2 produced clean, sidecar-matching
  step-10912 artifacts with exact 22,167-row coverage and validation-only
  macro NLL `0.9586805432462503/1.004664310741637`; b1 retains step 8184 and
  b2 retains step 10912. B1/B2/B3 remained healthy near
  `11301/11476/6827`, while b4 `1277379` stayed `AssocGrpGRES`-pending.
  Global freeze remains `7/8`, quota `88.6%/44.6%`, held-out access zero, and
  Sheet E/F/G blank.

- At 01:43 SAST on 30 August, General Stage-B b0 `1276431` completed `0:0`.
  Its sidecar-matching terminal step-10912 artifact SHA-256 is
  `41f7ab1a...97b4`, with exact 22,167-row coverage and validation-only macro
  NLL `0.9643478094125281`; it retains step 5456 at `0.9451928643283297`.
  CPU roundtrip verifier `1277380` was submitted once and is pending, so b0
  is not yet scientifically counted. Immutable-registry, absent-output, and
  no-duplicate checks passed before unchanged b4 was submitted once as
  A100-80GB job `1277379`; it is priority-pending. Stage-B remains `0/8`,
  global freeze `7/8`, held-out access zero, and Sheet E/F/G blank.

- At 00:38 SAST on 30 August, General Stage-B b3 `1276434` produced a clean,
  sidecar-matching step-5456 validation artifact at SHA-256
  `d868b01a...b3a9`, with exact 22,167-row six-family coverage and
  validation-only macro NLL `0.9549178716296577`. This improved b3's own
  step-2728 value and remains within-run retention evidence only. All four
  jobs remained healthy near `10564/10374/10533/5740` with empty fault scans.
  Stage-B stayed `0/8`, global freeze `7/8`, quota `88.6%/44.5%`, held-out
  access zero, and Sheet E/F/G blank.

- At 21:34 SAST on 29 August, General Stage-B b0/b1/b2/b3 remained healthy
  near `8919/8754/8892/4276` with empty fault scans and stable step times. All
  four A100-80GB cards stayed occupied by required work. B3's step-5456
  artifact remained due around 23:55--00:20, and b0--b2 step-10912 artifacts
  around 01:30--02:05 on 30 August. Stage-B stayed `0/8`, global freeze `7/8`,
  quota `88.6%/44.5%`, held-out access zero, and Sheet E/F/G blank.

- At 20:32 SAST on 29 August, General Stage-B b0/b1/b2 produced clean,
  sidecar-matching step-8184 artifacts with exact 22,167-row coverage and
  validation-only macro NLL `0.9508059313758785/0.9539537704610211/`
  `1.0190495201500418`. B0 had its first within-run non-improvement and kept
  step 5456; b1/b2 improved and retained step 8184. Artifact hashes were
  `d5e82ffe...9982`, `b42dd821...2def`, and `8c945d29...3205`. All four jobs
  remained healthy near `8376/8220/8350/3738`. Stage-B stayed `0/8`, global
  freeze `7/8`, quota `88.6%/44.5%`, held-out access zero, and Sheet E/F/G
  blank.

- At 19:31 SAST on 29 August, General Stage-B b3 `1276434` produced a clean,
  sidecar-matching step-2728 artifact at SHA-256 `4872027b...1849e`, with
  exact 22,167-row six-family coverage and validation-only macro NLL
  `1.006736571720808`. This is within-run retention evidence only. B3 resumed
  near step `3208`; b0/b1/b2 remained healthy near `7986/7842/7965` with
  empty fault scans and step-8184 artifacts due around 20:05--20:35. Stage-B
  stayed `0/8`, global freeze `7/8`, quota `88.6%/44.5%`, held-out access
  zero, and Sheet E/F/G blank.

- At 18:31 SAST on 29 August, General Stage-B b3 `1276434` reached exact step
  2728 and entered its first frozen validation; its artifact was not yet
  present and was expected around 18:45--19:05. B0/b1/b2 remained healthy near
  `7446/7311/7424` with empty fault scans and their next artifacts due around
  20:05--20:35. All four A100-80GB cards remained occupied by required work.
  Stage-B stayed `0/8`, global freeze `7/8`, quota `88.6%/44.4%`, held-out
  access zero, and Sheet E/F/G blank.

- At 17:30 SAST on 29 August, General Stage-B b0/b1/b2/b3 remained healthy
  near `6906/6781/6884/2287` with empty fault scans and stable `6.6--6.9`
  second step times. All four A100-80GB cards stayed occupied by required
  work. B3's first artifact remained due around 18:30--18:50 and the next
  b0--b2 artifacts around 20:05--20:35. Stage-B stayed `0/8`, global freeze
  `7/8`, quota `88.6%/44.4%`, held-out access zero, and Sheet E/F/G blank.

- At 16:31 SAST on 29 August, an apparent two-hour General Stage-B progress
  gap was disproved as a delayed heartbeat timestamp. Live HEX time, log
  mtimes, and consecutive progress records showed all four jobs advancing
  normally near `6.7` seconds per step with empty fault scans at
  `6380/6263/6362/1762`. No job was changed. B3's first frozen artifact is
  projected around 18:30--18:50, and the next b0--b2 artifacts around
  20:05--20:35. Stage-B remained `0/8`, global freeze `7/8`, quota
  `88.6%/44.4%`, held-out access zero, and Sheet E/F/G blank.

- At 16:24 SAST on 29 August, General Stage-B b0/b1/b2
  `1276431/1276432/1276433` produced clean, sidecar-matching step-5456
  artifacts with exact 22,167-row coverage. Validation-only macro NLL values
  were `0.9451928643283297/0.9699125326068634/1.0699096838669606`, improving
  each run's own step-2728 value and retaining step 5456 without ranking
  candidates. Artifact hashes were `b403ba75...50ce`, `e7c9a027...bd64`, and
  `fbdcc51b...2bcf`. All four jobs remained healthy; b0/b1/b2 resumed near
  `6332/6217/6317` and b3 `1276434` reached about step `1711`. Stage-B stayed
  `0/8`, global freeze `7/8`, quota `88.6%/44.4%`, held-out access zero, and
  Sheet E/F/G blank.

- At 13:10 SAST on 29 August, General Stage-B b3 job `1276434` started on the
  fourth A100-80GB card after its preregistered queue wait. Its execution
  manifest SHA-256 is `1344f1c7...255bdf`; the exact association resources,
  all 694 frozen source/config files, and the GatedDeltaNet fast path verified
  before healthy model loading. B0/b1/b2 remained healthy near
  `4743/4661/4737`. All four eligible cards are now occupied by required owned
  Stage-B jobs. Stage-B remains `0/8`, global freeze `7/8`, quota
  `88.6%/44.4%`, held-out access zero, and Sheet E/F/G blank.

- At 12:09 SAST on 29 August, General Stage-B b0/b1/b2
  `1276431/1276432/1276433` remained healthy near steps `4217/4144/4207` with
  empty fault scans. Their step-5456 artifacts were still absent as expected
  and are projected for approximately 14:40--15:10. B3 `1276434` remained
  `AssocGrpGRES`-pending behind other-user job `1274494`. Stage-B stayed `0/8`,
  global freeze `7/8`, quota `88.6%/44.4%`, held-out access zero, and Sheet
  E/F/G blank.

- At 10:09 SAST on 29 August, General Stage-B b0/b1/b2
  `1276431/1276432/1276433` had clean, sidecar-matching step-2728 artifacts
  with exact 22,167-row coverage. Validation-only macro NLL values were
  `0.9817947401411412/1.0586443713118137/1.240997314146131`; these retain
  checkpoints within each run but do not rank candidates. Artifact hashes were
  `ee49a2b0...8bb33`, `5159eae7...ffa5d`, and `44e2ce9e...b74d2`.
  All resumed healthy near steps `4173/4100/4167`; b3 `1276434` remained
  `AssocGrpGRES`-pending. Stage-B stayed `0/8`, global freeze `7/8`, and quota
  `88.6%/44.4%`.

- At 08:02 SAST on 29 August, General Stage-B b0/b1/b2
  `1276431/1276432/1276433` remained healthy near steps `2162/2122/2153` on
  three A100-80GB cards with empty fault scans. First frozen artifacts remained
  projected for 09:15--09:35. B3 `1276434` stayed `AssocGrpGRES`-pending;
  Stage-B remained `0/8`, global freeze `7/8`, and quota `88.6%/44.3%`.

- At 07:01 SAST on 29 August, General Stage-B b0/b1/b2
  `1276431/1276432/1276433` remained healthy near steps `1860/1822/1851` on
  three A100-80GB cards with empty fault scans. First frozen artifacts were
  projected for 08:50--09:20. B3 `1276434` remained
  `AssocGrpGRES`-pending behind other-user job `1274494`, with the same
  movable 13:08 estimate. Stage-B stayed `0/8` and global freeze `7/8`.

- At 05:59 SAST on 29 August, General Stage-B b0/b1/b2
  `1276431/1276432/1276433` remained healthy near steps `1044/1023/1034` on
  three A100-80GB cards with empty fault scans. First frozen boundaries
  remained projected for 09:10--09:40. B3 `1276434` stayed
  `AssocGrpGRES`-pending behind other-user job `1274494`, with a movable 13:08
  Slurm estimate. Stage-B remained `0/8` and global freeze `7/8`.

- At 04:58 SAST on 29 August, General Stage-B b0/b1/b2 jobs
  `1276431/1276432/1276433` were healthy near steps `488/484/488` on three
  A100-80GB cards with empty fault scans. Their first frozen boundaries were
  projected for 09:10--09:40 SAST. B3 `1276434` remained
  `AssocGrpGRES`-pending behind other-user job `1274494`, with a movable Slurm
  estimate of 13:08. Stage-B remained `0/8`, global freeze `7/8`, held-out
  access zero, Sheet E/F/G blank, and quota `88.6%/44.3%`.

- At 04:00 SAST on 29 August, corrected General a0 recovery `1276350` was
  scientifically terminal-valid after completing `0:0`, exact 22,167-row
  six-family coverage, sidecar-verified artifact SHA-256 `95c3abc9...e05`,
  and terminal validation-only macro NLL `0.9830990526022072`. Checkpoint
  13640 is retained. CPU verifier `1276429` completed `0:0` and proved exact
  retained-to-final equality across 424 keys and 71,762,560 values. General
  Stage-A closes at `3/3`. Stage-B b0--b3 jobs `1276431--1276434` were
  submitted once after fail-closed preflight; b0--b2 are running on three
  A100-80GB cards and b3 is `AssocGrpGRES`-pending behind other-user job
  `1274494`. Global freeze remains `7/8`, held-out access zero, Sheet E/F/G
  blank, and quota `88.6%/44.3%`.

- At 02:53 SAST on 29 August, corrected General a0 recovery `1276350` remained
  healthy at step `13236/13640` on A100-80GB with an empty fault scan and
  about 45 minutes of training remaining. Its terminal artifact window stayed
  04:15--05:00. General remained Stage-A `2/3`, global freeze `7/8`, held-out
  access zero, and Sheet E/F/G blank.

- At 01:53 SAST on 29 August, corrected General a0 recovery `1276350` remained
  healthy at step `12684/13640` on A100-80GB, steady near `6.6` seconds per
  step with an empty fault scan. Training-only completion remained near 03:38
  SAST and the terminal artifact window 04:15--05:00. General stayed Stage-A
  `2/3`, global freeze `7/8`, held-out access zero, and Sheet E/F/G blank.

- At 00:52 SAST on 29 August, corrected General a0 recovery `1276350` remained
  healthy at step `12141/13640` on A100-80GB, steady near `6.6` seconds per
  step with an empty fault scan. Training-only completion remained near 03:38
  SAST and the terminal artifact window 04:15--05:00. General stayed Stage-A
  `2/3`, global freeze `7/8`, held-out access zero, and Sheet E/F/G blank.
  Quota was `88.6%/44.3%`.

- At 23:52 SAST on 28 August, corrected General a0 recovery `1276350` remained
  healthy at step `11587/13640` on A100-80GB, steady near `6.6` seconds per
  step with an empty fault scan. Its terminal window remained 04:15--05:00
  SAST on 29 August. General stayed Stage-A `2/3`, global freeze `7/8`,
  held-out access zero, and Sheet E/F/G blank. Quota was `88.6%/44.3%`.

- At 22:52 SAST on 28 August, corrected General a0 recovery `1276350` remained
  healthy at step `11043/13640` on A100-80GB, holding near `6.7` seconds per
  step with no runtime fault. Training-only completion projects near 03:43
  SAST on 29 August and the terminal validation/artifact window near
  04:15--05:00. General remains Stage-A `2/3`, global freeze `7/8`, held-out
  access zero, and Sheet E/F/G blank. Quota was `88.6%/44.3%`; two A100-80GB
  cards were free but no Stage-B work was yet scientifically eligible.

- At 22:40 SAST on 28 August, corrected General a0 recovery `1276350` was
  healthy at step `10940/13640` on A100-80GB. It resumed directly at step
  `10913` from checkpoint-10912, proving the corrected path did not restart
  from step 0. One other association job was active, leaving two A100-80GB
  cards free; no later General trial is scientifically eligible before this
  Stage-A result verifies terminally. Quota was `88.6%/44.3%`.

- At 22:34 SAST on 28 August, explicitly authorized corrected General a0
  recovery `1276350` started on one A100-80GB after a complete fail-closed
  preflight. All five checkpoint-10912 hashes remain exact, no final adapter or
  duplicate existed, and the immutable 695-file correction snapshot verifies.
  Launcher/batch/deployment-manifest hashes are `45fec06b...ceab`,
  `7f3d0644...0c2f`, and `8b6a744b...d9bb6`; the resume-specific execution
  manifest sidecar matches `c47a2cf9...879bd` and records the exact checkpoint.
  Trainer-level resume is confirmed: progress jumped directly to step
  `10913/13640` and continued past `10919`, with no step-0 restart or fault
  marker. General stays `2/3` Stage-A terminal-valid, global freeze `7/8`,
  held-out access zero, and Sheet E/F/G blank.

- At 11:49 SAST on 28 August, authorized General a0 recovery `1275241` was
  cancelled after 5:49 because its log proved `resume_from_checkpoint=None`
  and a fresh step-0 start. It produced no checkpoint or validation metric.
  The immutable 11 August launcher never consumed the exported resume variable;
  all five checkpoint-10912 hashes remain unchanged. A validated prospective
  resume overlay is prepared at SHA-256 `45fec06b...ceab`, but the explicit
  no-second-recovery rule requires new authorization before deployment or one
  corrected replacement. General Stage-A remains `2/3`, global freeze `7/8`,
  held-out access zero, and Sheet E/F/G blank.

- At 11:45 SAST on 28 August, General a2 `1274502` became scientifically
  terminal-valid. It completed `0:0` after step-10912 patience-2 early stopping;
  sidecar-verified terminal artifact SHA-256 is `98dffa8e...c1cb` with exact
  22,167-row six-family coverage. Checkpoint 5456 remains best at validation-only
  macro NLL `0.9284030074439155`. CPU verifier `1275799` completed `0:0` and
  proved exact retained-to-final equality across 424 keys and 71,762,560 values.
  General Stage-A advances to `2/3`; exact a0 recovery `1275241` started on the
  released A100-80GB at 11:39:37. Global freeze remains `7/8`, quota home
  `88.6%` and scratch `44.2%`, held-out access zero, and Sheet E/F/G blank.

- At 09:55 SAST on 28 August, General a2 `1274502` remained healthy near
  `10074/13640` on one A100-80GB and was projected to enter its step-10912
  validation boundary around 11:25. Exact a0 recovery `1275241` remained the
  next pending owned job. The shared `nlpgroup80` four-GPU cap was saturated:
  another group member held the other three cards with jobs
  `1274494/1274495/1274496` and had more queued. This shared-cap contention is
  the current critical-path constraint; General remains `1/3` terminal-valid,
  global freeze `7/8`, quota home `88.6%` and scratch `44.1%`, held-out access
  zero, and Sheet E/F/G blank.

- At 06:37 SAST on 28 August, General a2 `1274502` completed its frozen
  step-8184 validation and resumed healthy A100-80GB training. Sidecar-verified
  artifact SHA-256 is `1b93d30a...7cea3`, with exact 22,167-row six-family
  coverage including all 3,082 AfriHG rows. Validation-only macro NLL
  `0.9491996698865588` did not improve the retained step-5456 value
  `0.9284030074439155`, so checkpoint 5456 remains the within-run best. This
  is interim evidence, not a General selection. A0 recovery `1275241` remains
  queued for 28 August 14:23. General is `1/3` terminal-valid, global freeze
  `7/8`, quota home `88.6%` and scratch `44.1%`, held-out access zero, and
  Sheet E/F/G blank.

- At 01:32 SAST on 28 August, General a2 `1274502` completed its frozen
  step-5456 validation and resumed healthy A100-80GB training. Sidecar-verified
  artifact SHA-256 is `9db07a48...6505`; coverage is exact across all 22,167
  rows, including all 3,082 AfriHG rows. Validation-only macro NLL improved
  from `0.9487367487872976` to `0.9284030074439155`. This is interim evidence,
  not a selection. A0 recovery `1275241` remains queued for the current 28
  August 14:23 estimate. General remains `1/3` terminal-valid, global freeze
  `7/8`, held-out access zero, and Sheet E/F/G blank.

- At 23:31 SAST on 27 August, General a2 `1274502` remained healthy at
  `4755/13640` on A100-80GB. The scheduler estimate for the already-submitted
  a0 recovery `1275241` improved from 29 August 13:08 to 28 August 14:23 SAST,
  aligned with a2's current allocation end. General remains `1/3`
  terminal-valid and global freeze remains `7/8`; held-out access is zero and
  Sheet E/F/G remain blank.
- Intent a1 confirmation `1287272`, News English Mono `1287274`, and SIB
  Afrikaans/English Mono `1288159/1288160` completed `0:0`. CPU verifiers
  `1288213`-`1288216` proved exact retained-to-final equality for each across
  `424` keys and `71,762,560` values. Ranking job `1288218` froze Intent to a1
  at family LR `8e-5`; ranking SHA-256 is
  `5151121aa34b5f7299f6aa830c10b70d37c821a7308eb3e10813123b7b01e89f`.
  Twelve of 21 Mono checkpoints are now frozen, including the already-frozen
  T2X arm. Released cards run News Xhosa `1288219`, SIB Northern Sotho
  `1288220`, SIB Southern Sotho `1288221`, and queue Intent English `1288222`.
  Held-out and Sheet E/F/G are unchanged.

- At 20:58 SAST on 27 August, the one-time prospective Kombuys RTX 5090
  validation-only equivalence gate completed but failed scientifically. Exact
  source, model, adapter, validation boundary, hardware, and environment checks
  passed, but predictions and aggregate metrics differed. Maximum/mean score
  differences were `0.053466796875/0.015472412109375`, exceeding frozen
  `0.05/0.01` limits. RTX scoring took `204.61164229223505` seconds versus
  `36.0765298968181` on A100-80GB, about 5.67 times slower. Candidate and
  comparator SHA-256 values are `929eb1b5...c6c2` and `51fc366d...9492`.
  The result is preserved, the gate will not be rerun or relaxed, and both
  Kombuys GPUs are excluded from pure-GDN model work. HEX remains the only
  valid GPU environment; held-out access remains zero and Sheet E/F/G blank.

- At 20:38 SAST on 27 August, the user authorized the preregistered exact
  same-trial recovery of General a0 after its clean wall-time timeout. The
  fail-closed preflight confirmed source job `1271042` is terminal `TIMEOUT`,
  no active a0 duplicate or final adapter exists, no prior resume manifest
  exists, and all five checkpoint-10912 hashes match the frozen recovery
  protocol. Resume job `1275241` was submitted once on the currently ratified
  A100-80GB family with exact `nlpgroup80/a100/nlpgroup80`,
  `gpu:ampere80:1`, 24 hours, eight CPUs, and the unchanged immutable
  `uniform-adapter-hpo-20260811-6aabf717` launcher. Its batch script exports
  only the frozen checkpoint resume path in addition to the existing runtime
  variables. It is `AssocGrpGRES` pending behind the four occupied A100-80GB
  cards; a2 `1274502` remains running. General remains `1/3` terminal-valid,
  global freeze `7/8`, held-out access zero, and Sheet E/F/G blank.

- At 20:28 SAST on 27 August, General Stage-A a2 job `1274502` remained
  healthy on A100-80GB near step `3115/13640`. Its first frozen boundary at
  step `2728` produced a sidecar-verified, exact 22,167-row six-family
  validation artifact with equal-family macro NLL `0.9487367487872976` and
  SHA-256 `889c17edad1bc6f749337e838f2007ff5ecf80463d63aa8e4d62db80e59cf68f`.
  This is interim validation-only evidence, not a winner. Other-user jobs
  `1274494/1274495/1274496` occupy the other three A100-80GB cards; the
  A100-40GB node is also full and L40S is saturated with a long queue.
  General remains `1/3` terminal-valid, global freeze `7/8`, and the exact
  a0 checkpoint-10912 resume remains unsubmitted pending scientific
  authorization. Quota is home `88.6%`, scratch `44.1%`; held-out access is
  zero and Sheet E/F/G remain blank.

- At 17:26 SAST on 27 August, General a0 job `1271042` was confirmed
  `TIMEOUT` at `16:41:29` after reaching step `12131/13640` without a model,
  data, evaluator, or training error. No final adapter or post-step-10912
  checkpoint exists. All five complete checkpoint-10912 hashes reverified,
  confirming wall time as the sole failure mechanism. The prospective exact-
  resume recovery remains unsubmitted pending scientific authorization. A2
  `1274502` remains healthy near step `1585`; other-user jobs
  `1274494/1274495/1274496` occupy the other three A100-80GB cards. General
  Stage-A remains `1/3` terminal-valid and global freeze remains `7/8`.

- At 16:27 SAST on 27 August, General a0 job `1271042` remained healthy near
  step `11964/13640`, but its `16:41:18` 24-hour limit makes terminal
  completion impossible at the observed rate. The latest complete
  checkpoint-10912 includes adapter, optimizer, scheduler, RNG, and trainer
  state and remains best at validation-only macro NLL `0.9838042431257504`.
  A prospective exact-resume protocol is drafted but not authorized or
  submitted; the original terminal state must be preserved first. A2
  `1274502` remains healthy, General Stage-A remains `1/3` terminal-valid, and
  the global freeze remains `7/8`.

- At 15:24 SAST on 27 August, General Stage-A a1 job `1271043` completed
  `0:0`; artifact SHA-256 `acd76a93...ebc6` verifies exact 22,167-row
  six-family coverage, and validation-only checkpoint `8184` remains best at
  macro NLL `0.9499744008920644`. CPU verifier `1274655` completed `0:0` and
  proved exact retained-to-final equality across 424 tensor keys, making
  General Stage-A `1/3` terminal-valid. A0 `1271042` remains healthy and its
  step-10912 full-coverage artifact is preserved at SHA-256
  `439f3119...124d`, but its continued improvement leaves insufficient time
  to reach terminal within the current 24-hour allocation; any timeout will
  be preserved and diagnosed before recovery. A2 `1274502` started as soon as
  a1 released its A100-80GB card. Global family freeze remains `7/8`, held-out
  access is zero, and Sheet E/F/G remain blank.

- At 13:24 SAST on 27 August, AfriHG froze scientifically to seed-42 b7
  checkpoint `15410`. Seed-87 a2/b7 jobs `1271041/1271354` completed `0:0`
  with clean 128-row exact artifacts; CPU verifiers `1274503/1274504` proved
  exact retained-to-final tensor equality across all 424 keys. The complete
  validation-only freeze artifact has SHA-256 `3cc31f0b...0f5db`; seed 42
  remains decisive and seeds 13/87 are robustness only. Global family freeze
  advances to `7/8`. Corrected General Stage-A a2 seed-42 job `1274502` was
  submitted after absent-output/no-duplicate checks and is `AssocGrpGRES`
  pending because other-user jobs `1274494/1274495` occupy the two A100-80GB
  cards released by AfriHG. General a0/a1 `1271042/1271043` remain healthy.
  Held-out access is zero, Sheet E/F/G remain blank, and the working complete-
  table ETA remains 1 September with 2 September buffer.

- At 22:57 SAST on 26 August, corrected General Stage-A a0/a1
  `1271042/1271043` produced the first scientifically complete step-2728
  validation artifacts and resumed training. Sidecar-verified SHA-256 values
  are `0e41f378...a5ce` and `ab5a8b49...fd0d`; both persist exact 22,167-row
  six-family coverage including all 3,082 AfriHG rows, raw summed NLL and
  valid-token counts, per-family NLL, and the preregistered equal-family macro
  NLL. A0/a1 macro NLL is `1.1208475241267701/0.988150101648468`. This proves
  the corrected General coverage and aggregation path is operating as
  registered; the values remain interim validation-only evidence.

- At 16:51 SAST on 26 August, all four ratified A100-80GB jobs remained
  healthy and running. Live positions were a2 seed-87 `1271041` at
  `144/15410`, General a0/a1 `1271042/1271043` at `61/13640` and `62/13640`,
  and isolated b7 seed-87 `1271354` at `88/15410`. Compute-side sampling found
  about `11.6--17.6 GiB` used per card with bursty utilization, consistent
  with the previously documented small-model pipeline rather than a fallback.
  All four cards are occupied; changing microbatch inside either active grid
  would require matched reruns, so the frozen recipes remain unchanged.

- At 16:45 SAST on 26 August, the corrected A100 gate passed scientifically.
  Reference `1271037` and candidate `1271038` completed `0:0` sequentially;
  corrected comparator `1271352` passed every frozen check with exact
  predictions, zero score difference, and artifact SHA-256
  `9f699ff2...fa377`. A100-80GB is ratified. Four jobs now run on
  `srvrocgpu011`: a2 seed-87 `1271041`, General a0/a1 `1271042/1271043`, and
  output-isolated b7 seed-87 `1271354`. Initial b7 `1271040` is preserved and
  quarantined after its obsolete wrapper reopened the cancelled partial path;
  the corrected overlay and new manifest are hash-recorded. AfriHG remains
  confirmations `2/4`, global freeze `0/8`, held-out access zero, and Sheet
  E/F/G blank. Working full-table ETA is 1 September, with late 31 August best
  case and 2 September buffer.

- At 15:45 SAST on 26 August, the user authorised one prospective A100-80GB
  launcher correction after failed reference `1269243` produced no science or
  result. Immutable amendment/wrapper hashes are `73988566...3f4a` and
  `36022115...3c81`. B7 seed-87 `1270629` was cancelled and preserved after
  `02:48:42`; pending `1270630/1270631` never started. The final strict chain
  is A100-40GB reference `1271037`, A100-80GB candidate `1271038`, and CPU
  comparator `1271039`; the reference has Slurm reservation `17:41:17`.
  Comparator success releases four dependency-held A100-80GB jobs: b7/a2
  seed-87 `1271040/1271041` and General a0/a1 `1271042/1271043`. Held-out
  access remains zero and Sheet E/F/G blank. Conditional on a pass, full-table
  ETA returns to 1 September, with late 31 August best case and 2 September
  buffer.

- At 13:49 SAST on 26 August, AfriHG b7 seed-87 confirmation `1270629` was
  healthy on A100-40GB at `1113/15410`; its immutable execution manifest
  re-verified at SHA-256 `8888487e...bf584`. A2 seed-87 `1270630` remained
  `Resources`-pending with Slurm reservation `27 August 12:52:33`, while
  corrected General a0 `1270631` remained `Priority`-pending without an
  estimate. AfriHG confirmations remain `2/4`, global freeze `0/8`, held-out
  access zero, and Sheet E/F/G blank. The A100-40GB queue is the blocker;
  complete-table ETA remains 2--3 September best case and 3--5 September with
  queue/failure buffer.

- At 12:47 SAST on 26 August, corrected A100-40GB gate reference `1269243`
  failed `126:0` before science because its immutable nested launcher had mode
  `0444` and could not execute. No result artifact exists. The gate is not
  rerun or relaxed; A100-80GB is excluded, and never-started dependencies
  `1269245/1269246` were cancelled. Frozen A100-40GB jobs b7 seed-87 `1270629`,
  a2 seed-87 `1270630`, and corrected General a0 `1270631` were submitted after
  absent-output/no-duplicate checks and are queue-pending. Best-case full table
  moves to 2--3 September; queue/failure-aware ETA is 3--5 September. Held-out
  access remains zero and Sheet E/F/G remain blank.

- At 09:42 SAST on 26 August, A100-40GB gate reference `1269243` remained
  `Priority`-pending and its Slurm estimate slipped to `19:22`. All four
  A100-80GB devices were idle, but the sequential gate correctly prevents use
  before the reference and comparator pass. Two `alsilo001` jobs are now
  queued ahead as well as the four running A100-40GB jobs, so the useful
  intervention is priority or allocation for the short reference itself. The
  working complete-table ETA remains 1 September.

- At 07:39 SAST on 26 August, b7 verifier `1269961` completed `0:0` and proved
  exact retained-to-final equality across 424 keys and 76,410,112 values, so
  AfriHG confirmations advance to `2/4`. A100-40GB gate reference `1269243`
  remained `Priority`-pending and its Slurm estimate slipped to `13:19:00`;
  all four A100-80GB devices remained idle. The queue is the only blocker and
  the working complete-table ETA remains 1 September.

- At 06:39 SAST on 26 August, b7 seed-13 `1267877` completed `0:0` with a
  clean 128-row terminal artifact and validation-selected checkpoint 15410 at
  mean chrF `25.376360945254902`. Roundtrip verifier `1269961` is
  `Priority`-pending, so AfriHG confirmations remain `1/4`. A100-40GB gate
  reference `1269243` is `Resources`-pending with current Slurm estimate
  `11:19:14`; all four A100-80GB devices are idle. The A100-40GB queue is the
  critical blocker, while the working complete-table ETA remains 1 September.

- At 05:39 SAST on 26 August, b7 seed-13 `1267877` had reached `15410/15410`
  and entered its frozen terminal exact callback. A2 seed-13 was already
  scientifically terminal-valid, so AfriHG confirmations remained `1/4` until
  b7's terminal artifact and exact roundtrip verification complete. The
  automatic A100 gate remained correctly held.

- At 04:39 SAST on 26 August, a2 seed-13 confirmation `1267878` became
  scientifically terminal-valid: completion `0:0`, clean 128-row terminal
  artifact, validation-selected checkpoint 15410 with mean chrF
  `24.895106041491907`, and CPU verifier `1269959` proving exact
  retained-to-final equality across 424 keys and 71,762,560 values. AfriHG
  confirmations advance to `1/4`. B7 seed-13 `1267877` remained healthy at
  `15035/15410`; the automatic A100 gate remained correctly dependency-held.

- At 03:37 SAST on 26 August, a2 seed-13 `1267878` had reached `15410/15410`
  and entered its frozen terminal exact callback; b7 seed-13 `1267877` was
  healthy at `13869/15410`. No incomplete callback or health-only loss is used
  for selection. The automatic A100 gate remained correctly held.

- At 02:37 SAST on 26 August, b7 seed-13 `1267877` had completed a clean,
  hash-recorded step-12328 artifact and resumed at `12750/15410`; a2 seed-13
  `1267878` was healthy at `14509/15410`, 901 optimizer steps from its
  terminal boundary. The A100 gate remained correctly held and held-out access
  remained zero.

- At 01:34 SAST on 26 August, a2 seed-13 `1267878` had completed a clean,
  hash-recorded step-12328 exact artifact and resumed at `13313/15410`; b7
  seed-13 `1267877` had entered its step-12328 exact callback. Both remained
  healthy on A100-40GB and the automatic A100 gate remained dependency-held.
  Held-out access is still zero and Sheet E/F/G remain blank.

- At 00:33 SAST on 26 August, a2 seed-13 `1267878` entered its frozen
  step-12328 exact callback after complete declared validation; b7 seed-13
  `1267877` remained healthy at `11404/15410`. The automatic A100 hardware
  gate remains correctly dependency-held, with no held-out access and Sheet
  E/F/G still blank.

- At 23:33 SAST on 25 August, b7 seed-13 confirmation `1267877` had completed
  a clean, hash-recorded step-9246 exact artifact and resumed at
  `10254/15410`; a2 seed-13 `1267878` was healthy at `12003/15410`. The
  automatic `1269243 -> 1269245 -> 1269246` hardware gate remains correctly
  dependency-held. No held-out data has been accessed and Sheet E/F/G remain
  blank.

- At 22:05 SAST on 25 August, the remaining full downstream scope was
  re-audited explicitly: after eight Multilingual-family winners freeze, 21
  Monolingual adapters still require validation-only training/checkpoint
  selection, followed by one official held-out pass over all applicable
  Mono/Multi rows and the General 16-lane matrix. Four A100-80GB GPUs keep late
  31 August possible only as a best case; the working complete-table ETA is
  1 September with 2 September failure/queue buffer.

- At 22:00 SAST on 25 August, the four-A100-80GB plan moves the best-case
  global Multilingual freeze to 29--30 August and best-case full downstream
  artifacts to late 31 August. The defensible working ETA is 1 September,
  with 2 September retained for queue or run failure. Active confirmations
  remain healthy and the automatic gate chain is queued.

- At 21:53 SAST on 25 August, the corrected hardware-gate chain was fully
  queued: A100-40GB reference `1269243`, sequential A100-80GB candidate
  `1269245`, then CPU comparator `1269246`. Hard dependencies prevent any
  pre-pass A100-family overlap. A passing comparator releases the planned
  four-job A100-80GB validation wave.

- At 21:51 SAST on 25 August, the user directed immediate use of the confirmed
  four-job allowance. A prospective queue-order clarification allows the
  corrected A100-80GB gate candidate to be submitted now as a dependency-held
  fourth job. It cannot run until `1267877/1267878` are terminal and A100-40GB
  reference `1269243` succeeds, so no pre-pass A100-family overlap is possible.

- At 21:49 SAST on 25 August, live Slurm accounting confirmed that the existing
  `nlpgroup80` association already allows four jobs and its QOS allows four
  A100-80GB GPUs per user. The user authorized a prospective capacity-only
  four-job amendment. It becomes active only after the corrected sequential
  A100 equivalence gate passes and changes no scientific selection rule.

- At 21:45 SAST on 25 August, the user authorized moving pending and future
  validation work to A100-80GB after a fresh sequential equivalence check. A
  narrow runtime-correction amendment was preregistered at SHA-256
  `14b2d2dd...25cf54`; it keeps the original validation data, metrics,
  thresholds, and fail-closed rule, while forcing both jobs through the actual
  current HPO runtime. Never-started b7 seed-87 job `1269234` was cancelled
  with no output to preserve the three-job cap. Corrected A100-40GB reference
  `1269243` now waits for untouched seed-13 confirmations `1267877/1267878` to
  finish; the A100-80GB candidate will run sequentially after it. A pass moves
  b7 seed-87, a2 seed-87, and General a0 to the idle A100-80GB pool. Until
  then there is no A100-family overlap. Held-out access remains zero and Sheet
  E/F/G remain blank.

- At 21:31 SAST on 25 August, owned jobs use two of four A100-40GB `ampere`
  GPUs (`1267877/1267878`), with third job `1269234` pending. The other two
  GPUs run one `a100free` job each for `chkkar002` and `bxxjin001`. The latter
  has one task running plus 45 collapsed array tasks pending, but those pending
  tasks have lower priority (`7795/7784`) than owned `1269234` (`7847`). The
  queue is heavily populated, but only one GPU is currently used by that
  large-array submitter; current evidence does not show their pending tasks
  positioned to jump ahead of the owned job.

- At 21:28 SAST on 25 August, Jan confirmed `gpu:amperemk` is reserved for
  Michelle's group and `gpu:ampere80` is accessible. He offered to address
  another student's contention on the A100-40GB `ampere` node. Live Slurm
  evidence shows seed-87 b7 `1269234` is blocked by `Priority`, with
  `a100free` jobs `1263877_2/1267954_2` occupying the two A100-40GB devices
  not used by owned confirmations `1267877/1267878`. Resolving that contention
  can release the third permitted slot without any protocol change.

- At 21:26 SAST on 25 August, a live capacity check found all four A100-80GB
  GPUs idle and both Kombuys GPUs idle. A100-80GB is the only useful major
  speed option, but remains scientifically excluded by the frozen gate whose
  sole mismatch was `pip` `23.3.1` versus `26.2.1`, despite exact predictions
  and selected scores. It may be reconsidered only through an explicit
  prospective protocol amendment, identical pinned runtime, and fresh
  untouched validation-only equivalence gate. Kombuys remains read-only with
  RTX 5090 untouched; its 12GB RTX 3080 Ti is unsuitable for unchanged full
  General/AfriHG training. No new job or protocol amendment was made. Current
  owned jobs are `1267877/1267878` running and `1269234` Priority-pending on
  A100-40GB; quota is `88.6%/43.0%`, held-out access `0`, and Sheet E/F/G
  remain blank.

- At 21:09 SAST on 25 August, the deadline plan was tightened without changing scientific selection: the workstream monitor now checks every 15 minutes, corrected General Stage-A a0 will start at the first slot released by a seed-13 confirmation while AfriHG seed-87 work continues, and a2 seed-87 will use the next free slot. The cap remains three identical A100-40GB jobs, every submission still requires absent-output/no-duplicate and immutable-provenance checks, and no held-out or recipe rule changes. The `gpu:amperemk` request remains unsent without explicit approval.

- At 21:06 SAST on 25 August, frozen AfriHG b7 seed-87 confirmation `1269234` was submitted once from the unchanged immutable snapshot after absent-output and no-duplicate checks. It is `Priority`-pending because all four A100-40GB devices are allocated; seed-13 confirmations `1267877/1267878` remain running, so the three-owned-job cap is full. No A100-80GB or L40S is used. A full-table critical-path audit shows that 31 August is not defensible under the frozen protocol: after AfriHG, corrected General still needs 11 seed-42 trials plus four confirmations at about 18 GPU-hours each, or about 108 best-case wall-clock hours in dependency waves on three continuously available GPUs, before Monolingual validation and one-time held-out evaluation. Best-case planning is global freeze around 31 August--1 September and full downstream artifacts around 2--3 September; queue/failure-aware delivery is 3--5 September. No scientific gate may be removed to meet the target.

- At 20:57 SAST on 25 August, a2 seed-13 `1267878` reached `9246/15410`, completed full declared validation at health-only loss `2.2032243356079655`, and entered its frozen third exact-generation callback at `20:16:38`; it remained active at `20:56` with normal generation progress, no artifact, and no fault marker. B7 seed-13 `1267877` remains healthy at `8693/15410`, about 553 steps from its third boundary. Both are `RUNNING` after `10:58:47` on separate A100-40GB `gpu:ampere:1` allocations on `srvrocgpu010`, with empty fault scans. A2's artifact is expected around `21:10--21:30`; b7 should reach step 9246 around `21:24--21:30` and produce its artifact around `22:50--23:20`. Terminal completion remains `04:45--08:15` on 26 August. Combined optimizer progress is about `58%`; scientifically AfriHG remains seed-42 `11/11`, confirmations `0/4`, global freeze `0/8`. These are the only owned jobs, no A100-80GB or L40S work exists, quota is `88.6%/43.0%`, Kombuys is read-only, Sheet E/F/G are blank, held-out access is `0`, and terminal confirmation compute remains the only blocker.

- At 20:04 SAST on 25 August, a2 seed-13 `1267878` and b7 seed-13 `1267877` are healthy at `9056/15410` and `7680/15410`, respectively, after `10:05:40` on separate A100-40GB `gpu:ampere:1` allocations on `srvrocgpu010`. Throughput is about `3.09--3.15 s/step` and targeted fault scans are empty. A2 should reach step 9246 around `20:13--20:16` and produce its third exact artifact around `21:15--21:40`; b7 should reach the boundary around `21:22--21:30` and produce its artifact around `22:50--23:20`. Terminal completion remains `04:45--08:15` on 26 August. Combined optimizer progress is about `54%`; scientifically no new artifact exists since the clean second boundaries, so AfriHG remains seed-42 `11/11`, confirmations `0/4`, global freeze `0/8`. These are the only owned jobs, no A100-80GB or L40S work exists, quota is `88.6%/43.0%`, Kombuys is read-only, Sheet E/F/G are blank, held-out access is `0`, and terminal confirmation compute remains the only blocker.

- At 19:04 SAST on 25 August, b7 seed-13 `1267877` completed a clean step-6164 exact callback and resumed near `6527/15410`. Its immutable 128-row artifact passes exact `64/64` Xho/Zul coverage and uniqueness, zero-empty, prompt-boundary, and generated-EOS checks. Xho/Zul chrF is `23.23858364622246/24.275093281531426`; mean `23.756838463876943` improves step 3082, so checkpoint 6164 becomes its current validation-only retained checkpoint. Artifact/trainer-state/adapter hashes are `dd46c013...89ff3`/`2aaeb8fc...cdec1`/`a2d3665d...2cd89`. A2 seed-13 `1267878` remains healthy near `7895/15410`; b7's second-boundary mean is slightly above a2's `23.683208682030707`, but this is interim validation-only evidence and no winner is frozen. Both remain `RUNNING` after `9:05:08` on separate A100-40GB `gpu:ampere:1` allocations on `srvrocgpu010`, with empty fault scans. A2 should reach step 9246 around `20:10--20:20`, b7 around `21:20--21:35`, with exact artifacts one to two hours later; terminal completion remains `04:45--08:15` on 26 August. Combined optimizer progress is about `47%`; scientifically AfriHG remains seed-42 `11/11`, confirmations `0/4`, global freeze `0/8`. These are the only owned jobs, no A100-80GB or L40S work exists, quota is `88.6%/43.0%`, Kombuys is read-only, Sheet E/F/G are blank, held-out access is `0`, and terminal confirmation compute remains the only blocker.

- At 18:03 SAST on 25 August, a2 seed-13 `1267878` completed a clean step-6164 exact callback and resumed near `6735/15410`. Its immutable 128-row artifact passes exact `64/64` Xho/Zul coverage and uniqueness, zero-empty, prompt-boundary, and generated-EOS checks. Xho/Zul chrF is `22.871254941246914/24.4951624228145`; mean `23.683208682030707` improves step 3082, so checkpoint 6164 becomes the current validation-only retained checkpoint. Artifact/trainer-state/adapter hashes are `0edaa940...cd84e`/`6a9ef5e8...e51b6`/`c010c281...436b`. B7 seed-13 `1267877` reached `6164/15410`, completed full validation at health-only loss `2.172864157684742`, and entered its frozen second exact callback at `17:16:24`; it remained active at `18:02` with normal generation progress, no artifact, and no fault marker. Both remain `RUNNING` after `8:04:34` on separate A100-40GB `gpu:ampere:1` allocations on `srvrocgpu010`. B7's artifact is expected around `18:20--18:40`; terminal completion remains `04:45--08:15` on 26 August. Combined optimizer progress is about `42%`; scientifically AfriHG remains seed-42 `11/11`, confirmations `0/4`, global freeze `0/8`. These are the only owned jobs, no A100-80GB or L40S work exists, quota is `88.6%/43.0%`, Kombuys is read-only, Sheet E/F/G are blank, held-out access is `0`, and terminal confirmation compute remains the only blocker.

- At 17:04 SAST on 25 August, a2 seed-13 `1267878` reached `6164/15410`, completed full declared validation at health-only loss `2.18823875907487`, and entered its frozen second exact-generation callback at `16:33:56`; no step-6164 artifact exists yet, so no incomplete output or health-only loss is used for selection. B7 seed-13 `1267877` remains healthy at `6062/15410`, about 102 steps from the same boundary. Both are `RUNNING` after `7:05:20` on separate A100-40GB `gpu:ampere:1` allocations on `srvrocgpu010`, with no runtime fault marker. A2's artifact is expected around `17:25--17:45`; b7 should enter its boundary around `17:08--17:12` and produce its artifact around `18:35--19:00`. The conditional terminal window remains `04:45--08:15 SAST` on 26 August. These are the only owned jobs, with no A100-80GB or L40S work. Combined optimizer progress is about `40%`; scientifically AfriHG remains seed-42 `11/11`, confirmations `0/4`, global freeze `0/8`. Quota is `88.6%/43.0%`, Kombuys is read-only, Sheet E/F/G are blank, held-out access is `0`, and terminal confirmation compute remains the only blocker.

- At 16:02 SAST on 25 August, AfriHG seed-13 confirmations b7 `1267877` and a2 `1267878` are healthy at `4892/15410` and `5651/15410`, respectively, after `6:04:05` on separate A100-40GB `gpu:ampere:1` allocations on `srvrocgpu010`. Throughput is about `3.14--3.21 s/step`; targeted fault scans are empty. A2 should reach step 6164 around `16:29--16:33 SAST` and produce its second exact artifact around `17:25--17:45`; b7 should reach the boundary around `17:08--17:15` and produce its artifact around `18:35--19:00`. The conditional terminal window remains `04:45--08:15 SAST` on 26 August. These are the only owned jobs, both A100-40GB, with no owned A100-80GB or L40S work. Combined optimizer progress is about `34%`; scientifically AfriHG remains seed-42 `11/11`, confirmations `0/4`, global freeze `0/8`. Quota is `88.6%/43.0%`, Kombuys is read-only, Sheet E/F/G are blank, held-out access is `0`, and terminal confirmation compute remains the only blocker.

- At 15:01 SAST on 25 August, b7 seed-13 `1267877` completed a clean step-3082 exact callback and resumed near `3734/15410`. Its immutable 128-row artifact passes exact `64/64` Xho/Zul coverage and uniqueness, zero-empty, prompt-boundary, and generated-EOS checks. Xho/Zul chrF is `22.957096156670843/23.350442862175072`; mean `23.153769509422958` makes checkpoint 3082 the current validation-only retained checkpoint. Artifact/trainer-state/adapter hashes are `a9db8e2c...975034`/`c7c9ed89...1eaa6e`/`b293626f...113640`. A2 seed-13 `1267878` is healthy near `4493/15410`; b7's first-boundary mean is above a2's `22.872278601860273`, but this is interim validation-only evidence and no winner is frozen. Both are the only owned jobs, both A100-40GB, with empty fault scans and no owned A100-80GB or L40S work. Combined optimizer progress is about `27%`; observed callback runtimes widen the conditional terminal window to `04:45--08:15 SAST` on 26 August. Scientifically AfriHG remains seed-42 `11/11`, confirmations `0/4`, global freeze `0/8`. Quota is `88.6%/43.0%`, Kombuys is read-only, Sheet E/F/G are blank, held-out access is `0`, and terminal confirmation compute remains the only blocker.

- At 13:59 SAST on 25 August, a2 seed-13 `1267878` completed a clean step-3082 exact callback and resumed near `3324/15410`. Its immutable 128-row artifact passes exact `64/64` Xho/Zul coverage and uniqueness, zero-empty, prompt-boundary, and generated-EOS checks. Xho/Zul chrF is `22.690663935823533/23.053893267897013`; mean `22.872278601860273` makes checkpoint 3082 the current validation-only retained checkpoint. Artifact/trainer-state/adapter hashes are `c38bfc67...eb6fc9`/`15bf56b4...a261e`/`8e76fd66...c2c0c8`. B7 seed-13 `1267877` remains healthy in its step-3082 exact callback, with log progress through `13:28:36` and no exact artifact or fault marker yet. These are the only owned jobs, both A100-40GB; no owned A100-80GB or L40S work exists. B7's artifact is expected around `14:00--14:20 SAST`, terminal artifacts around `05:30--07:00 SAST` on 26 August if uninterrupted. A2 is valid within-run evidence but not terminal confirmation, so AfriHG remains seed-42 `11/11`, confirmations `0/4`, global freeze `0/8`. Quota is `88.6%/42.9%`, Kombuys is read-only, Sheet E/F/G are blank, held-out access is `0`, and b7 callback plus terminal confirmation compute remain the blockers.

- At 12:57 SAST on 25 August, both AfriHG seed-13 confirmations entered their first frozen exact-generation callbacks. A2 `1267878` reached `3082/15410`, completed full declared validation at health-only loss `2.232720357113887`, and entered exact generation at `12:42:37`; b7 `1267877` did the same with health-only loss `2.2235934426458064` at `12:44:00`. Both remain `RUNNING` on separate A100-40GB `gpu:ampere:1` allocations on `srvrocgpu010`, and targeted fault scans are empty. No exact artifact exists yet, so no callback result or health-only loss is used for selection. These are the only owned jobs, with no owned A100-80GB or L40S work. First exact artifacts are expected around `13:40--14:10 SAST`, terminal artifacts around `05:30--07:00 SAST` on 26 August if uninterrupted. Operational optimizer progress is `20%`; scientifically AfriHG remains seed-42 `11/11`, confirmations `0/4`, global freeze `0/8`. Quota is `88.6%/42.9%`, Kombuys is read-only, Sheet E/F/G are blank, held-out access is `0`, and the active callback is the only blocker.

- At 11:57 SAST on 25 August, AfriHG seed-13 confirmations b7 `1267877` and a2 `1267878` are healthy at `2257/15410` and `2272/15410`, respectively, after `1:59:07` on separate A100-40GB `gpu:ampere:1` allocations on `srvrocgpu010`. Throughput remains about `3.06--3.12 s/step`; targeted fault scans are empty. These are the only owned jobs, with no owned A100-80GB or L40S work. First `3082` boundaries remain projected around `12:38--12:43 SAST`, exact callback artifacts around `13:45--14:45 SAST`, and terminal artifacts around `05:30--07:00 SAST` on 26 August if uninterrupted. Operational seed-13 training is about `15%`; scientifically AfriHG remains seed-42 `11/11`, confirmations `0/4`, global freeze `0/8`. Quota is `88.6%/42.9%`, Kombuys is read-only, Sheet E/F/G are blank, held-out access is `0`, and confirmation compute is the only blocker.

- At 10:56 SAST on 25 August, AfriHG seed-13 confirmations b7 `1267877` and a2 `1267878` are healthy at `1070/15410` and `1083/15410`, respectively, after `58:16` on separate A100-40GB `gpu:ampere:1` allocations on `srvrocgpu010`. Throughput is about `3.05--3.10 s/step`; targeted fault scans are empty. These are the only owned jobs, with no owned A100-80GB or L40S work. First `3082` boundaries are projected around `12:35--12:45 SAST`, exact callback artifacts around `13:45--14:45 SAST`, and terminal artifacts around `05:30--07:00 SAST` on 26 August if uninterrupted. Operational seed-13 training is about `7%`; scientifically AfriHG remains seed-42 `11/11`, confirmations `0/4`, global freeze `0/8`. Quota is `88.6%/42.9%`, Kombuys is read-only, Sheet E/F/G are blank, held-out access is `0`, and confirmation compute is the only blocker.

- At 10:00 SAST on 25 August, the prospective A100 hardware-equivalence gate failed closed. Candidate `1258898` completed `0:0`; original comparator `1258899` failed before science because a stale verifier rejected manifest arguments. Execution-only retry `1267937` used the frozen verifier (`64c8e42a...1a89`) and unchanged inputs, then exited `1:0` with artifact `8142fb8b...cfe73`: predictions and selected scores were exactly equal, but the runtime environment differed only in `pip` (`23.3.1` A100-40GB versus `26.2.1` A100-80GB). Under the preregistered any-check-fails rule, A100-80GB is excluded with no rerun or threshold relaxation. Frozen seed-13 confirmations b7 `1267877` and a2 `1267878` were released from the now-obsolete failed dependency and both started at `09:57:54 SAST` on separate A100-40GB devices on `srvrocgpu010`; manifests `966088ea...12fa`/`8b829180...a99` pass. At `10:01 SAST`, both were healthy at step `27/15410` near `3.15 s/step` with no fault marker. First callbacks are expected around `13:30--14:30 SAST`, terminal artifacts around `05:30--07:00 SAST` on 26 August if uninterrupted. AfriHG remains seed-42 `11/11`, confirmations `0/4`, global freeze `0/8`; quota `88.6%/42.9%`, Kombuys read-only, Sheet E/F/G blank, held-out `0`, and these are the only two owned jobs, both A100-40GB.

- At 07:59 SAST on 25 August, AfriHG seed-42 closed scientifically at `11/11`. B7 `1262543` completed `0:0` with a clean terminal artifact and Xho/Zul chrF `23.814926201664736/26.355581419154543` (mean `25.08525381040964`); terminal checkpoint 15410 is retained using validation evidence only. Verifier `1267874` proved exact equality across 424 keys and `76,410,112` values. Artifact/state and retained/final hashes are `845187a0...c857d77`/`6779f1ef...1adb4e1` and `9dfc09f8...af236bf`/`450e4163...7dbfbc`. Ranking job `1267875` froze b7 and a2 as Stage-C candidates at `25.08525381040964/24.910865619073284`; artifact `sallm_memory/artifacts/2026-08-25-pure-gdn-afrihg-seed42-ranking.json` has SHA-256 `b252ddcc...5f70df`. Seed-13 confirmation jobs `1267877/1267878` are dependency-pending after comparator `1258899`; gate `1258898` is `AssocGrpGRES`-pending with estimate `21:47:49 SAST`. Exactly three GPU jobs are owned and the chain prevents family overlap. No GPU active now; quota `88.6%/42.5%`, Kombuys read-only, Sheet E/F/G blank, held-out `0`, AfriHG confirmations `0/4`, global freeze `0/8`.

- At 06:58 SAST on 25 August, b6 `1262542` is scientifically terminal-valid after completion `0:0`, a clean 128-row terminal artifact, terminal Xho/Zul chrF `21.906310485665017/22.74221693861099` (mean `22.324263712138006`), and CPU verifier `1267864` proving exact retained-to-final equality across 424 keys and `71,762,560` values. Terminal checkpoint 15410 is retained using validation evidence only. Artifact/trainer-state hashes are `76be85b2...adf4f5a`/`1a957eff...0a7ea7`; retained/final hashes are `fed9fd82...8e19522`/`bdae26c8...84c40a`. Failed verifier `1267863` is preserved as a zero-second missing-venv-path launcher failure. AfriHG advances to `10/11`. B7 `1262543` has reached `15410/15410`, completed full validation at health-only loss `2.317962587073126`, and is healthy in its frozen terminal exact callback. Gate `1258898` and comparator `1258899` remain dependency-safe. Quota `88.6%/42.5%`; only A100-40GB active, no A100-80GB or L40S overlap, Kombuys read-only, Sheet E/F/G blank, held-out `0`, global freeze `0/8`.

- At 05:52 SAST on 25 August, b6 `1262542` remained active in its frozen terminal exact callback after `15410/15410`; its log advanced through `05:26:51` with normal generation warnings and no fault marker, so this is not a stall. No terminal artifact exists yet. B7 `1262543` remains healthy near `14425/15410` (`93.6%`) with about 52 minutes of displayed training remaining before terminal validation/callback. The terminal window is roughly `06:30--08:30 SAST`, callback-dependent. Gate `1258898` and comparator `1258899` remain dependency-safe. Quota `88.6%/42.4%`; only A100-40GB active, no A100-80GB or L40S overlap, Kombuys read-only, Sheet E/F/G blank, held-out `0`, AfriHG `9/11`, global freeze `0/8`.

- At 04:52 SAST on 25 August, b7 `1262543` completed a clean step-12328 exact callback and resumed near `13283/15410`. Its 128-row artifact passes exact coverage, uniqueness, zero-empty, prompt-boundary, and generated-EOS checks. Xho/Zul chrF is `23.826584265027677/26.270991275007738`; mean `25.048787770017707` improves step 9246, so checkpoint 12328 becomes retained using validation evidence only. Artifact/trainer-state hashes are `f5c261d0...689a3e`/`98153bf3...00ebb2`. B6 `1262542` reached `15410/15410`, completed terminal validation at health-only loss `2.276732130997526`, and entered its frozen terminal exact callback; no terminal artifact exists yet, so AfriHG remains `9/11` terminal-valid. Gate `1258898` and comparator `1258899` remain dependency-safe. Quota `88.6%/42.4%`; only A100-40GB active, no A100-80GB or L40S overlap, Kombuys read-only, Sheet E/F/G blank, held-out `0`, global freeze `0/8`.

- At 03:52 SAST on 25 August, b7 `1262543` reached `12328/15410`, completed declared validation at health-only loss `2.2675734513150956`, and entered its frozen step-12328 exact callback. No exact artifact exists yet, so no metric or checkpoint decision is available. B6 `1262542` remains healthy near `14399/15410` (`93.4%`); targeted fault scans are empty. The terminal window is roughly `06:00--08:00 SAST`, callback-dependent. Gate `1258898` and comparator `1258899` remain dependency-safe. Quota `88.6%/42.4%`; only A100-40GB active, no A100-80GB or L40S overlap, Kombuys read-only, Sheet E/F/G blank, held-out `0`, AfriHG `9/11`, global freeze `0/8`.

- At 02:51 SAST on 25 August, b6 `1262542` completed a clean step-12328 exact callback and resumed near `13219/15410`. Its 128-row artifact passes exact `64/64` Xho/Zul coverage and uniqueness, zero-empty, prompt-boundary, and generated-EOS checks. Xho/Zul chrF is `21.830004875620137/22.782335134481`; mean `22.306170005050568` improves step 9246, so checkpoint 12328 becomes retained using validation evidence only. Artifact/trainer-state hashes are `e915cd82...242c50`/`4c4660ea...56cbf`. B7 `1262543` remains healthy near `12240/15410`; gate `1258898` and comparator `1258899` remain blocked in order. Quota `88.6%/42.4%`; only A100-40GB active, no A100-80GB or L40S overlap, Kombuys read-only, Sheet E/F/G blank, held-out `0`, AfriHG `9/11`, freeze `0/8`.

- At 01:50 SAST on 25 August, b6 `1262542` reached `12328/15410`, completed declared validation at health-only loss `2.277246473981992`, and entered its frozen step-12328 exact callback. No exact artifact exists yet, so no metric or checkpoint decision is available. B7 `1262543` remains healthy near `11078/15410`; targeted fault scans are empty. Gate `1258898` and comparator `1258899` remain dependency-safe. Quota `88.6%/42.4%`; only A100-40GB active, no A100-80GB or L40S overlap, Kombuys read-only, Sheet E/F/G blank, held-out `0`, AfriHG `9/11`, global freeze `0/8`.

- At 00:49 SAST on 25 August, b7 `1262543` completed a clean step-9246 exact callback and resumed near `9918/15410`. Its 128-row artifact passes exact `64/64` Xho/Zul coverage and uniqueness, zero-empty, prompt-boundary, and generated-EOS checks. Xho/Zul chrF is `24.154545975888247/25.85751534959767`; mean `25.00603066274296` improves step 6164, so checkpoint 9246 becomes retained using validation evidence only. Artifact/trainer-state hashes are `bb3a7f7f...1b342`/`78c213ca...bdd11`. B6 `1262542` remains healthy near `12296/15410`, just before its step-12328 boundary; gate `1258898` and comparator `1258899` remain blocked in order. Quota `88.6%/42.4%`; only A100-40GB active, no A100-80GB or L40S overlap, Kombuys read-only, Sheet E/F/G blank, held-out `0`, AfriHG `9/11`, freeze `0/8`.

- At 23:46 SAST on 24 August, b7 `1262543` reached `9246/15410`, completed declared validation at health-only loss `2.2039773533825437`, and entered its frozen step-9246 exact callback. No exact artifact exists yet, so no metric or checkpoint decision is available. B6 `1262542` remains healthy near `11080/15410`; targeted fault scans are empty. Gate `1258898` and comparator `1258899` remain dependency-safe. Quota `88.6%/42.4%`; only A100-40GB active, no A100-80GB or L40S overlap, Kombuys read-only, Sheet E/F/G blank, held-out `0`, AfriHG `9/11`, global freeze `0/8`.

- At 22:50 SAST on 24 August, the user approved continuing the frozen pure-GDN anchor-plus-Sobol validation HPO and declined a midstream switch to the historical xLSTM Bayesian process. This changes no candidate, search space, checkpoint, prompt, retry, seed, or hardware rule. Report future comparisons as best observed performance under each architecture's documented validation-selected tuning process, disclosing method and budget, rather than as a strict equal-HPO-budget causal comparison. B6 `1262542` and b7 `1262543` remain healthy near `10025/15410` and `8986/15410`; held-out access `0`, Sheet E/F/G blank, AfriHG `9/11`, global freeze `0/8`.

- At 22:46 SAST on 24 August, b6 `1262542` completed a clean step-9246 exact callback and resumed near `9948/15410`. Its 128-row artifact passes exact `64/64` Xho/Zul coverage and uniqueness, zero-empty, prompt-boundary, and generated-EOS checks. Xho/Zul chrF is `21.828465231626993/22.32754911383624`; mean `22.078007172731617` improves step 3082, so checkpoint 9246 becomes retained using validation evidence only. Artifact/trainer-state hashes are `3c42fa22...3b9ef`/`efc063f9...534e5`. B7 `1262543` remains healthy near `8900/15410`; gate `1258898` and comparator `1258899` remain blocked in order. Quota `88.6%/42.4%`; only A100-40GB active, no A100-80GB or L40S overlap, Kombuys read-only, Sheet E/F/G blank, held-out `0`, AfriHG `9/11`, freeze `0/8`.

- At 20:59 SAST on 24 August, b6 `1262542` reached `9246/15410` (`60.0%`)
  and entered its third frozen validation boundary; b7 `1262543` remained
  healthy near `6877/15410` (`44.6%`). Together the two remaining AfriHG
  seed-42 trials have completed `52.3%` of their optimizer steps, with empty
  targeted fault scans. AfriHG remains `9/11` terminal-valid; after b6/b7,
  hardware gate `1258898`, comparator `1258899`, and four confirmations
  remain before winner freeze. Quota `88.6%/42.5%`; only A100-40GB active,
  no A100-80GB or L40S overlap, Kombuys read-only, Sheet E/F/G blank,
  held-out `0`, global freeze `0/8`.

- At 20:45 SAST on 24 August, b7 `1262543` completed a clean step-6164 exact
  callback and resumed near `6603/15410`. Its immutable 128-row artifact
  passes exact `64/64` Xho/Zul coverage and uniqueness, zero-empty, prompt-
  boundary, and generated-EOS checks. Xho/Zul chrF is
  `23.435519836035333/25.083127573554826`; mean `24.25932370479508` improves
  step 3082, so checkpoint 6164 becomes retained using validation evidence
  only. Artifact/trainer-state hashes are
  `d09e81b6...f10048`/`3630238e...c9667d4`. Concurrent b6 `1262542` remains
  healthy near `8996/15410`; gate `1258898` and comparator `1258899` remain
  blocked in order. Quota `88.6%/42.5%`; only A100-40GB active, no A100-80GB
  or L40S overlap, Kombuys read-only, Sheet E/F/G blank, held-out `0`, AfriHG
  `9/11`, freeze `0/8`.

- At 19:42 SAST on 24 August, b7 `1262543` reached `6164/15410`, completed
  declared validation at health-only loss `2.1724466154282935`, and entered
  its frozen second exact-generation callback. Its exact artifact is not yet
  present, so no new validation metric is available or used. Concurrent b6
  `1262542` remains healthy near `7787/15410` at about `3.12 s/step`; neither
  run has a runtime fault marker. Gate `1258898` remains dependency-blocked
  on both, followed by comparator `1258899`. Quota `88.6%/42.5%`; only
  A100-40GB active, no A100-80GB or L40S overlap, Kombuys read-only, Sheet
  E/F/G blank, held-out `0`, AfriHG `9/11`, freeze `0/8`.

- At 18:42 SAST on 24 August, b6 `1262542` completed a clean step-6164 exact
  callback and resumed near `6654/15410`. Its immutable 128-row artifact
  passes exact `64/64` Xho/Zul coverage and uniqueness, zero-empty, prompt-
  boundary, and generated-EOS checks. Xho/Zul chrF is
  `20.629001232654016/21.55695707470458`; mean `21.0929791536793` is below
  step 3082 mean `21.270276801059854`, so checkpoint 3082 remains retained.
  Artifact/trainer-state hashes are `83d4c9e8...09b53`/`36a56215...9cf46`.
  Concurrent b7 `1262543` remains healthy near `5568/15410`; gate `1258898`
  and comparator `1258899` remain blocked in order. Quota `88.6%/42.5%`;
  only A100-40GB active, no A100-80GB or L40S overlap, Kombuys read-only,
  Sheet E/F/G blank, held-out `0`, AfriHG `9/11`, freeze `0/8`.

- At 17:28 SAST on 24 August, b6 `1262542` reached `6164/15410`, completed
  declared validation at health-only loss `2.306739406102977`, and entered
  its frozen second exact-generation callback. The callback was active
  through `17:24:44` with only normal context-truncation warnings and no
  runtime fault marker; its exact step-6164 artifact was still absent, so no
  new validation metric is available or used. Concurrent b7 `1262543`
  remains healthy near `4166/15410` at about `3.15 s/step`. Gate `1258898`
  remains dependency-blocked on both jobs, followed by comparator `1258899`.
  Quota `88.6%/42.4%`; only A100-40GB active, no A100-80GB or L40S overlap,
  Kombuys read-only, Sheet E/F/G blank, held-out `0`, AfriHG `9/11`, freeze
  `0/8`.

- At 16:40 SAST on 24 August, b7 `1262543` completed a clean step-3082 exact
  callback and resumed near `3250/15410`. Its 128-row artifact passes exact
  Xho/Zul coverage and uniqueness, zero-empty, prompt-boundary, and generated-
  EOS checks. Xho/Zul chrF is `22.39658842179247/23.98035435684408`; mean
  `23.188471389318273` exactly matches retained checkpoint 3082. Artifact/
  trainer-state hashes are `c2d9a125...b58f2f`/`4c2be62b...f7d510`. This is
  validation-only within-run evidence. Concurrent b6 `1262542` remains
  healthy near `5698/15410`; gate `1258898` and comparator `1258899` remain
  blocked in order. Quota `88.6%/42.4%`; only A100-40GB active, no L40S,
  Kombuys read-only, Sheet E/F/G blank, held-out `0`, AfriHG `9/11`, freeze
  `0/8`.

- At 15:37 SAST on 24 August, b7 `1262543` reached `3082/15410`, completed
  declared validation at health-only loss `2.223826493171033`, and entered
  its first frozen exact-generation callback without a fault marker. No exact
  artifact exists yet; it is tentatively due `16:20--16:35`. B6 `1262542`
  remains healthy near `4497/15410`, with its next exact artifact roughly
  `18:00--18:20`. Gate `1258898` remains blocked on both and comparator
  `1258899` remains after it. Quota `88.6%/42.4%`; only A100-40GB active, no
  L40S, Kombuys read-only, Sheet E/F/G blank, held-out `0`, AfriHG `9/11`,
  freeze `0/8`.

- At 14:36 SAST on 24 August, b6 `1262542` completed a clean step-3082 exact
  callback and resumed near `3342/15410`. Its 128-row artifact passes exact
  `64/64` Xho/Zul coverage and uniqueness, zero-empty, prompt-boundary, and
  generated-EOS checks. Xho/Zul chrF is
  `21.216082571156228/21.32447103096348`; mean `21.270276801059854` exactly
  matches retained checkpoint 3082. Artifact/trainer-state hashes are
  `6d5cf744...4ef0e5`/`df2b9fc4...134871`. This is validation-only within-run
  evidence, not terminal status. Concurrent b7 `1262543` is healthy near
  `2291/15410`; gate `1258898` and comparator `1258899` remain blocked in
  order. Quota `88.6%/42.4%`; only A100-40GB active, no L40S, Kombuys read-
  only, Sheet E/F/G blank, held-out `0`, AfriHG `9/11`, freeze `0/8`.

- At 13:36 SAST on 24 August, b6 `1262542` reached `3082/15410`, completed
  declared validation at health-only loss `2.4353304596244323`, and entered
  the frozen exact-generation callback without a fault marker. No step-3082
  exact artifact exists yet, so no metric decision was made; it is tentatively
  due around `14:10--14:30`. Concurrent b7 `1262543` is healthy near
  `1109/15410` at about `3.11 s/step`, with its first exact artifact roughly
  `16:15--16:45`. Gate `1258898` remains blocked on both and comparator
  `1258899` remains after it. Quota `88.6%/42.3%`; only A100-40GB is active,
  no L40S, Kombuys read-only, Sheet E/F/G blank, held-out `0`, AfriHG `9/11`,
  global freeze `0/8`.

- At 12:36 SAST on 24 August, available same-family capacity was used without
  changing science. A prospective amendment was frozen at SHA-256
  `5c4490b9...25c7af` before edits, with b7 roots absent and no candidate or
  held-out metric inspected. B6 `1262542` remains healthy near `2266/15410`;
  b7 `1262543` started at `12:35:59` on a second A100-40GB, wrote execution-
  manifest SHA-256 `ab0a2d96...c45c79`, verified all `694/694` files, passed
  fast-GDN, and loaded the canonical model with exact b7 trainable parameter
  count `9,296,640`.
  A100-80GB gate `1258898` now depends on successful completion of both runs,
  followed by CPU comparator `1258899`, so no A100-family overlap is possible.
  Owned GPU jobs are exactly three; no L40S, Kombuys read-only, quota
  `88.6%/42.2%`, Sheet E/F/G blank, held-out `0`, AfriHG `9/11`, freeze `0/8`.

- At 11:34 SAST on 24 August, corrected AfriHG b6 `1262542` remains healthy
  on A100-40GB `srvrocgpu010`, near `1053/15410` at about `2.97--3.00 s/step`
  with no runtime fault marker. Its first frozen step-3082 exact artifact is
  tentatively due around `14:00 SAST`; terminal verification remains roughly
  early 25 August. B7 `1262543`, A100-80GB gate `1258898`, and CPU comparator
  `1258899` remain strictly dependency-serial. Quota is `88.6%/42.2%`; no
  A100-family overlap or L40S is present, Kombuys remains read-only, Sheet
  E/F/G blank, held-out access `0`, AfriHG `9/11`, and global freeze `0/8`.

- At 10:40 SAST on 24 August, stale-launcher b6 job `1258900` was preserved
  after failing `1:0` in zero seconds before model/data/evaluator access; its
  only output is a 157-byte missing-`scripts/hpo_protocol.py` log with SHA-256
  `76ca9c0c...de94f`, and candidate roots stayed absent. A prospective
  execution-only correction was frozen at SHA-256 `e5c08c40...4ed63b` after
  the immutable sidecar, all `694/694` source hashes, registry hash, and
  absent b6/b7 roots passed. Corrected unchanged b6 `1262542` started at
  `10:37:56 SAST` on A100-40GB `srvrocgpu010`, verified all `694/694`
  immutable files, wrote execution-manifest SHA-256 `94a1587e...a3bb6b`,
  passed fast-GDN, and loaded the canonical model cleanly. B7 `1262543` and
  A100-80GB gate `1258898` remain strictly serial after it, followed by CPU
  comparator `1258899`. No A100-family overlap, candidate metric, or held-out
  access occurred. AfriHG remains `9/11`, global freeze `0/8`, Sheet E/F/G
  blank, quota `88.6%/42.0%`, no L40S, and Kombuys read-only.

- At 09:37 SAST on 24 August, existing unchanged A100-40GB AfriHG b6 job
  `1258900` was safely released ahead of the capacity-only A100-80GB gate
  after Slurm moved gate candidate `1258898` to an estimated
  `2026-09-01 17:31:43 SAST` start. The prospective execution-only amendment
  was frozen at SHA-256 `e8dbf98e...5c83d` before edits and uses no candidate
  or held-out metric. B6 is now eligible and `Resources`-pending with an
  absent output root; `1258898` is dependency-pending after b6 and comparator
  `1258899` remains after the gate. No A100-family overlap occurred. All
  approved A100-40GB devices remain occupied by other users, quota is
  `88.6%/42.0%`, no L40S is used, Kombuys remains read-only, held-out access
  is `0`, global freeze is `0/8`, and Sheet E/F/G remain blank.

- At 05:32 SAST on 24 August, AfriHG b5 `1258485` is scientifically
  terminal-valid after completion `0:0`, a clean 128-row terminal exact
  artifact (Xho/Zul chrF `23.320999482462366/23.993243166470638`, mean
  `23.657121324466502`, SHA-256 `880bf0b4...3b5ba`), retained checkpoint
  12328 at validation-only mean `23.750084550710223`, and CPU verifier
  `1261967` proving exact equality for all 424 retained/final tensor keys and
  71,762,560 values. AfriHG advances to `9/11` terminal-valid. Gate candidate
  `1258898` is now `AssocGrpGRES`-pending with Slurm estimate
  `2026-08-25 21:47:49 SAST`; all approved A100-80GB and A100-40GB devices
  are occupied by other users, so no approved capacity is being left idle.
  Comparator `1258899` and b6 `1258900` remain dependency-pending. Quota is
  `88.6%/42.0%`, no L40S is used, Kombuys remains read-only, held-out access
  is `0`, global freeze is `0/8`, and Sheet E/F/G remain blank.

- At 04:30 SAST on 24 August, isolated AfriHG b5 `1258485` completed exactly
  `15410/15410` training steps and full declared terminal validation at
  health-only loss `2.2310394148483126`, then entered its frozen terminal
  exact-generation callback. It remains healthy on `srvrocgpu010` A100-40GB
  with no fault marker, but the terminal artifact does not exist yet; AfriHG
  therefore remains `8/11` terminal-valid. Serial jobs
  `1258898 -> 1258899 -> 1258900` remain dependency-pending. Quota is
  `88.6%/41.9%`, no L40S is used, Kombuys remains read-only, held-out access
  is `0`, global freeze is `0/8`, and Sheet E/F/G remain blank.

- At 01:26 SAST on 24 August, isolated AfriHG b5 `1258485` completed its
  clean step-12328 exact callback and resumed healthy A100-40GB training
  beyond step 12929/15410. The 128-row artifact passes exact `64/64` Xho/Zul
  coverage and uniqueness, zero-empty, and prompt-boundary checks. Xho/Zul
  chrF is `23.192972608788125/24.307196492632322`; registered mean
  `23.750084550710223` improves step 9246 and matches retained checkpoint
  12328. Artifact/trainer-state SHA-256 values are `ca46edf3...77c2bde` and
  `d45e3874...0a8b16`. This remains validation-only within-run evidence:
  AfriHG is `8/11` terminal-valid, global freeze `0/8`, held-out `0`, and
  Sheet E/F/G blank. Terminal verification is tentatively due around
  04:30--05:15 SAST; serial jobs `1258898 -> 1258899 -> 1258900` remain
  dependency-pending. Quota is `88.6%/41.9%`, no L40S is used, and Kombuys
  remains read-only.

- At 21:46 SAST on 23 August, a live capacity audit found no avoidable idle
  approved compute. B5 `1258485` was actively training at step 9985/15410;
  all four `gpu:ampere` devices and all four `gpu:ampere80` devices were
  allocated across cluster jobs. The dependency chain `1258485 -> 1258898 ->
  1258899 -> 1258900` already removes manual gate/b6 hand-off delay. Four idle
  `gpu:amperemk` devices remain inaccessible to the current association; the
  prepared support request is the only plausible capacity-side acceleration
  and remains unsent pending explicit approval. The observed low-utilization
  training/evaluation shape and exact callbacks are frozen comparability
  costs, not scheduler idle. No L40S or Kombuys work was started.

- At 21:18 SAST on 23 August, isolated AfriHG b5 `1258485` is healthy beyond
  step 9405/15410 on `srvrocgpu010` A100-40GB after clean exact callbacks at
  steps 6164 and 9246. Both 128-row artifacts pass full `64/64` Xho/Zul
  coverage and uniqueness, zero-empty, and prompt-boundary checks. Registered
  mean chrF improved from `21.770125040860544` at step 6164 to
  `23.38957146182795` at step 9246, so checkpoint 9246 is the current
  validation-only within-run best. Artifact SHA-256 values are
  `f7887231...40e10f8` and `dba75b13...036c78`; trainer-state SHA-256 is
  `ee5e84b3...e71a05`. B5 is not terminal and is tentatively due around
  05:00--06:00 SAST on 24 August. The strict serial chain remains b5
  `1258485` -> A100-80 gate candidate `1258898` -> CPU comparator `1258899`
  -> A100-40 b6 `1258900`. Quota is `88.6%/41.9%`; no L40S is used, Kombuys
  remains read-only, AfriHG remains `8/11` terminal-valid, global freeze is
  `0/8`, held-out access is `0`, and Sheet E/F/G are blank.
  Live metadata now names target sheetId `202608060` `GDN Results`, not
  `Pure GatedDeltaNet Results`, and exposes multiple visible tabs; the stable
  sheetId remains the authoritative target. Read-only checks confirmed E2:G43
  blank and quarantined rows 41--43 confined to column D. No Sheet write was
  made.

- At 14:01 SAST on 23 August, isolated AfriHG b5 `1258485` completed its
  first exact validation callback at step 3082 and resumed healthy training
  beyond step 3498. The 128-row Xho/Zul artifact is clean (64/64 coverage and
  unique predictions per language, no empties, correct boundaries), with mean
  chrF `21.685455872673925` and SHA-256 `38f0591c...718850`. This is
  validation-only within-run evidence, not a terminal result; AfriHG remains
  `8/11` terminal-valid, global freeze `0/8`, held-out access `0`, and Sheet
  E/F/G blank. The serial hardware-gate/b6 chain remains dependency-pending.

- At 09:47 SAST on 23 August, AfriHG b4 `1253374` is scientifically
  terminal-valid after completion `0:0`, a clean 128-row terminal exact
  artifact (mean chrF `23.919296880215992`, SHA-256 `bd52036d...573f24`),
  retained validation-only winner checkpoint 12328 (mean chrF
  `23.9692081245192`), and CPU verifier `1258498` proving exact equality for
  all 424 retained/final tensor keys and 69,438,784 values. AfriHG advances
  to `8/11` terminal-valid. Isolated A100-40GB b5 `1258485` is the sole
  running owned GPU job on `srvrocgpu010`, healthy beyond step 80/15410 at
  about 3.1 seconds per step. The final race-free chain is queued exactly as
  A100-80GB gate candidate `1258898` after b5, CPU comparator `1258899`, then
  mandatory A100-40GB b6 `1258900`; only successful dependencies release the
  next job. Quota is `88.6%/41.9%`, no L40S is used, Kombuys remains
  read-only, held-out access is `0`, Sheet E/F/G are blank, and global freeze
  remains `0/8`.

- At 09:40 SAST on 23 August, corrected A100-80GB candidate `1258453`
  unexpectedly ran from `09:36:18--09:37:21` while A100-40GB b4 `1253374`
  remained active. Its contents were not inspected; it is quarantined as an
  ordering failure and cannot establish hardware equivalence. Isolated
  A100-40GB b5 job `1258485` was submitted dependency-pending after b4, but
  remains unstarted. A prospective amendment freezes one isolated A100-80GB
  candidate replacement after b4, then a fail-closed CPU comparator, then b5
  only after comparator success. Held-out access remains `0`, Sheet E/F/G are
  blank, and global freeze is `0/8`.

- At 09:35 SAST on 23 August, an isolated b5 dry-run preflight failed closed
  before submission because the immutable wrapper ignored newer
  run/output/logging override variables. It rewrote exactly the three metadata
  files in the already-quarantined default b5 root; preserved partial
  checkpoint tensors were not modified. The default root remains quarantined
  and no b5 job was submitted. The frozen recovery is to invoke the same
  immutable scientific runner directly with registry-resolved variables,
  prove an isolated dry-run manifest first, and preserve all failure
  provenance. B4 `1253374` remains in terminal exact generation; A100-80GB
  gate candidate `1258453` remains capacity-blocked; held-out access is `0`
  and global freeze is `0/8`.

- At 09:01 SAST on 23 August, AfriHG b4 `1253374` had reached exactly
  `15410/15410`, completed full declared terminal validation at health-only
  loss `2.2331547721649905`, and entered its frozen exact-generation callback.
  No terminal artifact or fault marker exists yet; checkpoint 12328 remains
  the non-terminal validation-only winner at mean chrF `23.9692081245192`.
  Terminal output is tentatively due around `09:30--09:50 SAST`. Corrected
  A100-80GB gate candidate `1258453` remains `AssocGrpGRES`-pending with a
  `2026-08-24 22:17:48 SAST` scheduler estimate. Quota is `88.6%/41.7%`; no
  L40S is used, Kombuys remains read-only, held-out access is `0`, Sheet E/F/G
  are blank, and global freeze is `0/8`.

- At 07:31 SAST on 23 August, corrected A100-40GB gate reference `1258452`
  completed `0:0` in `00:01:00` and recorded the required
  `slurm_job_gres="gpu:ampere:1"`. Its result/execution-manifest SHA-256
  values are `bb9fbf65...76b47` and `0b11f970...07e0`; this is operational
  completion only, not a passing hardware gate. Corrected A100-80GB candidate
  `1258453` is now `AssocGrpGRES`-pending because another user occupies all
  four A100-80GB devices; Slurm estimates `2026-08-24 22:17:48 SAST`.
  AfriHG b4 `1253374` remains healthy beyond `14195/15410` on A100-40GB,
  with terminal verification tentatively due around `10:00--10:30 SAST`.
  Quota is `88.6%/41.7%`; no L40S is used, Kombuys remains read-only,
  held-out access is `0`, Sheet E/F/G are blank, and global freeze is `0/8`.

- At 06:08 SAST on 23 August, AfriHG b4 `1253374` completed its step-12328
  exact validation callback and resumed beyond step `12451/15410`. Its clean
  `128`-row artifact has Xho/Zul chrF
  `23.322881784975422/24.61553446406298`, mean `23.9692081245192`, making
  checkpoint 12328 the current within-run best. Artifact/trainer-state
  SHA-256 values are `4823dd05...b7f26` and `867ecef5...2384`. This remains
  non-terminal validation-only evidence. Corrected A100 gate jobs
  `1258452/1258453` remain Priority/Dependency-pending; A100-80GB remains
  unratified, held-out access `0`, and global freeze `0/8`.

- At 05:35 SAST on 23 August, the original paired A100 gate failed closed on
  metadata instrumentation only. Reference `1257517` completed `0:0`; frozen
  comparison with sequential candidate `1257520` passed every scientific,
  artifact, manifest, runtime, prediction, metric, and score-difference check
  (maximum/mean score difference both `0.0`) but failed
  `hardware_pair_match` because both artifacts recorded
  `slurm_job_gres=null`. Verification SHA-256 is `e0aa48c...42edc6`.
  A100-80GB remains unratified and sealed b5 `1257792` must be cancelled and
  quarantined. It was cancelled at `05:33:46` after `06:38:26`, preserving
  `16` partial files without inspecting their contents. A dated correction
  protocol freezes a single sequential rerun pair that changes only explicit
  process-level GRES metadata; thresholds, data, code, and held-out boundary
  remain unchanged. Corrected reference `1258452` is Priority-pending and
  corrected candidate `1258453` is dependency-pending, alongside running b4
  `1253374`; the three-job cap is full and no L40S is in use.

- At 02:33 SAST on 23 August, AfriHG b4 `1253374` completed its step-9246
  exact validation callback and resumed beyond step `9591/15410`. The exact
  `128`-row artifact has complete `64/64` Xho/Zul coverage, zero empty
  predictions, `64/64` unique predictions per language, clean prompt
  boundaries, and Xho/Zul chrF `23.331670222361144/23.887552144959805`
  (mean `23.609611183660476`). Checkpoint 9246 is the within-run best;
  artifact/trainer-state SHA-256 values are `d4772261...ab941d` and
  `7baaf968...dbc3c`. This remains non-terminal validation-only evidence.
  Sealed A100-80GB b5 `1257792` is running without result inspection;
  A100-40GB reference `1257517` remains Priority-pending. AfriHG stays
  `7/11` terminal-valid, global freeze `0/8`, held-out adapter access `0`,
  and Sheet E/F/G blank.

- At 22:55 SAST on 22 August, user-approved sealed A100-80GB execution began.
  Equivalence replacement `1257520` completed `0:0` in `00:01:04`; its result
  was not inspected and remains sealed at SHA-256
  `d90f1a645716a46eb82eb5972bb7422d4fc1e9e8042ac09c270858b9dd38ae98`
  pending paired comparison with A100-40GB reference `1257517`. Exact frozen
  AfriHG b5 was submitted once as A100-80GB job `1257792` from the immutable
  HPO snapshot and started on `srvrocgpu011` at `22:55:20` SAST. B4
  `1253374` remains the sole running owned A100-40GB job; `1257517` is
  pending, no L40S is in use, and the three-job cap is full. Scientifically
  AfriHG remains `7/11` terminal-valid, global freeze `0/8`, held-out adapter
  access `0`, and Sheet E/F/G blank.

- At 21:00 SAST on 22 August, a read-only queue audit ruled out a local
  wall-time fix for A100 equivalence reference `1257517`: exact `sbatch
  --test-only` probes at 10 minutes, 30 minutes, 2 hours, and 24 hours all
  projected the same `2026-08-24 08:23:59 SAST` start. The job is blocked by
  Priority, not chiefly by its 24-hour request. Both owned A100 associations
  reject idle `gpu:amperemk` capacity with `AssocGrpGRES`, so there is no
  self-service route to the four idle A100-40GB devices; scheduler/support
  action is required. No job was changed. B4 `1253374` remains healthy near
  `5733/15410`, replacement `1257520` remains dependency-pending, quota is
  `88.6%/41.5%`, no L40S work exists, Kombuys remains read-only, held-out
  access is `0`, global freeze is `0/8`, and Sheet E/F/G remain blank.

- At 20:41 SAST on 22 August, the user-approved A100 hardware-equivalence gate
  is queued without weakening any scientific grid. B5/b6 `1257468/1257469`
  were confirmed never-started with absent outputs and cancelled to free the
  slots. B4 `1253374` remains healthy beyond `5347/15410`; A100-40GB reference
  `1257517` is Priority-pending. A first A100-80GB canary `1257518` completed
  before cancellation while b4 was active, so its uninspected result is
  preserved but quarantined under the user's no-pre-pass-overlap rule.
  Replacement `1257520` is dependency-gated after b4 ends and `1257517`
  succeeds. The new 702-file read-only snapshot has full-manifest SHA-256
  `ccef0a5174067d3db485ec5bcc4912e82df1c42f41a613f893f4d6068e7b31d4`.
  Independent review found and closed one comparator gap: the read-only
  verification script now requires exact source/model/runtime manifest
  identity at SHA-256
  `64c8e42aeba1e32fed011e901cd2ebecf677190e75d628f984b046c979f81a89`.
  Owned work is exactly three jobs; quota is `88.6%/41.5%`, no L40S work
  exists, Kombuys remains read-only, held-out access is `0`, global freeze is
  `0/8`, and Sheet E/F/G remain blank.

- At 18:55 SAST on 22 August, AfriHG b4 `1253374` completed its first exact
  generation callback and resumed beyond step `3322/15410`. The step-3082
  validation-only artifact has `128` rows, exact `64/64` Xho/Zul coverage,
  zero empty predictions, `128/128` unique predictions, and Xho/Zul chrF
  `21.277088395946436/22.53969039418806` (mean `21.908389395067248`). Its
  SHA-256 is
  `72999ffbce1405585374c8aa54698ebd31694dd65a640254728e653d48443e96`.
  B5/b6 `1257468/1257469` remain Priority-pending; owned work is exactly three
  A100-40GB jobs. Quota is `88.6%/41.5%`, Kombuys remains read-only, held-out
  access is `0`, global freeze is `0/8`, and Sheet E/F/G remain blank.

- At 17:52 SAST on 22 August, CPU-only exact verifier `1257467` completed
  `0:0` in four seconds and proved exact retained/final tensor equality for
  AfriHG b2 and b3. Its log SHA-256 is
  `6cee58f46d85859c4eee30a0704532469b54a6f5dda15f29a6e12d84bd1d626d`;
  AfriHG scientific progress advances from `5/11` to `7/11`. Never-started
  GPU verifier jobs `1253806/1256680` remain preserved as cancelled
  provenance. B4 `1253374` is healthy in its epoch-one exact-generation
  callback after `eval_loss=2.3000728302695084`. With two owned slots free,
  absent outputs, no duplicates, and frozen registry plus dry runs verified,
  b5/b6 were submitted once as jobs `1257468/1257469`; both are
  Priority-pending on the required `gpu:ampere` A100-40GB queue. Owned work is
  exactly three jobs. A fresh `gpu:amperemk` test-only request still fails
  `AssocGrpGRES`, A100-80GB remains excluded, quota is `88.6%/41.4%`, Kombuys
  remains read-only, held-out access is `0`, global freeze is `0/8`, and Sheet
  E/F/G remain blank.

- At 17:31 SAST on 22 August, AfriHG b4 `1253374` was healthy near
  `3075/15410`; verifiers `1253806/1256680` remained Priority-pending.
  After b4, AfriHG still needs b5--b7 plus four confirmations, approximately
  `133` GPU-hours at observed runtime. Corrected General still needs eleven
  seed-42 trials plus four confirmations, provisionally about `270`
  GPU-hours. One continuously available eligible GPU therefore implies about
  `17` days after b4 just to close these Multilingual gates; three eligible
  GPUs imply roughly six days. Live capacity offers no second allowed device:
  all four `gpu:ampere` GPUs on `srvrocgpu010` are occupied, while idle
  `amperemk` and `ampere80` devices are excluded by the frozen A100-40GB
  protocol. Monolingual selection remains unstarted, held-out access remains
  `0`, Sheet E/F/G remain blank, and a full-table date is not yet defensible.

- At 17:15 SAST on 22 August, a read-only throughput audit of AfriHG b4 job
  `1253374` confirmed the intended FLA GatedDeltaNet chunk fast path and no
  torch fallback, but found that the overall pipeline does not saturate its
  A100-40GB: a 15-second sample averaged `15.9%` SM utilization and used
  `13,147/40,960 MiB`. The frozen batch-4/accumulation-2 LoRA workload is
  launch/overhead bound, while five full 3,082-row validations and five exact
  beam-search generation callbacks account for roughly 5--6 of the observed
  approximately 19 hours per candidate. A larger microbatch may improve
  throughput, but it is unbenchmarked and cannot be introduced mid-grid
  because effective batch construction is preregistered; b4 remains unchanged.

- At 15:13 SAST on 22 August, AfriHG b4 job `1253374` was healthy beyond
  step `347/15410` after starting at `14:53:59 SAST` on `srvrocgpu010`
  A100-40GB. Its live Slurm envelope matches the required settings, execution
  manifest SHA-256 is
  `cd32f4e1a65d8046c78b2cf3f63ae3b1a14303206d0564f97c8c830a7ff1bf85`,
  and all `694` source/config files verified before training. The frozen
  seed-42 recipe uses LR `8.981661817441004e-05` and validation-only
  `eval_all_chrf` selection. Its first exact artifact is tentatively due
  around `18:35--19:00 SAST`, with terminal output around `2026-08-23 10:00
  SAST`. Exact b2/b3 verifier jobs `1253806/1256680` remain Priority-pending.
  Owned work is exactly three A100-40GB jobs, quota is `88.6%/41.4%`, AfriHG
  remains `5/11`, global freeze remains `0/8`, Kombuys remains read-only,
  held-out access is `0`, Sheet E/F/G remain blank, and
  General/Monolingual/publication remain blocked.

- At 12:43 SAST on 22 August, AfriHG b3 job `1250970` completed `0:0` after
  `19:05:12`. Its terminal 128-row artifact has complete `64/64` Xho/Zul
  coverage, zero empty outputs, `64/64` unique predictions, and clean prompt
  boundaries. Terminal Xho/Zul chrF is
  `23.614333434018096/24.862300215885675`, mean `24.238316824951886`, so
  validation-only checkpoint 12328 remains selected at mean
  `24.344455545302008`. Terminal-artifact, retained-state, retained-BIN, and
  final-safetensors SHA-256 values are
  `27ad84c93ec691143f80bc4ee1b15ab6f2bb2a139612617a95e540f998393c32`,
  `d3935936721e6a2a6b20b674e15ee70df153ea3e48f1e2890d700f49f94a4c29`,
  `4e7fa2497781f545659791a45fe27e60da5e570f16dae4d9de47c4ae8c013b5e`,
  and `ead9ba99c7833cc1aec3999f5f90741639e0d1ebce5fea0dbc4c0ca6b99aa333`;
  retained/final configs are byte-identical. Exact tensor verifier job
  `1256680` was submitted once in the freed slot and is Priority-pending, so
  b3 is operationally complete but AfriHG stays `5/11` terminal-valid and
  global freeze stays `0/8`. B2 verifier `1253806` is Priority-pending and b4
  `1253374` is Resources-pending with provisional start `2026-08-22 19:20:51
  SAST`. Owned work is exactly three A100-40GB jobs, quota is `88.6%/41.3%`,
  Kombuys remains read-only, held-out access is `0`, Sheet E/F/G remain
  blank, and General/Monolingual/publication remain blocked.

- At 11:13 SAST on 22 August, AfriHG b3 job `1250970` reached exactly
  `15410/15410` after `17:55:32` of training and entered its frozen terminal
  validation path on `srvrocgpu010` A100-40GB. No terminal exact-generation
  artifact or fault marker exists yet; validation-only checkpoint 12328
  remains the within-run winner at mean chrF `24.344455545302008`. Terminal
  output is tentatively due around `12:15--12:45 SAST`. Verifier `1253806`
  and b4 `1253374` remain Priority-pending; b4 has provisional start
  `2026-08-23 00:15 SAST`. Owned work remains exactly three A100-40GB jobs,
  quota is `88.6%/41.2%`, AfriHG remains `5/11`, global freeze remains `0/8`,
  Kombuys remains read-only, held-out access is `0`, Sheet E/F/G remain
  blank, and General/Monolingual/publication remain blocked.

- At 08:44 SAST on 22 August, AfriHG b3 job `1250970` completed its frozen
  step-12328 callback and resumed healthy training beyond step `12550/15410`
  on `srvrocgpu010` A100-40GB. Its exact 128-row artifact has complete
  `64/64` Xho/Zul coverage, zero empty outputs, `64/64` unique predictions,
  and clean prompt boundaries. Xho/Zul chrF is
  `23.541671781341/25.147239309263014`; mean `24.344455545302008` exactly
  matches `trainer_state.json` and moves b3's validation-only retained
  checkpoint to 12328. Artifact/trainer-state SHA-256 values are
  `6290e2088ad381225e53c5401e1245e79abd9564eea8dcd59915c23cf4a3ff78`
  and `d3935936721e6a2a6b20b674e15ee70df153ea3e48f1e2890d700f49f94a4c29`.
  This remains within-run evidence only, so AfriHG stays `5/11`
  terminal-valid and global freeze stays `0/8`. Terminal output is tentatively
  due around `12:30--13:00 SAST`. Verifier `1253806` and b4 `1253374` remain
  Priority-pending; b4 has a provisional `2026-08-23 00:15 SAST` start.
  Owned work is exactly three A100-40GB jobs, quota is `88.6%/41.2%`,
  Kombuys remains read-only, held-out access is `0`, Sheet E/F/G remain
  blank, and General/Monolingual/publication remain blocked.

- At 07:44 SAST on 22 August, AfriHG b3 job `1250970` reached step
  `12328/15410`, completed full declared validation at health-only loss
  `2.2188514744129217`, and entered its frozen exact-generation callback on
  `srvrocgpu010` A100-40GB. No step-12328 artifact exists yet, so retained
  checkpoint 9246 at validation-only mean chrF `24.12720516406921` remains
  the audited within-run winner. The artifact is tentatively due around
  `08:40--09:00 SAST`, with terminal output around `12:30--13:00 SAST`.
  Verifier `1253806` and b4 `1253374` remain Priority-pending; b4 has a
  provisional `2026-08-23 00:15 SAST` start and the verifier has no ETA.
  Owned work remains exactly three A100-40GB jobs, quota is `88.6%/41.2%`,
  AfriHG remains `5/11`, global freeze remains `0/8`, Kombuys remains
  read-only, held-out access is `0`, Sheet E/F/G remain blank, and
  General/Monolingual/publication remain blocked.

- At 06:15 SAST on 22 August, AfriHG b2 job `1248577` completed `0:0` after
  `19:09:46`. Its terminal 128-row artifact has complete `64/64` Xho/Zul
  coverage, zero empty outputs, `64/64` unique predictions, and clean prompt
  boundaries. Terminal Xho/Zul chrF is
  `21.745224131354718/22.99258739484475`, mean `22.368905763099733`, so
  validation-only retained checkpoint 12328 remains selected at mean
  `22.382494324095585`. Terminal-artifact, retained-state, retained-BIN, and
  final-safetensors SHA-256 values are
  `7b3802ebb13659e5c7d7eca619e662bcf827cf5ad97b723b4e5e74d7d1054091`,
  `87c0a37744edab36842293eb2f7878212d41a7a8bf8cca0f2227903e8c1f658a`,
  `567f63f405236b0e12894ddfbc26ad4b5683d73a5fa8183daa0ce94f2607d1b5`,
  and `36017276eec12f1a30da5a49d83dd0a7ba46a8f9717898d9fbe732e5efd2c15a`;
  retained/final configs are byte-identical. Exact tensor verifier job
  `1253806` was submitted once in the freed slot and is Priority-pending, so
  b2 is operationally complete but AfriHG stays `5/11` terminal-valid and
  global freeze stays `0/8`. B3 `1250970` is healthy beyond step 10887; b4
  `1253374` remains Priority-pending. Owned work is exactly three A100-40GB
  jobs, quota is `88.6%/41.2%`, Kombuys remains read-only, held-out access is
  `0`, Sheet E/F/G remain blank, and General/Monolingual/publication remain
  blocked.

- At 05:13 SAST on 22 August, AfriHG b3 job `1250970` completed its frozen
  step-9246 callback and resumed healthy training beyond step 9730 on
  `srvrocgpu010` A100-40GB. Its 128-row artifact has complete `64/64`
  Xho/Zul coverage, zero empty outputs, `64/64` unique predictions, and clean
  prompt boundaries. Xho/Zul chrF is
  `23.597619040378838/24.656791287759578`; mean `24.12720516406921` exactly
  matches `trainer_state.json` and moves b3's validation-only retained
  checkpoint to 9246. Artifact/trainer-state SHA-256 values are
  `2847bbd52d82ec4a1f8d53d6ad18a316f02c4873747eef5d1588c201d5488e5c`
  and `6619a7336c797836bbe285ceaa9e250df90c79e8b43133443492f6d2a05715c8`.
  B2 `1248577` reached `15410/15410`, completed full declared validation at
  health-only loss `2.272301319278269`, and entered its frozen terminal exact
  callback; b4 `1253374` remains Priority-pending. This is within-run evidence
  only, so AfriHG stays `5/11` terminal-valid and global freeze stays `0/8`.
  Owned work is exactly three A100-40GB jobs, quota is `88.6%/41.2%`,
  Kombuys remains read-only, held-out access is `0`, Sheet E/F/G remain
  blank, and General/Monolingual/publication remain blocked.

- At 02:14 SAST on 22 August, AfriHG b2 job `1248577` completed its frozen
  step-12328 exact callback and resumed healthy training beyond step 12356 on
  `srvrocgpu010` A100-40GB. Its 128-row artifact has complete `64/64`
  Xho/Zul coverage, zero empty outputs, `64/64` unique predictions, and clean
  prompt boundaries. Xho/Zul chrF is
  `21.913107235865905/22.851881412325266`; mean `22.382494324095585`
  exactly matches `trainer_state.json` and moves b2's validation-only retained
  checkpoint to 12328. Artifact/trainer-state SHA-256 values are
  `3760fd2074244b1fb7d90066d3a0c3c2a3795595e0dc129145904576f268262b`
  and `87c0a37744edab36842293eb2f7878212d41a7a8bf8cca0f2227903e8c1f658a`.
  B3 `1250970` remains healthy beyond step 7579 and b4 `1253374` remains
  Priority-pending without an ETA. This is within-run evidence only, so
  AfriHG stays `5/11` terminal-valid and global freeze stays `0/8`. Owned
  work is exactly three A100-40GB jobs, quota is `88.6%/41.2%`, Kombuys
  remains read-only, held-out access is `0`, Sheet E/F/G remain blank, and
  General/Monolingual/publication remain blocked.

- At 01:14 SAST on 22 August, AfriHG b3 job `1250970` completed its frozen
  step-6164 exact callback and resumed healthy training beyond step 6420 on
  `srvrocgpu010` A100-40GB. Its 128-row artifact has complete `64/64`
  Xho/Zul coverage, zero empty outputs, `64/64` unique predictions, and clean
  prompt boundaries. Xho/Zul chrF is
  `22.305296547403668/23.369240354846387`; mean `22.837268451125027`
  exactly matches `trainer_state.json` and moves b3's validation-only retained
  checkpoint to 6164. Artifact/trainer-state SHA-256 values are
  `de331d569cce31014f80045689514c372c090d4def9dc77d8654051ff000f369`
  and `993bd7de2f85cfd1b737f4482abe81279bcee7cda8986521457824523f29ddb9`.
  B2 `1248577` reached step 12328, completed all 3,082 validation rows at
  health-only loss `2.2725518763181376`, and entered frozen exact generation;
  no selector artifact exists yet. B4 `1253374` remains Priority-pending.
  This is within-run evidence only, so AfriHG stays `5/11` terminal-valid and
  global freeze stays `0/8`. Owned work is exactly three A100-40GB jobs,
  quota is `88.6%/41.2%`, Kombuys remains read-only, held-out access is `0`,
  Sheet E/F/G remain blank, and General/Monolingual/publication remain
  blocked.

- At 00:17 SAST on 22 August, exact read-only verifier retry `1252840`
  completed `0:0` and proved retained-checkpoint/final-adapter tensor equality
  for AfriHG b0 and b1. B0 has `424` keys and `69,438,784` values in each
  artifact; b1 has `424` keys and `76,410,112` values. Both have zero missing
  keys and zero shape, dtype, or value mismatches. The verifier output SHA-256
  is `98fd80db0522250e91ec7d42074c0aa2d3f59d56fcbf8a9c8c9f2a8857194eef`.
  B0 and b1 are now scientifically terminal-valid, advancing AfriHG from
  `3/11` to `5/11`; the global freeze remains `0/8`. B2 `1248577` is healthy
  beyond step `11487/15410`, and b3 `1250970` remains healthy in its frozen
  step-6164 exact-generation callback. With the third slot free, the next
  preregistered candidate b4 was submitted once as job `1253374` under the
  required A100-40GB envelope; it is Priority-pending with provisional start
  `13:51 SAST`. Owned work is exactly three A100-40GB jobs, quota is
  `88.6%/41.2%`, Kombuys remains read-only with RTX 5090 untouched, held-out
  access is `0`, Sheet E/F/G remain blank, and
  General/Monolingual/publication remain blocked.

- At 23:49 SAST on 21 August, AfriHG b3 job `1250970` reached step
  `6164/15410`, completed all `3,082` declared validation rows at health-only
  loss `2.2234693127433487`, and entered its frozen exact-generation path on
  `srvrocgpu010` A100-40GB. No exact artifact exists yet, so checkpoint 3082
  remains the audited within-run best and scientific progress does not change;
  the artifact is tentatively expected around `00:55--01:10 SAST` on 22
  August. B2 `1248577` remains healthy beyond step 10844, retaining
  checkpoint 9246 at validation-only mean chrF `22.230781529182863`.
  Verifier retry `1252840` remains Resources-pending with scheduler estimate
  `05:38:36 SAST`. Owned work is exactly three A100-40GB jobs, quota is
  `88.6%/41.2%`, AfriHG remains `3/11`, global freeze remains `0/8`, Kombuys
  remains read-only, held-out access is `0`, Sheet E/F/G remain blank, and
  General/Monolingual/publication remain blocked.

- At 22:45 SAST on 21 August, AfriHG b2 job `1248577` completed its frozen
  step-9246 callback and resumed fault-free training beyond step 9736 on
  `srvrocgpu010` A100-40GB. Its exact 128-row artifact has complete `64/64`
  Xho/Zul coverage, zero empty outputs, `64/64` unique predictions, and clean
  prompt boundaries. Xho/Zul chrF is
  `21.729798398243634/22.73176466012209`; mean `22.230781529182863` exactly
  matches `trainer_state.json` and moves the retained validation-only
  checkpoint to 9246. Artifact/trainer-state SHA-256 values are
  `14044d771f1955b19a6872e63a4d661ce483581acaffbf87d51c82edf7c919c6`
  and `c5a92ec3970c2bc5643afaaa01da6a4f6726093d08ebd09c3037e8a7dd94a059`.
  Verifier `1252793` failed immediately `1:0` on a shell-quoting `SyntaxError`
  before reading adapters; output SHA-256 is
  `69d68b830a11a7f7e88600487768838a2ffb4e382f02e4b2151eddf0fadbf363`.
  The corrected, `bash -n` clean verifier is preserved at SHA-256
  `b10ae7326bc1e52a1888689388678a5a571ca1b84aab57fd8a428cefc9a9e056`
  and retry job `1252840` is Resources-pending with scheduler estimate
  `2026-08-22 05:38:36 SAST`. B3 `1250970` is healthy beyond step 4987.
  Owned work is exactly three A100-40GB jobs, quota is `88.6%/41.2%`, AfriHG
  remains `3/11`, global freeze remains `0/8`, Kombuys remains read-only,
  held-out access is `0`, Sheet E/F/G remain blank, and
  General/Monolingual/publication remain blocked.

- At 21:47 SAST on 21 August, AfriHG b1 job `1248576` completed `0:0` after
  `19:02:10`. Its terminal 128-row artifact has complete `64/64` Xho/Zul
  coverage, zero empty outputs, `64/63` unique predictions, and clean prompt
  boundaries. Terminal Xho/Zul chrF is
  `22.999692656448786/23.8565376695875`, mean `23.42811516301814`; retained
  checkpoint 12328 remains selected by validation-only mean
  `23.50555969452345`. Terminal-artifact, retained-trainer-state,
  retained-BIN, and final-safetensors SHA-256 values are
  `f7c2465d1ca50ff9092476b013ce4fe16b47205c578733e2c39c3296b175bb20`,
  `1807ebff82c0488b2a77f87e2b575316b55be386875c9bf7ae359aae682ac129`,
  `32c3378f8a1d7418e5f277e97274fdc82121f8e97d50c5074d84ae0a87ce36ef`,
  and `0228762dff097e00bc7654cb9406bfb6d0a47ee3245eb281cb519132cbd3cb16`;
  retained/final configs are byte-identical. Read-only exact tensor verifier
  job `1252793` was submitted within the freed third A100-40GB slot and is
  Priority-pending with scheduler estimate `2026-08-22 11:51 SAST`. B0 and b1
  remain operationally complete but scientifically pending until it passes,
  so AfriHG stays `3/11` and global freeze stays `0/8`. B2 `1248577` remains
  healthy in step-9246 exact generation with no artifact yet (tentative ETA
  `22:15--22:30 SAST`); b3 `1250970` resumed healthy training beyond step
  3900. Owned work is exactly three A100-40GB jobs, quota is `88.6%/41.2%`,
  Kombuys remains read-only, held-out access is `0`, Sheet E/F/G remain blank,
  and General/Monolingual/publication remain blocked.

- At 21:14 SAST on 21 August, AfriHG b3 job `1250970` completed its frozen
  step-3082 callback and resumed fault-free training beyond step 3245 on
  `srvrocgpu010` A100-40GB. Its exact 128-row artifact has complete `64/64`
  Xho/Zul coverage, zero empty outputs, `64/64` unique predictions, and clean
  prompt boundaries. Xho/Zul chrF is
  `21.110816397043155/22.68749899428495`; registered mean
  `21.89915769566405` exactly matches `trainer_state.json` and retains
  checkpoint 3082. Artifact/trainer-state SHA-256 values are
  `d326ba97a910df8d9a36c6badccf6393dacee72b4f324d1530f1c9bf984bbbfa`
  and `99b92b87235c9a1af4c056b37d3e261f4a01076be4fefb1ffbada5625bbd74d9`.
  B2 `1248577` reached step 9246 and completed all `3,082` validation rows at
  health-only loss `2.2812004114110156`; b1 `1248576` remains in terminal
  exact generation. This is validation-only within-run evidence, so AfriHG
  remains `3/11`, global freeze `0/8`, and b0 tensor equivalence is pending.
  Quota is `88.6%/41.1%`, all owned work is A100-40GB, Kombuys remains
  read-only, held-out access is `0`, and Sheet E/F/G remain blank.

- At 20:44 SAST on 21 August, AfriHG b1 job `1248576` reached exactly
  `15410/15410`, completed all `3,082` declared validation rows at health-only
  loss `2.2297809261690866`, and entered its frozen terminal exact generation
  callback on `srvrocgpu010` A100-40GB with automatic batch size 64. No
  terminal artifact or final reconciliation exists yet, so b1 remains
  non-terminal-valid and checkpoint 12328 remains the audited within-run best.
  B3 `1250970` remains in step-3082 exact generation; b2 `1248577` is healthy
  near step 8723. All three owned jobs are running on A100-40GB, quota is
  `88.6%/41.1%`, AfriHG stays `3/11`, global freeze stays `0/8`, b0 tensor
  equivalence is pending, Kombuys remains read-only, held-out access is `0`,
  and Sheet E/F/G remain blank.

- At 20:14 SAST on 21 August, AfriHG b3 job `1250970` reached step
  `3082/15410`, completed all `3,082` declared validation rows at health-only
  loss `2.2798393393088903`, and entered frozen exact generation on
  `srvrocgpu010` A100-40GB with automatic batch size 64. No step-3082 exact
  artifact exists yet, so there is no selector or scientific-progress change.
  B1 `1248576` is fault-free at step 15123, about 15 minutes from its terminal
  training boundary; b2 `1248577` is healthy near step 8149. All three owned
  jobs are running on A100-40GB, quota is `88.6%/41.1%`, AfriHG stays `3/11`,
  global freeze stays `0/8`, b0 tensor equivalence is pending, Kombuys remains
  read-only, held-out access is `0`, and Sheet E/F/G remain blank.

- At 18:44 SAST on 21 August, AfriHG b2 job `1248577` completed its frozen
  step-6164 callback and resumed fault-free training beyond step 6455 on
  `srvrocgpu010` A100-40GB. Its exact 128-row artifact has complete `64/64`
  Xho/Zul coverage, zero empty outputs, `64/64` unique predictions, and clean
  prompt boundaries. Xho/Zul chrF is
  `20.71843131877858/21.26318054903395`; mean `20.990805933906266` does not
  improve the registered best `21.120202579737125`, so checkpoint 3082 remains
  retained. Artifact/trainer-state SHA-256 values are
  `f0339961bc79557317bc9d272e4fb15092590f57e4ec6aee5c8db81ed31f10d9`
  and `047ddfda82e48aac5173a219b0fabee8bb1026b7873898bb6031e75636087de3`.
  B1 `1248576` is healthy near step 13424 and b3 `1250970` near step 1694.
  This is validation-only within-run evidence, so AfriHG remains `3/11`,
  global freeze `0/8`, and b0 tensor equivalence is pending. Quota is
  `88.6%/41.1%`, all owned work is A100-40GB, Kombuys remains read-only,
  held-out access is `0`, and Sheet E/F/G remain blank.

- At 18:14 SAST on 21 August, AfriHG b1 job `1248576` completed its frozen
  step-12328 callback and resumed fault-free training beyond step 12853 on
  `srvrocgpu010` A100-40GB. Its exact 128-row artifact has complete `64/64`
  Xho/Zul coverage, zero empty outputs, `64/63` unique predictions, and clean
  prompt boundaries. Xho/Zul chrF is
  `22.873427986122337/24.137691402924556`; registered mean
  `23.50555969452345` exactly matches `trainer_state.json`, improves b1's
  prior best, and retains checkpoint 12328. Artifact/trainer-state SHA-256
  values are
  `cd151e6d409410635d42ea5df4ac47b82cf657cbd5decb4bdaa0d1331e25d008`
  and `1807ebff82c0488b2a77f87e2b575316b55be386875c9bf7ae359aae682ac129`.
  B2 `1248577` remains in step-6164 exact generation; b3 `1250970` is healthy
  near step 1098. This is validation-only within-run evidence, so AfriHG
  remains `3/11`, global freeze `0/8`, and b0 tensor equivalence is pending.
  Quota is `88.6%/41.1%`, all owned work is A100-40GB, Kombuys remains
  read-only, held-out access is `0`, and Sheet E/F/G remain blank.

- At 17:45 SAST on 21 August, AfriHG b3 job `1250970` started at `17:15:39`
  on `srvrocgpu010` A100-40GB, verified its immutable 694-file execution
  manifest at SHA-256
  `71a5e984da15fd42c49ef6f0acfb013155f65e902ff23e3eeb657867f36b78e3`,
  loaded the canonical pure GDN with the CUDA fast path, declared complete
  `24,649/3,082` train/validation coverage, and was fault-free near step 525.
  B2 `1248577` reached step `6164/15410`, completed all `3,082` validation
  rows at health-only loss `2.3051664338801916`, and entered exact generation;
  b1 `1248576` remains in its step-12328 exact callback. No new exact artifact
  exists yet. All three owned jobs are running on A100-40GB with zero targeted
  faults; quota is `88.6%/41.1%`. AfriHG stays `3/11`, global freeze stays
  `0/8`, b0 tensor equivalence remains pending, Kombuys remains read-only,
  held-out access is `0`, and Sheet E/F/G remain blank.

- At 17:02 SAST on 21 August, AfriHG b1 job `1248576` reached step
  `12328/15410`, completed all `3,082` declared validation rows at health-only
  loss `2.2270502267450114`, and entered its frozen exact generation callback
  on `srvrocgpu010` A100-40GB with automatic batch size 64. No step-12328
  exact artifact exists yet, so there is no selector or scientific-progress
  change. B2 `1248577` is fault-free near step `5774/15410`; b3 `1250970`
  remains Resources-pending with provisional start `2026-08-22 02:34:16`.
  Owned work remains at the three-job A100-40GB cap, quota is
  `88.6%/40.9%`, AfriHG stays `3/11`, global freeze stays `0/8`, b0 tensor
  equivalence is pending, Kombuys remains read-only, held-out access is `0`,
  and Sheet E/F/G remain blank.

- At 15:02 SAST on 21 August, AfriHG b2 job `1248577` completed its frozen
  step-3082 callback and resumed fault-free training beyond step 3464 on
  `srvrocgpu010` A100-40GB. Its exact 128-row artifact has complete `64/64`
  Xho/Zul coverage, zero empty outputs, `64/64` unique predictions per
  language, and clean prompt boundaries. Xho/Zul chrF is
  `21.24350617202978/20.996898987444474`; registered mean
  `21.120202579737125` exactly matches retained checkpoint 3082.
  Artifact/trainer-state SHA-256 values are
  `1f43c032f88bb1bb46b4ce8df0e5d5abdd5014ce9c13ec1234578151df52d15b`
  and `4f6ce1239e86c4f7138e52b0fd0ddc54841e59b10758020e8081110c2849ba8d`.
  B1 `1248576` remains healthy near step 10505; b3 `1250970` is
  Resources-pending with provisional start `2026-08-22 02:34:16`. This is
  validation-only within-run evidence, so AfriHG remains `3/11`, global
  freeze `0/8`, and b0 tensor equivalence is pending. Quota is
  `88.6%/40.9%`, all owned work is A100-40GB, Kombuys remains read-only,
  held-out access is `0`, and Sheet E/F/G remain blank.

- At 14:02 SAST on 21 August, AfriHG b1 job `1248576` completed its frozen
  step-9246 callback and resumed fault-free training beyond step 9350 on
  `srvrocgpu010` A100-40GB. Its exact 128-row artifact has complete `64/64`
  Xho/Zul coverage, zero empty outputs, `64/64` unique predictions per
  language, and clean prompt boundaries. Xho/Zul chrF is
  `23.160436060121018/23.693880347472295`; mean `23.427158203796658` exactly
  matches the newly retained checkpoint 9246. Artifact/trainer-state SHA-256
  values are
  `7762ee77ec8f1d5d9befd58a7001f2b3590eb6bbcd5023a457dbb3073fbfeef8`
  and `7fa3ac45f71768986f8f3db90ee3cd16b8f0dc3c855e32c428bcb11ee0b81f42`.
  B2 `1248577` completed its first full validation at step 3082 and entered
  exact scoring; b3 `1250970` remains Priority-pending. This is validation-only
  within-run evidence, so AfriHG remains `3/11`, global freeze `0/8`, and b0
  tensor equivalence is pending. Quota is `88.6%/40.9%`, all owned work is
  A100-40GB, Kombuys remains read-only, held-out access is `0`, and Sheet
  E/F/G remain blank.

- At 13:02 SAST on 21 August, AfriHG b1 job `1248576` reached step
  `9246/15410`, completed all `3,082` declared validation rows at health-only
  loss `2.2297211188453114`, and entered its frozen exact generation callback
  on `srvrocgpu010` A100-40GB with automatic batch size 64. No step-9246
  exact artifact exists yet, so there is no selector or scientific-progress
  change. B2 `1248577` is fault-free near step `2487/15410`; b3 `1250970`
  remains Priority-pending. Owned work remains at the three-job A100-40GB cap,
  quota is `88.6%/40.9%`, AfriHG stays `3/11`, global freeze stays `0/8`,
  b0 tensor equivalence is pending, Kombuys remains read-only, held-out access
  is `0`, and Sheet E/F/G remain blank.

- At 11:02 SAST on 21 August, AfriHG b2 job `1248577` started on
  `srvrocgpu010` with one A100-40GB. It verified the immutable 694-file
  execution manifest, loaded the canonical pure `GatedDeltaNetForCausalLM`
  with its CUDA fast path, and declared complete `24,649/3,082`
  train/validation coverage; the targeted log check found no fault. B1
  `1248576` is healthy at step `7189/15410`, retaining checkpoint 6164 at
  validation-only mean chrF `22.030107111540246`; b3 `1250970` remains
  Priority-pending. Owned work is at the three-job cap, entirely A100-40GB.
  Scientific progress remains AfriHG `3/11` and global freeze `0/8` because
  b0 tensor equivalence is pending. Quota is `88.6%/40.9%`, Kombuys remains
  read-only, held-out access is `0`, and Sheet E/F/G remain blank.

- At 10:32 SAST on 21 August, AfriHG b1 job `1248576` completed its frozen
  step-6164 callback and resumed healthy training beyond step 6608 on
  `srvrocgpu010` A100-40GB. Its exact 128-row artifact has complete `64/64`
  Xho/Zul coverage, zero empty outputs, `64/64` unique predictions per
  language, and clean prompt boundaries. Xho/Zul chrF is
  `21.641679914561536/22.418534308518957`; mean `22.030107111540246` exactly
  matches the newly retained checkpoint 6164. Artifact SHA-256 is
  `0de4e760a86cbbf12805d4fb50f5b70c5c1f2b8f79a692f58c9417cb9b97aac3`.
  This is validation-only within-run selector evidence, not terminal-valid:
  AfriHG remains `3/11`, global freeze `0/8`, and b0 tensor equivalence is
  pending. B2 `1248577` and b3 `1250970` remain Resources/Priority-pending;
  quota is `88.6%/40.7%`, all owned work is A100-40GB, Kombuys remains
  read-only, held-out access is `0`, and Sheet E/F/G remain blank.

- At 09:02 SAST on 21 August, AfriHG b1 job `1248576` reached step
  `6164/15410`, completed all `3,082` declared validation rows at health-only
  loss `2.2457383123022794`, and entered its frozen exact generation callback
  on `srvrocgpu010` A100-40GB. No step-6164 exact artifact exists yet, so
  there is no selector or scientific-progress change: AfriHG remains `3/11`,
  global freeze `0/8`, and b0 tensor equivalence is pending. B2 `1248577` and
  b3 `1250970` remain Resources/Priority-pending; quota is `88.6%/40.7%`, all
  owned work is A100-40GB, Kombuys remains read-only, held-out access is `0`,
  and Sheet E/F/G remain blank.

- At 06:25 SAST on 21 August, AfriHG b1 job `1248576` completed its frozen
  step-3082 callback and resumed healthy training beyond step 3176 on
  `srvrocgpu010` A100-40GB. The exact 128-row artifact has complete `64/64`
  Xho/Zul coverage, zero empty outputs, `64/64` unique predictions per
  language, and clean prompt boundaries. Xho/Zul chrF is
  `21.196464787124476/22.282564090083`; mean `21.739514438603738` exactly
  matches retained checkpoint 3082. Artifact SHA-256 is
  `96642a367fbe5cf9bd5c6131e075e5e3100d170ebb7fe3e7558a640bf0ad4a39`.
  This is validation-only within-run selector evidence, not terminal-valid:
  AfriHG remains `3/11`, global freeze `0/8`, and b0 tensor equivalence is
  pending. B2 `1248577` and b3 `1250970` remain Resources/Priority-pending;
  quota is `88.6%/40.7%`, all owned work is A100-40GB, Kombuys remains
  read-only, held-out access is `0`, and Sheet E/F/G remain blank.

- At 05:24 SAST on 21 August, AfriHG b1 job `1248576` reached step
  `3082/15410`, completed all `3,082` declared validation rows at health-only
  loss `2.3020349857793234`, and entered its frozen exact generation callback
  with automatic batch size 64. No exact artifact exists yet, so there is no
  selector or scientific-progress change: AfriHG remains `3/11` and global
  freeze `0/8`; b0 retained/final tensor equivalence remains pending. B2
  `1248577` and b3 `1250970` remain Resources/Priority-pending, all owned work
  is A100-40GB, quota is `88.6%/40.6%`, Kombuys remains read-only, held-out
  access is `0`, and Sheet E/F/G remain blank.

- At 04:54 SAST on 21 August, AfriHG b1 job `1248576` remained healthy near
  step `2747/15410` on `srvrocgpu010` A100-40GB and b2 `1248577` remained
  Resources-pending. With two owned jobs, no duplicate, and no b3 output,
  preregistered Stage-B b3 was submitted once as job `1250970` from the same
  immutable snapshot. It is Priority-pending and requests the required single
  A100-40GB `gpu:ampere`; owned work is now at the three-job cap with no
  A100-80GB/L40S overlap. Scientific progress remains AfriHG `3/11` and
  global freeze `0/8` because b0 retained/final tensor equivalence is still
  pending. Quota is `88.6%/40.6%`, Kombuys remains read-only, held-out access
  is `0`, and Sheet E/F/G remain blank.

- At 02:55 SAST on 21 August, AfriHG Stage-B b0 job `1248575` completed
  `0:0` after `18:47:37`. Its audited terminal 128-row artifact has complete
  `64/64` Xho/Zul coverage, no empty/debug-empty outputs, `64/64` unique
  predictions per language, and clean prompt boundaries. Terminal Xho/Zul
  chrF is `23.246064347495587/25.29784315222218` (mean
  `24.27195374985888`), leaving checkpoint 12328 as the validation-only best
  at `24.419330770518705`. Artifact SHA-256 is
  `3963d5be67a7492ed468e1fcc561ef394429e7715af62b7703e36538dc63939d`.
  Exact retained-versus-final tensor equivalence is still pending a
  compute-node check, so b0 is operationally complete but AfriHG remains
  `3/11` scientifically terminal-valid and the global freeze stays `0/8`.
  B1 job `1248576` started at `02:34:16` on `srvrocgpu010`, verified its
  immutable manifest and `24,649/3,082` coverage, and is healthy; b2
  `1248577` remains pending for resources. Quota is `88.6%/40.6%`, owned work
  remains A100-40GB only, Kombuys remains read-only, held-out access is `0`,
  and Sheet E/F/G remain blank.

- At 13:53 SAST on 20 August, the `GDN Results` presentation was normalized
  to the shared architecture result-table contract. The operational dashboard
  in rows 46--59 was removed. Column J now shows 42 representative
  `Input / Gold / Pure GDN` examples rather than artifact paths; the exact
  corrected artifact paths, selectors, and prior provenance remain in cell
  notes. Verification found no visible JSON/scratch paths, no missing notes,
  intact wrapping, and E/F/G still entirely blank. Future monitoring belongs
  in dated notes and `sallm_progress.md`, not in the result table.

- At 11:43 SAST on 20 August, AfriHG Stage-B b0 job `1248575` had completed
  its first frozen exact callback and resumed training. The step-3082
  artifact contains exactly `128` rows (`64/64` Xho/Zul), zero empty or
  whitespace-only predictions, `64/64` unique predictions per language, and
  a clean `[BOS]` to `[EOS]<|assistant|>` prompt contract. Xho/Zul chrF is
  `21.196932693799432/23.036447759744906`; registered mean chrF is
  `22.11669022677217`, matching `trainer_state.json` `best_metric` and
  retained checkpoint 3082. Artifact/trainer-state SHA-256 values are
  `825e6a931dfc37a728e739e7df7bf725c1e1e2c1cace17e461b258f0b9d26a4c`
  and `848ac3e516b906bf15c6cca308297654a9376e61a3469bd4e79fd40429d00779`.
  This is eligible validation-only selector evidence, not a terminal-valid
  candidate: AfriHG remains `3/11` terminal-valid, no winner is frozen, and
  the global freeze stays `0/8`. Jobs `1248576/1248577` remain pending;
  quota is `88.6%/40.4%`, held-out access is `0`, and Kombuys remains
  read-only with RTX 5090 untouched.

- At 11:07 SAST on 20 August, AfriHG Stage-B b0 job `1248575` had reached
  step `3082/15410`, completed full declared validation over all `3,082`
  rows (`1,305` Xho, `1,777` Zul), and entered the frozen exact generation
  callback. Health-only validation loss is `2.269708057876379` over all
  declared rows; it is not selector evidence. Both exact-generation language
  segments have begun at automatic batch size 64, the job remains healthy on
  `srvrocgpu010` A100-40GB, and no exact JSONL exists yet. Scientifically
  AfriHG therefore remains `3/11` terminal-valid and the global freeze stays
  `0/8`. Jobs `1248576/1248577` remain pending, quota is `88.6%/40.4%`,
  held-out access is `0`, and Kombuys remains read-only with RTX 5090
  untouched.

- At 08:08 SAST on 20 August, AfriHG Stage-B b0 job `1248575` was healthy
  near `370/15410` after starting at `07:46:38` on `srvrocgpu010` A100-40GB.
  The GatedDeltaNet fast path, immutable source, frozen b0 registry values,
  and `24,649/3,082` train/validation coverage all reconcile; targeted fault
  scans are empty. Manifest/trial SHA-256 values are
  `630e9a5506288e57a8b6440e30941b1d1786d3870a039f26b39ec29b915da36a`
  and `c3067b07bc81719ca08eae359e2c8131fd195f61821a4b921e40ef399b1a0fcc`.
  Its first training boundary is tentatively near `10:25--10:40`, with the
  first exact selector artifact around `11:00--11:30`. B1/b2 jobs
  `1248576/1248577` remain pending; b1 has provisional start `18:31:54` and
  b2 has no estimate. Scientifically AfriHG remains `3/11` terminal-valid and
  no family winner is frozen (`0/8` globally). All owned work requests only
  A100-40GB `gpu:ampere`; quota is `88.6%/40.4%`, Kombuys remains read-only
  with RTX 5090 untouched, and held-out access is `0`. The live Sheet tracker
  was updated and verified with E/F/G blank; General, Monolingual, and
  publication remain blocked.

- At 07:42 SAST on 20 August, AfriHG Stage-A a2 job `1246940` had completed
  `0:0` at `07:20:44` and passed terminal reconciliation. Its exact 128-row
  step-15410 artifact has `64/64` Xho/Zul coverage, no empty or
  normalized-empty output, `64/64` unique predictions per language, and a
  clean BOS/assistant-marker contract. Xho/Zul chrF is
  `23.800277043809448/26.021454194337124`, mean
  `24.910865619073284`, and retained checkpoint 15410 exactly equals the
  final adapter across `424` keys and `71,762,560` tensor values. This closes
  Stage A at `3/3`; a2 is the Stage-A leader, not a frozen family winner.
  AfriHG is `3/11` terminal-valid at seed 42 because b0-b7 and subsequent
  top-two three-seed confirmation remain mandatory; the global final-family
  freeze stays `0/8`. With zero owned jobs and absent outputs, Stage-B jobs
  `1248575/1248576/1248577` for b0/b1/b2 were submitted once from the verified
  immutable snapshot. All are pending and request only one A100-40GB
  `gpu:ampere`; b0 has provisional start `18:31:54 SAST`, while b1/b2 have no
  estimate. Quota is `88.6%/40.2%`, no A100-80GB/L40S work is owned, Kombuys
  remains read-only with RTX 5090 untouched, and held-out access is `0`. The
  live Sheet tracker was updated and verified with the correct Stage-B/Stage-C
  remaining work; E/F/G remain blank. General, Monolingual, and publication
  remain blocked.

- At 06:37 SAST on 20 August, AfriHG Stage-A a2 job `1246940` had reached
  exactly `15410/15410`, completed terminal full declared validation over
  all `3,082` rows at health-only loss `2.26641358530885`, and entered its
  frozen terminal exact callback on `srvrocgpu010` A100-40GB. Automatic
  generation batch size 64 was established at `06:19:19`; the terminal
  artifact and final adapter remained absent, with no targeted fault marker
  and terminal reconciliation tentatively due around `07:15--07:30 SAST`.
  The health-only loss is not selector evidence; a2 retains audited
  checkpoint 12328 at mean chrF `24.78084394766916` and remains
  non-terminal-valid until exact reconciliation. Operationally jobs
  `1246938/1246939` are complete and `1246940` is in its terminal callback;
  scientifically AfriHG remains `2/3` terminal-valid with no winner frozen
  (`0/8` globally). Quota is `88.6%/40.2%`, the only owned work is one
  A100-40GB `gpu:ampere`, Kombuys remains read-only with RTX 5090 untouched,
  held-out adapter evaluation is `0`, Sheet E/F/G are blank, and
  General/Monolingual/publication remain blocked.

- At 05:39 SAST on 20 August, AfriHG Stage-A a2 job `1246940` had completed
  its frozen step-9246 and step-12328 exact callbacks and resumed healthy,
  fault-free training near `14717/15410` on `srvrocgpu010` A100-40GB. The
  exact step-12328 artifact has `64/64` Xho/Zul coverage, no empty or
  normalized-empty output, `64/64` unique predictions per language, and a
  clean BOS/assistant-marker contract. Xho/Zul chrF is
  `23.738394297462413/25.823293597875907`; mean
  `24.78084394766916` exactly matches newly retained checkpoint 12328.
  Artifact and trainer-state hashes are
  `d9285b011c04c1b14cdff110a8929786f3e6d761d414a51b3cd4e0a1e23cd5b0`
  and `2a06be19720b2b9ff0c4a06fb15f02fb351a043058cea451ace2b9f07dec43ef`.
  This is within-run validation evidence only: a2 remains non-terminal-valid,
  so AfriHG stays `2/3` terminal-valid with no winner frozen (`0/8`
  globally). Jobs `1246938/1246939` are complete and `1246940` is running,
  with terminal reconciliation tentatively due around `07:15--07:30 SAST`.
  Quota is `88.6%/40.2%`, the only owned work is one A100-40GB
  `gpu:ampere`, Kombuys remains read-only with RTX 5090 untouched, held-out
  adapter evaluation is `0`, Sheet E/F/G are blank, and
  General/Monolingual/publication remain blocked.

- At 22:21 SAST on 19 August, AfriHG Stage-A a2 job `1246940` remained
  healthy and fault-free at `8669/15410` after `9:54:50` on
  `srvrocgpu010` A100-40GB. It retains checkpoint 6164 at validation-only
  mean chrF `23.449880263733377`; its checkpoint-9246 boundary remains near
  `22:50--23:00`, with tentative terminal completion around `07:30--08:00`
  on 20 August after the remaining full validations and frozen exact
  callbacks. Jobs `1246938/1246939` are complete and `1246940` is running;
  scientifically AfriHG remains `2/3` terminal-valid and no winner is frozen
  (`0/8` globally). Quota is `88.6%/40.2%`, the only owned work is one
  A100-40GB `gpu:ampere`, Kombuys remains read-only with RTX 5090 untouched,
  held-out adapter evaluation is `0`, Sheet E/F/G are blank, and
  General/Monolingual/publication remain blocked.

- At 21:50 SAST on 19 August, AfriHG Stage-A a1 job `1246939` had completed
  `0:0` at `21:38:34` and passed terminal reconciliation. Its exact 128-row
  step-15410 artifact has `64/64` Xho/Zul coverage, no empty or
  normalized-empty output, `64/63` unique predictions, and a clean
  BOS/assistant-marker contract. Terminal Xho/Zul chrF is
  `23.35450117522189/25.122629092549037`, mean `24.238565133885464`, so the
  retained validation-only best remains checkpoint 12328 at mean chrF
  `24.316085107021614`. The retained BIN and final safetensors have the same
  424 keys and all `71,762,560` tensor values compare exactly. Terminal
  artifact and retained/final weight hashes are
  `1a6963e6a1308ad2c77fc0df50b58f8a66035b64d011d0df2139e9bbb7c9b295`,
  `317dcf88b4a92a8852159186184f837205b88643acd2ccdea8c0fe7879734876`,
  and `f43e5ed0f199026e13421ed2b3f3d610a46d2e5c18e2a534f4c48c2cf0982ee8`.
  A1 is now terminal-valid, advancing AfriHG to `2/3`. A2 `1246940` remained
  healthy near `8087/15410`, retaining checkpoint 6164 at mean chrF
  `23.449880263733377`; its next boundary 9246 is expected around
  `22:50--23:00`, with tentative terminal completion around `07:30--08:00`
  on 20 August. No winner is frozen. Quota is `88.6%/40.2%`, the only owned
  work is A100-40GB, Kombuys remains read-only with RTX 5090 untouched,
  held-out adapter evaluation is `0`, Sheet E/F/G are blank, and
  General/Monolingual/publication remain blocked.

- At 21:20 SAST on 19 August, AfriHG Stage-A a1 job `1246939` remained
  healthy in its frozen terminal exact callback on `srvrocgpu010` A100-40GB.
  Its second language phase established automatic generation batch size 64
  at `21:04:36`; the terminal step-15410 JSONL remained absent at
  `21:20:47`, with no targeted fault marker and an artifact ETA of
  `21:35--21:50`. A1 retains checkpoint 12328 at mean chrF
  `24.316085107021614` and remains non-terminal-valid until exact
  reconciliation. A2 `1246940` remained healthy near `7515/15410`, retaining
  checkpoint 6164 at mean chrF `23.449880263733377`. Operationally AfriHG is
  one complete, one in terminal callback, and one running; scientifically it
  remains `1/3` terminal-valid with no winner frozen. Quota is
  `88.6%/40.1%`, all owned jobs are A100-40GB, Kombuys remains read-only with
  RTX 5090 untouched, held-out adapter evaluation is `0`, Sheet E/F/G are
  blank, and General/Monolingual/publication remain blocked.

- At 20:51 SAST on 19 August, AfriHG Stage-A a1 job `1246939` had reached
  exactly `15410/15410`, completed terminal full declared validation over all
  `3,082` rows at health-only loss `2.2272494524659314`, and entered its
  frozen terminal exact callback on `srvrocgpu010` A100-40GB. Automatic
  generation batch size 64 was established at `20:36:31`; the terminal
  step-15410 JSONL remained absent at `20:51:55`, with no targeted fault
  marker and a tentative artifact ETA of `21:35--21:50`. The health-only loss
  is not selector evidence; a1 retains audited checkpoint 12328 at mean chrF
  `24.316085107021614` and is not terminal-valid until exact reconciliation.
  A2 `1246940` remained healthy near `6933/15410`, retaining checkpoint 6164
  at mean chrF `23.449880263733377`. Operationally AfriHG is one complete,
  one in terminal callback, and one running; scientifically it remains `1/3`
  terminal-valid with no winner frozen. Quota is `88.6%/40.1%`, all owned
  jobs are A100-40GB, Kombuys remains read-only with RTX 5090 untouched,
  held-out adapter evaluation is `0`, Sheet E/F/G are blank, and
  General/Monolingual/publication remain blocked.

- At 20:20 SAST on 19 August, AfriHG Stage-A a2 job `1246940` had completed
  its frozen step-6164 exact callback and resumed healthy training near
  `6362/15410` on `srvrocgpu010` A100-40GB. Its exact 128-row artifact has
  `64/64` Xho/Zul coverage, no empty or normalized-empty output, `64/64`
  unique predictions per language, and a clean BOS/assistant-marker contract.
  Xho/Zul chrF is `22.762274809600555/24.137485717866202`; mean
  `23.449880263733377` exactly matches newly retained checkpoint 6164 and
  improves a2 checkpoint 3082. Artifact and trainer-state hashes are
  `7d4ec097ce2ec64cfbafe079c93e0276f66cb6d0612381103314f8748413d08e` and
  `e9e9425e5448a6d4edc8d14c6721d870b830977ca1d21c6aec30a14070823091`.
  This is within-run validation evidence only: a2 is not terminal-valid and
  no AfriHG winner is frozen. A1 `1246939` remained healthy near
  `15192/15410`, retaining validated checkpoint 12328 at mean chrF
  `24.316085107021614`, about 11 minutes from its final step boundary before
  full validation and its final exact callback. Operationally AfriHG remains
  one complete plus two running; scientifically it is `1/3` terminal-valid.
  Quota is `88.6%/40.1%`, all owned jobs are A100-40GB, Kombuys remains
  read-only with RTX 5090 untouched, held-out adapter evaluation is `0`,
  Sheet E/F/G are blank, and General/Monolingual/publication remain blocked.

- At 19:20 SAST on 19 August, AfriHG Stage-A a2 job `1246940` reached
  exactly `6164/15410`, completed full declared validation over all `3,082`
  rows at health-only loss `2.188938227510545`, and entered its frozen exact
  callback with automatic generation batch size 64. The loss is not selector
  evidence; a2 retains validated checkpoint 3082 at mean chrF
  `22.763135043827063` until the step-6164 artifact completes around
  `20:05--20:20`. A1 `1246939` remained healthy near `14047/15410`,
  retaining audited checkpoint 12328 at mean chrF `24.316085107021614`,
  with its final step boundary still near `20:32` before full validation and
  its final exact callback. Operationally AfriHG is one complete plus two
  running; scientifically it remains `1/3` terminal-valid with no winner
  frozen. Both jobs are clean A100-40GB work; quota is `88.6%/40.1%`,
  Kombuys remains read-only with RTX 5090 untouched, held-out adapter
  evaluation is `0`, Sheet E/F/G are blank, and
  General/Monolingual/publication remain blocked.

- At 18:21 SAST on 19 August, AfriHG Stage-A a1 job `1246939` had completed
  its frozen step-12328 exact callback and resumed healthy training near
  `12923/15410` on `srvrocgpu010` A100-40GB. Its exact 128-row artifact has
  `64/64` Xho/Zul coverage, no empty or normalized-empty output, `64/63`
  unique predictions, and a clean BOS/assistant-marker contract. Xho/Zul
  chrF is `23.372141107936447/25.260029106106785`; mean
  `24.316085107021614` exactly matches newly retained checkpoint 12328 and
  narrowly improves a1 checkpoint 9246. Artifact and trainer-state hashes
  are `b097141d7926648a5b87b3a75769c8b342699ebc0ac7af243fa12bfb2f12ea5c`
  and `37e19be9ba81365273f6d5b4f7be521cd335d63825e392ad5c329cc59bcd16ab`.
  This is within-run validation evidence only: a1 is not terminal-valid and
  no AfriHG winner is frozen. A2 `1246940` was also RUNNING cleanly near
  `5456/15410`; a1/a2 targeted fault scans were empty. Operationally AfriHG
  remains one complete plus two running; scientifically it is `1/3`
  terminal-valid. A1's final step boundary is tentatively near `20:32`,
  followed by its final exact callback; a2's step-6164 boundary is around
  `19:00`. Quota is `88.6%/40.1%`, all owned jobs are A100-40GB, Kombuys
  remains read-only with RTX 5090 untouched, held-out adapter evaluation is
  `0`, Sheet E/F/G are blank, and General/Monolingual/publication remain
  blocked.

- At 16:46 SAST on 19 August, AfriHG Stage-A a2 job `1246940` completed its
  frozen step-3082 exact callback and resumed clean training past
  `3658/15410` on `srvrocgpu010` A100-40GB. Its exact 128-row artifact has
  `64/64` Xho/Zul coverage, no empty or normalized-empty output, `64/64`
  unique predictions per language, and a clean BOS/assistant-marker contract.
  Xho/Zul chrF is `21.862952647458833/23.663317440195296`; mean
  `22.763135043827063` exactly matches retained checkpoint 3082. Artifact and
  trainer-state hashes are
  `00adcb61c3432a3c448db240e3da7ed5657e2e962d538625a35f9bc15c9ef761` and
  `14c1b5f4f4b866b454072a2cdf70def39b5692f632598b7a261ef448aaee0182`.
  This is within-run validation evidence only: a2 is not terminal-valid and
  no AfriHG winner is frozen. A1 `1246939` reached step 12328, completed full
  declared validation at health-only loss `2.2190199647144713`, and entered
  its frozen exact callback with a tentative artifact ETA of
  `17:45--18:00`. Operationally AfriHG is one complete plus two running;
  scientifically it remains `1/3` terminal-valid. Quota is `88.6%/40.1%`,
  all owned jobs are A100-40GB, Kombuys remains read-only with RTX 5090
  untouched, held-out adapter evaluation is `0`, Sheet E/F/G are blank, and
  General/Monolingual/publication remain blocked.

- At 16:16 SAST on 19 August, AfriHG Stage-A a2 job `1246940` remained
  RUNNING cleanly in its frozen step-3082 exact callback on `srvrocgpu010`
  A100-40GB. No completed artifact or targeted fault marker existed, shifting
  its exact 128-row artifact ETA slightly to `16:20--16:35`. A1 `1246939`
  was healthy near `11824/15410`, retaining audited checkpoint 9246 at mean
  chrF `24.23494200032799`, about 26 minutes from its step-12328 boundary
  near `16:42`. Operationally AfriHG is one complete plus two running;
  scientifically it remains `1/3` terminal-valid with no winner frozen.
  Quota is `88.6%/40.1%`, all owned jobs are A100-40GB, Kombuys remains
  read-only with RTX 5090 untouched, held-out is `0`, Sheet E/F/G are blank,
  and General/Monolingual/publication remain blocked.

- At 15:46 SAST on 19 August, AfriHG Stage-A a2 job `1246940` remained
  healthy in its frozen step-3082 exact callback on `srvrocgpu010`
  A100-40GB. Its second language phase established automatic generation batch
  size 64 at `15:36:04`; no complete artifact or targeted fault marker
  existed, keeping the exact 128-row artifact ETA around `16:10--16:25`.
  A2 still has no selector evidence. A1 `1246939` was healthy near
  `11254/15410`, retaining audited checkpoint 9246 at mean chrF
  `24.23494200032799`; its step-12328 boundary is tentatively due around
  `16:42`, with the exact artifact around `17:45--18:00`. Operationally
  AfriHG is one complete plus two running; scientifically it remains `1/3`
  terminal-valid with no winner frozen. Quota is `88.6%/40.1%`, all owned
  jobs are A100-40GB, Kombuys remains read-only with RTX 5090 untouched,
  held-out is `0`, Sheet E/F/G are blank, and
  General/Monolingual/publication remain blocked.

- At 15:16 SAST on 19 August, AfriHG Stage-A a2 job `1246940` had reached
  exactly `3082/15410` cleanly on `srvrocgpu010` A100-40GB. Full declared
  validation covered all `3,082` rows at health-only loss
  `2.233357958635829` in `140.3220 s`, and its frozen exact callback
  established automatic generation batch size 64 at `15:09:52`. No complete
  step-3082 artifact or targeted fault marker existed yet, keeping its exact
  128-row artifact ETA around `16:10--16:25`; the loss is not selector
  evidence. A1 `1246939` remained healthy near `10682/15410`, retaining
  audited checkpoint 9246 at mean chrF `24.23494200032799`. Operationally
  AfriHG is one complete plus two running; scientifically it remains `1/3`
  terminal-valid with no winner frozen. Quota is `88.6%/40.1%`, all owned
  jobs are A100-40GB, Kombuys remains read-only with RTX 5090 untouched,
  held-out is `0`, Sheet E/F/G are blank, and
  General/Monolingual/publication remain blocked.

- At 14:46 SAST on 19 August, AfriHG Stage-A jobs `1246939` and `1246940`
  remained healthy and RUNNING on `srvrocgpu010` A100-40GB. A1 was near
  `10124/15410`, retaining its audited validation-only checkpoint 9246 at
  mean chrF `24.23494200032799`. A2 was near `2704/15410` at about
  `3.04 s/step`, roughly 19 minutes from its first full-validation boundary
  at step 3082 around `15:05`; its frozen exact artifact remains tentatively
  due around `16:10--16:25`. A2 has no selector evidence yet. Operationally
  AfriHG is one complete plus two running; scientifically it remains `1/3`
  terminal-valid with no winner frozen. Quota is `88.6%/40.1%`, all owned
  jobs are A100-40GB, Kombuys remains read-only with RTX 5090 untouched,
  held-out is `0`, Sheet E/F/G are blank, and
  General/Monolingual/publication remain blocked.

- At 14:17 SAST on 19 August, AfriHG Stage-A a1 job `1246939` had completed
  its frozen step-9246 exact callback and resumed training on
  `srvrocgpu010` A100-40GB. Its 128-row artifact covers exactly `64` Xho and
  `64` Zul, has no empty/whitespace-only raw or normalized-empty output, and
  has `64/64` unique predictions per language. Xho/Zul chrF is
  `23.517448141947057/24.952435858708927`; mean chrF `24.23494200032799`
  exactly matches retained checkpoint 9246 and supersedes checkpoint 6164
  within a1. Artifact and trainer-state SHA-256 values are
  `26aec28ad225009aa6295ea7059e8c365cf70c1e49a1f1ed5eb0f02adfbee9ff`
  and `92ce92d557b3c9a644cef34445329080f529a135c26eb60f49082de46caf609d`;
  all prompts pass the BOS/assistant-marker contract. This is within-run
  validation evidence only: a1 is not terminal-valid and no winner is
  selected. A2 `1246940` remained healthy near `2107/15410`, with its first
  exact artifact tentatively due around `16:10--16:25`. Operationally
  AfriHG is one complete plus two running; scientifically it is `1/3`
  terminal-valid. Quota is `88.6%/40.1%`, all jobs are A100-40GB, Kombuys
  remains read-only with RTX 5090 untouched, held-out is `0`, Sheet E/F/G are
  blank, and General/Monolingual/publication remain blocked.

- At 13:46 SAST on 19 August, AfriHG Stage-A a1 job `1246939` remained
  healthy in its frozen step-9246 exact callback on `srvrocgpu010`
  A100-40GB. Its second language phase established automatic batch size 64 at
  `13:26:44`; no exact JSONL or targeted fault marker existed yet, keeping
  the complete-artifact ETA around `14:00--14:15`. A2 `1246940` was healthy
  near `1507/15410`, with its first exact artifact tentatively due around
  `16:10--16:25`. Operationally AfriHG is one complete plus two running;
  scientifically it is `1/3` terminal-valid. Quota is `88.6%/40.1%`, all
  jobs are A100-40GB, Kombuys remains read-only with RTX 5090 untouched,
  held-out is `0`, Sheet E/F/G are blank, and General/Monolingual/publication
  remain blocked.

- At 13:16 SAST on 19 August, AfriHG Stage-A a1 job `1246939` reached
  `9246/15410` cleanly on `srvrocgpu010` A100-40GB. Full declared validation
  covered all `3,082` rows at health-only loss `2.211229248839025` in
  `139.0600 s`; its frozen exact callback established automatic batch size 64
  at `12:59:09`. No exact JSONL or targeted fault marker existed yet, keeping
  the complete-artifact ETA around `14:00--14:15`; checkpoint 6164 remained
  a1's eligible validation-only best. A2 `1246940` was healthy near
  `910/15410`, with its first exact artifact tentatively due around
  `16:10--16:25`. Operationally AfriHG is one complete plus two running;
  scientifically it is `1/3` terminal-valid. Quota is `88.6%/40.1%`, all
  jobs are A100-40GB, Kombuys remains read-only with RTX 5090 untouched,
  held-out is `0`, Sheet E/F/G are blank, and General/Monolingual/publication
  remain blocked.

- At 12:48 SAST on 19 August, AfriHG Stage-A a0 job `1246938` was
  scientifically terminal-valid after completing `0:0` at `12:26:26` in
  `19:08:18`. Its terminal step-15410 artifact has exact `128`-row, `64/64`
  language coverage, no empty/whitespace-only raw or normalized-empty output,
  `64/64` unique predictions per language, and a clean BOS/assistant-marker
  contract. Terminal Xho/Zul chrF is
  `21.82177467761243/23.335156227554975`; mean `22.578465452583703` is below
  retained checkpoint 12328 best `22.843278831141802`. Terminal artifact,
  retained state, and trial SHA-256 values are
  `a8b785757a497e8780c5d0b5f609e42ffa2dbcc63a43ea826165afc7ca483b75`,
  `6ceeefc3bfc763016687c9966467fdbc15c8b28ce3fe32b510b9a6ae2651dbab`,
  and `01d022dc5d714dd4844fee33b504b0b2af13cdff26600cfcad0c5ed8f3a76f62`.
  A local read-only comparison proved all `424/424` tensors (`71,762,560`
  values) exactly equal between retained checkpoint and `final_adapter`;
  retained/final weight hashes are
  `0a9f21a998c9c4dcbc54e0db9f9f0b5f85c69309f076928809cd6da1408d8350`/
  `ab7331f5d0c88fe102e2d1e8621ed171056959f20588875393b533d7ef66c728`,
  with byte-identical config hash
  `0ca737e9db8182cb61e882fb827f1d9917015e7743c168b5156f1cbe208f3aff`.
  AfriHG advances to `1/3` terminal-valid; no winner is frozen. A2 `1246940`
  started on the freed A100-40GB at `12:26:26` and was healthy past step 355;
  its execution/trial hashes are
  `03ac17c961a2f9ca96644f11b8bd3058f70b6f37fa57b777cb6c4d4b910e5d3d`/
  `a85ed5f5e3e4d15f8e3ffb00157fafc0236cd1363d88797ad764ab505c07ab11`.
  A1 `1246939` remained healthy near `9073/15410`, retaining checkpoint 6164.
  Operationally AfriHG is one complete plus two running; scientifically it is
  `1/3` terminal-valid. Quota is `88.6%/40.1%`, all jobs are A100-40GB,
  Kombuys remains read-only with RTX 5090 untouched, held-out is `0`, Sheet
  E/F/G are blank, and General/Monolingual/publication remain blocked.

- At 12:16 SAST on 19 August, AfriHG Stage-A a0 job `1246938` remained
  healthy in its frozen terminal exact callback on `srvrocgpu010`
  A100-40GB. Its second language phase established automatic batch size 64 at
  `11:50:42`; no exact JSONL or targeted fault marker existed yet, narrowing
  the terminal-artifact ETA to `12:20--12:35`. A0 remains non-terminal-valid
  until exact reconciliation. A1 `1246939` was healthy near `8501/15410`,
  retaining validated checkpoint 6164, with its step-9246 boundary due near
  `12:55`; a2 `1246940` remained Resources-pending with estimate `14:10:13`.
  AfriHG is operationally two running plus one pending but scientifically
  `0/3` terminal-valid. Quota is `88.6%/39.8%`, all jobs are A100-40GB,
  Kombuys remains read-only with RTX 5090 untouched, held-out is `0`, Sheet
  E/F/G are blank, and General/Monolingual/publication remain blocked.

- At 11:46 SAST on 19 August, AfriHG Stage-A a0 job `1246938` had completed
  terminal full declared validation over all `3,082` rows at health-only loss
  `2.25601756735176` in `140.8837 s` and entered the frozen terminal exact
  callback on `srvrocgpu010` A100-40GB. Automatic generation batch size 64
  was established at `11:21:02`; no exact JSONL or targeted fault marker
  existed yet, keeping the terminal-artifact ETA around `12:20--12:35`. The
  health-only loss is not selector evidence, and a0 is not terminal-valid
  until exact reconciliation. A1 `1246939` was healthy near `7924/15410`,
  retaining validated checkpoint 6164; a2 `1246940` remained Resources-
  pending with estimate `14:10:13`. AfriHG is operationally two running plus
  one pending but scientifically `0/3` terminal-valid. Quota is
  `88.6%/39.8%`, all jobs are A100-40GB, Kombuys remains read-only with RTX
  5090 untouched, held-out is `0`, Sheet E/F/G are blank, and
  General/Monolingual/publication remain blocked.

- At 11:17 SAST on 19 August, AfriHG Stage-A a0 job `1246938` reached exactly
  `15410/15410` training steps cleanly on `srvrocgpu010` A100-40GB and
  entered the frozen terminal evaluation path. No terminal validation metric,
  exact JSONL, or targeted fault marker existed yet; allowing the full
  declared pass and exact callback keeps the terminal-artifact ETA around
  `12:20--12:35`. Checkpoint 12328 remains a0's latest eligible within-run
  selector evidence, and a0 is not terminal-valid yet. A1 `1246939` remained
  healthy near `7348/15410`, retaining validated checkpoint 6164; a2
  `1246940` remained Resources-pending with estimate `14:10:13`. AfriHG is
  operationally two running plus one pending but scientifically `0/3`
  terminal-valid. Quota is `88.6%/39.8%`, all jobs are A100-40GB, Kombuys
  remains read-only with RTX 5090 untouched, held-out is `0`, Sheet E/F/G are
  blank, and General/Monolingual/publication remain blocked.

- At 10:46 SAST on 19 August, AfriHG Stage-A a0 job `1246938` remained
  healthy near `14822/15410` on `srvrocgpu010` A100-40GB, with no targeted
  fault marker. Its final training boundary is projected around
  `11:16--11:20`, followed by full declared validation and the frozen exact
  callback; terminal-artifact ETA is `12:20--12:35`. A1 `1246939` was healthy
  near `6780/15410`, retaining validated checkpoint 6164; a2 `1246940`
  remained Resources-pending with estimate `14:10:13`. AfriHG is
  operationally two running plus one pending but scientifically `0/3`
  terminal-valid. Quota is `88.6%/39.8%`, all jobs are A100-40GB, Kombuys
  remains read-only with RTX 5090 untouched, held-out is `0`, Sheet E/F/G are
  blank, and General/Monolingual/publication remain blocked.

- At 10:16 SAST on 19 August, AfriHG Stage-A a1 job `1246939` had completed
  its frozen step-6164 exact callback and resumed training on
  `srvrocgpu010` A100-40GB. Its 128-row artifact covers exactly `64` Xho and
  `64` Zul, has no empty/whitespace-only raw or normalized-empty output, and
  has `64/64` unique predictions per language. Xho/Zul chrF is
  `22.77167471216128/23.627781327855192`; mean chrF `23.199728020008237`
  exactly matches retained checkpoint 6164 and supersedes checkpoint 3082
  within a1. Artifact and trainer-state SHA-256 values are
  `59af1a4e3022fd45d100fe590545f0a5bb9c70c5f6b1f28b623201cf0cae2a82`
  and `f0a135e9c85bf7f0086dbd6ed0b750a2b289561d47cc47dad5055345c9469996`;
  all prompts pass the BOS/assistant-marker contract. This is within-run
  validation evidence only: a1 is not terminal-valid and no winner is
  selected. A0 `1246938` remained healthy near `14257/15410`, retaining
  validated checkpoint 12328; a2 `1246940` remained Resources-pending with
  estimate `14:10:13`. AfriHG is operationally two running plus one pending
  but scientifically `0/3` terminal-valid. Quota is `88.6%/39.8%`, all jobs
  are A100-40GB, Kombuys remains read-only with RTX 5090 untouched, held-out
  is `0`, Sheet E/F/G are blank, and General/Monolingual/publication remain
  blocked.

- At 09:46 SAST on 19 August, AfriHG Stage-A a1 job `1246939` remained
  healthy in its frozen step-6164 exact callback on `srvrocgpu010`
  A100-40GB. Its second language phase established automatic batch size 64 at
  `09:38:49`; no exact JSONL or targeted fault marker existed yet, keeping
  the complete-artifact ETA around `10:10--10:25`. A0 `1246938` was healthy
  near `13668/15410`, retaining validated checkpoint 12328; a2 `1246940`
  remained Resources-pending with estimate `14:10:13`. AfriHG is
  operationally two running plus one pending but scientifically `0/3`
  terminal-valid. Quota is `88.6%/39.8%`, all jobs are A100-40GB, Kombuys
  remains read-only with RTX 5090 untouched, held-out is `0`, Sheet E/F/G are
  blank, and General/Monolingual/publication remain blocked.

- At 09:16 SAST on 19 August, AfriHG Stage-A a1 job `1246939` reached
  `6164/15410` cleanly on `srvrocgpu010` A100-40GB. Full declared validation
  covered all `3,082` rows at health-only loss `2.217037688595068` in
  `139.9795 s`; its frozen exact callback established automatic batch size 64
  at `09:11:53`. No step-6164 JSONL or targeted fault marker existed yet,
  keeping the complete-artifact ETA around `10:15--10:30`; checkpoint 3082
  remained a1's only eligible within-run selector evidence. A0 `1246938`
  remained healthy near `13092/15410`, retaining validated checkpoint 12328;
  a2 `1246940` remained Resources-pending with estimate `14:10:13`. AfriHG
  is operationally two running plus one pending but scientifically `0/3`
  terminal-valid. Quota is `88.6%/39.8%`, all jobs are A100-40GB, Kombuys
  remains read-only with RTX 5090 untouched, held-out is `0`, Sheet E/F/G are
  blank, and General/Monolingual/publication remain blocked.

- At 08:43 SAST on 19 August, AfriHG Stage-A a0 job `1246938` had completed
  its frozen step-12328 exact callback and resumed training on
  `srvrocgpu010` A100-40GB. Its 128-row artifact covers exactly `64` Xho and
  `64` Zul, has no empty/whitespace-only raw or normalized-empty output, and
  has `64/64` unique predictions per language. Xho/Zul chrF is
  `22.22350325246101/23.46305440982259`; mean chrF `22.843278831141802`
  exactly matches retained checkpoint 12328 and supersedes checkpoint 9246
  within a0. Artifact and trainer-state SHA-256 values are
  `c11f2a435bbab829d038496c34b7ba131ea1a7f0ab33041ed07c6c41cbfeaba3`
  and `6ceeefc3bfc763016687c9966467fdbc15c8b28ce3fe32b510b9a6ae2651dbab`;
  all prompts pass the BOS/assistant-marker contract. This is within-run
  validation evidence only: a0 is not terminal-valid and no winner is
  selected. At `08:43`, a0 `1246938` and a1 `1246939` were RUNNING at
  elapsed `15:24:31` and `6:11:19`; a2 `1246940` remained Resources-pending
  with estimate `14:10:13`. AfriHG is operationally two running plus one
  pending but scientifically `0/3` terminal-valid. Quota is `88.6%/39.8%`,
  all jobs are A100-40GB, Kombuys remains read-only with RTX 5090 untouched,
  held-out is `0`, Sheet E/F/G are blank, and
  General/Monolingual/publication remain blocked.

- At 07:43 SAST on 19 August, AfriHG Stage-A a0 job `1246938` reached
  `12328/15410` cleanly on `srvrocgpu010` A100-40GB. Full declared validation
  covered all `3,082` rows at health-only loss `2.2554911044725423` in
  `139.1445 s`; its frozen exact callback established automatic batch size 64
  at `07:29:03`. No exact JSONL or fault marker existed yet, keeping the
  complete-artifact ETA around `08:25--08:40`; checkpoint 9246 remained the
  only eligible within-run selector evidence. A1 `1246939` was healthy past
  `4556/15410`, retaining validated checkpoint 3082; a2 `1246940` remained
  Resources-pending with estimate `14:10:13`. AfriHG is operationally two
  running plus one pending but scientifically `0/3` terminal-valid. Quota is
  `88.6%/39.8%`, all jobs are A100-40GB, Kombuys remains read-only with RTX
  5090 untouched, held-out is `0`, Sheet E/F/G are blank, and
  General/Monolingual/publication remain blocked.

- At 07:13 SAST on 19 August, AfriHG Stage-A a0 job `1246938` remained
  healthy near `12105/15410` on `srvrocgpu010` A100-40GB, about 223 training
  steps from its frozen step-12328 boundary. Current throughput projects the
  boundary near `07:25` and the complete exact artifact around
  `08:25--08:40`. A1 `1246939` was healthy near `3991/15410`, retaining
  validated checkpoint 3082; a2 `1246940` remained Resources-pending with
  estimate `14:10:13`. AfriHG is operationally two running plus one pending
  but scientifically `0/3` terminal-valid. Quota is `88.6%/39.8%`, all jobs
  are A100-40GB, Kombuys remains read-only with RTX 5090 untouched, held-out
  is `0`, Sheet E/F/G are blank, and General/Monolingual/publication remain
  blocked.

- At 06:43 SAST on 19 August, AfriHG Stage-A a1 job `1246939` had completed
  its frozen step-3082 exact callback and resumed past `3420/15410` on
  `srvrocgpu010` A100-40GB. Its 128-row artifact covers exactly `64` Xho and
  `64` Zul, has no empty/whitespace-only raw or normalized-empty output, and
  has `64/64` unique predictions per language. Xho/Zul chrF is
  `21.14544404556832/23.06803041933297`; mean chrF `22.106737232450644`
  exactly matches retained checkpoint 3082. Artifact and trainer-state
  SHA-256 values are
  `05d2969eddf94ad424067a4806d48ddae2e804ab4058b96ff95fd349fc82c0c9`
  and `086ef14fa6fcb1f4f5e84b487a98279152059f026a4ca346f82be5cda0956adb`;
  all prompts pass the BOS/assistant-marker contract. This is within-run
  validation evidence only: a1 is not terminal-valid and no winner is
  selected. A0 `1246938` remained healthy past `11536/15410`, retaining
  validated checkpoint 9246; a2 `1246940` remained Resources-pending with
  estimate `14:10:13`. AfriHG is operationally two running plus one pending
  but scientifically `0/3` terminal-valid. Quota is `88.6%/39.8%`, all jobs
  are A100-40GB, Kombuys remains read-only with RTX 5090 untouched, held-out
  is `0`, Sheet E/F/G are blank, and General/Monolingual/publication remain
  blocked.

- At 06:13 SAST on 19 August, AfriHG Stage-A a1 job `1246939` remained
  healthy in its frozen step-3082 exact callback on `srvrocgpu010`
  A100-40GB. Its second language phase established batch size 64 at
  `05:49:17`; no exact JSONL or fault marker existed yet, revising the
  complete-artifact ETA slightly to `06:20--06:35`. A0 `1246938` remained
  healthy past `10964/15410`, retaining validated checkpoint 9246; a2
  `1246940` remained Resources-pending with estimate `14:10:13`. AfriHG is
  operationally two running plus one pending but scientifically `0/3`
  terminal-valid. Quota is `88.6%/39.8%`, all jobs are A100-40GB, Kombuys
  remains read-only with RTX 5090 untouched, held-out is `0`, Sheet E/F/G
  are blank, and General/Monolingual/publication remain blocked.

- At 05:43 SAST on 19 August, AfriHG Stage-A a1 job `1246939` completed full
  declared validation over all `3,082` rows at health-only loss
  `2.26994530679652` in `139.6339 s`, then entered its frozen step-3082 exact
  callback with automatic batch size 64 at `05:18:01`. No exact JSONL or
  fault marker existed yet; its complete-artifact ETA remained
  `06:15--06:30`. A0 `1246938` remained healthy past `10395/15410`, retaining
  validated checkpoint 9246; a2 `1246940` remained Resources-pending with
  estimate `14:10:13`. AfriHG is operationally two running plus one pending
  but scientifically `0/3` terminal-valid. Quota is `88.6%/39.8%`, all jobs
  are A100-40GB, Kombuys remains read-only with RTX 5090 untouched, held-out
  is `0`, Sheet E/F/G are blank, and General/Monolingual/publication remain
  blocked.

- At 05:13 SAST on 19 August, AfriHG Stage-A a1 job `1246939` reached
  `3082/15410` cleanly around `05:11` on `srvrocgpu010` A100-40GB and entered
  its first full declared validation. No step-3082 exact JSONL or fault marker
  existed yet; its complete-artifact ETA is tentatively `06:15--06:30`. A0
  `1246938` remained healthy past `9821/15410`, retaining validated checkpoint
  9246 at mean chrF `22.440017494077004`; a2 `1246940` remained
  Resources-pending with estimate `14:10:13`. AfriHG is operationally two
  running plus one pending but scientifically `0/3` terminal-valid. Quota is
  `88.6%/39.8%`, all jobs are A100-40GB, Kombuys remains read-only with RTX
  5090 untouched, held-out is `0`, Sheet E/F/G are blank, and
  General/Monolingual/publication remain blocked.

- At 04:43 SAST on 19 August, AfriHG Stage-A a0 job `1246938` completed its
  frozen step-9246 exact callback and resumed past `9266/15410` on
  `srvrocgpu010` A100-40GB. Its 128-row artifact covers exactly `64` Xho and
  `64` Zul, has no empty/whitespace-only raw or normalized-empty output, and
  has `64/64` unique predictions per language. Xho/Zul chrF is
  `22.038503951986627/22.841531036167382`; mean chrF
  `22.440017494077004` exactly matches retained checkpoint 9246. Artifact and
  trainer-state SHA-256 values are
  `39ab1091aec93f691b653e74e70b1bb70edb4bd8d8e7bb5fc539fa0a5984bc55`
  and `c2128f4a7545b33b8acd7c01ae292525bd3aedccd150c89fc58e11daabd6abc8`.
  All prompts pass the BOS/assistant-marker contract, and an independent Luna
  audit agrees on every count, hash, metric, and retained-checkpoint field.
  This is within-run validation evidence only: a0 is not terminal-valid and
  no winner is selected. A1 `1246939` remained healthy near `2505/15410`;
  a2 `1246940` remained Resources-pending with estimate `14:10:13`. AfriHG is
  operationally two running plus one pending but scientifically `0/3`
  terminal-valid. Quota is `88.6%/39.8%`, all jobs are A100-40GB, Kombuys
  remains read-only with RTX 5090 untouched, held-out is `0`, Sheet E/F/G
  are blank, and General/Monolingual/publication remain blocked.

- At 04:13 SAST on 19 August, AfriHG Stage-A a0 job `1246938` remained
  healthy in its frozen step-9246 exact callback on `srvrocgpu010`
  A100-40GB. Its second language phase established batch size 64 at
  `04:07:04`; no step-9246 JSONL or fault marker existed yet, keeping the
  complete-artifact ETA around `04:35--04:50`. Checkpoint 6164 remains the
  only eligible within-run selector evidence. A1 `1246939` was healthy near
  `1927/15410`; a2 `1246940` remained Resources-pending with estimate
  `14:10:13`. AfriHG is operationally two running plus one pending but
  scientifically `0/3` terminal-valid. Quota is `88.6%/39.8%`, all jobs are
  A100-40GB, Kombuys remains read-only with RTX 5090 untouched, held-out is
  `0`, Sheet E/F/G are blank, and General/Monolingual/publication remain
  blocked.

- At 03:43 SAST on 19 August, AfriHG Stage-A a0 job `1246938` reached
  `9246/15410` cleanly on `srvrocgpu010` A100-40GB. Full declared validation
  covered all `3,082` rows at health-only loss `2.2620294312236373` in
  `140.9852 s`; the frozen exact callback is active with automatic batch size
  64, but no step-9246 JSONL exists yet. Its complete artifact is tentatively
  due around `04:35--04:50`; checkpoint 6164 remains the only eligible
  within-run selector evidence. A1 `1246939` remained healthy near
  `1349/15410`, while a2 `1246940` remained Resources-pending with estimate
  `14:10:13`. AfriHG is operationally two running plus one pending but
  scientifically `0/3` terminal-valid. Quota is `88.6%/39.8%`, all jobs are
  A100-40GB, Kombuys remains read-only with RTX 5090 untouched, held-out is
  `0`, Sheet E/F/G are blank, and General/Monolingual/publication remain
  blocked.

- At 03:13 SAST on 19 August, AfriHG Stage-A a0 job `1246938` remained
  healthy near `8850/15410` on `srvrocgpu010` A100-40GB. Its step-9246
  boundary is projected near `03:34`, with the complete exact artifact around
  `04:35--04:50`; retained checkpoint 6164 and mean chrF
  `21.242485538593336` remain the only eligible within-run selector evidence.
  A1 `1246939` was healthy near `778/15410`, while a2 `1246940` remained
  Resources-pending with estimate `14:10:13`. AfriHG is operationally two
  running plus one pending but scientifically `0/3` terminal-valid. Quota is
  `88.6%/39.8%`, all jobs are A100-40GB, Kombuys remains read-only with RTX
  5090 untouched, held-out is `0`, Sheet E/F/G are blank, and
  General/Monolingual/publication remain blocked.

- At 02:43 SAST on 19 August, AfriHG Stage-A a1 job `1246939` had started at
  `02:31:20` on a second `srvrocgpu010` A100-40GB. It wrote execution-
  manifest SHA-256
  `32e43a2bd448aadf34002bcc9078768a0db618278439b960561e3c3e3382fd72`,
  verified all `694` source/config files, passed the GatedDeltaNet fast-path
  gate, loaded exactly `24,649/3,082` train/validation rows, and reached about
  `202/15410` near `3.13 s/step` without a fault marker. A0 `1246938`
  remained healthy near `8267/15410`, retaining checkpoint 6164; a2
  `1246940` remained Resources-pending with estimate `14:10:13`. AfriHG is
  operationally two running plus one pending but scientifically `0/3`
  terminal-valid. Quota is `88.6%/39.8%`, all jobs are A100-40GB, Kombuys
  remains read-only with RTX 5090 untouched, held-out is `0`, Sheet E/F/G are
  blank, and General/Monolingual/publication remain blocked.

- At 01:13 SAST on 19 August, AfriHG Stage-A a0 job `1246938` had completed
  its second frozen exact callback and resumed near `6545/15410` on
  `srvrocgpu010` A100-40GB. Its 128-row step-6164 artifact covers exactly
  `64` Xho plus `64` Zul, with no empty/whitespace-only or normalized-empty
  output and `64/64` unique predictions per language. Xho/Zul chrF is
  `20.742026697780208/21.742944379406463`; mean chrF
  `21.242485538593336` exactly matches retained checkpoint 6164. Artifact
  SHA-256 is
  `5fc6747ca7cd80f35a51fa22ae996bf63214e75fe04386cd2b4d432fe21a1e9f`;
  independent Sol/Luna audits agree. This is within-run validation evidence
  only: a0 is not terminal-valid and AfriHG remains open. Jobs
  `1246939/1246940` remain Resources/Priority-pending, with a1 projected at
  `03:18:32` and no a2 estimate. Quota is `88.6%/39.6%`, all jobs are
  A100-40GB, Kombuys remains read-only with RTX 5090 untouched, held-out is
  `0`, Sheet E/F/G are blank, and General/Monolingual/publication remain
  blocked.

- At 00:13 SAST on 19 August, AfriHG Stage-A a0 job `1246938` reached step
  `6164/15410` cleanly on `srvrocgpu010` A100-40GB. Full declared validation
  covered all `3,082` rows at health-only loss `2.2817668778644142`, exactly
  `1305` Xho and `1777` Zul, in `140.4197 s`; the frozen 128-prompt exact
  callback is active with automatic batch size 64 and no fault marker. No
  step-6164 exact artifact exists yet, so step 3082 remains the only
  scientific selector evidence and a0 is not terminal-valid. Jobs
  `1246939/1246940` remain Resources/Priority-pending, with a1 projected at
  `03:18:32` and no a2 estimate. Quota is `88.6%/39.6%`, all jobs are
  A100-40GB, Kombuys remains read-only with RTX 5090 untouched, held-out is
  `0`, Sheet E/F/G are blank, and corrected General, Monolingual, and
  publication remain blocked.

- At 21:44 SAST on 18 August, AfriHG Stage-A a0 job `1246938` remained
  healthy and fault-free at `3828/15410` on `srvrocgpu010` A100-40GB, near
  `3.07 s/step`. Its step-3082 exact artifact remains the only scientifically
  valid within-run checkpoint; a0 is not terminal-valid and AfriHG remains
  open. Jobs `1246939/1246940` remain Resources/Priority-pending, with a1
  projected at `03:18:32` on 19 August and no a2 estimate. All three request
  one `gpu:ampere` A100-40GB and eight CPUs; quota is `88.6%/39.6%`, Kombuys
  remains read-only with RTX 5090 untouched, held-out is `0`, Sheet E/F/G are
  blank, and global freeze/publication gates remain closed.

- At 21:08 SAST on 18 August, AfriHG Stage-A a0 job `1246938` had a clean,
  scientifically valid first exact checkpoint at step 3082 and resumed near
  `3136/15410` on `srvrocgpu010` A100-40GB. Its 128-row artifact is exactly
  `64` Xho plus `64` Zul, with no empty or whitespace-only raw output and
  `64/64` unique predictions per language. Xho/Zul chrF is
  `20.87606798645913/21.52051846942758`; mean chrF
  `21.198293227943353` exactly matches retained `checkpoint-3082`. Artifact
  SHA-256 is
  `2f71557104f7c27ce60bdd890dcc21602c0768edc8f4918bb27a718576cf6e0c`.
  This is within-run validation evidence only: a0 is not terminal-valid and
  AfriHG remains open. Jobs `1246939/1246940` remain Resources/Priority-
  pending; a1 is projected at `03:18:32` on 19 August and a2 has no estimate.
  All jobs are A100-40GB, quota is `88.6%/39.6%`, Kombuys remains read-only
  with RTX 5090 untouched, held-out is `0`, Sheet E/F/G are blank, and global
  freeze/publication gates remain closed.

- At 20:44 SAST on 18 August, AfriHG Stage-A a0 job `1246938` remained
  healthy in its step-3082 frozen exact callback on `srvrocgpu010` A100-40GB.
  The second language phase began at `20:32:58`, fresh generation activity
  continued through `20:41:03`, and no fault marker appeared. No exact JSONL
  exists yet; because automatic batch-size probing plus beam generation took
  about 33 minutes for the first language, the revised artifact ETA is
  `21:00--21:15`. Jobs `1246939/1246940` remain Priority-pending; a1 is
  projected at `06:18:00` on 19 August and a2 has no estimate. AfriHG remains
  scientifically `0/3` terminal-valid. All three jobs use A100-40GB, quota is
  `88.6%/39.6%`, Kombuys remains read-only with RTX 5090 untouched, held-out
  is `0`, Sheet E/F/G are blank, and global freeze/publication gates remain
  closed.

- At 20:14 SAST on 18 August, AfriHG Stage-A a0 job `1246938` completed its
  first full validation-loss boundary at step 3082 on `srvrocgpu010`
  A100-40GB: `eval_loss=2.3395319597663855` over all `3082` declared rows,
  exactly `1305` Xho and `1777` Zul. This is operational evidence only, not
  the frozen selector. Its 128-prompt beam callback is active, with fresh
  generation activity through `20:09:32`, no fault marker, and no exact
  artifact yet; the artifact is expected around `20:15--20:25`. Jobs
  `1246939/1246940` remain Priority-pending; a1 is estimated at `06:18:00`
  on 19 August and a2 has no estimate. AfriHG remains scientifically `0/3`
  terminal-valid. All three jobs use A100-40GB, quota is `88.6%/39.6%`,
  Kombuys remains read-only with RTX 5090 untouched, held-out is `0`, Sheet
  E/F/G are blank, and global freeze/publication gates remain closed.

- At 19:44 SAST on 18 August, AfriHG Stage-A a0 job `1246938` remained
  healthy and fault-free at `2826/15410` on `srvrocgpu010` A100-40GB, running
  near `3.04 s/step`. Its first epoch boundary is due around `19:57`; the
  first exact artifact remains tentatively due `20:40--21:30 SAST`. Jobs
  `1246939/1246940` remain Priority-pending without model/data access;
  Slurm's a1 estimate slipped to `06:18:00 SAST` on 19 August and a2 has no
  estimate. AfriHG remains scientifically `0/3` terminal-valid. These are the
  only owned jobs, all A100-40GB; home/scratch quota is `88.6%/39.6%`,
  Kombuys is read-only with RTX 5090 untouched, held-out is `0`, Sheet E/F/G
  are blank, and global freeze/publication gates remain closed.

- At 18:43 SAST on 18 August, AfriHG Stage-A a0 job `1246938` remained
  healthy and fault-free at about `1628/15410` on `srvrocgpu010` A100-40GB,
  running near `2.98 s/step`. Its first epoch boundary remains near `19:56`
  and first exact artifact is tentatively due `20:40--21:30 SAST`. Jobs
  `1246939/1246940` remain Resources/Priority-pending without model/data
  access; Slurm's a1 estimate regressed to `03:18:32 SAST` on 19 August and
  a2 has no estimate. AfriHG remains scientifically `0/3` terminal-valid.
  These are the only owned jobs, all A100-40GB; home/scratch quota is
  `88.6%/39.6%`, Kombuys is read-only with RTX 5090 untouched, held-out is
  `0`, Sheet E/F/G are blank, and global freeze/publication gates remain
  closed.

- At 17:43 SAST on 18 August, AfriHG Stage-A a0 job `1246938` was healthy at
  about `453/15410` steps on `srvrocgpu010` A100-40GB, with the immutable
  694-file manifest and GatedDeltaNet fast-kernel gate passed and no targeted
  fault marker. At `3.1--3.2 s/step`, the first epoch boundary is near
  `20:00`; the tentative first complete exact-artifact window is
  `20:40--21:30 SAST`. Jobs `1246939/1246940` remain
  Resources/Priority-pending; Slurm tentatively projects a1 at `19:09:50`
  and gives a2 no estimate. AfriHG is scientifically `0/3` terminal-valid.
  These are the only owned jobs, all A100-40GB; home/scratch quota is
  `88.6%/39.6%`, Kombuys remains read-only with RTX 5090 untouched, held-out
  remains `0`, Sheet E/F/G remain blank, and global freeze/publication gates
  remain closed.

- At 17:18 SAST on 18 August, NER a2 seed-13 job `1245393` completed cleanly
  `0:0` after `16:36:09`. Its terminal step-7574 mean exact span F1 is
  `0.6733876882791980`; the second frozen threshold miss preserves retained
  checkpoint 6492 at `0.6774368136184235`. The 192-row artifact reconciles
  exactly at 64 rows per language with no literal empty raw outputs or parser
  failures; debug SHA-256 is
  `80c82291449e84b07b353d305e7791137c3494fd7d52de77a8a24dbc67efc4af`
  and final-adapter SHA-256 is
  `8261fc03703a34b10150d6903043617c561b45476ead0caad98a7582e8f486ec`.
  NER is now `4/4` terminal-valid and its validation-only three-seed ranking
  selects b7 over a2 at `0.6825166742032610` versus `0.6778547247706191`;
  ranking-artifact SHA-256 is
  `fa0f76dcf79f8907a72dcc2bf658dc5dfdd043df962f125f237f4febafbcb6fd`.
  AfriHG Stage-A jobs `1246938/1246939/1246940` were then submitted once from
  the immutable 694-file snapshot. A0 `1246938` is running on A100-40GB and
  passed manifest and GatedDeltaNet fast-kernel gates; a1/a2 are
  Resources/Priority-pending. AfriHG is scientifically `0/3` terminal-valid.
  Home/scratch quota is `88.6%/39.4%`; there is no owned A100-80GB/L40S work,
  Kombuys stays read-only with RTX 5090 untouched, held-out remains `0`, Sheet
  E/F/G remain blank, and global freeze/publication gates remain closed.

- At 16:42 SAST on 18 August, NER a2 seed-87 job `1245394` completed cleanly
  `0:0` after `16:16:36`, making NER scientifically `3/4` terminal-valid. Its
  step-7574 exact artifact scores mean span F1 `0.6766047840600772`; the
  `0.0000560398915701` gain over step 6492 is below frozen threshold `0.001`,
  so the second threshold miss correctly terminated training while trainer
  state retained numerical-best checkpoint 7574. The 192-row artifact has
  exact 64-per-language coverage, no literal empty raw output or parser
  failure, an independently reconciled mean, and SHA-256
  `77e3efd49ecef909362f4d8cf24cbbfb2044521ac2bc06e9e35c12c8913681a1`;
  retained adapter-model SHA-256 is
  `9cac1283266e0cc07e8947fa06cacc71feecb54758f57c5f800498f0fa90f784`.
  Seed-13 job `1245393` remains healthy in its matching step-7574 exact
  callback with fresh probes through `16:34`; its artifact is tentatively due
  `17:00--17:20 SAST`, after which it will either terminate on a second miss
  or run the final epoch on a qualifying improvement. It is the only owned
  job, on A100-40GB `gpu:ampere`, with no owned A100-80GB/L40S overlap. Quota
  is home `88.6%`, scratch `39.5%`; held-out remains `0`, Sheet E/F/G remain
  blank, Kombuys remains read-only at its last verified idle state, and no
  winner is frozen.

- At 16:12 SAST on 18 August, NER a2 seed-13 job `1245393` scored mean exact
  span F1 `0.6723811055906449` at step 7033, below retained step-6492 best
  `0.6774368136184235`; checkpoint 6492 remains retained and patience
  advances to `1/2`, matching seed 87. Its 192-row exact artifact has exact
  64-per-language coverage, no literal empty raw output or parser failure,
  an independently reconciled mean, and SHA-256
  `fd4ec8df85f899777e951df0cbe65cf28de3730d2bd6631f13cbdb622170b75e`.
  Seed 13 resumed near `7459/8115`; seed-87 job `1245394` reached step 7574
  and entered its next exact callback with a fresh probe at `16:07`. A
  non-improvement there would satisfy seed 87's frozen patience. The next
  artifacts are tentatively due `16:40--17:30 SAST`. These remain interim
  results: NER is scientifically `2/4` terminal-valid and no winner is
  frozen. The two fast-path jobs are the only owned jobs, both A100-40GB
  `gpu:ampere`, with no owned A100-80GB/L40S overlap. Quota is home `88.6%`,
  scratch `39.4%`; held-out remains `0`, Sheet E/F/G remain blank, Kombuys
  remains read-only at its last verified idle state, and all gates remain
  unchanged.

- At 15:42 SAST on 18 August, NER a2 seed-87 job `1245394` scored mean exact
  span F1 `0.6758961608549887` at step 7033, below retained step-6492 best
  `0.6765487441685071`; checkpoint 6492 remains retained and patience
  advances to `1/2`. Its 192-row exact artifact has exact 64-per-language
  coverage, no literal empty raw output or parser failure, an independently
  reconciled mean, and SHA-256
  `811f7ff7f8c631ed848c8955c283c993a6cdfd6aed4423c69e0d5a6fa2a2c754`.
  Seed 87 resumed near `7247/8115`; seed-13 job `1245393` remains healthy in
  its matching step-7033 callback with fresh probes through `15:39`, and its
  artifact is tentatively due `15:55--16:15 SAST`. These remain interim
  results: NER is scientifically `2/4` terminal-valid and no winner is
  frozen. The two fast-path jobs are the only owned jobs, both A100-40GB
  `gpu:ampere`, with no owned A100-80GB/L40S overlap. Quota is home `88.6%`,
  scratch `39.3%`; held-out remains `0`, Sheet E/F/G remain blank, Kombuys
  remains read-only at its last verified idle state, and all gates remain
  unchanged.

- At 15:12 SAST on 18 August, both NER a2 jobs reached step `7033/8115`.
  Seed-87 job `1245394` completed declared validation over all `10,760` rows
  at provenance-only loss `0.392970134068599` and entered its exact callback,
  with fresh probes through `15:05`; seed-13 job `1245393` is at the matching
  validation boundary. No step-7033 exact artifact exists yet, so both retain
  checkpoint 6492 with patience `0/2`. Expected artifact timing is
  `15:30--16:10 SAST`. Both A100-40GB fast-path jobs remain operationally
  healthy and fault-free, while NER stays scientifically `2/4` terminal-valid
  and no winner is frozen. They are the only owned jobs, with no owned
  A100-80GB/L40S overlap. Quota is home `88.6%`, scratch `39.3%`; held-out
  remains `0`, Sheet E/F/G remain blank, Kombuys remains read-only at its last
  verified idle state, and all gates remain unchanged.

- At 14:42 SAST on 18 August, both NER a2 confirmations improved at the
  step-6492 exact boundary. Job `1245393` seed 13 now retains checkpoint 6492
  at mean span F1 `0.6774368136184235` and job `1245394` seed 87 retains it at
  `0.6765487441685071`; frozen patience is `0/2` for both. Each exact artifact
  has `192` rows, `64` per language, no literal empty raw output or parser
  failure, and an independently reconciled language mean. Debug artifact
  SHA-256 values are
  `287600a8d55420e4d6bd33902a25ae4f95ce9dea309272b769f7d2d12adc633c`
  and
  `54b1fa2e88c2fc211b508527348bea7f29e185ac00ff1de22e5f72a96286d3b8`.
  Both A100-40GB fast-path jobs remain healthy: seed 13 resumed near
  `6517/8115`, while seed 87 resumed near `6892/8115`; next exact artifacts
  are tentatively due `15:30--16:15 SAST`. These remain interim results, so
  NER stays scientifically `2/4` terminal-valid and no winner is frozen.
  They are the only owned jobs, with no owned A100-80GB/L40S overlap. Quota
  is home `88.6%`, scratch `39.3%`; held-out remains `0`, Sheet E/F/G remain
  blank, Kombuys remains read-only at its last verified idle state, and all
  gates remain unchanged.

- At 14:12 SAST on 18 August, NER a2 jobs `1245393/1245394` both reached
  step `6492/8115`, completed declared validation over all `10,760` rows at
  provenance-only losses `0.379897379254763/0.3869488684218169`, and entered
  their exact 192-prompt callbacks. Fresh generation probes continued through
  `14:07/14:09` with no targeted fault marker. No step-6492 exact artifact
  exists yet: seed 13 retains checkpoint 5951 with patience `0/2`, while seed
  87 retains checkpoint 5410 with patience `1/2`. Expected artifact timing is
  `14:25--15:00 SAST`. Both A100-40GB fast-path jobs remain operationally
  healthy, while NER stays scientifically `2/4` terminal-valid and no winner
  is frozen. They are the only owned jobs, with no owned A100-80GB/L40S
  overlap. Quota is home `88.6%`, scratch `39.3%`; held-out remains `0`,
  Sheet E/F/G remain blank, Kombuys remains read-only at its last verified
  idle state, and all gates remain unchanged.

- At 13:42 SAST on 18 August, NER a2 seed-13 job `1245393` improved at
  step 5951 to mean exact span F1 `0.6755429869223111`, retaining checkpoint
  5951 and resetting patience from `1/2` to `0/2`. Its 192-row exact artifact
  has exact 64-per-language coverage, no literal empty raw output or parser
  failure, an independently reconciled mean, and SHA-256
  `d71dff7d7ccfb0de9c1453103757074711a820bf66766ecff1f85e8ee2154a7e`.
  Seed 13 resumed near `6161/8115`; seed-87 job `1245394` reached its next
  step-6492 validation boundary. The next artifacts are tentatively due
  `14:20--15:10 SAST`. These remain interim results: NER is scientifically
  `2/4` terminal-valid and no winner is frozen. The two fast-path jobs are
  the only owned jobs, both A100-40GB `gpu:ampere`, with no owned
  A100-80GB/L40S overlap. Quota is home `88.6%`, scratch `39.3%`; held-out
  remains `0`, Sheet E/F/G remain blank, Kombuys remains read-only at its
  last verified idle state, and all gates remain unchanged.

- At 13:12 SAST on 18 August, NER a2 seed-87 job `1245394` scored mean exact
  span F1 `0.6714130986587395` at step 5951, below retained step-5410 best
  `0.6743755031188132`; checkpoint 5410 remains retained and patience
  advances to `1/2`. Its 192-row exact artifact has exact 64-per-language
  coverage, no literal empty raw output or parser failure, an independently
  reconciled mean, and SHA-256
  `28e6d10f7897f6fcdf3f01ccc4b56e7c6fc1400435d20a68be775b3d01a02a84`.
  Seed 87 resumed just after step 5951; seed-13 job `1245393` remains healthy
  in its matching step-5951 callback with fresh probes through `13:05`, and
  its artifact is tentatively due `13:30--13:50 SAST`. These remain interim
  results: NER is scientifically `2/4` terminal-valid and no winner is
  frozen. The two fast-path jobs are the only owned jobs, both A100-40GB
  `gpu:ampere`, with no owned A100-80GB/L40S overlap. Quota is home `88.6%`,
  scratch `39.4%`; held-out remains `0`, Sheet E/F/G remain blank, Kombuys
  remains read-only at its last verified idle state, and all gates remain
  unchanged.

- At 12:42 SAST on 18 August, NER a2 seed-13 job `1245393` scored mean exact
  span F1 `0.6697110944357454` at step 5410, below retained step-4869 best
  `0.6700645113839289`; checkpoint 4869 remains retained and patience
  advances to `1/2`. Its 192-row exact artifact has exact 64-per-language
  coverage, no literal empty raw output or parser failure, an independently
  reconciled mean, and SHA-256
  `beea360ceb0ad66e362670c560f403a488134eeaa1db60581d79d018381128ee`.
  Seed 13 resumed near `5817/8115`; seed-87 job `1245394` reached step 5951
  and entered its next exact callback with a fresh probe at `12:37`. The next
  artifacts are tentatively due `13:10--14:00 SAST`. These remain interim
  results: NER is scientifically `2/4` terminal-valid and no winner is
  frozen. The two fast-path jobs are the only owned jobs, both A100-40GB
  `gpu:ampere`, with no owned A100-80GB/L40S overlap. Quota is home `88.6%`,
  scratch `39.3%`; held-out remains `0`, Sheet E/F/G remain blank, Kombuys
  remains read-only at its last verified idle state, and all gates remain
  unchanged.

- At 12:12 SAST on 18 August, NER a2 seed-87 job `1245394` improved at
  step 5410 to mean exact span F1 `0.6743755031188132`, retaining checkpoint
  5410 with patience `0/2`. Its 192-row exact artifact has exact
  64-per-language coverage, no literal empty raw output or parser failure,
  an independently reconciled mean, and SHA-256
  `82e43c8e60003b98522861bc475566079bb10c3464f04e168e3178424e3fcdf8`.
  Seed 87 resumed near `5627/8115`; seed-13 job `1245393` remains healthy in
  its matching step-5410 exact callback with fresh probes through `12:10`,
  and its artifact is tentatively due `12:20--12:35 SAST`. These remain
  interim results: NER is scientifically `2/4` terminal-valid and no winner
  is frozen. The two fast-path jobs are the only owned jobs, both A100-40GB
  `gpu:ampere`, with no owned A100-80GB/L40S overlap. Quota is home `88.6%`,
  scratch `39.3%`; held-out remains `0`, Sheet E/F/G remain blank, Kombuys
  remains read-only at its last verified idle state, and all gates remain
  unchanged.

- At 11:42 SAST on 18 August, NER a2 seed-13 job `1245393` improved at
  step 4869 to mean exact span F1 `0.6700645113839289`, retaining checkpoint
  4869 with patience `0/2`. Its 192-row exact artifact has exact
  64-per-language coverage, no literal empty raw output or parser failure,
  an independently reconciled mean, and SHA-256
  `624f6ed2d6932ed21d5eac250b8983c7533575d481ddd64f78be97ca5c737db6`.
  Both seed 13 and seed 87 have reached step `5410/8115` and remain healthy
  in the next validation/exact-callback cycle, with fresh probes through
  `11:40/11:34`; the next artifacts are tentatively due
  `12:00--12:30 SAST`. These remain interim results: NER is scientifically
  `2/4` terminal-valid and no winner is frozen. The two fast-path jobs are
  the only owned jobs, both A100-40GB `gpu:ampere`, with no owned
  A100-80GB/L40S overlap. Quota is home `88.6%`, scratch `39.3%`; held-out
  remains `0`, Sheet E/F/G remain blank, Kombuys remains read-only at its
  last verified idle state, and all gates remain unchanged.

- At 11:12 SAST on 18 August, NER a2 seed-87 job `1245394` improved at
  step 4869 to mean exact span F1 `0.6675332242184386`, retaining checkpoint
  4869 and resetting frozen patience from `1/2` to `0/2`. Its 192-row exact
  artifact has 64 rows per language, no literal empty raw output or parser
  failure, an independently reconciled mean, and SHA-256
  `dd1fb6f424ec39490c8a535c653422f28fe89ed4f89d2820861c5cdd27209dff`.
  Seed 87 resumed near `5278/8115`. Seed-13 job `1245393` remains healthy in
  its matching step-4869 exact callback with fresh probes through `11:01`;
  its artifact is expected around `11:13--11:20 SAST`. Both A100-40GB
  fast-path jobs remain healthy and fault-free. These are interim results:
  NER remains scientifically `2/4` terminal-valid and no winner is frozen.
  They are the only owned jobs, with no owned A100-80GB/L40S overlap. Quota
  is home `88.6%`, scratch `39.3%`; held-out remains `0`, Sheet E/F/G remain
  blank, Kombuys remains read-only at its last verified idle state, and all
  gates remain unchanged.

- At 10:42 SAST on 18 August, NER a2 jobs `1245393/1245394` both reached
  step `4869/8115`, completed declared validation over all `10,760` rows at
  provenance-only losses `0.3637493601518936/0.37202444537421586`, and
  entered their exact 192-prompt callbacks. Fresh generation probes continued
  through `10:40/10:39` with no targeted fault marker. No step-4869 exact
  artifact exists yet: seed 13 retains checkpoint 4328 with patience `0/2`,
  while seed 87 retains checkpoint 3787 with patience `1/2`. Expected artifact
  timing is `10:50--11:20 SAST`. Both A100-40GB fast-path jobs remain
  operationally healthy, while NER stays scientifically `2/4` terminal-valid
  and no winner is frozen. They are the only owned jobs, with no owned
  A100-80GB/L40S overlap. Quota is home `88.6%`, scratch `39.3%`; held-out
  remains `0`, Sheet E/F/G remain blank, Kombuys remains read-only at its last
  verified idle state, and all gates remain unchanged.

- At 10:12 SAST on 18 August, NER a2 seed-13 job `1245393` improved at
  step 4328 to mean exact span F1 `0.6548998792345401`, retaining checkpoint
  4328 and resetting frozen patience from `1/2` to `0/2`. Its 192-row
  artifact has exact 64-per-language coverage, no literal empty raw output or
  parser failure, an independently reconciled mean, and SHA-256
  `89094dfb926d4fe1494711b252e6bcb0dc1c906e90f7b991a5420707fe59436d`.
  Seed 13 resumed near `4478/8115`; seed-87 job `1245394` reached its next
  step-4869 validation boundary. Next exact artifacts are tentatively due
  `11:00--11:30 SAST`, with terminal timing still governed by frozen patience.
  Both A100-40GB fast-path jobs remain healthy and fault-free. These are
  interim results: NER remains scientifically `2/4` terminal-valid and no
  winner is frozen. They are the only owned jobs, with no owned
  A100-80GB/L40S overlap. Quota is home `88.6%`, scratch `39.3%`; held-out
  remains `0`, Sheet E/F/G remain blank, Kombuys remains read-only at its last
  verified idle state, and all gates remain unchanged.

- At 09:42 SAST on 18 August, NER a2 seed-87 job `1245394` produced valid
  step-4328 mean exact span F1 `0.6534121964705447`, below its retained
  step-3787 best `0.6576473598211583`. Checkpoint 3787 remains retained and
  frozen patience advances to `1/2`. The 192-row artifact has exact
  64-per-language coverage, no literal empty raw output or parser failure,
  independently reconciled mean, and SHA-256
  `21ce3dd9736868fa56275d4ee926292d362ed54e3ef70c6b49c5885c7e8aadf4`.
  Seed 87 resumed just after step 4328. Seed-13 job `1245393` completed
  declared step-4328 validation at provenance-only loss
  `0.35468309820806226` and remains in its exact callback with fresh probes
  through `09:38`; its artifact is expected around `09:50--10:00 SAST`.
  Both A100-40GB fast-path jobs remain healthy and fault-free. These are
  interim results: NER remains scientifically `2/4` terminal-valid and no
  winner is frozen. They are the only owned jobs, with no owned
  A100-80GB/L40S overlap. Quota is home `88.6%`, scratch `39.4%`; held-out
  remains `0`, Sheet E/F/G remain blank, Kombuys remains read-only at its last
  verified idle state, and all gates remain unchanged.

- At 09:12 SAST on 18 August, NER a2 seed-13 job `1245393` produced a valid
  step-3787 mean exact span F1 of `0.6387472783390758`, below its retained
  step-3246 best `0.6461748976768455`. Checkpoint 3246 remains retained and
  frozen patience advances to `1/2`. The 192-row artifact has exact
  64-per-language coverage, no literal empty raw output or parser failure,
  independently reconciled mean, and SHA-256
  `03aadc062f64aad6c1bcfd8c6469778666f562c7b36d014e3e3acf4799a3b5e1`.
  Seed 13 resumed near `4136/8115`. Seed-87 job `1245394` reached step 4328,
  completed declared validation at provenance-only loss
  `0.3662617509693018`, and entered its exact callback with a fresh probe at
  `09:08`; next artifacts are tentatively due `09:40--10:10 SAST`. Both
  A100-40GB fast-path jobs remain healthy and fault-free. These are interim
  results: NER remains scientifically `2/4` terminal-valid and no winner is
  frozen. They are the only owned jobs, with no owned A100-80GB/L40S overlap.
  Quota is home `88.6%`, scratch `39.3%`; held-out remains `0`, Sheet E/F/G
  remain blank, Kombuys remains read-only at its last verified idle state,
  and all gates remain unchanged.

- At 08:42 SAST on 18 August, NER a2 seed-87 job `1245394` improved at
  step 3787 to mean exact span F1 `0.6576473598211583`, retaining checkpoint
  3787 with patience `0/2`. Its exact artifact has `192` rows, `64` per
  language, no literal empty raw output or parser failure, and SHA-256
  `50f264a373b8698a3fbd3c100b51b2d175ffbec9d51524b4e9d21ed4338bb320`;
  the independently recomputed mean matches trainer state. Seed 87 resumed
  near `3973/8115`. Seed-13 job `1245393` completed declared step-3787
  validation at provenance-only loss `0.36098667981456206` and remains in its
  exact callback with fresh probes through `08:28`; its artifact is expected
  around `08:50--09:00 SAST`. Both A100-40GB fast-path jobs remain healthy
  and fault-free. These are interim results: NER remains scientifically `2/4`
  terminal-valid and no winner is frozen. They are the only owned jobs, with
  no owned A100-80GB/L40S overlap. Quota is home `88.6%`, scratch `39.3%`;
  held-out remains `0`, Sheet E/F/G remain blank, Kombuys remains read-only at
  its last verified idle state, and all gates remain unchanged.

- At 08:12 SAST on 18 August, NER a2 seed-13 job `1245393` improved at
  step 3246 to mean exact span F1 `0.6461748976768455`, retaining checkpoint
  3246 with patience `0/2`. Its 192-row artifact has exact 64-per-language
  coverage, no literal empty raw output or parser failure, and SHA-256
  `6bc90db1ee5cf914d6b81fc4abbd58bfa28db2cc2c3373d46530c2bbcba026d6`;
  the independently recomputed language mean matches trainer state. Seed 13
  reached step 3787. Seed-87 job `1245394` completed its declared step-3787
  validation at provenance-only loss `0.3610744887568251` and entered the
  exact callback, with fresh probes through `08:05`; next artifacts are
  tentatively due `08:20--08:50 SAST`. Both A100-40GB fast-path jobs remain
  healthy and fault-free. These are interim results: NER remains
  scientifically `2/4` terminal-valid and no winner is frozen. They are the
  only owned jobs, with no owned A100-80GB/L40S overlap. Quota is home
  `88.6%`, scratch `39.3%`; held-out remains `0`, Sheet E/F/G remain blank,
  Kombuys remains read-only at its last verified idle state, and all gates
  remain unchanged.

- At 07:42 SAST on 18 August, NER a2 seed-87 job `1245394` improved at
  step 3246 to mean exact span F1 `0.6326330113434602` and now retains
  checkpoint 3246 with patience `0/2`. Its exact artifact has `192` rows,
  `64` per language, no literal empty raw output or parser failure, and SHA-256
  `6c7d21200f31db08f833f6d530a2fda79c1126660408a78780e9d3f6aa6aba72`;
  the independently recomputed language mean matches trainer state. Seed 87
  resumed near `3633/8115`. Seed-13 job `1245393` remains healthy in its
  matching step-3246 exact callback with fresh probes through `07:30`; its
  artifact is expected around `07:43--07:50 SAST`. Both A100-40GB fast-path
  jobs have no targeted fault marker. These are interim results: NER remains
  scientifically `2/4` terminal-valid and no winner is frozen. They are the
  only owned jobs, with no owned A100-80GB/L40S overlap. Quota is home
  `88.6%`, scratch `39.3%`; held-out remains `0`, Sheet E/F/G remain blank,
  Kombuys remains read-only at its last verified idle state, and all gates
  remain unchanged.

- At 07:12 SAST on 18 August, NER a2 jobs `1245393/1245394` both reached
  step `3246/8115`, completed declared validation over all `10,760` rows at
  provenance-only losses `0.33066272381069933/0.3471481550138679`, and
  entered their separate exact 192-prompt callbacks. Fresh generation probes
  continued through `07:07/07:10` with no targeted fault marker. No
  step-3246 exact artifact exists yet, so both still retain checkpoint 2705
  with patience `0/2`; expected artifact timing is `07:20--07:45 SAST`.
  Both A100-40GB fast-path jobs remain operationally healthy, while NER stays
  scientifically `2/4` terminal-valid and no winner is frozen. They are the
  only owned jobs, with no owned A100-80GB/L40S overlap. Quota is home
  `88.6%`, scratch `39.3%`; held-out remains `0`, Sheet E/F/G remain blank,
  Kombuys remains read-only at its last verified idle state, and all gates
  remain unchanged.

- At 06:42 SAST on 18 August, both NER a2 confirmations improved at the
  step-2705 exact boundary. Job `1245393` seed 13 now retains checkpoint 2705
  at mean span F1 `0.6263447401956433` and job `1245394` seed 87 retains it at
  `0.6288979354137386`; frozen patience is `0/2` for both. Each exact artifact
  has `192` rows, `64` per language, no literal empty raw output or parser
  failure, and an independently reconciled language mean. Debug artifact
  SHA-256 values are
  `ae27fb507e8fe8f8dde361c79da2f2f51746be2d1639d00dd3a7fe2d04359bed`
  and
  `e074f236fbad6d1be3da43f6c387e3d27d690bfbcc3add2897d18a2d06fb05c3`.
  Both A100-40GB fast-path jobs remain healthy: seed 13 resumed near
  `2935/8115`, while seed 87 reached the next step-3246 validation boundary;
  next exact artifacts are tentatively due `07:15--07:45 SAST`. These remain
  interim results, so NER stays scientifically `2/4` terminal-valid and no
  winner is frozen. They are the only owned jobs, with no owned
  A100-80GB/L40S overlap. Quota is home `88.6%`, scratch `39.3%`; held-out
  remains `0`, Sheet E/F/G remain blank, Kombuys remains read-only at its last
  verified idle state, and all gates remain unchanged.

- At 06:12 SAST on 18 August, both NER a2 jobs remained healthy inside their
  step-2705 exact callbacks. Declared validation again covered all `10,760`
  rows, with provenance-only losses `0.3368947479361495` for job `1245393`
  and `0.3409823563019139` for job `1245394`; fresh automatic generation
  probes continued through `06:05` and `06:00`, with no targeted fault
  marker. Neither step-2705 artifact existed yet, so checkpoint 2164 and
  patience `0/2` remain unchanged for both seeds. Observed callback timing
  revises the artifact window to about `06:15--06:40 SAST`. Both A100-40GB
  fast-path jobs are operationally healthy, while NER remains scientifically
  `2/4` terminal-valid and no winner is frozen. They are the only owned jobs,
  with no owned A100-80GB/L40S overlap. Quota is home `88.6%`, scratch
  `39.3%`; held-out remains `0`, Sheet E/F/G remain blank, Kombuys remains
  read-only at its last verified idle state, and all gates remain unchanged.

- At 05:42 SAST on 18 August, NER a2 seed-13 job `1245393` completed and
  independently reconciled its step-2164 exact artifact, improving retained
  mean span F1 to `0.5956411041394506`, with Tsn/Xho/Zul
  `0.5865787136876325/0.5823108384457577/0.6180337602849617`. Its 192-row
  artifact SHA-256 is
  `792e84420538b9ddb6b28df78ec6453d9883799d5beaa5f80985f961646d9caf`;
  coverage is `64` rows per language with no literal empty raw output or
  parser failure, and the recomputed mean matches trainer state. Both seeds
  now retain checkpoint 2164 with patience `0/2`. Seed 13 resumed near
  `2567/8115`; seed-87 job `1245394` reached step 2705, completed declared
  validation at provenance-only loss `0.3409823563019139`, and entered its
  exact callback. The next exact artifacts are tentatively due around
  `06:20--06:45 SAST`. These are interim results: NER remains `2/4`
  terminal-valid and no winner is frozen. Both A100-40GB fast-path jobs are
  healthy and the only owned work, with no owned A100-80GB/L40S overlap.
  Quota is home `88.6%`, scratch `39.3%`; held-out remains `0`, Sheet E/F/G
  remain blank, Kombuys remains read-only at its last verified idle state,
  and all gates remain unchanged.

- At 05:12 SAST on 18 August, NER a2 seed-87 job `1245394` completed and
  independently reconciled its step-2164 exact artifact, improving retained
  mean span F1 to `0.5960721696811287`, with Tsn/Xho/Zul
  `0.5945072697899338/0.5718134715025407/0.6218957677509119`. Its exact
  192-row artifact SHA-256 is
  `f4c12ee4479593cd0edfa6f9af81f70cad370140e0e705a3a44bee4816f25d62`;
  it has `64` rows per language, no literal empty raw output, and no parser
  failure, and its recomputed mean matches trainer state. Checkpoint 2164 is
  retained with patience `0/2`, and seed 87 resumed near `2341/8115`.
  Seed-13 job `1245393` remained healthy inside its matching step-2164 exact
  callback after declared validation loss `0.3380058487108649`; fresh probes
  continued through `05:07`, with its artifact expected around
  `05:15--05:30 SAST`. These remain interim results: NER is `2/4`
  terminal-valid and no winner is frozen. Both A100-40GB fast-path jobs are
  healthy and the only owned work, with no owned A100-80GB/L40S overlap.
  Quota is home `88.6%`, scratch `39.3%`; held-out remains `0`, Sheet E/F/G
  remain blank, Kombuys remains read-only at its last verified idle state,
  and all gates remain unchanged.

- At 04:31 SAST on 18 August, NER a2 seed-13 job `1245393` completed and
  independently reconciled its step-1623 exact artifact, improving retained
  mean span F1 to `0.569087134514023`, with Tsn/Xho/Zul
  `0.5734701831882052/0.5484253666953773/0.5853658536584866`. The 192-row
  artifact SHA-256 is
  `76a68338d87a62949e0e024ca073ebcf6abd97e7fdb4670366d2e5be11e8fd0b`;
  coverage is exactly `64` rows per language, with no literal empty raw output
  or parser failure, and the recomputed mean matches trainer state. Both seeds
  now retain checkpoint 1623 with patience `0/2`. Seed 13 resumed near
  `2081/8115`; seed-87 job `1245394` reached step 2164, completed declared
  validation at provenance-only loss `0.33073989697991696`, and entered its
  next exact callback. The next exact artifacts are tentatively due around
  `05:10--05:35 SAST`. These remain interim results: NER is `2/4`
  terminal-valid and no winner is frozen. Both A100-40GB fast-path jobs are
  healthy and are the only owned work, with no owned A100-80GB/L40S overlap.
  Quota is home `88.6%`, scratch `39.3%`; held-out remains `0`, Sheet E/F/G
  remain blank, Kombuys remains read-only at its last verified idle state,
  and all gates remain unchanged.

- At 04:01 SAST on 18 August, NER a2 seed-87 job `1245394` completed and
  independently reconciled its step-1623 exact artifact, improving retained
  mean span F1 to `0.5725123790157849` with Tsn/Xho/Zul
  `0.5664436573526983/0.5478807765927861/0.6032127031018703`. The exact
  192-row artifact has SHA-256
  `559de3d5bd0dc5c8a04498a7e180b0e32272db8d998bbdc741e590400bccdcab`,
  exactly `64` rows per language, no literal empty raw output, and no parser
  failure; its recomputed language mean matches trainer state. Checkpoint
  1623 is retained with patience `0/2`, and the healthy job resumed near
  `1829/8115`. Seed-13 job `1245393` remained healthy inside its matching
  step-1623 callback; its artifact was still absent and expected around
  `04:05--04:20 SAST`. These are interim results: NER remains `2/4`
  terminal-valid confirmations and no winner is frozen. Both are A100-40GB
  fast-path jobs and the only owned work, with no owned A100-80GB/L40S
  overlap. Quota is home `88.6%`, scratch `39.3%`; held-out remains `0`,
  Sheet E/F/G remain blank, Kombuys remains read-only at its last verified
  idle state, and all gates remain unchanged.

- At 03:49 SAST on 18 August, NER a2 jobs `1245393` and `1245394` were both
  healthy inside their step-1623 exact callbacks after complete `10,760`-row
  declared validation. Provenance-only losses were `0.34193392048094795` and
  `0.34355421668977987`; fresh generation probes continued through `03:40`
  and `03:37`, with no targeted fault marker. No step-1623 exact artifact was
  complete yet; observed callback timing puts the likely window around
  `03:50--04:15 SAST`. A transient SSH monitoring connection reset returned
  no state, and the immediate quota-first retry succeeded; no job action was
  taken. Both A100-40GB fast-path jobs remain operational, while NER stays
  scientifically `2/4` terminal-valid confirmations. They are the only owned
  jobs, with no owned A100-80GB/L40S overlap. Quota is home `88.6%`, scratch
  `39.3%`; held-out remains `0`, Sheet E/F/G remain blank, Kombuys remains
  read-only at its last verified idle state, and all gates remain unchanged.

- At 02:58 SAST on 18 August, both NER a2 confirmations completed and
  independently reconciled their step-1082 exact artifacts, improving their
  retained checkpoints with patience still `0/2`. Job `1245393` (seed 13)
  reached mean span F1 `0.4957424333994835`, with Tsn/Xho/Zul
  `0.48288476613530407/0.4777564813880263/0.5265860526751202`; its 192-row
  artifact SHA-256 is
  `aabe0cc3394c455fb4ed27857aeea00b90f1653b70b7be0a8d376e859e3510dc`.
  Job `1245394` (seed 87) reached `0.529091241481777`, with
  `0.5227586206896052/0.5089807162533941/0.5555343875023318`; its artifact
  SHA-256 is
  `b2273b799e8e563d9f0a88845748d6d5c83d7536bd2c1dcc36c0c5b0e7833819`.
  Both artifacts have exactly `192` rows, `64` per language, no literal empty
  raw output, and no parser failure; their recomputed language means match the
  trainer-state metrics exactly. The jobs resumed healthy near `1125/8115`
  and `1453/8115`; terminal timing remains governed by frozen early stopping.
  These are valid interim artifacts, so NER remains `2/4` terminal-valid and
  no winner is frozen. Both are A100-40GB fast-path jobs and the only owned
  work, with no owned A100-80GB/L40S overlap. Quota is home `88.6%`, scratch
  `39.3%`; held-out remains `0`, Sheet E/F/G remain blank, Kombuys remains
  read-only at its last verified idle state, and all gates remain unchanged.

- At 02:28 SAST on 18 August, NER a2 confirmation jobs `1245393` and
  `1245394` remained healthy inside their step-1082 exact callbacks. Declared
  validation again covered all `10,760` rows, with provenance-only losses
  `0.3635498784288598` and `0.3648764287672078`; fresh generation batch-size
  probes continued through `02:26` and `02:25`, with no targeted fault
  marker. Neither second 192-row exact artifact is complete yet; observed
  callback timing puts them around `02:40--03:00 SAST`. Both jobs remain
  operationally running on A100-40GB with the verified GDN fast path, while
  NER remains scientifically `2/4` terminal-valid confirmations. They are the
  only owned jobs, with no owned A100-80GB/L40S overlap. Quota is home
  `88.6%`, scratch `39.3%`; held-out remains `0`, Sheet E/F/G remain blank,
  Kombuys remains read-only at its last verified idle state, and all gates
  remain unchanged.

- At 01:58 SAST on 18 August, both NER a2 confirmations produced clean first
  validation-only exact artifacts at step 541 and resumed training. Job
  `1245393` (seed 13) scored mean span F1 `0.2972696517612993`, with
  Tsn/Xho/Zul `0.27350049707274504/0.2900679456434352/0.32824051256771775`;
  its 192-row debug SHA-256 is
  `a045389ad4f1143d2ba672b7c51aa3bb82de17ddba77cef93d69c4afc2df53f0`.
  Job `1245394` (seed 87) scored `0.36267885666568933`, with
  `0.36175455889596053/0.32256819351513666/0.4037138175859708`; its debug
  SHA-256 is
  `bb53213ddb0fab8d10f0913af59a5d52d2719e78a0b07393c503c0bc5918a79f`.
  Each artifact has exactly `64` rows per language, no literal empty raw
  output, and no parser failure; recomputed language means exactly match the
  retained trainer metrics. The jobs were healthy near `877/8115` and
  `1082/8115`, with seed 87 entering its next validation boundary. These are
  valid interim artifacts, not terminal confirmations: NER remains `2/4`
  terminal-valid and no winner is frozen. They remain the only owned jobs,
  both A100-40GB with the GDN fast path and no owned A100-80GB/L40S overlap.
  Quota is home `88.6%`, scratch `39.3%`; held-out remains `0`, Sheet E/F/G
  remain blank, Kombuys remains read-only at its last verified idle state,
  and all scientific gates remain unchanged.

- At 01:22 SAST on 18 August, NER a2 confirmation jobs `1245393` (seed 13)
  and `1245394` (seed 87) had both crossed the first step-541 boundary and
  completed declared validation over all `10,760` rows. Their validation
  losses were `0.491191163470754` and `0.5266696859026487`; these losses are
  health/provenance only and are not used for recipe selection. Both jobs are
  now inside the separate exact 192-prompt generation callback, with fresh
  automatic batch-size probes, the verified GatedDeltaNet fast path, and no
  targeted fault marker. No complete step-541 exact artifact exists yet;
  prior callback timing puts the revised window around `02:05--02:25 SAST`.
  NER therefore remains `2/4` terminal-valid confirmations. These are the only
  owned jobs, both A100-40GB, with no owned A100-80GB/L40S overlap. Quota is
  home `88.6%`, scratch `39.2%`; held-out remains `0`, Sheet E/F/G remain
  blank, Kombuys remains read-only at its last verified idle state, and all
  scientific gates remain unchanged.

- At 00:52 SAST on 18 August, NER a2 confirmation jobs `1245393` (seed 13)
  and `1245394` (seed 87) were healthy and fault-free on separate
  `srvrocgpu010` A100-40GB devices at `511/8115` and `502/8115` training
  steps. Both retain the verified GatedDeltaNet fast path and are approaching
  the first step-541 validation boundary; first complete exact validation
  artifacts are estimated around `01:45--02:15 SAST`. This is operational
  progress only: NER still has `2/4` terminal-valid confirmations. These are
  the only owned jobs, with no A100-80GB/L40S overlap. Quota is home `88.6%`,
  scratch `39.2%`; held-out remains `0`, Sheet E/F/G remain blank, Kombuys
  remains read-only at its last verified idle state, and all scientific gates
  remain unchanged.

- At 00:26 SAST on 18 August, POS a1 seed-13 job `1241720` was
  terminal-valid: `COMPLETED 0:0` at `00:08:59` after `17:59:32`. Its final
  step-1981 exact artifact covers `1,800` rows, `12` cells, and `17` labels,
  scores `0.8427841065626649`, and independently verifies at SHA-256
  `160ced2753810e8980abda4d54c804c15396103bed7d02d1c4c61741eca518f3`.
  Checkpoint 1415 remains the seed-13 retained winner at
  `0.8452102586327794`; a1's two-seed mean is therefore
  `0.8463392380469843`. The preregistered, validation-only POS close-out
  selects a2 at two-seed mean `0.8606543221600175`; held-out remains
  untouched and the global eight-family freeze is still blocked by remaining
  grids. With no owned job left, the two already-frozen NER a2 confirmations
  were submitted and started together as jobs `1245393` (seed 13) and
  `1245394` (seed 87) on `srvrocgpu010`, one A100-40GB each. Both verified
  all `694` immutable source/config files and the GatedDeltaNet fast path;
  execution-manifest SHA-256 values are
  `9c18f338c4c41402217c8669c3e078c4be303d65f66d498b9efef07f952af602`
  and `c12d0921153bec2da784e05752d140ca261e94a1ffcc1e9cda76583842017fc8`.
  No A100-80GB/L40S work overlaps them. Quota is home `88.6%`, scratch
  `39.2%`; Sheet E/F/G remain blank, Kombuys remains read-only at its last
  verified idle state, and quarantine/publication gates remain unchanged.

- At 23:52 SAST on 17 August, POS a1 seed-13 job `1241720` remained healthy
  and fault-free on `srvrocgpu010` A100-40GB at `1,550/1,800` rows in its
  step-1981 exact callback. The remaining `250` rows project completion near
  `00:10--00:15 SAST` on 18 August. This remains operational only: a1 is not
  terminal-valid and POS is not frozen. It is the sole owned job; no
  A100-80GB/L40S work overlaps it. Quota is home `88.6%`, scratch `39.2%`;
  held-out `0`, Sheet E/F/G blank, last verified Kombuys state read-only/idle,
  quarantine, and publication gates remain unchanged.

- At 23:22 SAST on 17 August, POS a1 seed-13 job `1241720` remained healthy
  and fault-free on `srvrocgpu010` A100-40GB at `1,150/1,800` rows in its
  step-1981 exact callback. Observed throughput keeps the complete-artifact
  ETA near `00:05--00:15 SAST` on 18 August. This remains operational only:
  a1 is not terminal-valid and POS is not frozen. It is the sole owned job;
  no A100-80GB/L40S work overlaps it. Quota is home `88.6%`, scratch `39.2%`;
  held-out `0`, Sheet E/F/G blank, last verified Kombuys state read-only/idle,
  quarantine, and publication gates remain unchanged.

- At 22:52 SAST on 17 August, POS a1 seed-13 job `1241720` remained healthy
  and fault-free on `srvrocgpu010` A100-40GB at `750/1,800` rows in its
  step-1981 exact callback. Observed throughput keeps the complete-artifact
  ETA near `00:05--00:15 SAST` on 18 August. This remains operational only:
  a1 is not terminal-valid and POS is not frozen. It is the sole owned job;
  no A100-80GB/L40S work overlaps it. Quota is home `88.6%`, scratch `39.2%`;
  held-out `0`, Sheet E/F/G blank, last verified Kombuys state read-only/idle,
  quarantine, and publication gates remain unchanged.

- At 22:22 SAST on 17 August, POS a1 seed-13 job `1241720` remained healthy
  and fault-free on `srvrocgpu010` A100-40GB at `350/1,800` rows in its
  step-1981 exact callback. Observed throughput projects the complete artifact
  near `00:05--00:15 SAST` on 18 August. This is operational progress only:
  a1 is not terminal-valid and POS is not frozen. It is the sole owned job;
  no A100-80GB/L40S work overlaps it. Quota is home `88.6%`, scratch `39.2%`;
  held-out `0`, Sheet E/F/G blank, last verified Kombuys state read-only/idle,
  quarantine, and publication gates remain unchanged.

- At 21:52 SAST on 17 August, POS a1 seed-13 job `1241720` independently
  hash-verified its complete step-1698 exact artifact over all `1,800` rows,
  `12` cells, and `17` labels. Token accuracy was `0.843005859282623`;
  artifact SHA-256 is
  `ea0ac77781f762cb871143a2bfd951bc1c8b87399de35b2fbf5efe62a332c5f0`.
  The retained best remains checkpoint 1415 at `0.8452102586327794`, so the
  provisional a1 two-seed mean remains `0.8463392380469843`. The healthy,
  fault-free job has entered its step-1981 declared validation; its exact
  callback is expected next, with a complete artifact projected near
  `00:05--00:20 SAST` on 18 August. This remains operational only: a1 is not
  terminal-valid and POS is not frozen. It is the sole owned A100-40GB job,
  with no A100-80GB/L40S overlap. Quota is home `88.6%`, scratch `39.2%`;
  held-out `0`, Sheet E/F/G blank, Kombuys read-only, quarantine, and
  publication gates remain unchanged.

- At 21:22 SAST on 17 August, POS a1 seed-13 job `1241720` remained healthy
  on `srvrocgpu010` A100-40GB at `1,600/1,800` rows in its step-1698 exact
  callback, with no targeted fault marker. Current throughput puts the complete
  artifact near `21:37--21:42 SAST`. This remains operational only: a1 is not
  terminal-valid and POS is not frozen. It is the sole owned job; no
  A100-80GB/L40S work overlaps it. `srvrocgpu009` is idle but is the excluded
  `gpu:amperemk` GRES, not an eligible acceleration path. Quota is home
  `88.6%`, scratch `39.2%`; held-out `0`, Sheet E/F/G blank, Kombuys
  read-only, quarantine, and publication gates remain unchanged.

- At 20:52 SAST on 17 August, POS a1 seed-13 job `1241720` remained healthy
  and fault-free on `srvrocgpu010` A100-40GB. Its step-1698 exact validation
  reached `1,200/1,800` rows; current throughput projects the complete artifact
  near `21:35--21:40 SAST`. This remains operational progress only: a1 is not
  terminal-valid and POS is not frozen. The sole owned job is A100-40GB, with
  no A100-80GB/L40S overlap. Quota is home `88.6%`, scratch `39.2%`;
  held-out remains `0`, Sheet E/F/G remain blank, Kombuys remains read-only,
  and quarantine/publication gates are unchanged.

- At 19:22 SAST on 17 August, a1 seed-13 POS job `1241720` independently
  hash-verified its step-1415 exact artifact over all `1,800` rows, `12`
  cells, and `17` labels. Token accuracy improved to
  `0.8452102586327794`; artifact SHA-256 is
  `4406ed9b0d44c8c386904744d912a5ac6e7e61b64c94503daeb49320ad70eb1f`.
  Its provisional two-seed mean is now `0.8463392380469843`, below
  terminal-valid a2's `0.8606543221600175`, but a1 remains incomplete and
  cannot be pruned or selected from this interim result. It reached the
  step-1698 boundary healthy. One A100-40GB job is owned; no A100-80GB/L40S
  work is active. Quota is home `88.6%`, scratch `39.2%`; held-out `0`,
  Sheet E/F/G blank, Kombuys read-only, quarantine, and publication gates
  remain unchanged.

- At 16:46 SAST on 17 August, a1 seed-13 POS confirmation job `1241720`
  independently hash-verified its step-1132 exact artifact over all `1,800`
  rows, `12` cells, and `17` labels. Token accuracy improved to
  `0.8363515921069141`; artifact SHA-256 is
  `e76d7a1ba485336c597e2243db0e01c5fd9cb6169403d3ba81db169b3c85c2c1`.
  Its provisional two-seed mean with the fixed seed-42 score is
  `0.8419099047840517`, still below terminal-valid a2's
  `0.8606543221600175`, but the preregistered a1 run is incomplete and must
  not be selected or pruned from this interim comparison. It resumed healthy
  training toward step 1415. Owned GPU state is one running A100-40GB job
  and no A100-80GB/L40S work; quota is home `88.6%`, scratch `39.2%`.
  Held-out `0`, blank Sheet E/F/G, read-only Kombuys, quarantine, and
  publication gates are unchanged.

- At 14:16 SAST on 17 August, a2 seed-13 POS confirmation job `1241719`
  completed `0:0` after `15:17:59`. Its terminal step-1698 exact artifact
  covers all `1,800` rows, `12` cells, and `17` labels, scores
  `0.8538223932360355`, and independently verifies at SHA-256
  `be14c27a3ea8c882eb49ecdb2822e13c68ca538381acee2606b6456e4d780879`.
  The seed's retained validation winner is therefore step 1132 at
  `0.8618166664037479`; combined with preregistered seed-42 score
  `0.8594919779162872`, a2's final two-seed arithmetic mean is
  `0.8606543221600175`. A1 seed-13 job `1241720` also independently
  hash-verified its step-849 artifact at `0.8326089557196429` (SHA-256
  `ef6ec65f0406f8e35b31e9404f19abca287d421079850a72889bc4a4db5b401a`)
  and began step-1132 exact validation. A2 is terminal-valid, but POS remains
  unfrozen until a1 completes under the preregistered protocol. Owned GPU
  state is one running A100-40GB job and no A100-80GB/L40S work. Quota is
  home `88.6%`, scratch `39.2%`; held-out `0`, blank Sheet E/F/G, read-only
  Kombuys, quarantine, and publication gates are unchanged.

- At 11:46 SAST on 17 August, both reduced POS confirmations produced new
  independently hash-verified exact-validation artifacts. A2 seed-13 job
  `1241719` scored `0.8601064478765359` at step 1415 over all `1,800` rows,
  `12` cells, and `17` labels (SHA-256
  `22315c34915b1e38b665394f54e166bc08cad6a32714a435e8bba14ac1439bea`);
  this does not exceed its eligible step-1132 best
  `0.8618166664037479`. A1 seed-13 job `1241720` improved to
  `0.8065868856875102` at step 566 with the same complete coverage (SHA-256
  `2ef415d2dea2592b635428a176b9e09ccb387b8bfdbed94a8dba636b50e5a960`)
  and began its step-849 callback. Both jobs remain healthy and running on
  separate A100-40GB GPUs; these are eligible interim checkpoints, not
  terminal seed results or a frozen POS winner. Quota is home `88.6%`,
  scratch `39.1%`; no A100-80GB/L40S work is owned. Held-out remains `0`,
  Sheet E/F/G remain blank, Kombuys remains read-only, and quarantine and
  publication gates are unchanged.

- At 09:24 SAST on 17 August, a2 seed-13 job `1241719` completed and
  independently hash-verified its step-1132 exact POS artifact over all
  `1,800` rows, `12` cells, and `17` labels. Token accuracy improved to
  `0.8618166664037479`; artifact SHA-256 is
  `bf82daa0fb5f7261e130985ff6ee35cc5e40cb102830025c1ddd9ec5bfdd8b94`.
  This is an eligible within-run checkpoint, not the terminal seed result or
  POS winner; the job reached the next step-1415 evaluation boundary. A1
  seed-13 job `1241720` remained fault-free at `250/1,800` rows in its
  step-566 exact callback. Both jobs remain on separate A100-40GB GPUs; no
  A100-80GB/L40S work is owned. Quota is home `88.6%`, scratch `39.1%`;
  held-out `0`, blank Sheet E/F/G, read-only Kombuys, quarantine, and
  publication gates are unchanged.

- At 08:54 SAST on 17 August, a1 seed-13 job `1241720` completed and
  independently hash-verified its first exact POS artifact at step 283 over
  all `1,800` rows, `12` cells, and `17` labels. Token accuracy is
  `0.726226396953919`; artifact SHA-256 is
  `4057cf4e88a2f1a7d07abab8a69017b2e6e0e367b43e407285118798ab6bc75c`.
  This is an eligible within-run checkpoint, not the terminal seed result or
  POS winner; the job resumed healthy training at step `423`. A2 seed-13 job
  `1241719` remained fault-free at `1,600/1,800` rows in its step-1132 exact
  callback, projecting completion near `09:10`. Both jobs remain on separate
  A100-40GB GPUs; no A100-80GB/L40S work is owned. Quota is home `88.6%`,
  scratch `39.1%`; held-out `0`, blank Sheet E/F/G, read-only Kombuys,
  quarantine, and publication gates are unchanged.

- At 06:54 SAST on 17 August, a2 seed-13 job `1241719` completed and
  independently hash-verified its step-849 exact POS artifact over all
  `1,800` rows, `12` cells, and `17` labels. Token accuracy improved to
  `0.8535822546623176`; artifact SHA-256 is
  `f28d700a95045225f8513a13749a119875d658eae7b1c4b4e8a453971ffc3b23`.
  This is another eligible within-run checkpoint, not the terminal seed result
  or POS winner; its step-1132 callback had begun at `50/1,800` rows. A1
  seed-13 job `1241720` remained healthy in its first step-283 exact callback
  at `300/1,800` rows. Both jobs were fault-free on separate A100-40GB GPUs
  on `srvrocgpu010`; no A100-80GB/L40S work is owned. Quota is home `88.6%`,
  scratch `39.1%`; held-out `0`, blank Sheet E/F/G, read-only Kombuys,
  quarantine, and publication gates are unchanged.

- At 06:24 SAST on 17 August, both reduced POS confirmations were running
  concurrently on separate A100-40GB devices on `srvrocgpu010`. A1 seed-13
  job `1241720` started at `06:09:27`, verified all `695` immutable files
  (execution-manifest SHA-256
  `b723f33b64a6e9fb895a9b370dffed357fca9ea31356877ad8f1070c314e46f2`),
  loaded the canonical pure-GDN checkpoint, and reached training step `276`
  without a fault marker. A2 seed-13 job `1241719` remained healthy at
  `1,650/1,800` rows in its step-849 exact callback, still projecting a
  complete artifact around `06:35`. Owned state is two running A100-40GB
  jobs and no pending/A100-80GB/L40S work. Quota is home `88.6%`, scratch
  `39.1%`; held-out remains `0`, Sheet E/F/G remain blank, Kombuys remains
  read-only, and all quarantine/publication gates remain unchanged.

- At 05:54 SAST on 17 August, POS a2 seed-13 job `1241719` remained
  fault-free on A100-40GB at `1,250/1,800` rows in its step-849 exact
  callback, projecting completion around `06:35`. Slurm advanced the
  Resources-pending a1 seed-13 job `1241720` estimate from `11:58:10` to
  `06:38:15`, so it may start shortly after a2 releases the GPU. Owned state
  remains one running plus one pending A100-40GB job, with no A100-80GB/L40S.
  Quota is home `88.6%`, scratch `39.1%`; held-out remains `0`, Sheet E/F/G
  remain blank, Kombuys remains read-only, and all quarantine/publication
  gates remain unchanged.

- At 04:24 SAST on 17 August, POS a2 seed-13 job `1241719` completed and
  independently hash-verified its step-566 exact validation artifact over all
  `1,800` rows, `12` cells, and `17` labels. Token accuracy improved to
  `0.8276377533234429`; artifact SHA-256 is
  `db177f15b2486d540757883eb082936d8d1a46385971c35113bf253bed3e5fca`.
  This remains an eligible within-run checkpoint, not the terminal seed result
  or POS winner. The step-849 callback is active at `100/1,800` rows. Job
  `1241720` remains Resources-pending for `2026-08-17 11:58:10`. Owned state
  is one running plus one pending A100-40GB job, no A100-80GB/L40S; quota is
  home `88.6%`, scratch `39.1%`. Held-out `0`, blank Sheet E/F/G, read-only
  Kombuys, quarantine, and publication gates are unchanged.

- At 01:54 SAST on 17 August, POS reduced-confirmation a2 seed-13 job
  `1241719` remained healthy on `srvrocgpu010` A100-40GB. Its first exact
  full-validation callback completed at step 283 over `1,800` rows, `12`
  cells, and `17` labels with token accuracy `0.7823488792993896`; artifact
  SHA-256 `d605b4289e770194da5849e1ce85c03828372e42c67db0671ef60775199f8f7c`
  independently verifies. This is an eligible within-run checkpoint, not the
  terminal seed result and not a frozen winner. The step-566 callback had
  reached `150/1,800` rows. A1 seed-13 job `1241720` remains
  Resources-pending with Slurm estimate `2026-08-17 11:58:10`. Owned state is
  one running plus one pending A100-40GB job, no A100-80GB/L40S; quota is home
  `88.6%`, scratch `39.1%`. Held-out remains `0`; Sheet E/F/G are blank,
  Kombuys remains read-only, and all quarantine/publication gates remain.

- At 22:53 SAST on 16 August, the user approved a deadline-driven,
  validation-only POS close-out before any cancellation. The enhanced POS
  Stage-B branch is uniformly quarantined and may not affect selection:
  pending resumes `1241580/1241581` were cancelled at zero runtime, and b2
  `1241578` was cancelled at a hash-verified step-283 checkpoint boundary
  after `02:36:12`. POS selection is now restricted to completed Stage-A
  seed-42 candidates; fixed finalists are a2 (`0.8594919779162872`) and a1
  (`0.8474682174611892`). Exactly one additional validation-only seed, 13,
  will run for each finalist, and the winner will be selected by their
  two-seed arithmetic mean. Confirmation a2 seed-13 job `1241719` started on
  `srvrocgpu010` A100-40GB at `22:52:14`, verified all `695` immutable files
  and manifest SHA-256 `b00cc625...144e`; a1 seed-13 job `1241720` is
  Resources-pending with current estimate `2026-08-17 11:58:10`. This is a
  transparently budget-limited coarse POS search, not completion of the
  planned 11-candidate enhanced grid. Held-out remains `0`; Sheet E/F/G,
  quarantine, Kombuys read-only, and publication gates are unchanged. HEX
  quota is home `88.6%`, scratch `39.0%`; only A100-40GB `gpu:ampere` is in
  use.

- At 21:54 SAST POS b2 `1241578` remained healthy on `srvrocgpu010`
  A100-40GB; its frozen checkpoint-283 exact callback reached `1,000/1,800`
  rows at 21:50 with a fresh fault-free log, keeping the complete-artifact
  ETA around `22:45--23:00`. Slurm's dynamic estimate for Resources-pending
  b0 resume `1241580` regressed to `2026-08-17 11:58:10 SAST`; Priority-
  pending b1 `1241581` still has no estimate. Owned state remains one running
  plus two pending A100-40GB jobs, no A100-80GB/L40S; quota is home `88.6%`,
  scratch `39.0%`. Trusted counts remain base `16/16`, NER `11/11` plus
  `2/4`, T2X `11/11` plus `4/4`, POS `3/11`, winners `0/8`, held-out `0`,
  Mono not started. Kombuys is read-only and idle; Sheet E/F/G remain blank
  and quarantine/publication gates remain.

- At 21:24 SAST POS b2 `1241578` remained healthy on `srvrocgpu010`
  A100-40GB; its frozen checkpoint-283 exact callback reached `650/1,800`
  rows with a fresh fault-free log, preserving a first complete-artifact ETA
  around `22:45--23:00`. Exact b0/b1 checkpoint-849 resumes
  `1241580/1241581` remain Resources/Priority-pending; Slurm projects b0 at
  `2026-08-16 22:43:34 SAST` and gives b1 no estimate. Owned state is one
  running plus two pending A100-40GB jobs, no A100-80GB/L40S; quota is home
  `88.6%`, scratch `39.0%`. Trusted progress remains base `16/16`, NER
  `11/11` plus `2/4` confirmations, T2X `11/11` plus `4/4`, POS `3/11`,
  winners `0/8`, held-out `0`, Mono not started. Kombuys remains read-only
  and idle; Sheet E/F/G are blank and quarantine/publication gates remain.

- At 19:46 SAST POS recovery jobs `1238989/1238990` remained healthy on
  `srvrocgpu010` A100-40GB; both checkpoint-1132 exact callbacks had reached
  `1,050/1,800` rows with fresh fault-free logs, keeping complete artifacts
  at about 20:35--20:50. Partial rows are operational only. Row-batch v3
  gate/b2 `1239042` remains Resources-pending, conservatively scheduled for
  `2026-08-17 10:26:08` but eligible when a recovery GPU releases. Owned
  state is two running plus one pending A100-40GB job, no A100-80GB/L40S;
  quota is home 70.3%, scratch 38.9%. Terminal trusted POS remains `3/11`;
  all other trusted counts, held-out `0`, Sheet E/F/G, publication gates,
  and latest verified read-only/idle Kombuys state are unchanged.

- At 19:16 SAST POS recovery jobs `1238989/1238990` remained healthy on
  `srvrocgpu010` A100-40GB; both checkpoint-1132 exact callbacks had reached
  `650/1,800` rows with fresh fault-free logs, keeping complete artifacts at
  about 20:35--20:50. Partial rows are operational only. Row-batch v3
  gate/b2 `1239042` remains Resources-pending, conservatively scheduled for
  `2026-08-17 10:26:08` but eligible when a recovery GPU releases. Owned
  state is two running plus one pending A100-40GB job, no A100-80GB/L40S;
  quota is home 70.3%, scratch 38.9%. Terminal trusted POS remains `3/11`;
  all other trusted counts, held-out `0`, Sheet E/F/G, publication gates,
  and latest verified read-only/idle Kombuys state are unchanged.

- At 18:46 SAST POS recovery jobs `1238989/1238990` remained healthy on
  `srvrocgpu010` A100-40GB in their checkpoint-1132 exact callbacks: b0 was
  at `300/1,800` rows and b1 at `250/1,800`, with fresh fault-free logs and
  complete artifacts projected around 20:35--20:50. Partial rows are
  operational only. Row-batch v3 gate/b2 `1239042` remains
  Resources-pending, conservatively scheduled for `2026-08-17 10:26:08`
  but eligible when a recovery GPU releases. Owned state is two running plus
  one pending A100-40GB job, no A100-80GB/L40S; quota is home 70.3%, scratch
  38.9%. Terminal trusted POS remains `3/11`; all other trusted counts,
  held-out `0`, Sheet E/F/G, publication gates, and latest verified
  read-only/idle Kombuys state are unchanged.

- At 18:17 SAST POS recovery jobs `1238989/1238990` had each completed and
  hash-verified its checkpoint-849 exact artifact over all `1,800/1,800`
  rows, 12 cells, and 17 labels. B0 accuracy is `0.830626634400054`
  (artifact SHA-256
  `9574e4c17f3804c7b43adc6dca127999e17f0e41bf85034fa5b90fe8cb1bca7a`);
  b1 is `0.8016046208083835` (SHA-256
  `6c217453c5e4318897eba4f8fceba8ea1087a512f89991fbb665a2d7b03f2d2c`).
  Both improved over checkpoint 566 and retained checkpoint 849 with
  patience `0/2`; cancelled b0 `1238876` remains provenance-only. Both
  resumed healthy A100-40GB training at steps 1022/975, with checkpoint-1132
  exact artifacts estimated around 20:30--20:50. Row-batch v3 gate/b2
  `1239042` remains Resources-pending, conservatively scheduled for
  `2026-08-17 10:26:08` but eligible when a recovery GPU releases. Owned
  state is two running plus one pending A100-40GB job, no A100-80GB/L40S;
  quota is home 70.3%, scratch 38.9%. These are within-candidate milestones,
  so terminal trusted POS remains `3/11`; all other trusted counts, held-out
  `0`, Sheet E/F/G, publication gates, and latest verified read-only/idle
  Kombuys state remain unchanged.

- At 17:46 SAST POS recovery jobs `1238989/1238990` remained healthy on
  `srvrocgpu010` A100-40GB in their checkpoint-849 exact callbacks: b0 was
  at `1,500/1,800` rows and b1 at `1,450/1,800`, with fresh fault-free logs
  and complete artifacts projected around 18:05--18:15. Partial rows are
  operational only. Row-batch v3 gate/b2 `1239042` remains
  Resources-pending, conservatively scheduled for `2026-08-17 10:26:08`
  but eligible when a recovery GPU releases. Owned state is two running plus
  one pending A100-40GB job, no A100-80GB/L40S; quota is home 70.3%, scratch
  38.9%. Terminal trusted POS remains `3/11`; all other trusted counts,
  held-out `0`, Sheet E/F/G, publication gates, and latest verified
  read-only/idle Kombuys state are unchanged.

- At 17:14 SAST POS recovery jobs `1238989/1238990` remained healthy on
  `srvrocgpu010` A100-40GB in their checkpoint-849 exact callbacks: b0 was
  at `1,100/1,800` rows and b1 at `1,050/1,800`, with fresh fault-free logs
  and complete artifacts projected around 18:05--18:15. Partial rows are
  operational only. Row-batch v3 gate/b2 `1239042` remains
  Resources-pending, conservatively scheduled for `2026-08-17 10:26:08`
  but eligible when a recovery GPU releases. Owned state is two running plus
  one pending A100-40GB job, no A100-80GB/L40S; quota is home 70.3%, scratch
  38.9%. Terminal trusted POS remains `3/11`; all other trusted counts,
  held-out `0`, Sheet E/F/G, publication gates, and latest verified
  read-only/idle Kombuys state are unchanged.

- At 15:42 SAST POS recovery jobs `1238989/1238990` had each completed and
  hash-verified its checkpoint-566 exact artifact over all `1,800/1,800`
  rows, 12 cells, and 17 labels. B0 accuracy is `0.7890084636039543`
  (artifact SHA-256
  `ea7de0a378816d5cf35435d1dca440c6044e2f30f3e54c547cf5105893c8daf4`);
  b1 is `0.7616054032322993` (SHA-256
  `9acee96952db70a1dbf2432449cd441c75d8da67a54d41f67370abf96ff52a48`).
  Both improved over their own recovery checkpoint-283 scores and retained
  checkpoint 566 with patience `0/2`; cancelled b0 `1238876` remains
  provenance-only. Both resumed healthy A100-40GB training at steps 738/686,
  with checkpoint-849 exact artifacts estimated around 17:50--18:10.
  Row-batch v3 gate/b2 `1239042` remains Resources-pending, conservatively
  scheduled for `2026-08-17 10:26:08` but eligible when a recovery GPU
  releases. Owned state is two running plus one pending A100-40GB job, no
  A100-80GB/L40S; quota is home 70.3%, scratch 38.9%. These are
  within-candidate milestones, so terminal trusted POS remains `3/11`; all
  other trusted counts, held-out `0`, Sheet E/F/G, publication gates, and
  latest verified read-only/idle Kombuys state remain unchanged.

- At 15:11 SAST POS recovery jobs `1238989/1238990` remained healthy on
  `srvrocgpu010` A100-40GB; both checkpoint-566 exact callbacks had reached
  `1,450/1,800` rows with fresh fault-free logs. The remaining 350 rows per
  job project complete artifacts around 15:35--15:45; partial rows are
  operational only. Row-batch v3 gate/b2 `1239042` remains
  Resources-pending, conservatively scheduled for `2026-08-17 10:26:08`
  but eligible when a recovery GPU releases. Owned state is two running plus
  one pending A100-40GB job, no A100-80GB/L40S; quota is home 70.3%, scratch
  38.9%. Terminal trusted POS remains `3/11`; all other trusted counts,
  held-out `0`, Sheet E/F/G, publication gates, and latest verified
  read-only/idle Kombuys state are unchanged.

- At 14:42 SAST POS recovery jobs `1238989/1238990` remained healthy on
  `srvrocgpu010` A100-40GB in their checkpoint-566 exact callbacks: b0 was
  at `1,100/1,800` rows and b1 at `1,050/1,800`, with fresh fault-free logs
  and complete artifacts still projected around 15:30--15:40. Partial rows
  are operational only. Row-batch v3 gate/b2 `1239042` remains
  Resources-pending, conservatively scheduled for `2026-08-17 10:26:08`
  but eligible when a recovery GPU releases. Owned state is two running plus
  one pending A100-40GB job, no A100-80GB/L40S; quota is home 70.3%, scratch
  38.9%. Terminal trusted POS remains `3/11`; all other trusted counts,
  held-out `0`, Sheet E/F/G, publication gates, and latest verified
  read-only/idle Kombuys state are unchanged.

- At 13:59 SAST POS recovery jobs `1238989/1238990` remained healthy on
  `srvrocgpu010` A100-40GB; both checkpoint-566 exact callbacks had reached
  `500/1,800` rows with fresh fault-free logs, projecting complete artifacts
  around 15:30--15:40. Partial rows are operational only. Row-batch v3
  gate/b2 `1239042` remains Resources-pending, conservatively scheduled for
  `2026-08-17 10:26:08` but eligible when a recovery GPU releases. Owned
  state is two running plus one pending A100-40GB job, no A100-80GB/L40S;
  quota is home 70.3%, scratch 38.9%. Terminal trusted POS remains `3/11`;
  all other trusted counts, held-out `0`, Sheet E/F/G, publication gates,
  and latest verified read-only/idle Kombuys state are unchanged.

- At 13:32 SAST POS recovery b0/b1 jobs `1238989/1238990` had each completed
  and hash-verified its checkpoint-283 exact artifact over all `1,800/1,800`
  rows, 12 cells, and 17 labels. B0 accuracy is `0.7144075908238836`
  (artifact SHA-256
  `477026720accb5060d8fdbc68247c0c6fa93b5d4cb6eb0207587dd02b8b9984f`);
  b1 is `0.621204427395874` (SHA-256
  `b7a8ddb64bd89e7beaa222ae4c37301f76ec6e0ef01c67251dc1bf754596e4d0`).
  Each retained its own recovery checkpoint 283; cancelled b0 `1238876`
  remains provenance-only. Both jobs are healthy on `srvrocgpu010`
  A100-40GB and their second exact callbacks are active at checkpoint 566
  (b0 `150/1,800`, b1 `100/1,800`), with complete artifacts estimated around
  15:25--15:40 SAST. Row-batch v3 gate/b2 `1239042` remains
  Resources-pending, conservatively scheduled for `2026-08-17 10:26:16` but
  eligible when a recovery GPU releases. Owned state is two running plus one
  pending A100-40GB job, no A100-80GB/L40S; quota is home 70.3%, scratch
  38.9%. These are within-candidate validation milestones, so terminal
  trusted POS remains `3/11`; all other trusted counts, held-out `0`, Sheet
  E/F/G, publication gates, and latest verified read-only/idle Kombuys state
  remain unchanged.

- At 13:00 SAST full-prefix POS recovery jobs `1238989/1238990` remained
  healthy on `srvrocgpu010` A100-40GB, each at `1,750/1,800` rows in its
  first exact callback with fresh fault-free logs. Only 50 rows remain per
  job, so complete artifacts are expected around 13:03--13:08; partial
  results remain operational only. Fused row-batch gate/b2 `1239042` remains
  Resources-pending on `gpu:ampere`, with a conservative displayed start of
  `2026-08-17 10:26:16` but possible earlier start after a recovery GPU
  releases. Owned state is two running plus one pending A100-40GB job, no
  A100-80GB/L40S execution. Quota is home 70.3% and scratch 38.8%; Kombuys
  remains read-only with no execution in this pass. Trusted counts, held-out
  state, Sheet E/F/G, and publication gates are unchanged.

- At 12:29 SAST full-prefix POS recovery jobs `1238989/1238990` remained
  healthy on `srvrocgpu010` A100-40GB, each at `1,350/1,800` rows in its
  first exact callback with fresh fault-free logs. The remaining 450 rows
  keep complete-artifact ETA around 13:00--13:10 SAST; partial results are
  operational only. Fused row-batch gate/b2 `1239042` remains
  Resources-pending on `gpu:ampere`, conservatively projected for
  `2026-08-17 10:26:16` but eligible to start when a recovery GPU releases.
  Owned state is two running plus one pending A100-40GB job, no
  A100-80GB/L40S execution. Quota is home 70.3% and scratch 38.8%; Kombuys
  remains read-only with no execution in this pass. Trusted counts, held-out
  state, Sheet E/F/G, and publication gates are unchanged.

- At 12:00 SAST full-prefix POS recovery jobs `1238989/1238990` remained
  healthy on `srvrocgpu010` A100-40GB, with each first exact callback at
  `950/1,800` rows and fresh fault-free logs. Sustained throughput still
  projects complete artifacts around 13:00--13:10 SAST; partial results are
  operational only. Fused row-batch gate/b2 `1239042` remains
  Resources-pending on `gpu:ampere`, conservatively projected for
  `2026-08-17 10:26:16` but eligible to start when a recovery GPU releases.
  Owned state is two running plus one pending A100-40GB job; no
  A100-80GB/L40S execution exists. Quota remains home 70.3% and scratch
  38.8%. Kombuys remains read-only with no execution in this pass. Trusted
  counts, held-out status, Sheet E/F/G, and publication gates are unchanged.

- At 11:30 SAST full-prefix POS recovery jobs `1238989/1238990` remained
  healthy on `srvrocgpu010` A100-40GB, with each first exact callback at
  `550/1,800` rows and fresh fault-free logs. Sustained throughput projects
  complete artifacts around 13:00--13:10 SAST; partial results remain
  operational only. Fused row-batch gate/b2 `1239042` remains
  Resources-pending on `gpu:ampere`, conservatively projected for
  `2026-08-17 10:26:16`. Owned state is two running plus one pending
  A100-40GB job; no A100-80GB/L40S execution exists. Quota remains home
  70.3% and scratch 38.8%. Kombuys is read-only and idle at RTX 5090
  `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch 60%, with only the Tailscale
  tmux. Trusted counts, held-out status, Sheet E/F/G, and publication gates
  are unchanged.

- At 10:48 SAST cross-row full-prefix POS batching v3 was prospectively
  preregistered, implemented, locally verified, immutably deployed, and fused
  into pending b2. Fixed batch size 8 preserves `use_cache=False`, prompts,
  labels, row order, coverage, cell/aggregate metrics, and all selection rules;
  it must match the original serial scorer and achieve at least 3x on frozen
  b0 checkpoint 566 before any b2 training. Sixteen targeted tests and all
  lint/format/shell/whitespace checks passed. Read-only snapshot
  `/home/lmbanr001/masters/sallm_snapshots/pure-gdn-hpo-pos-rowbatch-v3p-20260816-ad16dfe5`
  verifies 698 source/config hashes; deployment manifest SHA-256 is
  `4ab77afa26415706beff8524bb6a5970512db9b76b91d0734b04b04ab324a930`
  on an independent inode. Pending-only full-prefix b2 `1238991` was cancelled
  at exactly `00:00:00`, with no checkpoint/logging output directories, and
  replaced by fused gate/b2 `1239042`. Intermediate `1239041` was also
  cancelled pending-only at `00:00:00` after a pre-start audit found that its
  execution manifest would record batch/output overrides only in launcher
  source rather than the captured environment; the corrected wrapper exports
  them before manifest creation. Job `1239042` is pending on the original
  A100-40GB `gpu:ampere`; its start estimate is not yet reliable.
  Full-prefix recovery jobs `1238989/1238990` remain untouched and healthy on
  the same node, both reaching step 283 and completing health-only validation.
  Quota is home 70.3%, scratch 38.8%; `gpu:amperemk` remains idle but forbidden
  by association, and no L40S/A100-80GB, Kombuys, held-out, Sheet, or Hugging
  Face action occurred. The v3 speedup is queued but not yet measured or
  ratified.

- At 10:09 SAST the POS equivalence gate was fused into b1's own allocation
  to avoid a separate A100 queue turn. Immutable snapshot
  `/home/lmbanr001/masters/sallm_snapshots/pure-gdn-hpo-poscache-fused-20260816-5ed8c304`
  passed 18 targeted local tests and all static checks. Standalone pending
  gate `1238975` was cancelled and replaced by b1 `1238979`, which will test
  frozen b0 checkpoint 566 on 12 validation-only cells before any b1 training;
  it aborts unless all preregistered equivalence and >=3x callback-speed gates
  pass, then enables incremental cache scoring in the same allocation. Job
  `1238979` is Resources-pending with no reliable start time. B0 `1238876`
  remains untouched and healthy, reaching `1,100/1,800` legacy rows at 10:03.
  The optimization is not active yet and applies only to POS exact callbacks,
  not training or other families. `gpu:amperemk` remains unavailable under
  `nlpgroup`; no held-out, Sheet, Kombuys, A100-80GB, or L40S action occurred.

- At 10:00 SAST the POS runtime correction is implemented and locally
  verified but not yet scientifically ratified. A dated prospective protocol
  fixes exact prediction/cell/accuracy, score-drift, and speed gates; 17
  targeted tests plus lint/syntax checks pass. Immutable gate snapshot
  `/home/lmbanr001/masters/sallm_snapshots/pure-gdn-hpo-poscache-20260816-34a3f7d5`
  is read-only and byte-verified. Pending-only Stage-B b1/b2 jobs
  `1238877/1238878` were cancelled before start/data access and preserved as
  provenance; replacement gate `1238975` is Resources-pending on
  `gpu:ampere`, conservatively projected for `2026-08-17 03:20:32`, while
  b0 `1238876` continues unchanged. Idle A100-40GB node `srvrocgpu009` is not
  currently usable: `nlpgroup` has live `GrpTRES gpu:amperemk=0`, and Slurm
  rejects that GRES with `AssocGrpGRES`. Thus cross-GRES expansion and b1/b2
  resubmission remain blocked until both implementation equivalence and an
  allowed `amperemk` association exist. Home quota recovered from a transient
  88.6% duplicate-snapshot charge to 83.4% after a hard-linked rebuild;
  scratch is 38.8%. No held-out, Sheet E/F/G, Kombuys, A100-80GB, or L40S
  action occurred; trusted counts remain base `16/16`, NER `11/11` plus
  `2/4` confirmations, T2X `11/11` plus `4/4`, POS `3/11`, winners `0/8`,
  held-out `0`, Mono not started.

- At 09:17 SAST POS Stage-B b0 `1238876` remained healthy on
  `srvrocgpu010` A100-40GB, reaching `450/1,800` rows in checkpoint 849's
  third frozen exact callback at `09:15:52`. Its log is fresh with no fault
  marker, and sustained throughput narrows the complete-artifact ETA to
  `10:55--11:05`. Partial output is operational only and cannot alter
  retention or ranking; checkpoint 566 remains retained at exact accuracy
  `0.7845538768361983`, patience `0/2`. Jobs `1238877/1238878` remain
  Resources/Priority-pending; b1 projects `2026-08-17 03:20:32`, b2 has no
  projection. Owned state is one running plus two pending A100-40GB jobs at
  cap, no A100-80GB/L40S; quota is `70.3%/38.8%`. Kombuys remains read-only
  and idle at RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `60%`.
  Terminal trusted counts remain base `16/16`, NER `11/11` plus `2/4`
  confirmations, T2X `11/11` plus `4/4` confirmations, POS `3/11`, winners
  `0/8`, held-out `0`, Mono not started. Sheet E/F/G stay blank; quarantine,
  publication, remaining-validation, and winner-freeze gates remain.

- At 08:48 SAST POS Stage-B b0 `1238876` had completed checkpoint 566's
  frozen exact artifact at accuracy `0.7845538768361983` over all `1,800`
  rows, 12 cells, and 17 labels, improving on checkpoint 283's
  `0.7099399341886862`. Checkpoint 566 is retained at patience `0/2`;
  artifact/state/adapter hashes are `027c3365...2ed`/`29754726...b04`/
  `5449ef4b...f6c`. This is valid within-candidate validation evidence, not a
  terminal result or cross-candidate ranking. B0 remains healthy on
  `srvrocgpu010` A100-40GB: checkpoint `849/4245` covered all `1,800` rows at
  health-only loss `0.14097159915500218`, and its third frozen callback
  reached `50/1,800` rows at `08:46:18`. The step-849 exact artifact is
  estimated around `10:55--11:10`. Jobs `1238877/1238878` remain
  Resources/Priority-pending; b1 projects `2026-08-17 03:20:32`, b2 has no
  projection. Owned state is one running plus two pending A100-40GB jobs at
  cap, no A100-80GB/L40S; quota is `70.3%/38.8%`. Kombuys remains read-only
  and idle at RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `60%`.
  Terminal trusted counts remain base `16/16`, NER `11/11` plus `2/4`
  confirmations, T2X `11/11` plus `4/4` confirmations, POS `3/11`, winners
  `0/8`, held-out `0`, Mono not started. Sheet E/F/G stay blank; quarantine,
  publication, remaining-validation, and winner-freeze gates remain.

- At 08:18 SAST POS Stage-B b0 `1238876` remained healthy on
  `srvrocgpu010` A100-40GB, reaching `1,650/1,800` rows in checkpoint 566's
  second frozen exact callback at `08:15:40`. Its log is fresh with no fault
  marker; only 150 rows remain, narrowing the complete-artifact ETA to
  `08:25--08:30`. Partial output is operational only and cannot alter
  retention or ranking; checkpoint 283 remains provisionally retained at
  exact accuracy `0.7099399341886862`, patience `0/2`. Jobs
  `1238877/1238878` remain Resources/Priority-pending; b1 projects
  `2026-08-17 03:20:32`, b2 has no projection. Owned state is one running
  plus two pending A100-40GB jobs at cap, no A100-80GB/L40S; quota is
  `70.3%/38.8%`. Kombuys remains read-only and idle at RTX 5090
  `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `60%`. Terminal trusted
  counts remain base `16/16`, NER `11/11` plus `2/4` confirmations, T2X
  `11/11` plus `4/4` confirmations, POS `3/11`, winners `0/8`, held-out `0`,
  Mono not started. Sheet E/F/G stay blank; quarantine, publication,
  remaining-validation, and winner-freeze gates remain.

- At 07:48 SAST POS Stage-B b0 `1238876` remained healthy on
  `srvrocgpu010` A100-40GB, reaching `1,250/1,800` rows in checkpoint 566's
  second frozen exact callback at `07:45:20`. Its log is fresh with no fault
  marker, and sustained throughput narrows the complete-artifact ETA to
  `08:25--08:35`. Partial output is operational only and cannot alter
  retention or ranking; checkpoint 283 remains provisionally retained at
  exact accuracy `0.7099399341886862`, patience `0/2`. Jobs
  `1238877/1238878` remain Resources/Priority-pending; b1 projects
  `2026-08-17 03:20:32`, b2 has no projection. Owned state is one running
  plus two pending A100-40GB jobs at cap, no A100-80GB/L40S; quota is
  `70.3%/38.8%`. Kombuys remains read-only and idle at RTX 5090
  `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `60%`. Terminal trusted
  counts remain base `16/16`, NER `11/11` plus `2/4` confirmations, T2X
  `11/11` plus `4/4` confirmations, POS `3/11`, winners `0/8`, held-out `0`,
  Mono not started. Sheet E/F/G stay blank; quarantine, publication,
  remaining-validation, and winner-freeze gates remain.

- At 07:18 SAST POS Stage-B b0 `1238876` remained healthy on
  `srvrocgpu010` A100-40GB, reaching `850/1,800` rows in checkpoint 566's
  second frozen exact callback at `07:15:05`. Its log is fresh with no fault
  marker, and sustained throughput keeps the complete-artifact ETA at
  `08:25--08:40`. Partial output is operational only and cannot alter
  retention or ranking; checkpoint 283 remains provisionally retained at
  exact accuracy `0.7099399341886862`, patience `0/2`. Jobs
  `1238877/1238878` remain Resources/Priority-pending; b1 projects
  `2026-08-17 03:20:32`, b2 has no projection. Owned state is one running
  plus two pending A100-40GB jobs at cap, no A100-80GB/L40S; quota is
  `70.3%/38.8%`. Kombuys remains read-only and idle at RTX 5090
  `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `60%`. Terminal trusted
  counts remain base `16/16`, NER `11/11` plus `2/4` confirmations, T2X
  `11/11` plus `4/4` confirmations, POS `3/11`, winners `0/8`, held-out `0`,
  Mono not started. Sheet E/F/G stay blank; quarantine, publication,
  remaining-validation, and winner-freeze gates remain.

- At 06:48 SAST POS Stage-B b0 `1238876` remained healthy on
  `srvrocgpu010` A100-40GB, reaching `450/1,800` rows in checkpoint 566's
  second frozen exact callback at `06:46:04`. Its log is fresh with no fault
  marker, and observed throughput supports an `08:25--08:40` complete
  artifact. Partial output is operational only and cannot alter retention or
  ranking; checkpoint 283 remains provisionally retained at exact accuracy
  `0.7099399341886862`, patience `0/2`. Jobs `1238877/1238878` remain
  Resources/Priority-pending; b1 projects `2026-08-17 03:20:32`, b2 has no
  projection. Owned state is one running plus two pending A100-40GB jobs at
  cap, no A100-80GB/L40S; quota is `70.3%/38.8%`. Kombuys remains read-only
  and idle at RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `60%`.
  Terminal trusted counts remain base `16/16`, NER `11/11` plus `2/4`
  confirmations, T2X `11/11` plus `4/4` confirmations, POS `3/11`, winners
  `0/8`, held-out `0`, Mono not started. Sheet E/F/G stay blank; quarantine,
  publication, remaining-validation, and winner-freeze gates remain.

- At 06:18 SAST POS Stage-B b0 `1238876` remained healthy on
  `srvrocgpu010` A100-40GB. It reached checkpoint `566/4245`, completed all
  `1,800/1,800` declared rows at health-only loss
  `0.18320615980360244`, and reached `50/1,800` rows in its second frozen
  exact callback. No step-566 artifact or selection decision exists yet;
  completion remains estimated around `08:25--08:40`. Checkpoint 283 remains
  provisionally retained at exact accuracy `0.7099399341886862`, patience
  `0/2`. Jobs `1238877/1238878` remain Resources/Priority-pending; b1
  projects `2026-08-17 03:20:32`, b2 has no projection. Owned state is one
  running plus two pending A100-40GB jobs at cap, no A100-80GB/L40S; quota
  is `70.3%/38.8%`. Kombuys remains read-only and idle at RTX 5090
  `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `60%`. Terminal trusted
  counts remain base `16/16`, NER `11/11` plus `2/4` confirmations, T2X
  `11/11` plus `4/4` confirmations, POS `3/11`, winners `0/8`, held-out `0`,
  Mono not started. Sheet E/F/G stay blank; quarantine, publication,
  remaining-validation, and winner-freeze gates remain.

- At 05:59 SAST POS Stage-B b0 `1238876` produced its first frozen exact
  artifact at checkpoint 283: accuracy `0.7099399341886862` over all `1,800`
  rows, 12 cells, and 17 labels. Artifact/state/adapter hashes are
  `394219b3...40b`/`8dbdb7f7...b27`/`fa518b2e...548`; checkpoint 283 is
  retained at patience `0/2`. This is valid within-candidate validation
  evidence, not a terminal b0 result or cross-candidate ranking. B0 resumed
  healthily on `srvrocgpu010` A100-40GB; step 566's exact artifact is
  estimated around `08:25--08:40`. B1/b2 `1238877/1238878` remain
  Resources/Priority-pending; b1 projects `2026-08-17 03:20:32`, b2 has no
  projection. Owned state is one running plus two pending A100-40GB jobs at
  cap, no A100-80GB/L40S; quota is `70.3%/38.8%`. Kombuys remains read-only
  and idle at RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `60%`.
  Terminal trusted counts remain base `16/16`, NER `11/11` plus `2/4`
  confirmations, T2X `11/11` plus `4/4` confirmations, POS `3/11`, winners
  `0/8`, held-out `0`, Mono not started. Sheet E/F/G stay blank; quarantine,
  publication, remaining validation, and winner-freeze gates remain.

- At 05:18 SAST POS Stage-B b0 `1238876` remained healthy on A100-40GB at
  `1,250/1,800` frozen constrained rows, with a fresh log, no fault marker,
  and no premature selection artifact. Sustained throughput keeps the first
  complete exact accuracy near `05:55--06:05`; no checkpoint or candidate
  comparison is permitted before closeout. B1/b2 `1238877/1238878` remain
  Resources/Priority-pending without data access; b1 projects
  `2026-08-17 03:20:32`, b2 has no projection. Owned state is one running
  plus two pending A100-40GB jobs at cap, no A100-80GB/L40S; quota is
  `70.3%/38.8%`. Kombuys remains read-only and idle at RTX 5090
  `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `60%`. Trusted counts remain
  base `16/16`, NER `11/11` plus `2/4` confirmations, T2X `11/11` plus `4/4`
  confirmations, POS `3/11`, winners `0/8`, held-out `0`, Mono not started.
  Sheet E/F/G stay blank; quarantine, publication, and held-out gates remain.

- At 04:48 SAST POS Stage-B b0 `1238876` remained healthy on A100-40GB at
  `850/1,800` frozen constrained rows, with a fresh log, no fault marker, and
  no complete selection artifact. Sustained throughput projects the first
  exact accuracy around `05:55--06:05`; partial progress cannot alter
  retention or ranking. B1/b2 `1238877/1238878` remain
  Resources/Priority-pending without data access; b1 projects
  `2026-08-17 03:20:32`, b2 has no projection. Owned state is one running
  plus two pending A100-40GB jobs at cap, no A100-80GB/L40S; quota is
  `70.3%/38.8%`. Kombuys remains read-only and idle at RTX 5090
  `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `60%`. Trusted counts remain
  base `16/16`, NER `11/11` plus `2/4` confirmations, T2X `11/11` plus `4/4`
  confirmations, POS `3/11`, winners `0/8`, held-out `0`, Mono not started.
  Sheet E/F/G stay blank; quarantine, publication, and held-out gates remain.

- At 04:18 SAST POS Stage-B b0 `1238876` remained healthy on A100-40GB and
  reached `450/1,800` frozen constrained rows at `04:14:54`, with a fresh log,
  no fault marker, and no complete selection artifact. Epoch-1 health-only
  loss remains `0.34238827175564235`; observed throughput keeps the first
  exact-accuracy ETA near `05:55--06:10`. B1/b2 `1238877/1238878` remain
  Resources/Priority-pending without data access; b1 projects
  `2026-08-17 03:20:32`, while b2 has no projection. Owned state is one
  running plus two pending A100-40GB jobs at cap, no A100-80GB/L40S; quota is
  `70.3%/38.8%`. Kombuys remains read-only and idle at RTX 5090
  `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `60%`, only
  `tailscale-kombuys` tmux. Trusted counts remain base `16/16`, NER `11/11`
  plus `2/4` confirmations, T2X `11/11` plus `4/4` confirmations, POS `3/11`,
  winners `0/8`, held-out `0`, Mono not started. Sheet E/F/G stay blank;
  quarantine and publication blocks remain.

- At 03:48 SAST POS Stage-B b0 `1238876` was healthy on `srvrocgpu010`
  A100-40GB: it completed epoch-1 step `283/4245`, evaluated exact
  `1,800/1,800` rows at health-only loss `0.34238827175564235`, and reached
  `100/1,800` rows in the frozen constrained callback. No exact accuracy,
  selection artifact, checkpoint comparison, or ranking exists yet; current
  throughput places the first artifact around `05:50--06:10`. B1/b2 jobs
  `1238877/1238878` remain Resources/Priority-pending without data access;
  b1's dynamic projection remains `2026-08-17 03:20:32`, while b2 has none.
  Owned state is one running plus two pending A100-40GB jobs at cap, with no
  A100-80GB/L40S. Quota is `70.3%/38.8%`. Kombuys remains read-only and idle
  at GPU0 RTX 5090 `10 MiB/0%`, GPU1 RTX 3080 Ti `1 MiB/0%`, scratch `60%`,
  only `tailscale-kombuys` tmux. Trusted counts remain base `16/16`, NER
  seed-42 `11/11`, NER confirmations `2/4`, T2X seed-42 `11/11`, T2X
  confirmations `4/4`, POS seed-42 `3/11`, winners `0/8`, held-out `0`, Mono
  not started. Sheet E/F/G stay blank; quarantine and publication blocks
  remain, and no fixed full-program ETA is supportable.

- At 03:21 SAST on 2026-08-16 POS a2 `1238228` was terminal-valid: completed
  `0:0` at `03:04:33` after `22:53:12`, with step-2547 exact accuracy
  `0.8509148607619882` over all `1,800` rows, 12 cells, and 17 labels. This
  second frozen miss retains checkpoint 1981 at `0.8594919779162872`.
  Final-artifact/state/retained/final/config hashes are
  `913219f0...c5c`/`829494f8...82d`/`9683f802...1c0`/`2a750177...099`/
  `b7033604...ade`; exact read-only comparison proved all `424/424` tensors
  (`71,762,560` values) match the final adapter. POS Stage A is therefore
  terminal-valid `3/3`, without freezing a winner. Fixed Stage-B b0/b1/b2
  are jobs `1238876/1238877/1238878`. B0 started on `srvrocgpu010` at
  `03:20:32`, verified all `694` immutable files and the pure-GDN fast path,
  loaded exact `2,259/1,800` train/validation rows, and has manifest/trial
  hashes `06b825ae...2b1`/`e3cb9752...abf`; b1/b2 are
  Resources/Priority-pending without model/data access. Slurm projects b1
  for `2026-08-17 03:20:32` and gives b2 no current projection. Owned state
  is one running plus two pending A100-40GB jobs, exactly at cap, with no
  A100-80GB/L40S.
  Quota is `70.3%/38.8%`. Kombuys remains read-only and idle at GPU0 RTX 5090
  `10 MiB/0%`, GPU1 RTX 3080 Ti `1 MiB/0%`, scratch `60%`, only
  `tailscale-kombuys` tmux. Trusted counts are base `16/16`, NER seed-42
  `11/11`, NER confirmations `2/4`, T2X seed-42 `11/11`, T2X confirmations
  `4/4`, POS seed-42 `3/11`, winners `0/8`, held-out `0`, Mono not started.
  Sheet E/F/G stay blank, quarantine and publication blocks remain, and the
  remaining validation work prevents a fixed full-program ETA.

- At 00:49 SAST on 2026-08-16 POS a2 `1238228` step 2264 scored exact
  validation accuracy `0.8594007031274665` over all `1,800` rows, 12 cells,
  and 17 labels. It is `0.0000912747888207` below retained checkpoint 1981
  accuracy `0.8594919779162872`, making this the first frozen threshold miss
  at patience `1/2`. Artifact/state/current-adapter hashes are
  `e6c755c8...647`/`4330749c...78d`/`c0efc576...ff5`. This remains
  operational validation-only evidence, not terminal trusted evidence. The
  sole owned A100-40GB job is healthy at step 2547 loss
  `0.12763774447970921`; its exact artifact and possible terminal decision
  are estimated around `02:55--03:15`, before the `04:11` wall-time limit.
  Stage B remains blocked and no continuation or additional job was
  submitted. There is no pending owned job or A100-80GB/L40S work. Quota is
  `70.3%/38.8%`. Kombuys remains read-only and idle at GPU0 `10 MiB/0%`,
  GPU1 `1 MiB/0%`, scratch `60%`. Trusted counts remain base `16/16`, NER
  seed-42 `11/11`, NER confirmations `2/4`, T2X seed-42 `11/11`, T2X
  confirmations `4/4`, POS seed-42 `2/11`, winners `0/8`, held-out `0`, and
  Mono not started. Sheet E/F/G stay blank; quarantine and publication blocks
  remain. Remaining validation grids prevent a fixed full-results ETA.

- At 22:19 SAST POS a2 `1238228` step 1981 improved exact validation accuracy
  to `0.8594919779162872` over all `1,800` rows, 12 cells, and 17 labels.
  Its `0.0031854468158129` gain over step 1698 exceeds the frozen `0.001`
  threshold, so checkpoint 1981 is the new raw best and patience reset to
  `0/2`. Artifact/state/adapter hashes are
  `917ff951...4c3`/`829494f8...82d`/`9683f802...1c0`. This remains
  operational validation-only progress, not terminal trusted evidence. The
  job is healthy as the sole owned A100-40GB job and reached step 2264 loss
  `0.12269888136121962`; its next artifact is estimated around
  `00:25--00:40`, with the earliest possible terminal decision at the
  following step-2547 artifact around `02:55--03:20`. The `04:11` 24-hour
  limit is now a provenance risk if either artifact resets patience again;
  no unfrozen continuation was submitted. Stage B remains blocked. There is
  no pending owned job or A100-80GB/L40S work. Quota is `70.3%/38.7%`.
  Kombuys remains read-only and idle at GPU0 `10 MiB/0%`, GPU1 `1 MiB/0%`,
  scratch `60%`. Trusted counts remain base `16/16`, NER seed-42 `11/11`,
  NER confirmations `2/4`, T2X seed-42 `11/11`, T2X confirmations `4/4`, POS
  seed-42 `2/11`, winners `0/8`, held-out `0`, and Mono not started. Sheet
  E/F/G stay blank; quarantine and publication blocks remain. Remaining
  validation grids prevent a fixed full-results ETA.

- At 19:52 SAST POS a2 `1238228` step 1698 improved exact validation accuracy
  to `0.8563065311004743` over all `1,800` rows, 12 cells, and 17 labels.
  Its `0.0025767203700779` gain over step 1415 exceeds the frozen `0.001`
  threshold, so checkpoint 1698 is the new raw best and patience reset to
  `0/2`. Artifact/state/adapter hashes are
  `aa6aa0f3...ba8`/`bafa8eb9...9f7`/`9327a404...3b1`. This remains
  operational validation-only progress, not a terminal trusted result. The
  job is healthy as the sole owned A100-40GB job and entered step-1981
  constrained validation after loss `0.11816371493869357`; its next artifact
  is estimated around `21:55--22:10`, and the reset patience moves the
  earliest possible terminal decision to roughly `00:25--00:55`. Stage B
  remains blocked; no additional job was submitted. There is no pending
  owned job or A100-80GB/L40S work. Quota is `70.3%/38.7%`. Kombuys remains
  read-only and idle at GPU0 `10 MiB/0%`, GPU1 `1 MiB/0%`, scratch `60%`.
  Trusted counts remain base `16/16`, NER seed-42 `11/11`, NER confirmations
  `2/4`, T2X seed-42 `11/11`, T2X confirmations `4/4`, POS seed-42 `2/11`,
  winners `0/8`, held-out `0`, and Mono not started. Sheet E/F/G stay blank;
  quarantine and publication blocks remain. Remaining validation grids
  prevent a fixed full-results ETA.

- At 18:21 SAST NER seed-87 confirmation `1233035` was terminal-valid:
  completed `0:0` at `18:07:40` after `17:21:13`, with final step-8115 mean
  F1 `0.6894898423629795` (Tsn/Xho/Zul
  `0.6711970726606924/0.6959178298656337/0.7013546245626127`) over exact
  `192`, `0` literal empties, `54` whitespace-only outputs, `39/56/37`
  unique outputs, and `0` parser failures. This second frozen miss leaves
  checkpoint 7033 as the validation winner at `0.6904636994389453`.
  Final-debug/state/retained/final/config hashes are
  `5b94ebaf...fa8a`/`a000611a...3f9`/`107a0242...6c4`/`f0c4d157...0c2f`/
  `62af9169...973`; exact read-only comparison proved all `424/424` tensors
  (`76,410,112` values) match the final adapter. Trusted NER confirmations
  are now `2/4`. POS a2 `1238228` is the sole owned running A100-40GB job and
  was healthy at `900/1,800` rows in step-1698 constrained validation, after
  operational loss `0.11603702757093641`; its exact artifact and earliest
  terminal decision remain estimated around `19:15--19:30`. No third job was
  submitted because Stage B waits for a2's terminal decision. There is no
  pending owned job or A100-80GB/L40S work. Quota is `70.3%/38.7%`.
  Kombuys remains read-only and idle at GPU0 `10 MiB/0%`, GPU1 `1 MiB/0%`,
  scratch `60%`. Trusted counts are base `16/16`, NER seed-42 `11/11`, NER
  confirmations `2/4`, T2X seed-42 `11/11`, T2X confirmations `4/4`, POS
  seed-42 `2/11`, winners `0/8`, held-out `0`, and Mono not started. Sheet
  E/F/G remain blank; quarantine and publication blocks remain. Remaining
  validation grids prevent a fixed full-results ETA.

- At 17:04 SAST NER seed-87 `1233035` step 7574 scored exact mean F1
  `0.6898049384622436` (Tsn/Xho/Zul
  `0.6729387168430181/0.6964671194650163/0.7000089790786965`) over exact
  `192`, with no literal empties or parser failures. It is the frozen first
  miss against retained step 7033 `0.6904636994389453`, at patience `1/2`;
  debug/state/adapter hashes are
  `372b2a48...204`/`77f976d7...0dd`/`c5ee12ba...709`. It resumed near step
  `7701/8115`, with the final artifact and terminal decision estimated around
  `18:05--18:20`. POS a2 `1238228` step 1415 set a new raw-best exact
  accuracy `0.8537298107303964` over `1,800` rows, but its
  `0.0007278971717521` gain is below the frozen `0.001` threshold, so it is
  patience `1/2` while checkpoint 1415 remains Trainer's raw best.
  Artifact/state/adapter hashes are
  `9da1ee04...a7c`/`157bce34...998`/`c9170922...ced`. It resumed near step
  `1598/4245`; the next artifact and earliest terminal decision are estimated
  around `19:15--19:30`. Exactly two A100-40GB jobs are running on
  `srvrocgpu010`, with no pending owned job or A100-80GB/L40S work. The third
  slot remains intentionally idle because Stage B waits for a2's terminal
  decision. Quota is `70.3%/38.7%`. Kombuys remains read-only and idle:
  foreign GPU 0 is `10 MiB/0%`, assigned GPU 1 is `1 MiB/0%`, and scratch is
  `60%` used. Scientifically trusted terminal counts remain base `16/16`, NER
  seed-42 `11/11`, NER confirmations `1/4`, T2X seed-42 `11/11`, T2X
  confirmations `4/4`, POS seed-42 `2/11`, winners `0/8`, held-out `0`, and
  Mono not started. Sheet E/F/G stay blank; quarantine and publication blocks
  remain. Active grids prevent a fixed full-results ETA.

- At 16:04 SAST NER seed-87 `1233035` improved at step 7033 to exact mean
  validation F1 `0.6904636994389453` (Tsn/Xho/Zul
  `0.6727676820498912/0.6974781048854626/0.7011453113814818`) over exact
  `192`, with `0` literal empties, `54` whitespace-only outputs, `39/56/37`
  unique outputs, and `0` parser failures. Checkpoint 7033 is retained at
  patience `0/2`; debug/state/adapter hashes are
  `3425049c...c8a`/`a000611a...3f9`/`107a0242...6c4`. It resumed near step
  `7354/8115`, with the next artifact estimated around `16:55--17:10`. POS
  a2 `1238228` advanced to `1,100/1,800` step-1415 constrained rows,
  retaining checkpoint 1132 accuracy `0.8530019135586443`; its artifact ETA
  remains `16:50--17:05`. Exactly two A100-40GB jobs are running on
  `srvrocgpu010`, with no pending owned job or A100-80GB/L40S work. The third
  slot remains intentionally idle because Stage B waits for a2's terminal
  decision. Quota is `70.3%/38.7%`. Kombuys remains read-only and idle:
  foreign GPU 0 is `10 MiB/0%`, assigned GPU 1 is `1 MiB/0%`, and scratch is
  `60%` used. Scientifically trusted terminal counts remain base `16/16`, NER
  seed-42 `11/11`, NER confirmations `1/4`, T2X seed-42 `11/11`, T2X
  confirmations `4/4`, POS seed-42 `2/11`, winners `0/8`, held-out `0`, and
  Mono not started. Sheet E/F/G stay blank; quarantine and publication blocks
  remain. Active grids prevent a fixed full-results ETA.

- At 15:05 SAST NER seed-87 `1233035` improved at step 6492 to exact mean
  validation F1 `0.6880731826771278` (Tsn/Xho/Zul
  `0.6716242661447642/0.692505916381754/0.7000893655048651`) over exact
  `192`, with `0` literal empties, `54` whitespace-only outputs, `39/56/37`
  unique outputs, and `0` parser failures. Checkpoint 6492 is retained at
  patience `0/2`; debug/state/adapter hashes are
  `49673d05...ef4`/`79b25e9c...e7a`/`435f1185...f78`. It reached step
  `7033/8115`, with the next exact artifact estimated around
  `15:45--16:00`. POS a2 `1238228` reached step 1415, completed operational
  loss `0.12002206590440538/1,800`, and advanced to `350/1,800` constrained
  rows, retaining checkpoint 1132 accuracy `0.8530019135586443`; its artifact
  ETA remains `16:50--17:05`. Exactly two A100-40GB jobs are running on
  `srvrocgpu010`, with no pending owned job or A100-80GB/L40S work. The third
  slot remains intentionally idle because Stage B waits for a2's terminal
  decision. Quota is `70.3%/38.7%`. Kombuys remains read-only with assigned
  GPU 1 idle at `1 MiB/0%`; foreign GPU 0 is active at `14,772 MiB/89%` and
  untouched; scratch is `60%` used. Scientifically trusted terminal counts
  remain base `16/16`, NER seed-42 `11/11`, NER confirmations `1/4`, T2X
  seed-42 `11/11`, T2X confirmations `4/4`, POS seed-42 `2/11`, winners
  `0/8`, held-out `0`, and Mono not started. Sheet E/F/G stay blank;
  quarantine and publication blocks remain. Active grids prevent a fixed
  full-results ETA.

- At 14:35 SAST POS a2 `1238228` improved at step 1132 to exact validation
  accuracy `0.8530019135586443` over `1,800` rows, 12 cells, and 17 labels;
  checkpoint 1132 is retained at patience `0/2`. Artifact/state/adapter hashes
  are `572122f7...d87`/`60add673...474`/`19028705...820`. It resumed healthy
  to step `1411/4245`, with the next exact artifact estimated around
  `16:50--17:05`. NER seed-87 `1233035` completed step-6492 operational loss
  `0.3785168956203532/10,760` and all three automatic generation segments
  without faults; retained checkpoint 5951 mean F1 remains
  `0.6865433259628579`, with the next exact artifact expected around
  `14:40--14:50`. Exactly two A100-40GB jobs are running on `srvrocgpu010`,
  with no pending owned job or A100-80GB/L40S work. The third slot remains
  intentionally idle because Stage B waits for a2's terminal decision. Quota
  is `70.3%/38.7%`. Kombuys remains read-only with assigned GPU 1 idle at
  `1 MiB/0%`; foreign GPU 0 is active at `19,318 MiB/97%` and untouched;
  scratch is `60%` used. Scientifically trusted terminal counts remain base
  `16/16`, NER seed-42 `11/11`, NER confirmations `1/4`, T2X seed-42 `11/11`,
  T2X confirmations `4/4`, POS seed-42 `2/11`, winners `0/8`, held-out `0`,
  and Mono not started. Sheet E/F/G stay blank; quarantine and publication
  blocks remain. Active grids prevent a fixed full-results ETA.

- At 14:02 SAST POS a1 `1233034` became terminal-valid after local read-only
  proof of exact equality for all `424/424` tensors (`71,762,560` values)
  between retained checkpoint 1415 and `final_adapter`; POS seed-42 terminal
  progress is now `2/11`. Redundant pending verifier `1238386` never
  allocated or accessed model/data and was cancelled at elapsed `00:00:00`.
  NER seed-87 `1233035` step 5410 scored mean F1
  `0.6796900433721967`, then step 5951 improved to retained
  `0.6865433259628579` (Tsn/Xho/Zul
  `0.6641949152541873/0.6956059720524448/0.6998290905819415`) at patience
  `0/2`. Coverage is exact `192`; step 5951 has `0` literal empties, `54`
  whitespace-only outputs, `39/56/37` unique outputs, and `0` parser
  failures. Debug/state/adapter hashes are
  `9d0c1252...b46`/`73f1dda5...7a5`/`86a27eeb...afa`. It entered step-6492
  full validation, with the next artifact estimated around `14:50--15:05`.
  POS a2 `1238228` is healthy at `1,450/1,800` step-1132 constrained rows,
  retaining `0.8411869496098022`, with an artifact estimated around
  `14:20--14:35`. Stage B waits for a2's terminal decision. Exactly two
  A100-40GB jobs are running; no pending owned job or A100-80GB/L40S work
  exists. Quota is `70.3%/38.7%`. Kombuys remains read-only with assigned
  GPU 1 idle at `1 MiB/0%`; foreign GPU 0 is active at `16,992 MiB/96%` and
  untouched; scratch is `60%` used. Other trusted counts remain base `16/16`,
  NER seed-42 `11/11`, NER confirmations `1/4`, T2X seed-42 `11/11`, T2X
  confirmations `4/4`, winners `0/8`, held-out `0`, Mono not started. Sheet
  E/F/G remain blank; publication stays blocked. Active grids prevent a
  fixed full-results ETA.

- At 12:23 SAST POS a1 `1233034` completed `0:0` after `17:53:22`.
  Step-1981 accuracy `0.8432799447332778` left checkpoint 1415 raw-best
  `0.8474682174611892` retained after the frozen second miss. Artifact,
  retained-state, retained-adapter, final-adapter, and final-config hashes are
  `a9d43240...73f`/`118b3055...324`/`9711a463...0f1`/
  `14f37519...db5`/`d828b30f...99b`. Exact retained/final tensor verification
  job `1238386` is priority-pending without allocation; a1 is operationally
  complete but not terminal-valid, so POS terminal progress stays `1/11`.
  POS a2 `1238228` improved at step 849 to exact accuracy
  `0.8411869496098022`, retained at patience `0/2`; hashes are
  `1662c7b2...df9`/`e7b81446...15d`/`5d65fc9a...332`. It entered step-1132
  validation at `100/1,800`, with its next artifact estimated around
  `14:15--14:30`. NER `1233035` remains healthy in step-5410 generation,
  retaining `0.6802812718619565`, with an artifact estimated around
  `12:25--12:35`. Owned state is two running plus pending `1238386`, all
  A100-40GB; no A100-80GB/L40S work exists. Quota is `70.3%/38.7%`.
  Kombuys remains read-only with assigned GPU 1 idle at `1 MiB/0%`; foreign
  GPU 0 is active at `20,144 MiB/99%` and untouched; scratch is `60%` used.
  Other trusted counts remain base `16/16`, NER seed-42 `11/11`, NER
  confirmations `1/4`, T2X seed-42 `11/11`, T2X confirmations `4/4`, winners
  `0/8`, held-out `0`, Mono not started. Sheet E/F/G remain blank;
  publication stays blocked. Active grids and the a1 proof prevent a fixed
  full-results ETA.

- At 11:46 SAST POS a1 `1233034` and a2 `1238228` reached `1,600/1,800` and
  `1,700/1,800` constrained rows, retaining eligible accuracies
  `0.8474682174611892` and `0.817024570725743`; both exact artifacts are
  estimated around `11:55--12:05`. NER seed-87 `1233035` reached step
  `5410/8115`, full operational validation loss
  `0.3694716556364719/10,760`, and entered its tenth frozen generation
  callback without faults. The loss is not a selection metric; retained
  step-4869 mean F1 remains `0.6802812718619565`, with the next artifact
  estimated around `12:15--12:30`. Exactly three healthy A100-40GB jobs are
  running on `srvrocgpu010`; no A100-80GB/L40S work exists. Quota is
  `70.3%/38.7%`. Kombuys remains read-only with assigned GPU 1 idle at
  `1 MiB/0%`; foreign GPU 0 is active at `21,508 MiB/99%` and untouched;
  scratch is `60%` used. Trusted terminal counts remain base `16/16`, NER
  seed-42 `11/11`, NER confirmations `1/4`, T2X seed-42 `11/11`, T2X
  confirmations `4/4`, POS seed-42 `1/11`, winners `0/8`, held-out `0`, Mono
  not started. Sheet E/F/G remain blank; publication stays blocked. Active
  grids prevent a fixed full-results ETA.

- At 11:11 SAST NER seed-87 `1233035` improved at step 4869 to exact mean
  validation F1 `0.6802812718619565` (Tsn/Xho/Zul
  `0.6592776111617704/0.6831280071907713/0.6984381972333281`) over exact
  `192`, with `0` literal empties, `49` whitespace-only values, `42/56/39`
  unique values, and `0` parser failures. Checkpoint 4869 is retained at
  patience `0/2`; debug/state/adapter hashes are
  `1687c8f7...280`/`6f32f792...4d9`/`74f79666...d3e`. It resumed near
  `4986/8115`, with the next artifact estimated around `12:10--12:25`. POS
  a1 `1233034` and a2 `1238228` advanced to `1,150/1,800` and
  `1,250/1,800` constrained rows, retaining accuracies
  `0.8474682174611892` and `0.817024570725743`; their next artifacts are
  estimated around `11:50--12:10`. Exactly three healthy A100-40GB jobs are
  running on `srvrocgpu010`; no A100-80GB/L40S work exists. Quota is
  `70.3%/38.7%`. Kombuys remains read-only with assigned GPU 1 idle at
  `1 MiB/0%`; foreign GPU 0 is active at `24,130 MiB/95%` and untouched;
  scratch is `60%` used. Trusted terminal counts remain base `16/16`, NER
  seed-42 `11/11`, NER confirmations `1/4`, T2X seed-42 `11/11`, T2X
  confirmations `4/4`, POS seed-42 `1/11`, winners `0/8`, held-out `0`, Mono
  not started. Sheet E/F/G remain blank; publication stays blocked. Active
  grids prevent a fixed full-results ETA.

- At 10:41 SAST NER seed-87 `1233035` reached step `4869/8115`, full
  operational validation loss `0.361292127871602/10,760`, and entered its
  ninth frozen generation callback without faults. The loss is not a
  selection metric; retained step-4328 mean F1 remains
  `0.6748547444980355`, with the next exact artifact estimated around
  `11:00--11:15`. POS a1 `1233034` and a2 `1238228` advanced to
  `750/1,800` and `850/1,800` constrained rows, retaining eligible accuracies
  `0.8474682174611892` and `0.817024570725743`; their next artifacts remain
  estimated around `11:50--12:10`. Exactly three healthy A100-40GB jobs are
  running on `srvrocgpu010`; no A100-80GB/L40S work exists. Quota is
  `70.3%/38.7%`. Kombuys remains read-only with assigned GPU 1 idle at
  `1 MiB/0%`; foreign GPU 0 is active at `17,238 MiB/97%` and untouched;
  scratch is `60%` used. Trusted terminal counts remain base `16/16`, NER
  seed-42 `11/11`, NER confirmations `1/4`, T2X seed-42 `11/11`, T2X
  confirmations `4/4`, POS seed-42 `1/11`, winners `0/8`, held-out `0`, Mono
  not started. Sheet E/F/G remain blank; publication stays blocked. Active
  grids prevent a fixed full-results ETA.

- At 10:10 SAST NER seed-87 `1233035` improved twice: step 3787 mean F1
  `0.6673562272065903`, then retained step 4328 mean F1
  `0.6748547444980355` (Tsn/Xho/Zul
  `0.6579163248563898/0.6699258445481515/0.6967220640895654`). Coverage is
  exact `192`; step 4328 has `0` literal empties, `54` whitespace-only values,
  `39/56/37` unique values, and `0` parser failures. It is retained at
  patience `0/2`; debug/state/adapter hashes are
  `d1324aff...14e`/`12039d02...a86`/`12d4a3ce...6c1`. The job resumed near
  `4586/8115`, with its next exact artifact estimated around
  `11:00--11:15`. POS a1 `1233034` step 1698 scored
  `0.8429252468406908`, leaving step 1415 `0.8474682174611892` retained at
  patience `1/2`; hashes are `50e7069b...260`/`a7b00a8b...3a5`/
  `21c03971...0dc`. POS a2 `1238228` improved at step 566 to
  `0.817024570725743`, retained at patience `0/2`; hashes are
  `6916f581...93a`/`a5a8790a...d27`/`d6e14db8...5f6`. Both cover exact
  `1,800` rows, 12 cells, and 17 labels. Operationally, their next callbacks
  are step 1981 at `350/1,800` and step 849 at `400/1,800`, with artifacts
  estimated around `11:50--12:10`. Exactly three healthy A100-40GB jobs are
  running on `srvrocgpu010`; no A100-80GB/L40S work exists. Quota is
  `70.3%/38.7%`. Kombuys remains read-only with assigned GPU 1 idle at
  `1 MiB/0%`; foreign GPU 0 is active at `16,854 MiB/88%` and untouched;
  scratch is `60%` used. Trusted terminal counts remain base `16/16`, NER
  seed-42 `11/11`, NER confirmations `1/4`, T2X seed-42 `11/11`, T2X
  confirmations `4/4`, POS seed-42 `1/11`, winners `0/8`, held-out `0`, Mono
  not started. Sheet E/F/G remain blank; publication stays blocked. Active
  grids prevent a fixed full-results ETA.

- At 07:58 SAST NER seed-87 `1233035` improved at step 3246 to exact mean
  validation F1 `0.6536194168330306` (Tsn/Xho/Zul
  `0.6384674723578827/0.6380013420739707/0.6843894360672385`) over exact
  `192`, with `0` literal empties, `46` whitespace-only values, `43/57/41`
  unique values, and `0` parser failures. Checkpoint 3246 is retained at
  patience `0/2`; debug/state/adapter hashes are
  `88ac1f96...63b`/`f0b81e9a...593`/`723d528a...764`. It resumed near
  `3609/8115`, with the next exact artifact estimated around
  `08:55--09:10`. POS a1 `1233034` and a2 `1238228` are healthy at
  `600/1,800` and `700/1,800` constrained rows, retaining accuracies
  `0.8474682174611892` and `0.7964588530699107`; their next exact artifacts
  are estimated around `09:10--09:30`. Exactly three A100-40GB jobs are
  running on `srvrocgpu010`; no A100-80GB/L40S work exists. Quota is
  `70.3%/38.6%`. Kombuys remains read-only with assigned GPU 1 idle at
  `1 MiB/0%`; foreign GPU 0 is active at `25,626 MiB/97%` and untouched;
  scratch is `60%` used. Trusted terminal counts remain base `16/16`, NER
  seed-42 `11/11`, NER confirmations `1/4`, T2X seed-42 `11/11`, T2X
  confirmations `4/4`, POS seed-42 `1/11`, winners `0/8`, held-out `0`, Mono
  not started. Sheet E/F/G remain blank; publication stays blocked. Active
  grids prevent a fixed full-results ETA.

- At 07:27 SAST NER seed-87 `1233035` completed step-3246 full validation
  loss `0.33468470236625814/10,760` and all three generation segments without
  faults; retained eligible mean F1 remains `0.648578345565961`, with the
  next artifact expected around `07:40--07:50`. POS a1 `1233034` reached
  step `1698/4245`, full loss `0.1237556160820855/1,800`, and `200/1,800`
  constrained rows. POS a2 `1238228` reached step `566/4245`, full loss
  `0.13736961364746095/1,800`, and `300/1,800` constrained rows. The losses
  are operational only; retained accuracies remain `0.8474682174611892` and
  `0.7964588530699107`, with next artifacts expected around
  `09:15--09:35`. Exactly three A100-40GB jobs remain healthy on
  `srvrocgpu010`; no A100-80GB/L40S work exists. Quota is `70.3%/38.6%`.
  Kombuys remains read-only with assigned GPU 1 idle at `1 MiB/0%`; foreign
  GPU 0 is active at `17,290 MiB/95%` and untouched; scratch is `61%` used.
  Trusted terminal counts remain base `16/16`, NER seed-42 `11/11`, NER
  confirmations `1/4`, T2X seed-42 `11/11`, T2X confirmations `4/4`, POS
  seed-42 `1/11`, winners `0/8`, held-out `0`, Mono not started. Sheet E/F/G
  remain blank; publication stays blocked. Active grids prevent a fixed
  full-results ETA.

- At 06:58 SAST three exact validation-only artifacts landed. NER seed-87
  `1233035` improved at step 2705 to mean F1 `0.648578345565961`
  (Tsn/Xho/Zul `0.6576375314157558/0.6211193703541262/0.666978134928001`)
  over exact `192`, with `0` literal empties, `54` whitespace-only values,
  `39/56/37` unique values, and `0` parser failures. Checkpoint 2705 is
  retained at patience `0/2`; debug/state/adapter hashes are
  `fbe0c973...e36`/`1bd1b618...b23`/`150322c4...14f`. It reached its next
  step `3246/8115` callback, with another artifact expected around
  `07:40--07:55`. POS a1 `1233034` improved at step 1415 to exact accuracy
  `0.8474682174611892`; hashes are `f779b245...c48`/`118b3055...324`/
  `9711a463...0f1`. POS a2 `1238228` produced its first exact step-283
  accuracy `0.7964588530699107`; hashes are `9f7fd5ac...0cf`/
  `9d341b9f...fcf`/`75c0c7a5...498`. Both cover exact `1,800` rows, 12 cells,
  and 17 labels at patience `0/2`; their next artifacts are expected around
  `09:00--09:25`. Exactly three A100-40GB jobs remain healthy on
  `srvrocgpu010`; no A100-80GB/L40S work exists. Quota is `70.3%/38.6%`.
  Kombuys remains read-only with assigned GPU 1 idle at `1 MiB/0%`; foreign
  GPU 0 is active at `14,972 MiB/97%` and untouched. These are valid interim
  artifacts; terminal trusted counts remain base `16/16`, NER seed-42
  `11/11`, NER confirmations `1/4`, T2X seed-42 `11/11`, T2X confirmations
  `4/4`, POS seed-42 `1/11`, winners `0/8`, held-out `0`, Mono not started.
  Sheet E/F/G remain blank; publication stays blocked. Active grids prevent a
  fixed full-results ETA.

- At 06:27 SAST all three callbacks are near artifact completion. NER
  seed-87 `1233035` advanced its step-2705 generation segments at `05:55:11`,
  `06:01:06`, and `06:15:44` without faults; retained eligible mean F1 stays
  `0.5993678536371205`, with the next artifact expected around
  `06:30--06:40`. POS a1 `1233034` reached `1,450/1,800` constrained rows,
  retaining `0.8268296279293154`; POS a2 `1238228` reached `1,500/1,800`.
  Their next/first exact artifacts are expected around `06:45--07:00`.
  Exactly three A100-40GB jobs are running on `srvrocgpu010`; no
  A100-80GB/L40S work exists. Quota remains `70.3%/38.6%`. Kombuys remains
  read-only with assigned GPU 1 idle at `1 MiB/0%`; foreign GPU 0 is active
  at `15,652 MiB/99%` and untouched. Trusted terminal progress remains base
  `16/16`, NER seed-42 `11/11`, NER confirmations `1/4`, T2X seed-42
  `11/11`, T2X confirmations `4/4`, POS seed-42 `1/11`, global winners `0/8`,
  held-out `0`, Mono not started. Sheet E/F/G remain blank; publication stays
  blocked. Active grids prevent a fixed full-results ETA.

- At 05:57 SAST NER seed-87 `1233035` reached step `2705/8115`, completed
  full validation loss `0.331830769932403/10,760`, and entered its fifth
  frozen generation callback without faults. The loss is operational only;
  retained eligible mean F1 remains `0.5993678536371205`, with the next exact
  artifact expected around `06:30--06:45`. POS a1 `1233034` advanced to
  `1,050/1,800` constrained rows, retaining `0.8268296279293154`; POS a2
  `1238228` advanced to `1,150/1,800`. Both are healthy, with their
  next/first exact artifacts expected around `06:45--07:00`. Exactly three
  A100-40GB jobs are running on `srvrocgpu010`; no A100-80GB/L40S work
  exists. Quota remains `70.3%/38.6%`. Kombuys remains read-only with
  assigned GPU 1 idle at `1 MiB/0%`; foreign GPU 0 is active at
  `20,682 MiB/92%` and untouched. Trusted terminal progress remains base
  `16/16`, NER seed-42 `11/11`, NER confirmations `1/4`, T2X seed-42
  `11/11`, T2X confirmations `4/4`, POS seed-42 `1/11`, global winners `0/8`,
  held-out `0`, Mono not started. Sheet E/F/G remain blank; publication stays
  blocked. Active grids prevent a fixed full-results ETA.

- At 05:29 SAST NER seed-87 confirmation `1233035` improved at step 2164 to
  exact mean validation F1 `0.5993678536371205` (Tsn/Xho/Zul
  `0.5816062176165304/0.578320705118243/0.6381766381765882`) over exact
  `192`, `64/language`. Raw outputs have `0` literal empties, `56`
  whitespace-only values (`21/8/27`), `39/56/37` unique values, and `0`
  parser failures. Checkpoint 2164 is retained at patience `0/2`;
  debug/state/adapter hashes are `c28c9781...31d`/`3cafef57...930`/
  `2eaa8c7d...96b`. It resumed healthy near `2315/8115`, with the next
  artifact expected around `06:30--06:45`. POS a1 `1233034` is healthy at
  `650/1,800` constrained rows, retaining `0.8268296279293154`, with its next
  artifact expected around `06:50--07:05`; POS a2 `1238228` is healthy at
  `750/1,800`, with its first artifact expected around `06:40--06:55`.
  Exactly three A100-40GB jobs are running on `srvrocgpu010`; no
  A100-80GB/L40S work exists. Quota remains `70.3%/38.6%`. Kombuys remains
  read-only with assigned GPU 1 idle at `1 MiB/0%`; foreign GPU 0 is active
  at `14,902 MiB/89%` and untouched. Trusted terminal progress remains base
  `16/16`, NER seed-42 `11/11`, NER confirmations `1/4`, T2X seed-42
  `11/11`, T2X confirmations `4/4`, POS seed-42 `1/11`, global winners `0/8`,
  held-out `0`, Mono not started. Sheet E/F/G remain blank; publication stays
  blocked. Active grids prevent a fixed full-results ETA.

- At 04:57 SAST all three owned A100-40GB jobs remain healthy on
  `srvrocgpu010`. POS a1 `1233034` reached step `1415/4245`, completed full
  validation loss `0.12730493757459851/1,800`, and advanced to `250/1,800`
  constrained rows; retained eligible accuracy remains
  `0.8268296279293154`, with the next artifact expected around
  `06:45--07:00`. NER seed-87 `1233035` reached step `2164/8115`, completed
  full loss `0.32311224387924026/10,760`, and entered its fourth generation
  callback without faults; retained mean F1 remains `0.5768414049419194`,
  with the next artifact expected around `05:15--05:30`. POS a2 `1238228`
  reached step `283/4245`, completed first full validation loss
  `0.23404678344726562/1,800`, and advanced to `350/1,800` constrained rows;
  its first selection artifact is expected around `06:40--06:55`. No
  A100-80GB/L40S work exists. Quota remains `70.3%/38.6%`. Kombuys remains
  read-only with assigned GPU 1 idle at `1 MiB/0%`; foreign GPU 0 is active
  at `14,590 MiB/92%` and untouched. Trusted terminal progress is unchanged:
  base `16/16`, NER seed-42 `11/11`, NER confirmations `1/4`, T2X seed-42
  `11/11`, T2X confirmations `4/4`, POS seed-42 `1/11`, global winners `0/8`,
  held-out `0`, Mono not started. Sheet E/F/G remain blank; publication stays
  blocked. Active grids prevent a fixed full-results ETA.

- At 04:27 SAST POS a1 `1233034` improved at step 1132 to exact accuracy
  `0.8268296279293154` over `1,800` rows, 12 cells, and 17 labels;
  checkpoint 1132 is retained. Artifact/state/adapter hashes are
  `60be578c...743`/`609cc077...dd6`/`8d62e2ae...46e`. NER b7 seed-87
  `1233035` improved at step 1623 to exact mean F1 `0.5768414049419194`
  (Tsn/Xho/Zul `0.564731240738196/0.557582976880128/0.608209997207434`)
  over exact `192` rows; debug/state/adapter hashes are
  `73228958...fa1`/`477cb51d...725`/`cffe3dfa...326`. Both resumed healthy,
  with next artifacts expected around `06:40--07:00` and `05:10--05:30`.
  POS a2 `1238228` started at `04:11:21`, verified `694/694` immutable files
  and exact `2,259/1,800` train/validation rows, and was healthy near
  `275/4245`; its first artifact is expected around `06:35--06:55`.
  Manifest/trial hashes are `65b62cb0...472`/`4edadeb3...122`. Owned state is
  exactly three running A100-40GB jobs; no A100-80GB/L40S work exists. Quota
  is `70.3%/38.6%`. Kombuys remains read-only with assigned GPU 1 idle at
  `1 MiB/0%`; foreign GPU 0 is active at `18,394 MiB/92%` and untouched.
  Trusted terminal progress remains base `16/16`, NER seed-42 `11/11`, NER
  confirmations `1/4`, T2X seed-42 `11/11`, T2X confirmations `4/4`, POS
  seed-42 `1/11`, global winners `0/8`, held-out `0`, Mono not started. Sheet
  E/F/G remain blank; publication stays blocked. Active validation grids
  prevent a fixed full-results ETA.

- At 03:31 SAST POS a0 `1232181` is terminal-valid: completed `0:0` at
  `03:09:18` after `22:49:21`, with step-2547 raw-best accuracy
  `0.8211477429884227`. The `0.0006173890501758` improvement was below the
  frozen `0.001` threshold, so patience reached `2/2` and training stopped;
  checkpoint 2547 remained the trainer's raw best. Artifact/state/current-
  adapter/final-adapter/config hashes are `c1c8b115...837`/
  `833600b5...690`/`402447d6...dcd`/`a8cf6bc3...059`/`d7682a34...546`.
  Local read-only verification proved exact equality for `424/424` tensors
  (`71,762,560` values) between retained and final serialization. POS
  seed-42 terminal progress is now `1/11`. Preregistered POS a2 seed-42 was
  submitted as job `1238228` and is resource-pending without model/data
  access. POS a1 `1233034` is healthy at `1150/1800` rows in step-1132
  validation, retaining `0.8225303477478206`, with its next artifact expected
  around `04:15--04:30`. NER seed-87 `1233035` entered its step-1623 third
  callback, retaining `0.5257548232368509`, with its next artifact expected
  around `04:00--04:20`. Owned state is two running plus one pending
  A100-40GB job; no A100-80GB/L40S work exists. Quota is `70.3%/38.6%`.
  Kombuys remains read-only with assigned GPU 1 idle at `1 MiB/0%`; foreign
  GPU 0 is active at `17,084 MiB/97%` and untouched. Trusted progress is base
  `16/16`, NER seed-42 `11/11`, NER confirmations `1/4`, T2X seed-42
  `11/11`, T2X confirmations `4/4`, POS seed-42 `1/11`, global winners
  `0/8`, held-out `0`, Mono not started. Sheet E/F/G remain blank;
  publication stays blocked. A2 allocation and remaining validation grids
  prevent a fixed full-results ETA.

- At 02:57 SAST NER b7 seed-87 confirmation `1233035` improved at step 1082
  to exact mean F1 `0.5257548232368509` (Tsn/Xho/Zul
  `0.5114958074113647/0.5188186965692855/0.5469499657299023`) over exact
  `192` rows, `64/language`; debug/state/adapter hashes are
  `9af4d0f3...672`/`4fbe87a2...ab3`/`cff24f23...a88`. It resumed healthy
  near `1102/8115`, with the next artifact expected around `04:00--04:20`.
  POS a0 `1232181` is healthy at `1600/1800` rows in step-2547 validation,
  retaining `0.8205303539382469` at patience `1/2`, with its next decision
  expected around `03:08--03:15`. POS a1 `1233034` is healthy at `700/1800`
  rows in step-1132 validation, retaining `0.8225303477478206`, with its next
  artifact expected around `04:05--04:25`. All three owned jobs remain on
  A100-40GB; no A100-80GB/L40S work exists. Quota is `70.3%/38.6%`.
  Kombuys remains read-only with assigned GPU 1 idle at `1 MiB/0%`; foreign
  GPU 0 is active at `23,408 MiB/92%` and untouched. Terminal trusted
  progress remains base `16/16`, NER seed-42 `11/11`, NER confirmations
  `1/4`, T2X seed-42 `11/11`, T2X confirmations `4/4`, POS `0/11` with 11
  valid interim artifacts, global winners `0/8`, held-out `0`, Mono not
  started. Sheet E/F/G remain blank; publication stays blocked. Remaining
  validation grids prevent a fixed full-results ETA.

- At 01:58 SAST POS a1 `1233034` improved at step 849 to exact accuracy
  `0.8225303477478206`, covering `1,800` rows, 12 cells, and 17 labels;
  checkpoint 849 is retained. Artifact/state/adapter hashes are
  `fd975808...480`/`3a0670b8...1ba`/`051dcb84...40a`. It resumed healthy
  near `1087/4245`, with the next artifact expected around `04:05--04:25`.
  NER b7 seed-87 `1233035` produced its first exact step-541 mean F1
  `0.32363493061792803` (Tsn/Xho/Zul
  `0.3202108384120759/0.2890758511896485/0.36161810225205965`) over exact
  `192` rows, `64/language`; debug/state/adapter hashes are
  `d5109492...206`/`fa19ae0e...624`/`53d6def6...911`. It resumed healthy near
  `825/8115`, with the next artifact expected around `02:35--02:55`. POS a0
  `1232181` is healthy at `800/1800` rows in step-2547 validation, retaining
  `0.8205303539382469` at patience `1/2`, with its next artifact expected
  around `03:05--03:20`. All three owned jobs are running on A100-40GB; no
  A100-80GB/L40S work exists. Quota is `70.3%/38.6%`. Kombuys remains
  read-only with assigned GPU 1 idle at `1 MiB/0%`; foreign GPU 0 is active
  at `24,334 MiB/94%` and untouched. Terminal trusted progress remains base
  `16/16`, NER seed-42 `11/11`, NER confirmations `1/4`, T2X seed-42
  `11/11`, T2X confirmations `4/4`, POS `0/11` with 11 valid interim
  artifacts, global winners `0/8`, held-out `0`, Mono not started. Sheet
  E/F/G remain blank; publication stays blocked. Remaining validation grids
  prevent a fixed full-results ETA.

- At 00:57 SAST POS a0 `1232181` produced step-2264 exact accuracy
  `0.8153398128421713`, below retained step-1981 best
  `0.8205303539382469`; frozen patience is `1/2`. Exact coverage remains
  `1,800` rows, 12 cells, and 17 labels. Artifact/state/current-adapter hashes
  are `6d376484...af8`/`0ae11dc6...872`/`8ecc7f65...93d`. It is healthy at
  `50/1800` rows in step-2547 validation, with the next artifact expected
  around `03:05--03:20`. NER b7 seed-87 confirmation `1233035` started at
  `00:46:27`, passed immutable startup and exact `4,323` training-row
  tokenization, and is healthy near `191/8115`; its first artifact is
  expected around `01:45--02:10`. POS a1 `1233034` is healthy at `1150/1800`
  rows in step-849 validation, retaining `0.7763574498537653`, with its next
  artifact expected around `01:40--01:55`. All three owned jobs are running
  on A100-40GB; no A100-80GB/L40S work exists. Quota is `70.3%/38.5%`.
  Kombuys remains read-only with assigned GPU 1 idle at `1 MiB/0%`; foreign
  GPU 0 is active at `19,230 MiB/97%` and untouched. Trusted terminal
  progress remains base `16/16`, NER seed-42 `11/11`, NER confirmations
  `1/4`, T2X seed-42 `11/11`, T2X confirmations `4/4`, POS `0/11` with ten
  valid interim artifacts, global winners `0/8`, held-out `0`, Mono not
  started. Sheet E/F/G remain blank; publication stays blocked. Remaining
  validation grids prevent a fixed full-results ETA.

- At 23:27 SAST POS a1 `1233034` improved at step 566 to exact
  `all_token_accuracy=0.7763574498537653`, covering all `1,800` rows, 12
  language/template cells, and 17 labels; checkpoint 566 is retained.
  Artifact/trainer-state/adapter hashes are `184caafe...a774`/
  `fe311cc7...a3ee`/`c630ecf3...e107`. It remains healthy near the step-849
  boundary, with its next artifact expected around `01:40--01:55`. POS a0
  `1232181` is healthy at `850/1800` rows in step-2264 validation, retaining
  `0.8205303539382469`, with its next artifact expected around
  `00:30--00:45`. NER b7 seed-87 `1233035` remains resource-pending without
  model/data access. Owned state is two running plus one pending A100-40GB
  job and no A100-80GB/L40S work; quota is `70.3%/38.5%`. Kombuys remains
  read-only with assigned GPU 1 idle at `1 MiB/0%`; foreign GPU 0 is active
  at `23,040 MiB/96%` and untouched. Trusted terminal progress is unchanged:
  base `16/16`, NER seed-42 `11/11`, NER confirmations `1/4`, T2X seed-42
  `11/11`, T2X confirmations `4/4`, POS `0/11` with nine valid interim
  artifacts, global winners `0/8`, held-out `0`, Mono not started. Sheet
  E/F/G remain blank; publication stays blocked. Queue time and remaining
  validation grids still prevent a fixed full-results ETA.

- At 22:27 SAST corrected POS a0 `1232181` improved at step 1981 to exact
  `all_token_accuracy=0.8205303539382469`, covering all `1,800` rows, 12
  language/template cells, and 17 labels; checkpoint 1981 is retained.
  Artifact/trainer-state/adapter hashes are `64fb032a...ebd9`/
  `0d3ec38c...49a4`/`75235996...960b`. It is healthy at `50/1800` rows in
  step-2264 validation, with the next artifact expected around
  `00:30--00:45`. POS a1 `1233034` is healthy at `1150/1800` rows in
  step-566 validation, with its next artifact expected around
  `23:10--23:25`; NER b7 seed-87 `1233035` remains resource-pending without
  model/data access. Owned state is two running plus one pending A100-40GB
  job and no A100-80GB/L40S work; quota is `70.3%/38.5%`. Kombuys remains
  read-only with assigned GPU 1 idle at `1 MiB/0%`; foreign GPU 0 is active
  at `15,426 MiB/99%` and untouched. Trusted terminal progress is unchanged:
  base `16/16`, NER seed-42 `11/11`, NER confirmations `1/4`, T2X seed-42
  `11/11`, T2X confirmations `4/4`, POS `0/11` with eight valid interim
  artifacts, global winners `0/8`, held-out `0`, Mono not started. Sheet
  E/F/G remain blank; publication stays blocked. Queue time and remaining
  validation grids still prevent a fixed full-results ETA.

- At 20:57 SAST POS a1 `1233034` produced its first exact constrained metric
  at step 283: `all_token_accuracy=0.722852150097027`, covering all `1,800`
  rows, 12 language/template cells, and 17 labels; checkpoint 283 is
  retained. Artifact/trainer-state/adapter hashes are
  `511d540b...8438`/`a97fcf74...d54c`/`79e6f10a...dce2`. The job remains
  healthy at its next epoch boundary, with its next exact artifact expected
  around `23:10--23:25`. POS a0 `1232181` is healthy at `850/1800` rows in
  step-1981 validation, retaining `0.8153800322428616`, with its next
  artifact expected around `22:00--22:15`. NER b7 seed-87 `1233035` remains
  resource-pending without model/data access. Owned state is two running plus
  one pending A100-40GB job and no A100-80GB/L40S work; quota is
  `70.3%/38.5%`. Kombuys remains read-only with assigned GPU 1 idle at
  `1 MiB/0%`; foreign GPU 0 is active at `17,692 MiB/99%` and untouched.
  Trusted terminal progress is unchanged: base `16/16`, NER seed-42
  `11/11`, NER confirmations `1/4`, T2X seed-42 `11/11`, T2X confirmations
  `4/4`, POS `0/11` with seven valid interim artifacts, global winners
  `0/8`, held-out `0`, Mono not started. Sheet E/F/G remain blank;
  publication stays blocked. Queue time and remaining validation grids still
  prevent a fixed full-results ETA.

- At 19:57 SAST corrected POS a0 `1232181` improved at step 1698 to exact
  `all_token_accuracy=0.8153800322428616`, covering all `1,800` rows, 12
  language/template cells, and 17 labels; checkpoint 1698 is retained.
  Artifact/trainer-state/adapter hashes are `a0e327fc...05b2`/
  `51bd5edf...4c7a`/`9d34c70a...0d8e`. It is healthy at `50/1800` rows in
  step-1981 validation, with the next artifact expected around
  `22:00--22:15`. POS a1 `1233034` is healthy at `1200/1800` rows in its
  first constrained callback, with its first metric expected around
  `20:35--20:50`; NER b7 seed-87 `1233035` remains resource-pending without
  model/data access. Owned state is two running plus one pending A100-40GB
  job and no A100-80GB/L40S work; quota is `70.3%/38.4%`. Kombuys remains
  read-only with assigned GPU 1 idle at `1 MiB/0%`; foreign GPU 0 is active
  at `17,700 MiB/94%` and untouched. Trusted terminal progress is unchanged:
  base `16/16`, NER seed-42 `11/11`, NER confirmations `1/4`, T2X seed-42
  `11/11`, T2X confirmations `4/4`, POS `0/11` with six valid interim
  artifacts, global winners `0/8`, held-out `0`, Mono not started. Sheet
  E/F/G remain blank; publication stays blocked. Queue time and remaining
  validation grids still prevent a fixed full-results ETA.

- At 18:27 SAST POS a1 environment-only retry `1233034` is healthy on
  `srvrocgpu010` A100-40GB. It passed Python 3.12, `694/694` immutable-source,
  pure-GDN kernel, and exact `2,259` train/`1,800` validation row gates,
  then completed step-283 full validation loss `0.3147279442681207/1,800`
  and entered constrained validation. Its first exact metric is expected
  around `20:35--20:50`. The immutable wrapper resolved the run/output to
  the canonical a1 seed-42 path rather than the submitted `sourcefix2`
  override; no prior artifact was overwritten because a1 jobs `1232086` and
  `1232226` never reached manifest/model/data access there. Manifest/trial
  hashes are `ba9623b6...44f6`/`8dc2990d...a5f`. POS a0 `1232181` remains
  healthy at `900/1800` rows in step-1698 validation, retaining
  `0.8096575744368201`, with its next artifact expected around
  `19:30--19:45`. NER b7 seed-87 `1233035` remains resource-pending without
  model/data access. Owned state is two running plus one pending A100-40GB
  job and no A100-80GB/L40S work; quota is `70.3%/38.4%`. Kombuys remains
  read-only with assigned GPU 1 idle at `1 MiB/0%`; foreign GPU 0 is active
  at `17,446 MiB/87%` and untouched. Scientific terminal counts, blank Sheet
  E/F/G, held-out gate, and publication block are unchanged.

- At 17:27 SAST corrected POS a0 `1232181` improved at step 1415 to exact
  `all_token_accuracy=0.8096575744368201`, covering all `1,800` rows, 12
  language/template cells, and 17 labels under the frozen constrained
  protocol; checkpoint 1415 is retained. Artifact/trainer-state/adapter
  hashes are `de039e95...9873`/`49e831d1...a2a`/`1c018f67...e3b`.
  It is healthy at `100/1800` rows in step-1698 validation, with the next
  artifact expected around `19:35--19:50`. POS a1 retry `1233034` remains
  resource-pending and NER b7 seed-87 `1233035` remains priority-pending,
  both without model/data access. Owned state is one running plus two pending
  A100-40GB jobs and no A100-80GB/L40S work; quota is `70.3%/38.4%`.
  Kombuys remains read-only with assigned GPU 1 idle at `1 MiB/0%`; foreign
  GPU 0 is active at `17,310 MiB/96%` and untouched. Trusted terminal
  progress remains base `16/16`, NER seed-42 `11/11`, NER confirmations
  `1/4`, T2X seed-42 `11/11`, T2X confirmations `4/4`, POS `0/11` with five
  valid interim artifacts, global winners `0/8`, held-out `0`, Mono not
  started. Sheet E/F/G remain blank; publication stays blocked. Queue time
  and remaining validation grids still prevent a fixed full-results ETA.

- At 15:32 SAST NER b7 seed-13 confirmation `1232204` completed `0:0` after
  `10:47:18`. Step 4869 mean validation F1
  `0.6681165754371922` (Tsn/Xho/Zul
  `0.6494543518764473/0.6725933719094714/0.6823020025256581`) was the frozen
  second patience miss, so checkpoint 3787 best `0.6698278113149699` was
  restored. Exact coverage is `192`, `64/language`, with no literal empty
  outputs or parser failures. Debug/retained-state/retained-adapter/final-
  adapter/config hashes are `c5df3f29...393a`/`b7f838c3...752a`/
  `4d52f4f3...e675`/`91c16983...05b7`/`8d05d65a...604c`; a local read-only
  comparison proved all `424/424` tensors (`76,410,112` values) exactly
  equal between retained and final serialization. NER confirmations are now
  terminal-valid `1/4`. POS a1 environment-only retry `1233034` is
  resource-pending, and the already-frozen NER b7 seed-87 confirmation
  `1233035` is priority-pending; neither has accessed model/data. POS a0
  `1232181` remains healthy at `600/1800` rows in step-1415 validation,
  retaining `0.7947789577505381`, with its next artifact expected around
  `16:50--17:10`. Owned state is one running plus two pending A100-40GB jobs
  and no A100-80GB/L40S work; quota is `70.3%/38.4%`. Kombuys remains
  read-only with assigned GPU 1 idle at `1 MiB/0%`; foreign GPU 0 is active
  at `21,800 MiB/90%` and untouched. Trusted progress is base `16/16`, NER
  seed-42 `11/11`, NER confirmations `1/4`, T2X seed-42 `11/11`, T2X
  confirmations `4/4`, POS `0/11`, global winners `0/8`, held-out `0`, Mono
  not started. Sheet E/F/G remain blank; publication stays blocked. Queue
  time and the remaining validation grids prevent a fixed full-results ETA.

- At 15:30 SAST POS a1 `1232226` is preserved as a zero-runtime
  infrastructure failure: its submission omitted `SALLM_RUNTIME_REPO`, so
  the immutable snapshot fell back to system Python 3.9 and failed importing
  `datetime.UTC` before manifest creation, model/data loading, or any metric.
  One environment-only retry is preregistered with identical candidate,
  seed, validation protocol, source snapshot, registry, and resources; it
  will explicitly use the already-validated Python 3.12 runtime and new
  `sourcefix2` paths. No held-out evidence informed the retry.

- At 14:58 SAST corrected POS a0 `1232181` improved at step 1132 to exact
  `all_token_accuracy=0.7947789577505381`, covering all `1,800` rows, 12
  language/template cells, and 17 labels under the frozen constrained
  protocol; checkpoint 1132 is retained. Artifact/trainer-state/adapter
  hashes are `d072e54d...26d82`/`78dfca25...afbc`/`e7c2c049...0d56`.
  It is healthy at `150/1800` rows in step-1415 validation, with the next
  artifact expected around `16:50--17:10`. NER b7 seed-13 `1232204`
  completed step-4869 full validation loss `0.36888760137735244/10,760` and
  is entering its exact decision callback; checkpoint 3787 mean F1
  `0.6698278113149699` remains retained at patience `1/2`, with the next
  artifact expected around `15:10--15:25`. POS a1 `1232226` remains
  resource-pending without model/data access. Owned state is two running plus
  one pending A100-40GB job and no A100-80GB/L40S work; HEX quota is
  `70.3%/38.4%`. Kombuys remains read-only with assigned GPU 1 idle at
  `1 MiB/0%`; foreign GPU 0 is active at `19,828 MiB/94%` and untouched.
  Trusted terminal progress is unchanged: base `16/16`, NER seed-42
  `11/11`, NER confirmations `0/4`, T2X seed-42 `11/11`, T2X confirmations
  `4/4`, POS `0/11`, global winners `0/8`, held-out `0`, Mono not started;
  there are now eight valid interim NER-confirmation and four valid interim
  POS artifacts. Sheet E/F/G remain blank and publication stays blocked.

- At 14:29 SAST NER b7 seed-13 confirmation `1232204` completed its exact
  step-4328 artifact at mean F1 `0.6624468056515672` (Tsn/Xho/Zul
  `0.6457120682964189/0.6680659363161167/0.6735624123421661`), below
  retained step-3787 best `0.6698278113149699`; frozen patience is now `1/2`.
  Coverage remains exact `192`, `64/language`, with no literal empty raw
  outputs or parser failures. Debug/trainer-state/current-adapter hashes are
  `5f0f168f...0ab8`/`30998847...0809`/`75b77836...4d96`. It resumed healthy
  near `4787/8115`; its next artifact is expected around `15:10--15:25` and
  will terminate the run on another frozen patience miss. Corrected POS a0
  `1232181` remains healthy at `1750/1800` rows in its step-1132 constrained
  callback, retaining checkpoint 849 accuracy `0.7794016731767367`; its next
  artifact is imminent. Corrected POS a1 `1232226` remains resource-pending
  without model/data access. Owned state is two running plus one pending
  A100-40GB job and no A100-80GB/L40S work; HEX quota is `70.3%/38.4%`.
  Kombuys remains read-only: assigned GPU 1 is idle at `1 MiB/0%`, while
  foreign GPU 0 is active at `18,112 MiB/96%` and untouched. No held-out
  metric was accessed; trusted terminal counts, blank Sheet E/F/G, and
  publication gates are unchanged.

- At 13:57 SAST NER b7 seed-13 confirmation `1232204` completed its
  step-4328 full validation loss at `0.3567892875813197/10,760` and remains
  healthy in generation with fresh probes through `13:49:39`; checkpoint
  3787 mean F1 `0.6698278113149699` remains retained and its next artifact is
  expected around `14:00--14:15`. Corrected POS a0 `1232181` is healthy at
  `1350/1800` rows in its step-1132 constrained callback, retaining
  checkpoint 849 accuracy `0.7794016731767367`; its next artifact is expected
  around `14:25--14:40`. Corrected POS a1 `1232226` remains resource-pending
  without model/data access. Owned state is two running plus one pending
  A100-40GB job and no A100-80GB/L40S work; HEX quota is `70.3%/38.4%`.
  Kombuys remains read-only: assigned GPU 1 is idle at `1 MiB/0%`, while
  foreign GPU 0 is active at `13,494 MiB/95%` and untouched. No new selection
  metric is accepted; trusted counts, blank Sheet E/F/G, and publication
  gates are unchanged.

- At 13:15 SAST NER b7 seed-13 confirmation `1232204` improved at step 3787
  to mean validation F1 `0.6698278113149699` (Tsn/Xho/Zul
  `0.6403700372606466/0.6753356591605381/0.693777737523725`) with exact
  `192`, `64/language` coverage, no literal empty raw outputs or parser
  failures, and debug/trainer-state/adapter hashes
  `4766e171...21e6`/`b7f838c3...752a`/`4d52f4f3...e675`; checkpoint 3787
  is retained. It resumed healthy near `4231/8115`, with the next artifact
  expected around `14:00--14:15`. Corrected POS a0 `1232181` remains healthy
  at `800/1800` rows in its step-1132 constrained callback, retaining
  checkpoint 849 accuracy `0.7794016731767367`; its next artifact is expected
  around `14:25--14:40`. Corrected POS a1 `1232226` remains resource-pending
  without model/data access. Owned state is two running plus one pending
  A100-40GB job and no A100-80GB/L40S work; HEX quota is `70.3%/38.4%`.
  Kombuys remains read-only: assigned GPU 1 is idle at `1 MiB/0%`, while
  foreign GPU 0 is active at `13,476 MiB/97%` and untouched. Operational base
  artifacts remain `16/16`; scientifically trusted progress remains base
  `16/16`, NER seed-42 `11/11`, NER confirmations terminal `0/4` with seven
  valid interim artifacts, T2X seed-42 `11/11`, T2X confirmations `4/4`, POS
  terminal `0/11` with three valid interim artifacts, global winners `0/8`,
  held-out `0`, and Mono not started. Sheet E/F/G remain blank and Hugging
  Face publication remains blocked.

- At 12:06 SAST corrected POS a0 `1232181` improved at step 849 to exact
  `all_token_accuracy=0.7794016731767367`, covering all `1,800` rows, three
  languages, four templates, and 17 labels. Artifact/trainer-state/adapter
  hashes are `6f2c308b...8078`/`292d9139...20c1`/
  `d5aeb30c...a8cf`; checkpoint 849 is retained. It resumed healthy near
  `1063/4245`, with its next artifact expected around `14:15--14:35`. NER b7
  seed-13 confirmation `1232204` improved at step 3246 to mean F1
  `0.6488434791145067` (Tsn/Xho/Zul
  `0.6291104415039458/0.6451244813277509/0.6722955145118235`) with exact
  `192`, `64/language` coverage, no literal empty raw outputs or parser
  failures, and debug/trainer-state/adapter hashes
  `6eeaeaaa...4394`/`e12e44d6...ae45`/`508c88d1...2a8`; checkpoint 3246 is
  retained. It resumed healthy near `3735/8115`, with its next artifact
  expected around `12:45--13:00`. Corrected POS a1 `1232226` remains
  resource-pending without model/data access. Owned state is two running plus
  one pending A100-40GB job and no A100-80GB/L40S work; HEX quota is
  `70.3%/38.4%`. Kombuys remains read-only: assigned GPU 1 is idle at
  `1 MiB/0%`, while foreign GPU 0 holds `27,404 MiB` at `0%` and is
  untouched. Operational base artifacts remain `16/16`; scientifically
  trusted progress remains base `16/16`, NER seed-42 `11/11`, NER
  confirmations terminal `0/4` with six valid interim artifacts, T2X
  seed-42 `11/11`, T2X confirmations `4/4`, POS terminal `0/11` with three
  valid interim artifacts, global winners `0/8`, held-out `0`, and Mono not
  started. Sheet E/F/G remain blank and Hugging Face publication remains
  blocked.

- At 11:36 SAST corrected POS a0 `1232181` is healthy at `1500/1800` rows
  in its step-849 constrained callback, retaining checkpoint 566 accuracy
  `0.7194463042526343`; its next artifact is expected around
  `11:55--12:05`. NER b7 seed-13 confirmation `1232204` remains healthy in
  step-3246 generation with fresh probes through `11:28:11`, retaining
  checkpoint 2705 mean F1 `0.6363575862728579`; its next artifact is expected
  around `11:45--12:00`. Corrected POS a1 `1232226` remains
  resource-pending without model/data access. Owned state is two running plus
  one pending A100-40GB job and no A100-80GB/L40S work; HEX quota is
  `70.3%/38.4%`. Kombuys remains read-only: assigned GPU 1 is idle at
  `1 MiB/0%`, while foreign GPU 0 is active at `15,978 MiB/91%` and
  untouched. No new selection metric is accepted; trusted counts, blank
  Sheet E/F/G, and publication gates are unchanged.

- At 11:06 SAST NER b7 seed-13 confirmation `1232204` completed its
  step-3246 full validation loss at `0.3455629483474675/10,760` and entered
  generation; checkpoint 2705 mean F1 `0.6363575862728579` remains retained.
  Corrected POS a0 `1232181` remains healthy at `1100/1800` rows in its
  step-849 callback, retaining checkpoint 566 accuracy
  `0.7194463042526343`; corrected POS a1 `1232226` remains resource-pending
  without model/data access. Owned state is two running plus one pending
  A100-40GB job, no A100-80GB/L40S work, and HEX quota `70.3%/38.4%`.
  Kombuys remains read-only: assigned GPU 1 is idle at `1 MiB/0%`, while
  foreign GPU 0 is active at `18,698 MiB/97%` and untouched. No new selection
  metric is accepted; trusted counts, blank Sheet E/F/G, and publication
  gates are unchanged.

- At 11:05 SAST NER b7 seed-13 confirmation `1232204` improved at step 2705
  to mean validation F1 `0.6363575862728579` (Tsn/Xho/Zul
  `0.6111820270614768/0.6341540737609139/0.6637366579961829`) with exact
  `192`, `64/language` coverage, no literal empty raw outputs or parser
  failures, and debug/trainer-state/adapter hashes
  `b5dce357...e6ca`/`680bb667...bf14`/`6f567a9a...7f48`; checkpoint 2705
  is retained. It reached step `3246/8115` and entered its next callback,
  with the next artifact expected around `11:45--12:00`. Corrected POS a0
  `1232181` remains healthy at `1100/1800` rows in its step-849 constrained
  callback, retaining checkpoint 566 accuracy `0.7194463042526343`; its next
  artifact is expected around `11:50--12:05`. Corrected POS a1 `1232226`
  remains resource-pending without model/data access. Owned state is two
  running plus one pending A100-40GB job and no A100-80GB/L40S work; HEX
  quota is `70.3%/38.4%`. Kombuys remains read-only: assigned GPU 1 is idle
  at `1 MiB/0%`, while foreign GPU 0 is active at `17,178 MiB/99%` and
  untouched. Operational base artifacts remain `16/16`; scientifically
  trusted progress remains base `16/16`, NER seed-42 `11/11`, NER
  confirmations terminal `0/4` with five valid interim artifacts, T2X
  seed-42 `11/11`, T2X confirmations `4/4`, POS terminal `0/11` with two
  valid interim artifacts, global winners `0/8`, held-out `0`, and Mono not
  started. Sheet E/F/G remain blank and Hugging Face publication remains
  blocked.

- At 09:35 SAST corrected POS a0 `1232181` improved at step 566 to exact
  `all_token_accuracy=0.7194463042526343`, covering all `1,800` rows, three
  languages, four templates, and 17 labels. Artifact/trainer-state/adapter
  hashes are `b053ef9c...bea59`/`c7e4c20c...3a21c`/
  `28e8ed0c...f2e5d`; checkpoint 566 is retained. It resumed through step
  `849/4245` and entered its next callback, with the next metric expected
  around `11:50--12:10`. NER b7 seed-13 confirmation `1232204` also improved
  at step 2164 to mean F1 `0.6007840028814361` (Tsn/Xho/Zul
  `0.5953342890655046/0.5901889092016562/0.6168288103771477`) with exact
  `192`, `64/language` coverage, no literal empty raw outputs or parser
  failures, and debug/trainer-state/adapter hashes
  `cb97b672...2a8e`/`b1a398df...6c0`/`242064cf...2f81`; checkpoint 2164 is
  retained. It resumed healthy near `2505/8115`, with the next artifact
  expected around `10:25--10:40`. Corrected POS a1 `1232226` remains
  resource-pending without model/data access. Owned state is two running plus
  one pending A100-40GB job and no A100-80GB/L40S work; HEX quota is
  `70.3%/38.4%`. Kombuys remains read-only: assigned GPU 1 is idle at
  `1 MiB/0%`, while foreign GPU 0 is active at `15,534 MiB/98%` and
  untouched. Operational base artifacts remain `16/16`; scientifically
  trusted progress remains base `16/16`, NER seed-42 `11/11`, NER
  confirmations terminal `0/4` with four valid interim artifacts, T2X
  seed-42 `11/11`, T2X confirmations `4/4`, POS terminal `0/11` with two
  valid interim artifacts, global winners `0/8`, held-out `0`, and Mono not
  started. Sheet E/F/G remain blank and Hugging Face publication remains
  blocked.

- At 09:05 SAST corrected POS a0 `1232181` is healthy at `1500/1800` rows
  in its step-566 constrained callback, with the next complete artifact
  expected around `09:25--09:30`; checkpoint 283 accuracy
  `0.49199491017457003` remains retained. NER b7 seed-13 confirmation
  `1232204` completed its step-2164 full validation loss at
  `0.33058257262502905/10,760` and remains healthy in generation with fresh
  probes through `09:05:22`; checkpoint 1623 mean F1
  `0.5765106149612969` remains retained and the next artifact is expected
  around `09:20--09:35`. Corrected POS a1 `1232226` remains
  resource-pending without model/data access. Owned state is two running plus
  one pending A100-40GB job and no A100-80GB/L40S work; HEX quota is
  `70.3%/38.4%`. Kombuys remains read-only: assigned GPU 1 is idle at
  `1 MiB/0%`, while foreign GPU 0 is active at `20,590 MiB/96%` and
  untouched. No new metric is accepted; operational and scientifically
  trusted progress, Sheet E/F/G, and publication gates are unchanged.

- At 08:35 SAST corrected POS a0 `1232181` is healthy at `1150/1800` rows
  in its step-566 constrained callback; checkpoint 283 accuracy
  `0.49199491017457003` remains retained and the next complete artifact is
  expected around `09:20--09:30`. NER b7 seed-13 confirmation `1232204`
  reached its step-2164 epoch-4 boundary after healthy training and entered
  the next validation pass; checkpoint 1623 mean F1
  `0.5765106149612969` remains retained, with the next artifact expected
  around `09:10--09:25`. Corrected POS a1 `1232226` remains resource-pending
  without model/data access. Owned state is two running plus one pending
  A100-40GB job and no A100-80GB/L40S work; HEX quota is `70.3%/38.4%`.
  Kombuys remains read-only: assigned GPU 1 is idle at `1 MiB/0%`, while
  foreign GPU 0 is active at `16,892 MiB/95%` and untouched. No new metric
  is accepted: operational base artifacts remain `16/16`; scientifically
  trusted progress remains base `16/16`, NER seed-42 `11/11`, NER
  confirmations terminal `0/4` with three valid interim artifacts, T2X
  seed-42 `11/11`, T2X confirmations `4/4`, POS terminal `0/11` with one
  valid interim artifact, global winners `0/8`, held-out `0`, and Mono not
  started. Sheet E/F/G remain blank and Hugging Face publication remains
  blocked.

- At 08:20 SAST NER b7 seed-13 confirmation `1232204` improved at step 1623
  to mean validation F1 `0.5765106149612969` (Tsn/Xho/Zul
  `0.582905544147794/0.5562117696687964/0.5904145310673001`) with exact
  `192`, `64/language` coverage, no literal empty raw outputs or parser
  failures, and debug/state hashes `ad3c06ee...2b3cc4`/
  `53266a58...2fb67`. It resumed healthy near `1883/8115`; its next artifact
  is expected `09:05--09:20`. Corrected POS a0 `1232181` is healthy at
  `950/1800` rows in its step-566 callback, retaining checkpoint 283
  accuracy `0.49199491017457003`; its next artifact is expected
  `09:15--09:30`. Corrected POS a1 `1232226` remains resource-pending without
  model/data access. Owned state is two running plus one pending A100-40GB
  job and no A100-80GB/L40S work; HEX quota is `70.3%/38.4%`. Kombuys GPU 1
  is idle at `1 MiB/0%`, while foreign GPU 0 is active at
  `16,734 MiB/97%` and untouched. Operational base artifacts remain `16/16`;
  scientifically trusted progress is base `16/16`, NER seed-42 `11/11`, NER
  confirmations terminal `0/4` with three valid interim artifacts, T2X
  seed-42 `11/11`, T2X confirmations `4/4`, POS terminal `0/11` with one
  valid interim artifact, global winners `0/8`, held-out `0`, and Mono not
  started. Sheet E/F/G remain blank and Hugging Face publication remains
  blocked.

- At 07:30 SAST corrected POS a0 `1232181` is healthy in its epoch-2
  constrained callback at `250/1800` rows after full validation loss
  `0.32744540744357636`; checkpoint 283 accuracy `0.49199491017457003`
  remains retained and the next metric ETA is `09:10--09:25`. NER b7
  seed-13 confirmation `1232204` is healthy in its step-1623 epoch-3 full
  validation pass, with checkpoint 1082 mean F1 `0.49773337486849173`
  retained and the next artifact expected `08:10--08:25`. Corrected POS a1
  `1232226` remains resource-pending without model/data access, with dynamic
  start projection `2026-08-15 00:45:59`. Owned state is two running plus
  one pending A100-40GB job and no A100-80GB/L40S work; HEX quota is
  `70.3%/38.4%`. Kombuys GPU 1 is idle at `1 MiB/0%`, while foreign GPU 0 is
  active at `24,304 MiB/99%` and untouched. No new metric is accepted;
  operational base artifacts remain `16/16`, while scientifically trusted
  progress remains base `16/16`, NER seed-42 `11/11`, NER confirmations
  terminal `0/4` with two valid interim artifacts, T2X seed-42 `11/11`, T2X
  confirmations `4/4`, POS terminal `0/11` with one valid interim artifact,
  global winners `0/8`, held-out `0`, and Mono not started. Sheet E/F/G
  remain blank and Hugging Face publication remains blocked.

- At 07:00 SAST corrected POS a0 `1232181` has one valid interim step-283
  artifact at `all_token_accuracy=0.49199491017457003`, with exact
  `1,800`-row coverage across three languages, four templates, and all 17
  labels. Artifact/state hashes are `2b083e77...a1669` and
  `b4950070...b7980`; checkpoint 283 is retained and the healthy job resumed
  near `434/4245`, with its next artifact expected `09:15--09:30`. NER b7
  seed-13 confirmation `1232204` improved at step 1082 to mean F1
  `0.49773337486849173` (Tsn/Xho/Zul
  `0.4795533845079752/0.4802905110256856/0.5333562290718142`) with exact
  `192`, `64/language` coverage, no literal empty raw outputs or parser
  failures, and debug/state hashes `dd6f8159...b48fb9`/
  `bc32d5a9...10246`. It resumed near `1189/8115`; next artifact ETA is
  `08:05--08:20`. Corrected POS a1 `1232226` remains resource-pending
  without model/data access. Owned state is two running plus one pending
  A100-40GB job and no A100-80GB/L40S work; HEX quota is `70.3%/38.4%`.
  Kombuys GPU 1 is idle at `1 MiB/0%`, while foreign GPU 0 is active at
  `20,590 MiB/98%` and untouched. Operational base artifacts remain `16/16`;
  scientifically trusted progress is base `16/16`, NER seed-42 `11/11`, NER
  confirmations terminal `0/4` with two valid interim artifacts, T2X
  seed-42 `11/11`, T2X confirmations `4/4`, POS terminal `0/11` with one
  valid interim artifact, global winners `0/8`, held-out `0`, and Mono not
  started. Sheet E/F/G remain blank and Hugging Face publication remains
  blocked.

- At 06:30 SAST corrected POS a0 `1232181` is healthy at `1450/1800` rows in
  its first constrained callback, with the complete validation metric
  expected `06:50--07:00`. NER b7 seed-13 confirmation `1232204` is healthy
  in its step-1082 generation callback after full epoch-2 validation loss
  `0.36973950127243554/10,760`; fresh probes continue through `06:24:26`,
  and the new 192-row artifact is expected `06:55--07:10`. Checkpoint 541
  mean F1 `0.30889695092008335` remains its only accepted interim metric.
  Corrected POS a1 `1232226` remains resource-pending without model/data
  access, with dynamic start projection `2026-08-15 00:45:59`. Owned state
  is two running plus one pending A100-40GB job and no A100-80GB/L40S work;
  HEX quota is `70.3%/38.3%`. Kombuys GPU 1 is idle at `1 MiB/0%`, while
  foreign GPU 0 is active at `15,542 MiB/94%` and untouched. Operational base
  artifacts remain `16/16`; scientifically trusted progress remains base
  `16/16`, NER seed-42 `11/11`, NER confirmations terminal `0/4` with one
  valid interim artifact, T2X seed-42 `11/11`, T2X confirmations `4/4`, POS
  `0/11`, global winners `0/8`, held-out `0`, and Mono not started. Sheet
  E/F/G remain blank and Hugging Face publication remains blocked.

- At 06:00 SAST NER b7 seed-13 confirmation `1232204` has one valid interim
  step-541 artifact at mean validation F1 `0.30889695092008335` (Tsn/Xho/Zul
  `0.2943698990177948/0.2845754318153887/0.3477455219270665`). Exact coverage
  is `192`, `64/language`, with no literal empty raw outputs, `55`
  whitespace-only raw outputs, and two parser failures preserved as model
  behavior. Debug/state hashes are `ff0cc3f1...9efb8` and
  `6c9fa18d...3faad`; checkpoint 541 is retained and the healthy job resumed
  near `936/8115`, with its next artifact expected `06:45--07:05`. Corrected
  POS a0 `1232181` is healthy at `1100/1800` rows in its first constrained
  callback, with first metric expected `06:45--07:00`; corrected a1
  `1232226` remains resource-pending without model/data access. Owned state
  is two running plus one pending A100-40GB job and no A100-80GB/L40S work;
  HEX quota is `70.3%/38.3%`. Kombuys GPU 1 is idle at `1 MiB/0%`, while
  foreign GPU 0 is active at `21,376 MiB/95%` and untouched. Operational base
  artifacts remain `16/16`; scientifically trusted progress is base
  `16/16`, NER seed-42 `11/11`, NER confirmations terminal `0/4` with one
  valid interim artifact, T2X seed-42 `11/11`, T2X confirmations `4/4`, POS
  `0/11`, global winners `0/8`, held-out `0`, and Mono not started. Sheet
  E/F/G remain blank and Hugging Face publication remains blocked.

- At 05:30 SAST corrected POS a0 `1232181` is healthy at `650/1800` rows in
  its epoch-1 constrained callback, with first complete validation evidence
  still projected around `06:45--07:00`. NER b7 seed-13 confirmation
  `1232204` completed full epoch-1 validation loss
  `1.0411281429702022/10,760` and remains healthy in generation with fresh
  probes through `05:26:50`; no 192-row metric artifact exists yet and the
  updated evidence window is `05:40--06:00`. Corrected POS a1 `1232226`
  remains resource-pending without model/data access; Slurm's dynamic start
  projection is `2026-08-15 00:45:59`. Owned state is two running plus one
  pending A100-40GB job and no A100-80GB/L40S work; HEX quota is
  `70.3%/38.3%`. Kombuys GPU 1 is idle at `1 MiB/0%`, while foreign GPU 0 is
  active at `17,478 MiB/98%` and untouched. No new selection metric is
  accepted: operational base artifacts remain `16/16`, while scientifically
  trusted progress remains base `16/16`, NER seed-42 `11/11`, NER
  confirmations `0/4`, T2X seed-42 `11/11`, T2X confirmations `4/4`, POS
  `0/11`, global winners `0/8`, held-out `0`, and Mono not started. Sheet
  E/F/G remain blank and Hugging Face publication remains blocked.

- At 05:00 SAST corrected POS a0 `1232181` and NER b7 seed-13 confirmation
  `1232204` are healthy in their first full validation callbacks on
  `srvrocgpu010` A100-40GB. POS reached step `283/4245`, completed full
  validation loss `0.7721000501844618`, and advanced to `250/1800`
  constrained rows; observed throughput puts its first complete metric near
  `06:45--07:00`. NER reached step `541/8115` and entered its full
  `10,760`-example callback; its first complete 192-row confirmation artifact
  remains due about `05:25--05:50`. No new selection metric is accepted yet.
  Because a0 passed the frozen source/count/startup gate, never-started
  old-source a1 `1232086` was cancelled after zero run time, with Slurm
  provenance preserved and no artifacts removed. Corrected immutable a1
  replacement `1232226` is resource-pending with the required
  `nlpgroup/a100/nlpgroup`, one-A100-40GB, 24-hour, eight-CPU envelope. Owned
  state is two running plus one pending A100-40GB job and no
  A100-80GB/L40S work; HEX quota is `70.3%/38.3%`. Kombuys GPU 1 is idle at
  `1 MiB/0%`, while foreign GPU 0 is active at `23,148 MiB/97%` and
  untouched. Operational base artifacts remain `16/16`; scientifically
  trusted progress remains base `16/16`, NER seed-42 `11/11`, NER
  confirmations `0/4`, T2X seed-42 `11/11`, T2X confirmations `4/4`, POS
  `0/11`, global winners `0/8`, held-out `0`, and Mono not started. Full
  downstream timing remains gated by the active callbacks and remaining
  validation grids. Sheet E/F/G remain blank and Hugging Face publication
  remains blocked.

- At 04:30 SAST corrected POS a0 `1232181` and frozen NER b7 seed-13
  confirmation `1232204` are both running healthy on `srvrocgpu010`
  A100-40GB. POS passed its implementation-correction startup gate: all
  `694` immutable files verified, exact `2,259/1,800` train/validation rows
  loaded and tokenized, and training reached step `188/4245`; the first
  complete validation artifact is expected `05:00--05:30`. NER verified the
  same immutable file count and reached step `67/8115`; its first complete
  confirmation artifact is expected `05:25--05:50`. Manifest/trial hashes
  are POS `da37437b...74eb6`/`17059aaa...9c22` and NER
  `12bd5577...d60d`/`c5037722...10e`. Old-source POS a1 `1232086` remains
  held. Owned state is exactly two running plus one held A100-40GB job, with
  no A100-80GB/L40S work; HEX quota is `70.3%/38.3%`. Kombuys GPU 1 is idle
  at `1 MiB/0%`; foreign GPU 0 is active at `19,790 MiB/91%` and untouched.
  Operational base artifacts remain `16/16`; scientifically trusted
  progress remains base `16/16`, NER seed-42 `11/11`, NER confirmations
  `0/4`, T2X seed-42 `11/11`, T2X confirmations `4/4`, POS `0/11`, global
  winners `0/8`, held-out `0`, and Mono not started. The full-result ETA
  remains gated by corrected POS and the remaining validation grids. Sheet
  E/F/G remain blank and Hugging Face publication remains blocked.

- At 03:30 SAST NER b7 job `1230086` is terminal-valid `0:0` after
  `16:35:49`, completing the seed-42 NER grid at `11/11`. Step-7574 mean
  validation F1 `0.6848459167143336` was the frozen second patience miss, so
  retained checkpoint 6492 best `0.6872585118558678` was restored. The exact
  192-row artifact has full `64/language` coverage, no literal empties or
  parse failures, and final/retained adapter equality passed all `424/424`
  tensors (`76,410,112` values). The hashed validation-only 11-candidate
  ranking `f51a8a3b...f9b1e2` freezes b7 and a2 for seeds 13/87 confirmation;
  no test metric was consulted. First confirmation b7 seed 13 is job
  `1232204`, priority-pending without scheduler ETA. Corrected POS a0
  `1232181` is resource-pending with current projection `06:12:27`; old a1
  `1232086` remains held. Owned state is exactly two schedulable plus one held
  A100-40GB job, with no A100-80GB/L40S work. HEX quota is `70.3%/38.3%`.
  Kombuys GPU 1 is idle; foreign GPU 0 is active at `16,908 MiB/91%` and
  untouched. Operational base artifacts remain `16/16`; scientifically
  trusted progress is base `16/16`, NER seed-42 `11/11`, NER confirmations
  `0/4`, T2X seed-42 `11/11`, T2X confirmations `4/4`, POS `0/11`, global
  winners `0/8`, held-out `0`, and Mono not started. Sheet E/F/G remain blank
  and Hugging Face publication remains blocked.

- At 03:00 SAST NER b7 job `1230086` remains healthy after `16:29:51` on
  A100-40GB. Its step-7574 full validation loss is complete at
  `0.39439457889826324` over all `10,760` examples, and generation remains
  active with fresh probes through `02:54:47`; no complete 192-row selection
  artifact exists yet. Checkpoint 6492 best F1 `0.6872585118558678` therefore
  remains unchanged. Updated conditional terminal evidence is due about
  `03:20--03:40` on a second patience miss or `04:35--04:50` after a
  qualifying improvement. Corrected POS a0 `1232181` remains
  priority-pending with dynamic start projection `14:39`; a1 `1232086`
  remains held and failed `1231553` preserved. HEX quota is `70.3%/38.3%`,
  with one running, one pending, and one held owned A100-40GB job and no
  A100-80GB/L40S work. Kombuys GPU 1 is idle; foreign GPU 0 is active at
  `15,504 MiB/92%` and untouched. Operational base artifacts remain `16/16`;
  scientifically trusted progress remains base `16/16`, NER `10/11`, T2X
  `11/11`, confirmations `4/4`, POS `0/11`, global winners `0/8`, held-out
  `0`, and Mono not started. Sheet E/F/G remain blank and Hugging Face
  publication remains blocked.

- At 02:30 SAST NER b7 job `1230086` entered its step-7574 epoch-14
  validation callback healthy on A100-40GB, with no complete new artifact;
  checkpoint 6492 best F1 `0.6872585118558678` remains unchanged. Conditional
  terminal evidence is due about `02:55--03:10` on a second patience miss or
  `04:15--04:20` after a qualifying improvement. Corrected POS a0 retry
  `1232181` remains priority-pending without model/data access; Slurm's
  dynamic start projection is `14:39 SAST`, subject to movement when NER
  releases its GPU. A1 `1232086` remains held and failed `1231553` preserved.
  HEX quota is `70.3%/38.3%`, with one running, one pending, and one held
  owned A100-40GB job and no A100-80GB/L40S work. Kombuys GPU 1 is idle and
  foreign GPU 0 is active and untouched. Trusted progress remains base
  `16/16`, NER `10/11`, T2X `11/11`, confirmations `4/4`, POS `0/11`, global
  winners `0/8`, held-out `0`, and Mono not started. Sheet E/F/G remain blank
  and Hugging Face publication remains blocked.

- At 02:00 SAST the POS source correction was frozen before rerun at
  preregistration SHA `0f2544b...3f48b`: upstream commit
  `376f4161...26d3`, commit-addressed raw API access, and transient-only
  retries, with no change to data content or scientific selection. Local
  gates passed `117` tests plus exact count/token audits. New read-only HEX
  snapshot `pure-gdn-hpo-possourcefix-20260814-d9501087` verified `694/694`
  files with source-set/deployment hashes `59459749...4f60` and
  `d996a19b...8c0a`. The single preregistered a0 infrastructure retry is job
  `1232181`, priority-pending with the required A100-40GB envelope and a new
  output path; failed job `1231553` is preserved. A1 `1232086` remains held
  until a0 passes startup. NER b7 `1230086` declined at step 7033 to mean F1
  `0.6827893429348121`, so checkpoint 6492 best `0.6872585118558678` remains
  retained and patience is `1/2`; it is healthy with conditional terminal
  ETA `03:05--04:20`. HEX quota after deployment is `70.3%/38.3%`, with one
  running, one
  pending, and one held owned A100-40GB job and no A100-80GB/L40S work.
  Kombuys GPU 1 is idle and foreign GPU 0 remains active and untouched.
  Trusted progress remains base `16/16`, NER `10/11`, T2X `11/11`,
  confirmations `4/4`, POS `0/11`, global winners `0/8`, held-out `0`, and
  Mono not started. Sheet E/F/G remain blank; Hugging Face publication is
  blocked.

- At 01:30 SAST POS a0 job `1231553` failed `1:0` after 53 seconds, after
  immutable-manifest and canonical-model verification but before dataset
  load, training, or any metric. The failure is a MasakhaPOS source-loader
  HTTP 404; direct checks also found intermittent GitHub raw `503` responses
  for Xho/Zul train files. Its provenance is preserved. Structurally
  identical POS a1 job `1232086` was put on reversible user hold before
  allocation to prevent a known duplicate failure; it remains blocked on a
  documented immutable source-availability correction. NER b7 job `1230086`
  remains healthy in its step-7033 generation callback; latest accepted
  checkpoint remains step 6492 at mean validation F1
  `0.6872585118558678`, with terminal ETA about 03:30--04:00. HEX quota is
  `52.0%/38.2%`, with one running and one held owned A100-40GB job and no
  A100-80GB/L40S work. Kombuys GPU 1 is idle and foreign GPU 0 remains
  active and untouched. Trusted progress remains base `16/16`, NER `10/11`,
  T2X `11/11`, confirmations `4/4`, POS `0/11`, global winners `0/8`,
  held-out `0`, and Mono not started. Sheet E/F/G remain blank and Hugging
  Face publication remains blocked.

- At 01:00 SAST NER b7 job `1230086` improved at step 6492 to exact mean
  validation F1 `0.6872585118558678`, retaining the checkpoint with exact
  192-row coverage, no parse failures, and debug/state hashes
  `b7631734...e545c`/`b2ac7c4a...d0d29`. It resumed near `6711/8115` with
  conditional terminal ETA about 03:30--04:00. POS a0/a1 jobs
  `1231553/1232086` remain resources/priority-pending. HEX quota is
  `52.0%/38.2%`, with one running plus two pending owned A100-40GB jobs and
  no A100-80GB/L40S work. Trusted progress remains base `16/16`, NER `10/11`,
  T2X `11/11`, confirmations `4/4`, POS `0/11`, global winners `0/8`,
  held-out `0`, and Mono not started. No held-out metric was accessed; Sheet
  E/F/G remain blank and Hugging Face publication remains blocked.

- At 00:30 SAST NER b7 job `1230086` remains healthy in its step-6492
  validation callback with a fresh log and no fault marker; no complete new
  artifact exists yet. POS a0/a1 jobs `1231553/1232086` remain
  resources/priority-pending. HEX quota is `52.0%/38.2%`, with one running
  plus two pending owned A100-40GB jobs and no A100-80GB/L40S work. Kombuys
  assigned GPU 1 remains idle and foreign GPU 0 remains untouched. Trusted
  progress remains base `16/16`, NER `10/11`, T2X `11/11`, confirmations
  `4/4`, POS `0/11`, global winners `0/8`, held-out `0`, and Mono not started.
  No held-out metric was accessed; Sheet E/F/G remain blank and Hugging Face
  publication remains blocked.

- At 00:00 SAST NER b7 job `1230086` improved at step 5951 to exact mean
  validation F1 `0.6854031406360509`, retaining the checkpoint with exact
  192-row coverage, no parse failures, and debug/state hashes
  `64808307...9adc`/`70ec9593...f0dc`. It resumed near `6340/8115` with
  conditional terminal ETA about 03:30--04:00. POS a0/a1 jobs
  `1231553/1232086` remain resources/priority-pending. HEX quota is
  `52.0%/38.2%`, with one running plus two pending owned A100-40GB jobs and
  no A100-80GB/L40S work. Kombuys assigned GPU 1 remains idle and foreign GPU
  0 remains untouched. Trusted progress remains base `16/16`, NER `10/11`,
  T2X `11/11`, confirmations `4/4`, POS `0/11`, global winners `0/8`,
  held-out `0`, and Mono not started. No held-out metric was accessed; Sheet
  E/F/G remain blank and Hugging Face publication remains blocked.

- At 23:30 SAST NER b7 job `1230086` remains healthy in its step-5951
  validation callback with a fresh log and no fault marker. POS a0/a1 jobs
  `1231553/1232086` remain resources/priority-pending. HEX quota is
  `52.0%/38.2%`, with one running plus two pending owned A100-40GB jobs and
  no A100-80GB/L40S work. Kombuys assigned GPU 1 is idle and foreign GPU 0
  remains untouched. Trusted progress remains base `16/16`, NER `10/11`,
  T2X `11/11`, confirmations `4/4`, POS `0/11`, global winners `0/8`,
  held-out `0`, and Mono not started. No held-out metric was accessed; Sheet
  E/F/G remain blank and Hugging Face publication remains blocked.

- At 23:00 SAST NER b6 job `1227987` completed `0:0`, terminal-valid with
  final/retained checkpoint 8115 mean validation F1 `0.4347279479413982`,
  raising trusted NER progress to `10/11`. NER b7 job `1230086` improved at
  step 5410 to `0.6834830425594811`, retained the checkpoint, and remains
  healthy. After the frozen count/hash/path gates, pure-GDN POS a1 job
  `1232086` was submitted with the required immutable A100-40GB launcher and
  is priority-pending; POS a0 job `1231553` remains resources-pending. HEX
  quota is `52.0%/38.2%`, with one running plus two pending owned A100-40GB
  jobs and no A100-80GB/L40S work. Kombuys assigned GPU 1 is idle and foreign
  GPU 0 remains untouched. Trusted progress is base `16/16`, NER `10/11`,
  T2X `11/11`, confirmations `4/4`, POS `0/11`, global winners `0/8`,
  held-out `0`, and Mono not started. No held-out metric was accessed; Sheet
  E/F/G remain blank and Hugging Face publication remains blocked.

- At 22:30 SAST NER b6 job `1227987` completed all `8115/8115` training
  steps and entered its final generation callback; b7 job `1230086` remains
  healthy inside its step-5410 callback. Conditional complete evidence is
  expected around 22:35--22:50. POS a0 job `1231553` remains
  priority-pending. HEX quota is `52.0%/38.2%`, with two running plus one
  pending owned A100-40GB job and no A100-80GB/L40S work. Kombuys assigned
  GPU 1 is idle and foreign GPU 0 remains untouched. Trusted progress remains
  base `16/16`, NER `9/11`, T2X `11/11`, confirmations `4/4`, POS `0/11`,
  global winners `0/8`, held-out `0`, and Mono not started. No held-out metric
  was accessed; Sheet E/F/G remain blank and Hugging Face publication remains
  blocked.

- At 22:00 SAST T2X Stage C completed: a2 seed 87 finished terminal-valid at
  validation chrF `50.74381008283938`, and all `424/424` final-adapter
  tensors exactly match retained checkpoint 1932. The preregistered
  three-seed rule selects b7 over a2 by mean chrF
  `51.987594971037375` versus `50.75818404404604`; the hashed validation-only
  ranking artifact is `86f7655d...20e36`. This is family-local only; the
  eight-family global freeze remains closed. HEX NER b6 job `1227987`
  declined slightly at step 7574 to mean F1 `0.4345354420684952`, retaining
  checkpoint 7033 best `0.43460119481609377`; b7 job `1230086` remains
  healthy in its step-5410 callback. POS a0 job `1231553` remains
  priority-pending. HEX quota is `52.0%/38.2%`, with two running plus one
  pending owned A100-40GB job and no A100-80GB/L40S work. Kombuys GPU 1 is
  idle and foreign GPU 0 remains untouched. Trusted progress is base `16/16`,
  NER `9/11`, T2X `11/11`, confirmations `4/4`, POS `0/11`, global winners
  `0/8`, held-out `0`, and Mono not started. No held-out metric was accessed;
  Sheet E/F/G remain blank and Hugging Face publication remains blocked.

- At 21:30 SAST T2X a2 seed 87 improved through validation chrF
  `44.40289293605772/47.67471900310504/50.21260712607607` at steps
  `483/966/1449`, retaining checkpoint 1449 with a clean exact 64-row
  artifact and debug/state hashes `3a7ae03c...134c`/`27aa6296...1381`. It
  resumed on assigned Kombuys GPU 1 with terminal ETA about 21:50; foreign
  GPU 0 remains untouched. HEX NER b7 job `1230086` improved at step 4869 to
  exact mean validation F1 `0.6738201916539253`, retaining checkpoint 4869;
  b6 job `1227987` remains healthy inside its step-7574 callback. POS a0 job
  `1231553` remains priority-pending. Quota is `52.0%/38.2%`, with two
  running plus one pending owned A100-40GB job and no A100-80GB/L40S work.
  Trusted progress remains base `16/16`, NER `9/11`, T2X `11/11`,
  confirmations `3/4`, POS `0/11`, winners `0/8`, held-out `0`, and Mono not
  started. No held-out metric was accessed; Sheet E/F/G remain blank and
  Hugging Face publication remains blocked.

- At 21:00 SAST the final T2X confirmation, a2 seed 87, produced a clean
  step-483 validation artifact at chrF `44.40289293605772`: exact `64/64`
  rows, no empty raw predictions, `60` unique outputs, and debug/state hashes
  `ac96581d...03df3`/`c4fa6deb...53f0`. It resumed near `741/1932` on
  assigned Kombuys GPU 1 with terminal ETA about 21:50; foreign GPU 0 remains
  untouched. HEX NER b6/b7 jobs `1227987/1230086` remain healthy in their
  step `7574/4869` callbacks, while POS a0 job `1231553` remains
  priority-pending. Quota is `52.0%/38.2%`, with two running plus one pending
  owned A100-40GB job and no A100-80GB/L40S work. Trusted progress remains
  base `16/16`, NER `9/11`, T2X `11/11`, confirmations `3/4`, POS `0/11`,
  winners `0/8`, held-out `0`, and Mono not started. Sheet E/F/G remain blank
  and Hugging Face publication remains blocked; no held-out metric was
  accessed.

- At 20:38 SAST T2X a2 seed 13 completed terminal-valid at validation chrF
  `50.22442959480154`; all `424/424` final-adapter tensors exactly match
  retained checkpoint 1932. The last confirmation, a2 seed 87, passed the
  frozen gates and is running on assigned Kombuys GPU 1 with terminal ETA
  about 21:50; foreign GPU 0 remains untouched. HEX NER b5 job `1226719`
  completed `0:0`, retaining checkpoint 7574 best mean validation F1
  `0.5743710116912446`; b6/b7 jobs `1227987/1230086` remain healthy on
  A100-40GB. Pure-GDN POS Stage-A a0 job `1231553` was submitted with the
  immutable launcher and is pending for priority. HEX quota is `52.0%/38.2%`
  with two running plus one pending owned job and no A100-80GB/L40S work.
  Trusted progress is base `16/16`, NER `9/11`, T2X `11/11`, confirmations
  `3/4`, POS `0/11` terminal, winners `0/8`, held-out `0`, and Mono not
  started. No held-out metric was accessed; Sheet E/F/G remain blank and
  Hugging Face publication remains blocked.

- At 20:00 SAST Kombuys T2X a2 seed 13 improved from validation chrF
  `44.201928271802885` at step 483 to `47.23730282734459` at step 966,
  retaining checkpoint 966 with exact clean debug/state hashes
  `e7b28077...8caa`/`ef49422e...4ccb`. It reached its step-1449 callback on
  assigned GPU 1 with terminal ETA about 20:25; foreign GPU 0 remains
  occupied and untouched. HEX NER jobs `1226719/1227987/1230086` remain
  healthy A100-40GB runs on `srvrocgpu010`, respectively inside the final
  step-8115, step-7033, and step-4328 callbacks, with no new complete artifact
  or fault marker. Quota is `52.0%/38.2%`; no A100-80GB/L40S work exists.
  Trusted progress remains base `16/16`, NER `8/11`, T2X `11/11`,
  confirmations `2/4`, winners `0/8`, held-out `0`, Sheet E/F/G blank, and
  Hugging Face blocked. No held-out metric was accessed.

- At 19:30 SAST Kombuys T2X a2 seed 13 produced a valid first artifact at
  step 483 with validation chrF `44.201928271802885`, no empty predictions,
  and debug/state hashes `f5f03ae2...cea32`/`d65ca8da...83a4`; it resumed
  near `688/1932` on assigned GPU 1 with terminal ETA about 20:25. Foreign
  GPU 0 remains occupied and untouched. HEX NER b7 job `1230086` improved at
  step 3787 to exact mean validation F1 `0.6708899559501509`, retaining the
  checkpoint, while b6 job `1227987` recorded its first post-improvement miss
  at step 6492 with `0.42658069985026664` and retained checkpoint 5951 best
  `0.4270927910241913`. B5 job `1226719` completed all 8115 training steps
  and entered its final callback. All three HEX jobs remain healthy
  A100-40GB runs on `srvrocgpu010`; quota is `52.0%/38.2%`, with no
  A100-80GB/L40S work. Trusted progress remains base `16/16`, NER `8/11`,
  T2X `11/11`, confirmations `2/4`, winners `0/8`, held-out `0`, Sheet E/F/G
  blank, and Hugging Face blocked. No held-out metric was accessed.

- At 19:03 SAST Kombuys T2X b7 seed 87 completed terminal-valid with retained
  checkpoint 1932 and validation chrF `51.72122400804512`; all `424/424`
  final-adapter tensors exactly match the retained checkpoint. B7's three
  validation-only seed scores are now
  `52.26645731208185/51.975103592985164/51.72122400804512`, but no winner is
  frozen before a2 completes. After the frozen idle/hash/path gates, a2 seed
  13 started in tmux `sallm-t2x-confirm-a2-s13`, verified 694 immutable files
  and exact `3,859/460` rows, and entered its 1,932-step run on assigned GPU
  1; manifest/trial hashes are `bac8a685...ba46`/`696b37f9...7ca`, with
  first-artifact/terminal ETAs about 19:30/20:25. Foreign GPU 0 remains
  occupied and untouched. HEX NER b5 job `1226719` improved at step 7574 to
  exact mean validation F1 `0.5743710116912446`; jobs
  `1226719/1227987/1230086` remain exactly three healthy A100-40GB runs on
  `srvrocgpu010`, quota is `52.0%/38.1%`, and no A100-80GB/L40S work exists.
  Trusted progress remains base `16/16`, NER `8/11`, T2X `11/11`, winners
  `0/8`, held-out `0`, Sheet E/F/G blank, and Hugging Face blocked. No
  held-out metric was accessed.

- At 18:30 SAST Kombuys T2X b7 seed-87 confirmation improved through
  validation chrF `44.99226017209242/49.92559710705884/51.497458946416664`
  at steps `483/966/1449`, retaining checkpoint 1449; exact step-1449
  debug/state hashes are `047d12b3...cebb20`/`41b89b1c...c7439f`. Epoch-4
  training resumed on assigned GPU 1 with terminal ETA about 18:50; foreign
  GPU 0 remains occupied and untouched. HEX NER jobs
  `1226719/1227987/1230086` remain exactly three healthy A100-40GB jobs;
  quota is `52.0%/38.2%`, with no A100-80GB/L40S work. Trusted progress
  remains base `16/16`, NER `8/11`, T2X `11/11`, winners `0/8`, held-out
  `0`, Sheet E/F/G blank, and Hugging Face blocked. No held-out metric was
  accessed.

- At 18:00 SAST HEX NER b5/b6/b7 jobs `1226719/1227987/1230086`
  produced exact step `7033/5951/3246` mean validation F1
  `0.5722423856201379/0.4270927910241913/0.642549691285009`. B5 records its
  first miss and retains checkpoint 6492 best `0.5733473836832604`; b6/b7
  improve and retain their new checkpoints. All three remain healthy
  A100-40GB jobs on `srvrocgpu010`; quota is `52.0%/38.2%`, with no
  A100-80GB/L40S work. Kombuys b7 seed-87 confirmation produced a valid
  initial chrF `44.99226017209242` at step 483 and is healthy near
  `732/1932` on assigned GPU 1, terminal ETA about 18:50; foreign GPU 0
  remains occupied and untouched. Trusted progress remains base `16/16`, NER
  `8/11`, T2X `11/11`, winners `0/8`, held-out `0`, Sheet E/F/G blank, and
  Hugging Face blocked. No held-out metric was accessed.

- At 17:31 SAST Kombuys T2X b7 seed-13 confirmation is terminal-valid with
  retained checkpoint 1932 and best validation chrF `52.26645731208185`; all
  `424/424` final-adapter tensors exactly match the retained checkpoint.
  After the frozen idle/hash/path gates, b7 seed 87 started in tmux
  `sallm-t2x-confirm-b7-s87`, verified 694 immutable files and exact
  `3,859/460` rows, and entered its 1,932-step run on assigned GPU 1;
  manifest/trial hashes are `421e9e2a...c1334`/`2d0fd4f1...6063e`, with
  first-artifact/terminal ETAs about 17:55/18:50. Foreign GPU 0 remains
  occupied and untouched. HEX NER jobs `1226719/1227987/1230086` remain
  exactly three healthy A100-40GB jobs; quota is `52.0%/38.1%`, with no
  A100-80GB/L40S work. Trusted progress remains base `16/16`, NER `8/11`,
  T2X `11/11`, winners `0/8`, held-out `0`, Sheet E/F/G blank, and Hugging
  Face blocked. No held-out metric was accessed.

- At 17:00 SAST HEX NER b6/b7 jobs `1227987/1230086` improved at steps
  `5410/2705` to exact mean validation F1
  `0.4199830275844996/0.6384214393951778`, retaining both checkpoints and
  resetting patience. Together with b5 `1226719`, all three remain healthy
  A100-40GB jobs on `srvrocgpu010`; quota is `52.0%/38.1%`, with no
  A100-80GB/L40S work. Kombuys b7 seed-13 confirmation improved from chrF
  `45.60933714692265` at step 483 to `49.040358075702585` at step 966 and is
  healthy near `1393/1932` on assigned GPU 1, terminal ETA about 17:25;
  foreign GPU 0 remains occupied and untouched. Trusted progress remains
  base `16/16`, NER `8/11`, T2X `11/11`, winners `0/8`, held-out `0`, Sheet
  E/F/G blank, and Hugging Face blocked. No held-out metric was accessed.

- At 16:30 SAST HEX NER b5 job `1226719` improved at step 6492 to exact mean
  validation F1 `0.5733473836832604`, retaining checkpoint 6492; its exact
  192-row debug/state/adapter hashes are
  `054b855c...04613d`/`9a8cea1f...0cc34`/`3b6c5ebd...a8391c`. B6/b7 jobs
  `1227987/1230086` remain in incomplete step-5410/2705 callbacks. All three
  are healthy A100-40GB jobs on `srvrocgpu010`; quota is `52.0%/38.2%`, with
  no A100-80GB/L40S work. Kombuys b7 seed-13 confirmation is healthy near
  `613/1932` on assigned GPU 1 with first-artifact/terminal ETAs about
  16:35/17:25; foreign GPU 0 remains occupied and untouched. Trusted progress
  remains base `16/16`, NER `8/11`, T2X `11/11`, winners `0/8`, held-out
  `0`, Sheet E/F/G blank, and Hugging Face blocked. No held-out metric was
  accessed.

- At 16:06 SAST Kombuys T2X b7 is terminal-valid, completing the seed-42
  grid `11/11` with retained checkpoint 1932 and best validation chrF
  `51.975103592985164`; all `424/424` final-adapter tensors exactly match the
  retained checkpoint. The hashed validation-only ranking
  `cd532dec...a2cb7` freezes b7 (`51.975103592985164`) and a2
  (`51.306312454497196`) for confirmation. B7 seed 13 then started in tmux
  `sallm-t2x-confirm-b7-s13`, verified 694 immutable files and exact
  `3,859/460` rows, and entered its 1,932-step run on assigned Kombuys GPU 1;
  manifest/trial hashes are `d1286b12...4aaa`/`d103dfb8...6ba4`, with
  first-artifact/terminal ETAs about 16:35/17:25. Foreign GPU 0 remains
  occupied and untouched. HEX NER jobs `1226719/1227987/1230086` remain
  exactly three healthy A100-40GB runs; quota is `52.0%/38.2%`, with no
  A100-80GB/L40S work. Trusted progress is base `16/16`, NER `8/11`, T2X
  `11/11`, winners `0/8`, held-out `0`, Sheet E/F/G blank, and Hugging Face
  blocked. No held-out metric was accessed.

- At 15:30 SAST HEX NER b5/b6/b7 jobs `1226719/1227987/1230086`
  produced exact step `5951/4869/2164` mean validation F1
  `0.5719458254448195/0.40108590595274/0.6165239138215678`; b5 and b7
  improve and retain their new checkpoints, while b6 records its first miss
  and retains checkpoint 4328 best `0.4045046237234104`. All three remain
  healthy A100-40GB jobs on `srvrocgpu010`; quota is `52.0%/38.2%` and no
  A100-80GB/L40S work exists. Kombuys T2X b7 improved at step 966 to
  validation chrF `49.68565734381204`, retaining checkpoint 966 with exact
  64-row coverage and no empty raw predictions; its step-1449 corrected
  callback is active on assigned GPU 1. Foreign GPU 0 remains occupied and
  untouched; root/scratch free are `23 GB/2.1 TB`, with b7 terminal ETA about
  16:00--16:10. Trusted terminal progress remains base `16/16`, NER `8/11`,
  T2X `10/11`, winners `0/8`, held-out `0`, Sheet E/F/G blank, and Hugging
  Face blocked. No ranking or held-out metric access occurred.

- At 15:00 SAST Kombuys T2X b7 produced a valid initial step-483 validation
  chrF `44.69243255853447`, exact 64-row coverage, no empty raw predictions,
  and debug/state hashes `48d2f3d5...c6b95f`/`47f0421f...316cb9`. It is
  healthy near `734/1932` on assigned GPU 1, ETA about 16:00--16:10; foreign
  GPU 0 remains occupied and untouched, with root/scratch free
  `23 GB/2.1 TB`. HEX NER jobs `1226719/1227987/1230086` remain exactly
  three healthy A100-40GB runs in step `5951/4869/2164` corrected callbacks;
  no new complete artifact exists. Quota `52.0%/38.2%`, no A100-80GB/L40S.
  Trusted terminal progress remains base `16/16`, NER `8/11`, T2X `10/11`,
  winners `0/8`, held-out `0`, Sheet E/F/G blank, and Hugging Face blocked.
  No held-out metric was accessed.

- At 14:32 SAST Kombuys T2X b6 is terminal-valid `10/11`, retaining
  checkpoint 1449 best validation chrF `38.35361029340392` after final
  `38.13150671845184`; final debug/state/adapter hashes are
  `32b12f27...40940`/`9e977c48...31f47`/`3b84725d...aa55`, and all
  `424/424` final-adapter tensors exactly match the retained checkpoint.
  After the idle-GPU gate, final seed-42 candidate b7 started in tmux
  `sallm-t2x-hpo-b7`, verified 694 immutable files and exact `3,859/460`
  rows, and entered training on assigned GPU 1; manifest/trial hashes are
  `c7710142...d66b`/`ce1e3298...18d0`, first-artifact/terminal ETAs about
  14:55/15:55. Foreign GPU 0 remains occupied and untouched. HEX NER b5/b6/b7
  jobs `1226719/1227987/1230086` produced valid step `5410/4328/1623` mean
  F1 `0.5525407691385177/0.4045046237234104/0.5725969497508221`; b5 records
  its first patience miss while b6/b7 improve. All three remain healthy on
  A100-40GB; quota `52.0%/38.2%`, no A100-80GB/L40S. Trusted progress is
  base `16/16`, NER `8/11`, T2X `10/11`, winners `0/8`, held-out `0`, Sheet
  E/F/G blank, and Hugging Face blocked. No held-out metric was accessed.

- At 14:00 SAST Kombuys T2X b6 improved at step 966 to validation chrF
  `36.85528231891048`, retaining checkpoint 966; its exact 64-row artifact
  has no empty raw predictions and debug/state hashes
  `d78c9297...a313fc`/`c7f22c6b...d6b815`. It remains healthy on isolated
  GPU 1 with terminal ETA about 15:00; foreign GPU 0 is occupied and
  untouched, with Kombuys root/scratch free `23 GB/2.1 TB`. HEX NER jobs
  `1226719/1227987/1230086` remain exactly three healthy A100-40GB runs in
  step `5410/4328/1623` corrected callbacks, with no complete new artifact;
  quota `52.0%/38.2%` and no A100-80GB/L40S. The HPO design remains
  appropriately rigorous: validation candidates separate while incomplete
  evidence is excluded, and multi-seed confirmation plus the one-time test
  gate still protect against validation overfitting. Trusted terminal
  progress remains base `16/16`, NER `8/11`, T2X `9/11`, winners `0/8`,
  held-out `0`, Sheet E/F/G blank, and Hugging Face blocked.

- At 13:32 SAST HEX NER b7 job `1230086` improved at step 1082 to exact mean
  validation F1 `0.4907328629996628`, retaining checkpoint 1082; its exact
  192-row debug/state hashes are `1f347f3f...8013f`/
  `a9ca20de...5850e`. B5/b6 jobs `1226719/1227987` are in their step
  `5410/4328` corrected callbacks with no complete new artifact. All three
  jobs remain healthy on `srvrocgpu010` A100-40GB; quota `52.0%/38.2%`, no
  A100-80GB/L40S. Kombuys T2X b6 produced a valid initial step-483 chrF
  `31.376884549616697`, exact 64-row coverage, no empty raw predictions, and
  debug/state hashes `36b64e6e...fd91f`/`31aeaeff...340e5`; it is healthy
  near `775/1932` on assigned GPU 1 with terminal ETA about 14:25--14:35.
  Foreign GPU 0 remains occupied and untouched. Operational artifacts are
  healthy; scientifically terminal-valid progress remains base `16/16`, NER
  `8/11`, T2X `9/11`, winners `0/8`, held-out `0`, Sheet E/F/G blank, and
  Hugging Face blocked. No held-out metric was accessed.

- At 13:00 SAST HEX NER b5 job `1226719` improved at step 4869 to mean
  validation F1 `0.5584894571005082`, retaining the new checkpoint; b6 job
  `1227987` declined at step 3787 to `0.3539458751417704`, its first patience
  miss, and correctly retains checkpoint 3246 best `0.3750359191540918`.
  Debug/state hashes are `ef6b7676...2db17e`/`65394c17...d47a4f` and
  `2e696a6a...e414e`/`e3c7b35c...7258c`. B7 job `1230086` remains in its
  step-1082 callback. All three A100-40GB jobs are healthy; quota
  `52.0%/38.2%`, no A100-80GB/L40S. Kombuys T2X b6 attempt 0 failed before
  model/data/metric access due to a wrong environment-derived model path and
  is hash-preserved. The environment-only retry `sallm-t2x-hpo-b6-r1`
  verified 694 immutable files, loaded the canonical BF16 model, verified
  exact `3,859/460` rows, and entered training on GPU 1; retry manifest/trial
  hashes are `75340d5f...cf104`/`7e9be9d1...3148e`,
  first-artifact/terminal ETAs about 13:25/14:25, and foreign GPU 0 remains
  untouched. Base `16/16`, NER terminal-valid `8/11`, T2X terminal-valid
  `9/11`, winners `0/8`, held-out `0`, Sheet E/F/G blank, Hugging Face
  blocked. No held-out metric was accessed.

- At 12:32 SAST Kombuys T2X b5 is terminal-valid `9/11`, completing all
  `1932/1932` steps with final-best validation chrF `44.54370114772246` at
  checkpoint 1932. Final debug/state/adapter hashes are
  `5d546045...9aed7b`/`266e5ff9...0aaf2`/`dd285ba3...22895`. After the
  idle-GPU gate, preregistered b6 started in tmux `sallm-t2x-hpo-b6`, verified
  all 694 immutable files, and reached the kernel gate; manifest/trial hashes
  are `8882d9f0...f20b1d`/`60519a3e...deec59`, with first-artifact/terminal
  ETAs about 12:55/13:55. Foreign GPU 0 remains untouched. HEX NER jobs
  `1226719/1227987/1230086` remain exactly three healthy A100-40GB runs in
  corrected callbacks; no new complete artifact exists. Quota `52.0%/38.1%`,
  no A100-80GB/L40S. Base `16/16`, NER terminal-valid `8/11`, winners `0/8`,
  held-out `0`, Sheet E/F/G blank, Hugging Face blocked. These remain
  validation-only results.

- At 12:00 SAST HEX NER b5/b6/b7 jobs `1226719/1227987/1230086` produced
  valid step `4328/3246/541` mean F1
  `0.5504264830571982/0.3750359191540918/0.23712363123925098`; b5/b6 improve
  and b7 initializes, so all retain the new checkpoints and reset patience.
  Exact 192-row debug/state hashes are
  `1a0c248a...6d0787`/`ca384ab3...708d9`,
  `76ebca0c...f8389`/`4c527dc9...6088ba`, and
  `8e41259c...b78347`/`182d6451...4f3c7`. The three A100-40GB jobs remain
  healthy; next callbacks are expected about 12:15--13:00. Kombuys T2X b5
  improved at epoch 2 to validation chrF `43.72808904321918`, retaining
  checkpoint 966 with exact 64-row coverage and debug/state hashes
  `b33e0687...181f2e`/`16674327...2a911`; its epoch-3 callback is active,
  terminal ETA about 12:25, and foreign GPU 0 remains untouched. This is not
  test evidence. Quota `52.0%/38.1%`, no A100-80GB/L40S. Base `16/16`, NER
  terminal-valid `8/11`, T2X terminal-valid `8/11`, winners `0/8`, held-out
  `0`, Sheet E/F/G blank, Hugging Face blocked.

- At 11:30 SAST Kombuys T2X b5 produced a valid initial step-483 chrF
  `35.59502883566279`, exact 64-row coverage, no empty raw predictions, and
  debug/state hashes `b06d679e...2841d`/`3800f5d2...5bfa8`. It is healthy
  near `645/1932`, ETA about 12:23; foreign GPU 0 remains untouched. HEX NER
  jobs `1226719/1227987/1230086` remain exactly three healthy A100-40GB runs
  in epoch-8/6/1 corrected callbacks at steps `4328/3246/541`; their new
  artifacts are incomplete, so no decision was made. Quota `52.0%/38.1%`,
  no A100-80GB/L40S. Base `16/16`, NER terminal-valid `8/11`, T2X
  terminal-valid `8/11`, winners `0/8`, held-out `0`, Sheet E/F/G blank,
  Hugging Face blocked.

- At 11:01 SAST Kombuys T2X b4 is terminal-valid `8/11`, with final-best
  checkpoint 1932 chrF `45.69104321766449` and exact 64-row coverage; final
  debug/state/adapter hashes are `15f4d344...bc317`/
  `cb2174e4...732211`/`5eab5498...555551`. After the idle-GPU gate,
  preregistered b5 initially failed before model/data/metric access because
  its shell lacked W&B authentication; that attempt is hash-preserved. The
  environment-only offline retry in tmux `sallm-t2x-hpo-b5-r1` verified `694`
  immutable files and exact `3,859/460` rows, then entered training;
  manifest/trial hashes are `181bfc56...e3303`/`1eeb76c5...fc46e`, with
  first-artifact/terminal ETAs about 11:25/12:23. Foreign GPU 0 remains
  untouched. HEX NER jobs
  `1226719/1227987/1230086` are exactly three healthy A100-40GB runs; b5/b6
  are in corrected callbacks at steps `4328/3246` and b7 is near step 528.
  Quota `52.0%/38.1%`, no A100-80GB/L40S. Base `16/16`, NER terminal-valid
  `8/11`, winners `0/8`, held-out `0`, Sheet E/F/G blank, Hugging Face
  blocked.

- At 10:33 SAST NER b4 job `1224858` is terminal-valid `8/11`, completing
  `0:0` after the second frozen patience miss and retaining checkpoint 5951
  mean F1 `0.5859339434694067`; final debug/state/adapter hashes are
  `2ecdf302...aa5af`/`a5d56635...c6a60`/`4cef1566...d4d4`. The released
  slot started b7 job `1230086` with exact A100-40GB settings and verified
  `694` immutable files, exact `4,323/10,760` rows, and entered training;
  manifest/trial hashes are
  `9c35e22e...c3448`/`43c483cd...66a6c`, first-artifact ETA about 11:45.
  B5/b6 jobs `1226719/1227987` improved at steps `3787/2705` to mean F1
  `0.5352507187972697/0.3471549886942178` and continue. These are exactly
  three healthy HEX A100-40GB jobs; quota `52.0%/38.1%`, no A100-80GB/L40S.
  Kombuys T2X b4 improved at epochs 2/3 to chrF
  `43.83300504241281/45.204596233784045`, retaining checkpoint 1449 and
  remaining healthy for a roughly 10:50 terminal ETA; foreign GPU 0 remains
  untouched. Base `16/16`, T2X terminal-valid `7/11`, winners `0/8`,
  held-out `0`, Sheet E/F/G blank, Hugging Face blocked.

- At 10:01 SAST Kombuys T2X b4 produced a valid initial step-483 chrF
  `37.277711104727736`, exact 64-row Xhosa coverage, no empty predictions,
  and debug/state hashes `4227b2c5...8668`/`6ed5208c...52b7e`. It is healthy
  near `718/1932`, ETA about 10:50; foreign GPU 0 remains untouched. HEX NER
  jobs `1224858/1226719/1227987` remain exactly three healthy A100-40GB runs
  in epoch-11/7/5 corrected callbacks at steps `5951/3787/2705`; their new
  artifacts are incomplete, so no decision was made and b7 stays
  unsubmitted. Quota `52.0%/38.0%`, no A100-80GB/L40S. Base `16/16`, NER
  terminal-valid `7/11`, T2X terminal-valid `7/11`, winners `0/8`, held-out
  `0`, Sheet E/F/G blank, Hugging Face blocked.

- At 09:33 SAST Kombuys T2X b3 is terminal-valid `7/11`, retaining epoch-3
  checkpoint 1449 best chrF `46.10443672013474` after final
  `46.08058576186493`; final debug/state/adapter hashes are
  `4690290f...32871`/`51a6bb55...d6477`/`b8d96f0c...730c7`. After the
  idle-GPU gate, preregistered b4 started in tmux `sallm-t2x-hpo-b4`, verified
  all 694 immutable files and exact `3,859/460` rows, and entered training;
  manifest/trial hashes are `1f7dfc9a...6bba0`/`82f5e201...fc27`, ETA about
  10:50. Foreign GPU 0 remains untouched. HEX NER jobs `1226719/1227987`
  improved at steps `3246/2164` to exact mean F1
  `0.5095075141003677/0.2797155538913259`, resetting patience. Job `1224858`
  is in its step-5951 corrected callback; all three A100-40GB jobs are
  healthy, b7 stays unsubmitted, quota `52.0%/38.0%`, no A100-80GB/L40S.
  Base `16/16`, NER terminal-valid `7/11`, winners `0/8`, held-out `0`, Sheet
  E/F/G blank, Hugging Face blocked.

- At 09:00 SAST Kombuys T2X b3 improved epoch 2 step 966 to chrF
  `45.124063374893765`, retaining checkpoint 966; its exact 64-row artifact
  has no empty predictions and debug/state hashes `ed8c3f78...78ad0`/
  `8fbb32ef...14d7a`. Epoch-3 corrected validation is active, ETA about 09:25,
  and foreign GPU 0 remains untouched. HEX NER b4 `1224858` recorded its
  first patience miss at step 5410, exact mean F1 `0.57713788562457324`, and
  correctly retains checkpoint 4869 best `0.5858902577772452`; its exact
  192-row debug/state hashes are `b28e56ad...ab5aa`/
  `b41cb64e...e80af`. Jobs `1226719/1227987` remain in corrected callbacks;
  all three A100-40GB jobs are healthy, b7 stays unsubmitted, quota
  `52.0%/38.0%`, and no A100-80GB/L40S is active. Base `16/16`, NER
  terminal-valid `7/11`, T2X terminal-valid `6/11`, winners `0/8`, held-out
  `0`, Sheet E/F/G blank, Hugging Face blocked.

- At 08:30 SAST Kombuys T2X b3 produced a valid initial step-483 chrF
  `38.65717411733255`, exact 64-row coverage, no empty predictions, and
  debug/state hashes `453cb3cf...0cabf`/`fdfed85b...b2558`. It is healthy
  near `663/1932`, ETA about 09:25; foreign GPU 0 remains untouched. HEX NER
  jobs `1224858/1226719/1227987` remain exactly three healthy A100-40GB runs
  in epoch-10/6/4 corrected callbacks at steps `5410/3246/2164`; their new
  artifacts are incomplete, so no decision was made and b7 stays
  unsubmitted. Quota `52.0%/38.0%`, no A100-80GB/L40S. Base `16/16`, NER
  terminal-valid `7/11`, T2X terminal-valid `6/11`, winners `0/8`, held-out
  `0`, Sheet E/F/G blank, Hugging Face blocked.

- At 08:05 SAST Kombuys T2X b2 is terminal-valid `6/11`, with final-best
  chrF `40.32726992706406` at checkpoint 1932 and exact 64-row coverage. Its
  final debug/state/adapter hashes are `3654f374...2f5e8`/
  `33452a28...91ba1`/`86baa82d...7e02`. After the idle-GPU gate,
  preregistered b3 started in tmux `sallm-t2x-hpo-b3`, verified all 694
  immutable files and exact `3,859/460` rows, and entered training;
  manifest/trial hashes are `99a4d9d8...327e`/`8156ee05...5918`, ETA about
  09:25. Foreign GPU 0 remains untouched. HEX NER jobs
  `1224858/1226719/1227987` improved at steps `4869/2705/1623` to exact mean
  F1 `0.5858902577772452/0.4874091148048638/0.2044077624651216`, resetting
  patience; all three exact 192-row artifacts have no literal empties or parse
  failures. They remain healthy near `5290/2899/1937`; b7 stays unsubmitted,
  quota `52.0%/38.0%`, no A100-80GB/L40S. Base `16/16`, NER terminal-valid
  `7/11`, winners `0/8`, held-out `0`, Sheet E/F/G blank, Hugging Face
  blocked.

- At 07:30 SAST Kombuys T2X b2 improved epoch 2 step 966 to chrF
  `37.9190530917995`, retaining checkpoint 966; its exact 64-row artifact has
  no empty predictions and debug/state hashes `5b2c4078...1102`/
  `7f2a2918...f2be`. Epoch-3 corrected generation is active, ETA about 08:00,
  and foreign GPU 0 remains untouched. HEX NER jobs
  `1224858/1226719/1227987` remain exactly three healthy A100-40GB runs in
  epoch-9/5/3 corrected callbacks; their artifacts are incomplete, so no
  decision was made and b7 stays unsubmitted. Quota `52.0%/38.0%`, no
  A100-80GB/L40S. Base `16/16`, NER terminal-valid `7/11`, T2X terminal-valid
  `5/11`, winners `0/8`, held-out `0`, Sheet E/F/G blank, Hugging Face
  blocked.

- At 07:00 SAST HEX NER b4/b5/b6 jobs `1224858/1226719/1227987` improved at
  steps `4328/2164/1082` to exact mean F1
  `0.5711115998283094/0.4162427285350683/0.19493532967233085`, resetting
  patience and retaining those checkpoints. Exact 192-row debug/state hashes
  are `bf22345c...b6de8`/`40f4b816...4362`,
  `07b8942a...f2db`/`f8625b53...9932`, and
  `d363a3df...d389a`/`3db07dcf...55b3`; b6 retains one explicit parse
  failure without dropping coverage. The three A100-40GB jobs remain healthy;
  b7 stays unsubmitted, quota `52.0%/38.0%`, no A100-80GB/L40S. Kombuys T2X
  b2 produced a valid initial step-483 chrF `31.31387618368498`, exact 64-row
  coverage, no empty predictions, and hashes `94457434...2e18`/
  `43701ea5...d3c3`; it is healthy near `719/1932`, ETA about 07:55, with
  foreign GPU 0 untouched. Base `16/16`, NER terminal-valid `7/11`, T2X
  terminal-valid `5/11`, winners `0/8`, held-out `0`, Sheet E/F/G blank,
  Hugging Face blocked.

- Deferred future work: after the pure-GDN program is complete and frozen,
  apply the same scientifically clean validation-only HPO design to every
  other architecture in scope. This does not expand the current run plan.
  Preregister each architecture independently, preserve immutable provenance,
  confirm winners across seeds, and keep held-out tests out of selection.

- At 06:31 SAST Kombuys T2X b1 is terminal-valid `5/11`, with final-best chrF
  `43.70046004673441` at checkpoint 1932; final debug/state/adapter hashes are
  `c7bc6c5d...20dd`/`61d1efbd...69ff`/`865465c0...f4e3`. No ranking
  occurred. After the idle-GPU gate, preregistered b2 started in tmux
  `sallm-t2x-hpo-b2`, verified all 694 immutable files and exact `3,859/460`
  rows, and entered training; manifest/trial hashes are
  `5929088d...39c6`/`36161943...daba`, ETA about 07:55. Foreign GPU 0 remains
  untouched. HEX NER jobs `1224858/1226719/1227987` remain exactly three
  healthy A100-40GB runs in corrected callbacks; b7 stays unsubmitted, quota
  `52.0%/38.0%`, no A100-80GB/L40S. Base `16/16`, NER terminal-valid `7/11`,
  winners `0/8`, held-out `0`, Sheet E/F/G blank, Hugging Face blocked.

- At 05:59 SAST Kombuys T2X b1 improved epoch 2 step 966 to chrF
  `42.91101817994199`, retaining checkpoint 966; its exact 64-row artifact has
  no empty predictions and debug/state hashes `cefdf05c...d431`/
  `077383bf...a562`. Epoch-3 corrected generation is active, ETA about 06:25,
  and foreign GPU 0 remains untouched. HEX NER jobs
  `1224858/1226719/1227987` remain exactly three healthy A100-40GB runs in
  epoch-8/4/2 corrected callbacks; their new artifacts are incomplete, so no
  decision was made and b7 stays unsubmitted. Quota `52.0%/38.0%`, no
  A100-80GB/L40S. Base `16/16`, NER terminal-valid `7/11`, T2X terminal-valid
  `4/11`, winners `0/8`, held-out `0`, Sheet E/F/G blank, Hugging Face
  blocked.

- At 05:31 SAST HEX NER b4/b5/b6 jobs `1224858/1226719/1227987` produced
  valid step `3787/1623/541` artifacts with exact mean F1
  `0.5692895173118145/0.3072057532693912/0.1216232572664236`. B4 and b5
  improved and b6 initialized, so all three reset patience and retain those
  checkpoints. Exact 192-row debug/state hashes are
  `d9bc5d6c...790c5`/`e9fc1493...29e4`,
  `b9640ba0...f9c7`/`c1efb382...182c`, and
  `02c9efe8...b082`/`1a277f7d...64fc`. The three A100-40GB jobs remain
  healthy on `srvrocgpu010`; b7 stays unsubmitted, quota `52.0%/38.0%`, no
  A100-80GB/L40S. Kombuys T2X b1 produced a valid initial step-483 chrF
  `36.10345122148267`, exact 64-row coverage, no empty predictions, and
  debug/state hashes `97d6084b...12a1`/`e6f2b620...aab6`; it is healthy near
  `732/1932`, ETA about 06:30, while foreign GPU 0 remains untouched. Base
  `16/16`, NER terminal-valid `7/11`, T2X terminal-valid `4/11`, winners
  `0/8`, held-out `0`, Sheet E/F/G blank, Hugging Face blocked.

- At 05:01 SAST Kombuys T2X b0 is terminal-valid `4/11`, retaining epoch-3
  checkpoint 1449 best chrF `47.56264117884068` after final
  `47.55999905312901`; its epoch-3/final debug, retained-state, and adapter
  hashes are `ff52d061...08a44`/`342791db...96f5`/
  `5d0d9194...ca4ac`/`705bd44d...f1b46`. No ranking occurred. After the
  idle-GPU gate, preregistered b1 started in tmux `sallm-t2x-hpo-b1`, verified
  all 694 immutable files and exact `3,859/460` rows, and entered training;
  manifest/trial hashes are `b154bcd5...b35e`/`bfca2071...aa7`, ETA about
  06:30. GPU 1 is healthy and foreign GPU 0 remains untouched. HEX NER jobs
  `1224858/1226719/1227987` remain exactly three healthy A100-40GB runs on
  `srvrocgpu010`; b7 stays unsubmitted, quota `52.0%/38.0%`, and no
  A100-80GB/L40S is owned. Base `16/16`, NER terminal-valid `7/11`, winners
  `0/8`, held-out `0`, Sheet E/F/G blank, Hugging Face blocked.

- At 04:29 SAST HEX NER b4 `1224858` improved epoch 6 step 3246 to F1
  `0.5391488200979206`, and b5 `1226719` improved epoch 2 step 1082 to
  `0.2718237001045021`; both reset patience. Their exact 192-row debug/state
  hashes are `17f0d581...41771`/`d0d23830...6aa6c` and
  `9950be4d...a29e4`/`44fbbf16...3ad75`. B6 `1227987` is running healthily
  near step 531 with verified manifest/trial hashes `83791021...bdbe`/
  `6fb77d13...f074`. These are exactly three A100-40GB jobs, so b7 remains
  unsubmitted; quota `52.0%/38.0%`, no A100-80GB/L40S. Kombuys T2X b0
  improved epoch 2 step 966 to chrF `46.74057155779831`, retaining checkpoint
  966; its exact 64-row artifact has no empty predictions and debug/state
  hashes `c5b60c9d...e6ad5`/`37d52f7e...f26d1`. Epoch-3 generation is active,
  ETA about 05:00, with foreign GPU 0 untouched. Base `16/16`, NER
  terminal-valid `7/11`, T2X terminal-valid `3/11`, winners `0/8`, held-out
  `0`, Sheet E/F/G blank.

- At 04:01 SAST NER b3 job `1222348` is terminal-valid `COMPLETED 0:0`,
  stopping after its second patience miss and retaining checkpoint 4328 best
  F1 `0.6027308342440768`; final exact 192-row debug/state/adapter hashes are
  `33cb6d3f...b7af5`/`2a520f87...c5160`/`ad6ab2aa...859e5`. NER seed-42
  progress is now `7/11`. The open slot was filled with preregistered b6 job
  `1227987`, currently `PENDING (Priority)` with the exact A100-40GB Slurm
  contract; b7 remains unsubmitted. b4 `1224858` and b5 `1226719` are healthy
  in corrected callbacks with no complete new artifacts. Kombuys T2X b0 has
  a valid epoch-1 step-483 chrF `40.382906319702634`, exact 64-row coverage,
  no empty predictions, and debug/state hashes `8217da07...a1ae`/
  `9b789220...675e`; it is healthy near `692/1932`, ETA about 04:50, with
  foreign GPU 0 untouched. HEX quota `52.0%/38.0%`, no A100-80GB/L40S. Base
  `16/16`, T2X terminal-valid `3/11`, winners `0/8`, held-out `0`, Sheet E/F/G
  blank.

- At 03:32 SAST Kombuys T2X a2 is terminal-valid `3/11`, retaining epoch-3
  checkpoint 1449 best chrF `51.306312454497196` after final
  `51.14499797288408`; final debug/state/adapter hashes are
  `bef02bd5...f6c0`/`d1f2ff9f...1867c`/`9f4991a1...04d4e`. After the idle-GPU
  gate, preregistered Stage-B b0 started in tmux `sallm-t2x-hpo-b0-r1` with
  exact `3,859/460` rows and manifest/trial hashes `758fc88a...4af01`/
  `a28ed57c...1afc`. Its first invocation failed before process/artifact/data/
  metric access because the log directory was absent; the active retry changes
  only that directory precreation. GPU 1 is healthy, ETA about 04:50, and
  foreign GPU 0 remains untouched. HEX NER jobs `1222348/1224858/1226719`
  remain three healthy A100-40GB runs in corrected validation/training, so b6
  remains unsubmitted; quota `52.0%/37.9%`, no A100-80GB/L40S. Base `16/16`,
  NER terminal-valid `6/11`, winners `0/8`, held-out `0`, Sheet E/F/G blank.

- At 03:00 SAST Kombuys T2X a2 improved through epoch-3 step 1449 to chrF
  `51.306312454497196`, retaining checkpoint 1449; its exact 64-row artifact
  has no empty predictions and debug/state hashes `89ac3c5d...90548`/
  `d1f2ff9f...1867c`. Epoch 4 is healthy near `1492/1932`, ETA about 03:20,
  with assigned GPU 1 healthy and foreign GPU 0 untouched. HEX NER b4
  `1224858` improved epoch 5 step 2705 to F1 `0.5347543586486823`, resetting
  patience; b5 `1226719` produced its valid initial step-541 F1
  `0.12884601234004364`. Their exact 192-row debug/state hashes are
  `f9167b25...e9be`/`e6b7c56d...af7d6` and
  `686f4335...d59c`/`a226fc99...ffa1`. b3 `1222348` is in its step-5410
  corrected callback with no complete artifact. Exactly three A100-40GB jobs
  are healthy, so b6 remains unsubmitted; quota `52.0%/37.9%`, no
  A100-80GB/L40S. Base `16/16`, NER terminal-valid `6/11`, T2X terminal-valid
  `2/11`, winners `0/8`, held-out `0`, Sheet E/F/G blank.

- At 02:29 SAST Kombuys T2X a2 has a valid epoch-1 step-483 chrF
  `44.05334578338157`, exact 64-row coverage, no empty predictions, and
  debug/state hashes `aa96f2a5...a75c7`/`179b9c20...f8eba`; checkpoint 483
  is retained and the run is healthy near `730/1932`, ETA about 03:25.
  Assigned GPU 1 is healthy and foreign GPU 0 remains untouched. HEX NER b3
  `1222348` narrowly missed at epoch 9 step 4869 with F1
  `0.6025279148686585`, advancing patience to `1/2` while retaining checkpoint
  4328 best `0.6027308342440768`; exact 192-row debug/state hashes are
  `37c2651c...be97`/`b1291b4f...8944`. b4 `1224858` and b5 `1226719` are in
  corrected callbacks with no complete new artifacts, so no decisions were
  made from health-only losses. Exactly three A100-40GB jobs are healthy; b6
  remains unsubmitted, quota `52.0%/37.9%`, no A100-80GB/L40S. Base `16/16`,
  NER terminal-valid `6/11`, T2X terminal-valid `2/11`, winners `0/8`,
  held-out `0`, Sheet E/F/G blank.

- At 02:01 SAST T2X a1 is terminal-valid `2/11`, retaining epoch-3 checkpoint
  1449 chrF `46.33302341833507` after final `46.01656370856435`; final debug/
  state/adapter hashes are `65242c33...2fe5f`/`b5165888...8c65c`/
  `9024c825...80ef5`. After an idle-GPU gate, preregistered a2 started in tmux
  `sallm-t2x-hpo-a2` with exact `3,859/460` rows and manifest/trial hashes
  `0084c85f...d85f8d`/`39fc8c5e...1e213`; it is healthy, ETA about 03:25,
  and foreign GPU 0 remains untouched. HEX NER b4 `1224858` improved epoch 4
  step 2164 to F1 `0.4720515545248202`, resetting patience; exact 192-row
  debug/state hashes are `ca54665c...3d19b`/`fb183bd1...ee2346`. b5 job
  `1226719` is now running with verified manifest/trial hashes
  `70d8469a...faa534`/`ed5188e3...238381`. Exactly three A100-40GB jobs
  `1222348/1224858/1226719` are healthy, so b6 remains unsubmitted; quota
  `52.0%/37.9%`, no A100-80GB/L40S. Base `16/16`, NER terminal-valid `6/11`,
  winners `0/8`, held-out `0`, Sheet E/F/G blank.

- At 01:29 SAST Kombuys T2X a1 improved epoch 2 step 966 to chrF
  `45.13571777938961`, retaining checkpoint 966; its exact 64-row artifact has
  no empty predictions and debug/state hashes `f3e3c2bb...e56ad`/
  `0f7599a9...aa831`. It is healthy near `1362/1932`, ETA about 02:10, with
  assigned GPU 1 healthy and foreign GPU 0 untouched. HEX NER b3 `1222348`
  improved epoch 8 step 4328 to F1 `0.6027308342440768`, resetting patience;
  its exact 192-row debug/state hashes are `e1795ae6...6a47a`/
  `2a520f87...c5160`. b4 `1224858` is in its epoch-4 corrected callback with
  no complete step-2164 artifact. b5 `1226719` remains `PENDING (Resources)`;
  owned HEX state is two running plus one pending A100-40GB job, so b6 is not
  submitted. Quota `52.0%/37.9%`, no A100-80GB/L40S. Base `16/16`, NER
  terminal-valid `6/11`, T2X terminal-valid `1/11`, winners `0/8`, held-out
  `0`, Sheet E/F/G blank.

- At 01:01 SAST NER b2 job `1220056` is terminal-valid `COMPLETED 0:0`, taking
  NER seed-42 progress to `6/11`. Final step-8115 F1 `0.4670792452595483` is
  the trainer best and exported checkpoint; debug/state/adapter hashes are
  `8d7e836b...36b6df`/`47b7a3ee...df26c6`/`d5decfc9...fdce2`, with exact
  192-row coverage and no literal empties or parse failures. A one-second
  mistaken login-node b5 preflight was stopped and left no persisted output,
  model, data, or metric artifact. Correct Slurm b5 job `1226719` was then
  submitted with the frozen A100-40GB settings and is `PENDING (Resources)`;
  b6 remains unsubmitted. Kombuys T2X a1 has a valid epoch-1 step-483 chrF
  `39.13618399069993`, exact 64-row coverage, no empty predictions, and
  debug/state hashes `fe0c54ec...2f99b4`/`18bb9d8f...03bffb`; it is healthy
  near `630/1932`, ETA about 02:10, with foreign GPU 0 untouched. HEX quota
  `52.0%/37.9%`, no A100-80GB/L40S. Base `16/16`, T2X terminal-valid `1/11`,
  winners `0/8`, held-out `0`, Sheet E/F/G blank.

- At 00:34 SAST T2X `a0` is terminal-valid `1/11`: final epoch-4 chrF
  `41.01639999061889` correctly retained epoch-3 checkpoint 1449 best
  `41.19802690700617`; final debug/adapter hashes are
  `7fe475c1...f6c71`/`9bc4b3f5...f84ca`. T2X `a1` initially failed after
  manifest-only startup, then a durable retry proved the generic wrapper used
  a nonexistent Kombuys model path and failed at tokenizer opening before
  data/metrics. Both attempts are preserved. Host-path-only retry tmux
  `sallm-t2x-hpo-a1-r2` is healthy with exact `3,859/460` rows and active
  manifest/trial hashes `889e4a62...a28f9`/`901eb352...8afd0`; ETA about
  02:10. Foreign GPU 0 remains untouched. HEX NER b3 `1222348` improved epoch
  7 step 3787 to F1 `0.583489364755344`, and b4 `1224858` improved epoch 3
  step 1623 to `0.3707816090929401`, both resetting patience; hashes are
  `3f093f94...91bda5`/`90dd1249...a33239` and
  `b0ad1b16...5e5006`/`874c77ef...5fddf1`. b2 `1220056` remains in its final
  callback, so b5 stays unsubmitted. Three A100-40GB jobs healthy; quota
  `52.0%/37.8%`, no A100-80GB/L40S. Base `16/16`, NER terminal-valid `5/11`,
  winners `0/8`, held-out `0`, Sheet E/F/G blank.

- At 23:59 SAST Kombuys T2X `a0` improved epoch 3 step 1449 to chrF
  `41.19802690700617`, retaining checkpoint 1449; its exact 64-row artifact
  has no empty predictions and debug/state hashes `c3941759...74ec8`/
  `06d9e738...572dd`. It then completed `1932/1932` training steps and entered
  its final callback after health-only loss `2.4602493949558424`; step 1932
  remained absent at 00:01 SAST, so `a1` was not launched. GPU 1 remained
  healthy and foreign GPU 0 untouched. HEX NER `b2` job `1220056` recorded
  epoch-14 step-7574 F1 `0.4658432729648961`, a small miss that correctly
  advanced patience to `1/2` and retained checkpoint 7033; its exact 192-row
  debug/state hashes are `02756eab...6b8d15`/`dd84afcf...e5331`. It completed
  `8115/8115` training steps and entered its final callback. `b3` `1222348`
  and `b4` `1224858` remain in epoch-7/3 corrected callbacks with no complete
  new artifacts. Three healthy A100-40GB jobs keep b5 unsubmitted; quota
  `52.0%/37.8%`, no A100-80GB/L40S. Base `16/16`, NER terminal-valid `5/11`,
  T2X terminal-valid `0/11`, winners `0/8`, held-out `0`, Sheet E/F/G blank.

- At 23:29 SAST Kombuys T2X Stage-A `a0` has valid step-483/966 validation
  artifacts: Xhosa chrF improved `33.96025389749359` ->
  `40.5024801694161`, retaining checkpoint 966. Both contain exact 64-row
  coverage, no empty predictions, and debug hashes `085700d9...0ac92`/
  `1041cb5f...61025`; trainer-state hash is `7fbf1ea1...c8d6`. The run is
  healthy in epoch 3 near `1214/1932` on assigned Kombuys GPU 1, with a
  tentative 45--60 minute ETA; foreign GPU 0 remains untouched. This is only
  within-run validation evidence and no T2X candidate ranking occurred. HEX
  NER `b4` job `1224858` improved epoch 2 step 1082 to exact mean F1
  `0.2959148954716149` (Tsn/Xho/Zul
  `0.2919144497/0.2855311355/0.3102991012`), resetting patience and retaining
  checkpoint 1082; its exact 192-row artifact has no literal empties or parse
  failures and hashes `839a6ba7...2c347`/`ccaa0731...152d8`. `b2` `1220056`
  and `b3` `1222348` remain in epoch-14/7 corrected callbacks without complete
  new artifacts. Three healthy A100-40GB jobs keep b5 unsubmitted; quota
  `52.0%/37.8%`, no A100-80GB/L40S. Base `16/16`, NER terminal-valid `5/11`,
  T2X terminal-valid `0/11`, winners `0/8`, held-out `0`, Sheet E/F/G blank.

- At 22:59 SAST Stage-B `b3` job `1222348` improved epoch 6 step 3246 to
  exact mean F1 `0.5652522468935781` (Tsn/Xho/Zul
  `0.5474323299/0.5605612999/0.5877631110`), a `0.0091709042897022` gain
  resetting patience and retaining checkpoint 3246. Its exact 192-row,
  64/language artifact has no literal empty raw predictions or parse failures;
  debug/state hashes are `25efeaef...0e935`/`b4314c65...169d6`. `b2`
  `1220056` is in epoch-14 corrected generation after health-only loss
  `0.4200252717312384`; `b4` `1224858` is in epoch-2 corrected generation,
  with neither complete next selection artifact available. All three remain
  healthy A100-40GB jobs, so b5 is unsubmitted; quota `52.0%/37.8%`, no
  A100-80GB/L40S. Kombuys T2X `a0` reached epoch-1 step 483 and entered its
  corrected generation callback after health-only loss `2.658723781419837`;
  no complete chrF artifact exists and no T2X decision was made. GPU 1 is
  healthy near `8,250 MiB`, `96%`, `62 C`; foreign GPU 0 remains untouched.
  Sheet E/F/G remain blank, base `16/16`, NER terminal-valid `5/11`, T2X
  terminal-valid `0/11`, winners `0/8`, held-out `0`.

- At 22:28 SAST the cross-host assignment was frozen before T2X HPO metrics at
  SHA-256 `ca7078aa...62d51`: NER stays wholly on HEX A100-40GB and T2X stays
  wholly on Kombuys GPU 1 RTX 3080 Ti for all Stage-A/B and confirmation
  seeds. The read-only enhanced snapshot verified `694/694` files, registry
  `8fdd6ea5...bb726`, six canonical model hashes, and exact T2X train/validation
  hashes. Kombuys environment manifest `e0023257...0af5` and direct-validation
  batch-4 canary `1d5cb4bd...00c36` passed; the canary recorded no selection
  score. The first T2X `a0` launch failed before model/data/metric access for a
  missing W&B key and is preserved (`3fb1c271...8086f`/`6517f1de...54a65`).
  Its unchanged W&B-offline retry is active in tmux `sallm-t2x-hpo-a0-r1`
  with new output suffix `seed_42-wandb-retry1`, verified `3,859/460` rows and
  frozen batch `4/4`, accumulation 2, BF16 semantics, and began `1,932` steps;
  manifest/trial hashes are `58e90c6d...8dbe9`/`11e8aef8...a76`. Kombuys GPU
  1 used about `1,466 MiB` at 60%; foreign GPU 0 work remains untouched. On
  HEX, b2 job `1220056` improved epoch-13 step-7033 to F1
  `0.4662058582807271`, resetting patience; jobs
  `1220056/1222348/1224858` remain healthy A100-40GB, so b5 is unsubmitted.
  Quota `52.0%/37.8%`, no A100-80GB/L40S, Sheet E/F/G blank, base `16/16`,
  NER terminal-valid `5/11`, T2X terminal-valid `0/11`, winners `0/8`,
  held-out `0`.

- At 21:59 SAST Stage-B `b3` job `1222348` improved epoch 5 step 2705 to
  mean F1 `0.5560813426038759` (Tsn/Xho/Zul
  `0.5512972475/0.5505144746/0.5664323058`), a
  `0.05113046799774067` gain resetting patience; epoch 6 resumed near step
  3136. Fixed b4 retry `1224858` produced its valid initial epoch-1 step-541
  mean F1 `0.17146060376217687` (`0.1542506573/0.1629525711/
  0.1971785829`) and resumed epoch 2 near step 772 without low-fidelity
  pruning. Both artifacts have exact 192 rows and 64/language, finite
  metrics, no literal empty raw strings or parse failures, and intact frozen
  prompt contracts. b3 debug/state hashes are `344e0f85...c833ad`/
  `11446d4a...d71364`; b4 values are `6ef46c8c...b589c9`/
  `b5c7c3ec...b9269a`. `b2` `1220056` is in epoch-13 corrected generation
  after loss `0.4188957015821039`, with step 7033 tentatively due
  `22:10--22:25`. Three A100-40GB jobs remain healthy on `srvrocgpu010`, no
  A100-80GB/L40S; quota `52.0%/37.8%`, Kombuys untouched, Sheet E/F/G blank,
  base `16/16`, NER seed-42 terminal-valid `5/11`, frozen winners `0/8`,
  held-out `0`.

- At 21:29 SAST Stage-B `b2` job `1220056` produced valid epoch-12 step
  6492 mean F1 `0.4554093541784874` (Tsn/Xho/Zul
  `0.4552082046/0.4398419131/0.4711779449`), `0.0031122173985638133`
  below retained epoch-11 checkpoint 5951 best `0.4585215715770512`.
  Patience correctly advanced to `1/2` and epoch 13 resumed near step 6980.
  Coverage is exact 192 rows and 64/language, with finite metrics, no literal
  empty raw strings or parse failures, and intact frozen prompt contract;
  debug/state hashes are `4c6df0b8...4499be`/`c669358c...6bd84`. `b3`
  `1222348` is in epoch-5 corrected generation after loss
  `0.351184297582917`, with step 2705 tentatively due `21:35--21:45`; b4
  retry `1224858` is in epoch-1 corrected generation after loss
  `1.2492069286927858`, with step 541 tentatively due `21:45--22:00`.
  Three A100-40GB jobs remain healthy on `srvrocgpu010`, no A100-80GB/L40S;
  quota `52.0%/37.8%`, Kombuys untouched, Sheet E/F/G blank, base `16/16`,
  NER seed-42 terminal-valid `5/11`, frozen winners `0/8`, held-out `0`.

- At 20:59 SAST `b2` `1220056` remains in epoch-12 corrected generation
  after loss `0.42394250352143353`; all three segments started by 20:49:03,
  but step 6492 is absent, tentatively due `21:00--21:10`. `b3` `1222348`
  is in its epoch-5 validation callback with no complete loss or step-2705
  artifact, tentatively due `21:20--21:40`. Fixed b4 retry `1224858` remains
  healthy near its first boundary at step 528, with step 541 tentatively due
  `21:35--21:55`. Targeted fault scans are empty. All three run on
  A100-40GB `srvrocgpu010`, with no A100-80GB/L40S; quota `52.0%/37.8%`,
  Kombuys untouched, Sheet E/F/G blank, base `16/16`, NER seed-42
  terminal-valid `5/11`, frozen winners `0/8`, held-out `0`.

- At 20:31 SAST Stage-B `b1` job `1219931` is terminal-valid `COMPLETED
  0:0`. Epoch-13 mean F1 `0.5523719658486961` was the second frozen-threshold
  miss, so checkpoint 6492 remains terminal best at `0.5576102928857579`.
  Terminal debug/state/manifest/final-weight/config hashes are
  `bac19c95...2d8f1`/`f56a1348...085538`/`f67827a8...8dd22`/
  `94f6028b...2c57c`/`5ecb209c...390e3`. `b3` `1222348` improved epoch 4
  step 2164 to `0.5049508746061352` (`0.4985616010/0.5021662469/
  0.5141247760`), a `0.1014977493973674` gain resetting patience; its
  debug/state hashes are `4bef8fa3...298ab`/`13c3ed83...a0fd6`. Both
  artifacts have exact 192 rows and 64/language, finite metrics, no literal
  empty raw strings or parse failures, and intact frozen prompt contracts.
  Released slot started fixed b4: job `1224853` failed at zero seconds before
  model/data/manifest/metric access because immutable source was not exported;
  preserved unchanged retry `1224858` started at 20:30:28 with explicit
  immutable source/runtime paths, verified 694/694 files and fast pure-GDN
  A100-40GB path. Manifest/trial hashes are `f4675ec4...96b1a`/
  `62c6e81b...25db5`. Running A100-40GB jobs are
  `1220056/1222348/1224858`; no A100-80GB/L40S. Quota `52.0%/37.8%`,
  Kombuys untouched, Sheet E/F/G blank, base `16/16`, NER seed-42
  terminal-valid `5/11`, frozen winners `0/8`, held-out `0`.

- At 19:59 SAST Stage-B `b2` job `1220056` improved at epoch 11 step 5951
  to mean F1 `0.4585215715770512` (Tsn/Xho/Zul
  `0.4614395887/0.4381976023/0.4759275237`), matching retained state to
  float precision. Its `0.0026630207781532933` gain exceeds frozen `0.001`,
  so patience reset and epoch 12 resumed near step 6144. Coverage is exact
  192 rows and 64/language, with finite metrics, no literal empty raw strings
  or parse failures, and the frozen prompt contract intact; debug/state
  hashes are `0ac527c8...fa139`/`c4f18951...f26e1`. `b1` `1219931` is in
  epoch-13 corrected generation after loss `0.4200027976337419`, with step
  7033 tentatively due `20:00--20:10`; `b3` `1222348` is in epoch-4 corrected
  generation after loss `0.36654010758524047`, with step 2164 tentatively
  due `20:15--20:30`. Three A100-40GB jobs remain healthy on
  `srvrocgpu010`, no A100-80GB/L40S; quota `52.0%/37.7%`, Kombuys untouched,
  Sheet E/F/G blank, base `16/16`, NER seed-42 terminal-valid `4/11`, frozen
  winners `0/8`, held-out `0`.

- At 19:29 SAST Stage-B `b3` job `1222348` improved at epoch 3 step 1623
  to mean F1 `0.4034531252087678` (Tsn/Xho/Zul
  `0.3728004772/0.4128303868/0.4247285116`), matching retained state to
  float precision. Its `0.08573398241760233` gain reset frozen patience;
  epoch 4 resumed near step 1993. Coverage is exact 192 rows and 64/language,
  with finite metrics, no literal empty raw strings or parse failures, and
  the frozen prompt contract intact; debug/state hashes are
  `80d0d48d...5b29be`/`526bb94f...882d4`. `b2` `1220056` is in epoch-11
  corrected generation after loss `0.4216838056713232`, with step 5951
  tentatively due `19:45--20:00`; `b1` `1219931` is in epoch-13 corrected
  generation after loss `0.4200027976337419`, with step 7033 tentatively due
  `20:00--20:20`. Three A100-40GB jobs remain healthy on `srvrocgpu010`, no
  A100-80GB/L40S; quota `52.0%/37.7%`, Kombuys untouched, Sheet E/F/G blank,
  base `16/16`, NER seed-42 terminal-valid `4/11`, frozen winners `0/8`,
  held-out `0`.

- At 19:00 SAST Stage-B `b1` job `1219931` produced a valid epoch-12
  step-6492 mean F1 `0.5576102928857579` (Tsn/Xho/Zul
  `0.5487421384/0.5537291856/0.5703595547`), matching its newly retained
  numerical-best checkpoint. The `0.0006878515836052` gain is below frozen
  threshold `0.001`, so patience correctly advanced to `1/2` and epoch 13
  resumed near step 6666. Coverage is exact 192 rows and 64/language, with
  finite metrics, no literal empty raw strings or parse failures, and the
  frozen prompt contract intact; debug/state hashes are
  `1f133cd1...e7baf3b`/`f56a1348...085538`. `b3` `1222348` remains in
  epoch-3 corrected generation after loss `0.4227915597227869`, with all
  segments started and step 1623 tentatively due `19:05--19:15`; `b2`
  `1220056` is healthy near epoch-11 boundary step 5951, with an
  output-dependent artifact window `19:45--20:05`. All three A100-40GB jobs
  remain healthy on `srvrocgpu010`, no A100-80GB/L40S; quota `52.0%/37.7%`,
  Kombuys untouched, Sheet E/F/G blank, base `16/16`, NER seed-42
  terminal-valid `4/11`, frozen winners `0/8`, held-out `0`.

- At 18:38 SAST Stage-B `b3` job `1222348` improved at epoch 2 step 1082 to
  mean F1 `0.3177191427911655` (Tsn/Xho/Zul
  `0.3086306801/0.3198029817/0.3247237666`), exactly matching retained state;
  its `0.14118867509099828` gain reset patience. It is now in epoch-3
  corrected generation after loss `0.4227915597227869`. `b2` `1220056`
  improved at epoch 10 step 5410 to `0.45585855079889787`
  (`0.4542906713/0.4345031089/0.4787818722`), a `0.020229851946633903`
  gain over retained epoch 8, resetting patience; epoch 11 resumed near step
  5424. b3 debug/state hashes are `4974e299...31aac3`/`bf9b997f...6ebb8e`;
  b2 values are `d0e4bf49...53c793`/`8da358de...396d18`. `b1` `1219931`
  remains in epoch-12 corrected generation after loss `0.41578371409589915`;
  step 6492 is absent. ETAs: b1 `18:45--19:00`, b3 `19:10--19:30`, b2 next
  `19:45--20:05`. Three A100-40GB jobs remain healthy on `srvrocgpu010`, no
  A100-80GB/L40S; quota `52.0%/37.7%`, Kombuys untouched, Sheet E/F/G blank,
  base `16/16`, NER seed-42 terminal-valid `4/11`, frozen winners `0/8`,
  held-out `0`.

- At 17:49 SAST Stage-B `b1` job `1219931` improved at epoch 11 step 5951
  to mean F1 `0.5569224413021527` (Tsn/Xho/Zul
  `0.5409836066/0.5556978233/0.5740858940`), exactly matching retained state.
  Its `0.010461328483170385` gain exceeds frozen `0.001`, so patience reset
  and epoch 12 resumed near step 6120. `b2` `1220056` epoch-9 step-4869 mean
  F1 `0.4320187263597735` is `0.003609972492490454` below retained epoch-8
  best `0.43562869885226396`; this is its first patience miss, so it correctly
  resumed epoch 10 near step 5400. b1 debug/state hashes are
  `3cbc0b36...b57778`/`bcc862af...c6e009`; b2 values are
  `61e1afce...24c63`/`cf48650c...571461`. `b3` `1222348` remains in
  epoch-2 corrected generation after loss `0.4814153054389812`; step 1082 is
  absent, ETA `18:05--18:20`. Three A100-40GB jobs remain healthy on
  `srvrocgpu010`, no A100-80GB/L40S; quota `52.0%/37.7%`, Kombuys untouched,
  Sheet E/F/G blank, base `16/16`, NER seed-42 terminal-valid `4/11`, frozen
  winners `0/8`, held-out `0`.

- At 17:19 SAST Stage-B `b1/b2/b3` jobs `1219931/1220056/1222348` are all
  healthy in corrected generation for epochs `11/9/2`, after exact losses
  `0.41051052618204/0.427205513312471/0.4814153054389812`. Step
  `5951/4869/1082` artifacts are absent, so no checkpoint or patience
  decision was made from loss. Output-dependent ETA is b2 `17:25--17:35`,
  b1 `17:35--17:50`, b3 `17:55--18:15`. Three A100-40GB jobs remain on
  `srvrocgpu010`, no A100-80GB/L40S; targeted scans empty, quota
  `52.0%/37.7%`, Kombuys untouched, Sheet E/F/G blank, base `16/16`, NER
  seed-42 terminal-valid `4/11`, frozen winners `0/8`, held-out `0`.

- At 16:49 SAST Stage-B `b1` job `1219931` produced epoch-10 step-5410 mean
  F1 `0.5464611128189824` (Tsn/Xho/Zul
  `0.5296850800/0.5441524933/0.5655457652`), exactly matching its new
  numerical-best checkpoint. The `0.00009749732803343569` gain is below
  frozen `0.001`, so the trainer retained the numerical best while patience
  correctly continued; epoch 11 resumed near step 5851. `b3` job `1222348`
  produced its valid initial epoch-1 step-541 mean F1 `0.1765304677001672`
  (`0.1557108539/0.1744019519/0.1994785973`) and resumed epoch 2 near step
  699; no low-fidelity pruning is allowed. b1 debug/state hashes are
  `1bfb9ead...b5e8ca`/`79e3fb55...af6f37`; b3 values are
  `12ea8e6f...f8a12b`/`741f4979...3a234`. `b2` `1220056` entered epoch-9
  corrected generation after loss `0.427205513312471`; step 4869 is absent,
  ETA `17:10--17:25`. Three A100-40GB jobs remain healthy on srvrocgpu010,
  no A100-80GB/L40S; quota `52.0%/37.7%`, Kombuys untouched, Sheet E/F/G
  blank, base `16/16`, NER seed-42 terminal-valid `4/11`, frozen winners
  `0/8`, held-out `0`.

- At 16:19 SAST Stage-B `b2` job `1220056` improved at epoch 8 step 4328 to
  mean F1 `0.43562869885226396` (Tsn/Xho/Zul
  `0.4571794872/0.3972940374/0.4524125719`), exactly matching retained state.
  Its `0.03476144609081261` gain exceeds frozen `0.001`, so patience reset
  and epoch 9 resumed near step 4604. Coverage is exact 192 rows,
  64/language, finite, with no literal empty raw strings; debug/state hashes
  are `3969d89b...b98b98`/`11e19866...20262`. `b1` `1219931` and `b3`
  `1222348` remain in corrected epoch-10/1 generation after exact losses
  `0.40588744209601535/1.21032742071329`; step 5410/541 artifacts are absent,
  so no decision was made from loss. Both are tentatively due
  `16:25--16:40`, output-dependent. Three A100-40GB jobs remain healthy on
  `srvrocgpu010`, no A100-80GB/L40S; quota `52.0%/37.7%`, Kombuys untouched,
  Sheet E/F/G blank, base `16/16`, NER seed-42 terminal-valid `4/11`, frozen
  winners `0/8`, held-out `0`.

- At 15:48 SAST Stage-B `b1/b2` jobs `1219931/1220056` are healthy in
  corrected generation after exact epoch-10/8 losses
  `0.40588744209601535/0.4334614580005518`; step `5410/4328` F1 artifacts
  are absent, so no decision was made from loss. Tentative artifact windows
  are b2 `16:00--16:15`, b1 `16:20--16:35`. New `b3` job `1222348` is
  healthy in epoch 1 near step `442/8115`, with first F1 tentatively
  `16:45--17:05`. Three A100-40GB jobs remain on `srvrocgpu010`, no
  A100-80GB/L40S; targeted scans empty, quota `52.0%/37.7%`, Kombuys
  untouched, Sheet E/F/G blank, base `16/16`, NER seed-42 terminal-valid
  `4/11`, frozen winners `0/8`, held-out `0`.

- At 15:26 SAST Stage-B `b0` job `1218997` is terminal-valid `COMPLETED 0:0`.
  Epoch-14 mean F1 `0.6479530629541063` was the second threshold-nonimproving
  epoch, so frozen patience correctly restored epoch-12 checkpoint 6492 best
  `0.6488713709971404`. Debug/state hashes are
  `28a48acc...74f7b`/`f8319343...e21fed`; manifest/final-weight/config hashes
  are `f9f0c203...5c14f2`/`2e0684c0...cd412c`/`5c95d408...9dec8`.
  `b1` `1219931` improved epoch 9 step 4869 to `0.5463636154909489`, and
  `b2` `1220056` improved epoch 7 step 3787 to `0.40086725276145135`; both
  exactly match retained state and reset patience. Their debug/state hashes
  are `c9a5caf0...fb29c7`/`f83e63e9...67738f` and
  `cc50ddc6...f58e67`/`959b4db7...9affcc`. The released slot started fixed
  `b3` job `1222348`, verified `694/694` immutable files, pure GDN BF16 fast
  path and `4,323/10,760` train/validation rows, and reached step 10. Frozen
  b3 LR/rank-alpha/dropout/warmup are `0.00007207193245743205`, `16/32`,
  `0.06486568450927735`, `0.04297354072332382`; manifest/trial hashes are
  `43ecf3ab...772b0`/`abac3919...1e664c`. Owned state is three running
  A100-40GB jobs `1219931/1220056/1222348`, no A100-80GB/L40S; quota
  `52.0%/37.7%`, Kombuys untouched, Sheet E/F/G blank, base `16/16`, NER
  seed-42 terminal-valid `4/11`, frozen winners `0/8`, held-out `0`.

- At 14:50 SAST Stage-B `b0/b1/b2` jobs `1218997/1219931/1220056` are all
  healthy in corrected generation for epochs `14/9/7`, after exact losses
  `0.40557230555878254/0.39639591940273583/0.44957441025063893`. Their
  step `7574/4869/3787` artifacts are absent, so no scientific decision was
  made from loss. Targeted fault scans are empty. All three remain on
  `srvrocgpu010`, one A100-40GB each; b2 is tentatively due `14:55--15:05`
  and b0/b1 `15:05--15:25`, output-dependent. No A100-80GB/L40S work; quota
  `52.0%/37.6%`, Kombuys untouched, Sheet E/F/G blank, base `16/16`, Stage-A
  NER terminal `3/3`, frozen winners `0/8`, held-out `0`, Mono not started.

- At 14:19 SAST Stage-B `b1` job `1219931` improved at epoch 8 step 4328 to
  mean F1 `0.5393736962954022` (Tsn/Xho/Zul
  `0.5355132672/0.5267403315/0.5558674902`), matching trainer state to float
  precision. Its `0.011490621822435987` gain exceeds frozen `0.001`, so
  checkpoint 4328 was retained and epoch 9 resumed near step 4759. `b0` job
  `1218997` epoch-13 step-7033 mean F1 `0.6475630097288816` is
  `0.0013083612682588397` below its retained epoch-12 checkpoint 6492 F1
  `0.6488713709971404`; this is its first non-improving epoch under patience
  2, and epoch 14 resumed near step 7391. Both artifacts have exact 192-row,
  64/language coverage, finite metrics, no literal empty raw strings, and 58
  whitespace-only outputs that normalize empty. b0 debug/state SHA-256 values
  are `ce7cfa40...42c51b2`/`52ec29b5...600079`; b1 values are
  `af2434eb...be19b3`/`caf0f02c...fc1e7`. `b2` `1220056` entered epoch-7
  corrected generation after loss `0.44957441025063893`; step 3787 is absent.
  Three A100-40GB jobs remain healthy, no A100-80GB/L40S; quota
  `52.0%/37.6%`, Kombuys untouched, Sheet E/F/G blank, base `16/16`, frozen
  winners `0/8`, held-out `0`.

- At 13:49 SAST Stage-B `b2` job `1220056` improved at epoch 6 step 3246 to
  mean F1 `0.3862521680863425` (Tsn/Xho/Zul
  `0.3792564072/0.3698532409/0.4096468562`), exactly matching checkpoint
  3246. The `0.03569960217619822` gain exceeds frozen `0.001`, resetting
  patience; it resumed epoch 7 near step 3529. Coverage is exact 192 rows and
  64/language, no literal empty raw strings, and `58/192` whitespace-only
  outputs (Tsn/Xho/Zul `23/7/28`). These normalize empty with
  `parse_failed=false`, as in earlier corrected NER artifacts, and are scored
  by the preregistered metric rather than indicating the former evaluator-wide
  prompt-EOS fault. Debug/state SHA-256 values are
  `8eebe2ac...146d7f`/`66bdf88c...49f6d2`. `b0` `1218997` and `b1`
  `1219931` remain healthy in corrected generation for epochs 13/8; their
  step 7033/4328 artifacts are absent. Owned state is three running
  A100-40GB jobs, no A100-80GB/L40S; quota `52.0%/37.6%`, Kombuys untouched,
  Sheet E/F/G blank, trusted base `16/16`, frozen winners `0/8`, held-out `0`.

- At 13:18 SAST Stage-B `b0` job `1218997` improved at epoch 12 step 6492 to
  mean F1 `0.6488713709971404` (Tsn/Xho/Zul
  `0.6443671394/0.6459540674/0.6562929062`), exactly matching checkpoint
  6492. The `0.0049355574309944` gain exceeds frozen `0.001`, resetting
  patience; it resumed epoch-13. The 192-row artifact has 64 rows/language and
  all raw predictions nonempty; debug/state SHA-256 values are
  `88a43ed1...1be3ff`/`f8319343...e21fed`. `b1` `1219931` completed exact
  epoch-8 validation loss `0.39547758988731413`, and `b2` `1220056`
  completed exact epoch-6 loss `0.46967894827123025`; both are in corrected
  generation without next F1 artifacts, so no decisions were made from loss.
  Owned state is three running A100-40GB jobs, no A100-80GB/L40S; quota
  `52.0%/37.6%`, Kombuys untouched, Sheet E/F/G blank, trusted base `16/16`,
  frozen winners `0/8`, held-out `0`.

- At 12:48 SAST Stage-B `b1` job `1219931` improved at epoch 7 step 3787 to
  mean F1 `0.5278830744729662` (Tsn/Xho/Zul
  `0.5178216510/0.5036417002/0.5621858722`), exactly matching checkpoint
  3787. The `0.0241741561402960` gain exceeds frozen `0.001`, resetting
  patience; it resumed epoch-8 near step 3919. The 192-row artifact has 64
  rows/language and all raw predictions nonempty; debug/state SHA-256 values
  are `712c9e3a...bd92e1`/`d20882ca...6f9f06`. `b0` `1218997` remains in
  epoch-12 corrected generation after loss `0.3988653133349791`, without F1
  artifact. `b2` `1220056` is healthy in epoch-6 near step 3223. Owned state
  is three running A100-40GB jobs, no A100-80GB/L40S; quota `52.0%/37.6%`,
  Kombuys untouched, Sheet E/F/G blank, trusted base `16/16`, frozen winners
  `0/8`, held-out `0`.

- At 12:23 SAST Stage-B `b2` job `1220056` improved at epoch 5 step 2705 to
  mean F1 `0.3505525659101443` (Tsn/Xho/Zul
  `0.3512463986/0.3318047488/0.3686065503`), exactly matching checkpoint
  2705. The `0.07628408463817873` gain exceeds frozen `0.001`, resetting
  patience; it resumed epoch-6. The 192-row artifact has 64 rows/language and
  all raw predictions nonempty; debug/state SHA-256 values are
  `74aeda3e...95f475`/`a15c9c10...73678d`. `b0` `1218997` completed exact
  epoch-12 validation loss `0.3988653133349791`, and `b1` `1219931`
  completed exact epoch-7 loss `0.393537137056372`; both are in corrected
  generation without next F1 artifacts, so no decisions were made from loss.
  Owned state is three running A100-40GB jobs, no A100-80GB/L40S; quota
  `52.0%/37.6%`, Kombuys untouched, Sheet E/F/G blank, trusted base `16/16`,
  frozen winners `0/8`, held-out `0`.

- At 11:48 SAST Stage-B `b0` job `1218997` improved at epoch 11 step 5951
  to mean F1 `0.643935813566146` (Tsn/Xho/Zul
  `0.6350053362/0.6418292293/0.6549728752`), exactly matching checkpoint
  5951; its `0.005867249566207` gain exceeds frozen `0.001`. Stage-B `b1`
  job `1219931` improved at epoch 6 step 3246 to mean F1
  `0.5037089183326702` (Tsn/Xho/Zul
  `0.4964539007/0.4963758803/0.5182969740`), exactly matching checkpoint
  3246; its `0.0234528348096561` gain also exceeds `0.001`. Both reset
  patience and resumed training. Each artifact has 192 rows, 64/language,
  all raw predictions nonempty; b0 debug/state SHA-256 values are
  `5f81d7ae...c8ffbd`/`1fea1e72...954aa7`, and b1 values are
  `7c8aed47...f2c5f1`/`338f5822...bb0dc`. `b2` `1220056` completed exact
  epoch-5 loss `0.5249240350546004` and is in corrected generation without
  F1 artifact. Owned state is three running A100-40GB jobs, no
  A100-80GB/L40S; quota `52.0%/37.6%`, Kombuys untouched, Sheet E/F/G blank,
  trusted base `16/16`, frozen winners `0/8`, held-out `0`.

- At 11:18 SAST Stage-B `b2` job `1220056` improved at epoch 4 step 2164 to
  mean F1 `0.27426848127196557` (Tsn/Xho/Zul
  `0.2680390934/0.2518824610/0.3028838895`), exactly matching checkpoint
  2164. The `0.10095193978780567` gain exceeds frozen `0.001`, resetting
  patience; it resumed epoch-5 near step 2380. The 192-row artifact has 64
  rows/language and all raw predictions nonempty; debug/state SHA-256 values
  are `e08e539c...af6c4b`/`8a2f35e4...ad9445`. `b0` `1218997` completed
  exact epoch-11 validation loss `0.39432813226068775`, and `b1` `1219931`
  completed exact epoch-6 loss `0.3903183947708527`; both are in corrected
  generation without next F1 artifacts, so no decisions were made from loss.
  Owned state is three running A100-40GB jobs, no A100-80GB/L40S; quota
  `52.0%/37.6%`, Kombuys untouched, Sheet E/F/G blank, trusted base `16/16`,
  frozen winners `0/8`, held-out `0`.

- At 10:48 SAST Stage-B `b0` job `1218997` improved at epoch 10 step 5410 to
  mean F1 `0.638068563999939` (Tsn/Xho/Zul
  `0.6276216858/0.6371529639/0.6494310423`), exactly matching checkpoint
  5410. The `0.0089627700204281` gain exceeds frozen `0.001`, resetting
  patience; it resumed epoch-11 near step 5757. The 192-row artifact has 64
  rows/language and all raw predictions nonempty; debug/state SHA-256 values
  are `665ce314...5a77e1`/`107ab9af...4ea3a6`. `b1` `1219931` is in
  epoch-6 corrected generation; `b2` `1220056` completed exact epoch-4 loss
  `0.7704469021368204` and is also in corrected generation. Neither next F1
  artifact exists, so no decisions were made from loss. Owned state is three
  running A100-40GB jobs, no A100-80GB/L40S; quota `52.0%/37.6%`, Kombuys
  untouched, Sheet E/F/G blank, trusted base `16/16`, frozen winners `0/8`,
  held-out `0`.

- At 10:18 SAST Stage-B `b1` job `1219931` improved at epoch 5 step 2705 to
  mean F1 `0.4802560835230141` (Tsn/Xho/Zul
  `0.4790034673/0.4651410325/0.4966237508`), exactly matching checkpoint
  2705; its `0.06356442360022315` gain exceeds frozen `0.001`. Stage-B `b2`
  job `1220056` improved at epoch 3 step 1623 to mean F1
  `0.1733165414841599` (Tsn/Xho/Zul
  `0.1626785871/0.1680166236/0.1892544138`), exactly matching checkpoint
  1623; its `0.0094945849198557` gain also exceeds `0.001`. Both reset
  patience and resumed training. Each artifact has 192 rows, 64/language,
  all raw predictions nonempty; b1 debug/state SHA-256 values are
  `2f0c4326...6efd29`/`01a5c15f...c2bd6`, and b2 values are
  `f02f04d0...c40787`/`4c54f284...24072e`. `b0` `1218997` completed exact
  epoch-10 validation loss `0.38368800280262544` and entered corrected
  generation without F1 artifact. Owned state is three running A100-40GB
  jobs, no A100-80GB/L40S; quota `52.0%/37.6%`, Kombuys untouched, Sheet
  E/F/G blank, trusted base `16/16`, frozen winners `0/8`, held-out `0`.

- At 09:48 SAST Stage-B `b0` job `1218997` improved at epoch 9 step 4869 to
  mean F1 `0.6291057939795109` (Tsn/Xho/Zul
  `0.6252717391/0.6267364822/0.6353091607`), exactly matching checkpoint
  4869. The `0.0087232233184932` gain exceeds frozen `0.001`, resetting
  patience. The 192-row artifact has 64 rows/language and all raw predictions
  nonempty; debug/state SHA-256 values are
  `10010a53...2dbfd6`/`27b87619...db7afa`. `b1` `1219931` completed exact
  epoch-5 validation loss `0.39540263463130226`, while `b2` `1220056` remains
  in epoch-3 corrected generation after loss `1.1244368429077602`; neither
  next F1 artifact exists, so no decisions were made from loss. Owned state
  is three running A100-40GB jobs, no A100-80GB/L40S; quota `52.0%/37.6%`,
  Kombuys untouched, Sheet E/F/G blank, trusted base `16/16`, frozen winners
  `0/8`, held-out `0`.

- At 09:18 SAST Stage-B `b1` job `1219931` improved at epoch 4 step 2164 to
  mean F1 `0.41669165992279095` (Tsn/Xho/Zul
  `0.4073300858/0.4091173055/0.4336275885`), exactly matching checkpoint
  2164. The `0.09695622609201065` gain exceeds frozen `0.001`, resetting
  patience; it resumed epoch-5 near step 2566. The 192-row artifact has 64
  rows/language and all raw predictions nonempty; debug/state SHA-256 values
  are `259253bb...ad40bf`/`4c3e60e2...c0cbb4`. `b0` `1218997` remains in
  epoch-9 corrected generation after loss `0.3812508735514928`; `b2`
  `1220056` completed exact epoch-3 loss `1.1244368429077602` and entered
  corrected generation. Neither next F1 artifact exists, so no decisions were
  made from loss. Owned state is three running A100-40GB jobs, no
  A100-80GB/L40S; quota `52.0%/37.6%`, Kombuys untouched, Sheet E/F/G blank,
  trusted base `16/16`, frozen winners `0/8`, held-out `0`.

- At 08:53 SAST Stage-B `b2` job `1220056` improved at epoch 2 step 1082 to
  mean F1 `0.1638219565643042` (Tsn/Xho/Zul
  `0.1788502484/0.1369325612/0.1756830601`), exactly matching checkpoint
  1082. The `0.11986505071325975` gain exceeds frozen `0.001`, resetting
  patience; it resumed epoch-3 near step 1421. The 192-row artifact has 64
  rows/language and all raw predictions nonempty; debug/state SHA-256 values
  are `9a122808...5ad69d`/`b69bb8bb...ae681`. `b0` `1218997` completed exact
  epoch-9 validation loss `0.3812508735514928`, and `b1` `1219931` completed
  exact epoch-4 loss `0.438888345597845`; both are in corrected generation
  without task-native F1 artifacts, so no decisions were made from loss.
  Owned state is three running A100-40GB jobs, no A100-80GB/L40S; quota
  `52.0%/37.6%`, Kombuys untouched, Sheet E/F/G blank, trusted base `16/16`,
  frozen winners `0/8`, held-out `0`.

- At 08:11 SAST Stage-B `b0` job `1218997` improved at epoch 8 step 4328 to
  mean F1 `0.6203825706610177` (Tsn/Xho/Zul
  `0.6221399286/0.6079748164/0.6310329670`), exactly matching checkpoint
  4328; the `0.0047530834308320` gain exceeds frozen `0.001`. Stage-B `b1`
  job `1219931` improved at epoch 3 step 1623 to mean F1
  `0.3197354338307803` (Tsn/Xho/Zul
  `0.3180771157/0.3051722678/0.3359569180`), exactly matching checkpoint
  1623; the `0.02923439043360694` gain also exceeds `0.001`. Both runs reset
  patience and continued. Each artifact has 192 rows, 64/language, all raw
  predictions nonempty; b0 debug/state SHA-256 values are
  `46de0d74...0ac3c5`/`16fb1dfc...b1ab2a`, and b1 values are
  `29bfb39c...ec2141`/`510e461b...48adf1`. `b2` `1220056` completed exact
  epoch-2 loss `1.3991323931952835` and is in corrected generation without F1
  artifact. Owned state is three running A100-40GB jobs, no A100-80GB/L40S;
  quota `52.0%/37.6%`, Kombuys untouched, Sheet E/F/G blank, trusted base
  `16/16`, frozen winners `0/8`, held-out `0`.

- At 07:41 SAST Stage-B `b2` job `1220056` produced valid epoch-1 step-541
  mean F1 `0.04395690585104445` (Tsn/Xho/Zul
  `0.0514285714/0.0285934423/0.0518487038`), exactly matching retained
  checkpoint 541. The low initial result is not used for pruning or
  cross-candidate selection; it resumed epoch-2 near step 931. Its 192-row
  artifact has 64 rows/language and all raw predictions nonempty; debug/state
  SHA-256 values are `6347350a...b36ad`/`ac30c40b...9c03e`. `b0` `1218997`
  completed exact epoch-8 validation loss `0.37637544653229554`; `b0` and
  `b1` `1219931` are in corrected generation without their next F1 artifacts.
  Owned state is three running A100-40GB jobs, no A100-80GB/L40S; quota
  `52.0%/37.6%`, Kombuys untouched, Sheet E/F/G blank, trusted base `16/16`,
  frozen winners `0/8`, held-out `0`.

- At 07:11 SAST Stage-B `b0` job `1218997` improved at epoch 7 step 3787 to
  mean F1 `0.6156294872301857` (Tsn/Xho/Zul
  `0.6145573770/0.6090975943/0.6232334904`), exactly matching newly retained
  checkpoint 3787. The `0.0171293534161445` gain exceeds frozen `0.001`,
  resetting patience; it resumed epoch-8 near step 4010. The 192-row artifact
  has 64 rows/language and all raw predictions nonempty; debug/state SHA-256
  values are `694fa03f...78140`/`4b622bfb...e1729`. `b1` `1219931`
  completed exact epoch-3 validation loss `0.723133299873664` and `b2`
  `1220056` completed exact epoch-1 loss `1.9991583501539265`; both are in
  corrected generation without task-native F1 artifacts, so no decisions
  were made from loss. Owned state is three running A100-40GB jobs, no
  A100-80GB/L40S; quota `52.0%/37.6%`, Kombuys untouched, Sheet E/F/G blank,
  trusted base `16/16`, frozen winners `0/8`, held-out `0`.

- At 06:41 SAST Stage-B `b1` job `1219931` improved at epoch 2 step 1082 to
  mean F1 `0.29050104339717336` (Tsn/Xho/Zul
  `0.3032039594/0.2633148262/0.3049843446`), exactly matching newly retained
  checkpoint 1082. The `0.11547740388034549` gain exceeds frozen `0.001`,
  resetting patience; it resumed epoch-3 near step 1389. The 192-row artifact
  has 64 rows/language and all raw predictions nonempty; debug/state SHA-256
  values are `e275e4e5...182135`/`fd2fb9a2...210475`. `b0` `1218997`
  completed exact epoch-7 validation loss `0.3684530038372735` and entered
  corrected generation without F1 artifact. `b2` `1220056` remains healthy
  in epoch-1 training. Owned state is three running A100-40GB jobs, no
  A100-80GB/L40S; quota `52.0%/37.6%`, Kombuys untouched, Sheet E/F/G blank,
  trusted base `16/16`, frozen winners `0/8`, held-out `0`.

- At 06:11 SAST Stage-B `b0` job `1218997` improved at epoch 6 step 3246 to
  mean F1 `0.5985001338140412` (Tsn/Xho/Zul
  `0.5929181316/0.5861427094/0.6164395604`), exactly matching newly retained
  checkpoint 3246. The `0.0155290351307471` gain exceeds frozen `0.001`,
  resetting patience; it resumed healthy epoch-7. The 192-row artifact has 64
  rows/language and all raw predictions nonempty; debug/state SHA-256 values
  are `8e2f4ae7...34c170`/`be463970...97d47`. Fixed `b2` job `1220056`
  started at 06:06:29 on A100-40GB, verified `694/694` immutable files, and
  loaded the canonical pure GDN BF16 path with `4,323` train/`10,760`
  validation rows. Manifest/trial SHA-256 values are
  `00a0a053...54abe`/`e97b1ce2...626e`; its frozen seed-42 parameters match
  the registry. `b1` `1219931` completed exact epoch-2 validation loss
  `1.0695723636442844` and entered corrected generation without F1 artifact.
  Owned state is three running A100-40GB jobs, no A100-80GB/L40S; quota
  `52.0%/37.6%`, Kombuys untouched, Sheet E/F/G blank, trusted base `16/16`,
  frozen winners `0/8`, held-out `0`.

- At 05:41 SAST Stage-B `b0` job `1218997` completed exact 10,760-row
  epoch-6 validation loss `0.35770683430384526` and entered final corrected
  generation. No step-3246 F1 artifact exists yet, so no checkpoint or
  patience decision was made from loss; retained best remains epoch-5 F1
  `0.5829710986832941`. `b1` `1219931` remains healthy entering epoch-2
  validation preparation after its reconciled epoch-1 F1. Fixed `b2`
  `1220056` is priority-pending with dynamic 12:27 SAST projection. Owned
  state is two running plus one pending A100-40GB jobs, no A100-80GB/L40S;
  quota `52.0%/37.6%`, Kombuys untouched, Sheet E/F/G blank, trusted base
  `16/16`, frozen winners `0/8`, held-out `0`.

- At 05:11 SAST enhanced Stage-B `b1` job `1219931` produced valid epoch-1
  step-541 mean F1 `0.17502363951682787` (Tsn/Xho/Zul
  `0.1639749192/0.1542686901/0.2068273092`), exactly matching retained
  checkpoint 541. Its 192-row artifact has 64 rows/language and all raw
  predictions nonempty; debug/state SHA-256 values are
  `fc851a41...c1b31`/`751efbd2...9b96f`. This initial result is not used for
  pruning or cross-candidate selection; the job resumed healthy epoch-2.
  `b0` `1218997` remains healthy in epoch-6 validation with retained epoch-5
  F1 `0.5829710986832941`; fixed `b2` `1220056` is priority-pending with
  dynamic 12:27 SAST projection. Owned state is two running plus one pending
  A100-40GB jobs, no A100-80GB/L40S; quota `52.0%/37.6%`, Kombuys untouched,
  Sheet E/F/G blank, trusted base `16/16`, frozen winners `0/8`, held-out `0`.

- At 04:42 SAST enhanced Stage-B `b0` job `1218997` improved at epoch 5
  step 2705 to mean F1 `0.5829710986832941` (Tsn/Xho/Zul
  `0.5850447604/0.5677351550/0.5961333806`), exactly matching newly retained
  checkpoint 2705. The `0.0275625934877679` gain exceeds frozen `0.001`,
  resetting patience; it resumed healthy epoch-6 training. The 192-row
  artifact has 64 rows/language and all raw predictions nonempty; debug/state
  SHA-256 values are `de189383...ca665c`/`50f76630...f263e`. Stage-B `b1`
  `1219931` completed exact epoch-1 validation loss `1.4194149882376859` and
  entered corrected generation without an F1 artifact. Fixed `b2` `1220056`
  remains priority-pending, dynamically projected for 12:27 SAST. Owned state
  is two running plus one pending A100-40GB jobs, no A100-80GB/L40S; quota
  `52.0%/37.5%`, Kombuys untouched, Sheet E/F/G blank, trusted base `16/16`,
  frozen winners `0/8`, held-out `0`.

- At 04:12 SAST Stage-A NER LR `1.5e-4` job `1218345` was verified terminal
  `0:0` after epoch-14 F1 `0.678259542642304`, the second consecutive
  non-improvement after retained epoch-12 best `0.6795225766333571`; frozen
  patience correctly stopped and restored checkpoint 6492. Its epoch-14
  192-row artifact has 64 rows/language and all raw predictions nonempty;
  debug SHA-256 is `7548e8d0...5c8ca`, final-adapter weight/config hashes are
  `f82ad9ba...e337`/`d085fe1a...c826`. Stage-A is now valid `3/3`:
  `a0`/`1218343` `0.500502987595103`, `a1`/`1218344`
  `0.6173130857642662`, `a2`/`1218345` `0.6795225766333571`; `a2` is only
  provisional until all Stage-B/confirmation work completes. The released
  GPU started `b1` job `1219931`, which verified `694/694` files and was
  healthy near `350/8115`; manifest/trial SHA-256 values are
  `f67827a8...8dd22`/`cc5b4391...b88e`. Fixed `b2` was submitted as job
  `1220056` and is priority-pending with LR `0.00003617950373518279`, rank/
  alpha `8/16`, dropout `0.019731278717517856`, warmup
  `0.09582568347454071`. Owned state is two running plus one pending
  A100-40GB jobs, no A100-80GB/L40S; quota `52.0%/37.5%`, Kombuys untouched,
  Sheet E/F/G blank, trusted base `16/16`, frozen winners `0/8`, held-out `0`.

- At 03:44 SAST Stage-B `b0` job `1218997` improved at epoch 4 step 2164 to
  mean F1 `0.5554085051955262` (Tsn/Xho/Zul
  `0.5657188841/0.5305276252/0.5699790063`), exactly matching the newly
  retained checkpoint. The `0.07601672733558295` gain exceeds frozen
  `0.001`, resetting patience; it resumed healthy epoch-5 training. The
  192-row artifact has 64 rows per language and all raw predictions nonempty;
  debug/state SHA-256 values are
  `0036f193...dd5783`/`17b023d4...da191`. Stage-A `1218345` completed exact
  epoch-14 validation loss `0.3901939760796643` and entered corrected
  generation without an F1 artifact yet; retained best remains
  `0.6795225766333571` with patience state still one non-improving epoch.
  `b1` `1219931` remains resources-pending. No candidate-level selection or
  held-out access occurred; trusted base `16/16`, frozen Multilingual winners
  `0/8`, held-out `0`, Sheet E/F/G blank.

- At 03:14 SAST Stage-B `b0` job `1218997` completed exact 10,760-row
  epoch-4 validation loss `0.3470055222068134` and entered its final corrected
  generation segment. No step-2164 F1 artifact exists yet, so no checkpoint,
  patience, pruning, or cross-candidate decision was made from loss. Stage-A
  `1218345` reached epoch-14 step 7574 and entered validation without a new
  loss or F1 artifact; retained best remains `0.6795225766333571` with one
  non-improving epoch. `b1` `1219931` remains priority-pending. Owned state is
  two running plus one pending A100-40GB jobs, no A100-80GB/L40S; quota home
  `52.0%`, scratch `37.5%`; Kombuys untouched, Sheet E/F/G blank, trusted base
  `16/16`, frozen Multilingual winners `0/8`, held-out `0`.

- At 02:42 SAST Stage-A NER LR `1.5e-4` job `1218345` produced a valid
  epoch-13 step-7033 mean F1 `0.6766570733246425` (Tsn/Xho/Zul
  `0.6644535918/0.6800707737/0.6854468545`). This is
  `0.0028655033087146` below retained epoch-12 best
  `0.6795225766333571`; checkpoint 6492 correctly remains best and the run
  resumed epoch 14 under the frozen patience rule. The 192-row artifact has
  64 rows per language and all raw predictions nonempty; debug/epoch-13
  state SHA-256 values are `baaa4527...73327`/`11296081...d20b7`.
  Stage-B `b0` `1218997` reached epoch-4 step 2164 and entered full
  validation without an F1 artifact yet; `b1` `1219931` remains
  priority-pending. No candidate-level selection or held-out access occurred;
  trusted base `16/16`, frozen Multilingual winners `0/8`, held-out `0`,
  Sheet E/F/G blank.

- At 02:12 SAST enhanced Stage-B `b0` job `1218997` improved at epoch 3
  step 1623 to mean F1 `0.47939177785994325` (Tsn/Xho/Zul
  `0.4617483233/0.4673776054/0.5090494049`), exactly matching the newly
  retained checkpoint. The `0.12854471703682655` gain exceeds frozen
  `0.001`, resetting patience. The 192-row artifact has 64 rows per language
  and all raw predictions nonempty; debug/state SHA-256 values are
  `5c95a258...746fa2`/`ed55337e...2d90c`. Stage-A `1218345` completed exact
  epoch-13 validation loss `0.38740929812746866` and entered corrected
  generation without an F1 artifact yet; retained best remains
  `0.6795225766333571`. `b1` `1219931` remains priority-pending. No
  candidate-level selection or held-out access occurred; trusted base
  `16/16`, frozen Multilingual winners `0/8`, held-out `0`, Sheet E/F/G blank.

- At 01:42 SAST Stage-A NER LR `1.5e-4` job `1218345` improved at epoch 12
  step 6492 to mean F1 `0.6795225766333571` (Tsn/Xho/Zul
  `0.6625962304/0.6855315747/0.6904399247`), exactly matching the newly
  retained checkpoint. The `0.0022997008228423` gain exceeds frozen `0.001`,
  resetting patience; it resumed healthy epoch-13 training. The 192-row
  artifact has 64 rows per language and all raw predictions nonempty;
  debug/state SHA-256 values are
  `f1b48ab5...19adee`/`c0e54e94...3fb66`. Stage-B `b0` `1218997` completed
  exact epoch-3 validation loss `0.3786283528494569` and entered corrected
  generation without an F1 artifact yet; its retained best remains
  `0.3508470608231167`. `b1` `1219931` remains priority-pending. No
  candidate-level selection or held-out access occurred; trusted base
  `16/16`, frozen Multilingual winners `0/8`, held-out `0`, Sheet E/F/G blank.

- At 01:12 SAST enhanced Stage-B `b0` job `1218997` produced a valid
  epoch-2 step-1082 mean F1 `0.3508470608231167` (Tsn/Xho/Zul
  `0.3492439964/0.3375142531/0.3657829329`), exactly matching the newly
  retained checkpoint. The `0.24554592951990876` gain over epoch 1 exceeds
  frozen `0.001`, resetting patience; it resumed healthy epoch-3 training.
  The 192-row artifact has 64 rows per language and all raw predictions
  nonempty; debug/state SHA-256 values are
  `1742612f...b0937b`/`3385bd0d...ef168`. Stage-A `1218345` completed exact
  epoch-12 validation loss `0.38388843181851184` and entered corrected
  generation without an F1 artifact yet; retained best remains
  `0.6772228758105148`. `b1` `1219931` remains priority-pending. No
  candidate-level selection or held-out access occurred; trusted base
  `16/16`, frozen Multilingual winners `0/8`, held-out `0`, Sheet E/F/G blank.

- At 00:46 SAST enhanced Stage-B `b0` job `1218997` had completed exact
  10,760-row epoch-2 validation loss `0.4526454996442263` and entered its
  final corrected-generation language segment. No step-1082 F1 artifact
  exists yet, so no checkpoint, patience, pruning, or cross-candidate
  decision was made from loss. Stage-A `1218345` remained healthy at
  `6463/8115` in epoch 12 with provisional best F1 `0.6772228758105148`;
  Stage-B `b1` `1219931` remained priority-pending. All owned work is two
  running plus one pending A100-40GB jobs, no A100-80GB/L40S; quota home
  `52.0%`, scratch `37.5%`; Kombuys untouched, `GDN Results` E/F/G blank,
  trusted base `16/16`, frozen Multilingual winners `0/8`, held-out `0`.

- At 00:21 SAST Stage-A NER LR `1.5e-4` job `1218345` improved at epoch 11
  step 5951 to mean F1 `0.6772228758105148` (Tsn/Xho/Zul
  `0.6594354417/0.6821315944/0.6901015913`), exactly matching the newly
  retained checkpoint. The `0.0034844431694402` gain exceeds the frozen
  `0.001` threshold, resetting patience; it resumed healthy epoch-12 training
  and NER remains unfrozen. The 192-row artifact has 64 rows per language and
  all raw predictions nonempty; debug/state SHA-256 values are
  `afc4907d...a0c78f`/`351fb3a2...df6a`. Stage-B `b0` `1218997` remains
  healthy in epoch-2 validation/generation without a step-1082 F1 artifact;
  `b1` `1219931` remains priority-pending. No pruning, cross-candidate
  selection, or held-out access occurred.

- At 00:16 SAST on 12 August, Stage-A NER LR `1.5e-4` job `1218345`
  completed exact 10,760-row epoch-11 validation loss
  `0.38024836543767426` and remained healthy in corrected generation; no
  step-5951 F1 artifact exists yet, so retained provisional best stays
  `0.6737384326410746`. Enhanced Stage-B `b0` job `1218997` reached epoch-2
  validation/generation without faults; no step-1082 F1 artifact exists yet
  and its epoch-1 metric is not used for pruning. After verifying the free
  protocol slot, immutable launcher/registry hashes, no duplicate/artifact,
  and the exact A100-40GB Slurm envelope, fixed Stage-B `b1` was submitted as
  job `1219931` and is priority-pending. Its frozen seed-42 configuration is
  LR `0.000030329114402234482`, rank/alpha `32/64`, dropout
  `0.08691746592521668`, warmup `0.029702768474817273`. Owned state is two
  running plus one pending A100-40GB jobs, no A100-80GB/L40S; quota home
  `52.0%`, scratch `37.5%`; Kombuys remains read-only/untouched. Trusted base
  `16/16`, frozen Multilingual winners `0/8`, held-out `0`, Mono not started,
  and `GDN Results` E/F/G blank.

- At 23:40 SAST enhanced NER Stage-B `b0` job `1218997` produced a valid
  epoch-1 step-541 mean F1 `0.10530113130320794` (Tsn/Xho/Zul
  `0.1190476190/0.1205028618/0.0763529130`), matching its retained checkpoint
  to floating-point representation. Its 192-row artifact has 64 rows per
  language and all raw predictions nonempty; debug/state SHA-256 values are
  `d9385ecd...addfe8`/`31722fd4...9380a`. This is only the first
  within-candidate point and the no-low-fidelity-pruning rule keeps it running
  into epoch 2. Stage-A `1218345` remained healthy in epoch-11 full validation
  with provisional best `0.6737384326410746`. Owned state is two running
  A100-40GB jobs, no A100-80GB/L40S; quota home `52.0%`, scratch `37.4%`;
  Kombuys remains read-only/untouched. Trusted base `16/16`, frozen
  Multilingual winners `0/8`, held-out `0`, Mono not started, and `GDN
  Results` E/F/G blank.

- At 23:10 SAST Stage-A NER LR `1.5e-4` job `1218345` materially improved at
  epoch 10 step 5410 to mean F1 `0.6737384326410746` (Tsn/Xho/Zul
  `0.6492476060/0.6827980168/0.6891696751`), exactly matching its retained
  checkpoint. The `0.0053915640314053` gain exceeds the frozen `0.001`
  threshold, resetting patience; it correctly continued into epoch 11 and NER
  remains unfrozen. Its 192-row artifact has all raw predictions nonempty;
  debug/state SHA-256 values are `2d5fe174...846eb`/`b3785a9f...b4988`.
  Enhanced `b0` job `1218997` completed exact 10,760-row epoch-1 validation
  loss `0.9400531300824814` and entered corrected generation; no Stage-B F1
  artifact exists yet. Both jobs are healthy with zero targeted faults. Owned
  state is two running A100-40GB jobs, no A100-80GB/L40S; quota home `52.0%`,
  scratch `37.4%`; Kombuys remains read-only/untouched. Trusted base `16/16`,
  frozen Multilingual winners `0/8`, held-out `0`, Mono not started, and
  `GDN Results` E/F/G blank.

- At 22:41 SAST enhanced NER Stage-B `b0` job `1218997` started at
  `22:25:31` on A100-40GB and passed startup verification: `694/694`
  immutable files, execution-manifest SHA-256 `f9f0c203...c14f2`, canonical
  pure GDN BF16 path, and exact validation-only seed-42 candidate LR
  `0.00015156541821567134`, rank/alpha `8/16`, dropout
  `0.028929591178894043`, warmup `0.061286738514900206`. Hashed trial
  SHA-256 is `9072fba1...36aa8`; `4,323` train and `10,760` declared
  validation rows, no held-out access. It was healthy near `268/8115`.
  Stage-A `1218345` remained healthy in epoch-10 generation with retained
  provisional best `0.6683468686096693`. Owned state is two running
  A100-40GB jobs, no A100-80GB/L40S; quota home `52.0%`, scratch `37.4%`;
  Kombuys remains read-only/untouched. Trusted base `16/16`, frozen
  Multilingual winners `0/8`, held-out `0`, Mono not started, and `GDN
  Results` E/F/G blank.

- At 22:10 SAST Stage-A NER LR `1.5e-4` job `1218345` materially improved at
  epoch 9 step 4869 to mean F1 `0.6683468686096693` (Tsn/Xho/Zul
  `0.6594493451/0.6606450242/0.6849462366`), exactly matching its retained
  checkpoint. The `0.0085636587751213` gain exceeds the frozen `0.001`
  threshold, resetting patience; it correctly continued into epoch 10 and NER
  remains unfrozen. The 192-row artifact has 64 rows per language and all raw
  predictions nonempty; debug/state SHA-256 values are
  `1db3d3a3...545f2`/`7cdfff25...5e641`. Enhanced `b0` job `1218997`
  remains Resources-pending with no output/metric and projected start
  `2026-08-12 00:48:20 SAST`. Owned state is one running plus one pending
  A100-40GB job, no A100-80GB/L40S; quota home `52.0%`, scratch `37.4%`;
  Kombuys remains read-only/untouched. Trusted base `16/16`, frozen
  Multilingual winners `0/8`, held-out `0`, Mono not started, and `GDN
  Results` E/F/G blank.

- At 21:10 SAST Stage-A NER LR `1.5e-4` job `1218345` materially improved at
  epoch 8 step 4328 to mean F1 `0.659783209834548` (Tsn/Xho/Zul
  `0.6519292605/0.6511328251/0.6762875440`), exactly matching its retained
  checkpoint. The `0.0073393242457308` gain exceeds the frozen `0.001`
  threshold, resetting patience; it correctly continued into epoch 9 and NER
  remains unfrozen. The 192-row artifact has 64 rows per language and all raw
  predictions nonempty; debug/state SHA-256 values are
  `0c441991...7cd93`/`868ad3b3...d2cab`. Enhanced `b0` job `1218997`
  remains Resources-pending with no output/metric and projected start
  `2026-08-12 00:48:20 SAST`. Owned state is one running plus one pending
  A100-40GB job, no A100-80GB/L40S; quota home `52.0%`, scratch `37.4%`;
  Kombuys remains read-only/untouched. Trusted base `16/16`, frozen
  Multilingual winners `0/8`, held-out `0`, Mono not started, and `GDN
  Results` E/F/G blank.

- At 20:10 SAST Stage-A NER LR `1.5e-4` job `1218345` remained healthy in
  epoch-8 full validation/generation with zero targeted faults and no new
  selection artifact; retained provisional best stays epoch 7 mean F1
  `0.6524438855888172`. Enhanced `b0` job `1218997` remains pending with no
  output/metric, but Slurm moved it to `Resources` and advanced its dynamic
  projected start to `2026-08-12 00:48:20 SAST`. Owned state is one running
  plus one pending A100-40GB job, no A100-80GB/L40S; quota home `52.0%`,
  scratch `37.4%`; Kombuys remains read-only/untouched. Trusted base `16/16`,
  frozen Multilingual winners `0/8`, held-out `0`, Mono not started, and
  `GDN Results` E/F/G blank.

- At 19:40 SAST Stage-A NER LR `1.5e-4` job `1218345` materially improved at
  epoch 7 step 3787 to mean F1 `0.6524438855888172` (Tsn/Xho/Zul
  `0.6412354805/0.6379301204/0.6781660559`), exactly matching its retained
  checkpoint. The `0.0161418127602831` gain exceeds the frozen `0.001`
  threshold, resetting patience; it correctly continued into epoch 8 and NER
  remains unfrozen. The 192-row artifact has 64 rows per language and all raw
  predictions nonempty; debug/state SHA-256 values are
  `79770d80...6b355d`/`ed2174ac...a1252`. Enhanced `b0` job `1218997`
  remains priority-pending with no output/metric and projected start
  `2026-08-12 07:07 SAST`. Owned state is one running plus one pending
  A100-40GB job, no A100-80GB/L40S; quota home `52.0%`, scratch `37.4%`;
  Kombuys remains read-only/untouched. Trusted base `16/16`, frozen
  Multilingual winners `0/8`, held-out `0`, Mono not started, and `GDN
  Results` E/F/G blank.

- At 18:40 SAST Stage-A NER LR `1.5e-4` job `1218345` improved
  provisionally at epoch 6 step 3246 to mean F1 `0.6363020728285341`
  (Tsn/Xho/Zul `0.6250316696/0.6259589467/0.6579156021`), exactly matching
  its retained checkpoint. The gain over epoch 5 is only
  `0.0001544210587123`, below the frozen `0.001` threshold, so it correctly
  continued into epoch 7; NER remains unfrozen. The 192-row artifact has 64
  rows per language and all raw predictions nonempty; debug/state SHA-256
  values are `2b53aee1...34a42`/`fd3e87b4...9e14`. Enhanced `b0` job
  `1218997` remains priority-pending with no output/metric and projected start
  `2026-08-12 07:07 SAST`. Owned state is one running plus one pending
  A100-40GB job, no A100-80GB/L40S; quota home `52.0%`, scratch `37.4%`;
  Kombuys remains read-only/untouched. Trusted base `16/16`, frozen
  Multilingual winners `0/8`, held-out `0`, Mono not started, and `GDN
  Results` E/F/G blank.

- At 18:12 SAST Stage-A NER LR `1.5e-4` job `1218345` became the
  provisional Stage-A leader with a scientifically valid epoch-5 step-2705
  mean F1 `0.6361476517698218` (Tsn/Xho/Zul
  `0.6310731447/0.6197408415/0.6576289692`), exactly matching its retained
  checkpoint. Its 192-row artifact has 64 rows per language and all raw
  predictions nonempty; debug/state SHA-256 values are
  `f43d7138...e3436c`/`55664938...7977`. It remains healthy in epoch-6
  generation, so NER is not frozen. Enhanced `b0` job `1218997` remains
  priority-pending with no output/metric and dynamic start projection
  `2026-08-12 07:07 SAST`. Owned state is one running plus one pending
  A100-40GB job, no A100-80GB/L40S; quota is home `52.0%`, scratch `37.4%`;
  Kombuys remains read-only/untouched. Trusted base `16/16`, frozen
  Multilingual winners `0/8`, held-out `0`, Mono not started, and `GDN
  Results` E/F/G blank.

- At 17:10 SAST Stage-A NER LR `3e-5` job `1218343` completed `0:0` after
  `18:35:25`, all 15 epochs, with zero targeted faults. Validation-only best
  is epoch 13 step `7033`, mean F1 `0.500502987595103`; state SHA-256 is
  `a483590e...ae04`. Final adapter weight/config SHA-256 values are
  `ed8d5c1c...7c50`/`ccedbaf6...c5f`; configs match retained checkpoint, while
  exact tensor equality remains unclaimed because checkpoint/final weights
  use bin/safetensors serialization. LR `1.5e-4` job `1218345` remains healthy
  in epoch-5 validation/generation. Enhanced `b0` job `1218997` remains
  priority-pending with no output/metric and dynamic start projection
  `2026-08-12 07:07 SAST`. Only A100-40GB is owned; quota is home `52.0%`,
  scratch `37.4%`; Kombuys remains read-only/untouched. Trusted base `16/16`,
  frozen Multilingual winners `0/8`, held-out `0`, Mono not started, and
  `GDN Results` E/F/G blank.

- At 14:24 SAST the enhanced uniform validation-only adapter HPO process was
  implemented and frozen for pure GDN, LLaMA-125M, Mamba-125M, xLSTM-125M,
  the historical Qwen-GDN hybrid, and explicit custom profiles. All profiles
  consume one architecture-neutral 11-candidate registry and identical
  Stage-A/Stage-B/three-seed confirmation/tie-break rules while preserving
  model-specific checkpoints and LoRA targets; cross-model metrics never
  select candidates. Full local verification is `115 passed`, with clean
  Ruff, shell syntax, and all-profile dry runs. Preregistration/registry
  SHA-256 values are `b7d50b8d...5c3c` and `8fdd6ea5...b726`.
  Read-only HEX snapshot
  `/home/lmbanr001/masters/sallm_snapshots/uniform-adapter-hpo-20260811-6aabf717`
  matches all `694/694` local source/config hashes at source-set SHA-256
  `6aabf717...b914`; deployment-manifest SHA-256 is `da1c4567...b285`.
  Pure-GDN NER Stage-B `b0` job `1218997` was submitted with LR
  `0.00015156541821567134`, rank/alpha `8/16`, dropout
  `0.028929591178894043`, warmup `0.061286738514900206`, and seed `42` under
  the exact `nlpgroup/a100/nlpgroup`, `gpu:ampere:1`, 24-hour, 8-CPU Slurm
  envelope. It is priority-pending with dynamic projection 12:32 SAST on 12
  August because all four A100-40GB devices are occupied. Owned Stage-A jobs
  `1218343/1218345` remain healthy with zero targeted faults; no other owned
  GPU family exists. Quota is home `52.0%`, scratch `37.3%`; Kombuys remains
  read-only/untouched. Trusted base is `16/16`, frozen Multilingual winners
  `0/8`, held-out adapter tests `0`, Mono not started, and `GDN Results`
  E/F/G blank. Next gate: `1218997` startup and execution-manifest/model/LoRA
  verification; no held-out access is authorized.

- At 14:04 SAST, NER LR `1.5e-4` job `1218345` produced a valid epoch-2
  step-1082 artifact with Tsn/Xho/Zul F1
  `0.4615384615/0.4418879056/0.5015956141`, exact mean
  `0.46834066041663247`, and retained `checkpoint-1082`. The artifact has
  exactly `192` rows (`64` per language), all raw predictions nonempty, and
  debug/state SHA-256 values `4f918661...51b9`/`8de32147...3804`. It resumed
  epoch 3 near `1576/8115`. LR `3e-5` job `1218343` reached epoch-13
  step `7033`, completed exact validation loss `0.40935960507304253`, and
  entered corrected generation; its epoch-13 F1 artifact was not yet present.
  Both jobs were healthy on A100-40GB `gpu:ampere`, with no owned
  A100-80GB/L40S work; quota was home `33.7%`, scratch `37.3%`; Kombuys
  remained read-only and untouched. The provisional cross-LR leader remains
  completed LR `8e-5` job `1218344` at mean F1 `0.6173130857642662`, but NER
  is not frozen until `1218343/1218345` terminate. Trusted base is `16/16`,
  frozen Multilingual winners `0/8`, held-out adapter tests `0`, Monolingual
  not started, and `GDN Results` E/F/G remain blank.

- At 13:31 SAST, NER LR `3e-5` job `1218343` produced a valid epoch-12
  step-6492 artifact with Tsn/Xho/Zul F1
  `0.4837698960/0.4780214176/0.5113529261`, exact mean
  `0.491048079892295`, and retained `checkpoint-6492`. This improves the
  epoch-11 best by `0.0033917618186639`, above the frozen `0.001` threshold,
  so the preregistered rule correctly continued into epoch 13. The artifact
  has exactly `192` rows (`64` per language), all raw predictions nonempty,
  and debug/state SHA-256 values `858611a1...4892`/`50da38ed...629f`.
  LR `1.5e-4` job `1218345` completed exact epoch-2 validation loss
  `0.3859550674608649` and remained healthy in corrected generation; its
  step-1082 F1 artifact was not yet present. Jobs `1218343/1218345` were the
  only active owned jobs, both on A100-40GB `gpu:ampere`; there was no owned
  A100-80GB/L40S work. Quota was home `33.7%`, scratch `37.3%`; Kombuys
  remained read-only and untouched. Trusted base is `16/16`, frozen
  Multilingual winners `0/8`, held-out adapter tests `0`, Monolingual not
  started. The workbook now labels the historical hybrid tab/dashboard
  `Qwen Results`/`Qwen` and the pure model tab `GDN Results`; E/F/G remain
  blank. NER remains unfrozen until jobs `1218343/1218345` terminate.

- At 12:31 SAST, NER LR `3e-5` job `1218343` produced a valid epoch-11
  artifact with Tsn/Xho/Zul F1
  `0.4857655809/0.4750207297/0.5021826436`, exact mean
  `0.4876563180736311`, and retained `checkpoint-5951`. The gain over epoch
  10 is only `0.0001763731566706`, below the frozen early-stopping threshold
  `0.001`; it resumed epoch 12 near `6304/8115` under the preregistered rule.
  NER LR `1.5e-4` job `1218345` produced its first valid epoch-1 artifact:
  Tsn/Xho/Zul `0.2152499767/0.2542787286/0.2983107340`, exact mean
  `0.255946479788738`, matching `checkpoint-541`, and resumed epoch 2 near
  `673/8115`. Both artifacts have exactly `192` rows (`64` per language),
  all raw predictions nonempty, and debug/state SHA-256 values
  `830075ff...2e30`/`6f60156d...a9bf` and
  `bfbb5797...5ac1`/`753e3ed5...3ffc`. Both jobs are healthy on A100-40GB
  with no A100-80GB/L40S overlap; quota is home `33.7%`, scratch `37.3%`;
  Kombuys remains read-only and untouched. NER is not frozen; trusted base is
  `16/16`, Multilingual winners `0/8`, held-out `0`, Mono not started, and
  Sheet E/F/G blank.

- Clean NER LR `8e-5` job `1218344` completed `0:0` at 11:07:19 SAST after
  `12:06:17`. Epoch-10 Tsn/Xho/Zul F1
  `0.5899299247/0.6182086718/0.6246112839`, mean
  `0.6109166268028269`, was its second non-improvement after epoch 8, so the
  frozen patience rule stopped at epoch 10 and retained `checkpoint-4328`
  with best mean `0.6173130857642662`. The terminal 192-row artifact is
  complete and all predictions are nonempty; debug SHA-256 is
  `523954f7...260a`. Exact comparison verified all `424/424` final-adapter
  tensors equal the retained checkpoint, with final adapter SHA-256
  `3bcf812b...1e40`. LR `1.5e-4` job `1218345` started immediately at
  11:07:19 on the released A100-40GB, verified its 691-file manifest
  `3181df56...beb1`, loaded the canonical pure GDN, and was healthy at
  `278/8115`; LR `3e-5` `1218343` remained healthy at `5854/8115`.
  Exactly two A100-40GB jobs are active with no A100-80GB/L40S; quota is home
  `33.7%`, scratch `37.3%`; Kombuys remains read-only and untouched. NER is
  not frozen until jobs `1218343/1218345` terminate. Trusted base is `16/16`,
  frozen Multilingual winners `0/8`, held-out `0`, Mono not started, and
  Sheet E/F/G blank.

- At 10:22 SAST, clean NER LR `8e-5` job `1218344` completed its valid
  epoch-9 artifact with Tsn/Xho/Zul F1
  `0.5981203535/0.6170811049/0.6231816049`, mean
  `0.6127943544298625`. This did not improve its epoch-8 retained best
  `0.6173130857642662`, so `checkpoint-4328` remains selected within the LR
  and the frozen patience rule continues. The 192-row artifact is complete
  and all predictions are nonempty; debug/state SHA-256 values are
  `2f3022f3...6098`/`ea32813e...b3ee`. Jobs `1218343/1218344` were healthy
  at the epoch-10 boundary (`5410/8115` and `5394/8115`) with zero targeted
  faults; LR `1.5e-4` job `1218345` remains resource-pending. Two A100-40GB
  jobs are active with no A100-80GB/L40S; quota is home `33.7%`, scratch
  `37.3%`; Kombuys remains read-only and untouched. Trusted base is `16/16`,
  frozen Multilingual winners `0/8`, held-out `0`, Mono not started, and
  Sheet E/F/G blank.

- At 09:52 SAST, clean NER LR `3e-5` job `1218343` improved at epoch 9 to
  Tsn/Xho/Zul F1 `0.4719526031/0.4591701499/0.4890492141`, exact mean
  `0.473390655677614`, matching newly retained `checkpoint-4869`. Its
  artifact has exactly `192` rows (`64` per language), all raw predictions
  nonempty, and debug/state SHA-256 `542aebb4...380a`/`ca09f509...b2ff`.
  It resumed epoch 10 near `4994/8115`. LR `8e-5` job `1218344` remained
  healthy in epoch-9 generation at step `4869`; no step-4869 artifact existed,
  so its provisional best remains `0.6173130857642662`. LR `1.5e-4` job
  `1218345` remains resource-pending. Two A100-40GB jobs are active with no
  A100-80GB/L40S; quota is home `33.7%`, scratch `37.2%`; Kombuys remains
  read-only and untouched. Trusted base is `16/16`, frozen Multilingual
  winners `0/8`, held-out `0`, Mono not started, and Sheet E/F/G blank.

- At 08:50 SAST, both clean NER trials produced scientifically valid epoch-8
  step-4328 artifacts and resumed epoch 9. LR `3e-5` job `1218343` improved
  to Tsn/Xho/Zul F1 `0.4713928712/0.4315817839/0.4954128440`, exact mean
  `0.46612916638577717`; LR `8e-5` job `1218344` improved to
  `0.6177868296/0.6111986097/0.6229538180`, exact mean
  `0.6173130857642662`. Each artifact has exactly `192` rows (`64` per
  language), all raw predictions nonempty, and its mean exactly matches the
  retained `checkpoint-4328` state. Debug/state SHA-256 values are
  `1468ea5f...7002`/`6aa18335...0236` and
  `ac1f0ac2...89c3`/`dabc4dd4...3935`. Jobs were healthy at
  `4699/8115` and `4406/8115`; LR `1.5e-4` `1218345` remains
  resource-pending. Two A100-40GB jobs are active with no A100-80GB/L40S;
  quota is home `33.7%`, scratch `37.2%`; Kombuys remains read-only and
  untouched. Trusted base is `16/16`, frozen Multilingual winners `0/8`,
  held-out `0`, Mono not started, and Sheet E/F/G blank.

- At 07:50 SAST, clean NER LR `8e-5` job `1218344` exceeded mean F1 `0.60`
  at epoch 7: Tsn/Xho/Zul `0.6016624885/0.6030139935/0.6232616941`, exact
  mean `0.6093127253514629`, matching newly retained `checkpoint-3787`.
  Its 192-row artifact is all nonempty with debug/state SHA-256
  `762e48dd...eee4`/`f58a7bea...9522`. It resumed epoch 8 and provisionally
  leads LR `3e-5` best `0.43220752568933674`; `1218343` entered epoch-8
  validation and LR `1.5e-4` `1218345` remains pending. Two A100-40GB
  active, no A100-80GB/L40S; quota home `33.7%`, scratch `37.2%`; Kombuys
  untouched, held-out `0`, winners `0/8`, Mono not started, Sheet E/F/G
  blank.

- At 07:20 SAST, clean NER LR `3e-5` job `1218343` improved at epoch 7:
  Tsn/Xho/Zul F1 `0.4530386740/0.3793400578/0.4642438453`, exact mean
  `0.43220752568933674`, matching newly retained `checkpoint-3787`. Its
  192-row artifact is all nonempty with debug/state SHA-256
  `f96f7d5d...3fb9`/`8ee69cba...e56e`. It resumed epoch 8 and remains behind
  LR `8e-5` best `0.5833389060125157`; `1218344` is healthy in epoch-7
  generation and LR `1.5e-4` `1218345` remains pending. Two A100-40GB
  active, no A100-80GB/L40S; quota home `33.7%`, scratch `37.2%`; Kombuys
  untouched, held-out `0`, winners `0/8`, Mono not started, Sheet E/F/G
  blank.

- At 06:50 SAST, clean NER LR `8e-5` job `1218344` improved modestly at
  epoch 6: Tsn/Xho/Zul F1 `0.5696314002/0.5751914242/0.6051938937`, exact
  mean `0.5833389060125157`, matching newly retained `checkpoint-3246`.
  Its 192-row artifact is all nonempty with debug/state SHA-256
  `f41dac64...bbcb`/`66cf5674...d762`. It resumed epoch 7 and provisionally
  leads LR `3e-5` best `0.42604881329910455`; `1218343` is healthy in
  epoch-7 generation and LR `1.5e-4` `1218345` remains pending. Two
  A100-40GB active, no A100-80GB/L40S; quota home `33.7%`, scratch `37.2%`;
  Kombuys untouched, held-out `0`, winners `0/8`, Mono not started, Sheet
  E/F/G blank.

- At 06:20 SAST, clean NER LR `3e-5` job `1218343` improved modestly at
  epoch 6: Tsn/Xho/Zul F1 `0.4330256922/0.4061216105/0.4389991372`, exact
  mean `0.42604881329910455`, matching newly retained `checkpoint-3246`.
  Its 192-row artifact is all nonempty with debug/state SHA-256
  `518cd110...400f`/`8d197eeb...931e`. It resumed epoch 7 and remains behind
  LR `8e-5` best `0.5766600986590078`; `1218344` is healthy in epoch-6
  generation and LR `1.5e-4` `1218345` remains pending. Two A100-40GB
  active, no A100-80GB/L40S; quota home `33.7%`, scratch `37.2%`; Kombuys
  untouched, held-out `0`, winners `0/8`, Mono not started, Sheet E/F/G
  blank.

- At 05:20 SAST, clean NER LR `8e-5` job `1218344` reached epoch-5 Tsn/Xho/Zul
  F1 `0.5732620321/0.5689924991/0.5877257648`, exact mean
  `0.5766600986590078`, matching newly retained `checkpoint-2705`. Its
  192-row artifact is all nonempty with debug/state SHA-256
  `0b0eec7f...57d8`/`a9ae28ab...83ed`. It resumed epoch 6 and provisionally
  leads LR `3e-5` best `0.41520939686372715`; `1218343` is healthy in its
  epoch-6 callback and LR `1.5e-4` `1218345` remains pending. Two A100-40GB
  active, no A100-80GB/L40S; quota home `33.7%`, scratch `37.2%`; Kombuys
  untouched, held-out `0`, winners `0/8`, Mono not started, Sheet E/F/G
  blank. The repaired, steadily rising NER curve confirms the prior near-zero
  adapters were trained under the invalid unshifted label-smoothed objective.

- At 04:50 SAST, clean NER LR `3e-5` job `1218343` improved at epoch 5:
  Tsn/Xho/Zul F1 `0.4270624415/0.3864130147/0.4321527344`, exact mean
  `0.41520939686372715`, matching newly retained `checkpoint-2705`. Its
  192-row artifact is all nonempty with debug/state SHA-256
  `d5c7a903...3afd`/`fb0d7b8a...1338`. It resumed epoch 6 and still trails
  LR `8e-5` best `0.5335878171485439`. Job `1218344` is healthy in epoch-5
  generation after loss `0.34560057345819295`; LR `1.5e-4` `1218345`
  remains pending. Two A100-40GB active, no A100-80GB/L40S; quota home
  `33.7%`, scratch `37.3%`; Kombuys untouched, held-out `0`, winners `0/8`,
  Mono not started, Sheet E/F/G blank.

- At 04:20 SAST, clean NER LR `8e-5` job `1218344` improved at epoch 4:
  Tsn/Xho/Zul F1 `0.5317041801/0.5205521316/0.5485071398`, exact mean
  `0.5335878171485439`, matching newly retained `checkpoint-2164`. Its exact
  192-row artifact is all nonempty with debug/state SHA-256
  `73e758f0...9c76`/`7c3910ab...1bd2`. It resumed epoch 5 and provisionally
  leads LR `3e-5` best `0.34181699883258126`. Job `1218343` is healthy in
  epoch-5 generation after loss `0.4703660702616752`; LR `1.5e-4` `1218345`
  is resource-pending. Two A100-40GB active, no A100-80GB/L40S; quota home
  `33.7%`, scratch `37.2%`; Kombuys untouched, held-out `0`, winners `0/8`,
  Mono not started, Sheet E/F/G blank.

- At 03:50 SAST, clean NER LR `3e-5` job `1218343` materially improved at
  epoch 4: Tsn/Xho/Zul F1 `0.3325619070/0.3274374255/0.3654516640`, exact
  mean `0.34181699883258126`, matching newly retained `checkpoint-2164`.
  Its exact 192-row artifact is all nonempty with debug/state SHA-256
  `7860c239...9fe8`/`e68b452e...e15e`. It resumed epoch 5 and still trails
  LR `8e-5` best `0.4557537887462699`. Job `1218344` remains healthy in
  epoch-4 generation after loss `0.35859204260390043`; LR `1.5e-4`
  `1218345` is resource-pending. Two A100-40GB active, no A100-80GB/L40S;
  quota home `33.7%`, scratch `37.2%`; Kombuys untouched, held-out `0`,
  winners `0/8`, Mono not started, Sheet E/F/G blank.

- At 02:50 SAST, clean NER LR `8e-5` job `1218344` improved again at epoch
  3: Tsn/Xho/Zul span micro-F1
  `0.4305254016/0.4578999858/0.4788359788`, exact mean
  `0.4557537887462699`, matching newly retained `checkpoint-1623`. Its
  exact 192-row artifact is all nonempty with debug/state SHA-256
  `3a7c37e1...fde5`/`46884992...a23e`. It provisionally leads LR `3e-5`
  best `0.22690209232181432`, but no winner is frozen. Job `1218344` resumed
  epoch 4; `1218343` entered its epoch-4 callback; LR `1.5e-4` `1218345`
  remains resource-pending. Two A100-40GB active, no A100-80GB/L40S; quota
  home `33.7%`, scratch `37.2%`; Kombuys untouched, held-out `0`, winners
  `0/8`, Mono not started, Sheet E/F/G blank.

- At 02:20 SAST, clean NER LR `3e-5` job `1218343` produced a valid epoch-3
  mean F1 `0.22690209232181432` from Tsn/Xho/Zul
  `0.2194001621/0.2188571907/0.2424489242`, exactly matching newly retained
  `checkpoint-1623`. This is only `0.0003750681` above epoch 2 and remains
  below LR `8e-5`'s provisional `0.34910683094139167`. Its 192-row artifact
  has `64/64/64` coverage and debug/state SHA-256
  `95b59051...0519`/`5e6e5d5e...97a9`. Job `1218343` continued under frozen
  patience; job `1218344` is healthy in epoch-3 generation after loss
  `0.3914661166393181`; LR `1.5e-4` `1218345` remains resource-pending.
  Two A100-40GB active, no A100-80GB/L40S; quota home `33.7%`, scratch
  `37.3%`; Kombuys untouched, held-out `0`, winners `0/8`, Mono not started,
  Sheet E/F/G blank.

- At 01:50 SAST, clean NER LR `8e-5` job `1218344` improved strongly at
  epoch 2: Tsn/Xho/Zul span micro-F1
  `0.3422339992/0.3469783855/0.3581081081`, exact mean
  `0.34910683094139167`, matching newly retained `checkpoint-1082`. Its
  192-row artifact has `64/64/64` coverage, all raw predictions nonempty,
  and debug/state SHA-256 `753d0d91...9922`/`e2cfaff0...fdb5`. This
  provisionally leads LR `3e-5` best `0.2265270242388299`, but no winner is
  frozen. Job `1218344` resumed epoch 3; job `1218343` is healthy in its
  terminal epoch-3 callback after health-only loss `0.9596275811744889`;
  LR `1.5e-4` `1218345` remains resource-pending. Two A100-40GB GPUs active,
  no A100-80GB/L40S. Quota home `33.7%`, scratch `37.2%`; Kombuys untouched,
  held-out `0`, Multilingual winners `0/8`, Mono not started, Sheet E/F/G
  blank.

- At 01:20 SAST, clean NER LR `3e-5` job `1218343` improved at epoch 2:
  full-grid Tsn/Xho/Zul span micro-F1
  `0.2412626832/0.2035084161/0.2348099734`, exact mean
  `0.2265270242388299`, matching newly retained `checkpoint-1082` state.
  Its 192-row artifact has `64/64/64` coverage, all raw predictions nonempty,
  and debug/state SHA-256 `5700d025...1da3`/`14cab016...502`. This is
  provisional only. Job `1218343` resumed epoch 3; LR `8e-5` job `1218344`
  remains healthy in its epoch-2 callback after health-only validation loss
  `0.45399808635498956`; LR `1.5e-4` `1218345` is resource-pending. Two
  A100-40GB GPUs are active, no A100-80GB/L40S. Quota home `33.7%`, scratch
  `37.2%`; Kombuys untouched, held-out `0`, Multilingual winners `0/8`, Mono
  not started, and Sheet E/F/G blank.

- At 00:50 SAST, clean NER jobs `1218343/1218344` both reached epoch-2 step
  `1082` and remain healthy in validation callbacks on two A100-40GB GPUs.
  LR `3e-5` completed exact 10,760-row health-only validation loss
  `1.221553873395388`; LR `8e-5` reached the boundary with causal token
  accuracy `0.8979236841`. No step-1082 F1 artifact or new selection exists;
  provisional epoch-1 ordering is unchanged. LR `1.5e-4` job `1218345`
  remains resource-pending. Quota is home `33.7%`, scratch `37.2%`; no
  A100-80GB/L40S, Kombuys, held-out, Sheet, or HF activity. Trusted base is
  `16/16`, frozen Multilingual winners `0/8`, held-out `0`, Monolingual not
  started, and Sheet E/F/G blank.

- At 00:22 SAST on 11 August, clean NER LR `8e-5` job `1218344` produced a
  scientifically valid step-541 artifact: exact 192-row Tsn/Xho/Zul coverage
  `64/64/64`, per-language span micro-F1
  `0.1328938975/0.1688608753/0.1756278443`, and arithmetic mean
  `0.15912753902811083`, matching retained `checkpoint-541` state exactly.
  Debug/state SHA-256 values are `1b6c9055...1647` and `ae6e68cb...c9a`.
  This provisionally exceeds LR `3e-5` epoch-1 mean `0.12351486799421617`,
  but no winner is frozen. Jobs `1218343/1218344` remain healthy on two
  A100-40GB GPUs at the epoch-2 callback boundary and near step `634/8115`;
  LR `1.5e-4` `1218345` is resource-pending. Quota is home `33.7%`, scratch
  `37.2%`; no A100-80GB/L40S, Kombuys, held-out, Sheet, or HF activity.
  Trusted base remains `16/16`, frozen Multilingual winners `0/8`, held-out
  tests `0`, Monolingual not started, and Sheet E/F/G blank.

- At 23:50 SAST clean NER LR `3e-5` job `1218343` produced the first
  scientifically valid repaired NER metric: exact 192-row Tsn/Xho/Zul coverage
  `64/64/64`, per-language span micro-F1
  `0.1254826255/0.0953115625/0.1497504160`, arithmetic mean
  `0.12351486799421617`, matching retained `checkpoint-541` state exactly.
  Debug/state SHA-256 values are `d938dce7...9d80` and `86dbed28...a1e8`.
  This is provisional within LR-0; no cross-LR winner is frozen. LR `8e-5`
  job `1218344` also completed shifted epoch-1 training at causal token
  accuracy `0.7557807088` and exact 10,760-row validation loss
  `1.04413329624332`; it is in corrected generation with first F1 ETA
  `00:05--00:20`. LR `1.5e-4` `1218345` remains resource-pending. Exactly two
  A100-40GB GPUs are owned; no A100-80GB/L40S. HEX quota home `33.7%`, scratch
  `37.2%`; no held-out, Kombuys, Sheet, or HF activity. Trusted state remains
  base `16/16`, frozen Multilingual winners `0/8`, held-out `0`, E/F/G blank.

- At 23:20 SAST clean NER LR `3e-5` job `1218343` has empirically confirmed
  the repaired objective: epoch-1 causal next-token training accuracy
  `0.7241757959`, loss `2.5004`, finite gradient norm `7.2572`, followed by
  exact 10,760-row Tsn/Xho/Zul validation loss `1.528678082888011`
  (`1.5848/1.3959/1.6249` per language). It is healthy in corrected NER
  generation at batch size 64; no span-F1 artifact or winner exists yet, ETA
  roughly `23:30--23:45`. LR `8e-5` job `1218344` started `23:01:02`, healthy
  around `338/8115` on a second A100-40GB; its 691-file execution-manifest
  SHA-256 is `e24806a4...027c` and corrected trainer hash matches
  `921327f8...97b1`. LR `1.5e-4` `1218345` remains resource-pending. HEX quota
  is home `33.7%`, scratch `37.2%`; no A100-80GB/L40S, held-out, Kombuys,
  Sheet, or HF activity. Trusted state remains base `16/16`, Multilingual
  winners `0/8`, held-out `0`, Sheet E/F/G blank.

- At 22:50 SAST clean causal-shift NER LR `3e-5` job `1218343` is healthy on
  one A100-40GB after starting `22:32:15`, approximately step `325/8115` at
  `3.02 s/step`. Its job-created 691-file execution manifest verifies at
  SHA-256 `cadacc9f...1ff` and contains corrected trainer SHA-256
  `921327f8...97b1`, Transformers `4.57.3`, TRL `0.26.2`, and FLA `0.5.1`.
  Runtime confirms canonical `GatedDeltaNetForCausalLM`, label smoothing
  `0.05`, assistant-only loss, and `4,649,088` trainable LoRA parameters; no
  targeted fault exists. First complete epoch-1 validation artifact is
  tentatively due `23:30--23:50 SAST`. NER LR `8e-5` `1218344` is
  resource-pending with projected `02:53:11` start, and LR `1.5e-4` `1218345`
  is priority-pending without ETA. HEX quota is home `33.7%`, scratch `37.2%`;
  no A100-80GB/L40S, held-out, Kombuys, Sheet, or HF activity. Trusted state
  remains base `16/16`, Multilingual winners `0/8`, held-out tests `0`, Sheet
  E/F/G blank.

- At 19:56 SAST the causal-label-shift correction is locally verified and
  immutably deployed. Preregistration SHA-256 is `29ddcaa0...fc13`; the
  regression test first failed with unshifted loss `7.9213` versus shifted
  `0.3213`, then passed after registering external FLA
  `GatedDeltaNetForCausalLM` in Transformers' causal-LM name mapping. Focused
  Ruff passes and the full suite is `113 passed`. Trainer/test SHA-256 values
  are `921327f8...97b1` and `54dd93c2...3a`; the read-only 691-file HEX
  snapshot is
  `/home/lmbanr001/masters/sallm_snapshots/pure-gdn-hpo-causalshift-20260810-921327f8`
  with deployment-manifest SHA-256 `821d9b61...aff6`. Invalid POS jobs
  `1217737/1217738/1217739` were cancelled at `19:53:09`, preserving all
  artifacts. Clean canonical-base validation-only NER replacements are
  `1218343` (LR `3e-5`), `1218344` (LR `8e-5`), and `1218345` (LR `1.5e-4`),
  initially priority-pending on A100-40GB `gpu:ampere`. HEX quota is home
  `33.6%`, scratch `37.1%`; no A100-80GB/L40S, held-out, Kombuys, or Sheet
  access occurred. At 19:56 SAST Slurm projected `1218343` to start at
  `02:53:11` on 11 August; `1218344/1218345` had no start estimates.
  Operational base is trusted `16/16`, scientifically frozen Multilingual
  winners `0/8`, held-out tests `0`, Sheet E/F/G blank.

- Critical 2026-08-10 causal-loss audit supersedes the earlier NER freeze and
  POS provisional interpretation. The shared pure-GDN validation launcher
  enables label smoothing (`0.05`), but Transformers `4.57.3` recognizes
  causal models by its internal class-name mapping. External FLA
  `GatedDeltaNetForCausalLM` is absent, so `Trainer.compute_loss` calls
  `LabelSmoother` without `shift_labels=True`: position-*t* logits are trained
  against position-*t* labels instead of causal position-*t+1* labels. The
  corrected evaluators are shifted correctly, making the low NER/POS metrics
  valid measurements of incorrectly trained adapters, not trustworthy
  architecture results. NER `1217444/1217445/1217446` and POS
  `1217737/1217738/1217739` are scientifically quarantined; frozen
  Multilingual winners revert to `0/8`, held-out tests remain `0`, and no
  Monolingual work is active or authorized. At 19:21 SAST the POS jobs were
  still operationally healthy at epoch-2 callback progress `1050/1800`,
  `350/1800`, and `300/1800` on exactly three A100-40GB GPUs. Quota was home
  `33.6%`, scratch `37.1%`; no A100-80GB/L40S work, held-out access, or Sheet
  write occurred. A preregistered causal-shift trainer correction, regression
  test, immutable redeployment, and validation-only rerun are now mandatory.

- Read-only base/pretraining dataset-lineage audit completed 2026-08-10. No
  retained LLaMA/Mamba/xLSTM/GDN pretraining config names the broken
  2,487,635-row raw `uctnlp/mzansi-text` train. Later bases use full
  `anrilombard/mzansi-text-tokenized` (now `uctnlp`, Hub/cache revision
  `5022487e437b120845b758fa48f8a28fd25ce58d`, exact splits
  `3,943,584/19,379/19,341`), while LLaMA-125M and early attempts use the
  older fixed-2048 local `sallm_processed` derivative of the complete filtered
  source. Thus WURA, ParaCrawl, and mC4 xh/zu coverage is not the cause of low
  NER/POS. Dataset representation remains a LLaMA-125M comparability confound.
  Hybrid GDN job `997801` and the initial pure-GDN run were fresh; pure-GDN
  `1179847` resumed only its own `checkpoint-29000`. Exact public Mamba parent
  checkpoint and exact deleted `sallm_processed` row counts/fingerprint remain
  unrecoverable. Full evidence is in
  `/Users/anrilombard/Desktop/Masters/Notes/mzansitext_huggingface_release_audit_2026-08-10.md`.

- At 18:52 SAST, all three corrected POS trials have complete, independently
  reconciled epoch-1 validation artifacts. Token accuracy is `0.10956738155126096`
  for `1217737` (LR `3e-5`), `0.23358808107576032` for `1217738` (LR `8e-5`),
  and `0.23815571879370404` for `1217739` (LR `1.5e-4`); exact recomputed
  12-cell means are floating-point equivalent. LR-2 is the provisional leader,
  not a frozen winner. Its artifact/trainer-state SHA-256 values are
  `4eda17d5...65c7` and `f25164ac...aea`; LR-1 values are
  `51e12eac...daf8` and `dcf5b848...da3e5`. All retain `checkpoint-283` and
  remain healthy: LR-0 is `700/1800` in epoch-2 constrained validation, while
  LR-1/LR-2 resumed training. Owned hardware is exactly three A100-40GB jobs;
  no A100-80GB/L40S or held-out work. HEX quota is home `33.6%`, scratch
  `37.1%`; Kombuys is read-only idle (RTX 5090 `10 MiB/0%`, RTX 3080 Ti
  `1 MiB/0%`, scratch `62%`). Base `16/16`, winners `1/8`, held-out `0`,
  Sheet E/F/G blank.

- At 18:20 SAST, POS LR-1/LR-2 `1217738/1217739` remain healthy at
  `1550/1800` and `1500/1800` in their first constrained callbacks; first
  metrics are expected around `18:40--18:45`. LR-0 `1217737` completed
  epoch-2 loss evaluation (`12.60320095486111`) and is `250/1800` in its
  second constrained callback, projected near `20:15--20:25`. No fault,
  held-out, or Sheet access occurred. Three owned A100-40GB only; HEX quota
  home `33.6%`, scratch `37.0%`; latest Kombuys state read-only idle. Base
  `16/16`, winners `1/8`, held-out `0`, Sheet E/F/G blank.

- At 17:49 SAST, POS LR-0 `1217737` completed its first corrected constrained
  validation: exact `1,800` rows / 12 language-prompt cells, arithmetic-mean
  token accuracy `0.10956738155126096`, retained `checkpoint-283`. Independent
  recomputation gives `0.10956738155126099`; artifact/trainer-state SHA-256
  values are `e39dbac8...5773` and `6d17c62b...f5ea`. LR-0 resumed epoch-2
  training. LR-1 `1217738` and LR-2 `1217739` remain healthy in epoch-1
  constrained validation at `1150/1800` and `1100/1800`, with first metrics
  expected around `18:40--18:50`. All three owned jobs remain A100-40GB only;
  no held-out or Sheet access. HEX quota is home `33.6%`, scratch `37.0%`;
  Kombuys read-only idle; base `16/16`, winners `1/8`, held-out `0`, Sheet
  E/F/G blank.

- At 17:19 SAST, POS `1217737/1217738/1217739` remain healthy in their first
  corrected constrained validation callbacks at `1450/1800`, `750/1800`, and
  `700/1800`. No selection metric exists yet and no fault marker appeared.
  Revised first-metric ETAs are approximately `17:45`, `18:35`, and `18:40`
  SAST. Owned hardware remains exactly three A100-40GB jobs with no
  A100-80GB/L40S or held-out work. HEX quota is home `33.6%`, scratch `37.0%`;
  latest Kombuys read-only state is idle. Base `16/16`, winners `1/8`,
  held-out `0`, Sheet E/F/G blank.

- At 16:49 SAST, all three clean POS retries `1217737/1217738/1217739` are
  healthy and running concurrently on the three permitted A100-40GB GPUs.
  Each completed epoch-1 training/loss evaluation over exact 1,800 validation
  rows and entered the preregistered constrained tuple/token-accuracy callback.
  Callback progress is LR-0 `1100/1800`, LR-1 `350/1800`, LR-2 `350/1800`;
  no runtime, provenance, coverage, CUDA, or NCCL fault is present. Execution
  manifest SHA-256 values are `a33ee863...1316`, `01935849...5e28`, and
  `3e6106f0...1280`. First epoch metrics are expected around `17:40` for LR-0
  and `18:35--18:50` for LR-1/LR-2; terminal ETAs remain output-dependent
  because patience-2 stopping requires complete callbacks. No A100-80GB/L40S
  or held-out work exists. HEX quota is home `33.6%`, scratch `37.0%`;
  Kombuys remains read-only idle; base `16/16`, winners `1/8`, held-out `0`,
  Sheet E/F/G blank.

- At 15:07 SAST, corrected NER grid is terminal and scientifically frozen.
  LR-2 `1217446` completed `0:0` at `14:42:33` in `03:50:04`, retained
  `checkpoint-541`, and scored corrected mean per-language span micro-F1
  `0.0`; all three LRs tie at `0.0`, so the frozen lower-LR/earlier-checkpoint
  rules select `1217444`, LR `3e-5`, checkpoint 541. Selection artifact
  SHA-256 is `dfbbae17d7a5f79bed113e1d0f64e298decd691503aa65797383fce45071d199`.
  Multilingual winners are now `1/8`.
- POS jobs `1217704/1217705` failed before model/data access in `00:00:45` /
  `00:00:39`: the sparse POS config lacks `adam_beta2`, while the shared
  launcher used a strict Hydra override. The root correction uses `++` for
  the five optional training keys and adds a validated attempt tag so failed
  directories remain untouched. Local syntax, 21 dry-runs, invalid-tag guard,
  and POS Hydra composition pass. Final read-only 691-file snapshot is
  `/home/lmbanr001/masters/sallm_snapshots/pure-gdn-hpo-posretryfix-20260810-20dd9439`;
  launcher/deployment-manifest SHA-256 values are `20dd9439...fbbe` and
  `8bb57c68...b2c3`. Clean POS retries are `1217737/1217738/1217739`; LR-0
  `1217737` is healthy training on one A100-40GB and LR-1/LR-2 are pending.
  No A100-80GB/L40S or held-out work occurred. HEX quota is home `33.6%`,
  scratch `37.0%`; Kombuys remains read-only idle; Sheet E/F/G remain blank.

- At 14:31 SAST, NER LR-2 `1217446` remains healthy in the epoch-3 corrected
  generation callback after step `1623`; the complete loss pass covers all
  `10,760` rows at loss `13.713622139289033`, and all three automatic
  generation batch-size probes have completed. The terminal 192-row debug
  artifact and selection metric are not yet saved, so NER remains unfrozen
  with ETA about `14:45--15:00 SAST`. POS `1217704/1217705` remain pending;
  these are the only three owned A100-40GB jobs. HEX quota is home `33.6%`,
  scratch `37.0%`; Kombuys remains read-only and idle, Sheet E/F/G blank,
  held-out adapter tests `0`.

- At 14:02 SAST, NER LR-2 `1217446` is healthy inside its epoch-3 full
  validation callback at step `1623/8115`; the loss pass has completed Tsn
  and Xho and is advancing through Zulu. Epochs 1--2 remain corrected mean
  per-language span micro-F1 `0.0`, so no cross-LR winner is frozen yet.
  Conditional terminal ETA is now roughly `14:40--15:00 SAST`. POS
  `1217704/1217705` remain resource/priority pending. These are the only
  owned jobs, all requesting one A100-40GB `gpu:ampere`; no A100-80GB/L40S
  work or held-out access occurred. HEX quota is home `33.6%`, scratch
  `37.0%`; Kombuys remains read-only and idle (RTX 5090 `10 MiB/0%`, RTX
  3080 Ti `1 MiB/0%`, scratch `61%`). Sheet E/F/G remain blank.

- At 13:43 SAST, NER LR-2 job `1217446` remains healthy and training toward
  its epoch-3 validation callback after completing epoch 2 at corrected mean
  per-language span micro-F1 `0.0`; its step-1082 debug artifact now exists.
  LR-0/LR-1 `1217444/1217445` remain terminal at `0.0`. A late LR-2 recovery
  is possible but unlikely because both completed epochs fail to emit
  parseable `LABEL: entity` spans. Conditional terminal ETA remains
  `14:35--14:50 SAST`. POS jobs `1217704/1217705` are pending; owned state is
  one running plus two pending A100-40GB jobs. HEX quota is home `33.6%`,
  scratch `37.0%`; no held-out work or Sheet write occurred.

- At 13:35 SAST, NER LR-0/LR-1 jobs `1217444/1217445` completed `0:0` in
  `03:54:28/03:49:07`, early-stopped after epoch 3, retained `checkpoint-541`,
  and saved hashed final adapters. Both validation-only best mean per-language
  span micro-F1 values are `0.0`; their epoch-3 192-row artifacts are nonempty
  but unparseable model outputs. NER LR-2 `1217446` remains healthy with
  conditional ETA `14:35--14:50`; NER winner is therefore not frozen.
  Freed capacity was used for independent POS validation-only LR `3e-5` job
  `1217704` and LR `8e-5` job `1217705`, both submitted from the same immutable
  snapshot and currently priority-pending. Owned state is one running plus two
  pending A100-40GB jobs; no A100-80GB/L40S or held-out work occurred. HEX
  quota is home `33.6%`, scratch `37.0%`; Kombuys is read-only and idle. Base
  trust `16/16`, Multilingual winners `0/8`, held-out adapter tests `0`, Sheet
  E/F/G blank.

- At 13:01 SAST, NER LR-0/LR-1 jobs `1217444/1217445` are healthy in epoch-3
  full callbacks at step `1623/8115`, with validation-only losses
  `13.51897326614777/13.434167707365242`; LR-2 `1217446` is healthy in its
  epoch-2 callback at step `1082/8115`, loss `13.502328502555763`. Primary F1
  artifacts remain pending. Frozen early-stopping patience is `2` with
  threshold `0.001`; correcting the prior patience-3 assumption gives
  conditional ETAs `13:20--13:30` for LR-0/1 and `14:35--14:50` for LR-2 if
  F1 remains `0.0`. No winner or held-out access exists; the three-job cap
  still blocks POS. Hardware is A100-40GB only, HEX quota home `33.6%`, scratch
  `36.9%`, Kombuys read-only and idle. Base trust `16/16`, winners `0/8`,
  held-out adapter tests `0`, Sheet E/F/G blank.

- At 12:31 SAST, NER `1217444/1217445/1217446` resumed training at steps
  `1568/1583/953` with no targeted fault. LR-0/LR-1 saved `checkpoint-1082`
  but retained earlier `checkpoint-541`, because epoch-2 corrected mean
  per-language span micro-F1 remained tied at `0.0`. Their exact 192-row epoch-2
  debug artifacts are nonempty but unparseable model outputs. LR-2 saved
  `checkpoint-541` at F1 `0.0`; its 192 predictions are all empty despite the
  identical corrected evaluator yielding nonempty LR-0/LR-1 outputs, so this is
  a high-LR model result and not a retry trigger. Conditional early-stop ETAs
  are roughly 15:00 SAST for LR-0/1 and 16:20 for LR-2 if F1 does not improve.
  POS remains blocked by the three-job cap. No held-out or A100-80GB/L40S work
  occurred; HEX quota is home `33.6%`, scratch `36.9%`; Kombuys is read-only
  and idle. Base trust `16/16`, winners `0/8`, held-out adapter tests `0`,
  Sheet E/F/G blank.

- At 12:02 SAST, NER `1217444/1217445/1217446` remain healthy inside full
  corrected callbacks on three A100-40GB GPUs at steps `1082/1082/541`.
  Epoch-1 LR-0/LR-1 debug artifacts each contain exact Tsn/Xho/Zul coverage
  `64/64/64`; every raw prediction is nonempty, but all normalize to no
  parseable NER span, scientifically validating selection F1 `0.0` as a model
  result rather than the old terminal-EOS evaluator failure. Artifact SHA-256
  values are `c48a7628...1126e` and `970e76c7...5d395`. No new retained
  checkpoint or winner exists yet. Callback completion and the three-job cap
  block POS. No held-out access or A100-80GB/L40S work occurred. HEX quota is
  home `33.6%`, scratch `36.8%`; Kombuys is read-only and idle. Base trust
  `16/16`, winners `0/8`, held-out adapter tests `0`, Sheet E/F/G blank.

- At 11:32 SAST, NER `1217444/1217445` reached epoch 2 (`1082/8115`) with
  validation-only losses `13.459356485246282/13.291437107922862`; `1217446`
  reached epoch 1 (`541/8115`) with loss `13.726716426579925`. All three are
  healthy in corrected full-metric callbacks on A100-40GB. LR-0/LR-1 retained
  `checkpoint-541` with provisional `best_metric=0.0`. Source wiring re-audit
  confirms `eval_all_f1` is the arithmetic mean of independently computed
  per-language span micro-F1 values, matching the preregistration. No winner is
  frozen. POS remains blocked by the three-job cap; no held-out access occurred.
  No A100-80GB/L40S work exists; HEX quota is home `33.6%`, scratch `36.8%`;
  Kombuys is read-only and idle. Base trust `16/16`, winners `0/8`, held-out
  adapter tests `0`, Sheet E/F/G blank.

- At 11:02 SAST, the full corrected NER grid is running on three A100-40GB
  GPUs. Jobs `1217444/1217445` resumed training after first validation and
  reached `787/8115` and `766/8115`; LR `1.5e-4` job `1217446` started at
  `10:52:29` and reached `147/8115`. Its verified 691-file execution-manifest
  SHA-256 is
  `c212b9b8fea653d52c033cfa6b91d81e3770b734ce65a567a7f6fd0d5695248e`;
  existing LR-0/LR-1 manifest hashes remain verified. All three use the
  canonical BF16 pure-GDN checkpoint and `4,649,088`-parameter LoRA, with no
  targeted fault. Training-only ETA is roughly `17:15--17:55 SAST`, but full
  callbacks and early stopping make terminal ETA uncertain. The three-job cap
  blocks POS submission. No A100-80GB/L40S work exists; HEX quota is home
  `33.6%`, scratch `36.8%`; Kombuys is read-only and idle. Base trust `16/16`,
  Multilingual winners `0/8`, held-out adapter tests `0`, Sheet E/F/G blank.

- At 10:32 SAST, NER jobs `1217444`/`1217445` remain healthy on two
  A100-40GB GPUs and completed their first loss pass over all `10,760`
  validation rows after step `541/8115`. Validation-only overall losses are
  `12.890100415311338` for LR `3e-5` and `13.469093793564127` for LR `8e-5`;
  both then entered corrected full NER metric/generation, so no primary span
  micro-F1, checkpoint selection, or winner exists yet. Job `1217446` (LR
  `1.5e-4`) remains resource-pending with a non-binding `13:55 SAST` Slurm
  estimate. All three owned jobs are A100-40GB only; no POS submission or
  held-out access occurred. HEX quota is home `33.6%`, scratch `36.7%`;
  Kombuys remains read-only and idle. Base trust `16/16`, Multilingual winners
  `0/8`, held-out adapter tests `0`, Sheet E/F/G blank.

- At 10:03 SAST, corrected NER validation jobs `1217444` (LR `3e-5`) and
  `1217445` (LR `8e-5`) are healthy on two A100-40GB GPUs. Both reached their
  first epoch validation callback after step `541/8115`, with finite losses
  and gradients and no targeted runtime/provenance/coverage fault. Their
  repetitive diagnostic generations are preserved model outputs, not retry
  triggers. LR `1.5e-4` job `1217446` remains resource-pending with a dynamic
  Slurm start estimate of `13:55 SAST`. The three-job cap is occupied, so POS
  is not yet submitted. HEX quota is home `33.6%`, scratch `36.7%`; no owned
  A100-80GB/L40S work exists. Kombuys remains read-only and idle with RTX 5090
  `10 MiB/0%`. Corrected base is trusted `16/16`, Multilingual winners `0/8`,
  held-out adapter tests `0`, and Sheet E/F/G remain blank.

- User ratified AfriHG's one whitespace-only Xhosa output as a valid model
  miss, with no rerun. Corrected base is now scientifically trusted `16/16`.
  Sheet rows 42--43 C/D are updated and verified at `7.6220/5.6752 chrF`;
  E/F/G remain blank. First NER HPO submissions `1217441--1217443` failed
  pre-training because immutable source had no venv and system Python 3.9
  cannot import `datetime.UTC`; no metric/artifact was produced. Minimal
  runtime-venv launcher correction is frozen in read-only snapshot
  `pure-gdn-hpo-runtimefix-20260810-05a4a619`, 691/691 deployment-manifest
  SHA-256 `a234fe873b665acafae3a0eba9fec087d1cb1cf0b6d0027bb1651d583b3c4c38`.
  Unchanged NER retries are `1217444/1217445/1217446`; LR-0 `1217444` is
  running on A100-40GB with its 691-file execution manifest, canonical BF16
  pure-GDN checkpoint, and `4,649,088`-parameter LoRA verified, while LR-1/LR-2
  are pending. HPO winners remain `0/8`; held-out adapter tests `0`.

- AfriHG `1216495_15` completed `0:0` in `05:43:37` on one A100-40GB.
  Frozen held-out coverage and token contract pass: Xhosa `1,305`, Zulu
  `1,776`, total `3,081`; exactly one leading BOS, no terminal EOS/chat
  markers, canonical BF16 zero-shot checkpoint, no adapter/merge, no runtime
  fault. Zulu has zero empty predictions; Xhosa row 1,099 decoded to one space
  and normalizes empty. This is a valid model miss, not the corrected
  evaluator failure, and cannot justify a held-out-driven rerun. Because the
  automation's literal zero-empty gate is not met, operational base is
  `16/16` but scientific ratification remains `15/16` pending adjudication.
  HPO stays `0/8`, held-out adapter tests `0`, no HEX GPU job is active,
  Sheet rows 42--43 remain quarantined, and E/F/G remain blank.

- At 06:29 SAST, AfriHG `1216495_15` remains healthy at `03:37:56` on the
  sole owned A100-40GB. Xhosa completed all 1,305 held-out test rows and saved
  artifacts; Zulu started with 1,776 rows and batch size 64. The prior 1,777
  Zulu/3,082 total gate was validation-only: read-only source CSV parsing
  confirms held-out test coverage is 1,305 Xhosa + 1,776 Zulu = 3,081, while
  dev/validation is 1,305 + 1,777 = 3,082. This count-only reconciliation
  changes no protocol or selection. Final trust remains `15/16` until job and
  artifacts pass terminal audit; HPO `0/8`, held-out adapter tests `0`, Sheet
  rows 42--43 quarantined, E/F/G blank.

- At 04:30 SAST, AfriHG `1216495_15` remains healthy at `01:39:59` on the
  sole owned A100-40GB, with fresh Xhosa generation activity at `04:17:13`
  and no runtime fault marker or summary yet. Conditional ETA remains near
  `08:41 SAST`. Corrected-base trust is `15/16`, HPO `0/8`, held-out adapter
  tests `0`, Sheet rows 42--43 remain quarantined, and E/F/G remain blank.

- At 03:58 SAST, AfriHG `1216495_15` remains healthy at `01:08:06` on the
  only owned A100-40GB, with fresh Xhosa generation activity at `03:32:27`
  and no runtime fault marker. Conditional ETA remains near `08:41 SAST`.
  Corrected-base trust is `15/16`, HPO `0/8`, held-out adapter tests `0`,
  Sheet rows 42--43 remain quarantined, and E/F/G remain blank.

- At 03:28 SAST, final base lane AfriHG `1216495_15` remains healthy at
  `00:38:02` on the only owned A100-40GB. It prepared all 1,305 Xhosa test
  examples, selected generation batch size 64, and logged fresh progress at
  `03:21:03`; no runtime fault marker exists. Conditional ETA remains near
  `08:41 SAST`. Corrected-base trust remains `15/16`, HPO `0/8`, held-out
  adapter tests `0`, and Sheet E/F/G blank.

- Corrected T2X Xhosa `1216462_14` completed `0:0` in `00:45:42` on
  A100-40GB. All 378 rows were evaluated with zero empty predictions, 235
  unique predictions, exactly one leading BOS, no terminal EOS, and no chat
  markers. chrF is `2.4064287155136577`; summary SHA-256 is
  `ecc78c42cb030ed38d69d377877d4f5d9a21dfa012f19687f9da389148564ab0`.
  Canonical Sheet row 41 was updated and verified with blank E/F/G.
  Scientifically trusted corrected-base progress is now `15/16`.
- AfriHG `1216495_15` started at `02:49:44 SAST` and is healthy on the only
  owned A100-40GB after preparing all 1,305 Xhosa test examples. Its
  runtime-only conditional ETA is approximately
  `08:41 SAST`; no A100-80GB/L40S overlap exists. Multilingual HPO remains
  `0/8` and held-out adapter evaluation remains blocked.

- At 02:28 SAST, T2X Xhosa `1216462_14` remains healthy at `00:21:48` after
  preparing all 378 test examples; conditional completion remains near
  `02:56 SAST`. AfriHG `1216495_15` is resource-pending with unknown ETA.
  These are the only owned jobs, both on the required A100-40GB family; no
  A100-80GB/L40S overlap exists. Corrected-base trust remains `14/16`, HPO
  `0/8`, held-out adapter tests `0`, and Sheet E/F/G blank.

- Corrected Belebele Zulu `1216461_13` completed `0:0` in `00:02:06`; all
  P1--P5 accuracies tie at `0.2288888888888889`, with verified summary/raw
  SHA-256 values `be74eaf0...b06d77e` and `a7d28b7c...9a81d37`.
  Canonical Sheet row 21 is updated and verified with E/F/G blank. Trusted
  corrected-base progress is now `14/16`.
- Final base lanes are in flight on the required A100-40GB family: T2X Xhosa
  `1216462_14` is healthy after preparing all 378 test examples, with a
  conditional ETA near `02:56 SAST`; AfriHG `1216495_15` is
  resource-pending with unknown ETA. There is no A100-80GB/L40S overlap.
  Multilingual HPO remains `0/8` and held-out adapter evaluation remains
  blocked.

- Corrected Belebele Xhosa `1216460_12` completed `0:0` in `00:02:08` with
  all P1--P5 accuracies tied at `0.2288888888888889`; summary/raw SHA-256
  values are `c76c07dd...d304a4d` and `bee8ec8e...f2d62c`. Canonical Sheet
  row 20 was updated and verified with blank E/F/G. Scientifically trusted
  corrected-base progress is now `13/16`; Zulu `1216461_13` is running and
  T2X `1216462_14` is pending on A100-40GB. HPO remains `0/8` and blocked.

- At 02:02 SAST on 10 August, corrected Belebele Swati `1215945_9`, Tswana
  `1215946_10`, and Tsonga `1215947_11` are artifact-verified `0:0`; every
  P1--P5 accuracy is `0.2288888888888889`. Their summary/raw SHA-256 values
  reconcile, and canonical Sheet rows 23/22/27 were updated and verified with
  preserved formats and blank E/F/G. Scientifically trusted corrected-base
  progress is `12/16`; Multilingual winners remain `0/8`, and no held-out
  adapter test has run.
- The next frozen lanes were submitted individually from immutable snapshot
  `pure-gdn-prompt-correction-20260809-693fe42c`: Belebele Xhosa
  `1216460_12`, Belebele Zulu `1216461_13`, and T2X Xhosa `1216462_14`.
  At the first check Xhosa was running on A100-40GB and the other two were
  pending. All have the required `nlpgroup/a100/nlpgroup`, `gpu:ampere:1`,
  24-hour, eight-CPU envelope. No A100-80GB/L40S overlap exists; HEX quota is
  home `33.5%`, scratch `36.7%`. HPO and held-out evaluation remain blocked.

## Durable correction execution — A100 canary passed and base rerun started (2026-08-09)

- Corrected base lane 3 SIB-200 `1211675_3` completed `0:0` in `00:07:39`; summary SHA-256 `e16b70d127d44ce812b4596a37a9791b70bcec47f7ea0279e2ff75f84cc485b6`. Corrected lane 0 MasakhaNews `1211671_0` completed `0:0` in `00:11:28`; summary SHA-256 `52679f0d7c021324c29fed2f56b161de9ccb5cf8d770d21f7e93c3f6a473f2c3`. Both verified raw zero-shot BF16/no-adapter prompts with `apply_chat_template=false` and exactly one BOS. Scientifically trusted base progress is now `2/16`; HPO winners remain `0/8`.
- Canonical Sheet `Pure GatedDeltaNet Results` rows 2--3 and 10--15 were re-read, replaced from those verified artifacts, and re-read successfully with dates, headline values, all prompt metrics, means/ranges, exact artifact paths/hashes, immutable source provenance, and unchanged wrapping/date formats. Other column-D rows remain quarantined; E/F/G remain blank.
- Corrected lanes `1211758_1` MasakhaNER and `1211759_2` MasakhaPOS started healthy on A100-40GB using the same immutable snapshot and prefix. Both log canonical checkpoint, no adapter/merge, BF16, zero-shot, `apply_chat_template=False`, and `add_bos_token=True`, with no fault markers. AfriHG `1210866` remains provenance-only; owned state is exactly three A100-40GB jobs and no other GPU family.
- Prompt-contract canary `1211316` completed `0:0` in `00:09:14` on `srvrocgpu010`, one A100-40GB `gpu:ampere`. Result SHA-256 is `79beda323ddab1d8c14dc755756ac4f7d607e2f324439bbf7287dd00081ca20e`; execution-manifest SHA-256 is `6104df2251b5f6f09d361a189405cd466c09bf7a91851c4fe9c5c813a40163b3`. It verified the canonical pure-GDN model, exactly one leading BOS/no terminal EOS, raw base prompts without chat markers, corrected NER/POS, and exact General coverage `22,167`. HPO remains blocked until all corrected base lanes pass.
- Immutable-source execution required separating the read-only source snapshot from the mutable runtime virtualenv and working directory, and resolving the batch launcher explicitly from the snapshot. Local suite remains `112 passed`; focused Ruff and shell checks pass. Final read-only HEX snapshot is `/home/lmbanr001/masters/sallm_snapshots/pure-gdn-prompt-correction-20260809-693fe42c`, with `691/691` local/HEX source hashes equal and deployment-manifest SHA-256 `0396a24cb9cd4f83e80406ec06fbf5d4f2532a245f3b2cc09f85acd3eacbbbb1`.
- Preserve pre-model/data infrastructure failures `1211606/1211607`, `1211637/1211638`, `1211649/1211650`, and `1211661`. None produced metrics or `evaluation_summary.json`; they successively exposed read-only `.venv`, relative batch-script resolution, and read-only Hydra-output assumptions.
- Corrected base jobs `1211671_0` lane 0 MasakhaNews and `1211675_3` lane 3 SIB-200 are healthy on A100-40GB under prefix `pure_gdn_base_0shot_bosraw_20260809r5`. Both log canonical `final_model`, BF16, zero-shot, `peft_adapter=None`, `merge_lora=False`, `apply_chat_template=False`, and `add_bos_token=True`; no fault markers or summaries yet. AfriHG job `1210866` continues unchanged for provenance only. Current trusted state remains base `0/16`, HPO winners `0/8`, held-out adapter tests `0`.

## Durable correction — base/chat prompt contract invalidates prior freeze (2026-08-09)

- Supersede the provisional statements below that T2X job `1210865` is a
  scientifically trusted negative result or that base trust is `15/16`.
  The marker-only held-out output triggered a deeper source-contract audit;
  it is quarantined and must not select a prompt variant.
- Canonical pretraining consumed `anrilombard/mzansi-text-tokenized`. Direct
  read-only dataset inspection confirms representative validation documents
  begin with `[BOS]` token `0` and end with `[EOS]` token `1`. The current
  SALLM chat template omits BOS, and the base lm-eval runner unconditionally
  forces `add_bos_token=false`.
- The canonical base tokenizer defines only BOS/EOS/PAD/UNK as special tokens.
  `<|system|>`, `<|user|>`, and `<|assistant|>` are not atomic vocabulary or
  added special tokens; they decompose into ordinary punctuation/word pieces.
  The pure-GDN base was pretrained on plain documents, not chat conversations.
- Therefore all 16 historical base lanes are operational artifacts but `0/16`
  are currently scientifically trusted under the corrected prompt contract:
  14 used chat wrapping/generation, while SIB and SA-general used raw prompts
  but still omitted the required BOS. Preserve every artifact and quarantine
  all Sheet column-D values pending one implementation-correction rerun.
- Adapter training adds and trains the three chat-marker embeddings, so chat
  format remains appropriate after resizing, but the shared chat template must
  begin with exactly one BOS. Because every prior HPO train/validation input
  omitted BOS, all eight Multilingual validation grids must rerun. The prior
  News/SIB/Intent reconciliations remain provenance only, not frozen winners.
- The prospective second correction is frozen at SHA-256
  `0f84b1f50df5d713164fc3ff62c7e9bc61c5be22505116287956f93c7ddb86d8`
  in
  `sallm_memory/notes/2026-08-09-pure-gdn-prompt-contract-correction-preregistration.md`.
  Base prompts are plain text with exactly one leading BOS, no chat markers,
  and no terminal EOS. Adapter chat prompts contain exactly one explicit BOS
  and no terminal EOS after the assistant generation marker.
- AfriHG base correction job `1210866` may finish unchanged for provenance but
  cannot become trusted under its observed chat/no-BOS contract. Do not launch
  HPO or held-out adapter evaluation until the second implementation tests,
  validation-only canary, immutable deployment, and corrected base reruns pass.
- The corrected local stack passes `112` tests, selected Ruff checks, JSON
  validation, launcher syntax, and dry-run overrides. The pretraining-contract
  audit artifact has SHA-256
  `327d10cadaa48a3718c102366b80a64ebcc7acfa4c553e045688455eda450203`.
- The final immutable HEX prompt-correction snapshot is
  `/home/lmbanr001/masters/sallm_snapshots/pure-gdn-prompt-correction-20260809-5a727849`.
  It is read-only, contains `691/691` verified source/config files, has source
  set SHA-256
  `5a72784950be3b09317ecf563242751da5d0d9ae79a4ebd406d4485725ac583b`,
  and deployment-manifest SHA-256
  `9d6fb4bee2b84f4764e17c2a82b3a241e526c09ade38072343e4f7d80d8dfd9c`.
  Preserve earlier `23bef54b`, `e3d2f83b`, and `85e36103` snapshots as
  pre-canary provenance only.
- A100 canary jobs `1211314` and `1211315` failed before model/data access and
  produced no metric: the first exposed an unset `$SCRATCH` wrapper assumption;
  the second rejected an assumed SXM GPU name because the node is
  `NVIDIA A100-PCIE-40GB`. Both failures are preserved. The exact-SKU retry is
  job `1211316`; it started on `srvrocgpu010` A100-40GB `gpu:ampere`, verified
  all 691 manifest hashes, and was healthy with no fault marker at 17:24 SAST.
  AfriHG provenance job `1210866` remains running unchanged on the same GPU
  family. There are no A100-80GB or L40S jobs.

## Durable update — pure-GDN correction canary passed (2026-08-09)

- The implementation-correction gate is locally clean (`109 passed`; selected
  Ruff checks clean) and the exact fallback chat template is now centralized
  across training, generation, constrained POS, classification, lm-eval, and
  evaluation-harness tokenizer paths. Regression tests prove direct
  `apply_chat_template(..., tokenize=True)` IDs equal rendered-text IDs with
  `add_special_tokens=False`, and reject a terminal EOS after the assistant
  marker.
- Preserve Kombuys canary attempt 1 at
  `/scratch/alombard/sallm/results/pure_gdn_hpo_correction_canary/2026-08-09/attempt-1`.
  It verified the then-current execution manifest but failed after model load
  because constrained POS received a tokenizer without the fallback template;
  it produced no HPO result.
- Kombuys validation-only canary attempt 2 passed end to end, but a subsequent
  full-manifest comparison found 13 source-set differences between the
  Kombuys working tree and the intended local/HEX snapshot, including
  `launch_finetune.sh`, `run_pure_gdn_validation_trial.sh`, and `disk.py`.
  Preserve attempt 2 and its result SHA-256
  `2c309c9f3f2fe0110f00bfc0a78973ff8acb55be52acef691b1560208b153ad5`,
  but do not use it as the deployment-equivalence gate.
- A clean exact-source Kombuys snapshot was then created and validation-only
  attempt 3 passed on the sole visible RTX 3080 Ti. Its `690`-file source-set
  digest is
  `5b8992861abaa6fe90904feafffc45552fef59ccaefecbaf99bd6672dc3228e8`;
  execution-manifest SHA-256 is
  `26647ba8d43d9e3fb3b7d5ea13a0f7e69bcb80033c7e06097faf559a92fc7fe3`;
  and `canary_result.json` SHA-256 is
  `ea386ef7a8e4fc8acb82fc09a2caaaeae4ae05bdb7ae055e08ab2b0df5ad682b`.
  The preregistration hash remains
  `a27f55cd35a48bb1d22c5ca101ec536ffb89a64229559ce6883adb6fe744a04c`.
- Attempt 3 verified canonical pure-GDN identity (`GatedDeltaNetForCausalLM`,
  `attn=None`, `127,425,448` parameters, BF16 load, no adapter, no merge), four
  prompt-token equality/no-terminal-EOS contracts, delimiter-faithful NER over
  all `10,760` prompt-expanded rows, constrained POS over all 12
  language/prompt cells, prefix-plus-extra score `2/3` rather than full credit,
  and exact General processed coverage `22,167` with
  SIB/News/NER/POS/AfriHG/T2X counts
  `2,970/3,095/10,760/1,800/3,082/460`.
- Attempt-3 generations were nonempty but repetitive (`<<<<<<<<`) for NER, T2X,
  and AfriHG. This is not an HPO-quality result and did not select a recipe; the
  preregistered canary validates implementation paths and coverage only.
  RTX 5090 remained untouched at `10 MiB/0%`; both Kombuys GPUs are idle.
- The News/SIB/Intent validation-only reconciliation artifact is deterministic
  at SHA-256
  `497ad12de2d397b82132fe39cb1e0f7cb07ad14f1f3449aaf17fbf4c8ef68839`.
  Its unchanged winners remain prepared but are not globally ratified.
- The exact attempt-3 source snapshot is deployed read-only on HEX at
  `/home/lmbanr001/masters/sallm_snapshots/pure-gdn-correction-20260809-5b899286`.
  Its HEX deployment-manifest SHA-256 is
  `c408a10ab5264b81962025c96f76c907a392abb2844d09688215021c3ccae99f`;
  all `690/690` hashes match attempt 3 with zero differences. The earlier
  mismatched candidate remains provenance only and is not runnable.
- Correction gates 2 and 3 are now passed. Operational/scientific state remains base
  `16/16` artifacts but `14/16` trusted, and HPO winners `0/8` globally frozen.
  Next rerun quarantined base lanes 14--15 once from the verified snapshot
  before corrected validation grids. No held-out adapter evaluation or Sheet
  E/F/G write is authorized.
- The one-time implementation-correction base reruns are now active from a
  runnable copy whose `690/690` source hashes exactly match the immutable
  attempt-3 snapshot. T2X lane 14 is job `1210825`; AfriHG lane 15 is job
  `1210826`. Their run-manifest SHA-256 values are
  `427d4ca61681d350b2fc542c7bd1a13b6a7082f9cbd3d4d829226d005f7ef90e`
  and
  `57d9f86c3b6d94fa61956e08559ebf199d219b3c9cbb007aa9bdf5d9db93c0b4`.
  Both manifests hash the six canonical model artifacts and exact source-set
  digest
  `5b8992861abaa6fe90904feafffc45552fef59ccaefecbaf99bd6672dc3228e8`.
- At 14:23 SAST, `1210825/1210826` were healthy on `srvrocgpu010`, each using
  one A100-40GB `gpu:ampere`. Both loaded the canonical local checkpoint and
  entered the frozen held-out generation task with BF16, zero-shot, no adapter,
  and no merge; no traceback, OOM, token-contract, CUDA, or NCCL marker exists.
  Conditional historical-runtime ETAs are about 17:05 SAST for T2X and 23:55
  SAST for AfriHG.
- Provenance-only invalid General LR-2 job `1207524` remains the third and last
  owned A100-40GB job. At step `8327/13640` its training ETA was almost equal
  to its remaining wall time, leaving no room for terminal validation; a
  `TIMEOUT` near 23:29 SAST is now likely. Preserve it and do not use it for
  selection. No owned A100-80GB or L40S job exists.
- HEX quota is home `3/10 GB` (`32.9%`) and scratch `108/300 GB` (`36.2%`).
  Kombuys is idle after attempt 3 (RTX 5090 `10 MiB/0%`, RTX 3080 Ti
  `1 MiB/0%`). Sheet column D remains operationally populated with rows
  41--43 quarantined; E/F/G remain blank and no Sheet write occurred.
- At the 14:29 SAST monitoring pass, corrected base jobs `1210825/1210826`
  remained healthy and running on `srvrocgpu010`, each at `00:07:55` on one
  A100-40GB `gpu:ampere`. Both had completed dataset expansion/filtering and
  were inside their frozen test-generation paths; neither corrected
  `evaluation_summary.json` existed yet and no fault marker appeared.
  Provenance-only invalid General job `1207524` remained the third owned
  A100-40GB job at step `8383/13640` and `14:59:16` elapsed. Quota and
  GPU-family/Sheet state were unchanged, so no additional job or Sheet write
  was authorized.
- At `14:39:35 SAST`, `srvrocgpu010` became `DOWN+NOT_RESPONDING`; Slurm marked
  `1210825`, `1210826`, and invalid General `1207524` `NODE_FAIL` at the same
  instant. Corrected base summaries were absent and their logs ended during
  automatic batch-size setup, so this is shared infrastructure failure rather
  than metric/model evidence. Preserve all three jobs and their outputs. Do
  not restart General. Base lanes 14--15 are eligible for one unchanged
  non-metric-driven infrastructure retry under new output prefixes from the
  same immutable source/model/protocol.
- After verifying no summary and no active duplicate, unchanged `r3`
  infrastructure retries were submitted: T2X `1210850` and AfriHG `1210851`.
  Both preserve the failed `r2` outputs and use the same immutable `5b899286`
  source, canonical model, frozen BF16 zero-shot no-adapter protocol, and
  `nlpgroup/a100/nlpgroup` `gpu:ampere:1`, 24-hour, 8-CPU envelope. At 14:59
  SAST both were pending because `srvrocgpu010` remained unavailable, so ETA
  is unknown. No General retry or third job was submitted. Quota remained
  home `32.9%`, scratch `36.2%`; no A100-80GB/L40S work exists. Kombuys was
  read-only and idle (RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`). Sheet
  rows 41--43 remain quarantined and E/F/G blank.
- At 15:14 SAST `srvrocgpu010` remained `DOWN+NOT_RESPONDING`; Slurm exposes
  no cause beyond loss of node response. Four jobs shared the `NODE_FAIL`
  window, confirming an infrastructure-wide event. Idle `srvrocgpu009` is
  A100-40GB but uses the different `gpu:amperemk` GRES class, so do not move
  the frozen correction jobs there without a prospective hardware-fallback
  amendment and replacement of—not duplication with—`1210850/1210851`.
- User-authorized `amperemk` fallback was preregistered prospectively at
  SHA-256
  `1f3ab735f34a5e79a4b25321dffba12d57ae6621852833bbbcb82391d9f53708`.
  Never-started jobs `1210850/1210851` were cancelled, but Slurm rejected the
  first replacement with `AssocGrpGRES` and did not attempt the second; no
  `r4` job or output exists. Accounting configuration explicitly gives
  `nlpgroup` `amperemk=0` (and `a100free` also has `amperemk=0`, with no user
  association), so idle `srvrocgpu009` is inaccessible. Preserve the unused
  amendment and restore clean `gpu:ampere` queue replacements under new
  prefixes; do not use A100-80GB/L40S or Kombuys.
- Original-hardware replacements are restored as T2X `1210865` and AfriHG
  `1210866`, using new `r5` prefixes and the unchanged immutable
  source/model/protocol. At 15:22 SAST both were correctly pending on
  `gpu:ampere:1` because `srvrocgpu010` remained unavailable; ETA is unknown.
  They are the only owned jobs, their summaries are absent, and General was
  not restarted.
- `srvrocgpu010` recovered and started `1210865/1210866` at `15:49:41 SAST`.
  At 15:56 both were healthy on A100-40GB `gpu:ampere`, had verified execution
  manifests (`5eed5d2b…cccff` / `8e4b6b71…f18c`), loaded canonical
  `final_model` with no adapter/merge, and entered the frozen T2X/AfriHG test
  paths with zero fault markers. Conditional ETAs are about 18:33 SAST and
  01:23 SAST respectively. Quota remained home `32.9%`, scratch `36.2%`; no
  A100-80GB/L40S work exists. Kombuys remained idle with RTX 5090 untouched.
  Sheet quarantine and blank adapter columns remain unchanged.
- Corrected T2X job `1210865` completed `0:0` in `00:49:51`. Summary SHA-256
  is `87627ec268bba04a34790f43378f0189e9302126f834f19d007815712ffdfbf7`;
  Xhosa/all chrF is `1.729633871926085` and BLEU/ROUGE are zero. Manifest,
  canonical pure-GDN/no-adapter protocol, A100-40GB allocation, and all 378
  rows verify. Outputs are nonempty but completely degenerate: 34 unique,
  `378/378` contain `<|assistant|>`, 347 contain `<|assisted|>`, and zero rows
  retain alphanumeric content after stripping those control markers. This is
  valid negative model performance after the preregistered implementation
  correction, not a rerun trigger. Base trust is now `15/16`; keep the T2X
  Sheet cell quarantined until corrected AfriHG and joint base closeout.

## Durable update — pure-GDN scientific audit and quarantine (2026-08-09)

- The downstream program is operationally advanced but has not passed its
  scientific freeze gate. All 16 base lanes produced hash-reconciled artifacts,
  but only `14/16` are trusted: base T2X and AfriHG must be rerun after a
  documented generation implementation correction. Quarantine canonical Sheet
  rows 41--43 while preserving their original artifacts and provenance.
- Corrected HPO is **no-go** until implementation tests and a validation-only
  Kombuys RTX 3080 Ti canary pass. The complete correction contract and frozen
  recovery order are preregistered before corrected metrics in
  `sallm_memory/notes/2026-08-09-pure-gdn-hpo-correction-preregistration.md`,
  SHA-256
  `a27f55cd35a48bb1d22c5ca101ec536ffb89a64229559ce6883adb6fe744a04c`.
- The generation evaluator retokenizes a rendered chat prompt with default
  `add_special_tokens=True`, adding BOS and terminal EOS after the assistant
  marker. Kombuys confirms direct chat-template tokens equal rendered tokens
  with `add_special_tokens=False`; bounded RTX 3080 Ti A/B output changed under
  the correction while RTX 5090 remained untouched. Across every LR/epoch,
  NER is `192/192` empty, T2X and AfriHG are `64/64` empty, and all POS files
  are empty except LR-0 epoch 1, whose `192/192` nonempty outputs all fail
  parsing. NER, POS, T2X, and AfriHG require full validation-only reruns.
- NER parsing is independently invalid: punctuation splitting and unsafe
  substring label replacement misparse `20/2,152` raw validation references
  (`100/10,760` P1--P5 examples), including `David A. Gross`, `Kazan, Russia`,
  `The Bomb Shelter Film Company`, and `Stimela`. Corrected HPO requires an
  explicit delimiter- and label-aware parser, not only tokenization repair.
- POS free-generation selection does not match the established final contract
  and gives a correct prefix plus extra tags full credit. Corrected HPO uses
  closed-label continuation-logprob tuple scoring, mean label-token score,
  exactly one UPOS label per token, token accuracy, and canonical P1--P4.
- News, SIB, and Intent selected on support-weighted F1 although the frozen
  protocol requires mean macro-F1. Validation-only W&B reconciliation gives
  corrected macro-F1 `0.0736133409/0.0576036866/0.0025019580`; each existing
  winner remains unchanged, but ratification requires a stored hashed
  reconciliation before any held-out evaluation. Retain the preregistered
  early-stopping threshold `0.001`; changing it now would be post-hoc and would
  require rerunning the affected Intent grid.
- General validation declared `12,209` examples but evaluated only `9,127`.
  The missing `3,082` are the entire AfriHG component, excluded because the HEX
  loader lost language labels. Its loss is also batch/sample weighted rather
  than assistant-token weighted. Corrected General selection asserts `task_name`
  and exact `22,167` canonical prompt-expanded rows, computes token-weighted
  assistant NLL within each family, and takes an equal macro mean across the six
  families. Jobs `1204261/1204262/1207524` are provenance only and General must
  rerun.
- HEX has no valid Git `HEAD` and zero tracked files; critical local/HEX SHA
  drift was observed for `afrihg.py` and `classification_metrics.py`. All
  correction runs require an immutable commit or complete imported-source,
  launcher, config, and environment hash manifest.
- At the 13:09 SAST pass, General LR-0/LR-1 `1204261/1204262` were complete
  `0:0`; LR-2 `1207524` was healthy at step `7693/13640` on the sole
  owned A100-40GB `gpu:ampere` job. Preserve it for provenance. HEX quota was
  home `3/10 GB` (`32.6%`) and scratch `108/300 GB` (`36.2%`); no owned
  A100-80GB/L40S job existed. Kombuys was idle after diagnostics (RTX 5090
  `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `61%`).
- No held-out adapter test has run. Keep Mono training, all held-out adapter
  evaluation, Sheet E/F/G writes, and publication blocked until a dated
  correction implementation passes its canary, base lanes 14--15 are corrected,
  News/SIB/Intent are hash-reconciled, NER/POS/T2X/AfriHG/General rerun, and all
  eight valid family winners are frozen. Adapter columns remain blank; no Sheet
  or Hugging Face write occurred. Full evidence is in
  `sallm_memory/notes/2026-08-09-pure-gdn-scientific-audit.md`.

## Durable update — pure-GDN base gate closed (2026-08-07)

- All 16 frozen canonical pure-GDN zero-shot base-evaluation lanes are complete
  and verified in `Pure GatedDeltaNet Results` (`sheetId=202608060`). The last
  lane, AfriHG `1185915_15`, completed `0:0` on A100-40GB after `09:33:50`.
- AfriHG combined summary SHA-256 is
  `089800aba8602dc7bec9795176e0a38aa1f7f17fcbb508c7903c210d241bad78`.
  Xhosa/Zulu chrF is `4.079834809867061` / `4.125909737887804`; English is
  structurally inapplicable and explicitly recorded as such. Rows 42--44 were
  published and re-read with complete provenance.
- No owned GPU jobs remained after base closure. Fine-tuning and held-out
  adapter evaluation remain paused until validation-only Monolingual,
  Multilingual, and General protocols are fully preregistered. Do not use any
  held-out base or adapter metric for recipe, checkpoint, prompt, rerun, or
  hyperparameter selection.
- Complete all experiments and evaluations by 31 August 2026. Paper drafting
  starts afterward, with a first advisor draft targeted for mid-September.

## Durable update — pure-GDN adapter protocol preregistered (2026-08-07)

- Before any post-base submission, the validation-only Mono/Multi/General
  protocol was frozen in
  `sallm_memory/notes/2026-08-07-pure-gdn-adapter-preregistration.md`
  (preregistration SHA-256
  `51ac66a4166a2f0c6e003cff1b3fb7ff51061d0a22f31d7c1618ecefb581a7d5`).
- Architecture-complete rank-16 LoRA, the three-point LR grid
  `3e-5/8e-5/1.5e-4`, task-native full-validation checkpoint metrics, prompt
  coverage, tie-breaks, retry rules, A100-40GB-only execution, applicability,
  and one-time held-out evaluation are fixed. Cancelled `1183162` remains
  cancelled and the completed News validation screen selects `3e-5` only for
  the News family.
- General uses the precomputed token-balanced six-family mix and selects LR and
  checkpoint on validation loss only. Historical auto-eval bulk launchers are
  disallowed because they chain held-out tests before the global freeze gate.
- No new training or adapter held-out job was submitted during preregistration.

## Durable update — validation runner ready, HEX sync blocked (2026-08-07)

- Validation-only family runner, loss-only General callback gate, Hydra
  add-or-override normalization, and override-aware GDN kernel check are locally
  implemented and verified. All 21 family/LR mappings dry-run; focused test is
  `1 passed`; NER and General Hydra compositions pass. The runner has no
  held-out-evaluation path and Hub writes are disabled.
- HEX remains idle at `100/300 GB` scratch (`33.5%`) with no existing screen
  artifact. No remote file changed and no job was submitted because targeted
  sync to `hex:~/masters/sallm/` was denied pending explicit user approval.
  After approval, sync the four runner files preserving their relative paths,
  then submit only SIB LR-0 as the first A100-40GB validation canary.

## Durable update — first adapter validation canary active (2026-08-07)

- Explicit approval was received and the four validated runner files were
  synced to their correct relative HEX paths with matching hashes. Initial SIB
  LR-0 canary `1189728` failed `1:0` after 50 seconds on a Hydra-only missing-key
  override, before model/data/metrics. The allowed implementation correction is
  `++training.gradient_checkpointing=false`; local syntax, diff, and real SIB
  composition checks passed. Preserve this failure provenance.
- Corrected canary `1189730` is the sole owned GPU job on A100-40GB
  `gpu:ampere`. It uses the canonical pure-GDN checkpoint, BF16, the exact ten
  architecture-complete LoRA targets, validation-only SIB data and all five
  prompts, fast GDN kernels, and no Hub push. W&B `86xxffgr` shows finite loss
  through step 90 (`6.1871`) and finite gradient norm (`25.6114`) with no
  traceback/OOM/NCCL or held-out-test activity.
- Keep all other screen trials gated until `1189730` demonstrates a complete
  validation epoch and valid selection metric. At the canary start, HEX scratch
  was `100/300 GB`; Kombuys remained read-only and no A100-80GB or L40S job was
  active or submitted. No sheet row changes until a verified selected adapter
  reaches its preregistered one-time held-out phase.

## Durable update — SIB validation grid released (2026-08-07)

- SIB LR-0 canary `1189730` passed its full epoch-1 gate: checkpoint `526`,
  validation loss `13.205979719065656`, and frozen validation macro-F1
  `0.10182469859889215`. All six languages and five prompts completed before
  epoch 2 began; no held-out data entered the run.
- The remaining preregistered SIB grid points now fill the two open slots:
  `1189838` (`8e-5`, W&B `v2b6kqxe`) and `1189839` (`1.5e-4`, W&B `web3ll2q`).
  Both passed pure-GDN, BF16, fast-kernel, ten-target, dataset, Hub-disable, and
  finite-loss/gradient startup gates. All three owned jobs use A100-40GB
  `gpu:ampere`; no A100-80GB/L40S work is active. Do not submit a fourth job.
- Keep recipe selection validation-only. Wait for all three SIB trials to stop
  under the frozen patience rule, then compare their best validation metrics
  with the lower-LR and earlier-checkpoint tie-breaks. No sheet write or
  official-test evaluation occurs during this phase.

## Durable update — SIB epoch-1 grid metrics complete (2026-08-07)

- All three SIB grid points have a complete epoch-1 validation checkpoint.
  Frozen macro-F1 is exactly `0.10182469859889215` for each of `1189730`,
  `1189838`, and `1189839`. Epoch-1 validation loss differs
  (`13.2059797191/13.4791498974/13.1062458899`) but is not a selection metric.
- LR-0 epoch 2 retains the exact same macro-F1 while validation loss improves
  to `13.0279887087`; this counts as one non-improving metric epoch under the
  frozen patience rule. All jobs continue cleanly. Do not select until every
  SIB job reaches terminal state; if the macro-F1 tie persists, freeze LR
  `3e-5` by the preregistered lower-LR tie-break and its earliest tied best
  checkpoint. Held-out test and sheet remain untouched.

## Durable update — SIB LR-0 complete, NER screen started (2026-08-07)

- SIB LR-0 `1189730` completed `0:0` in `02:06:28`. Frozen macro-F1 stayed
  `0.10182469859889215` for epochs 1--3, so patience stopped at epoch 3 and
  restored epoch-1 `checkpoint-526`. Final adapter SHA-256 is
  `c275ffa45b6ba53b3cb96f487bc3a4e81f3d49b13daca5a36bf631bd568b22f3`.
  SIB selection remains open until LR-1 `1189838` and LR-2 `1189839` finish.
- The released slot now runs preregistered NER LR-0 `1190665` (W&B
  `1nscfvh3`) on A100-40GB. It passed pure-GDN/BF16/ten-target/fast-kernel,
  validation-only dataset, Hub-disable, and finite loss/gradient startup gates.
  No A100-80GB or L40S work overlaps. No held-out or sheet action is due.

## Durable update — SIB winner frozen, full NER grid active (2026-08-07)

- SIB LR-1 `1189838` and LR-2 `1189839` both completed `0:0`. Every LR and
  every epoch tied at validation macro-F1 `0.10182469859889215`; therefore the
  preregistered lower-LR and earlier-checkpoint rules freeze SIB LR `3e-5`, job
  `1189730`, epoch-1 `checkpoint-526`. Do not run SIB held-out test until the
  global family-winner freeze gate.
- The full NER grid is active on A100-40GB: LR-0 `1190665`, LR-1 `1191239`,
  LR-2 `1191240`. LR-0 epoch 1 produced finite validation loss
  `12.8743516206` and frozen span-F1 `0.0` at checkpoint `541`, then resumed.
  LR-1/2 passed identical configuration and finite-startup gates. No other GPU
  family overlaps; no held-out or sheet update is allowed yet.
- At the 11:23 SAST pass, LR-0 had completed epoch 2 at `checkpoint-1082`
  (`eval_loss=13.4866886181`) with the frozen best span-F1 still `0.0` at
  epoch-1 `checkpoint-541`; LR-1/LR-2 completed epoch 1 at `checkpoint-541`
  with `eval_loss=13.5964215701/13.7191337273` and best span-F1 `0.0`.
  All three resumed with finite losses and gradients and no fault marker.
  Scratch is `102/300 GB` (`34.1%`); these are the only owned jobs and all use
  A100-40GB `gpu:ampere`. Expected terminal window is roughly 12:00 SAST for
  LR-0 and 12:45--13:00 for LR-1/LR-2 if the frozen patience rule stops after
  the third tied epoch. Kombuys remains read-only and idle. No sheet or
  held-out action is due.
- NER LR-0 `1190665` subsequently completed cleanly `0:0` in `01:55:27`.
  Span-F1 remained `0.0` at epochs 1--3, so frozen patience stopped after
  epoch 3 and restored epoch-1 `checkpoint-541`; validation losses were
  `12.8743516206/13.4866886181/13.6208044116`. Verified final adapter/model
  and config SHA-256 values are
  `075bcea23fc0fd517c1ab0067d570f555aa1f016ad67087e2e00a8f574ea84cb`
  and `47635d24822a96bb6a327ffe9687d9cb0fe7a8120f9cfc8a1e0f5e513fe02768`.
  NER LR-1/LR-2 epoch 2 also retained span-F1 `0.0`; validation losses are
  `13.5341346712/13.4914450947`, with epoch-1 checkpoints still best.
- After confirming no completed POS LR-0 artifact and no active duplicate,
  the released slot was filled by preregistered POS LR-0 job `1191712`
  (`3e-5`, W&B `366ujt2g`). It runs on A100-40GB with the canonical pure-GDN
  base, BF16, fast kernel, exact ten LoRA targets, `4,649,088` trainable
  parameters, Hub disabled, and validation-only POS data (`2,259/2,250`
  train/eval after five-prompt expansion). Training reached step 27 without a
  fault marker. Exactly three A100-40GB jobs remain active; no other GPU family,
  held-out evaluation, or sheet update overlaps.
- NER LR-1 `1191239` and LR-2 `1191240` completed cleanly `0:0` in
  `01:54:59` and `01:54:19`. Their epoch-3 validation losses were
  `13.5914469099/13.7344908080`; span-F1 remained `0.0` for every LR and
  epoch. The preregistered lower-LR then earlier-checkpoint rule therefore
  freezes the NER family to LR `3e-5`, job `1190665`, epoch-1
  `checkpoint-541`. Final adapter SHA-256 values for LR-1/LR-2 are
  `e3f02b9489226da5eaee1cd59e790aa161168923e22a7a6ffc58cc46412b9dda`
  and `671c1392fcfc0b71f7ae7bc57c9005b846a3beccfaf9b96ece67e2e10b65d69d`.
  No NER held-out evaluation is allowed before the global family freeze.
- POS LR-0 `1191712` completed epoch 1 at `checkpoint-283` with validation
  loss `11.0739157986` and frozen token accuracy `0.0`, then resumed. After
  verifying both newly released slots had no POS completion or duplicate,
  POS LR-1 `1192240` (`8e-5`, W&B `ojzj40j0`) and LR-2 `1192241`
  (`1.5e-4`, W&B `n1z4mr8r`) were submitted. Both passed canonical
  pure-GDN/BF16/fast-kernel/ten-target startup and began training on the same
  `2,259/2,250` validation-only datasets. These three POS jobs are the only
  owned GPU work, all on A100-40GB; scratch is `102/300 GB` (`34.3%`). No
  held-out or sheet action is due.
- POS LR-0 `1191712` completed cleanly `0:0` in `01:15:53`. Frozen token
  accuracy remained `0.0` at epochs 1--3; validation losses were
  `11.0739157986/12.5322222222/12.2372230903`, so the restored best is
  epoch-1 `checkpoint-283`. Final adapter/config SHA-256 values are
  `1817e8e2014b63bf24d96f499f417965bb137533f1fe5bfa87507e5071142d25`
  and `1045e80c1369142d0a29567f6bdba47018b7296fbb0ac87190c56f1cdf42dca1`.
- After confirming no completed Intent LR-0 artifact and no active duplicate,
  the released slot was filled by preregistered Intent LR-0 job `1192257`
  (`3e-5`, W&B `t7smhrtn`). It passed canonical pure-GDN/BF16/fast-kernel/
  ten-target startup, built `7,045/4,155` validation-only train/eval examples,
  and entered training without a fault marker. POS LR-1 `1192240`, POS LR-2
  `1192241`, and Intent LR-0 `1192257` are exactly the three active jobs, all
  on A100-40GB; scratch is `103/300 GB` (`34.4%`). Kombuys remains read-only
  and idle. No held-out or sheet action is due.
- POS LR-1 `1192240` and LR-2 `1192241` completed cleanly `0:0` in
  `00:56:16/00:56:09`. Frozen token accuracy was `0.0` for every LR and epoch,
  so the preregistered tie rules freeze POS to LR `3e-5`, job `1191712`,
  epoch-1 `checkpoint-283`. LR-1/LR-2 final adapter SHA-256 values are
  `6bfbf68ae775224bd43dd2d95a22e05f14b7297f91a1d09520e54b0219825381` and
  `d33ad0418adedf50eef8a31d24ce757ae89b718fbaf74ce45c2d60d20d1395db`.
- The released slots now run preregistered Intent LR-1 `1192267` and LR-2
  `1192268` beside LR-0 `1192257`. Both new jobs passed canonical pure-GDN,
  BF16, exact ten-target LoRA, validation-only `7,045/4,155` dataset, and
  finite-loss/gradient training-start gates. These are exactly three owned jobs, all on
  `srvrocgpu010` A100-40GB `gpu:ampere`; no A100-80GB/L40S work overlaps.
  Scratch is `103/300 GB` (`34.5%`), Kombuys is read-only/idle, and no
  held-out adapter or sheet action is allowed yet.
- Intent LR-0 `1192257` completed its full epoch-1 validation callback at
  `checkpoint-881`: validation loss `13.088938755076715`, frozen macro-F1
  `0.0012702529230128718`, then resumed epoch 2. LR-1 `1192267` and LR-2
  `1192268` completed epoch-1 trainer evaluation with validation losses
  `12.905990053399519/13.072481502895608`; their task-metric callbacks remain
  active. No LR selection, held-out evaluation, sheet write, or replacement
  submission is allowed until all three trials reach terminal state.
- All three Intent trials now have epoch-1 selection metrics. LR-0 `1192257`
  is the provisional leader at macro-F1 `0.0012702529230128718`; LR-1
  `1192267` and LR-2 `1192268` tie at `0.0010740856332265788`. Every current
  best is epoch-1 `checkpoint-881`, but no Intent recipe may be frozen until
  all three jobs terminate under the preregistered patience rule. LR-0 has
  entered its epoch-2 metric callback; LR-1/LR-2 continue epoch-2 training.
- Intent LR-0 `1192257` completed epoch 2 with validation loss
  `13.292073743983153` but no improvement over its epoch-1 macro-F1
  `0.0012702529230128718`; epoch-1 `checkpoint-881` remains selected. LR-0
  entered epoch 3. LR-1 `1192267` and LR-2 `1192268` have reached their
  epoch-2 validation phase and still retain their epoch-1 checkpoints. No
  family selection is permitted before all three terminal states.
- Intent LR-1 `1192267` and LR-2 `1192268` subsequently improved at epoch 2
  to validation macro-F1 `0.001265884217324858` and
  `0.0013538364263929801`, both at `checkpoint-1762`. LR-2 is the provisional
  family leader, but all three trials continue through their frozen terminal
  rule and no Intent winner is frozen yet. At 16:27 SAST LR-0 was in its
  epoch-3 validation callback and LR-1/LR-2 were in epoch-3 training; exactly
  three healthy A100-40GB jobs were active, so T2X remained unsubmitted.
- Intent LR-0 `1192257` completed `0:0` in `03:24:52`; its frozen best remains
  epoch-1 macro-F1 `0.0012702529230128718` at `checkpoint-881`. Final
  adapter/config SHA-256 values are
  `e541616d33084967d52cf12deeea4c49be8f19e8285e5ed3eb9a515061f30b11`
  and `a2ce56b4c9c4f51ee838d3c7bb1a489aaf763c6026f757f356b2ebaeaa5dd1f3`.
  After the required no-artifact/no-duplicate checks, the released slot was
  assigned to preregistered T2X LR-0 job `1192813` (`3e-5`, A100-40GB), which
  is pending for `Priority`. Intent selection remains open until LR-1/LR-2
  `1192267/1192268` terminate; no held-out or sheet action is allowed yet.
- Intent LR-1 `1192267` improved again at epoch 3 to frozen validation
  macro-F1 `0.002555133509628254` at `checkpoint-2643`, provisionally ahead of
  LR-0 and LR-2. LR-2 `1192268` is still completing its epoch-3 callback, so
  no family winner is frozen. T2X LR-0 `1192813` remains pending for
  `Priority`; with the two running Intent jobs this is the three-job limit.
- Intent LR-2 `1192268` completed `0:0` in `03:28:35`; its frozen best remains
  epoch-2 macro-F1 `0.0013538364263929801` at `checkpoint-1762`. Final
  adapter/config SHA-256 values are
  `4d87e8a8d2c411807108c2a8ec0c9fba2556737b378c08671a8fcbb7f375792b`
  and `b97c8fb363e6210080cb04592c239aee96b5958b7ed663ffff853520f51a84a6`.
  LR-1 `1192267` continues epoch 4 because its epoch-3 improvement reset the
  frozen patience counter, so Intent remains open. After no-artifact and
  no-duplicate checks, T2X LR-1 `1193166` (`8e-5`, A100-40GB) was submitted
  and is pending for `Priority`; T2X LR-0 `1192813` is pending for `Resources`
  with provisional start `2026-08-08 00:25:19 SAST`. No held-out or sheet
  action is allowed yet.
- Intent LR-1 `1192267` reached epoch-4 validation at step `3524` with no fault
  marker; its current frozen best remains epoch-3 macro-F1
  `0.002555133509628254` at `checkpoint-2643`. Intent remains open. Pending
  T2X jobs `1192813/1193166` plus running `1192267` are the three-job limit;
  no further submission or held-out action is due.
- Intent LR-1 `1192267` completed epoch 4 without improving its best
  macro-F1 `0.002555133509628254` at `checkpoint-2643`; it entered epoch 5
  with one non-improving epoch accumulated under patience 2. Intent remains
  unfrozen, and pending T2X jobs `1192813/1193166` keep the owned job count at
  three.
- Intent LR-1 `1192267` completed cleanly `0:0` in `05:47:47`; its frozen
  best is epoch-3 validation macro-F1 `0.002555133509628254` at
  `checkpoint-2643`. Final adapter/config SHA-256 values are
  `755c2e778162f6ee2880ae33b25fc32d05354bb87b3ba1932ee05830e77400b7`
  and `db9b46e1620e982d69e91a9b673d5c8a97d6474a177304ce1ab5daab088fbccc`.
  The completed validation grid therefore freezes Intent to LR `8e-5`, job
  `1192267`, checkpoint `2643`. Five total family winners are now frozen:
  News, SIB, NER, POS, and Intent; T2X, AfriHG, and General remain open.
- T2X LR-0 `1192813` is healthy on A100-40GB (W&B `esmggy0y`) after passing
  canonical pure-GDN/BF16/ten-target LoRA/validation-only startup gates.
  T2X LR-1 `1193166` remains pending for `Resources`; after no-artifact and
  no-duplicate checks, T2X LR-2 `1193880` was submitted and is pending for
  `Priority`. These are the three-job limit; no held-out or sheet action is
  due.

## Durable update — GDN Xhosa NER HPO final (2026-08-01)

- Validation-only 24-trial screen `1150968` plus three-seed finalist jobs `1151254/1151255` selected `tyvese0z` (`eval/all_f1` mean `0.6085`, sample SD `0.0097`, range `0.5978-0.6165`) over `b888dh4l` (`0.5858`, SD `0.0336`). Representative seed-42 `checkpoint-552` was frozen rather than selecting the lucky highest seed.
- Official held-out job `1151321` returned P1-P5 F1 `0.5192/0.6269/0.6239/0.6156/0.6287`; canonical headline explicitly **best prompt P5 `0.6287`**, mean `0.6029`, range `0.5192-0.6287`, labelled descriptive rather than unbiased. Result and full provenance are promoted to `GatedDeltaNet Results!E4`; `Comparison Data!I44=0.6287`.
- This strongly supports optimization/capacity as a major confound in the former mono-versus-multi NER gap, but matched Mono/Multi exposure controls across Xhosa/Zulu/Tswana remain required before attributing the residual pattern to architecture.

## Durable update — GDN General interpretation (2026-08-02)

- GDN General is not broadly superior: its matched advantage over Multi is confined to MasakhaPOS (`~0.015-0.019`), while it is worse on NER, SIB, Intent, AfriHG, and corrected News. The current General loader samples all six task families equally (`1/6` each) regardless of dataset size, a direct mixture/exposure confound that can oversample small structured tasks such as POS. Literature supports mixture balancing, capacity, and negative-transfer controls, but does not establish a GDN-specific General advantage. Do not make an architecture claim until update/token-matched mixture ablations and a matched non-GDN control are complete.
- The historical GDN General recipe is recovered from HEX job `1020357`, W&B
  `i77vz2jm`, and Hub commit `4c3635d3c15fac6d7cacc8904f389c17bbc3ab5f`:
  equal `1/6` task sampling, 43,637 examples, one epoch / 2,728 updates, max
  length 1024, batch 4 with accumulation 4, LR `8e-5`, cosine, warmup `0.03`,
  weight decay `0.01`, seed 42, and rank-16 / alpha-32 / dropout-0.05 LoRA on
  all seven GDN/attention projection targets. This is the reproducible anchor
  for the controlled mixture study, not evidence of a GDN-specific advantage.

## Purpose

Compare decoder-only architecture families for South African low-resource
language modelling and downstream task performance. The central question is not
only which model gets the highest score, but what each architecture costs and
where it fails under realistic low-resource constraints.

The comprehensive final registry for optimized results is the Google Sheet:
[SALLM results sheet](https://docs.google.com/spreadsheets/d/1Ph_zVcSuLZy0dUBnDPF4tkLqAfEVCwK9JVybB2_8x6U/edit?ouid=101226728847680131120&usp=sheets_home&ths=true).
Use this sheet as the source of final best-result rows. Use this Obsidian vault
for daily notes, experimental context, failure analysis, and defensibility
decisions around those final rows.

## Fair Comparison Policy

- Mamba rescue experiments must be interpreted against matched LLaMA controls
  whenever formulation, decoding, scoring, or result-selection changes.
- Each rescue must also be compared against the current best Mamba result for
  that task, so we can separate "Mamba got better" from "Mamba caught up to
  LLaMA" and from "both models benefit from a cleaner protocol".
- If an intervention improves both Mamba and LLaMA, record it as a shared
  decoder-only protocol improvement, not as a Mamba-only rescue. Shared
  improvements should be considered for later LLaMA reruns as well as Mamba.
- Keep a carry-forward backlog for shared improvements discovered during Mamba
  debugging, so useful protocol changes can be applied to LLaMA after the
  defensible Mamba recipe is selected.
- Matched xLSTM Base AfriXNLI/AfriMMLU 3-shot task-native accuracy is final
  from job `1139425` (`0:0`). Raw results verify `num_fewshot=3`, validation
  demonstrations, official test evaluation, and five complete prompts.
  Best-prompt headlines are AfriXNLI Eng/Xho/Zul/Sot
  `0.3433/0.3433/0.3517/0.3433` and AfriMMLU
  `0.2160/0.2440/0.2260/0.2180`; full provenance is in
  `XLSTM Results!D28:D35` notes. Legacy F1 rows are excluded. The shared
  summary writer now records effective/raw `num_fewshot` rather than the pack
  default.
- When a Mamba-focused experiment reveals a method that improves both Mamba and
  LLaMA, record that explicitly and plan to apply it to the final LLaMA rerun;
  those are protocol improvements, not evidence that Mamba alone was fixed.
- Corrected final Intent reporting now uses mean continuation-token
  log-probability, validation-F1 checkpoint selection, and one frozen official
  held-out test. Final clean multilingual-adapter best-prompt F1 values are:
  LLaMA Eng/Xho/Zul/Sot `0.0039/0.0033/0.0043/0.0043`, Mamba
  `0.0191/0.0096/0.0199/0.0250`, GDN `0.0058/0.0012/0.0091/0.0093`, and xLSTM
  `0.0733/0.1232/0.1025/0.1374`. These best-of-test-prompts values are
  descriptive, not unbiased; prompt means, ranges, winners, and all prompt
  values are retained in the canonical sheet provenance.
- Matched LLaMA and Mamba Base standardized suites are final from jobs
  `1139430/1139432` (`0:0`). Raw results verify exact base-checkpoint identity,
  effective `num_fewshot=3`, official test evaluation, and five complete
  prompts. Best-prompt AfriXNLI Accuracy for Xho/Zul/Sot/Eng is LLaMA
  `0.3533/0.3417/0.3767/0.3467` and Mamba
  `0.3567/0.3517/0.3400/0.3400`; AfriMMLU Accuracy is LLaMA
  `0.2320/0.2360/0.2560/0.2640` and Mamba
  `0.2140/0.2420/0.2380/0.2440`; AfriMGSM flexible exact match is LLaMA
  `0.0160/0.0080/0.0160/0.0120` and Mamba
  `0.0080/0.0120/0.0120/0.0160`. AfriXNLI/AfriMMLU demonstrations use
  validation. AfriMGSM exposes no separate few-shot split, so its three-shot
  results are matched benchmark-style values rather than clean independent
  train-demo estimates. Full prompt provenance is in the source-cell notes.
- Corrected General-adapter Intent evaluation is final for Mamba and GDN under
  the same mean continuation-token scorer and official held-out split. Mamba
  Eng/Xho/Zul/Sot best-prompt F1 is
  `0.0009909/0.0012195/0.0012195/0.0012195`; GDN is
  `0.0019991/0.0035620/0.0027329/0.0042349`. These are valid near-chance
  results, not missing runs. The clean xLSTM General replacement remains in
  training and is not promoted.
- Mamba General AfriMGSM is final under the standardized held-out
  task-native flexible-exact-match protocol. Xho/Zul/Sot/Eng best-prompt
  values are `0.0040/0.0080/0.0040/0.0040` from job `1135843`; prompt means,
  ranges, tied winners, all prompt values, artifact, and SHA are retained in
  `Mamba Results!G36:G39`.
- Corrected GDN monolingual Intent is final for Eng/Xho/Zul/Sot under the same
  official held-out mean continuation-token scorer. Best-prompt F1 is
  `0.0017254/0.0074846/0.0012214/0.0017021` from jobs
  `1137019/1137020/1137021/1137022`.
- Corrected xLSTM monolingual Intent is final for Eng/Xho/Zul/Sot under the
  same official held-out mean continuation-token scorer. Best-prompt F1 is
  `0.0016476/0.0053704/0.0057437/0.0106330` from jobs
  `1137487/1137489/1137491/1137493`. Validation-only selection froze
  checkpoints `165/252/63/504`. These are valid near-chance results, not
  missing runs.
- Mamba base POS is complete on the official held-out test under constrained
  mean label-token scoring: Xho `0.0360`, Zul `0.0211`, and Tsn `0.2157`
  best-prompt token accuracy (job `1131802`).
- Corrected Mamba monolingual Xhosa, Zulu, and Tswana POS are complete under
  the same constrained official-test protocol: best-prompt token accuracy is
  `0.0000`/`0.0421`/`0.0000` (array `1142391`; sacct components
  `1142392/1142393/1142391`). All three are verified negative results rather
  than missing runs: predictions collapse almost entirely to the legal `X`
  label.
- Mamba base MasakhaNER is complete on the corrected parquet-backed official
  held-out test: Xho, Zul, and Tsn are all `0.0000` F1 for every one of five
  prompts (jobs `1131946`-`1131948`). These are valid negative results, not
  missing runs: generation overgenerates/repeats and the entity extractor
  usually returns empty. The verified artifacts are under
  `/scratch/lmbanr001/masters/sallm/results/eval/mamba_base_closeout_20260729/`.
- The canonical four-architecture tabs contain no live `Queued`, `Running`,
  `Pending`, or `TODO` result statuses. Remaining unavailable values are
  explicitly labelled as quarantined or not applicable. Every populated
  task-language row now has an explicit Base/Mono/Multi/General entry rather
  than a blank cell. The legacy Transformer SIB200 Tswana row is not a missing
  run: the matched SIB200 suite intentionally covers only Afrikaans, English,
  Northern Sotho, Southern Sotho, Xhosa, and Zulu.
- If an intervention helps only one architecture, keep it clearly labelled as
  model-specific in the final comparison.
- Once the Mamba recipe is defensible, rerun or consolidate downstream
  evaluations so the final Mamba-vs-LLaMA comparison uses fair matched splits,
  prompts, metric scripts, decoding/result-selection rules, and documented
  model-specific exceptions.
- The final downstream suite should be treated as a confirmation phase: Mamba
  must be compared against both the previous best Mamba result and the matched
  best LLaMA result under fair conditions.
- Once the defensible Mamba recipe is selected, apply any confirmed shared
  decoder-only improvements to the LLaMA rerun where compatible, so the final
  result compares optimized recipes rather than a rescued Mamba against a stale
  transformer baseline.
- A fair final comparison must not compare a rescued Mamba recipe against a
  stale LLaMA result that lacks shared improvements found during Mamba rescue.
- Do not treat diagnostic improvements as final leaderboard claims until the
  matched final downstream evaluation has been run or an explicit exception is
  documented.

## Architecture Roadmap

1. **Transformer baseline**
   - First milestone: train and evaluate a transformer baseline.
   - This is the anchor for downstream metrics, training stability, and compute
     expectations.
   - Current comparison reference is the LLaMA-style decoder-only transformer
     baseline.

2. **Mamba**
   - Current milestone: train and evaluate a Mamba decoder-only model across
     the full downstream evaluation suite.
   - Mamba is being tested with the same broad downstream families where
     possible: classification, sequence labelling, headline generation, and
     translation.
   - The current work is still within decoder-only fine-tuning, decoding,
     formulation, metric-hygiene, and recipe-recovery gates. Architecture or
     pretraining changes should come only after these are exhausted.

3. **Future xLSTM / ExcelSTM-style model**
   - Planned future milestone: train an xLSTM-style model for the same
     downstream suite.
   - This should be compared against both the transformer baseline and Mamba.

4. **Other candidate architectures**
   - Add only when they answer a concrete comparison question: quality,
     data-efficiency, compute efficiency, memory footprint, multilingual
     transfer, or robustness under low-resource data.

## Metrics to Track Across Architectures

For every model family, record:

- Parameter count and trainable parameter count.
- Pretraining data size, language mixture, and training-token budget.
- Fine-tuning method: full fine-tune, LoRA, adapter, or other.
- Fine-tuning hyperparameters: learning rate, weight decay, schedule, warmup,
  batch size, gradient accumulation, epochs, label smoothing, sequence length,
  checkpoint-selection metric.
- Hardware and runtime: GPU type, number of GPUs, wall time, memory pressure,
  batch-size fallback, and failures.
- Evaluation harness version and task pack.
- Decoding strategy: greedy, beam, sampling, repetition penalty, length penalty,
  max tokens, and whether any constrained or guided output method was used.
- Task metrics:
  - Classification: accuracy, macro/weighted F1, per-language F1 where useful.
  - Sequence labelling: token accuracy, strict/length-penalized token accuracy,
    parseable-label rate, BIO/entity F1, non-O recall.
  - Generation: chrF, BLEU, ROUGE-L, plus raw-output sanity checks.
- Error profile: empty outputs, overgeneration, repetition, malformed labels,
  copied input, wrong language, no parseable tags, or task-template mismatch.
- Defensibility status: final, needs rerun, confounded, diagnostic only, or
  superseded.

## Current High-Level Status

### Public Release Usability

- The public `anrilombard/mzansilm-125m` Hugging Face model now has a repaired
  `model.safetensors` file and updated model-card usage instructions as of Hub
  commit `7f017bc71c53c19c1fd122e773ad1f60c5d30826`.
- The default Transformers 4.x load path was verified with the repaired
  safetensors file: no `meta` parameters remain after load and generation works.
- Transformers 5 still rejects the LLaMA config because the model uses explicit
  `head_dim=56` with `hidden_size=512` and `num_attention_heads=9`; the public
  instructions therefore pin `transformers>=4.52.4,<5`.

### Transformer Baseline

- Transformer/LLaMA baseline has already produced strong reference results for
  several generation tasks.
- Known references from the current local notes:
  - T2X Xho: LLaMA chrF about `53.98` best, with current-stack parity recheck
    around `53.48`.
  - AfriHG Xho: older local best chrF about `20.22`; current parity recheck
    `14.20`.
  - AfriHG Zul: older local best chrF about `23.00`; current parity recheck
    `21.56`.
- Exact final POS/NER baseline entries still need to be consolidated from the
  benchmark recordings/sheet before final writing.

### Mamba

- Mamba is not uniformly broken:
  - MasakhaNews monolingual parity recheck is strong after the task-template
    bug fix: Eng best F1 `0.773`, Xho best F1 `0.757`.
  - T2X is recipe-sensitive: recovered bqueawk-style full fine-tune reached
    chrF `32.18`, much better than weak current reproductions around chrF
    `20`, but still far below LLaMA around chrF `54`.
- Mamba remains weak or fragile on:
  - POS under official free-generation-style scoring and tag-sequence variants.
  - NER under free-generation BIO/tag-sequence scoring, despite signs of
    teacher-forced learning.
  - AfriHG, where recovered recipes improve over old weak runs but remain well
    below LLaMA.
- The final pre-hybrid POS/NER decoder-only rescue gate is now negative:
  D4 full-data atomic tag-sequence LoRA did not rescue POS or NER. Best D4 POS
  validation token accuracy is `0.1960` with exact length `0.0400` and high
  overgeneration on the better nonempty arm. D4 NER validation remains near the
  old narrow entity signal with non-`O` recall `0.1772`, BIO F1 `0.1769`, and
  all-`O` rate `0.9375`.

### xLSTM

- Strict-125M HF xLSTM is now viable in this repo after installing the official
  `xlstm` stack and adding xLSTM-specific chunked generation/eval handling.
  The current strict shape is `h736_l12_h4_chunk64` with `126,901,952`
  parameters.
- Corrected base MasakhaNER held-out test job `1126106` completed after the
  exact-head-dimension inference repair. Xhosa, Zulu, and Tswana are all
  `0.0000` F1 on every one of five prompts (best prompt = mean = range
  `0.0000`). Complete test aliases contain 1,000 Xhosa/Zulu and 996 Tswana
  rows per prompt. Treat this as a valid negative base result; generated
  outputs copy/continue input or extract no entities.
- Corrected POS test scoring has overturned the earlier "very strong POS"
  interpretation. The old POS scorer treated list-valued targets as multiple
  acceptable answers and inflated token accuracy. Under the patched serialized
  tag-sequence test pack, xLSTM multi-POS test is only moderate: best token
  accuracy is Tswana `0.328`, Xhosa `0.357`, and Zulu `0.390`, with exact
  length match below `0.20`. Treat earlier POS values around `0.93` as invalid.
- The fair full-base xLSTM run
  `xlstm_h736_ctx2048_native_4gpu_ddp_llama_budget_20260524` completed
  successfully as job `861849` in `1-05:59:05`, using a LLaMA-budget-style
  2048-context token-slot target of `4.758B` and reaching reported epoch
  `2.1513`. Its Trainer loss logs are affected by a Transformers/xLSTM
  loss-scaling issue under 4-GPU DDP, so use the clean/pretrain audit scripts
  rather than raw Trainer loss for final comparisons.
- Full xLSTM clean generation-loss validation is positive against the current
  Mamba base on all three audited generation gates, but still behind LLaMA:
  xLSTM final weighted NLL/PPL is T2X Xho `5.6584` / `286.70`, AfriHG Xho
  `5.7048` / `300.30`, and AfriHG Zul `5.8647` / `352.39`. Current Mamba base
  is `5.8515` / `347.77`, `6.0834` / `438.50`, and `6.4066` / `605.85`;
  LLaMA base is `4.1620` / `64.20`, `5.3254` / `205.48`, and `5.5885` /
  `267.32`.
- Full xLSTM status is now promising-positive as a Mamba replacement candidate
  for downstream evaluation. Repaired pretrain-loss audit confirms the same
  story: xLSTM final weighted NLL/PPL is `2.9250` / `18.63`, better than the
  current Mamba base at `3.8065` / `44.99` but behind LLaMA base at `2.3010` /
  `9.98`. The checkpoint-30000 loss-scale probe gives weighted NLL/PPL
  `2.6888` / `14.71`, confirming the raw 4-GPU Trainer eval-loss values were
  inflated and should not be read as perplexity. Final base-gate classification:
  positive versus Mamba, negative versus LLaMA; proceed to downstream xLSTM
  fine-tuning/evaluation if compute allows.
- The first strict-125M xLSTM 10k streaming base screen completed as jobs
  `861747` -> `861748`. It trained cleanly in `02:09:31`; the clean generation
  loss audit completed in `00:04:53`.
- xLSTM checkpoint-10000 has held-out trainer eval loss `3.310756`, down from
  `3.607671` at checkpoint-5000. On clean generation-loss validation it beats
  the current Mamba base on T2X Xho (`283.95` vs `347.77` PPL), but is worse
  than the current Mamba base on AfriHG Xho (`627.67` vs `438.50`) and AfriHG
  Zul (`852.06` vs `605.85`). LLaMA base remains substantially better on all
  three tasks (`64.20`, `205.48`, `267.32` PPL).
- Current xLSTM classification: ambiguous/promising. It is not a failed base
  screen like the recent fresh-Mamba and shallow-hybrid runs, but it is not yet
  a base-wide replacement for the current Mamba base or LLaMA. Prioritize
  xLSTM downstream adaptation or a longer xLSTM base screen before spending
  more compute on low-probability cheap Mamba rescue arms.

Current open gate:

- AfriHG lower-LR/no-smoothing is closed as a negative HPO gate: Xho chrF
  `3.19`, Zul chrF `4.87`, both below the recovered monolingual Mamba bests.
- Current best Mamba AfriHG is now the checkpoint-selected rerun: Xho chrF
  `11.53` and Zul chrF `13.57`.
- Checkpoint-selection-by-validation-generation-metric is producing useful
  signal. T2X Xho checkpoint-244 beat final on validation chrF (`32.68` vs
  `31.40`), AfriHG Xho checkpoint-656 beat final on validation chrF (`10.68`
  vs `9.54`), and AfriHG Zul checkpoint-892 beat final on validation chrF
  (`14.07` vs `13.09`). These three checkpoints should be promoted to official
  test-set lm-eval reruns. The official reruns are now submitted as serial
  L40S jobs `843499`-`843501`.
- Longer training remains a plausible next gate for Mamba generation if the
  checkpoint-selected test reruns improve, because all three validation wins
  came from later saved checkpoints rather than final/best-loss checkpoints.
- Official checkpoint-selected test reruns are complete and improve all three
  generation tasks checked:
  - T2X Xho checkpoint-244: chrF `32.61`, above prior Mamba best `32.18`.
  - AfriHG Xho checkpoint-656: chrF `11.53`, above prior Mamba best `9.90`.
  - AfriHG Zul checkpoint-892: chrF `13.57`, above prior Mamba best `12.09`.
- This closes the current saved-checkpoint selection gate, but it does not close
  the Mamba-vs-LLaMA generation gap. A small longer-training/continuation gate
  with validation chrF/ROUGE checkpoint selection is now justified before
  architecture-level claims.
- Root-cause forensics show Mamba generation is systematically shorter than
  LLaMA/reference outputs on T2X and AfriHG, with more repetition on T2X. The
  first longer-training gate has been submitted. Initial jobs `843512` ->
  `843513` and `843518` -> `843519` were pre-training Hydra override errors and
  were repaired; `843520` started training but failed at epoch eval because the
  inherited best-model metric still expected disabled trainer chrF metrics. The
  corrected T2X checkpoint-244 continuation jobs `843526` -> `843527`
  completed successfully. Official test chrF improved again to `33.1243`
  (BLEU `0.0577`, ROUGE-L `0.3015`), but output forensics still show Mamba
  under-generates: mean prediction length is about `10.22` tokens versus
  `27.37` reference tokens and about `12.74` for LLaMA.
- User approved deleting the old diagnostics checkpoint directory; it was
  removed, and `purequota` refreshed to scratch `72.7%`.
- The corrected serial AfriHG continuation gate completed on L40S one GPU:
  `843532` Xho finetune, `843533` Xho eval, `843534` Zul finetune, and repaired
  Zul eval `843571` after original eval `843535` failed before scoring due an
  `eval_model.adapter=null` config override. The continuation gate is negative:
  Xho chrF `10.3470` versus checkpoint-selected `11.5335`; Zul chrF `12.6551`
  versus checkpoint-selected `13.5726`. Both continuation runs shortened the
  outputs relative to their selected checkpoints. Do not promote AfriHG
  continuation; keep the checkpoint-selected AfriHG results as current Mamba
  best.
- Base-model lineage audit now has a matched held-out validation loss result:
  using the same tokenizer, same validation prompts, and same target token
  burden, Mamba base has substantially worse teacher-forced target likelihood
  than LLaMA base. Validation PPLs are T2X Xho `331.24` vs LLaMA `62.26`,
  AfriHG Xho `301.75` vs LLaMA `110.41`, and AfriHG Zul `401.35` vs LLaMA
  `157.94`. This makes base-model quality/lineage a real contributor to the
  generation gap, while tokenizer mismatch is ruled out as the architectures
  share the tokenizer.
- Decoder-only output-shape rescue is now checked on validation for current
  best Mamba generation checkpoints. T2X did not improve: stronger length
  controls caused severe overgeneration/repetition and lower chrF. AfriHG is
  more promising: beam5/lp1.2 improves Xho validation chrF from `10.95` to
  `12.81` and Zul validation chrF from `14.21` to `17.00`; Zul also improves
  ROUGE-L and length ratio cleanly. These AfriHG beam5/lp1.2 settings are
  validation-selected diagnostics and should be considered for official test
  reruns before changing any final Google Sheet rows.
- Base-issue validation control wave completed as jobs `848475` -> `848476`
  -> `848477`, all exit `0:0`, with artifacts pulled locally to
  `outputs/eval/diagnostics/mamba_base_issue_validation_20260519/`.
  The cleaner base generation-loss audit confirms the gap: Mamba base PPL is
  `347.81` vs LLaMA `64.20` on T2X Xho, `438.73` vs `205.48` on AfriHG Xho,
  and `606.04` vs `267.32` on AfriHG Zul. The selected Mamba fine-tuned
  checkpoints improve massively but remain behind task-specific LLaMA:
  T2X `18.73` vs LLaMA `5.31`, AfriHG Xho `21.04` vs `13.28`, AfriHG Zul
  `19.28` vs `10.79`; Mamba is close to SA-general LLaMA on AfriHG but not
  the task-specific LLaMA references. The held-out pretraining-style audit is
  the strongest root-cause signal: on the same `302278` validation tokens,
  Mamba base PPL is `45.00` versus LLaMA base `9.98`. Current recommendation:
  treat base checkpoint quality/training recipe as the main remaining root
  cause; run a short continued-pretraining recovery from the existing Mamba
  base before committing to a full new base pretrain.
- Scratch is now a blocker for any new HEX submission: quota reached `98.9%`
  after the HF dataset cache materialized. Top consumers are
  `/scratch/lmbanr001/masters` `64G` and `/scratch/lmbanr001/hf` `26G`. No
  cleanup was performed; deletion/cleanup approval is needed before further
  experiments.
- User approved proceeding; `/scratch/lmbanr001/hf` was deleted and scratch
  quota refreshed to `73.2%`. A bounded streaming continued-pretraining
  recovery wave is now active as jobs `848482` -> `848483` -> `848484` ->
  `848485` on L40S one GPU. The three recovery jobs run `300` steps from
  `anrilombard/sallm-mamba-125m` with LRs `5e-5`, `1e-4`, and `2e-4`, using HF
  streaming mode to avoid recreating the full dataset cache. The final job runs
  clean generation-loss audit over the three recovered checkpoints plus LLaMA
  base. This is a diagnostic/recovery gate, not a final fair-parity replacement
  for a fresh optimized base run.
- Base-model recommendation: if the short continued-pretraining wave improves
  Mamba base/conditional losses substantially, proceed to a fresh Mamba base
  HPO/retrain for the final architecture comparison. If it does not improve,
  skip longer continuation and move directly to fresh-base recipe search. A
  final defensible architecture comparison should include base validation PPL,
  downstream conditional PPL, task metrics, output shape metrics, token budget,
  GPU hours, parameter count, and inference throughput/memory.
- First continued-pretraining chain `848482` -> `848485` failed at eval due
  Mamba cache output hygiene (`Mamba2Cache` could not be fp32-cast by
  Accelerate), not due training divergence. The script was patched to disable
  `use_cache`, stranded jobs were cancelled, and the repaired chain is now
  `848486` -> `848487` -> `848488` -> `848489`; heartbeat updated accordingly.
- Second continued-pretraining chain `848486` -> `848489` passed step-100 eval
  (`eval_loss=3.8641`) but failed at checkpoint save because tied embeddings
  required `save_safetensors=false`. The script was patched with
  `save_safetensors=False`, stranded jobs were cancelled, and the active chain
  is now `848490` -> `848491` -> `848492` -> `848493`.
- Continued-pretraining recovery wave completed successfully as `848490` ->
  `848493`. On the 512-sample held-out pretraining slice, LR `5e-5` was best
  (`eval_loss=3.8625`), LR `1e-4` was nearly tied (`3.8635`), and LR `2e-4`
  was worse (`3.8902`). The downstream clean generation-loss audit did not
  improve over the starting Mamba base: LR `5e-5` PPLs were T2X `351.80`,
  AfriHG Xho `446.38`, AfriHG Zul `618.10`, versus the prior Mamba base
  `347.81` / `438.73` / `606.04`; higher LRs were worse. Conclusion: cheap
  300-step continued pretraining does not rescue the Mamba base. The next
  serious base step should be a fresh Mamba base recipe/HPO rather than longer
  blind continuation.
- Fresh Mamba base probe is active as `848494` -> `848495` -> `848496`, serial
  L40S one-GPU. It trains two fresh-from-config 3k-step Mamba probes
  (`lr2e-4_wu200` and `lr4e-4_wu200`) with streaming data and then runs the
  clean generation-loss audit against current Mamba and LLaMA bases. This is a
  bounded recipe diagnostic, not the final full Mamba base retrain.
- Fresh Mamba base probe completed successfully as `848494` -> `848496`. LR
  `4e-4` was clearly better than LR `2e-4` on held-out pretraining loss
  (`6.3838` vs `7.2619`), but both 3k-step fresh-from-random checkpoints were
  far worse than the current Mamba base on downstream clean generation loss
  (PPLs in the ~9.6k-17.3k range vs current Mamba base `347.81`/`438.73`/
  `606.04`). Interpretation: the short probe is useful for recipe direction
  but not a replacement base; a serious fresh Mamba base run needs a much
  longer token budget and should start from the faster-learning `4e-4` recipe
  direction unless longer-run stability argues otherwise.
- Longer fresh Mamba base pilot submitted as `850743` -> `850744`, serial L40S
  one-GPU. It trains fresh Mamba at LR `4e-4`, warmup `1000`, `20k` steps,
  saves/evals every `5k`, then audits clean generation loss for checkpoints
  `5k/10k/15k/20k/final` against current Mamba and LLaMA bases.
- Longer fresh Mamba base pilot completed successfully as `850743` -> `850744`
  and is negative for this exact pure-Mamba recipe. Best held-out pretraining
  loss was already at `checkpoint-5000` (`5.9383`), then worsened at `10k`
  (`6.0468`), `15k` (`6.1576`), and `20k` (`6.1647`). Clean generation-loss
  remains orders of magnitude worse than the current Mamba base and LLaMA base:
  best fresh PPLs are roughly T2X `15052`, AfriHG Xho `12102`, AfriHG Zul
  `11898`, versus current Mamba base `348`/`439`/`606` and LLaMA base
  `64`/`205`/`267`. This argues against simply extending the current
  fresh-from-random recipe; next pure-Mamba gate should be shape/recipe HPO in
  the same 120-130M band.
- The next pure-Mamba gate first ran as `852296` -> `852297`, serial one-GPU
  L40S. It keeps the parameter count matched but changes the base shape to
  `hidden_size=704`, `num_hidden_layers=24`, `expand=2`, `state_size=128`
  (`126,811,888` trainable params) and reruns the same 20k pretraining/audit
  structure. The first attempt reached the step-5000 evaluation point and then
  failed in the Mamba2 fast causal-conv kernel because the dynamic eval tensor
  layout did not satisfy the stride multiple-of-8 requirement; this is an
  implementation/kernel-layout issue, not a model-quality result. The repaired
  pad-to-multiple-of-8 chain ran as `854071` -> `854072` under run id
  `mamba_fresh_base_wide_e2_s128_pad8_20k_20260521`, but it failed at the same
  first-eval `causal_conv1d` stride constraint and the dependent audit was
  cancelled. Padding sequence length is therefore not enough; this is now an
  HF/Mamba2 fast-kernel layout blocker for the wide shape rather than a
  scientific loss result. The stale heartbeat was deleted.
- Mamba architecture research refresh on 2026-05-20 found hybrid Mamba2 to be
  the strongest current SSM-family literature direction, but the user wants to
  stay on pure Mamba/Mamba2 ("PMamba") for now before changing architecture.
  Immediate next gates should therefore be pure-Mamba base recipe/shape HPO,
  not hybrid. Mamba-3 is noted as a later research/feasibility item rather
  than an immediate pretraining-quality rescue.
- The wide pure-Mamba2 shape gate is now being retried with eval forced through
  the HF torch path while keeping fast fused CUDA training. The first
  torch-eval canary exposed a local patch bug; the second proved the binding
  fix but OOMed in torch eval at batch size `8` and context `2048`. The active
  corrected chain is now `854987` -> `854988` -> `854989` under run id
  `mamba_fresh_base_wide_e2_s128_torcheval3_20k_20260521`, using
  `eval_batch_size=1` and `eval_max_length=1024`, watched by heartbeat
  `sallm-mamba-wide-torch-eval-gate-watch`. This is still an
  implementation-repair gate, not yet a model-quality result.
- Root-cause rescue checklist has started under an active Codex goal. A1
  decoder-only POS/NER constrained label scoring completed as jobs `855022`-
  `855025`. It proves output shape alone is not enough: exact-length parseable
  outputs are `100%`, but POS remains very weak for both Mamba and LLaMA under
  this scoring prompt. Mamba NER sum scoring is a confirmed all-`O` collapse
  despite high token accuracy (`0.7420`): non-`O` recall and global entity F1
  are `0.0`. Mean scoring avoids all-`O` by overpredicting non-entity labels,
  not by recovering spans. Next narrow gate is A3 NER label-prior calibration.
- A3 Mamba NER scalar `O`-bias calibration completed as jobs `855026`-
  `855030` and is negative. Small penalties leave validation pure all-`O`;
  stronger penalties break all-`O` but flood the output with mostly `i-date`/
  `i-org` labels. Best global entity F1 is only `0.0019` at `o=-4.0` with
  `1` true-positive entity and `411` false-positive entities. Conclusion:
  NER is not rescued by a simple label-prior bias; next POS/NER gates should
  test constrained generation or formulation/training controls.
- A2 true prefix-constrained tag-sequence generation ran as jobs
  `855032`-`855035`, matched across Mamba/LLaMA and POS/NER. This tested the
  stricter version of the output-shape hypothesis by using actual
  decoder-only `model.generate` constrained to legal tag-label paths.
- A2 completed successfully and is negative as a rescue. It guarantees
  exact-length, parseable POS/NER outputs (`1.0` for all four runs), but Mamba
  POS still collapses mostly to `cconj`, Mamba NER floods `b-date`/`i-date`/
  `i-org`, and LLaMA controls are also weak under the same formulation.
  Conclusion: POS/NER failure is not simply malformed free generation; next
  POS/NER work should test formulation/training controls, with any shared
  improvements carried forward to both Mamba and LLaMA.
- C1 AfriHG official test rerun is active, testing whether validation-selected
  beam5/lp1.2 improves official AfriHG test metrics over the current
  checkpoint-selected beam5/lp0.7 baselines (Xho chrF `11.5335`, Zul chrF
  `13.5726`). If it improves Mamba, a matched LLaMA decode-variant check
  should be considered before final fair comparison.
- C1 is already positive for Xho: checkpoint-656 official test with beam5/lp1.2
  reached chrF `15.1702`, up from the prior checkpoint-selected Xho baseline
  `11.5335`. The first Zul job `855037` failed due Hydra override hygiene and
  was repaired/resubmitted as `855040`.
- C1 completed and is positive for Mamba AfriHG in both languages. Official
  test beam5/lp1.2 gives Xho chrF `15.1702` and Zul chrF `17.1245`, improving
  over checkpoint-selected baselines by about `+3.6` chrF each. Output length
  moves closer to references with low empty/repetition rates, but examples
  still show generic or wrong-focus headlines. Keep beam5/lp1.2 as the current
  Mamba AfriHG decoding recipe and run/plan a matched LLaMA decode check before
  final fair claims.
- C1b matched LLaMA AfriHG beam5/lp1.2 check is complete. The first attempt
  failed due stale LLaMA config checkpoint paths, but repaired durable-path jobs
  `855054` -> `855055` completed and were pulled locally. Beam5/lp1.2 is not a
  shared AfriHG protocol: it improves Mamba, but LLaMA Xho chrF `13.1149` and
  Zul chrF `20.8709` are below tracked LLaMA baselines and over-generate
  relative to concise headlines. Carry-forward tag: `Mamba-only`.
- C2 AfriHG hook/focus forensics is complete for the current C1/C1b evidence.
  Residual Mamba errors are mostly generic/off-topic and wrong-focus headline
  selection, not empty output: Xho off-topic/generic `37.2%` and too
  short/generic `31.9%`; Zul off-topic/generic `54.7%` and too short/generic
  `14.9%`. Matched LLaMA examples show stronger content preservation but
  over-generation under the Mamba-rescue decode setting, supporting semantic
  focus/headline planning and base quality as the remaining AfriHG issues.
- B3 T2X source-preservation forensics has started from current pulled
  outputs. Mamba checkpoint-selected T2X test has chrF `32.6125` versus LLaMA
  current-stack parity chrF `53.4796`. Crude exact source-token preservation is
  much lower for Mamba: entity coverage `0.241` vs LLaMA `0.511`, value
  coverage `0.202` vs `0.437`, and repetition `0.082` vs `0.034`. This
  supports entity/value preservation as a real T2X gap, while B1
  teacher-forced token-class loss remains needed for stronger causal evidence.
- B1 T2X teacher-forced token-class loss audit ran as job `855076` on one L40S
  GPU. The new script scores assistant/reference target
  tokens by exact source-token class (`source_entity`, `source_value`,
  `source_relation`, `other`) and compares current Mamba T2X continuation
  against LLaMA T2X opt-chrF on validation.
- B1 completed and strengthens the T2X root-cause story. On validation
  teacher-forced scoring, Mamba is worse than LLaMA on all target-token
  classes, but the gap is much larger for exact source entity/value tokens:
  entity NLL gap `+2.7656` with `15.89x` PPL ratio, value NLL gap `+2.3065`
  with `10.04x` PPL ratio, versus `other` token gap `+1.1783` with `3.25x`
  PPL ratio. This supports B2 placeholder/reinsertion as the next T2X rescue
  gate.
- D1 wide pure-Mamba2 base gate completed pretraining as job `854988`, but the
  dependent clean-loss audit `854989` failed before producing metrics. Final
  pretraining remains weak for the "train longer fixes it" hypothesis:
  checkpoint-5000 is still best with eval loss `5.6133`, while checkpoint-10000
  worsened to `5.6522`, checkpoint-15000 to `5.6954`, and checkpoint-20000 to
  `5.6942` despite improved training loss. The audit failed with the Mamba2
  fast-kernel channel-last stride error
  `causal_conv1d with channel last layout requires strides ... to be multiples
  of 8`, leaving the downstream clean generation-loss evidence missing. D1 is
  therefore incomplete/failed, not a defensible base-candidate decision yet.
  Next step is to repair and rerun only the clean-loss audit path once scratch
  pressure is safe; scratch was `89.3%` at the failure check.
- D1 clean-loss audit repair is now active as job `855987`. The local repair
  adds `--mamba-torch-forward` to `run_generation_loss_audit_clean.py`, using
  HF's torch Mamba2 eval path to avoid the fused causal-conv stride-layout
  failure. Local syntax/Ruff/help checks passed, the repair-only submitter was
  synced to HEX, and the job started on one L40S GPU with scratch still high
  at `89.3%`. A fresh 30-minute heartbeat
  `sallm-mamba-d1-clean-loss-repair-watch` is watching `855987` and should
  pull/summarize/classify D1 when `summary.json` is available.
- D1 wide pure-Mamba2 torch-eval base gate is now closed negative. Repaired
  audit job `855987` completed `0:0`, artifacts were pulled, and
  `base_gate_summary.md` was generated. The repair fixed the audit path, but
  the model-quality result is poor: best D1 fresh-wide clean-loss PPLs are T2X
  `8470`, AfriHG Xho `11315`, AfriHG Zul `11720`, versus current Mamba base
  `348`/`439`/`606` and LLaMA base `64`/`206`/`268`. D1 should not feed D3;
  no new base candidate exists. The D1-negative branch is now active: run E1
  implementation parity and B3/C3 formulation/decoding gates before designing
  another D2 base HPO matrix. Scratch remains high at `89.3%`, so no more HEX
  submissions without explicit user approval or cleanup.
- Post-D1 readiness audit: E1/B3/C3 submitters still pass local syntax checks,
  and E1/B3 diagnostic scripts pass local py_compile/Ruff. HEX queue is empty,
  but scratch remains `89.3%`; no job was submitted. Recommended next move is
  E1 first, because it is the lowest-footprint remaining gate and directly
  tests the HF-vs-official Mamba implementation-parity confounder. B3/C3 should
  wait until E1 closes or scratch pressure is reduced.
- E1 HF-vs-official Mamba logits parity completed as job `856253` and is a
  parity failure/caveat. HF and official `mamba_ssm` parameter counts match and
  the state load has no missing/unexpected keys, but logits are not close
  (`all_logits_close_at_1e-4=false`), greedy next token differs on two of three
  prompts, and max absolute logit deltas are `9.31`-`16.01`. This does not
  invalidate HF-Mamba vs HF-LLaMA downstream comparisons, because SALLM used
  the HF path, but it weakens broad architecture claims about official Mamba2.
  Continue B3/C3 as practical HF-Mamba rescue gates and keep final wording
  scoped to the implementation/config path unless official-path parity is
  repaired later.
- B3 T2X source-preservation decoding diagnostic completed as repaired job
  `856269` after two immediate structured-config failures were fixed. It is
  negative as a rescue. Mamba baseline validation chrF is `28.06`; the
  source-checklist prompt is only `28.15` and lowers entity/value coverage.
  Repetition-control variants reduce repetition but collapse chrF to `20.14`
  and `21.74`. LLaMA baseline remains much stronger: chrF `45.89`, entity
  coverage `0.763`, value coverage `0.594`, versus Mamba `0.410`/`0.320`.
  Conclusion: T2X source entity/value preservation is a real Mamba gap, but
  simple decoder-only prompt/checklist or repetition-control decoding does not
  fix it. Do not carry B3 prompt/decoding variants into final reruns.
- C3 AfriHG focus-prompt validation completed as repaired serial L40S chain
  `856290` -> `856293`. It is a positive Mamba validation candidate, especially
  for Zul: Mamba Xho improved from validation chrF `12.8140` to `13.1258`, and
  Mamba Zul improved from `17.0020` to `18.4924`. Matched focus-prompt LLaMA
  controls scored Xho chrF `13.2699` and Zul chrF `19.2154`, but the current
  pulled evidence lacks a matched LLaMA non-focus validation baseline, so this
  is not yet a shared protocol improvement. Carry-forward tag:
  `Mamba validation candidate / shared status ambiguous`; confirm on official
  Mamba test before final recipe adoption.
- C4 AfriHG focus-prompt official Mamba test confirmation is active as jobs
  `856306` -> `856307`, watched by heartbeat
  `sallm-c4-mamba-afrihg-focus-test-watch`. Scratch was `89.3%` at submission;
  top consumers were checked first and no deletion was performed. C4 compares
  focus-prompt test metrics against C1 official beam5/lp1.2 baselines: Xho chrF
  `15.1702`, Zul chrF `17.1245`.
- C4 completed and is ambiguous-to-negative as a final AfriHG recipe gate. Xho
  regressed from C1 chrF `15.1702` to `14.4983` and ROUGE-L `0.0572` to
  `0.0553`; Zul improved only slightly from chrF `17.1245` to `17.3358` and
  ROUGE-L `0.0736` to `0.0782`. Do not adopt `focus_v1` as final. Keep C1
  beam5/lp1.2 as current Mamba AfriHG official-test recipe; record focus_v1 as
  a validation-only/language-specific prompt idea.
- D2 pure-Mamba base HPO is now locally prepared but not submitted. Added
  `scripts/submit_d2_mamba_base_hpo_screen_2026_05_22.sh`, a one-arm-at-a-time
  screen with canary -> 10k train -> clean generation-loss audit. Local `bash -n`
  and usage checks pass. Do not submit while scratch is `89.4%`; cleanup/approval
  is needed before adding checkpoint-heavy base runs. Preferred first arm after
  cleanup is `current_lr2e4_wu2000_10k`.
- Cleanup audit for D2 is ready. Queue is empty; scratch is still `89.4%`.
  Four audited base-HPO checkpoint dirs could free roughly `10.6G`:
  old 3k probe, old expand4/state64 20k, D1 wide 20k, and the D1 canary. The
  three scientific runs have local summaries/loss rows and recorded negative
  decisions; the canary is not a scientific result. No deletion has been
  performed.
- D2 was patched to reduce scratch footprint before submission:
  `run_mamba_fresh_pretrain_streaming.py` now supports `--skip-final-save`, and
  the D2 submitter uses it so screening arms do not duplicate `final_model`.
  D2 will audit `checkpoint-5000` and `checkpoint-10000` only. Local validation
  passes (`bash -n`, `py_compile`, Python Ruff, and help output). Still do not
  submit at scratch `89.4%` without cleanup/approval.
- Lower-footprint D2 code is now also staged on HEX and passes remote
  `bash -n`/`py_compile`; no job submitted. Scratch remains `89.4%`, queue is
  empty, and deletion still requires explicit approval.
- Added an advisor-ready current evidence matrix:
  `sallm_memory/mamba_root_cause_rescue_evidence_matrix.md`. D2 submitter now
  supports `--dry-run`, and the remote dry run for first arm
  `current_lr2e4_wu2000_10k` works without submitting jobs.
- User approved cleanup for D2. Deleted only the audited old base-HPO checkpoint
  dirs, repaired the D2 scratch guard to use `/scratch/slurm/bin/purequota`,
  and submitted first D2 arm `current_lr2e4_wu2000_10k` as jobs `856320` ->
  `856322`. Heartbeat `sallm-d2-mamba-base-hpo-screen-watch` is active. Latest
  light status: canary `856320` completed, train `856321` running, clean-loss
  audit `856322` dependency-pending, scratch about `79.6%`. No D2 result yet.
- D2 first arm has a negative checkpoint-5000 signal: train loss `6.3609`,
  eval loss `6.5037`, eval PPL about `667.6`, worse than D1 wide checkpoint-5000
  eval loss `5.6133`. Keep waiting for checkpoint-10000 and clean-loss audit
  before final D2 classification.
- D2 checkpoint-10000 remains weak: train loss `6.2177`, eval loss `6.4936`,
  eval PPL about `660.9`. It is only slightly better than checkpoint-5000, so
  the current-shape lower-LR/longer-warmup arm is unlikely to be a credible
  base candidate unless the clean-loss audit surprisingly disagrees.
- D2 first arm `current_lr2e4_wu2000_10k` closed negative. Clean generation-loss
  checkpoint-10000 PPLs are far worse than the current Mamba base: T2X Xho
  `11925.1` vs `347.8`, AfriHG Xho `26758.1` vs `438.5`, AfriHG Zul
  `28854.2` vs `605.9`; LLaMA base remains lower still (`64.0`, `205.6`,
  `267.5`). Do not use this D2 base for D3. Lower LR/longer warmup alone does
  not rescue the current pure-Mamba shape.
- D2 second arm `wide_lr2e4_wu2000_10k` was submitted as jobs `861386` ->
  `861388`.
  This tests the wide expand2/state128 shape with lower LR `2e-4` and warmup
  `2000`, because the current-shape LR/warmup rescue failed. Scratch was
  `81.0%` at submission. This is retained as historical submission context;
  the arm has since closed below.
- D2 second arm `wide_lr2e4_wu2000_10k` closed negative. Jobs `861386` ->
  `861388` all completed `0:0`; artifacts were pulled and summarized locally.
  Trainer eval improved mildly by checkpoint-10000 (`eval_loss=6.4619`, PPL
  about `640.3`), but clean generation-loss rejected the arm: best wide
  checkpoint is checkpoint-5000 with PPLs T2X Xho `9983.6`, AfriHG Xho
  `21768.7`, AfriHG Zul `20378.4`, still far worse than current Mamba base
  (`347.8`, `438.5`, `605.9`) and LLaMA base (`64.0`, `205.6`, `267.5`).
  Do not use the wide D2 base for D3. Recommended base follow-up decision:
  stop pure-Mamba base HPO and report the base-rescue path as negative; the
  remaining conservative wide arm (`wide_lr1e4_wu1000_10k`) is optional only if
  an exhaustive ablation table is worth another low-probability run.
- B2 T2X placeholder/reinsertion diagnostic is now running as job `855304` on
  one L40S GPU. It replaces exact source entity/value strings with
  placeholders, generates with matched Mamba/LLaMA checkpoints, deterministically
  reinserts the original strings, then reports placeholder preservation,
  entity/value preservation, and final chrF/BLEU/ROUGE. This directly tests
  whether the B1/B3 T2X source-token gap is fixable by removing copy burden.
- B2 completed and is negative as a Mamba rescue. Mamba generated zero expected
  placeholders in the `216` examples where the placeholdered reference expected
  at least one placeholder, giving reinserted chrF `8.2455`. LLaMA generated at
  least one expected placeholder in `87.5%` of those examples, exact placeholder
  set `29.6%`, and reinserted chrF `27.2327`, but value preservation remains
  weak. Carry-forward tag: `LLaMA-only` diagnostic improvement / `Negative`
  Mamba rescue.
- B3 is now prepared but not submitted. The local diagnostic script
  `scripts/run_t2x_source_preservation_decoding_diagnostic.py` and submitter
  `scripts/submit_b3_t2x_source_preservation_decode_2026_05_21.sh` passed
  syntax/help/ruff checks. It will run matched Mamba/LLaMA validation variants
  for baseline beam3/lp1.2, greedy repetition control, conservative beam
  repetition control, and a source-preservation checklist prompt, then score
  chrF/BLEU/ROUGE plus entity/value preservation and repetition. The two B3
  scripts were narrowly rsynced to HEX. Hold submission while D1 is active and
  scratch remains above `85%`.
- Latest live state at 2026-05-21 23:15 SAST: C1b is closed; D1 job `854988`
  is still running around step `9140/20000` on L40S and D1 audit `854989` is
  dependency-pending. Scratch remains `86.9%`, with checkpoints `54G`, results
  `15G`, and logs `92M`; there is still no D1 artifact beyond
  `checkpoint-5000/trainer_state.json`.
- Follow-up live state at 2026-05-21 23:22 SAST: D1 job `854988` remains
  healthy around step `9528/20000`; `854989` is still dependency-pending.
  Scratch/top consumers are unchanged, and no new D1 artifact is available.
- Follow-up live state at 2026-05-21 23:25 SAST: D1 job `854988` remains
  healthy around step `9670/20000`; `854989` is still dependency-pending.
  Scratch/top consumers are unchanged, and no new D1 artifact is available.
- Follow-up live state at 2026-05-21 23:27 SAST: D1 job `854988` remains
  healthy around step `9750/20000`; `854989` is still dependency-pending.
  Scratch/top consumers are unchanged, and no new D1 artifact is available.
- Follow-up live state at 2026-05-21 23:29 SAST: D1 job `854988` remains
  healthy around step `9834/20000`; `854989` is still dependency-pending.
  Scratch/top consumers are unchanged, and no new D1 artifact is available.
- Follow-up live state at 2026-05-21 23:33 SAST: D1 job `854988` is still
  `RUNNING` on L40S `srvrocgpu012` after about `03:20:48`; `854989` remains
  dependency-pending. Scratch is still `86.9%` with top consumers checkpoints
  `54G`, results `15G`, and logs `92M`. The log tail is active inside an eval
  loop with no traceback, but the only discoverable D1 artifact remains
  `checkpoint-5000/trainer_state.json`, so no new local summary or decision is
  available yet.
- Follow-up live state at 2026-05-21 23:35 SAST: checkpoint-10000
  `trainer_state.json` appeared and was pulled locally. Checkpoint-10000 train
  loss improved to `5.2401`, but eval loss worsened to `5.6522` from
  checkpoint-5000's `5.6133`; trainer best still points to checkpoint-5000.
  This remains better than the earlier failed/current expand4/state64 20k
  run's 5k eval loss `5.9383`, but D1 is no longer a clean monotonic-loss
  story. Wait for checkpoint-15000/20000 and the clean generation-loss audit
  before deciding whether this is a real base-candidate improvement.
- Follow-up live state at 2026-05-21 23:38 SAST: D1 job `854988` is still
  running around step `10212/20000` (`51%`) after the checkpoint-10000 eval.
  Scratch remains `87.6%` with checkpoints `55G`, results `15G`, and logs
  `92M`; `854989` is still dependency-pending. No new D1 artifact is available
  beyond checkpoint-5000/checkpoint-10000 trainer states.
- Follow-up live state at 2026-05-21 23:39 SAST: D1 job `854988` remains
  healthy around step `10298/20000` (`51%`); `854989` is still
  dependency-pending. Scratch/top consumers are unchanged at `87.6%`,
  checkpoints `55G`, results `15G`, logs `92M`. No new artifact is available.
- Follow-up live state at 2026-05-21 23:41 SAST: D1 job `854988` remains
  healthy around step `10382/20000` (`52%`); `854989` is still
  dependency-pending. Scratch/top consumers are unchanged at `87.6%`,
  checkpoints `55G`, results `15G`, logs `92M`. No new artifact is available,
  so wait for the heartbeat rather than continuing manual minute-by-minute
  checks.
- Follow-up live state at 2026-05-21 23:43 SAST: D1 job `854988` remains
  healthy around step `10470/20000` (`52%`); `854989` is still
  dependency-pending. Scratch/top consumers are unchanged at `87.6%`,
  checkpoints `55G`, results `15G`, logs `92M`. No new artifact is available;
  leave the next check to the existing heartbeat unless a user decision is
  needed.
- Follow-up live state at 2026-05-21 23:48 SAST: D1 job `854988` remains
  healthy around step `10736/20000` (`54%`) on L40S `srvrocgpu012`; `854989`
  is still dependency-pending. Scratch/top consumers remain `87.6%`,
  checkpoints `55G`, results `15G`, logs `92M`. Artifact search still finds
  only checkpoint-5000/checkpoint-10000 trainer states, so no new D1 summary or
  recipe decision is available yet.
- Follow-up live state at 2026-05-21 23:51 SAST: D1 job `854988` remains
  healthy around step `10885/20000` (`54%`) on L40S `srvrocgpu012`; `854989`
  is still dependency-pending. Scratch/top consumers remain unchanged at
  `87.6%`, checkpoints `55G`, results `15G`, logs `92M`. There is still no
  checkpoint-15000, final pretrain summary, or clean-loss diagnostic, so D1
  remains active with no new decision.
- Follow-up live state at 2026-05-21 23:52 SAST: D1 job `854988` remains
  healthy around step `10972/20000` (`55%`) on L40S `srvrocgpu012`; `854989`
  is still dependency-pending. Scratch/top consumers remain unchanged at
  `87.6%`, checkpoints `55G`, results `15G`, logs `92M`. There is still no
  checkpoint-15000, final pretrain summary, or clean-loss diagnostic.
- Follow-up live state at 2026-05-21 23:54 SAST: D1 job `854988` remains
  healthy around step `11050/20000` (`55%`) on L40S `srvrocgpu012`; `854989`
  is still dependency-pending. Scratch/top consumers remain unchanged at
  `87.6%`, checkpoints `55G`, results `15G`, logs `92M`. There is still no
  checkpoint-15000, final pretrain summary, or clean-loss diagnostic.
- Follow-up live state at 2026-05-21 23:55 SAST: D1 job `854988` remains
  healthy around step `11124/20000` (`56%`) on L40S `srvrocgpu012`; `854989`
  is still dependency-pending. Scratch/top consumers remain unchanged at
  `87.6%`, checkpoints `55G`, results `15G`, logs `92M`. There is still no
  checkpoint-15000, final pretrain summary, or clean-loss diagnostic. Leave
  further watching to the heartbeat until a new artifact or failure appears.
- E2 Mamba generation parity smoke is now running as job `855312`. It checks
  representative Mamba T2X and AfriHG Xho generation under cache on/off and
  batch1/batch4 greedy settings to rule out evaluation-setting instability as
  a confounder. Immediate log tail showed Mamba CUDA kernels available.
- E2 completed and found an implementation-hygiene issue, not a rescue:
  Mamba generation did not crash, but outputs were not perfectly invariant to
  cache/batch settings. T2X exact match to batch1/cache-off was `7/8` under
  cache-on or batch4 variants; AfriHG Xho was `6/8` for cache-on variants and
  `7/8` for batch4/cache-off. Final Mamba generation evaluations should pin
  and document cache/batch settings, preferably with conservative
  batch1/cache-off reruns or explicit harness-setting documentation.
- Fair-comparison guardrail: any decoder-only improvement that helps both
  Mamba and LLaMA should be recorded as a shared protocol improvement and
  carried into later optimized LLaMA reruns where compatible, not treated as a
  Mamba-only win. Likewise, a LLaMA-only improvement should stay in the notes
  as a useful later LLaMA recipe candidate and as evidence about which fixes do
  or do not transfer to Mamba. Once the Mamba recipe is defensible enough for
  the downstream suite, final Mamba-vs-LLaMA reporting must rerun or align both
  architectures under the same splits, prompts, metrics, decoding/selection
  rules, and documented shared improvements before updating the comprehensive
  Google Sheet. Each checklist result should now carry a `Mamba-only`,
  `LLaMA-only`, `Shared`, or `Negative` tag so later recipe carry-forward is
  auditable. The final comparison has two distinct questions: did the chosen
  recipe improve over the previous best Mamba result, and is the optimized
  Mamba recipe fairly competitive with the optimized LLaMA transformer
  baseline?
- D5 Mamba-2 hybrid base screen completed as `861602` -> `861604` and is
  negative. The `hybrid_126m_waleffe8attn` arm (`126357846` params, LR `4e-4`,
  warmup `2000`, `10000` steps, 2 attention layers out of 24 / 8.3%
  attention) improved held-out pretraining eval from `5.7042` at
  checkpoint-5000 to `5.6123` at checkpoint-10000, which is better than recent
  pure-Mamba fresh HPO arms. However, clean generation-loss remains orders of
  magnitude worse than the current Mamba and LLaMA bases:
  checkpoint-10000 PPLs are T2X Xho `12364.7`, AfriHG Xho `14020.5`, AfriHG
  Zul `14634.8`, versus current Mamba base `347.8`/`438.5`/`605.9` and LLaMA
  base `64.0`/`205.6`/`267.5`. Do not use D5 as a base candidate; record it
  as architecture-follow-up evidence that shallow sparse attention improves
  training loss shape but does not rescue downstream conditional likelihood at
  this budget. Do not generalize this to all hybrids until the custom hybrid
  attention implementation is audited for positional encoding/RoPE and intended
  block structure.

### xLSTM

- xLSTM literature and citation notes are recorded in
  `sallm_memory/xlstm_architecture_literature.md`. The strongest directly
  relevant recipe signal is Beck et al. 2024: 125M-ish xLSTM uses embedding
  dim `768`, `24` blocks, `4` heads/head dim `384`, context `2048`, AdamW
  betas `(0.9, 0.95)`, eps `1e-5`, grad clip `1.0`, warmup `750`, cosine
  decay to 10% peak LR, weight decay `0.1`, and no positional encoding.
- HF/official xLSTM viability probes are complete enough to justify a strict
  125M xLSTM screen. HF xLSTM is viable for training on HEX only when
  `xlstm`/`mlstm-kernels` are installed. The repo-compatible `hidden=768`,
  12-layer HF shape has `135,368,544` params and passed forward/backward,
  finite gradients, save/load, and a 6-step overfit smoke on L40S. After the
  kernels were installed, the stricter `hidden=736`, 12-layer HF shape also
  passed forward/backward/save-load and tiny overfit at `126,901,952` params,
  so it is now the preferred xLSTM base-screen shape under the user's
  125M-parameter constraint.
- Official NX-AI xLSTM vanilla backend can instantiate and forward/backward
  under a comparable 12-layer config, but it is larger (`143,818,064` params)
  and not parameter-matched to the HF baseline. Official CUDA sLSTM backend
  failed extension build on HEX, so it is not the next mainline path.
- Current xLSTM generation status: stock HF `generate()` fails chunk-size
  assertions after the prompt chunk (`65` or `1` tokens not divisible by chunk
  size `64` depending on cache mode), but the SALLM evaluation path now has an
  xLSTM-only chunked generation helper and a passing smoke test. For base
  screening, use the streaming HF `hidden=736`, 12-layer path rather than TRL's
  full-dataset materialization path.
- Final xLSTM 3-epoch base retrain `xlstm_h736_ctx2048_native_4gpu_ddp_3epoch_resume_20260531`
  completed cleanly on L40S. The main train `880318` reached `67498/67498`
  steps in `1-17:39:46` with `train_loss=3.85037` and trainer-state epoch
  `3.0871`; the afterany resume job `880319` resumed from `checkpoint-67498`
  and no-opped cleanly; audits `880320` and `880321` completed cleanly. The
  held-out pretrain-style audit is positive versus Mamba but still behind
  LLaMA: xLSTM final mean NLL/token `2.97127`, PPL `17.48`; Mamba base PPL
  `44.99`; LLaMA base PPL `9.98`. The clean generation-loss audit is not
  competitive with LLaMA: xLSTM final PPLs are T2X Xho `479.63`, AfriHG Xho
  `497.88`, AfriHG Zul `568.79`, versus LLaMA `64.20`/`205.48`/`267.32`.
  Against current Mamba, xLSTM is worse on T2X Xho and AfriHG Xho but slightly
  better on AfriHG Zul (`568.79` vs `605.85`). Classify this run as useful
  architecture evidence, not a downstream base replacement for LLaMA.
- The completed xLSTM base export was published privately to Hugging Face as
  `anrilombard/sallm-xlstm-125m-native-3epoch-20260531` at commit
  `ba2ff845335c8cbf750f8f6f3ebc09468008fef9`. After publication, approved
  pretraining scratch cleanup removed old xLSTM canaries/intermediate
  checkpoints and the `anrilombard___mzansi-text-tokenized` cache; the retained
  current-run scratch payload is `final_model`, `fresh_pretrain_summary.json`,
  and best checkpoint `checkpoint-60000`.
- xLSTM downstream evaluation is complete through mono, multilingual, and
  general waves. The downstream story is split: task-specific mono/multi
  adapters are alive, but the single general adapter is not a good universal
  adapter. Corrected News evals recovered strongly for task-specific adapters
  (mono English/Xhosa best F1 about `0.913`/`0.920`; multilingual English/Xhosa
  about `0.881`/`0.919`). POS originally looked very strong in the multi gate,
  but the 2026-06-11 audit found that the old lm-eval target-list setup
  inflated the reported token accuracies. The final corrected constrained
  test-split POS result now shows a more defensible moderate xLSTM score, not
  the earlier near-`0.93` story: multilingual xLSTM best token accuracy is
  Tswana/Xhosa/Zulu `0.7245`/`0.6933`/`0.7313`.
  NER is modest but nonzero for task-specific adapters. In contrast, the general adapter largely
  collapses structured extraction/classification: News drops to English/Xhosa
  best F1 `0.6582`/`0.1929`, NER best F1s are near zero
  (`0.0094`/`0.0038`/`0.0035`), and POS best token accuracies are only
  `0.0933`/`0.0133`/`0.0200`. Treat task-specific mono/multi results as the
  defensible xLSTM downstream comparison; do not use the general adapter as
  xLSTM's best downstream form. One general eval, `afrihg_eng`, failed because
  no AFriHG English CSV exists on GitHub, so that is a dataset/config issue
  rather than model behavior.
- The xLSTM T2X mixed-source rescue did not transfer from validation to a
  strong official test result. Job `921485` completed cleanly on the official
  T2X Xhosa test split with chrF `34.1600`, BLEU `0.0454`, and ROUGE-L
  `0.2974`: only a small improvement over the prior xLSTM mono T2X official
  result around chrF `33.719`, and still far below LLaMA around chrF `53.976`.
  Treat T2X as an unresolved source-binding/structure problem for xLSTM; the
  next defensible rescue direction is structural delexicalization or
  source-placeholder reinsertion rather than broad decoding HPO.
- Final constrained POS audit update, 2026-06-12: evaluate MasakhaPOS as a
  closed-label token-tagging task on the test split by scoring every token
  against the fixed UPOS label set under the same tuple prompts. This confirms
  the old around-`0.93` xLSTM POS values were inflated, but POS is not a
  complete xLSTM failure under the correct protocol. Best multilingual
  constrained token accuracy by language is:
  - LLaMA: Tswana/Xhosa/Zulu `0.8460`/`0.8220`/`0.8409`.
  - xLSTM: Tswana/Xhosa/Zulu `0.7245`/`0.6933`/`0.7313`.
  - Mamba: Tswana/Xhosa/Zulu `0.2340`/`0.0674`/`0.0402`.
  Interpretation: LLaMA remains clearly strongest for POS under closed-label
  scoring; xLSTM is meaningfully competent but behind LLaMA; Mamba remains
  very weak even when free-generation formatting is removed. The Google Sheet
  multilingual tuned POS cells were updated and verified on 2026-06-12; base,
  monolingual, and general POS cells should not be reinterpreted from this
  constrained multilingual wave.
- xLSTM POS HPO held-out update, 2026-06-18: the HPO-selected multilingual
  xLSTM POS adapter passed the corrected constrained MasakhaPOS `test` gate.
  It uses the same closed-label token-logprob protocol, tuple prompts, and
  languages (`tsn`, `xho`, `zul`). Held-out test token accuracy is:
  - xLSTM HPO-selected multilingual POS: Tswana/Xhosa/Zulu
    `0.7906`/`0.7666`/`0.7855`; macro `0.7809`, micro `0.7824`.
  This supersedes the earlier corrected xLSTM multilingual POS baseline
  (`0.7245`/`0.6933`/`0.7313`) for the tuned xLSTM POS comparison, but it does
  not invalidate the LLaMA/Mamba corrected constrained POS results. Running
  mono/multi/general xLSTM HPO wave2 jobs are validation-only until their own
  selected adapters pass held-out test.
- xLSTM POS wave2 mono held-out update, 2026-06-19: the selected monolingual
  wave2 adapters for Tswana, Xhosa, and Zulu passed the corrected constrained
  MasakhaPOS `test` gate. They use the same closed-label token-logprob
  protocol, tuple prompts, and test split. Held-out token accuracy is Tswana
  `0.7650`, Xhosa `0.7839`, and Zulu `0.8201`; exact sequence accuracy is
  Tswana `0.0228`, Xhosa `0.0611`, and Zulu `0.1090`. These supersede the
  previous unproven monolingual POS sheet cells. Xhosa and Zulu improve over
  their promoted multilingual HPO-selected scores (`0.7666` and `0.7855`);
  Tswana does not improve over the promoted multilingual score `0.7906`.
  Wave2 multilingual held-out POS is defensible but lower than the already
  promoted multilingual HPO result: macro token accuracy `0.7676`, micro
  `0.7638`, with Tswana/Xhosa/Zulu `0.7462`/`0.7758`/`0.7808`, so it was
  recorded but not promoted. General remains HPO validation-only until a
  complete validation artifact and selected held-out eval exist.
- xLSTM HPO-selected held-out downstream update, 2026-06-25: the L40S
  post-HPO test wave completed cleanly for SIB, MasakhaNews, InjonGoIntent,
  AfriHG, T2X Xho, and MasakhaNER as jobs `952247`-`952252`. The held-out
  story is mixed rather than a broad HPO win. MasakhaNews remains strong
  (best F1 English/Xhosa `0.8686`/`0.9238`) and InjonGoIntent is strong for
  English/Xhosa but weaker for Sotho/Zulu (best F1
  `0.9064`/`0.8463` versus `0.6998`/`0.6226`). MasakhaNER is nonzero but still
  modest under flexible extraction (best F1 Tswana/Xhosa/Zulu
  `0.3174`/`0.2538`/`0.2545`). Generation remains below the strongest
  transformer references: T2X Xho chrF `30.81`, AfriHG Xho chrF `15.20`, and
  AfriHG Zul chrF `17.10`. SIB is not rescued by this HPO selection: best
  per-language F1 is only Afrikaans/English/Northern Sotho/Southern
  Sotho/Xhosa/Zulu `0.1260`/`0.1415`/`0.0800`/`0.1032`/`0.1175`/`0.0888`.
  Use the detailed 2026-06-25 note for run provenance and prompt means before
  making sheet-level leaderboard updates.
- xLSTM MasakhaNews protocol correction, 2026-07-28: historical News adapters
  and headline values predated selective training of newly added chat-role
  tokens and classification left truncation, so they are quarantined rather
  than treated as final architecture evidence. The corrected monolingual Xhosa
  recipe selected checkpoint-17 on validation F1 (`0.4095883`) and evaluated
  held-out test exactly once in job `1128440`. Weighted F1 across prompts
  P1-P5 is `0.2758567`/`0.2687654`/`0.1953027`/`0.1916278`/`0.2376085`.
  The canonical headline is explicitly **best prompt** P1 `0.2758567`; mean
  `0.2338322`, range `0.1916278-0.2758567`. This best-of-five test-prompt
  result is descriptive, not an unbiased estimate. Corrected multilingual
  validation selected epoch 4 / checkpoint-272 with
  `eval_classification/all_f1=0.6941930`; its single frozen held-out test job
  `1128810` is queued and must complete before multilingual News promotion.
- Task-head diagnostic update, 2026-06-29/30: held-out supervised task-head
  probes show that the poor generative adapter results do not mean the base
  representations lack task signal. In the matched trainable task-head setting,
  xLSTM is best on NER, Intent, and POS, while LLaMA is best on SIB; the same
  winner pattern appears in the frozen-backbone control. Three-seed core
  results after rerunning Mamba seed 13 with fast-path dependencies are:
  NER-all F1 xLSTM `0.4032 +/- 0.0236`, LLaMA `0.3553 +/- 0.0397`, Mamba
  `0.2491 +/- 0.0105`; SIB-all macro-F1 LLaMA `0.7355 +/- 0.0149`, xLSTM
  `0.7030 +/- 0.0032`, Mamba `0.6205 +/- 0.0115`. The frozen control summary
  confirms the pattern before backbone adaptation: xLSTM beats LLaMA/Mamba on
  NER (`0.3670` vs `0.2526`/`0.2414`), Intent (`0.6577` vs
  `0.4677`/`0.4835`), and POS (`0.6440` vs `0.6210`/`0.5844`), while LLaMA
  wins SIB (`0.6876` vs xLSTM `0.6491`, Mamba `0.6232`). Keep these results
  separate from the original decoder-only generative benchmark: they support a
  "formulation/adaptation failure, not base representation failure" conclusion
  and make task heads a defensible dissertation branch.
- GatedDeltaNet base update, 2026-07-09/14: the shallow/wide 125M GDN base
  pretraining run completed successfully and was published privately as
  `anrilombard/sallm-gated-deltanet-125m-shallowwide-4x40-20260707`. Final
  training job `997801` completed with exit `0:0` after `1-19:05:30`; final
  artifacts include `final_model`, `checkpoint-37500`, and `checkpoint-38590`.
  Final logged test loss was `3.3288254737854004`. Base zero-shot evaluation
  is now operationally covered across the intended packs after several launcher
  and evaluator repairs: generation path, lm-eval schema, POS JSON
  serialization, FLA TileLang dispatch, AfriMMLU import patch, and
  non-instruction tokenizer chat-template handling. The completed base eval
  directory contains results for AfriGSM, AfriMMLU, AfriXNLI, AfriSenti,
  InJoGoIntent, MasakhaNER, and MasakhaNEWS, plus earlier monolingual/base
  packs for T2X, AfriHG, MasakhaNER, MasakhaNEWS, MasakhaPOS, and SIB. Early
  base generation metrics are weak, as expected for an untuned 125M base:
  T2X Xhosa chrF `6.2571`, AfriHG Xhosa chrF `9.6993`, and AfriHG Zulu chrF
  `9.5554`. Treat GDN as a completed base-pretraining and evaluation-stack
  milestone, not yet as a competitive downstream result. Fine-tuned adapter
  evaluations remain pending because the first adapter eval pass retained old
  Mamba `peft_adapter` settings and was invalidated; corrected adapter evals
  are being rerun separately.
- GatedDeltaNet corrected POS update, 2026-07-27: the held-out constrained
  MasakhaPOS matrix now has defensible task-adapter results under the same
  closed-label token-logprob protocol used for the architecture comparison.
  Best-prompt multilingual Tswana/Xhosa/Zulu token accuracy is
  `0.8033`/`0.7855`/`0.8050`; best-prompt monolingual results are
  `0.8112`/`0.2361`/`0.7303`, where Xhosa is an audited genuine `NOUN`
  collapse; best-prompt general-adapter results are
  `0.8181`/`0.8102`/`0.8226`. The prompt means, ranges, and winning prompt IDs
  remain in `notes/2026-07-27-best-prompt-reporting.md`.
  The historical list-target POS scores remain invalid and quarantined.
  Corrected base POS is also complete and promoted. Best-prompt base
  Tswana/Xhosa/Zulu token accuracy is `0.2005`/`0.2217`/`0.2427`; all three
  genuinely collapse toward `NOUN` (`82.5%`/`89.0%`/`89.3%` of best-prompt
  predictions) and have zero exact-sequence accuracy.
- INJOngo Intent protocol audit, 2026-07-27: existing Mamba and LLaMA
  decoder-only Intent rows are not defensible architecture evidence. Summed
  continuation log-probability structurally favours short verbalizers; the old
  validation set included all 622 English test rows; those English test texts
  also occur in train; and the Zulu dev split contains corrupted labels. The
  shared local root fix now uses mean token log-probability, balanced
  prompt/label validation caps, test-text decontamination, and deterministic
  stratified validation derived from the remaining train rows. Focused tests
  pass, but retraining and final promotion remain pending.
- GatedDeltaNet 2/3-shot base closeout, 2026-07-27: all 23 completed non-NER
  artifacts from arrays `1118020/1118021` passed structural audit and were
  promoted using best-prompt headlines. Three-shot News F1 is Eng `0.1592`
  and Xho `0.2083`; SIB best-prompt F1 spans `0.1695-0.1893` (2-shot) and
  `0.1676-0.2094` (3-shot). AfriXNLI reaches `0.3317/0.3567`, AfriMMLU
  `0.2720/0.2360`, and AfriMGSM `0.0120/0.0160` for 2/3-shot respectively.
  Zulu few-shot Intent remains quarantined because its demonstration labels
  are corrupt. Full prompt-level provenance is in
  `notes/2026-07-27-gdn-base-fewshot-closeout.md`.
- GatedDeltaNet base NER few-shot closeout, 2026-07-27: two/three-shot arrays
  completed and were recorded as held-out validation diagnostics, not test
  results. Best-prompt F1 remains near zero: Xhosa `0.0132/0.0032`, Zulu
  `0.0021/0.0021`, and Tswana `0.0027/0.0022`. Outputs are complete but
  frequently copy or repeat prompts and fail extraction. Few-shot prompting
  therefore does not rescue base GDN NER.
- GatedDeltaNet monolingual Xhosa NER recovery, 2026-07-28: reducing effective
  batch from `256` to `64` produced a validation-selected checkpoint at epoch
  14 (`eval_all_f1=0.1343`). Its single frozen held-out test reached
  **best-prompt F1 `0.2780` on prompt 1**; five-prompt mean `0.1818`, range
  `0.0976-0.2780`. This replaces the prior monolingual zero and shows genuine
  extraction signal, although label confusion and repetition remain.
- GatedDeltaNet Xhosa NER HPO screen, 2026-08-01: the validation-only
  24-trial Bayesian/Hyperband screen completed cleanly in HEX job `1150968`.
  The top two recipes are `tyvese0z` (`eval/all_f1=0.6113`) and `b888dh4l`
  (`0.6086`), far above the earlier narrow recovery validation result.
  Three-seed stability is final: `tyvese0z` mean `0.6085`, sample SD `0.0097`,
  range `0.5978-0.6165`; `b888dh4l` mean `0.5858`, sample SD `0.0336`, range
  `0.5472-0.6086`. The validation winner is therefore `tyvese0z`; its
  representative seed-42 `checkpoint-552` is frozen rather than selecting the
  lucky highest seed. This strongly supports optimization/capacity as a major
  confound in the prior mono-versus-multi NER gap, though matched exposure
  controls are still required. Official frozen held-out job `1151321`
  completed: explicitly labelled best-prompt P5 F1 is `0.6287`, prompt mean
  `0.6029`, and range `0.5192-0.6287`. The verified result is promoted in
  `GatedDeltaNet Results!E4` and `Comparison Data!I44`; full prompt-level
  provenance and the descriptive-not-unbiased warning are retained.
- xLSTM base closeout update, 2026-07-27: POS job `1118354` is running.
  NER `1118353` and Intent `1118355` failed before scoring because the
  adapter-free xLSTM path left the model in chunkwise training mode and hit
  non-64-divisible sequence lengths. The shared runner fix is implemented
  locally with a focused passing regression test; cluster sync and isolated
  NER/Intent retries remain pending.
- xLSTM base POS closeout, 2026-07-27: job `1118354` completed exit `0:0` and
  passed the matched held-out constrained POS audit. Best-prompt base
  Xhosa/Zulu/Tswana token accuracy is `0.1536`/`0.1456`/`0.0746`; predictions
  overwhelmingly collapse to `PUNCT` and exact-sequence accuracy is zero.
  These are valid negative base-model results, now promoted with full
  four-prompt provenance.
- xLSTM corrected Xhosa News recovery, 2026-07-28: validation-only selection
  froze epoch-1 checkpoint `17` at validation F1 `0.4096`; its one held-out
  test reached **best-prompt F1 `0.2759` on P1**, mean `0.2338`, range
  `0.1916-0.2759`. This replaces the historical monolingual `0.9197` row,
  which is quarantined because it predates the role-token selective-training
  and left-truncation repairs. The corrected value and full provenance are
  promoted in `XLSTM Results!C3,E3,I3:J3`.
- xLSTM corrected multilingual News recovery, 2026-07-28: validation-only
  selection froze epoch-4 checkpoint `272` at validation F1 `0.6942`. Its
  single held-out test reached English **best-prompt F1 `0.2786` on P1**
  (mean `0.2586`, range `0.2458-0.2786`) and Xhosa **best-prompt F1 `0.5114`
  on P4** (mean `0.4910`, range `0.4674-0.5114`). These replace the historical
  multilingual `0.8810/0.9238` values, which predate the role-token and
  truncation repairs. Corrected values and provenance are promoted in
  `XLSTM Results!F2:F3,I2:J3`.
- xLSTM corrected monolingual English News recovery, 2026-07-28:
  validation-only selection froze epoch-2 checkpoint `104` at validation F1
  `0.5921`. Its single held-out test reached **best-prompt F1 `0.1700` on
  P2**, mean `0.1608`, range `0.1480-0.1700`. This replaces the historical
  monolingual `0.9126` value, which predates the role-token and truncation
  repairs. The corrected value and provenance are promoted in
  `XLSTM Results!E2,I2:J2`.
- xLSTM corrected base Intent closeout, 2026-07-29: adapter-free job `1130553`
  completed the official held-out test with mean continuation token
  log-probability scoring. Best-prompt F1 is English `0.0032`, Xhosa `0.0022`,
  Zulu `0.0037`, and Sotho `0.0042`. This is a defensible near-chance negative
  base result, now promoted in `XLSTM Results!D16:D19` with full prompt
  provenance. Historical tuned Intent cells remain quarantined pending clean
  validation-selected recovery.
- xLSTM corrected monolingual Xhosa AfriHG, 2026-07-30: validation-only job
  `1139443` selected checkpoint-328; frozen held-out job `1139444` produced
  chrF `17.9540` from one task-native prompt on 1,305 official test rows.
  This replaces the quarantined target-truncation-exposed historical value and
  is promoted with artifact hashes and output-quality counts.
- xLSTM corrected monolingual Zulu AfriHG, 2026-07-30: validation-only job
  `1139445` selected checkpoint-892 at validation chrF `19.5712`; frozen
  held-out job `1139446` produced chrF `19.6499` from one task-native prompt
  on 1,776 official test rows. The artifact has 36 empty predictions, 1,739
  unique predictions, and zero null predictions. This replaces the
  quarantined target-truncation-exposed historical value and is promoted with
  artifact and adapter hashes.

- Storage/phase decision, 2026-07-30: scratch capacity is expected to expand
  to approximately `300 GB`, and the user authorizes retaining the best
  validation-selected fine-tuned adapters/checkpoints for later analysis. The
  current deadline phase is therefore **benchmark-only**: finish already
  running training/evaluation dependencies, preserve their best artifacts,
  and do not launch new fine-tuning or broad HPO now. A later phase may rerun
  the full fine-tuning matrix from the retained models/configurations; this
  does not change the current held-out-test, provenance, or promotion rules.
- GatedDeltaNet corrected monolingual MasakhaNews closeout, 2026-07-30:
  validation-selected training array `1142362` and frozen held-out test array
  `1142368` completed under the matched seven-label, five-prompt News contract.
  Promoted best-prompt weighted F1 is English `0.2094` (P1; mean `0.1853`,
  range `0.1613–0.2094`) and Xhosa `0.5621` (P5; mean `0.4717`, range
  `0.3248–0.5621`). These replace the quarantined legacy mono-News zeros;
  full prompt-level provenance, artifacts, hashes, and the shared 2048-token
  truncation caveat are in `notes/2026-07-30.md` and the source-cell notes.
- xLSTM General matched-suite closeout, 2026-07-31: jobs `1137033–1137036`,
  isolated replacements `1145318/1145456`, and matched constrained POS
  `1145480` now provide the complete 41-row General matrix under the corrected
  protocol. Canonical `XLSTM Results` General cells are promoted with
  task-native held-out metrics and full source-cell provenance. Matched POS
  best-prompt token accuracy is Xhosa `0.7481`, Zulu `0.7685`, and Tswana
  `0.7699`; corrected mean-token Intent remains near chance
  (`0.0045–0.0086` F1); T2X Xhosa chrF is `18.9044`; AfriHG Xhosa/Zulu is
  `8.8127/7.6026`. Best-of-test-prompts is explicitly descriptive and not an
  unbiased estimate. The legacy free-generation POS artifact remains
  diagnostic-only and the stale validation-split POS artifact remains
  quarantined.
- The NER compatibility correction note is frozen at SHA-256
  `30ee60dcf73a01328932c88c6fcd5c4d3d2f6116d4c1db97952f358286298c11`.
  CPU probe `1282013` completed `0:0`, proving exact pinned Tsn/Xho/Zulu
  train/validation counts `1441/499`, `1441/817`, and `1441/836` through the
  alias-only bindings while canonical hashes stayed unchanged. Isolated Tsn
  `1282016` and Xho `1282017` replacements verified `715` immutable files,
  loaded/tokenized their exact train and five-template validation rows, and
  entered A100-80GB training. Intent a1/a2 plus these two jobs use all four
  eligible cards.
- Intent a1 `1281334` completed `0:0` in `05:51:19` with exact `4,155`-row
  coverage and retained checkpoint `2643`. CPU verifier `1282232` proved exact
  retained-to-final equality across `424` keys and `71,762,560` values. Eight
  of nine reduced seed-42 candidates are terminal-valid; Intent a2 remains.
- NER Zulu Mono `1282231` filled the released card after the frozen alias and
  recipe preflight. Manifest SHA-256 is
  `39dcb588f8cb3a429100f439127558dee39508869610b2d94621278f87d8f8ec`;
  all `715` files verified and exact `1,441` train plus `4,180` five-template
  validation rows loaded/tokenized. Tsn/Xho/Zulu and Intent a2 use all four
  eligible A100-80GB cards.
- Four-architecture Base matrix closure, 2026-07-31: the final genuine
  coverage gap, LLaMA/Transformer Base MasakhaNER Xhosa, is complete from job
  `1145524_1` and promoted in `Transformer Results!D4`. All five official-test
  prompts are `0.0000` F1 on 1,000 Xhosa documents, so the explicitly labelled
  best-prompt headline, mean, and range are all `0.0000`. Raw output collapses
  into repetitive malformed role/special-token strings and produces no
  extractable entities, making this a verified negative rather than a missing
  run. `Comparison Data!F4/O4` now resolve to `0.0000/4/4`; no `3/4` coverage
  row remains. Structural N/A rows remain non-experiments.

## Durable update — architecture label correction and pure-GDN priority (2026-08-02)

- The verified existing checkpoint is `Qwen3NextForCausalLM`, trained from
  scratch with 33 layers total: 25 `linear_attention` GDN layers and 8
  `full_attention` layers, hidden size 640, and 10 heads. The exact label is
  **GDN–Attention Hybrid (Qwen3Next implementation)**. Its results remain a
  valid secondary hybrid arm only and cannot fill the pure-GDN comparison slot;
  this hybrid implementation was deliberate in the early notes, while the
  pure-GDN wording was later reporting drift, not a new implementation.
- Pure GDN is installed on HEX through `flash-linear-attention==0.5.1` and
  `fla-core==0.5.1`, with `GatedDeltaNetConfig`, `GatedDeltaNetModel`, and
  `GatedDeltaNetForCausalLM` under
  `/home/lmbanr001/masters/sallm/.venv/lib/python3.12/site-packages/fla/models/gated_deltanet/`.
  With `attn=None`, every block is GatedDeltaNet; no official pretrained
  weights exist, so matched pretraining from scratch is required. The current
  `gated_deltanet` -> Qwen3Next mapping in `src/main/sallm/models/registry.py`
  is the integration defect to correct.
- The next architecture priority is pure GDN integration, not broad new hybrid
  HPO: preserve hybrid General equal-mixture control `1160839`, verify BF16
  forward/backward, FLA fast path, packed context, DDP, save/load, and
  generation, then run one A100-80GB canary before resumable full pretraining.
  Retain A100-80GB exclusivity and never tune on test.
- The parameter-matched pure shape is now frozen from CPU-only HEX job
  `1164029`: 21 all-GDN layers (`attn=None`), hidden size 512, intermediate
  size 1536, four 128-wide heads, `expand_v=2`, tied 65,536-token embeddings,
  and 2,048-token context, for exactly `127,425,448` parameters. This preserves
  the tied-embedding convention of the other 125M baselines. It is a config
  selection only until the required BF16/DDP/FLA-kernel/save-load/generation
  A100-80GB canary passes.
- Hybrid General equal-mixture control `1160839` completed cleanly `0:0` after
  2,728 updates with final validation loss `0.2754413566702204`. Preserve it as
  a **GDN--Attention Hybrid (Qwen3Next implementation)** control; do not use it
  as evidence for the pure-GDN architecture.
- Pure-GDN integration is locally accepted on
  `research/pure-gdn-baseline-20260802` at signed commit `4feaadf`: 80 tests
  and a third fresh Sol/high `ship` review cover pure/hybrid routing, wrapped
  2,048-token streaming canary batches, the FLA kernel probe, and DDP-safe
  saves. No HEX sync or GPU canary occurred because external-write approval is
  still required.
- The broad-pretraining budget is not yet scientifically frozen. The common
  tokenizer/optimizer/context intentions are aligned, but executed histories
  differ or are incomplete (LLaMA `48,403` steps; xLSTM `67,498` steps at
  epoch `3.0871`; current Mamba Hub lineage not fully recovered). Treat the
  pure config's five epochs as provisional; recover executed token/update
  contracts and pre-register one explicit token budget before full launch.

## Update Rule

- Update this progress note only when a result changes the high-level story.
- Put run-by-run details, logs, failures, and hypotheses into
  `notes/YYYY-MM-DD.md`.
- Current working comparison view: `final_results.html`.
- Current unresolved Mamba gap plan:
  `notes/2026-05-18-mamba-gap-investigation.md`.
- Advisor-ready Mamba classification/generation/base-model meeting note:
  `notes/2026-05-21-advisor-meeting-mamba-classification-generation-base.md`.
- Supervisor follow-up to-do list for Mamba output/root-cause forensics:
  `notes/2026-05-21-supervisor-todos-mamba-output-forensics.md`.
- Current post-advisor priorities from 2026-06-11: verify suspiciously strong
  xLSTM POS and InjongoIntent evals end to end; run focused xLSTM HPO for the
  weakest defensible tasks; train/evaluate GatedDeltaNet under the same
  downstream/test-set protocol for the next architecture comparison.
- 2026-06-16 sequencing update: do not start xLSTM HPO until a light
  evaluation/provenance cleanup is done. The corrected constrained POS result
  explains Mamba's updated POS failure as label-prior collapse under
  closed-label scoring (`DET`/`NOUN`/`PART`), not a free-generation formatting
  issue. HPO is now underway; GatedDeltaNet follows the xLSTM HPO/test gate.
- 2026-06-16 NER reformulation closeout: constrained BIO tag-sequence
  MasakhaNER test diagnostics did not rescue NER for any architecture and do
  not replace the official span-generation NER rows. Macro entity F1 remains
  near zero: Mamba `0.0119`, LLaMA `0.0353`, xLSTM `0.0367`. The models fail
  differently: Mamba overpredicts date labels, LLaMA overpredicts `b-per`, and
  xLSTM overpredicts repeated inside labels such as `i-loc`/`i-per`/`i-org`.
  This changes the root-cause story: poor NER is not only a span-copy or
  free-generation parsing problem; even legal closed-label BIO scoring exposes
  severe label-prior/calibration collapse. Diagnostic rows were written to the
  Google Sheet `Eval Provenance` tab only, marked `diagnostic_only`; official
  benchmark rows were left untouched.
- 2026-07-29 advisor comparison correction: General is a first-class variant,
  never an auxiliary ranking. The canonical `Variant Comparison` sheet now
  uses task-native held-out metrics and normalizes each cell against the best
  architecture × variant on the same task-language. Headline variant averages
  use only fully verified 12/12 Mono/Multi/General rows; exact scores and
  incomplete coverage remain visible without promotion.
## Pure-GDN runtime gate status — 2026-08-02 evening

- Pure arm remains FLA `GatedDeltaNetForCausalLM`, `attn=None`, exact `127,425,448` parameters; prior Qwen3Next results remain explicitly **GDN--Attention Hybrid**, not pure GDN.
- A100-80GB jobs `1164608`, `1164702`, and `1164809` collectively verify BF16 forward/backward, direct FLA chunk-kernel backward, two-rank startup, Hub streaming with `datasets 4.x`, exact model-size validation, a real wrapped `2048`-token batch, TileLang backend selection, and one optimizer step. They do not yet verify a complete canary/save/reload/generation chain.
- Runtime defects found and fixed on `research/pure-gdn-baseline-20260802`: Slurm submit-dir resolution (`8521c6ec...`), grouped Hydra overrides (`3b50f218...`), Hub `List` metadata compatibility (`581e95cf...`). A reviewed but currently uncommitted trainer fix marks only pure GDN `accepts_loss_kwargs=False` and defaults resolved iterable `dispatch_batches=False` when unset; affected suite `16` tests plus fresh Sol verdict `ship`.
- Decisive two-step canary rerun is pending because the external-write approval service rejected both staging and HEX sync with `unknown_parameter: input[6].namespace`. Exact-file sync must be explicitly re-authorized; do not bypass the approval boundary.
- Full pretraining remains gated after canary by storage (`~89.9/100 GB` used) and a preregistered matched token/update budget. Selection is validation loss/validation metrics only; never tune on held-out test.
- Fresh-lane update: reviewed trainer/provenance changes are committed on `research/pure-gdn-baseline-20260802` as `6812806bb92928cb78a912eb8ce2c67f86317e77`. Exact HEX rsync was separately rejected as remote source export, so no decisive canary exists yet. Latest quota is `89/100 GB` (`90.0%`), owned queue is empty, and three A100-80GB GPUs are nominally free. The prior `input[n].namespace` approval failure is publicly tracked in open `openai/codex#31754`; do not conflate it with the later rsync policy rejection.

## Pure-GDN hardware gate accepted — 2026-08-02

- Decisive two-GPU A100-80GB canary `1165989` completed `0:0` in `00:08:01`
  using pure FLA `GatedDeltaNetForCausalLM` with `attn=None`, exactly
  `127,425,448` parameters, and no concurrent owned A100-40GB or L40S work.
  Both wrapped 2,048-token optimizer steps completed with ordinary-scale
  losses `11.1940/11.1951`; both iterable validation passes completed at
  `eval_loss=11.1919603348`; the TileLang FLA backward path was exercised.
- Both step checkpoints and `final_model` saved without rank collisions.
  AutoModel reload, exact state-dict roundtrip, and deterministic greedy
  generation passed. Accepted artifact root:
  `/scratch/lmbanr001/masters/sallm/checkpoints/sallm-pure-gdn-125m-canary/a10080-1165989`
  (`1.7G`; final config SHA256
  `114fd68f5f522b3627aa10fc34d029e9e200adb5ebb228e8f826c64b93eddf5b`).
- The implementation/hardware gate is complete. Full pure-GDN pretraining is
  still not authorized to start: current scratch is `91/100 GB` (`91.6%`),
  and the executed LLaMA/Mamba/xLSTM token/update histories must be recovered
  and one matched token budget preregistered. Pretraining recipe selection is
  validation-only and held-out tests remain untouched.

## Pure-GDN three-epoch pretraining contract frozen — 2026-08-02

- After recovering the corrected xLSTM training standard and explicit user
  approval, full pure-GDN pretraining is frozen to `67,498` optimizer steps at
  global sequence batch `48` and context `2,048`, exactly `6,635,323,392`
  token slots. This is the corrected xLSTM-matched three-epoch-equivalent
  contract; the earlier `48,403`-step LLaMA-matched run was stopped and must
  not be reported as the final pure-GDN pretraining arm. Hub streaming is
  mandatory. Tokenizer vocabulary is `65,536`; optimizer is LR
  `4e-4`, cosine, `2,000` warmup steps, weight decay `0.01`, Adam betas
  `0.9/0.95`, and max gradient norm `1.0`.
- Reporting must call this a **6.635B token-slot, xLSTM-matched
  three-epoch-equivalent budget**, not “three epochs of GDN training”: Hub
  streaming plus repetition means it is not three literal finite-dataset
  passes.
- Model identity remains pure FLA `GatedDeltaNetForCausalLM`, `attn=None`,
  exactly `127,425,448` parameters. Historical Qwen3Next results remain
  explicitly **GDN--Attention Hybrid**, never evidence for this pure arm.
- Pretraining selection and monitoring use validation loss only. Downstream
  recipe/checkpoint selection must also remain validation-only before any
  one-time official held-out test.
- Verified cold archives now preserve `gdn_afrihg_hpo_r1` and `news_hpo_r1`
  on Kombuys with exact per-file SHA256 source/destination manifests (`119`
  and `126` files). CPU job `1166480` removed only those verified HEX roots
  plus the reproducible `655M` Triton cache; accepted canary `1165989` and all
  canonical winners remain untouched.
- The superseded `48,403`-step jobs `1166554/1166555` were explicitly
  cancelled after checkpoint `1000` and step `1599`; their artifacts are
  preserved as diagnostic provenance only. The corrected run must start from
  step zero under a fresh run ID so its cosine schedule is defined over all
  `67,498` steps. Selection remains validation-loss-only.
- Corrected primary job `1167989` is running from step zero on two A100-80GB
  GPUs, with sole `afterany` continuation `1167990`, under run ID
  `a10080-3epoch-20260802`. Early gates reconfirmed exact `127,425,448`
  parameters, real 2,048-token batches, the TileLang FLA backward path, and
  ordinary step-10 loss `11.1935`. Expected completion is about 52--55 hours
  across the two Slurm segments.
- HEX scratch expansion is now verified live by `purequota`: capacity is
  `300 GB`, with `90 GB` used (`30.1%`). The former 100 GB storage gate is
  resolved; preservation and copy-before-delete provenance rules still apply.

## Pure-GDN downstream protocol frozen — 2026-08-05

- First validation-only lane is joint English/Xhosa MasakhaNews classification
  on canonical job `1179847`'s verified `final_model`; held-out test remains
  untouched.
- Pure-GDN LoRA uses all recurrent mixing projections (`q/k/v/a/b/g/o_proj`)
  and MLP projections (`gate/up/down_proj`), rank 16, alpha 32, dropout 0.05.
  This architecture-complete target set was fixed before metrics and replaces
  the diagnostic-only `q_proj`/`v_proj` smoke choice.
- Frozen LR grid is `3e-5`, `8e-5`, `1.5e-4`, `3e-4`; select checkpoints and
  recipe only on validation macro-F1. Run serially on Kombuys RTX 3080 Ti GPU
  1; do not disturb RTX 5090.
- Private-checkpoint transfer to Kombuys was not authorized, so execution
  remains on HEX. Schema-only failure `1183133` did not load model/data;
  corrected trial-0 job `1183134` runs on one A100-80GB with canonical job
  `1179847` final model and has passed pure-model, LoRA attachment, dataset,
  and TileLang backward startup gates.
- Frozen MasakhaNews LR chain is `1183134` (`3e-5`) -> `1183160` (`8e-5`) ->
  `1183161` (`1.5e-4`) -> `1183162` (`3e-4`), all one A100-80GB and serial
  `afterok`. Trial 0 steadied near `1.73 s/step`; no test evaluation is queued
  before validation macro-F1 selects the recipe.
- Trial `1183134` epoch-1 trainer validation is finite
  (`eval_loss=13.105342222839257`); the validation-only classification callback
  is the dominant runtime at about 14 minutes per language. Worst-case frozen
  grid ETA is now about 36--40 hours, or roughly 11--15 hours if early stopping
  closes each trial after the minimum useful epochs. Held-out test untouched.
- Trial `1183134` validation macro-F1 is unchanged at
  `0.10811573554007083` for epochs 1 and 2 (Eng `0.043573575434573804`, Xho
  `0.17265789564556785`). This is provisional validation-only evidence; epoch
  3 is scoring and should trigger patience-2 early stopping if unchanged,
  reducing the likely four-trial ETA to about 7--9 hours from 00:04 SAST.
- Trial `1183134` subsequently completed `0:0` after epoch 3 with the same
  validation macro-F1 `0.10811573554007083` and saved its PEFT adapter; no
  traceback/OOM/NCCL failure occurred. Successor `1183160` (LR `8e-5`) released
  immediately on one A100-80GB and improved finite trainer validation loss
  from `12.9564904332` at epoch 1 to `12.6311550257` at epoch 2 while its
  validation classification callback continued. The remaining frozen chain
  is still `1183161` -> `1183162`; held-out test remains untouched until all
  four validation trials select one recipe.
- Trial `1183160` completed `0:0` after three epochs with validation macro-F1
  `0.06244142349681912`, below trial `1183134`'s current leader
  `0.10811573554007083`; both adapters are preserved. Trial `1183161` is now
  priority-pending for A100-80GB capacity (Slurm estimate 2026-08-08 00:54:36
  SAST) and `1183162` remains dependency-pending. No alternate-family job or
  duplicate was launched; held-out test remains untouched.
- Slurm subsequently identified `1183161`'s wait as `AssocGrpGRES` rather than
  generic priority: the A100-80GB association GPU limit is the explicit
  blocker. The estimated start remains 2026-08-08 00:54:36 SAST and the frozen
  serial chain remains intact; no alternate-family or duplicate job was
  launched.
- The wait cleared early: `1183161` started on one A100-80GB and passed healthy
  pure-GDN LoRA startup, so no hardware migration is warranted. A read-only
  authenticated Hugging Face inventory check found no canonical pure-GDN base
  repository; only the older GDN--Attention Hybrid lineage is present. A new
  private Hub upload of the verified `final_model` is pending explicit external
  write/repository approval.
- Hardware policy is now one A100 memory class at a time: A100-80GB or
  A100-40GB, never concurrent owned lanes. Let `1183161` finish on A100-80GB;
  pending successor `1183162` was cancelled before startup and no further
  fine-tuning will run. Next gate is the preregistered one-time zero-shot base
  evaluation across the established 16-lane held-out matrix, scheduled only
  after `1183161` on A100-40GB. Adapter evaluation remains paused. The exact
  base artifact will also be published privately as
  `anrilombard/sallm-pure-gdn-125m` with fresh-download hash verification.
- Pending fine-tuning successor `1183162` is cancelled. The A100-40GB base
  evaluation array is fully dry-run-verified but cannot be dependency-queued:
  Slurm rejects the future GRES with `AssocGrpGRES` while `1183161` occupies
  A100-80GB. Submit it immediately after `1183161` exits; no held-out data has
  yet been touched. The Hub publisher is ready but no external repository was
  created because exact payload/destination approval remains outstanding.

## Pure-GDN deterministic streaming-boundary failure — 2026-08-04

- Jobs `1167989` and `1167990` both reached step `29,954` and then failed with
  the same 30-minute NCCL watchdog timeout: rank 0 entered a one-element
  `_ALLGATHER_BASE` while rank 1 entered a 786,944-element gradient `ALLREDUCE`
  at the same collective sequence. Training and validation were numerically
  healthy through checkpoint 29,000 (`eval_loss=3.5169744492`). This is a
  reproducible finite-stream rank-divergence defect, not evidence of model
  divergence or a random GPU failure.
- The smallest local correction repeats only iterable training splits governed
  by an explicit positive `max_steps`; validation remains finite. Focused tests
  pass (`18 passed`). Do not resubmit blindly: first cross a deliberately tiny
  finite-stream boundary in a two-rank canary, then resume the canonical run
  from the verified `checkpoint-29000` if that gate passes.
- Two-A100-80GB boundary canary `1179842` passed the exact former failure point,
  completed step `30,010`, and passed model reload, exact save/load integrity,
  and deterministic generation (`0:0`). The finite-stream repeat fix is thus
  accepted for this failure. Strict `afterok` continuation `1179847` has been
  released and is resuming the canonical 67,498-step run from
  `checkpoint-29000`; all selection remains validation-loss-only.
- Canonical continuation `1179847` subsequently crossed the same boundary,
  saved `checkpoint-30000`, and improved validation loss from `3.5169744492`
  at 29k to `3.5101182461` at 30k. This confirms the recovery in the actual
  scientific run, not only the diagnostic canary. Continue ordinary
  validation/checkpoint monitoring through final-model verification.
- Canonical pure-GDN pretraining is complete: job `1179847` exited `0:0` at
  exactly `67,498` steps (`6,635,323,392` token slots) with final validation
  loss `3.4211919308`. The final artifact is pure FLA
  `GatedDeltaNetForCausalLM`, `attn=None`, exactly `127,425,448` parameters.
  AutoModel reload, exact state-dict roundtrip, and deterministic greedy
  generation passed. Canonical weights SHA256 are
  `37bcc3d080dafcc5d8812821b969a46ef3208b84341bc32ae2c54dd94a9d8108`.
  Downstream work may now begin on validation only; held-out test remains
  untouched.

## Pure-GDN base evaluation active — 2026-08-06

- Fine-tuning is frozen after clean completion of final allowed validation
  trial `1183161`; `1183162` is cancelled and adapter held-out evaluation is
  paused. Base-model characterization now has priority.
- The A100-40GB association accepts `gpu:ampere`, not `amperemk`; use only
  `ampere` jobs for this gate and never overlap them with owned A100-80GB work.
- The lm-eval include-path shim must derive its task root from the imported
  `lm_eval.tasks` package. A repo-`.venv` hard-code fails whenever execution
  imports lm-eval from the scratch environment.
- Corrected zero-shot base jobs `1184503/1184504/1184505` are healthy on
  A100-40GB and use the canonical local pure-GDN final model directly with no
  adapter. Continue the preregistered 16 lanes in non-repeating tranches of at
  most three; never use observed held-out metrics to alter the protocol.
- Canonical Hugging Face publication remains blocked pending explicit approval
  for private repo `anrilombard/sallm-pure-gdn-125m` and the exact staged
  payload.
- Full downstream mission is now Base first, then validation-selected
  Mono/Multi/General, with the canonical sheet updated after each verified
  result. Never use held-out metrics to select or change a recipe.
- Base job `1184503_0` completed cleanly and its MasakhaNews results are
  recorded in the new `Pure GatedDeltaNet Results` tab, separate from the old
  hybrid tab. Jobs `1184504_1`, `1184505_2`, and `1184508_3` continue the
  base matrix on A100-40GB; keep at most three active and submit remaining
  lanes individually as slots open.
- Base SIB-200 job `1184508_3` completed cleanly and its six language rows are
  recorded in `Pure GatedDeltaNet Results` with exact prompt provenance and
  summary hash. Active base jobs are `1184504_1` (MasakhaNER), `1184505_2`
  (MasakhaPOS), and `1184513_4` (InjongoIntent), all healthy on A100-40GB.
- Base InjongoIntent job `1184513_4` completed cleanly and its four language
  rows are recorded in `Pure GatedDeltaNet Results` with exact five-prompt
  provenance and summary hash. Frozen SA-general lane `1185142_5` was
  submitted into the freed A100-40GB slot; NER `1184504_1` and POS
  `1184505_2` remain active.
- Base MasakhaPOS job `1184505_2` completed cleanly; all four prompts returned
  `0.0000` flexible-extract token accuracy for Xhosa, Zulu, and Tswana. The
  verified negative rows are recorded in `Pure GatedDeltaNet Results` with
  exact provenance. Frozen Belebele Afrikaans lane `1185270_6` was submitted;
  it and SA-general `1185142_5` await A100-40GB capacity while NER continues.
- Base Belebele Afrikaans job `1185270_6` completed cleanly; all five frozen
  prompts tied at accuracy `0.2733333333333333` (summary SHA-256
  `2155482d50d16ec429c6e392044cb3455b1d44a3b38d637c69abe62a22768b87`).
  Row 26 of `Pure GatedDeltaNet Results` records the verified result and full
  provenance. Base progress is `5/16`; NER `1184504_1` and SA-general
  `1185142_5` run on A100-40GB while Belebele English `1185322_7` waits as the
  third owned lane.
- Base Belebele English job `1185322_7` completed cleanly; all five frozen
  prompts tied at accuracy `0.2733333333333333` (summary SHA-256
  `ce1f3dcfd8f7bb062b0a96f4f114b4089febc1435d651dbe9d43f229bd17bbbf`).
  Row 25 of `Pure GatedDeltaNet Results` records the verified result and full
  provenance. Base progress is `6/16`; NER `1184504_1`, SA-general
  `1185142_5`, and newly submitted Sesotho Belebele `1185855_8` are the three
  A100-40GB lanes.
- Base Belebele Sesotho job `1185855_8` completed cleanly; all five frozen
  prompts tied at accuracy `0.2733333333333333` (summary SHA-256
  `0a4892d547184febc8830f7a5304108bb62cd927717dce6cfe7f836cf28d5dcb`).
  Row 24 of `Pure GatedDeltaNet Results` records the verified result and full
  provenance. Base progress is `7/16`; siSwati Belebele `1185870_9` replaced
  it as the third A100-40GB lane.
- Base Belebele siSwati `1185870_9` and Setswana `1185875_10` both completed
  cleanly with all five prompts tied at accuracy `0.2733333333333333`; their
  verified rows are 23 and 22 respectively. Base progress is `9/16` and
  Xitsonga Belebele `1185878_11` is the third A100-40GB lane.
- Base Belebele Xitsonga `1185878_11` completed cleanly with all five prompts
  tied at accuracy `0.2733333333333333` (summary SHA-256
  `ac57800e3c1c67ee471c0217e2500d232a5d110aeb359c4ddc36af4676adb47e`).
  The pure-GDN tab's previously blank reserved row 27 now carries the missing
  `Tso` label and full verified provenance; historical tabs were not changed.
  Base progress is `10/16`, and Xhosa Belebele `1185883_12` is the third
  A100-40GB lane.
- Base Belebele Xhosa `1185883_12` and Zulu `1185884_13` completed cleanly;
  all five prompts tied at accuracy `0.2733333333333333`. Rows 21 and 20 of
  the pure-GDN tab contain their verified provenance. SA-general
  `1185142_5` also completed and rows 28--39 are published from its verified
  summary. Base progress is `13/16` until NER and the two generation lanes
  finish.
- The shared generation evaluator must respect the model artifact's saved
  `config.use_cache`. Forcing cache on broke five-beam generation for pure FLA
  GatedDeltaNet during cache reorder. Provenance jobs `1185900_14` and
  `1185901_15` failed before metrics; the focused fix is tested, and clean
  replacements `1185914_14` and `1185915_15` run on A100-40GB beside healthy
  NER `1184504_1`. This is an implementation compatibility correction, not a
  metric-driven protocol change.
- Base MasakhaNER `1184504_1` completed cleanly with flexible-extract F1
  `0.0000` for every frozen prompt in Xhosa, Zulu, and Setswana. The verified
  result and summary SHA-256
  `d01df9bcd264e398ebfdf5cc57f9af9d6e5e1d7e35e9f4577a2e3698c4cc18d3`
  are published in rows 4--6 of the pure-GDN tab. Base progress is `14/16`;
  only healthy A100-40GB generation jobs `1185914_14` and `1185915_15`
  remain before the Mono/Multi/General preregistration gate opens.
- Uncached five-beam generation is now the base-gate schedule risk. AfriHG
  `1185915_15` needed about 65 minutes for its automatic batch-size-64 probe;
  T2X `1185914_14` was still probing after 80 minutes. Both remain healthy,
  but the 24-hour wall-time may become binding. Do not change the frozen beam
  protocol from throughput evidence alone; first obtain actual post-probe
  progress or terminal evidence.
- Operational deadline: complete all experiments and evaluations by
  31 August 2026. Paper drafting is deferred until afterward, with a first
  advisor draft targeted for mid-September.
- T2X `1185914_14` cleared its expensive batch-size-64 probe after about
  1h52m and entered real generation; AfriHG `1185915_15` remains in real Xhosa
  generation. A probe-rate extrapolation puts AfriHG's combined Xhosa+Zulu
  lane beyond its 24-hour request, but that is not yet a measured task rate.
  Preserve the current run until it yields a task artifact or terminal state;
  then continue only the missing frozen task if needed, without repeating a
  completed held-out result.
- Base T2X Xhosa `1185914_14` completed cleanly in `02:44:37`: chrF
  `3.1401008038579232`, ROUGE-1/ROUGE-L `0.000816014651532441`, ROUGE-2 and
  BLEU `0.0`. Summary SHA-256
  `542a9729e92bd7afe44f68aa85c74d3fab218e1c6f9e662e82c78a98c34667d0`
  and full provenance are published in row 41 of the pure-GDN tab. Base is
  now `15/16`; healthy AfriHG `1185915_15` is the sole active A100-40GB job
  and the only remaining base gate.
- At 19:42 SAST, AfriHG `1185915_15` remained healthy in real Xhosa
  generation after `04:51:31`, with new progress at 18:48 and 18:59 and no
  fault marker or summary. It is the sole owned GPU job on A100-40GB; base
  progress remains `15/16`, fine-tuning remains frozen, and Mono/Multi/General
  work stays behind the base-completion and preregistration gate.
- AfriHG `1185915_15` saved its frozen Xhosa task at 20:56 SAST: chrF
  `4.079834809867061`, ROUGE-1/ROUGE-L `0.00012029488297117726`, ROUGE-2 and
  BLEU `0.0`; task-metrics SHA-256
  `9cbb9f7572794838c9cb2b53104ff814bd688bdf225d1b22ed7897b15001f198`.
  It then entered Zulu generation over `1,776` examples. Preserve Xhosa and
  never repeat it; wait for the combined summary before sheet publication.
  Base remains `15/16` until Zulu and the lane-level summary complete.
- At 22:12 SAST, AfriHG `1185915_15` logged real post-probe Zulu generation
  progress at 22:08 with no fault. Zulu and the combined summary remain the
  only base-gate blocker; Xhosa stays preserved and the sheet stays frozen.

## Pure-GDN adapter validation execution — 2026-08-07

- All 16 frozen base-model lanes and canonical pure-GDN sheet rows are
  complete. Adapter selection remains validation-only; no held-out adapter
  test is allowed until all eight task-family winners are frozen.
- Six family winners are frozen: News LR `3e-5`; SIB job `1189730`, LR
  `3e-5`, checkpoint `526`; NER job `1190665`, LR `3e-5`, checkpoint `541`;
  POS job `1191712`, LR `3e-5`, checkpoint `283`; and Intent job `1192267`,
  LR `8e-5`, checkpoint `2643` with validation macro-F1
  `0.002555133509628254`; T2X job `1192813`, LR `3e-5`, checkpoint `483`.
  AfriHG and General remain open.
- T2X LR-0 job `1192813` completed cleanly with validation chrF `0.0` at all
  three epochs and restored epoch-1 `checkpoint-483`; its final adapter SHA-256
  is `4b1dc2fcd3656809057ada44cb31b7eae66cbd229edac21760e560f0585fff96`.
  LR-1 `1193166` also completed cleanly with chrF `0.0` at all three epochs
  and restored its epoch-1 `checkpoint-483`; final adapter SHA-256 is
  `066a1ac354bedd81e98b2431cc008e852805e07a5f43526355081719e7ea35ba`.
  LR-2 `1193880` completed cleanly with chrF `0.0` at all three epochs and
  restored epoch-1 `checkpoint-483`; final adapter SHA-256 is
  `b6a5e5b132a0f53adc62efc26c482bf7f81e4c4bbbd00ec22e9a07ca4a0bb6fb`.
  Because every LR and epoch tied at chrF `0.0`, T2X is validation-frozen to
  the preregistered lower-LR/earlier-checkpoint winner: LR `3e-5`, job
  `1192813`, epoch-1 `checkpoint-483`.
- Preregistered AfriHG LR-0 job `1194878` (`3e-5`) was submitted after T2X
  LR-0 released a slot. AfriHG LR-1 job `1194979` (`8e-5`) and LR-2 job
  `1195667` (`1.5e-4`) complete the frozen validation grid. All three are
  schedulable on A100-40GB. LR-0 and LR-1 started together at 00:23:23 SAST on
  2026-08-08 and passed canonical pure-GDN, BF16, exact ten-target LoRA,
  Hub-disabled, and validation-only `24,649/3,082` dataset startup gates. LR-2
  remains resource-pending. Both running jobs entered finite training at about
  `3.0--3.1 s/step` and were healthy near steps `2925/15410` and `2848/15410`
  at 02:53 SAST; finite latest loss/gradient records had no fault signature.
  Both are minutes from epoch-1 validation; earliest patience-stop ETA remains
  roughly six raw training hours plus
  validation callback time. AfriHG and General remain open, and held-out
  adapter evaluation stays blocked behind the global family-winner freeze.
- AfriHG LR-0/LR-1 reached epoch-1 step `3082` and completed finite trainer
  validation (`eval_loss=0.7932677865028381/0.7838549017906189`), then entered
  the full task-native Xhosa/Zulu generation callbacks. At the 03:26 SAST
  pass both callbacks were still in automatic batch-size-64 probing, with no
  validation chrF or checkpoint artifact yet. Trainer loss is not the frozen
  selection metric, so no provisional winner is declared. LR-2 `1195667`
  remains resource-pending; all three schedulable jobs are A100-40GB only.
  AfriHG and General remain open, and every held-out adapter test and sheet
  update remains blocked.
- At 03:54 SAST the AfriHG epoch-1 task-native callbacks still had no chrF or
  checkpoint artifact. LR-0 batch-level events were roughly 19--21 minutes
  apart after the batch-size-64 probe; LR-1 showed the same order of cadence.
  A provisional `3,082/64` extrapolation places epoch-1 callback completion
  around `19:00--20:00 SAST`, but no explicit callback counter exists. The
  24-hour wall-time is therefore a material risk before patience-governed
  completion. Preserve `1194878/1194979`; do not alter the frozen recipe from
  throughput alone. If an actual wall-time termination occurs after a valid
  checkpoint, resume identically as infrastructure recovery. LR-2 `1195667`
  remains resource-pending. Held-out adapter evaluation and Sheet writes stay
  blocked.
- At 04:24 SAST the AfriHG callbacks remained healthy but their observed event
  intervals had increased to about 27 minutes on both `1194878/1194979`.
  Neither run had a chrF/checkpoint artifact. The superseding wide epoch-1
  callback estimate is now roughly `21:00 SAST` through the `00:23 SAST`
  wall-time boundary, so terminal completion within the original allocation
  is unlikely. This remains a log-event extrapolation, not a progress counter;
  preserve both jobs unchanged until artifact or terminal evidence, then use
  identical checkpoint resume only as infrastructure recovery. LR-2
  `1195667` remains resource-pending; AfriHG, General, held-out evaluation,
  and Sheet writes remain blocked.
- AfriHG LR-2 `1195667` started at `04:49:05 SAST` and passed the canonical
  pure-GDN/BF16/architecture-complete LoRA/validation-only/Hub-disabled
  startup gates, reaching step `60/15410` near `3.09 s/step` without a fault
  marker. The full frozen AfriHG grid `1194878/1194979/1195667` now occupies
  exactly three A100-40GB slots. LR-0/LR-1 still have no epoch-1 chrF or
  checkpoint artifact, so AfriHG remains open; no General trial, held-out
  adapter evaluation, or Sheet write is permitted yet.
- At 05:24 SAST LR-2 `1195667` was healthy at step `646/15410` with finite
  loss/gradient `1.1426/1.4747387170791626`, placing its raw epoch-1 training
  ETA near `07:25 SAST` before task-native callback overhead. LR-0/LR-1 still
  had no chrF/checkpoint artifact and no new callback log event for 37--41
  minutes; Slurm remained running and logs had no fault marker, so this is
  wall-time risk rather than proven hang. All three A100-40GB slots remain
  occupied; AfriHG, General, held-out evaluation, and Sheet writes are still
  blocked.
- AfriHG LR-0/LR-1 produced valid epoch-1 `checkpoint-3082` artifacts at
  `05:41:44/05:44:51 SAST`; both full-validation mean chrF values are `0.0`
  and each checkpoint is its trial's current best. The 64 saved debug examples
  are only the debug collector cap: the metric evaluator is explicitly built
  with `max_samples_per_lang=None`, so all `3,082` validation examples were
  scored. LR-0/LR-1 resumed epoch 2. Measured callback time is about `2h35m`,
  giving conditional tied-metric terminal ETAs near `16:10 SAST` for LR-0/1
  and `20:30 SAST` for healthy LR-2 `1195667`. Do not freeze AfriHG until all
  three terminal states; General, held-out evaluation, and Sheet writes remain
  blocked.
- LR-2 `1195667` completed epoch-1 trainer validation at 07:31 SAST with
  finite loss `0.7802404761314392` and entered its full task-native callback;
  no chrF/checkpoint exists yet. LR-0/LR-1 remain healthy near the end of
  epoch-2 training. Conditional first-checkpoint ETAs remain near
  `10:05/11:00 SAST`; AfriHG, General, held-out evaluation, and Sheet writes
  remain blocked.
- At 08:55 SAST, AfriHG LR-0/LR-1 `1194878/1194979` completed finite epoch-2
  trainer validation with health-only losses
  `0.7849799990653992/0.7786943912506104` and entered their full task-native
  callbacks; LR-2 `1195667` continued its epoch-1 callback with fresh progress.
  No new chrF/checkpoint or fault marker exists. Exactly three A100-40GB jobs
  remain active with no A100-80GB/L40S overlap, so no General slot is open.
  Base remains `16/16`, family winners `6/8`; held-out adapter evaluation,
  Sheet writes, and Hugging Face publication remain blocked.
- At 09:24 SAST, AfriHG jobs `1194878/1194979/1195667` remained healthy on
  exactly three A100-40GB GPUs with fresh task-native callback progress at
  `09:12:09/09:14:46/09:08:05`. No new chrF/checkpoint exists, no General
  slot is free, and the only fault-pattern grep match was the benign trainer
  setting `logging_nan_inf_filter=True`. Base remains `16/16`, family winners
  `6/8`; held-out adapter evaluation, Sheet writes, and publication remain
  blocked.
- At 09:54 SAST, AfriHG LR-0/LR-1 `1194878/1194979` continued epoch-2
  task-native callbacks with fresh progress; LR-2 `1195667` remained running
  in epoch-1 callback despite a 46-minute log interval and had no runtime
  fault marker. No new chrF/checkpoint or free General slot exists. Base
  remains `16/16`, family winners `6/8`; held-out adapter evaluation, Sheet
  writes, and publication remain blocked.
- AfriHG LR-2 `1195667` produced its valid epoch-1 `checkpoint-3082` at
  10:06 SAST with full-validation mean chrF `0.0`, then resumed epoch 2.
  Verified SHA-256 values for adapter/config/trainer state are
  `6c8a2711ece2cfddb0d87140093c5b1a905c3214efd86d21f5d1f91eca135a33`,
  `6f9169df6bdd3f6b1ef5cdbe2043f823c12bb60fbe0096e60bddabbdb79a03c7`,
  and `1b9dd239137717320d2ba07a8a3f643edde79137b4a394c7018b1a2f214a5c8a`.
  LR-0/LR-1 remain in epoch-2 callbacks. All three A100-40GB jobs remain
  active, so AfriHG is not yet frozen and General, held-out evaluation, Sheet
  writes, and publication remain blocked.
- At 10:54 SAST, LR-0/LR-1 `1194878/1194979` remained healthy in epoch-2
  callbacks with no `checkpoint-6164` yet; LR-2 `1195667` advanced normally
  to step `4005/15410` in epoch 2. Exactly three A100-40GB jobs remain active,
  so AfriHG is not terminal and General, held-out evaluation, Sheet writes,
  and publication remain blocked.
- AfriHG LR-0/LR-1 `1194878/1194979` produced valid epoch-2
  `checkpoint-6164` artifacts at 11:03/11:05 SAST. Both full-validation chrF
  values again tied `0.0`, so epoch-1 `checkpoint-3082` remains each trial's
  best. By 11:46 both had resumed epoch 3; LR-2 `1195667` continued epoch 2.
  All three A100-40GB jobs remain active, so AfriHG is not terminal and
  General, held-out evaluation, Sheet writes, and publication remain blocked.
- At 12:24 SAST, AfriHG LR-0/LR-1 `1194878/1194979` were healthy at epoch-3
  steps `7712/7676`; LR-2 `1195667` was healthy at epoch-2 step `5752`, about
  22 raw training minutes from its next validation. All three A100-40GB jobs
  remain active, so General, held-out evaluation, Sheet writes, and
  publication remain blocked.
- At 13:00 SAST, LR-2 `1195667` completed epoch-2 trainer validation with
  health-only loss `0.7753579020500183` and entered the full task-native
  callback; no chrF/checkpoint exists yet. LR-0/LR-1 `1194878/1194979` were
  healthy at epoch-3 steps `8406/8367`, about 45 raw minutes from validation.
  All three A100-40GB jobs remain active, so General, held-out evaluation,
  Sheet writes, and publication remain blocked.
- At 13:30 SAST, LR-0/LR-1 `1194878/1194979` were healthy at epoch-3 steps
  `8978/8940`, about 14--16 raw minutes from final validation. LR-2 `1195667`
  remained healthy in its epoch-2 task-native callback with no new chrF or
  checkpoint. All three A100-40GB jobs remain active, so General, held-out
  evaluation, Sheet writes, and publication remain blocked.
- At 14:19 SAST, LR-0/LR-1 `1194878/1194979` completed epoch-3 trainer
  validation with health-only losses `0.7823924422264099/0.7764996290206909`
  and entered their terminal task-native callbacks. LR-2 `1195667` continued
  its epoch-2 callback. No new chrF/checkpoint or terminal state exists yet;
  all three A100-40GB jobs remain active, so General, held-out evaluation,
  Sheet writes, and publication remain blocked.
- At 14:32 SAST, LR-0/LR-1 `1194878/1194979` remained healthy in their
  terminal task-native callbacks with fresh progress at `14:12/14:14`, and
  LR-2 `1195667` remained healthy in its epoch-2 callback with fresh progress
  at `14:27`. No new chrF/checkpoint or terminal state exists; all three
  A100-40GB jobs remain active, so General, held-out evaluation, Sheet writes,
  and publication remain blocked.
- At 15:01 SAST, LR-0/LR-1 `1194878/1194979` continued their terminal
  task-native callbacks with fresh progress at `14:33/14:35`; LR-2 `1195667`
  remained healthy in its epoch-2 callback. No new chrF/checkpoint or terminal
  state exists, so all three A100-40GB slots remain occupied and General,
  held-out evaluation, Sheet writes, and publication remain blocked.
- AfriHG LR-2 `1195667` produced a valid epoch-2 `checkpoint-6164` at 15:25
  SAST with full-validation mean chrF again `0.0`; epoch-1
  `checkpoint-3082` remains its best. Verified checkpoint-6164 SHA-256 values
  for adapter/config/trainer state are
  `ee89e2af74f8d3076bdfb7dc85d9c15aece1198cda8b8a84510c12baab2e98d6`,
  `6f9169df6bdd3f6b1ef5cdbe2043f823c12bb60fbe0096e60bddabbdb79a03c7`,
  and `6d4f7fba6fabf0130d54d4b0fafe1cff21ed86547ec8db1bf10a6ce1eac01c47`.
  At 16:02 SAST it had resumed epoch 3, while LR-0/LR-1 `1194878/1194979`
  remained healthy in terminal callbacks. All three A100-40GB slots remain
  occupied, so General, held-out evaluation, Sheet writes, and publication
  remain blocked.
- At 16:07 SAST, LR-0/LR-1 `1194878/1194979` remained `RUNNING` with active
  batch CPU accounting but no log progress for about 40 minutes and no new
  checkpoint, final adapter, terminal state, or fault marker. Preserve the
  frozen runs; this is not yet evidence of failure. LR-2 `1195667` continued
  epoch-3 training normally. All three A100-40GB slots remain occupied, so
  General, held-out evaluation, Sheet writes, and publication remain blocked.
- AfriHG LR-0 `1194878` and LR-1 `1194979` completed cleanly `0:0` after
  epoch 3, retaining epoch-1 `checkpoint-3082` with validation chrF `0.0`.
  Final adapter SHA-256 values are
  `dc6e3c1aee872b86bc005d0b375efc55eb03172d095a246789bd99ac8a90db09`
  and `5dbb6e3e68443d02215bf55b4aad793aa7683eca6e28b1e8701d14df3ba181a2`.
  AfriHG remains open until LR-2 `1195667` terminates.
- The two released A100-40GB slots were offered individually to the
  preregistered General screen after verifying matching runner/config hashes
  and no prior artifact or duplicate. General LR-0 `1204261` (`3e-5`) started
  on `srvrocgpu010` and passed pure-GDN/BF16/ten-target/fast-kernel,
  validation-loss-only, `43,637/12,209` token-balanced data, Hub-disable, and
  no-task-metric startup gates. It entered its `13,640`-step training loop at
  16:40 SAST; initial throughput is too early for a defensible ETA. General
  LR-1 `1204262` (`8e-5`) is resource-pending with conservative scheduler start
  `2026-08-09 04:49:05 SAST`. Held-out evaluation, Sheet writes, and
  publication remain blocked until AfriHG and General winners are frozen.
- At 17:06 SAST, AfriHG LR-2 `1195667` was healthy at epoch-3 step
  `8119/15410`, preserving its conditional terminal ETA near `20:40 SAST`.
  General LR-0 `1204261` was healthy at step `248/13640` near `6.16 s/step`,
  with a raw epoch-1 boundary near `21:20 SAST` before validation; terminal ETA
  remains unknown until the first validation runtime. General LR-1 `1204262`
  remained resource-pending. Only two A100-40GB jobs were active; held-out
  evaluation, Sheet writes, and publication remain blocked.
- At 17:36 SAST, AfriHG LR-2 `1195667` was healthy at epoch-3 step
  `8699/15410`, about 28 raw minutes from terminal validation; General LR-0
  `1204261` was healthy at step `534/13640`, with raw epoch-1 boundary near
  `21:24 SAST`. General LR-1 `1204262` remained resource-pending. Only two
  A100-40GB jobs were active; held-out evaluation, Sheet writes, and
  publication remain blocked.
- At 18:07 SAST, AfriHG LR-2 `1195667` had reached epoch-3 step `9246` and
  entered trainer validation, preserving terminal ETA near `20:40--20:45`
  SAST. General LR-0 `1204261` remained healthy near step `819/13640`.
  General LR-1 `1204262` started on A100-40GB, passed the frozen
  pure-GDN/BF16/ten-target/validation-loss/token-balanced/Hub-disabled gates,
  and entered training at 18:06 SAST. Exactly three A100-40GB jobs are active;
  General LR-2, held-out evaluation, Sheet writes, and publication remain
  blocked.
- AfriHG LR-2 `1195667` completed cleanly `0:0` at 20:43 SAST with validation
  chrF `0.0`, retaining epoch-1 `checkpoint-3082`; its final adapter SHA-256
  is `dc4cbbf124121aafb44ddddca775d5f16bbb878ea4fd700f9f4e15be3088689c`.
  All AfriHG LRs tied at `0.0`, so the preregistered tie-break freezes AfriHG
  to LR `3e-5`, job `1194878`, checkpoint `3082`. Frozen family winners are
  now `7/8`.
- General LR-0/LR-1 jobs `1204261/1204262` produced valid epoch-1
  `checkpoint-2728` validation-loss artifacts of
  `13.603466245839952/13.732546259124712` and remained healthy in epoch 2.
  After verifying the free A100-40GB slot and no artifact or duplicate,
  General LR-2 was submitted individually as job `1207524`; it started at
  `23:29:48 SAST` and passed the frozen pure-GDN/BF16/ten-target,
  validation-loss-only, token-balanced `43,637/12,209`, task-metric-disabled,
  and Hub-disabled startup gates, then entered finite training. Exactly three
  A100-40GB jobs are active with no A100-80GB/L40S overlap. General remains
  the sole open family; held-out adapter evaluation, Sheet writes, and
  publication remain blocked until all three General trials are terminal and
  the winner is frozen.
- General LR-0 job `1204261` produced its epoch-2 `checkpoint-5456` with
  validation loss `13.807909396570615`, worse than epoch-1
  `13.603466245839952`; validation-only selection therefore retains
  `checkpoint-2728` as the current best. LR-0 resumed epoch 3 under the frozen
  patience-2 rule. LR-1/LR-2 remain active and General remains the sole open
  family; held-out adapter evaluation, Sheet writes, and publication remain
  blocked until all three trials are terminal and the winner is frozen.
- General LR-1 job `1204262` produced its epoch-2 `checkpoint-5456` with
  validation loss `13.76491213184439`, worse than epoch-1
  `13.732546259124712`; validation-only selection therefore retains
  `checkpoint-2728` as its current best. LR-1 resumed epoch 3 under the frozen
  patience-2 rule. LR-2 remains active without a validation artifact, so
  General and all held-out adapter evaluation remain blocked.
- General LR-2 job `1207524` produced its epoch-1 `checkpoint-2728` with
  validation loss `13.611691059151418` and resumed epoch 2. Interim
  validation-loss bests are LR-0 `13.603466245839952`, LR-2
  `13.611691059151418`, and LR-1 `13.732546259124712`; this ordering is not a
  frozen winner until all three trials terminate. General and all held-out
  adapter evaluation remain blocked.
- General LR-0 `1204261` completed cleanly `0:0` after epoch 3 under the
  frozen patience-2 rule. Epoch-3 validation loss `13.80606733455681` did not
  improve epoch-1 best `13.603466245839952`, so the run restored retained
  `checkpoint-2728`; all `424/424` final-adapter tensors exactly match that
  checkpoint. Final adapter SHA-256 is
  `50ebbeddd2a45206f3d73c1eda62d188cf2d269848027e7227cffb4309381b8c`.
  General remains unfrozen until LR-1 `1204262` and LR-2 `1207524` terminate;
  held-out adapter evaluation and Sheet writes remain blocked.
- General LR-1 `1204262` completed cleanly `0:0` after epoch 3 under the
  frozen patience-2 rule. Epoch-3 validation loss `13.776524366234797` did
  not improve epoch-1 best `13.732546259124712`, so the run restored retained
  `checkpoint-2728`; all `424/424` final-adapter tensors exactly match that
  checkpoint. Final adapter SHA-256 is
  `c15d6079785714ea40881dbeba5f3be7e318b789cf7297f180ae6af8de328fb1`.
  General remains unfrozen only until LR-2 `1207524` terminates; no
  Monolingual training, held-out adapter evaluation, or Sheet write may begin
  before the validation-only General winner is frozen.
- General LR-2 `1207524` improved at epoch 2 to validation loss
  `13.511420388596335` at retained `checkpoint-5456`, provisionally beating
  completed LR-0/LR-1 bests `13.603466245839952/13.732546259124712`. It
  resumed epoch 3 under the frozen patience rule. Do not freeze General or
  start Monolingual/held-out work until LR-2 terminates and its restored final
  adapter is verified.
- At 18:59 SAST on 9 August, scientifically trusted corrected-base progress
  is `2/16`: News `1211671_0` and SIB `1211675_3` are verified and written to
  the canonical Sheet. Corrected NER `1211758_1` and POS `1211759_2` are
  healthy at `1057/14980` and `1025/7216` on A100-40GB; superseded AfriHG
  `1210866` is provenance-only. HPO remains `0/8` and blocked until all 16
  corrected raw-prompt, exactly-one-BOS base lanes pass. No held-out adapter
  test has run and Sheet E/F/G remain blank.
- At 19:27 SAST, corrected NER `1211758_1` and POS `1211759_2` remain healthy
  at `2193/14980` and `2137/7216`, with projected completion near `00:48` and
  `21:34` SAST. Superseded AfriHG `1210866` is active in Zulu but remains
  provenance-only. All owned jobs are A100-40GB; quota is home `33.5%`,
  scratch `36.4%`. Trusted progress is unchanged at `2/16`; HPO and held-out
  adapter evaluation remain blocked.
- At 19:57 SAST, corrected NER `1211758_1` and POS `1211759_2` advanced to
  `3377/14980` and `3329/7216`, with projected completion near `00:50` and
  `21:35` SAST. Provenance-only AfriHG `1210866` remains active in Zulu.
  Exactly three A100-40GB slots are occupied, so lane 4 cannot yet be
  submitted; no other GPU family is in use. Trusted progress remains `2/16`,
  HPO `0/8`, held-out adapter tests `0`.
- At 22:02 SAST, corrected POS `1211759_2` is verified `0:0` with summary SHA
  `c7e512d1c873435fa3c230c47775f94df3a915732605c65ddacca0ee934c4daf`;
  all Xho/Zul/Tsn P1--P4 token accuracies are `0.0`, a valid negative base
  result. Sheet rows 7--9 were updated and verified. Trusted base is `3/16`.
  Infrastructure-only jobs `1213980/1213981` failed before model/data because
  `SALLM_RUNTIME_REPO` was omitted; no summaries or metrics exist. Recovery
  Intent `1213986_4` and SA-general `1213987_5` are healthy under the immutable
  snapshot and correct mutable runtime. NER `1211758_1` remains healthy.
  Exactly three A100-40GB jobs are active; HPO and held-out evaluation remain
  blocked, and Sheet E/F/G remain blank.
- At 23:00 SAST, corrected Intent `1213986_4` is verified `0:0` with summary
  SHA `168cec2f9d755378edf08f62a31ab01f3d37d824c4a689cac76efce19f4b6b28`;
  all P1--P5 prompts tie at F1 `0.001290205525708353` for English and
  `0.0012195121951219512` for Xho/Zul/Sot. Sheet rows 16--19 were updated and
  verified. Trusted base is `4/16`. Corrected Belebele-Afrikaans lane 6 is
  queued as `1214451_6`; NER `1211758_1` and SA-general `1213987_5` remain
  healthy on A100-40GB. HPO and held-out adapter evaluation remain blocked;
  Sheet E/F/G remain blank.
- At 00:30 SAST on 10 August, corrected SA-general `1213987_5` and
  Belebele-Afrikaans `1214451_6` are verified. Summary SHAs are
  `733a494cf4d30fd9bc923c46bb47f1f1e009529a183e14598b08cf89d45e8d95`
  and `13efdc7ebeb11c555884b24a47dd7e2d2115637ec487e04a1999d289386b3520`.
  Corrected AfriMGSM is `0.0` across all languages/prompts; this is a valid
  negative result. Sheet row 26 and rows 28--39 were updated and verified.
  Trusted base is `6/16`. Belebele-English `1215825_7` is running and
  Belebele-Southern-Sotho `1215826_8` is resource-pending; NER `1211758_1`
  remains healthy near completion. HPO and held-out evaluation remain blocked,
  with Sheet E/F/G blank.
- At 01:00 SAST, corrected NER `1211758_1`, Belebele-English `1215825_7`, and
  Belebele-Sotho `1215826_8` are verified. Summary SHAs are
  `784a155538e47bb7df1752f6097e8c1d60c0d0fa3bef17aa57665183765614df`,
  `ebd0ddd24c2c22a123dbabe7715c8111ea8cafa4925e2a10540c1039f0ceb5ae`,
  and `a4d9ebe34463d6e97b9b09af75196aae7e75e4c2ac8349d5c4dd792bdde697ac`.
  Sheet rows 4--6 and 24--25 were updated and verified. Trusted base is
  `9/16`. Belebele-Swati `1215945_9` is running; Tswana `1215946_10` and
  Tsonga `1215947_11` are queued. HPO and held-out adapter evaluation remain
  blocked, with Sheet E/F/G blank.

- At 10:13 SAST on 16 August, user-authorized cancellation stopped slow POS
  Stage-B b0 `1238876` after `06:49:23`; step-283/566 validation artifacts
  and checkpoint 566 are preserved, but the interrupted run is
  provenance-only. Fused cache-gate/b1 `1238979` immediately used the freed
  `srvrocgpu010` A100-40GB and failed closed before training: all frozen
  12-cell validation-only predictions, counts, metrics, aggregate accuracy,
  and scores matched exactly, but runtime improved only `1.413645836445053x`
  (`48.3900309689343 s` to `34.23066069406923 s`) versus the preregistered
  3x gate. No held-out data was touched. There are zero owned jobs; trusted
  POS remains `3/11`, winners `0/8`, held-out `0`, Sheet E/F/G blank, and
  publication blocked.

- At 10:21 SAST on 16 August, POS cached last-logit v2 gate `1238982`
  also failed closed before training. It used immutable 698-file snapshot
  `pure-gdn-hpo-poscache-lastlogit-20260816-61ce254f` and matched the original
  scorer exactly on all frozen 12-cell validation-only outputs and scores,
  but achieved only `1.0367610778285992x` (`35.0182068439899 s` to
  `33.77654465707019 s`) versus the locked 3x gate. No held-out or training
  data was loaded. Zero owned jobs remain; cached scoring is not ratified,
  trusted POS stays `3/11`, winners `0/8`, and held-out `0`.

- At 10:28 SAST on 16 August, scientifically frozen full-prefix POS Stage-B
  recovery resumed. Clean snapshot
  `pure-gdn-hpo-pos-fullprefix-recovery-20260816-38aba8e3` verifies `694/694`
  files, contains no cache path, and has independent deployment-manifest SHA
  `819a4663b9a90ba65170c73c9dd10b30a16ec14d743c160c55b9c6b89a6df1dc`.
  B0 recovery `1238989` writes to a new isolated path; b1 `1238990` and b2
  `1238991` use their untouched canonical paths. B0/b1 started concurrently
  on two `srvrocgpu010` A100-40GB GPUs, verified manifests
  `916850c0...68f7`/`50960548...1d52`, passed fast-GDN, loaded exact
  `2,259/1,800` train/validation rows, and began training. B2 is
  Resources-pending. Owned state is two running plus one pending
  `gpu:ampere` jobs, with no other GPU family. Trusted POS remains `3/11`,
  winners `0/8`, held-out `0`, Sheet E/F/G blank, and publication blocked.
  The historical `possourcefix` deployment manifest was overwritten through
  a shared hard-link during rejected v2 manifest regeneration; its source is
  nevertheless independently proven intact by cancelled b0 execution
  manifest `06b825ae...68f7` verifying all original `694/694` hashes. The
  shared-manifest snapshots are excluded from future execution.
- At 20:19 SAST on 16 August, deadline acceleration preserved and hashed the
  complete checkpoint-849 adapter/optimizer/scheduler/RNG/trainer states from
  b0/b1 before cancelling jobs `1238989/1238990`; incomplete checkpoint-1132
  callbacks are excluded. Two prospective cross-row batch gates failed closed
  before training (`1239042` real-tokenizer multi-token assertion; `1241563`
  unsafe `38,942/40,960 MiB` use and no required speed artifact), so batching
  is not authorized. Exact same-trial resume support now passes remote Hydra
  composition from immutable 695-file snapshot
  `pure-gdn-hpo-pos-resume-v2-20260816-45fec06b` (manifest
  `bdfd7f2f...34e2`). Serial b2 `1241578` is healthy; exact b0/b1 step-849
  resumes are queued as `1241580/1241581`. Owned state is one running plus two
  pending A100-40GB jobs, no other GPU family; quota is home `88.6%`, scratch
  `39.0%`. Trusted POS remains `3/11`, winners `0/8`, held-out `0`, and Sheet
  E/F/G blank.
- At 15:54 SAST on 20 August, AfriHG Stage-B b0 job `1248575` completed its
  step-6164 exact validation callback. Xho/Zul chrF is
  `22.863993326854846/24.03782125622953`, mean `23.450907291542187`, making
  checkpoint 6164 the retained within-run best over step 3082. Artifact and
  trainer-state SHA-256 values are `6cc6ea0b...834ba7e` and
  `698b45e8...571c84`; coverage, uniqueness, prompt-boundary, and empty-output
  checks pass. The run is still non-terminal, so AfriHG remains `3/11`
  terminal-valid, global winners `0/8`, and held-out adapter access `0`.
- At 19:24 SAST on 20 August, AfriHG Stage-B b0 job `1248575` completed its
  step-9246 exact validation callback and resumed healthy training. Xho/Zul
  chrF is `23.769019260468166/24.8046626683949`, mean
  `24.286840964431534`; checkpoint 9246 is the retained within-run best.
  Artifact/trainer-state SHA-256 values are `47b9b7c7...f6ef8` and
  `635bea21...dfb91`; coverage, uniqueness, prompt-boundary, and empty-output
  checks pass. This remains non-terminal validation-only evidence: AfriHG is
  `3/11` terminal-valid, global winners `0/8`, held-out access `0`, and jobs
  `1248576/1248577` remain queued.

## 2026-08-31: non-General results no longer wait for General

- Before any current adapter official-test access, the user prospectively
  changed the order so each non-General family may test once its own corrected
  Multi winner and all applicable validation-frozen Mono adapters verify. The
  family-wise amendment SHA-256 is
  `dcfaa036979b6ef02f6d2cefde9b5118a282528d669fc9ac9ab451a7f33f641b`.
  General cells remain blank and early test scores may not affect unfinished
  HPO, Mono selection, retries, scheduling, or evaluator changes.
- Active General b1/b2/b4 continuations keep running; pending zero-second b5
  preflight `1279656` was cancelled and no further General work has priority.
- The next gate is metric-free exact-checkpoint CUDA verification. Post-
  checkpoint mode now includes BF16 forward/backward on the loaded checkpoint;
  the local verifier suite passes `10/10` and Ruff is clean. Then T2X can run
  its single official arm, while NER/POS/AfriHG complete eight new Mono
  adapters and News/SIB/Intent complete their already frozen reduced HPO plus
  twelve Mono adapters. Each complete family releases independently.
- Exact checkpoint CUDA gate `1279742` is queued once on A100-80GB with an
  immutable `1,424`-file manifest. It is metric-free and does not load a task
  or held-out split. After it passes, T2X is the first official-test release.
- Independent review caught gate binding/runtime/numerical gaps before
  `1279742` ran; it was cancelled at elapsed zero. Corrected replacement
  `1279752` is pending once with a fresh `713`-file immutable manifest, exact
  checkpoint-path binding, immutable Python/package runtime verification, and
  finite complete-gradient checks. Local verification is `13` tests passed and
  Ruff clean. Held-out access remains zero.
- Follow-up review caught missing manifest-root binding before `1279752` ran;
  it too was cancelled at elapsed zero. Final v4 gate `1279771` is pending with
  source, checkpoint, immutable runtime, snapshot-root, and artifact-root
  verification. Tests remain `13/13` and held-out access remains zero.
- At 22:54 SAST on 20 August, AfriHG Stage-B b0 job `1248575` completed its
  step-12328 exact validation callback and resumed healthy training. Xho/Zul
  chrF is `23.470367754445988/25.368293786591423`, mean
  `24.419330770518705`; checkpoint 12328 is the retained within-run best.
  Artifact/trainer-state SHA-256 values are `22a35faa...fdd50` and
  `728f1f52...add37`; coverage, uniqueness, prompt-boundary, and empty-output
  checks pass. This remains non-terminal validation-only evidence: AfriHG is
  `3/11` terminal-valid, global winners `0/8`, held-out access `0`, and jobs
  `1248576/1248577` remain queued.
- On 1 September, metric-free exact-checkpoint CUDA gate `1279771` completed
  `0:0`; result SHA-256 `d1f13d836e18b77f2af86db70466b5c19818a52a0e4daad34d2b645b81f1248e`
  verifies the immutable runtime/base, save-reload integrity, finite complete
  BF16 gradients, and deterministic generation. Frozen T2X b7 seed-42 adapter
  hashes transferred byte-identically to HEX. One-time official T2X job
  `1280426` then failed after opening and preparing all 378 test rows but before
  generation because the offline runtime lacked the Hugging Face `evaluate`
  BLEU module. It produced no predictions, metric, or summary. The family-wise
  fail-closed rule makes this arm terminally missing and forbids a retry.
- General b1/b2 continuations `1279470/1279471` completed `0:0` and await
  artifact verification; b4 `1279652` remains running. Released A100-80GB
  cards now run the frozen reduced non-General canaries: SIB a0 `1280437`,
  Intent a0 `1280444`, and corrected News a0 `1280455`. Original News a0
  `1280436` failed before model/data loading because `PURE_GDN_MODEL` was not
  exported; its metric-free correction is frozen at SHA-256
  `ced4b6a52f92871aa2689bd4699a02d693d449b5b6fe30a4326b12b0b22117e7`.
  Sheet E/F/G remain blank.
- General b4 continuation `1279652` completed `0:0` in `05:28:46`. The released
  card was filled in frozen candidate-major order by News a1 seed-42 job
  `1280921` after absent-output, no-duplicate, immutable-registry, and active-cap
  checks. News a0/a1, SIB a0, and Intent a0 now use all four A100-80GB cards;
  no held-out value or Sheet cell changed.
- SIB a0 seed-42 job `1280437` completed `0:0` in `02:08:43`; its retained
  checkpoint `526` has validation-only macro F1 `0.05760368663594471` with all
  `2,970` declared rows represented in the terminal log. It is not yet
  terminal-valid because sidecar and exact retained-to-final verification are
  pending. The released card now runs SIB a1 seed-42 job `1281116`, submitted
  after the frozen absence, duplicate, registry, and cap checks.
- Intent a0 seed-42 job `1280444` completed `0:0` in `03:31:07`; its retained
  checkpoint `881` has validation-only macro F1 `0.001135873084394561` with
  all `4,155` declared rows represented in the terminal log. Sidecar and exact
  retained-to-final verification remain pending. Intent a1 seed-42 job
  `1281334` now occupies the released card after the frozen preflight checks.
- Corrected News a0 `1280455` and News a1 `1280921` completed `0:0` with exact
  `3,095`-row terminal logs. They retained steps `1629` and `543` at
  validation-only macro F1 `0.47238216653156473` and
  `0.2079369214708845`; sidecar and exact retained-to-final verification remain
  pending. News a2 `1281511` and SIB a2 `1281512` immediately filled the two
  released cards after all frozen preflight checks passed.
- SIB a1 `1281116` completed `0:0` in `02:07:19`, retaining checkpoint `526`
  at validation-only macro F1 `0.05760368663594471` with exact `2,970`-row
  terminal coverage; sidecar and exact adapter roundtrip remain pending. Intent
  a2 `1281721` now runs in the released slot. All nine reduced News/SIB/Intent
  seed-42 candidates have therefore been launched exactly once.
- News a2 `1281511` and SIB a2 `1281512` completed `0:0`. CPU verifiers
  `1281966`-`1281972` proved exact retained-to-final equality across `424` keys
  and `71,762,560` values for every completed reduced candidate: News a0/a1/a2,
  SIB a0/a1/a2, and Intent a0. Intent a1/a2 remain running.
- NER Mono Tsn `1281964` and Xho `1281965` failed after `00:00:43` before any
  dataset row or validation metric because the current `datasets` runtime
  computed cache IDs not present under the exact extant offline cache. No
  checkpoint or final adapter exists. The prospective alias-only correction
  preserves the pinned data and recipe, requires an exact CPU row/hash probe,
  and uses isolated replacement roots; the original jobs and roots remain
  quarantined.
- Intent a1 `1281334` and verifier `1282232` completed `0:0`; exact
  retained-to-final equality covered `424` keys and `71,762,560` values.
  Corrected NER Zulu Mono `1282231` then started after the exact alias/data,
  snapshot, base, recipe, absence, duplicate, and active-cap checks passed.
- Intent a2 `1281721` completed `0:0` in `04:43:53`, covering all `4,155`
  validation rows and retaining checkpoint `1762`; exact CPU roundtrip verifier
  `1282671` is submitted once. All nine reduced News/SIB/Intent seed-42 runs
  are complete, with only that roundtrip pending. The released card now runs
  POS Tswana Mono `1282672` at the frozen POS LR `1.5e-4`; its immutable
  execution-manifest SHA-256 is
  `eaf017e309517cbe499ab1646761e4f85974ae4d9c6f03bd05793dbf2ea90ae0`.
  NER Tsn/Xho/Zulu and POS Tswana occupy all four A100-80GB cards; held-out
  adapter scores remain absent and Sheet E/F/G remain blank.
- The one-time NER family wrapper is ready prospectively. Independent review
  caught and blocked its initial generation-verifier mismatch before held-out
  access. Corrected task-pack verifier SHA-256
  `23523234228703e48339fde015100b7e076cc33d00ee0c3837eeaaf23c9a27dd`
  enforces exact `4,980/5,000/5,000` Tswana/Xhosa/Zulu coverage; focused tests
  pass `5/5`, and it structurally verifies all three finalized Base artifacts.
  Immutable manifest/preflight jobs `1283922/1283923` completed `0:0`. No NER
  held-out row has been loaded; the six-arm bundle is next when a card releases
  and Sheet E/F/G remain blank.
- POS Zulu Mono `1283304` completed `0:0` in `07:20:50`, covered all `750`
  declared validation rows at its terminal boundary, and wrote a final adapter;
  metric-free sidecar and exact roundtrip verification remain. Corrected NER
  official bundle `1285944` was then submitted exactly once and is
  `AssocGrpGRES`-pending behind another user's fourth association card. POS
  Tswana `1282788`, POS Xhosa `1282789`, and AfriHG Xhosa `1283372` run on the
  other three A100-80GB cards. No NER held-out row has loaded and Sheet E/F/G
  remain blank.
- POS Xhosa, POS Zulu, and AfriHG Xhosa are now terminal-valid: CPU jobs
  `1285992`-`1285994` proved exact retained-to-final equality across `424`
  keys and `71,762,560` values for each adapter. AfriHG Zulu `1285972` filled
  the released card, verified its frozen recipe and exact `14,209/1,777`
  train/validation rows, and entered the fixed `8,885`-step loop.
- Official NER job `1285944` failed in one second during runtime-manifest
  verification because only the Linux kernel patch string differed; its
  result root remained absent and no held-out row loaded. The public-seam
  correction normalizes Linux kernel release text while keeping OS,
  architecture, libc, Python, executable, and packages fail-closed. Fresh
  immutable manifest job `1285991` and metric-free preflight `1285995`
  completed `0:0` and reproduced every frozen config. Single terminal
  replacement `1285997` is running on A100-80GB. Sheet E/F/G remain blank.
- Official NER replacement `1285997` passed every manifest and loaded the
  model plus frozen adapter, then failed `1:0` after `00:01:38` when the
  offline runtime could not resolve the first held-out dataset. Its result
  root contains only empty directories: zero rows, files, predictions,
  summaries, or metrics. The frozen post-payload rule makes NER official
  results terminally missing; no retry is authorized. Independent review
  narrowed the shared Linux identity helper for future snapshots to retain
  kernel major/minor and RHEL family; five tests and Ruff pass. The released
  card now runs frozen News a0 seed-13 confirmation `1286000`. Sheet E/F/G
  remain blank.
- News a0 seed-13 `1286000` verified its immutable startup and exact first
  `3,095`-row validation boundary; execution-manifest SHA-256 is
  `a23ed55e921c78cd640cee0ba9b65ffea31fc4d943ce9e95054c85405d403ede`.
  POS Tswana `1282788` is still non-terminal at step `760/1,425` inside
  another fixed `750`-row boundary. AfriHG Zulu `1285972` is healthy at step
  `1,033/8,885`. Three owned A100-80GB jobs are active; held-out and Sheet
  state are unchanged.
- POS Tswana `1282788` and verifier `1286172` completed `0:0`; exact
  retained-to-final equality covered `424` keys and `71,762,560` values.
  POS Mono is frozen `3/3`. Independent review caught the NER-shaped verifier
  assumptions and mutable POS task URLs before held-out access. The corrected
  bundle pins revision `376f4161f0425584d4bd7664122b56fa026926d3`, passes
  nine focused checks and final review, and is frozen under note SHA-256
  `9b7fa2242c7d2fb4781a78df60d5fb3ee4a3732643edc917b35b2696ae33b79d`.
  Manifest/preflight jobs `1286173/1286174` completed `0:0`; one-time six-arm
  POS official job `1286189` is association-cap pending as the fourth owned
  submission. News `1286000`, SIB `1286171`, and AfriHG Zulu `1285972` run on
  A100-80GB. Sheet E/F/G remain blank.
- News a0 and SIB a0 seed-13 confirmations are terminal-valid: jobs
  `1286000/1286171` and CPU verifiers `1286716/1286717` completed `0:0` with
  exact retained-to-final equality at steps `1629/526`. News and SIB a1 jobs
  `1286721/1286722` are running. Preserve zero-second News launch `1286718`;
  its output root is absent.
- POS official job `1286189` failed before payload when execution attempted
  to rewrite a read-only resolved config; its result root is absent and no
  held-out row was touched. The isolated minimal correction is frozen under
  snapshot `pure-gdn-pos-official-test-20260902-v3-41626421` and note SHA-256
  `4be222798131779e6579fba75674c2154c23f6a377c9178abe8b5b75ca2b1ee0`.
  Manifest/preflight jobs `1286723/1286724` completed `0:0`; replacement POS
  official job `1286725` is association-cap pending as the fourth owned slot.
  AfriHG Zulu `1285972` remains running. Sheet E/F/G remain blank.
- News a1 and SIB a1 seed-13 jobs `1286721/1286722` completed `0:0` with
  exact fixed coverage; verifiers `1287267/1287268` proved exact
  retained-to-final equality across `424` keys and `71,762,560` values.
  Tested reduced-confirm ranker SHA-256
  `358d54fefdf4db60b6d041b28ca96ea07dca81ec1a5fbf51b0393668aba31147`
  requires exactly seeds 42/13. Ranking job `1287273` froze News to a0
  (`fd554517a4f1cbb81416e9522f3146ca0fe7b7944185a77f9b74380a748829a2`)
  and SIB to a0 under its exact-mean lower-LR tie-break
  (`24b3d47c2575a39d8c9c5378df4bdfec2f8c02c892e04104831e8e2fc4d9f346`).
- POS official replacement `1286725` passed immutable checks and entered the
  first task payload, then failed `1:0` after `42` seconds because the frozen
  runtime lacked `/home/lmbanr001/.cache/huggingface`. The result root has no
  files and therefore no held-out row or metric, but the one-time attempt is
  terminal and must never be retried. POS official remains missing.
- Intent a2/a1 seed-13 confirmations `1287271/1287272` are running. News
  English Mono `1287274` is association-cap pending at frozen a0 LR `3e-5`;
  AfriHG Zulu Mono `1285972` continues running. Four owned A100-80GB
  submissions are active. Sheet E/F/G remain blank.
- AfriHG Zulu Mono `1285972` completed `0:0` at retained checkpoint `7108`;
  CPU verifier `1287725` proved exact retained-to-final equality across `424`
  keys and `71,762,560` values. AfriHG Mono is frozen `2/2`. Cache-pinned
  AfriHG official preflight `1287733` verified six immutable manifests and
  four resolved configs after preserving pre-wrapper launcher failure
  `1287732`. One-time four-arm official job `1287734` is association-cap
  pending beside Intent `1287271/1287272` and News English Mono `1287274`.
  No AfriHG held-out payload is open and Sheet E/F/G remain blank.
- AfriHG official job `1287734` verified all manifests/configs and entered the
  first Xhosa test task, then failed `1:0` after `28` seconds when `datasets`
  attempted to create the absent home Hugging Face cache root. Its result tree
  has zero files: no prediction, summary, structural verification, or metric.
  The attempt is terminal and AfriHG current-adapter official results remain
  missing; never retry. Intent a2 confirmation `1287271` and verifier
  `1288158` completed `0:0` with exact `4,155`-row validation coverage and
  roundtrip equality across `424` keys and `71,762,560` values. Released
  slots now run SIB Afrikaans Mono `1288159` and queue SIB English Mono
  `1288160`; Intent a1 `1287272` and News English Mono `1287274` continue.
  Sheet E/F/G remain blank.
- News Xhosa `1288219`, SIB Northern/Southern Sotho `1288220/1288221`, and
  Intent English `1288222` completed `0:0`. CPU verifiers
  `1288931`-`1288934` proved exact retained-to-final equality for each across
  `424` keys and `71,762,560` values. Mono is now `16/21` frozen: News `2/2`,
  SIB `4/6`, Intent `1/4`, NER `3/3`, POS `3/3`, AfriHG `2/2`, and T2X `1/1`.
  Released cards run SIB Xhosa `1288936`, SIB Zulu `1288937`, Intent Xhosa
  `1288938`, and queue Intent Zulu `1288939`. Intent Southern Sotho is the
  sole unsubmitted Mono arm. Held-out and Sheet E/F/G are unchanged.
- SIB Xhosa/Zulu `1288936/1288937` and Intent Xhosa `1288938` completed
  `0:0`. CPU verifiers `1288988`-`1288990` proved exact retained-to-final
  equality across `424` keys and `71,762,560` values for each adapter. SIB
  Mono is frozen `6/6`, Intent is `2/4`, and total Mono progress is `19/21`
  frozen. Final Intent Southern Sotho job `1288991` was submitted once after
  the frozen absence, duplicate, config, active-cap, and recipe checks; it
  runs beside Intent Zulu `1288939` on A100-80GB. Held-out and Sheet E/F/G
  remain unchanged.
- Intent Zulu `1288939` and exact CPU roundtrip verifier `1289009` completed
  `0:0`; retained checkpoint `750` exactly matches the final adapter across
  `424` keys and `71,762,560` values. Intent is frozen `3/4` and total Mono
  progress is `20/21`. Intent Southern Sotho `1288991` is the sole remaining
  Mono job and is running on A100-80GB. Held-out and Sheet E/F/G remain
  unchanged.
- Intent Southern Sotho `1288991` and exact CPU roundtrip verifier `1289188`
  completed `0:0`; retained checkpoint `250` exactly matches the final adapter
  across `424` keys and `71,762,560` values. All `21/21` Mono adapters are
  frozen. SIB cache `1289189` verified six fixed configurations with `204`
  rows each and no model metric. Independent review blocked the first
  pre-payload snapshot because it omitted the structural verifier. The
  corrected snapshot, full cache seal, manifests, and metric-free preflight
  passed under jobs `1289206`-`1289208`; one-time twelve-arm SIB Mono/Multi
  official job `1289209` is submitted on A100-80GB. Official metrics remain
  uninspected and Sheet E/F/G remain blank.
- SIB official job `1289209` failed before payload in two seconds because its
  immutable `v2` snapshot compared raw CPU/GPU Linux kernel build strings.
  The result root remained absent with zero model, dataset, prediction,
  summary, structural-verification, or metric artifacts. Final prospective
  public-seam correction note SHA-256 is
  `4c6fd9934109174f9da8fad2445de20e1f6d86ab68065ebf35eec9c88710b93f`.
  Fresh immutable manifest/preflight jobs `1289210/1289211` completed `0:0`,
  all twelve configs were sealed, and single A100-80GB replacement `1289212`
  was submitted once. Sheet E/F/G remain blank.
- SIB official replacement `1289212` completed `0:0` in `00:18:42`. All
  twelve fixed Mono/Multi language arms and structural-verification sidecars
  passed with exactly five prompts and `1,020` expanded rows per arm. The
  sealed 48-file result-tree manifest SHA-256 is
  `ad98b6c0bbf2c03d101fbc79232654d4bfdb125abb317840290a18fc227136d9`.
  Report-only six-language macro held-out F1 is `0.1367692769759137` for
  Multi and `0.07555764695434825` for Mono. No score may affect remaining
  work; Sheet E/F/G remain blank pending the complete obtainable non-General
  set and exact readback.
- News official job `1290454` entered the first English held-out task and then
  failed `1:0` in the frozen `lm-eval` path resolver before any example,
  prediction, result, or metric artifact was written. Its result root has zero
  files. The one-time attempt is terminal and News current-adapter official
  results remain missing; never retry it under the original protocol.
- Intent's first official job `1290452` failed before payload because the
  protocol configs had not been sealed after an earlier setup command aborted.
  The prospective sealing-only correction is recorded under note SHA-256
  `63c3e13a96485018d14f817f404c218f6e67c5ba7be6d7ffa80a6016018d01e5`.
  Corrected manifest, preflight, seal, and official jobs
  `1290455`-`1290458` completed `0:0`. All eight fixed Mono/Multi arms, 32
  files, five prompts per language, exact expected expanded-row counts, and
  structural sidecars verified. The sealed result-tree manifest SHA-256 is
  `44a89fde7b63361e586a2135e1dc46a756032a6fad9b8613ac55174832534576`.
  Report-only four-language macro held-out F1 is
  `0.002614876682742` for Mono and `0.001161031235439` for Multi.
- The obtainable current-adapter non-General official set is closed: Base is
  accepted `16/16`; SIB and Intent provide `10/21` Mono language rows and
  `10/20` Multi language rows. News, T2X, NER, POS, and AfriHG remain
  terminally missing under the one-time protocol. Exact Google Sheet readback
  verified `GDN Results!C10:G19`: dates and SIB/Intent E/F values match the
  sealed artifacts, Base D is unchanged, wrapping/date formats are intact,
  and every General G cell remains blank. No GPU job is active for this
  close-out.
- The later 30 August correction amendment supersedes the old Base `16/16`
  shorthand for paper-ready use. Corrected Base preparation v1 job `1323700`
  failed before data/model/metrics because its offline cache path was absent;
  that evidence is preserved. V2 CPU audit `1323715` completed `0:0`, proving
  16 packs, 187 tasks, 110,251 rows, raw prompting, no adapter, and no
  generation lanes. Fresh metric-free A100-80GB checkpoint canary `1323738`
  is capacity-pending, with CPU seal `1323744` dependency-held after it. No
  corrected Base held-out payload or Sheet value has been opened.
- General g2 `1319959` reached its 24-hour scheduler limit. POS alone was
  claimed and has zero result files or structural sidecar after partial
  inference, making that unit terminally missing with no retry. AfriMMLU,
  BelebeleSot, BelebeleXho and AfriHGZulu remain unclaimed and output-absent;
  a 48-hour execution-only continuation for exactly those four untouched
  units was frozen before submission.
  Continuation `1325602` is now submitted once and capacity-pending behind the
  older Base canary; its four unit claims remain absent.
- At 10:12 SAST on 9 September, live capacity showed six physically free
  A100-40GB cards, all A100-80GB cards occupied, and a heavily queued L40S
  pool. The prospective A100-40GB switch amendment was frozen before action.
  Never-started jobs `1323738`, `1323744` and `1325602` were cancelled with
  zero runtime and absent target outputs. Exact-checkpoint A100-40GB canary
  `1326110` completed `0:0` in 42 seconds; CPU seal `1326111` completed `0:0`
  and bound the corrected Base bundle with digest
  `07aa04f2b4f6514fc2c7ba36344d98848a0509866893550a6a7f4646bef0cb80`.
  General remainder `1326112` is running on `gpu:ampere`; Base groups
  `1326143/1326144` are running and `1326145` is resource-pending on the same
  GPU family. Actual CUDA inference is confirmed for General AfriMMLU and Base
  News/NER. POS remains terminally missing and no held-out value was consulted.
- Live ETA at 10:26 SAST: General is 16/19 structurally verified and should
  finish its two remaining obtainable units around 10:45-11:00 today. Base
  News is verified; the NER-first group projects roughly 6.5 hours for NER
  before its shorter remaining units. The POS-first group is waiting for the
  General card and remains the critical path; prior measured pace puts full
  corrected-Base completion around late 10 September to early 11 September.
- The live ETA changed materially at 10:59 SAST. General remainder `1326112`
  structurally verified AfriMMLU, BelebeleSot and BelebeleXho, then failed
  before an AfriHGZulu result because its frozen offline loader found no Zulu
  CSV. That claimed unit is terminally missing; General closes at `17/19`
  structurally verified, with POS and AfriHGZulu missing. Base group `1326143`
  completed `0:0` with all six assigned units verified. Base group `1326145`
  failed before a POS result because its frozen offline loader could not find
  the fixed MasakhaPOS Tswana training URL; POS is terminally missing. Its four
  untouched later units remain eligible. Remainder `1326407` was submitted
  once after exact absence checks and is resource-pending on `gpu:ampere`.
  Group `1326144` remains healthy on NER (`1601/14980`, about 5.5 hours
  remaining at the latest sample). No metric value affected these decisions.
- Base remainder `1326407` started on `gpu:ampere` at 11:04 SAST. At 11:35,
  AfriMGSM was `1137/5000` with about 1h42m remaining. Group `1326144` was
  still healthy on NER at `3025/14980`, about 4h58m remaining. No new errors
  or sidecars appeared; all six earlier Base sidecars remain verified.
- Base remainder `1326407` completed `0:0` in `02:13:10`. AfriMGSM and
  Belebele Afrikaans/siSwati/Xhosa all passed structural verification, taking
  corrected Base to `10/16`; POS remains terminally missing. Group `1326144`
  is the only remaining GPU job. At 13:40 SAST NER was `7993/14980` with
  about 2h54m remaining before Intent, AfriXNLI and two Belebele units. No
  metric value has been opened.
- General POS slowness is now traced to a preparation defect: the frozen full
  matrix specified batch 8, but the sealed General config was overwritten to
  batch 1. Job `1319959` therefore processed 7,216 requests serially and timed
  out at 6,604/7,216. AfriHG Zulu separately failed because its frozen loader
  ignored already-present CSVs while offline. The user-directed completion
  work now permits a prospectively labelled infrastructure recovery for these
  two result-missing units; no failed metric artifact exists or was inspected.
- Recovery preparation v1/v2 jobs `1327418/1327432` failed before model or
  metrics on stale validation URLs and are preserved. V3 preparation `1327446`
  completed `0:0` using 12 deterministic synthetic, non-test gate rows and
  exact offline Zulu CSV hashes. Validation-only batch-1/batch-8 equivalence
  job `1327451` is queued on the ratified A100-40GB `gpu:ampere` family. It must
  pass exact raw/filtered response, coverage and metric equality before the
  two official recovery jobs can be sealed or launched.
- Corrected Base job `1326144` completed `0:0` at 17:32:52 SAST in
  `07:15:29`. Its five assigned sidecars all pass `verified=true` and
  `metric_values_included=false`; corrected Base closes at 15/16, with only
  the earlier result-missing POS unit absent. Another user's job took the
  released A100-40GB card, so General batch gate `1327451` remains
  priority-pending without an output root. No metric value was opened.
- General POS recovery `1332061` completed `0:0` after `1-00:46:21`;
  its sidecar verifies 12 tasks and 7,216 rows at SHA-256
  `6169b238...d50787`. General closes at 19/19 structurally verified.
  Exact held-out headlines were then promoted to all 41 General cells in
  `GDN Results`. Corrected Base is 15/16: 35 verified values were rebound
  to the corrected artifacts and its three superseded POS zeros were cleared.
  Readback verified dates, values, 79 provenance notes, wrapping, dependent
  comparison values, and zero formula errors. Belebele is correctly labelled
  `acc_norm`; the exact release is
  `.audit/pure-gdn-final-official-results-20260911.json`.
- The user authorized a post-hoc corrected-Base POS completeness amendment on
  12 September. Failed job `1326145` remains preserved with zero result files;
  no test row, prediction, or metric existed. CPU seal `1334558` passed and
  froze only cache availability plus a fresh output path under binding
  `e468d523...9483b`; model, task rows, prompts, batch size, decoding, metric,
  and aggregation are unchanged. Single `gpu:ampere` job `1334564` is pending
  with a 48-hour limit and must not be repeated. Results remain 118/121 until
  its 12-task, 7,216-row structural verification and exact Sheet promotion.
