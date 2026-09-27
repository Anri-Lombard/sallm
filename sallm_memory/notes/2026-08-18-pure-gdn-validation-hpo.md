# Pure-GDN validation-only HPO — 2026-08-18

## AfriHG a0 steady after first exact checkpoint — 21:44 SAST

- AfriHG Stage-A a0 job `1246938` remains healthy and fault-free on
  `srvrocgpu010` A100-40GB at `3828/15410` after `04:26:09`, running near
  `3.07 s/step`. Its scientifically valid step-3082 checkpoint remains the
  only exact artifact; a0 is not terminal-valid and no AfriHG winner is
  selected.
- Jobs `1246939/1246940` remain Resources/Priority-pending without model or
  data access. Slurm projects a1 at `03:18:32 SAST` on 19 August and gives a2
  no estimate. All three request exactly one `gpu:ampere` A100-40GB with
  eight CPUs under `nlpgroup/a100`; no A100-80GB/L40S work overlaps.
- HEX quota remains home `88.6%`, scratch `39.6%`. Kombuys remains read-only
  with RTX 5090 untouched; held-out remains `0`, Sheet E/F/G remain blank,
  and the global freeze/publication gates remain closed.

## AfriHG a0 first exact checkpoint reconciles — 21:08 SAST

- AfriHG Stage-A a0 job `1246938` completed its first frozen exact callback
  at step `3082` and resumed healthy training near `3136/15410` on
  `srvrocgpu010` A100-40GB. The 128-row artifact covers exactly `64` Xho and
  `64` Zul, with zero empty/whitespace-only raw outputs, zero empty normalized
  predictions, and `64/64` unique predictions per language.
- Xho/Zul validation chrF is
  `20.87606798645913/21.52051846942758`; the exact arithmetic mean is
  `21.198293227943353`, matching `checkpoint-3082/trainer_state.json`
  `best_metric` and retained checkpoint. Artifact SHA-256 is
  `2f71557104f7c27ce60bdd890dcc21602c0768edc8f4918bb27a718576cf6e0c`.
  This is a scientifically valid within-run checkpoint only: a0 is not
  terminal-valid, no AfriHG winner is selected, and held-out remains untouched.
- Jobs `1246939/1246940` remain Resources/Priority-pending without model or
  data access; Slurm currently projects a1 at `03:18:32 SAST` on 19 August
  and gives a2 no estimate. All three request A100-40GB `gpu:ampere`; no
  A100-80GB/L40S work overlaps. HEX quota remains home `88.6%`, scratch
  `39.6%`; Kombuys remains read-only with RTX 5090 untouched. Sheet E/F/G
  remain blank and global freeze/publication gates remain closed.

## AfriHG a0 exact callback healthy but slower — 20:44 SAST

- AfriHG Stage-A a0 job `1246938` remains running on `srvrocgpu010`
  A100-40GB with no fault marker. Its step-3082 frozen exact callback is still
  active: the second language phase began at `20:32:58` and fresh generation
  activity was logged through `20:41:03`.
- No step-3082 JSONL exists yet. Automatic batch-size probing plus beam
  generation took about 33 minutes for the first language, so the earlier
  `20:15--20:25` estimate was too optimistic; the revised exact-artifact ETA
  is `21:00--21:15 SAST`. AfriHG remains scientifically `0/3` terminal-valid.
- Jobs `1246939/1246940` remain Priority-pending without model or data access;
  a1 is still projected at `06:18:00 SAST` on 19 August and a2 has no
  estimate. All three jobs request A100-40GB `gpu:ampere`, with no
  A100-80GB/L40S overlap. HEX quota remains home `88.6%`, scratch `39.6%`;
  Kombuys remains read-only with RTX 5090 untouched. Held-out remains `0`,
  Sheet E/F/G remain blank, and global freeze/publication gates remain closed.

## AfriHG a0 enters first exact callback — 20:14 SAST

- AfriHG Stage-A a0 job `1246938` reached step `3082/15410` cleanly on
  `srvrocgpu010` A100-40GB. The full 3,082-row validation-loss pass completed
  in `139.0128 s` with `eval_loss=2.3395319597663855`, including exact
  Xho/Zul coverage `1305/1777`. This loss is operational health evidence, not
  the frozen recipe selector.
- The frozen 128-prompt beam callback began at `19:59:51`; automatic batch
  size 64 was established and fresh generation activity was logged through
  `20:09:32`. No fault marker or step-3082 exact artifact exists yet. With one
  64-row batch per language, the artifact is now expected around
  `20:15--20:25 SAST`; AfriHG remains scientifically `0/3` terminal-valid.
- Jobs `1246939/1246940` remain Priority-pending without model or data access;
  Slurm projects a1 at `06:18:00 SAST` on 19 August and gives a2 no estimate.
  All three jobs request A100-40GB `gpu:ampere`, with no A100-80GB/L40S
  overlap. HEX quota is home `88.6%`, scratch `39.6%`; Kombuys remains
  read-only with RTX 5090 untouched. Held-out remains `0`, Sheet E/F/G remain
  blank, and global freeze/publication gates remain closed.

## AfriHG a0 approaches first validation boundary — 19:44 SAST

- AfriHG Stage-A a0 job `1246938` remains healthy and fault-free on
  `srvrocgpu010` A100-40GB at `2826/15410` after `02:26:17`, running near
  `3.04 s/step`. Step 3082 is due around `19:57 SAST`; the frozen 128-prompt
  beam callback keeps the first exact-artifact window at `20:40--21:30 SAST`.
- Jobs `1246939/1246940` remain Priority-pending without model or data
  access. Slurm's dynamic a1 estimate slipped again to `06:18:00 SAST` on
  19 August; a2 still has no estimate. AfriHG remains scientifically `0/3`
  terminal-valid.
- These are the only owned jobs and all request A100-40GB `gpu:ampere`; no
  A100-80GB/L40S work overlaps them. HEX quota remains home `88.6%`, scratch
  `39.6%`; Kombuys remains read-only at its last verified idle state with RTX
  5090 untouched. Held-out remains `0`, Sheet E/F/G remain blank, and global
  freeze/publication gates remain closed.

## AfriHG a0 steady; queue estimates regress — 18:43 SAST

- AfriHG Stage-A a0 job `1246938` remains healthy and fault-free on
  `srvrocgpu010` A100-40GB at about `1628/15410` after `01:25:40`. Throughput
  improved to about `2.98 s/step`, keeping the first epoch boundary near
  `19:56 SAST` and the tentative first exact-artifact window at
  `20:40--21:30 SAST`.
- Jobs `1246939/1246940` remain Resources/Priority-pending without model or
  data access. Slurm's dynamic a1 estimate regressed from this evening to
  `03:18:32 SAST` on 19 August; a2 still has no estimate. AfriHG remains
  scientifically `0/3` terminal-valid.
- These are the only owned jobs and all request A100-40GB `gpu:ampere`; no
  A100-80GB/L40S work overlaps them. HEX quota is home `88.6%`, scratch
  `39.6%`; Kombuys remains read-only at its last verified idle state with RTX
  5090 untouched. Held-out remains `0`, Sheet E/F/G remain blank, and global
  freeze/publication gates remain closed.

## AfriHG a0 trains; a1/a2 remain queued — 17:43 SAST

- AfriHG Stage-A a0 job `1246938` remains healthy on `srvrocgpu010`
  A100-40GB after `00:25:38`. It verified the immutable 694-file manifest,
  passed the GatedDeltaNet fast-kernel gate, and reached about `453/15410`
  steps at roughly `3.1--3.2 s/step`; targeted fault scans are empty.
- The first epoch boundary is step 3082. Current training throughput places
  that boundary near `20:00 SAST`; allowing full validation and the frozen
  128-prompt beam callback gives a tentative first exact-artifact window of
  `20:40--21:30 SAST`. This is operational health only: AfriHG remains
  scientifically `0/3` terminal-valid.
- Jobs `1246939/1246940` remain Resources/Priority-pending with no model or
  data access; Slurm tentatively projects a1 `1246939` at `19:09:50 SAST`
  and gives a2 no start estimate. These are exactly three owned A100-40GB
  jobs, with no A100-80GB/L40S overlap. HEX quota is home `88.6%`, scratch
  `39.6%`; Kombuys remains read-only at its last verified idle state with RTX
  5090 untouched. Held-out remains `0`, Sheet E/F/G remain blank, and the
  global freeze/publication gates remain closed.

## NER closes and selects b7; AfriHG Stage A starts — 17:18 SAST

- NER a2 seed-13 job `1245393` completed cleanly `0:0` at `17:01:37 SAST`
  after `16:36:09`. Its terminal step-7574 exact artifact scores mean span
  F1 `0.6733876882791980`, with Tsn/Xho/Zul
  `0.6545890502197449/0.6764265868774882/0.6891474277403609`. This is the
  second frozen threshold miss, so retained checkpoint 6492 remains the seed
  winner at `0.6774368136184235`.
- The terminal artifact covers exactly `192` rows and `64` per language, has
  no literal empty raw output or parser failure, `58` whitespace-only raw
  outputs, and `39/54/37` unique Tsn/Xho/Zul predictions. Debug,
  retained-state, final-config, final-adapter, trial, and execution-manifest
  SHA-256 values are
  `80c82291449e84b07b353d305e7791137c3494fd7d52de77a8a24dbc67efc4af`,
  `a78a8cada97510b4ed3adc8103355e5a1eee2a73a457c0d4da0b1c8c654f79ae`,
  `3ffa005b2777c6c9e2f887484f764dfd4452d84f187fdbadbd590a8b347308ad`,
  `8261fc03703a34b10150d6903043617c561b45476ead0caad98a7582e8f486ec`,
  `389c7ab5072bbbe2e68cc7cf0b34e6da3c9ba9dafbb2316abc7e2a8d34df867a`,
  and `9c18f338c4c41402217c8669c3e078c4be303d65f66d498b9efef07f952af602`;
  the manifest sidecar passes `sha256sum -c`.
- All four NER confirmations are now terminal-valid. The preregistered
  validation-only three-seed ranking selects b7 over a2: means
  `0.6825166742032610` versus `0.6778547247706191`, with sample standard
  deviations `0.0111051221474177/0.0015031186491821`. The ranking artifact is
  `sallm_memory/artifacts/2026-08-18-pure-gdn-ner-confirmation-ranking.json`,
  SHA-256
  `fa0f76dcf79f8907a72dcc2bf658dc5dfdd043df962f125f237f4febafbcb6fd`;
  it records validation-only selection and no held-out metric access.
- With zero owned jobs and absent AfriHG Stage-A outputs, frozen candidates
  a0/a1/a2 were submitted once as jobs `1246938/1246939/1246940` from the
  694-file immutable snapshot. Job `1246938` started at `17:18:08` on
  `srvrocgpu010`, wrote manifest SHA-256
  `e870320d2f11324440c828437479a121dba8cc95bbdff3d106c02eb0923556a2`,
  verified all 694 files, and passed the GatedDeltaNet fast-kernel gate.
  Jobs `1246939/1246940` are Resources/Priority-pending with no model or data
  access. All three use `nlpgroup/a100/nlpgroup`, one A100-40GB
  `gpu:ampere`, 24 hours, eight CPUs, and the mandated home working directory.
- AfriHG Stage A is operationally `1 running + 2 pending` and scientifically
  `0/3` terminal-valid. Its first exact-metric ETA remains pending startup
  throughput; job `1246938` has a hard wall limit near `17:18 SAST` on
  19 August. There is no owned A100-80GB/L40S work. HEX quota is home `88.6%`
  and scratch `39.4%`; Kombuys remains read-only at its last verified idle
  state with RTX 5090 untouched. Held-out remains `0`, Sheet E/F/G remain
  blank, Hugging Face publication remains blocked, and the global eight-family
  freeze is still closed by AfriHG, General, and remaining family gates.

## NER a2 seed 87 terminal-valid; seed 13 at closing boundary — 16:42 SAST

- Seed-87 job `1245394` completed cleanly `0:0` at `16:42:04 SAST` after
  `16:16:36`. Its terminal step-7574 exact artifact scores mean span F1
  `0.6766047840600772`, with Tsn/Xho/Zul
  `0.6587131367291726/0.6804485305839966/0.6906526848670627`.
- The numerical gain over step-6492 best `0.6765487441685071` is only
  `0.0000560398915701`, below the frozen `0.001` early-stopping threshold.
  Trainer state therefore records the second consecutive threshold miss and
  terminates as preregistered, while retaining the strictly highest numerical
  checkpoint 7574.
- The terminal artifact covers exactly `192` rows and `64` per language, has
  no literal empty raw output or parser failure, `51` whitespace-only raw
  outputs, and `40/56/38` unique Tsn/Xho/Zul outputs. The independently
  recomputed language mean exactly matches trainer state.
- Debug/trainer-state/adapter-config/adapter-model SHA-256 values are
  `77e3efd49ecef909362f4d8cf24cbbfb2044521ac2bc06e9e35c12c8913681a1`,
  `78912f11a520f56eda83774512bcb75f3b69eea011a4ed7539fdfcdfc6fe3750`,
  `99132c152764b8965e4be2a757d975dcc4609debf27310db9ed1d5414b895520`,
  and `9cac1283266e0cc07e8947fa06cacc71feecb54758f57c5f800498f0fa90f784`.
  Trial and execution-manifest SHA-256 values are
  `ccb7d90af0d5b2f37ebdf84df10c07c63f28faba82ee5df49da4ec410a56227c`
  and `c12d0921153bec2da784e05752d140ca261e94a1ffcc1e9cda76583842017fc8`.
- NER is now scientifically `3/4` terminal-valid. Seed-13 job `1245393`
  reached step 7574, completed declared validation over all `10,760` rows at
  provenance-only loss `0.38513834786680995`, and remains healthy inside its
  exact callback with fresh probes through `16:34`. A second threshold miss
  will terminate it; a qualifying improvement will allow the final epoch.
  Its next artifact is tentatively expected around `17:00--17:20 SAST`.
- Exactly one owned A100-40GB `gpu:ampere` job remains; there is no owned
  A100-80GB/L40S overlap. HEX quota is home `88.6%`, scratch `39.5%`;
  held-out remains `0`, Sheet E/F/G remain blank, Kombuys remains read-only
  at its last verified idle state, and no winner is frozen yet.

## Both NER a2 seeds retain checkpoint 6492 after step 7033 — 16:12 SAST

- Seed-13 job `1245393` scored mean exact span F1
  `0.6723811055906449` at step 7033, with Tsn/Xho/Zul
  `0.6549707602338681/0.673098448909923/0.6890741076281436`. This did not
  exceed retained step-6492 best `0.6774368136184235`, so checkpoint 6492
  remains retained and frozen patience advances to `1/2`, matching seed 87.
- The seed-13 artifact covers exactly `192` rows and `64` per language, has
  no literal empty raw output or parser failure, `58` whitespace-only raw
  outputs, and `39/54/37` unique Tsn/Xho/Zul outputs. Its independently
  recomputed language mean exactly matches trainer state.
- Seed-13 debug/trainer-state/adapter-config SHA-256 values are
  `fd4ec8df85f899777e951df0cbe65cf28de3730d2bd6631f13cbdb622170b75e`,
  `2559666aa7985726af4103c365518f7ff3d1a2dcb0ca680371dede420d8043a8`,
  and `3ffa005b2777c6c9e2f887484f764dfd4452d84f187fdbadbd590a8b347308ad`.
- Seed 13 resumed healthy near `7459/8115`. Seed-87 job `1245394` reached
  step `7574/8115`, completed declared validation over all `10,760` rows at
  provenance-only loss `0.3943623709412756`, and entered its exact callback
  with a fresh generation probe at `16:07`. A non-improvement at this boundary
  would satisfy frozen patience for seed 87. The next complete artifacts are
  tentatively expected around `16:40--17:30 SAST`.
- Both jobs remain operationally healthy with the verified GDN fast path and
  no targeted fault marker. Scientifically NER remains `2/4` terminal-valid
  and no winner is frozen. They are the only owned jobs, both A100-40GB
  `gpu:ampere`, with no owned A100-80GB/L40S overlap. HEX quota is home
  `88.6%`, scratch `39.4%`; held-out remains `0`, Sheet E/F/G remain blank,
  Kombuys remains read-only at its last verified idle state, and all gates
  remain unchanged.

## NER a2 seed 87 first non-improvement after checkpoint 6492 — 15:42 SAST

- Seed-87 job `1245394` scored mean exact span F1
  `0.6758961608549887` at step 7033, with Tsn/Xho/Zul
  `0.6572920851747189/0.6795497982586038/0.6908465991316433`. This did not
  exceed retained step-6492 best `0.6765487441685071`, so checkpoint 6492
  remains retained and frozen patience advances to `1/2`.
- The artifact covers exactly `192` rows and `64` per language, has no
  literal empty raw output or parser failure, `51` whitespace-only raw
  outputs, and `40/56/38` unique Tsn/Xho/Zul outputs. The independently
  recomputed language mean exactly matches trainer state.
- Debug/trainer-state/adapter-config SHA-256 values are
  `811f7ff7f8c631ed848c8955c283c993a6cdfd6aed4423c69e0d5a6fa2a2c754`,
  `dcffd13439bb317e7473ffb7b1a10d9049fe7afda42c5ec148bc4b3d71dfa825`,
  and `99132c152764b8965e4be2a757d975dcc4609debf27310db9ed1d5414b895520`.
- Seed 87 resumed healthy near `7247/8115`. Seed-13 job `1245393` completed
  declared step-7033 validation over all `10,760` rows at provenance-only
  loss `0.3826227578088697` and remains inside its exact callback, with fresh
  generation probes through `15:39`; its artifact is tentatively expected
  around `15:55--16:15 SAST`.
- Both jobs remain operationally healthy with the verified GDN fast path and
  no targeted fault marker. Scientifically NER remains `2/4` terminal-valid
  and no winner is frozen. They are the only owned jobs, both A100-40GB
  `gpu:ampere`, with no owned A100-80GB/L40S overlap. HEX quota is home
  `88.6%`, scratch `39.3%`; held-out remains `0`, Sheet E/F/G remain blank,
  Kombuys remains read-only at its last verified idle state, and all gates
  remain unchanged.

## NER a2 step-7033 validation cycle active — 15:12 SAST

- Both jobs reached step `7033/8115`. Seed-87 job `1245394` completed
  declared validation over all `10,760` rows at provenance-only loss
  `0.392970134068599` and entered its 192-prompt exact callback, with fresh
  automatic generation probes through `15:05`. Seed-13 job `1245393` is at
  the matching validation boundary; its complete declared metric and exact
  artifact are not yet present.
- No step-7033 exact artifact exists for either seed, so both retain
  checkpoint 6492 with frozen patience `0/2`. Targeted fault scans remain
  empty. The expected artifact window is approximately
  `15:30--16:10 SAST`.
- Both jobs remain operationally healthy with the verified GDN fast path on
  separate A100-40GB `gpu:ampere` devices. Scientifically NER remains `2/4`
  terminal-valid and no winner is frozen. They are the only owned jobs, with
  no owned A100-80GB/L40S overlap. HEX quota is home `88.6%`, scratch
  `39.3%`; held-out remains `0`, Sheet E/F/G remain blank, Kombuys remains
  read-only at its last verified idle state, and all gates remain unchanged.

## Both NER a2 seeds improve at step 6492 — 14:42 SAST

- Seed-13 job `1245393` improved its retained mean exact span F1 to
  `0.6774368136184235`, with Tsn/Xho/Zul
  `0.6582143333777694/0.6783644558917725/0.6957316515857283`. Seed-87 job
  `1245394` improved to `0.6765487441685071`, with
  `0.6609694218186174/0.6782802075611065/0.690396603125797`. Both now retain
  checkpoint 6492 with frozen patience `0/2`.
- Each artifact covers exactly `192` rows and `64` per language, has no
  literal empty raw output or parser failure, and its independently recomputed
  language mean exactly matches trainer state. Seed 13 has `57`
  whitespace-only raw outputs and `39/55/37` unique Tsn/Xho/Zul outputs;
  seed 87 has `51` whitespace-only raw outputs and `40/57/38` unique outputs.
- Seed-13 debug/trainer-state/adapter-config SHA-256 values are
  `287600a8d55420e4d6bd33902a25ae4f95ce9dea309272b769f7d2d12adc633c`,
  `a78a8cada97510b4ed3adc8103355e5a1eee2a73a457c0d4da0b1c8c654f79ae`,
  and `3ffa005b2777c6c9e2f887484f764dfd4452d84f187fdbadbd590a8b347308ad`.
  Seed-87 values are
  `54b1fa2e88c2fc211b508527348bea7f29e185ac00ff1de22e5f72a96286d3b8`,
  `b00db5ccaf3457428b40e91be4b05e784228589f923cb746307de4263cb736cd`,
  and `99132c152764b8965e4be2a757d975dcc4609debf27310db9ed1d5414b895520`.
- Both remain operationally healthy with the verified GDN fast path and no
  targeted fault marker. Seed 13 resumed near `6517/8115`; seed 87 resumed
  near `6892/8115`. The next exact artifacts are tentatively expected around
  `15:30--16:15 SAST`; terminal timing remains governed by frozen patience.
- These are valid interim artifacts, not terminal confirmations: NER remains
  scientifically `2/4` terminal-valid and no winner is frozen. The two jobs
  are the only owned work, both A100-40GB `gpu:ampere`, with no owned
  A100-80GB/L40S overlap. HEX quota is home `88.6%`, scratch `39.3%`;
  held-out remains `0`, Sheet E/F/G remain blank, Kombuys remains read-only
  at its last verified idle state, and all gates remain unchanged.

## NER a2 step-6492 exact callbacks active — 14:12 SAST

- Both jobs reached step `6492/8115` and completed declared validation over
  all `10,760` rows. Job `1245393` recorded provenance-only loss
  `0.379897379254763`; job `1245394` recorded `0.3869488684218169`.
  Selection remains based only on the separate exact span-F1 artifacts.
- Both 192-prompt exact callbacks are active. Seed 13 produced a fresh
  generation batch-size probe at `14:07`; seed 87 produced probes through
  `14:09`. No step-6492 exact artifact exists yet and targeted fault scans
  are empty. Seed 13 therefore retains checkpoint 5951 with patience `0/2`,
  while seed 87 retains checkpoint 5410 with patience `1/2`.
- The expected artifact window is approximately `14:25--15:00 SAST`. Both
  jobs remain operationally healthy with the verified GDN fast path on
  separate A100-40GB `gpu:ampere` devices. Scientifically NER remains `2/4`
  terminal-valid and no winner is frozen. They are the only owned jobs, with
  no owned A100-80GB/L40S overlap. HEX quota is home `88.6%`, scratch
  `39.3%`; held-out remains `0`, Sheet E/F/G remain blank, Kombuys remains
  read-only at its last verified idle state, and all gates remain unchanged.

## NER a2 seed 13 improves at step 5951 — 13:42 SAST

- Seed-13 job `1245393` improved its retained mean exact span F1 to
  `0.6755429869223111`, with Tsn/Xho/Zul
  `0.6541046251152485/0.6727688787184858/0.6997554569331988`. It now retains
  checkpoint 5951 and resets frozen patience from `1/2` to `0/2`.
- The artifact covers exactly `192` rows and `64` per language, has no
  literal empty raw output or parser failure, `58` whitespace-only raw
  outputs, and `39/54/37` unique Tsn/Xho/Zul outputs. The independently
  recomputed language mean exactly matches trainer state.
- Debug/trainer-state/adapter-config SHA-256 values are
  `d71dff7d7ccfb0de9c1453103757074711a820bf66766ecff1f85e8ee2154a7e`,
  `1636030b16afc22d8ab1dbfcecb2d8fb506684ffe4f0bae785acef3dc11d6fc4`,
  and `3ffa005b2777c6c9e2f887484f764dfd4452d84f187fdbadbd590a8b347308ad`.
- Seed 13 resumed healthy near `6161/8115`. Seed-87 job `1245394` reached
  its next step `6492/8115` validation boundary. The next complete artifacts
  are tentatively expected around `14:20--15:10 SAST`; terminal timing remains
  governed by the frozen patience rule.
- Both jobs remain operationally healthy with the verified GDN fast path and
  no targeted fault marker. Scientifically NER remains `2/4` terminal-valid
  and no winner is frozen. They are the only owned jobs, both A100-40GB
  `gpu:ampere`, with no owned A100-80GB/L40S overlap. HEX quota is home
  `88.6%`, scratch `39.3%`; held-out remains `0`, Sheet E/F/G remain blank,
  Kombuys remains read-only at its last verified idle state, and all gates
  remain unchanged.

## NER a2 seed 87 first non-improvement after checkpoint 5410 — 13:12 SAST

- Seed-87 job `1245394` scored mean exact span F1
  `0.6714130986587395` at step 5951, with Tsn/Xho/Zul
  `0.6585719941929024/0.6750409035730752/0.6806263982102408`. This did not
  exceed retained step-5410 best `0.6743755031188132`, so checkpoint 5410
  remains retained and frozen patience advances to `1/2`.
- The artifact covers exactly `192` rows and `64` per language, has no
  literal empty raw output or parser failure, `51` whitespace-only raw
  outputs, and `40/57/38` unique Tsn/Xho/Zul outputs. The independently
  recomputed language mean exactly matches trainer state.
- Debug/trainer-state/adapter-config SHA-256 values are
  `28e6d10f7897f6fcdf3f01ccc4b56e7c6fc1400435d20a68be775b3d01a02a84`,
  `8399cbbabaad914817acea54fc653a1f8f1e7a1bdf756c0aa5f4cd651c116a6e`,
  and `99132c152764b8965e4be2a757d975dcc4609debf27310db9ed1d5414b895520`.
- Seed 87 resumed healthy just after step 5951. Seed-13 job `1245393`
  reached step `5951/8115`, completed declared validation over all `10,760`
  rows at provenance-only loss `0.3729733151574117`, and entered its exact
  callback with fresh generation probes through `13:05`; its artifact is
  tentatively expected around `13:30--13:50 SAST`.
- Both jobs remain operationally healthy with the verified GDN fast path and
  no targeted fault marker. Scientifically NER remains `2/4` terminal-valid
  and no winner is frozen. They are the only owned jobs, both A100-40GB
  `gpu:ampere`, with no owned A100-80GB/L40S overlap. HEX quota is home
  `88.6%`, scratch `39.4%`; held-out remains `0`, Sheet E/F/G remain blank,
  Kombuys remains read-only at its last verified idle state, and all gates
  remain unchanged.

## NER a2 seed 13 first non-improvement after checkpoint 4869 — 12:42 SAST

- Seed-13 job `1245393` scored mean exact span F1
  `0.6697110944357454` at step 5410, with Tsn/Xho/Zul
  `0.646886641830711/0.6691664881079425/0.6930801533685825`. This did not
  exceed retained step-4869 best `0.6700645113839289`, so checkpoint 4869
  remains retained and frozen patience advances to `1/2`.
- The artifact covers exactly `192` rows and `64` per language, has no
  literal empty raw output or parser failure, `52` whitespace-only raw
  outputs, and `41/56/38` unique Tsn/Xho/Zul outputs. The independently
  recomputed language mean exactly matches trainer state.
- Debug/trainer-state/adapter-config SHA-256 values are
  `beea360ceb0ad66e362670c560f403a488134eeaa1db60581d79d018381128ee`,
  `85238f041db70841b1dfa5036f077478e1f8971edf3c05cb23a53373d18c5f2d`,
  and `3ffa005b2777c6c9e2f887484f764dfd4452d84f187fdbadbd590a8b347308ad`.
- Seed 13 resumed healthy near `5817/8115`. Seed-87 job `1245394` reached
  step `5951/8115`, completed declared validation over all `10,760` rows at
  provenance-only loss `0.3803239560038627`, and entered its exact callback
  with a fresh generation probe at `12:37`. The next complete artifacts are
  tentatively expected around `13:10--14:00 SAST`.
- Both jobs remain operationally healthy with the verified GDN fast path and
  no targeted fault marker. Scientifically NER remains `2/4` terminal-valid
  and no winner is frozen. They are the only owned jobs, both A100-40GB
  `gpu:ampere`, with no owned A100-80GB/L40S overlap. HEX quota is home
  `88.6%`, scratch `39.3%`; held-out remains `0`, Sheet E/F/G remain blank,
  Kombuys remains read-only at its last verified idle state, and all gates
  remain unchanged.

## NER a2 seed 87 improves at step 5410 — 12:12 SAST

- Seed-87 job `1245394` improved its retained mean exact span F1 to
  `0.6743755031188132`, with Tsn/Xho/Zul
  `0.6692586137551452/0.6657387580299289/0.6881291375713656`. It now retains
  checkpoint 5410 with frozen patience `0/2`.
- The artifact covers exactly `192` rows and `64` per language, has no
  literal empty raw output or parser failure, `53` whitespace-only raw
  outputs, and `40/55/38` unique Tsn/Xho/Zul outputs. The independently
  recomputed language mean exactly matches trainer state.
- Debug/trainer-state/adapter-config SHA-256 values are
  `82e43c8e60003b98522861bc475566079bb10c3464f04e168e3178424e3fcdf8`,
  `055c6c5b10026802e5a0244762ffcb69adaa22a7e7358f5a0297b90694791a3b`,
  and `99132c152764b8965e4be2a757d975dcc4609debf27310db9ed1d5414b895520`.
- Seed 87 resumed healthy near `5627/8115`. Seed-13 job `1245393` remains
  healthy inside its matching step-5410 exact callback after complete
  declared validation over all `10,760` rows at provenance-only loss
  `0.36796710500043567`; fresh generation probes continued through `12:10`,
  and its artifact is tentatively expected around `12:20--12:35 SAST`.
- Both jobs remain operationally healthy with the verified GDN fast path and
  no targeted fault marker. Scientifically NER remains `2/4` terminal-valid
  and no winner is frozen. They are the only owned jobs, both A100-40GB
  `gpu:ampere`, with no owned A100-80GB/L40S overlap. HEX quota is home
  `88.6%`, scratch `39.3%`; held-out remains `0`, Sheet E/F/G remain blank,
  Kombuys remains read-only at its last verified idle state, and all gates
  remain unchanged.

## NER a2 seed 13 improves at step 4869 — 11:42 SAST

- Seed-13 job `1245393` improved its retained mean exact span F1 to
  `0.6700645113839289`, with Tsn/Xho/Zul
  `0.6535476718403048/0.6670677678717439/0.6895780944397377`. It now retains
  checkpoint 4869 with frozen patience `0/2`.
- The artifact covers exactly `192` rows and `64` per language, has no
  literal empty raw output or parser failure, `59` whitespace-only raw
  outputs, and `39/53/37` unique Tsn/Xho/Zul outputs. The independently
  recomputed language mean exactly matches trainer state.
- Debug/trainer-state/adapter-config SHA-256 values are
  `624f6ed2d6932ed21d5eac250b8983c7533575d481ddd64f78be97ca5c737db6`,
  `1a570285e60d732a46b53496273df1f622aee41652bea7c2c56cae3be41bc2f7`,
  and `3ffa005b2777c6c9e2f887484f764dfd4452d84f187fdbadbd590a8b347308ad`.
- Both seed 13 and seed 87 have reached step `5410/8115` and are inside the
  next validation/exact-callback cycle, with fresh exact-generation probes
  through `11:40/11:34`. Seed 87's declared step-5410 validation covered all
  `10,760` rows at provenance-only loss `0.37851249382841545`; selection
  remains based only on the separate exact artifact. The next complete
  artifacts are tentatively expected around `12:00--12:30 SAST`.
- Both jobs remain operationally healthy with the verified GDN fast path and
  no targeted fault marker. Scientifically NER remains `2/4` terminal-valid
  and no winner is frozen. They are the only owned jobs, both A100-40GB
  `gpu:ampere`, with no owned A100-80GB/L40S overlap. HEX quota is home
  `88.6%`, scratch `39.3%`; held-out remains `0`, Sheet E/F/G remain blank,
  Kombuys remains read-only at its last verified idle state, and all gates
  remain unchanged.

## NER a2 seed 87 improves at step 4869 — 11:12 SAST

- Seed-87 job `1245394` improved its retained mean exact span F1 to
  `0.6675332242184386`, with Tsn/Xho/Zul
  `0.6587944528353201/0.6679569892472622/0.6758482305727337`. It now retains
  checkpoint 4869 and resets frozen patience from `1/2` to `0/2`.
- The artifact covers exactly `192` rows and `64` per language, has no
  literal empty raw output or parser failure, `54` whitespace-only raw
  outputs, and `40/56/38` unique Tsn/Xho/Zul outputs. The independently
  recomputed language mean exactly matches trainer state.
- Debug/trainer-state/adapter-config SHA-256 values are
  `dd1fb6f424ec39490c8a535c653422f28fe89ed4f89d2820861c5cdd27209dff`,
  `c7c4c6f49741708cbb2b082078067f8d8359d1fd87f949894d1c4a6d809cc73f`,
  and `99132c152764b8965e4be2a757d975dcc4609debf27310db9ed1d5414b895520`.
- Seed 87 resumed healthy near `5278/8115`. Seed-13 job `1245393` remains
  healthy inside its matching step-4869 exact callback, with fresh generation
  probes through `11:01`; its artifact is expected around
  `11:13--11:20 SAST`.
- Both jobs retain the verified GDN fast path with no targeted fault marker.
  Scientifically NER remains `2/4` terminal-valid and no winner is frozen.
  They are the only owned jobs, both A100-40GB `gpu:ampere`, with no owned
  A100-80GB/L40S overlap. HEX quota is home `88.6%`, scratch `39.3%`;
  held-out remains `0`, Sheet E/F/G remain blank, Kombuys remains read-only
  at its last verified idle state, and all gates remain unchanged.

## NER a2 step-4869 exact callbacks active — 10:42 SAST

- Both jobs reached step `4869/8115` and completed declared validation over
  all `10,760` rows. Job `1245393` recorded provenance-only loss
  `0.3637493601518936`; job `1245394` recorded `0.37202444537421586`.
  Selection remains based only on the separate exact span-F1 artifacts.
- Both 192-prompt exact callbacks are active. Seed 13 produced a fresh
  generation batch-size probe at `10:40`; seed 87 produced probes through
  `10:39`. No step-4869 exact artifact exists yet and targeted fault scans are
  empty. Seed 13 therefore retains checkpoint 4328 with patience `0/2`; seed
  87 retains checkpoint 3787 with patience `1/2`.
- The expected artifact window is approximately `10:50--11:20 SAST`. Both
  jobs remain operationally healthy with the verified GDN fast path on
  separate A100-40GB `gpu:ampere` devices. Scientifically NER remains `2/4`
  terminal-valid and no winner is frozen. They are the only owned jobs, with
  no owned A100-80GB/L40S overlap. HEX quota is home `88.6%`, scratch
  `39.3%`; held-out remains `0`, Sheet E/F/G remain blank, Kombuys remains
  read-only at its last verified idle state, and all gates remain unchanged.

## NER a2 seed 13 improves at step 4328 — 10:12 SAST

- Seed-13 job `1245393` improved its retained mean exact span F1 to
  `0.6548998792345401`, with Tsn/Xho/Zul
  `0.6227373355905218/0.660350394325891/0.6816119077872071`. It now retains
  checkpoint 4328 and resets frozen patience from `1/2` to `0/2`.
- The artifact covers exactly `192` rows and `64` per language, has no
  literal empty raw output or parser failure, `55` whitespace-only raw
  outputs, and `40/55/37` unique Tsn/Xho/Zul outputs. The independently
  recomputed language mean exactly matches trainer state.
- Debug/trainer-state/adapter-config SHA-256 values are
  `89094dfb926d4fe1494711b252e6bcb0dc1c906e90f7b991a5420707fe59436d`,
  `955a1a5b974b63410450d02adcfeb021f42349ea00da7e19c2645b5be1cb2082`,
  and `3ffa005b2777c6c9e2f887484f764dfd4452d84f187fdbadbd590a8b347308ad`.
- Seed 13 resumed healthy near `4478/8115`. Seed-87 job `1245394` reached
  its next step-4869 validation boundary. The next exact artifacts are
  tentatively due around `11:00--11:30 SAST`; terminal timing remains
  governed by the frozen patience rule.
- Both jobs remain operationally healthy with the verified GDN fast path and
  no targeted fault marker. Scientifically NER remains `2/4` terminal-valid
  and no winner is frozen. They are the only owned jobs, both A100-40GB
  `gpu:ampere`, with no owned A100-80GB/L40S overlap. HEX quota is home
  `88.6%`, scratch `39.3%`; held-out remains `0`, Sheet E/F/G remain blank,
  Kombuys remains read-only at its last verified idle state, and all gates
  remain unchanged.

## NER a2 seed 87 first non-improvement — 09:42 SAST

- Seed-87 job `1245394` scored mean exact span F1
  `0.6534121964705447` at step 4328, with Tsn/Xho/Zul
  `0.6419981498611896/0.6541695677788587/0.6640688717715858`. This did not
  exceed retained step-3787 best `0.6576473598211583`, so checkpoint 3787
  remains retained and frozen patience advances to `1/2`.
- The step-4328 artifact covers exactly `192` rows and `64` per language,
  has no literal empty raw output or parser failure, `49` whitespace-only raw
  outputs, and `42/56/39` unique Tsn/Xho/Zul outputs. The independently
  recomputed language mean matches the step metric.
- Debug/trainer-state/adapter-config SHA-256 values are
  `21ce3dd9736868fa56275d4ee926292d362ed54e3ef70c6b49c5885c7e8aadf4`,
  `ca88bae4da94076737a0aca9a3307f7df3632d888ed75e34386b2feb819b9ac9`,
  and `99132c152764b8965e4be2a757d975dcc4609debf27310db9ed1d5414b895520`.
- Seed 87 resumed healthy just after step 4328. Seed-13 job `1245393`
  completed declared step-4328 validation over all `10,760` rows at
  provenance-only loss `0.35468309820806226` and remains inside its exact
  callback, with fresh generation probes through `09:38`; its artifact is
  expected around `09:50--10:00 SAST`.
- Both jobs remain operationally healthy with the verified GDN fast path and
  no targeted fault marker. Scientifically NER remains `2/4` terminal-valid
  and no winner is frozen. They are the only owned jobs, both A100-40GB
  `gpu:ampere`, with no owned A100-80GB/L40S overlap. HEX quota is home
  `88.6%`, scratch `39.4%`; held-out remains `0`, Sheet E/F/G remain blank,
  Kombuys remains read-only at its last verified idle state, and all gates
  remain unchanged.

## NER a2 seed 13 first non-improvement — 09:12 SAST

- Seed-13 job `1245393` scored mean exact span F1
  `0.6387472783390758` at step 3787, with Tsn/Xho/Zul
  `0.6138962181178043/0.6381893860561415/0.6641562308432815`. This did not
  exceed retained step-3246 best `0.6461748976768455`, so checkpoint 3246
  remains retained and frozen patience advances to `1/2`.
- The step-3787 artifact covers exactly `192` rows and `64` per language,
  has no literal empty raw output or parser failure, `50` whitespace-only raw
  outputs, and `41/56/39` unique Tsn/Xho/Zul outputs. The independently
  recomputed language mean matches the step metric.
- Debug/trainer-state/adapter-config SHA-256 values are
  `03aadc062f64aad6c1bcfd8c6469778666f562c7b36d014e3e3acf4799a3b5e1`,
  `7032794a1c5781fc2fd93c2f79a2350de1e99386768172950c3e1c79bf3b3448`,
  and `3ffa005b2777c6c9e2f887484f764dfd4452d84f187fdbadbd590a8b347308ad`.
- Seed 13 resumed healthy near `4136/8115`. Seed-87 job `1245394` reached
  step 4328, completed declared validation over all `10,760` rows at
  provenance-only loss `0.3662617509693018`, and entered its exact callback,
  with a fresh generation probe at `09:08`. The next exact artifacts are
  tentatively due around `09:40--10:10 SAST`.
- Both jobs remain operationally healthy with the verified GDN fast path and
  no targeted fault marker. Scientifically NER remains `2/4` terminal-valid
  and no winner is frozen. They are the only owned jobs, both A100-40GB
  `gpu:ampere`, with no owned A100-80GB/L40S overlap. HEX quota is home
  `88.6%`, scratch `39.3%`; held-out remains `0`, Sheet E/F/G remain blank,
  Kombuys remains read-only at its last verified idle state, and all gates
  remain unchanged.

## NER a2 seed 87 improves at step 3787 — 08:42 SAST

- Seed-87 job `1245394` improved its retained mean exact span F1 to
  `0.6576473598211583`, with Tsn/Xho/Zul
  `0.65420812046244/0.6522476675147932/0.6664862914862416`. It now retains
  checkpoint 3787 with frozen patience `0/2`.
- The artifact covers exactly `192` rows and `64` per language, has no
  literal empty raw output or parser failure, `52` whitespace-only raw
  outputs, and `40/57/38` unique Tsn/Xho/Zul outputs. The independently
  recomputed language mean exactly matches trainer state.
- Debug/trainer-state/adapter-config SHA-256 values are
  `50f264a373b8698a3fbd3c100b51b2d175ffbec9d51524b4e9d21ed4338bb320`,
  `913ea679b4c2aedfdb9ce7614a8c3068f2abc2e12fb32053137ddc4ff22a3b8e`,
  and `99132c152764b8965e4be2a757d975dcc4609debf27310db9ed1d5414b895520`.
- Seed 87 resumed healthy near `3973/8115`. Seed-13 job `1245393` completed
  declared step-3787 validation over all `10,760` rows at provenance-only loss
  `0.36098667981456206` and remains inside its exact callback, with fresh
  generation probes through `08:28`; its artifact is expected around
  `08:50--09:00 SAST`.
- Both jobs remain operationally healthy with the verified GDN fast path and
  no targeted fault marker. Scientifically NER remains `2/4` terminal-valid
  and no winner is frozen. They are the only owned jobs, both A100-40GB
  `gpu:ampere`, with no owned A100-80GB/L40S overlap. HEX quota is home
  `88.6%`, scratch `39.3%`; held-out remains `0`, Sheet E/F/G remain blank,
  Kombuys remains read-only at its last verified idle state, and all gates
  remain unchanged.

## NER a2 seed 13 improves at step 3246 — 08:12 SAST

- Seed-13 job `1245393` improved its retained mean exact span F1 to
  `0.6461748976768455`, with Tsn/Xho/Zul
  `0.6300866369125255/0.6362632160327732/0.6721748400852379`. It now retains
  checkpoint 3246 with frozen patience `0/2`.
- The artifact covers exactly `192` rows and `64` per language, has no
  literal empty raw output or parser failure, `50` whitespace-only raw
  outputs, and `41/57/39` unique Tsn/Xho/Zul outputs. The independently
  recomputed language mean exactly matches trainer state.
- Debug/trainer-state/adapter-config SHA-256 values are
  `6bc90db1ee5cf914d6b81fc4abbd58bfa28db2cc2c3373d46530c2bbcba026d6`,
  `b9e97d745da4f3fab81b90ea241be4749718a5437ac7ead289cc6cd4f2707fd9`,
  and `3ffa005b2777c6c9e2f887484f764dfd4452d84f187fdbadbd590a8b347308ad`.
- Seed 13 reached its next step-3787 validation boundary. Seed-87 job
  `1245394` completed declared step-3787 validation over all `10,760` rows at
  provenance-only loss `0.3610744887568251` and is inside its exact callback,
  with fresh generation probes through `08:05`. The next exact artifacts are
  tentatively due around `08:20--08:50 SAST`.
- Both jobs remain operationally healthy with the verified GDN fast path and
  no targeted fault marker. Scientifically NER remains `2/4` terminal-valid
  and no winner is frozen. They are the only owned jobs, both A100-40GB
  `gpu:ampere`, with no owned A100-80GB/L40S overlap. HEX quota is home
  `88.6%`, scratch `39.3%`; held-out remains `0`, Sheet E/F/G remain blank,
  Kombuys remains read-only at its last verified idle state, and all gates
  remain unchanged.

## NER a2 seed 87 improves at step 3246 — 07:42 SAST

- Seed-87 job `1245394` improved its retained mean exact span F1 to
  `0.6326330113434602`, with Tsn/Xho/Zul
  `0.6264628941208368/0.6207684771959883/0.6506676627135555`. It now retains
  checkpoint 3246 with frozen patience `0/2`.
- The artifact covers exactly `192` rows and `64` per language, has no
  literal empty raw output or parser failure, `55` whitespace-only raw
  outputs, and `40/56/37` unique Tsn/Xho/Zul outputs. The independently
  recomputed language mean exactly matches trainer state.
- Debug/trainer-state/adapter-config SHA-256 values are
  `6c7d21200f31db08f833f6d530a2fda79c1126660408a78780e9d3f6aa6aba72`,
  `52fe7b12ec242fdf698f56cff2ad3ddddf80e652f033018de41de350dd90eea5`,
  and `99132c152764b8965e4be2a757d975dcc4609debf27310db9ed1d5414b895520`.
- Seed 87 resumed healthy near `3633/8115`. Seed-13 job `1245393` remains
  healthy inside its matching step-3246 exact callback, with fresh generation
  probes through `07:30`; its artifact is expected around
  `07:43--07:50 SAST`. Both retain the verified GDN fast path and have no
  targeted fault marker.
- This remains an interim validation artifact: NER stays scientifically
  `2/4` terminal-valid and no winner is frozen. Both jobs are A100-40GB
  `gpu:ampere` and are the only owned work, with no owned A100-80GB/L40S
  overlap. HEX quota is home `88.6%`, scratch `39.3%`; held-out remains `0`,
  Sheet E/F/G remain blank, Kombuys remains read-only at its last verified
  idle state, and all gates remain unchanged.

## NER a2 step-3246 exact callbacks active — 07:12 SAST

- Both jobs reached step `3246/8115` and completed declared validation over
  all `10,760` rows. Job `1245393` recorded provenance-only loss
  `0.33066272381069933`; job `1245394` recorded `0.3471481550138679`.
  Selection remains based only on the separate exact span-F1 artifacts.
- Both 192-prompt exact callbacks are active. Seed 13 produced a fresh
  generation batch-size probe at `07:07`; seed 87 produced probes through
  `07:10`. No step-3246 artifact exists yet and targeted fault scans are
  empty, so both retain checkpoint 2705 with patience `0/2`. The artifact
  window remains approximately `07:20--07:45 SAST`.
- Both jobs remain operationally healthy with the verified GDN fast path on
  separate A100-40GB `gpu:ampere` devices. Scientifically NER remains `2/4`
  terminal-valid and no winner is frozen. They are the only owned jobs, with
  no owned A100-80GB/L40S overlap. HEX quota is home `88.6%`, scratch
  `39.3%`; held-out remains `0`, Sheet E/F/G remain blank, Kombuys remains
  read-only at its last verified idle state, and all gates remain unchanged.

## Both NER a2 seeds improve at step 2705 — 06:42 SAST

- Seed-13 job `1245393` improved its retained mean exact span F1 to
  `0.6263447401956433`, with Tsn/Xho/Zul
  `0.6132222800624175/0.6263140782485686/0.6394978622759439`. Seed-87 job
  `1245394` improved to `0.6288979354137386`, with
  `0.6386227960571486/0.6032306375694838/0.6448403726145834`. Both now retain
  checkpoint 2705 with frozen patience `0/2`.
- Each artifact covers exactly `192` rows and `64` per language, has no
  literal empty raw output or parser failure, and its independently recomputed
  language mean exactly matches trainer state. Seed 13 has `59`
  whitespace-only raw outputs and `39/53/36` unique Tsn/Xho/Zul outputs;
  seed 87 has `55` whitespace-only raw outputs and `39/56/37` unique outputs.
- Seed-13 debug/trainer-state/adapter-config SHA-256 values are
  `ae27fb507e8fe8f8dde361c79da2f2f51746be2d1639d00dd3a7fe2d04359bed`,
  `43f2391ec6bd384484f68d09be93eed5491a8eb58fd87b606d67e86506111cc4`,
  and `3ffa005b2777c6c9e2f887484f764dfd4452d84f187fdbadbd590a8b347308ad`.
  Seed-87 values are
  `e074f236fbad6d1be3da43f6c387e3d27d690bfbcc3add2897d18a2d06fb05c3`,
  `785b0b53a8f6249d7ad375e3cef1e3e0ad1ca4f26ccbefa80fc3c5ebf9c61ba9`,
  and `99132c152764b8965e4be2a757d975dcc4609debf27310db9ed1d5414b895520`.
- Both remain operationally healthy with the verified GDN fast path and no
  targeted fault marker. Seed 13 resumed near `2935/8115`; seed 87 reached
  its next step-3246 validation boundary. The next exact artifacts are
  tentatively due around `07:15--07:45 SAST`.
- These are valid interim artifacts, not terminal confirmations: NER remains
  scientifically `2/4` terminal-valid and no winner is frozen. The two jobs
  are the only owned work, both A100-40GB `gpu:ampere`, with no owned
  A100-80GB/L40S overlap. HEX quota is home `88.6%`, scratch `39.3%`;
  held-out remains `0`, Sheet E/F/G remain blank, Kombuys remains read-only
  at its last verified idle state, and all gates remain unchanged.

## NER a2 step-2705 callbacks continue — 06:12 SAST

- Both jobs reached step `2705/8115` and completed declared validation over
  all `10,760` rows. Job `1245393` recorded provenance-only loss
  `0.3368947479361495`; job `1245394` recorded `0.3409823563019139`.
- Both separate 192-prompt exact callbacks remain active, with fresh automatic
  generation batch-size probes through `06:05` and `06:00` and no targeted
  fault marker. No step-2705 exact artifact exists yet; checkpoint 2164 and
  frozen patience `0/2` therefore remain unchanged for each seed. Observed
  callback timing revises the likely artifact window to
  `06:15--06:40 SAST`.
- Operationally both A100-40GB jobs remain healthy with the verified GDN fast
  path. Scientifically NER remains `2/4` terminal-valid confirmations and no
  winner is frozen. They are the only owned jobs, with no owned
  A100-80GB/L40S overlap. HEX quota is home `88.6%`, scratch `39.3%`;
  held-out remains `0`, Sheet E/F/G remain blank, Kombuys remains read-only at
  its last verified idle state, and all gates remain unchanged.

## Both NER a2 seeds retain step 2164 — 05:42 SAST

- Seed-13 job `1245393` completed its step-2164 exact artifact at mean span
  F1 `0.5956411041394506`; Tsn/Xho/Zul values are
  `0.5865787136876325/0.5823108384457577/0.6180337602849617`. Checkpoint 2164
  is retained with frozen patience `0/2`, matching the seed-87 retained step.
- The artifact covers exactly `192` rows and `64` per language, with no
  literal empty raw output, `54` whitespace-only raw outputs, `40/57/38`
  unique Tsn/Xho/Zul raw outputs, and no parser failure. Its recomputed mean
  exactly matches trainer state. Debug/trainer-state/config SHA-256 values are
  `792e84420538b9ddb6b28df78ec6453d9883799d5beaa5f80985f961646d9caf`,
  `b2e69e5fa638728dfb7f773b499bcdc506799eabb219b2b9b04ff0d31d8b0a10`,
  and `3ffa005b2777c6c9e2f887484f764dfd4452d84f187fdbadbd590a8b347308ad`.
- Seed 13 resumed healthy near `2567/8115`. Seed-87 job `1245394` reached
  step 2705, completed declared validation over all `10,760` rows at
  provenance-only loss `0.3409823563019139`, and entered its exact callback.
  The next exact artifacts are tentatively due around `06:20--06:45 SAST`.
- These remain valid interim artifacts, not terminal confirmations: NER stays
  `2/4` terminal-valid and no winner is frozen. Both A100-40GB jobs retain the
  verified GDN fast path with no targeted fault marker and are the only owned
  work, with no owned A100-80GB/L40S overlap. HEX quota is home `88.6%`,
  scratch `39.3%`; held-out remains `0`, Sheet E/F/G remain blank, Kombuys
  remains read-only at its last verified idle state, and all gates remain
  unchanged.

## NER a2 seed 87 improves at step 2164 — 05:12 SAST

- Seed-87 job `1245394` completed its step-2164 exact artifact at mean span
  F1 `0.5960721696811287`; Tsn/Xho/Zul values are
  `0.5945072697899338/0.5718134715025407/0.6218957677509119`. Checkpoint 2164
  is retained with frozen patience `0/2`.
- The artifact covers exactly `192` rows and `64` per language, with no
  literal empty raw output, `55` whitespace-only raw outputs, `40/57/37`
  unique Tsn/Xho/Zul raw outputs, and no parser failure. Its recomputed mean
  exactly matches trainer state. Debug/trainer-state/config SHA-256 values are
  `f4c12ee4479593cd0edfa6f9af81f70cad370140e0e705a3a44bee4816f25d62`,
  `5b8f7fc2e1f070009b277709f7f301cb328b95c5bbcca345e4fcca72b031d9d3`,
  and `99132c152764b8965e4be2a757d975dcc4609debf27310db9ed1d5414b895520`.
- Seed 87 resumed healthy near `2341/8115`. Seed-13 job `1245393` remains
  healthy inside its matching step-2164 callback after complete declared
  validation at provenance-only loss `0.3380058487108649`; fresh generation
  probes continued through `05:07`, and its artifact is expected around
  `05:15--05:30 SAST`.
- These remain valid interim artifacts, not terminal confirmations: NER stays
  `2/4` terminal-valid and no winner is frozen. Both A100-40GB jobs retain the
  verified GDN fast path with no targeted fault marker and are the only owned
  work, with no owned A100-80GB/L40S overlap. HEX quota is home `88.6%`,
  scratch `39.3%`; held-out remains `0`, Sheet E/F/G remain blank, Kombuys
  remains read-only at its last verified idle state, and all gates remain
  unchanged.

## Both NER a2 seeds retain step 1623 — 04:31 SAST

- Seed-13 job `1245393` completed its step-1623 exact artifact at mean span
  F1 `0.569087134514023`; Tsn/Xho/Zul values are
  `0.5734701831882052/0.5484253666953773/0.5853658536584866`. Checkpoint 1623
  is retained with frozen patience `0/2`, matching the seed-87 retained step.
- The seed-13 artifact covers exactly `192` rows and `64` per language, with
  no literal empty raw output, `56` whitespace-only raw outputs, `40/55/36`
  unique Tsn/Xho/Zul raw outputs, and no parser failure. Its recomputed mean
  exactly matches trainer state. Debug/trainer-state/config SHA-256 values are
  `76a68338d87a62949e0e024ca073ebcf6abd97e7fdb4670366d2e5be11e8fd0b`,
  `9f2b9072773de4760f7c234638f77c8904b32b0ca62260f7c544593d3d929388`,
  and `3ffa005b2777c6c9e2f887484f764dfd4452d84f187fdbadbd590a8b347308ad`.
- Seed 13 resumed healthy near `2081/8115`. Seed-87 job `1245394` reached
  step 2164, completed declared validation over all `10,760` rows at
  provenance-only loss `0.33073989697991696`, and entered its exact callback.
  The next exact artifacts are tentatively due around `05:10--05:35 SAST`.
- These are valid interim artifacts, not terminal confirmations: NER remains
  `2/4` terminal-valid and no winner is frozen. Both A100-40GB jobs retain the
  verified GDN fast path with no targeted fault marker and are the only owned
  work, with no owned A100-80GB/L40S overlap. HEX quota is home `88.6%`,
  scratch `39.3%`; held-out remains `0`, Sheet E/F/G remain blank, Kombuys
  remains read-only at its last verified idle state, and all gates remain
  unchanged.

## NER a2 seed 87 improves at step 1623 — 04:01 SAST

- Seed-87 job `1245394` completed its step-1623 exact artifact at mean span
  F1 `0.5725123790157849`; Tsn/Xho/Zul values are
  `0.5664436573526983/0.5478807765927861/0.6032127031018703`. Checkpoint 1623
  is now retained with frozen patience `0/2`.
- The artifact covers exactly `192` rows and `64` per language, with no
  literal empty raw output, `50` whitespace-only raw outputs, `42/57/37`
  unique Tsn/Xho/Zul raw outputs, and no parser failure. The recomputed
  language mean exactly matches trainer state. Debug/trainer-state/config
  SHA-256 values are
  `559de3d5bd0dc5c8a04498a7e180b0e32272db8d998bbdc741e590400bccdcab`,
  `0f9eb2b0a769c486edd5dfac090c450968e537c1e624f58e10072b7da893d123`,
  and `99132c152764b8965e4be2a757d975dcc4609debf27310db9ed1d5414b895520`.
- Seed 87 resumed healthy near `1829/8115`. Seed-13 job `1245393` remained
  healthy inside its matching step-1623 exact callback; its artifact was not
  yet present and is expected around `04:05--04:20 SAST`. Both retain the
  verified GDN fast path with no targeted fault marker.
- These remain valid interim artifacts, not terminal confirmations: NER stays
  `2/4` terminal-valid and no winner is frozen. The two A100-40GB jobs are the
  only owned work, with no owned A100-80GB/L40S overlap. HEX quota is home
  `88.6%`, scratch `39.3%`; held-out remains `0`, Sheet E/F/G remain blank,
  Kombuys remains read-only at its last verified idle state, and all gates
  remain unchanged.

## NER a2 step-1623 callbacks active — 03:49 SAST

- Both jobs reached step `1623/8115` and completed declared validation over
  all `10,760` rows. Job `1245393` recorded provenance-only validation loss
  `0.34193392048094795`; job `1245394` recorded `0.34355421668977987`.
  Selection remains based only on the separate exact span-F1 artifacts.
- Both exact 192-prompt callbacks remain active, with fresh automatic
  generation batch-size probes through `03:40` and `03:37` and no targeted
  fault marker. No step-1623 artifact exists yet; observed callback timing
  puts the likely completion window around `03:50--04:15 SAST`.
- The first SSH monitoring attempt reset before returning state. The immediate
  quota-first read-only retry succeeded; no job action was taken. Both jobs
  remain healthy on A100-40GB with the verified GDN fast path. Scientifically
  NER remains `2/4` terminal-valid confirmations.
- They remain the only owned jobs, with no owned A100-80GB/L40S overlap. HEX
  quota is home `88.6%`, scratch `39.3%`; held-out remains `0`, Sheet E/F/G
  remain blank, Kombuys remains read-only at its last verified idle state,
  and all gates remain unchanged.

## NER a2 improves at step 1082 — 02:58 SAST

- Seed-13 job `1245393` improved at step 1082 to mean span F1
  `0.4957424333994835`; Tsn/Xho/Zul values are
  `0.48288476613530407/0.4777564813880263/0.5265860526751202`. Seed-87 job
  `1245394` improved to `0.529091241481777`, with
  `0.5227586206896052/0.5089807162533941/0.5555343875023318`. Both now retain
  checkpoint 1082 with frozen patience `0/2`.
- Both exact artifacts cover `192` rows and `64` per language. Seed 13 has no
  literal empty raw output, `58` whitespace-only raw outputs, `38/55/37`
  unique Tsn/Xho/Zul raw outputs, and no parser failure. Seed 87 has no
  literal empty raw output, `50` whitespace-only raw outputs, `42/57/37`
  unique raw outputs, and no parser failure. Recomputed language means match
  the trainer-state metrics exactly.
- Seed-13 debug/trainer-state/adapter-config SHA-256 values are
  `aabe0cc3394c455fb4ed27857aeea00b90f1653b70b7be0a8d376e859e3510dc`,
  `9f013e68ce4db9953d9fb06cba156750394c2336646235467fa1ba288c5634d0`,
  and `3ffa005b2777c6c9e2f887484f764dfd4452d84f187fdbadbd590a8b347308ad`.
  Seed-87 values are
  `b2273b799e8e563d9f0a88845748d6d5c83d7536bd2c1dcc36c0c5b0e7833819`,
  `1e1fb929d4672bf22710b617aab4db01a3941b3016734c0075cc2bf70afbfc25`,
  and `99132c152764b8965e4be2a757d975dcc4609debf27310db9ed1d5414b895520`.
- Both jobs resumed healthy near `1125/8115` and `1453/8115` with the
  verified GDN fast path and no fault marker. Their next exact artifacts are
  tentatively due around `03:45--04:25 SAST`; terminal timing remains governed
  by frozen early stopping. These are valid interim artifacts, not terminal
  confirmations: NER remains `2/4` terminal-valid and no winner is frozen.
- The two A100-40GB jobs are the only owned work, with no owned
  A100-80GB/L40S overlap. HEX quota is home `88.6%`, scratch `39.3%`.
  Held-out remains `0`, Sheet E/F/G remain blank, Kombuys remains read-only at
  its last verified idle state, and all global-freeze, quarantine, and
  publication gates remain unchanged.

## NER a2 second exact callbacks active — 02:28 SAST

- Both confirmations reached step `1082/8115` and completed their second
  declared validation over all `10,760` rows. Job `1245393` has
  provenance-only validation loss `0.3635498784288598`; job `1245394` has
  `0.3648764287672078`. These losses do not participate in frozen NER
  selection.
- Both are now inside the separate 192-prompt exact generation scorer. Fresh
  automatic batch-size probes continued through `02:26` and `02:25`, with no
  targeted fault marker. No step-1082 exact artifact exists yet; observed
  callback timing puts the likely completion window around
  `02:40--03:00 SAST`.
- Operationally both jobs are healthy on A100-40GB with the verified GDN fast
  path. Scientifically NER remains `2/4` terminal-valid confirmations until
  the runs terminate and all retained artifacts reconcile. These remain the
  only owned jobs, with no owned A100-80GB/L40S overlap. HEX quota is home
  `88.6%`, scratch `39.3%`; held-out remains `0`, Sheet E/F/G remain blank,
  Kombuys remains read-only at its last verified idle state, and all gates
  remain unchanged.

## NER a2 first exact artifacts reconcile — 01:58 SAST

- Seed-13 job `1245393` completed its step-541 exact artifact at mean span F1
  `0.2972696517612993`; Tsn/Xho/Zul values are
  `0.27350049707274504/0.2900679456434352/0.32824051256771775`. Seed-87 job
  `1245394` completed its step-541 artifact at mean
  `0.36267885666568933`, with
  `0.36175455889596053/0.32256819351513666/0.4037138175859708`.
- Both artifacts cover exactly `192` rows and `64` per language. Seed 13 has
  no literal empty raw output, `64` whitespace-only raw outputs, `39/53/34`
  unique Tsn/Xho/Zul raw outputs, and no parser failure. Seed 87 has no
  literal empty raw output, `61` whitespace-only raw outputs, `41/54/34`
  unique raw outputs, and no parser failure. Recomputed language means exactly
  match each retained `trainer_state.json` best metric.
- Seed-13 debug/trainer-state/adapter-config SHA-256 values are
  `a045389ad4f1143d2ba672b7c51aa3bb82de17ddba77cef93d69c4afc2df53f0`,
  `259bc90c580214550c43930d6fea3e784aa2197092e6173b6a66f5be3bcb6c98`,
  and `3ffa005b2777c6c9e2f887484f764dfd4452d84f187fdbadbd590a8b347308ad`.
  Seed-87 values are
  `bb53213ddb0fab8d10f0913af59a5d52d2719e78a0b07393c503c0bc5918a79f`,
  `b0563a6a3949bc3516c03a928b0c3681f3f7701d1d5e92912183caa7693cd94e`,
  and `99132c152764b8965e4be2a757d975dcc4609debf27310db9ed1d5414b895520`.
- Both jobs resumed healthy with the verified GatedDeltaNet fast path, near
  `877/8115` and `1082/8115`; seed 87 is entering its next validation
  boundary. The next exact artifacts are tentatively due around
  `02:25--03:00 SAST`, but terminal timing remains governed by frozen early
  stopping. These are accepted interim validation artifacts, not terminal
  confirmations: NER remains `2/4` terminal-valid and no family winner is
  frozen.
- They are the only owned jobs, both on A100-40GB `gpu:ampere`, with no owned
  A100-80GB/L40S overlap. HEX quota is home `88.6%`, scratch `39.3%`.
  Held-out remains `0`, Sheet E/F/G remain blank, Kombuys remains read-only at
  its last verified idle state, and all global-freeze, quarantine, and
  publication gates remain unchanged.

## NER a2 first exact callbacks active — 01:22 SAST

- Jobs `1245393` (seed 13) and `1245394` (seed 87) both reached step
  `541/8115` and completed declared validation over exactly `10,760` rows.
  Validation losses are `0.491191163470754` and `0.5266696859026487`.
  These are health/provenance values only; the frozen NER selector remains
  mean exact span F1 from the separate validation-only generation artifact.
- Both jobs are inside that exact 192-prompt generation callback. Their logs
  show fresh automatic batch-size probes, the verified GatedDeltaNet fast
  path, and zero targeted fault markers. No complete step-541 exact artifact
  exists yet; prior callback timing revises the likely completion window to
  about `02:05--02:25 SAST`.
- Operationally both jobs are healthy, but scientifically NER remains `2/4`
  terminal-valid confirmations. They are the only owned jobs, both on
  A100-40GB `gpu:ampere`, with no owned A100-80GB/L40S overlap. HEX quota is
  home `88.6%`, scratch `39.2%`; held-out remains `0`, Sheet E/F/G remain
  blank, Kombuys remains read-only at its last verified idle state, and all
  global-freeze, quarantine, and publication gates remain unchanged.

## NER a2 confirmations approach first validation — 00:52 SAST

- Jobs `1245393` (seed 13) and `1245394` (seed 87) remain healthy and
  fault-free on separate `srvrocgpu010` A100-40GB devices at `511/8115` and
  `502/8115` training steps. Both retain the verified GatedDeltaNet fast path
  and are approaching their first step-541 validation boundary.
- First complete exact validation artifacts are estimated around
  `01:45--02:15 SAST`. This is operational progress only: NER remains `2/4`
  terminal-valid confirmations until both jobs complete and their exact
  artifacts and hashes reconcile.
- These are the only owned jobs; no A100-80GB/L40S work overlaps them. HEX
  quota is home `88.6%`, scratch `39.2%`. Held-out remains `0`, Sheet E/F/G
  remain blank, Kombuys remains read-only at its last verified idle state,
  and all quarantine, global-freeze, and publication gates remain unchanged.

## POS closes; remaining NER confirmations start — 00:26 SAST

- POS a1 seed-13 job `1241720` completed `0:0` at `00:08:59 SAST` after
  `17:59:32`. Its terminal step-1981 exact artifact covers all `1,800` rows,
  `12` language/prompt cells, and `17` labels and scores
  `0.8427841065626649`. SHA-256
  `160ced2753810e8980abda4d54c804c15396103bed7d02d1c4c61741eca518f3`
  passes `sha256sum -c`; targeted fault scans are empty and the final adapter
  was saved.
- Checkpoint 1415 remains the seed-13 retained winner at
  `0.8452102586327794`. Combined with the fixed seed-42 score
  `0.8474682174611892`, a1's terminal two-seed arithmetic mean is
  `0.8463392380469843`. The preregistered POS close-out therefore selects a2,
  whose terminal two-seed mean is `0.8606543221600175`. This is a
  validation-only family selection; no held-out data were touched and the
  global eight-family freeze remains blocked by unfinished grids.
- After verifying zero owned jobs and absent a2 confirmation outputs, the two
  remaining frozen NER confirmations were submitted as a2 seed-13 job
  `1245393` and a2 seed-87 job `1245394`. Both started at `00:25:28 SAST` on
  separate `srvrocgpu010` A100-40GB devices with the required
  `nlpgroup/a100/nlpgroup`, `gpu:ampere:1`, 24-hour, eight-CPU, and home
  working-directory settings.
- Both NER jobs verified all `694` files from immutable snapshot
  `/home/lmbanr001/masters/sallm_snapshots/uniform-adapter-hpo-20260811-6aabf717`
  and passed the GatedDeltaNet fast-kernel gate. Execution-manifest SHA-256
  values are
  `9c18f338c4c41402217c8669c3e078c4be303d65f66d498b9efef07f952af602`
  for seed 13 and
  `c12d0921153bec2da784e05752d140ca261e94a1ffcc1e9cda76583842017fc8`
  for seed 87. They are the only owned jobs; no A100-80GB/L40S work overlaps
  them.
- HEX quota is home `88.6%`, scratch `39.2%`. Held-out remains `0`, Sheet
  E/F/G remain blank, Kombuys remains read-only at its last verified idle
  state, and quarantine and Hugging Face publication blocks remain.
