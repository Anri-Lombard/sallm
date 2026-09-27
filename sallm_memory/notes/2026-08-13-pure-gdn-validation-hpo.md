# Pure-GDN corrected validation-only HPO — 2026-08-13

## NER b7 reaches step-5951 callback; POS remains queued — 23:30 SAST

- NER b7 job `1230086` remains healthy on `srvrocgpu010` A100-40GB and has
  entered its step-5951 validation callback after `12:58:25` elapsed. Its log
  was fresh at `23:28:09 SAST`, with no traceback, OOM, CUDA, or other fault
  marker; no new complete 192-row artifact exists yet.
- POS a0/a1 jobs `1231553/1232086` remain pending for
  `(Resources)/(Priority)`. HEX has one running plus two pending owned
  A100-40GB jobs, quota home `52.0%` and scratch `38.2%`, and no owned
  A100-80GB or L40S work. Kombuys assigned GPU 1 is idle, root/scratch free
  are `23 GB/2.1 TB`, and foreign GPU 0 remains occupied and untouched.
- Trusted progress remains base `16/16`, NER `10/11`, T2X `11/11`, T2X
  confirmations `4/4`, POS `0/11`, global frozen winners `0/8`, held-out
  adapter evaluations `0`, and Mono not started. No held-out metric was
  accessed; Sheet E/F/G remain blank and Hugging Face publication remains
  blocked.

## NER b6 terminal-valid; b7 improves; POS a1 queues — 23:00 SAST

- NER b6 job `1227987` completed `0:0` at `22:46:02 SAST` after
  `18:45:06`, raising trusted NER grid progress to `10/11`. Its final step
  8115 mean validation F1 improved to `0.4347279479413982`, so checkpoint
  8115 is retained. Tsn/Xho/Zul F1 values are
  `0.44740209879698106/0.4081883946436317/0.4485933503835819`. The exact
  192-row, 64/language artifact has no literal empties or parse failures,
  `55` whitespace-only outputs, and `40/57/36` unique outputs. Final-debug,
  retained-state, retained-adapter, final-adapter, and config SHA-256 values
  are
  `7255dd470df17d220b6ab7709d16104b794f8cda96204e2d52cc19566761aaa0`,
  `4031ec6b4e40b73274ef18df908157757651132a0be81e42c2cbc5e8a1de2388`,
  `97f2d110bbfb5f461093075de0af68b299e40833642c09dfc1de58981c61239e`,
  `730520fec7d337a34fb7b5cfb287cc412785b1ac8454304011b42b9879c71dd6`,
  and `6fdebf186ba08ef151845ad1d542b78d8fb4d358c56d6cdfc1076dc77d8eb0a0`.
- NER b7 job `1230086` improved at epoch 10 step 5410 to exact mean
  validation F1 `0.6834830425594811`, retaining checkpoint 5410 and resetting
  patience. Tsn/Xho/Zul F1 values are
  `0.662327518875243/0.6900184356070084/0.6981031731961919`. Its exact
  192-row, 64/language artifact has no literal empties or parse failures,
  `55` whitespace-only outputs, and `40/56/38` unique outputs. Debug/state
  hashes are
  `e8acb836aec9c01cef71c43d2f6bf0888e2935a9501271f992728ae71b867ae1`/
  `667a54a701f2f029bd979f194fe9a52e0f800a36ba8c364c9da099540ed05b97`.
  B7 remains healthy on `srvrocgpu010` A100-40GB, elapsed `12:29:38`.
- After verifying the frozen registry hash
  `8fdd6ea5f192a1d2ef2267353162b9a640ccd09ae0d889cb5a999645198bb726`,
  absent a1 target, two owned jobs, and no A100-80GB/L40S work, pure-GDN POS
  Stage-A a1 job `1232086` was submitted from the immutable snapshot. Its
  settings verify as `nlpgroup/a100/nlpgroup`, one `gpu:ampere`, 24 hours,
  eight CPUs, and the required working directory. It is pending for
  `(Priority)`; POS a0 job `1231553` remains pending for `(Resources)`.
- HEX now has one running plus two pending owned A100-40GB jobs, quota home
  `52.0%` and scratch `38.2%`, and no owned A100-80GB or L40S work. Kombuys
  assigned GPU 1 remains idle and foreign GPU 0 remains untouched. Trusted
  progress is base `16/16`, NER seed-42 `10/11`, T2X seed-42 `11/11`, T2X
  confirmations `4/4`, POS enhanced grid `0/11`, global frozen winners
  `0/8`, held-out adapter evaluations `0`, and Mono not started. No held-out
  metric was accessed; Sheet E/F/G remain blank and Hugging Face publication
  remains blocked.

## NER b6 enters its terminal callback — 22:30 SAST

- NER b6 job `1227987` completed all `8115/8115` training steps and entered
  its final scheduled generation callback. NER b7 job `1230086` remains
  inside its step-5410 callback. Both remain healthy on `srvrocgpu010`
  A100-40GB, elapsed `18:28:32/11:58:26`, with no fault markers. Conditional
  complete-artifact/terminal evidence is expected around `22:35--22:50
  SAST`. POS a0 job `1231553` remains pending for `(Priority)`.
- HEX quota is home `52.0%`, scratch `38.2%`; there are two running plus one
  pending owned job and no A100-80GB or L40S work. Kombuys assigned GPU 1 RTX
  3080 Ti is idle, root/scratch free are `24 GB/2.1 TB`, and foreign GPU 0 RTX
  5090 remains occupied and untouched.
- Trusted progress remains base `16/16`, NER seed-42 `9/11`, T2X seed-42
  `11/11`, T2X confirmations `4/4`, POS enhanced grid `0/11`, global frozen
  winners `0/8`, held-out adapter evaluations `0`, and Mono not started. No
  held-out metric was accessed; Sheet E/F/G remain blank and Hugging Face
  publication remains blocked.

## T2X Stage C completes and selects b7; NER b6 reconciles — 22:00 SAST

- Kombuys T2X a2 seed 87 completed all `1932/1932` frozen steps in
  `1:18:34` and is terminal-valid. Its final validation chrF is
  `50.74381008283938`, retaining checkpoint 1932. The exact final 64-row
  artifact has no literal or whitespace-only empty raw predictions and `63`
  unique outputs. Final debug, retained-state, retained-adapter,
  final-adapter, and adapter-config SHA-256 values are
  `4f99237cb3388b3f7c87e1d8d9d55b1f686107b796e08ac2b432fbc429f5594a`,
  `8d4d82bb53be7886c4ff8fff04b636261a4b1564e87fc58493c26aba967dab56`,
  `5e1d051e520b7685bea64365ac28e98cafdfc42a0c7ca6bbbb690ff1eecc8722`,
  `80f2835ac3997651c7a10fd07c79fa34333bc785c0f83d75cd2a89731251bbd4`,
  and `c06c81ffb747cd4479f6bb8cf7c24d128afb6552ce3ac04632a752c421870ec9`.
  All `424/424` adapter tensors, totaling `71,762,560` values, exactly match
  between retained checkpoint 1932 and final serialization. Assigned GPU 1
  is idle; foreign GPU 0 RTX 5090 remains occupied and untouched. Kombuys
  root/scratch free are `23 GB/2.1 TB`.
- All four added T2X confirmation runs are now terminal-valid. The immutable
  HPO utility applied the preregistered arithmetic mean over seeds
  `13/42/87`. B7 scores
  `52.26645731208185/51.975103592985164/51.72122400804512`, mean
  `51.987594971037375`, sample SD `0.272831202123097`; a2 scores
  `50.22442959480154/51.306312454497196/50.74381008283938`, mean
  `50.75818404404604`, sample SD `0.5410846408801392`. T2X therefore selects
  b7, whose seed-42 retained checkpoint is the eventual production adapter.
  The reproducible validation-only ranking artifact is
  `sallm_memory/artifacts/2026-08-13-pure-gdn-t2x-confirmation-ranking.json`,
  SHA-256
  `86f7655d356aecf490f18fa30624faab2f79b5db91f29b8272f66a0545a20e36`.
  This is a family-local selection only; the eight-family global freeze gate
  remains closed and no held-out evaluation is authorized.
- HEX NER b6 job `1227987` declined at epoch 14 step 7574 to exact mean
  validation F1 `0.4345354420684952`, narrowly below retained checkpoint
  7033 best `0.43460119481609377`; this is its first patience miss after the
  improvement. Tsn/Xho/Zul F1 values are
  `0.4454649827783657/0.4098125576154371/0.44832878581168273`. Its exact
  192-row, 64/language artifact has no literal empties or parse failures,
  `55` whitespace-only outputs, and `40/57/36` unique outputs. Debug/state
  hashes are
  `64edd1798b95616020f61e6c5c3e9a6b37f97fb7e5ba973930d053554cb2c634`/
  `a837aa5bca7975d54c35cce08e267ce7b647c8dc1142c5b899323fa8d203aead`.
- NER b6/b7 jobs `1227987/1230086` remain healthy A100-40GB runs on
  `srvrocgpu010`, elapsed `17:58:32/11:28:26`; b7 is inside its step-5410
  callback with no new complete artifact yet. POS a0 job `1231553` remains
  pending for `(Priority)`. HEX quota is home `52.0%`, scratch `38.2%`; there
  are two running plus one pending owned job and no A100-80GB or L40S work.
- Trusted progress is base `16/16`, NER seed-42 `9/11`, T2X seed-42 `11/11`,
  T2X confirmations `4/4`, POS enhanced grid `0/11`, global frozen winners
  `0/8`, held-out adapter evaluations `0`, and Mono not started. No held-out
  metric was accessed; Sheet E/F/G remain blank and Hugging Face publication
  remains blocked.

## T2X a2 seed 87 reaches 50.21 chrF; NER b7 improves — 21:30 SAST

- Kombuys T2X a2 seed 87 improved from validation chrF
  `44.40289293605772` at step 483 through `47.67471900310504` at step 966
  to `50.21260712607607` at step 1449, retaining checkpoint 1449. The exact
  step-1449 64-row artifact has no literal or whitespace-only empty raw
  predictions and `63` unique outputs. Its debug/state SHA-256 values are
  `3a7ae03cad0761a3a8d347fa865be8fe260a99b070bf5207c5fa7ad1a08f134c`/
  `27aa6296a5eb8827ce695378df13a55c2281764260fdb0ce82b9174f65011381`.
  The run resumed after the callback in tmux `sallm-t2x-confirm-a2-s87` on
  assigned GPU 1 RTX 3080 Ti; conditional terminal ETA remains about `21:50
  SAST`. Foreign GPU 0 RTX 5090 remained occupied and untouched. Kombuys
  root/scratch free were `22 GB/2.1 TB`. This remains within-candidate
  validation evidence; no winner is frozen before terminal reconciliation.
- HEX NER b7 job `1230086` improved at epoch 9 step 4869 to exact mean
  validation F1 `0.6738201916539253`, retaining checkpoint 4869 and resetting
  patience. Tsn/Xho/Zul F1 values are
  `0.6456692913385328/0.6802415691304083/0.695549714492835`. Its exact
  192-row, 64/language artifact has no literal empties or parse failures,
  `56` whitespace-only outputs, and `39/56/38` unique outputs. Debug/state
  SHA-256 values are
  `90b15f2c30ca7c327b64176666ac493f7885f9a54d81c7544bc438f6ba9339a2`/
  `215acaf924966849250d9916dfb0e40203ed64c8b08d1321bb1597a8d6110685`.
  B7 resumed after its callback.
- NER b6 job `1227987` remains healthy inside its step-7574 callback; no new
  complete artifact exists yet. B6/b7 elapsed `17:28:34/10:58:28` on
  `srvrocgpu010` A100-40GB. Pure-GDN POS a0 job `1231553` remains pending for
  `(Priority)`. HEX quota is home `52.0%`, scratch `38.2%`; there are two
  running plus one pending owned job and no owned A100-80GB or L40S work.
- Trusted terminal progress remains base `16/16`, NER seed-42 `9/11`, T2X
  seed-42 `11/11`, T2X confirmations `3/4`, POS enhanced grid `0/11`, frozen
  winners `0/8`, held-out adapter evaluations `0`, and Mono not started. No
  held-out metric was accessed; Sheet E/F/G remain blank and Hugging Face
  publication remains blocked.

## Final T2X confirmation produces a clean first artifact — 21:00 SAST

- Kombuys T2X a2 seed 87 produced a valid epoch-1 step-483 artifact with
  validation chrF `44.40289293605772`, retaining checkpoint 483. The exact
  64-row artifact has no literal or whitespace-only empty raw predictions and
  `60` unique normalized outputs. Debug/state SHA-256 values are
  `ac96581d38e36e75402f7ca289dcc185e6af84d0a810d3e3f224af69d3103df3`/
  `c4fa6debf39f74f6a94c787eb6ef893bc193fff674a57b9faeb60d551c3753f0`.
  The run resumed and was healthy near step `741/1932` in tmux
  `sallm-t2x-confirm-a2-s87` on assigned GPU 1 RTX 3080 Ti; conditional
  terminal ETA remains about `21:50 SAST`. Foreign GPU 0 RTX 5090 remained
  occupied and untouched. Kombuys root/scratch free were `23 GB/2.1 TB`.
  This is validation-only confirmation evidence; no winner is frozen.
- HEX NER b6/b7 jobs `1227987/1230086` remain healthy A100-40GB runs on
  `srvrocgpu010`, elapsed `16:58:42/10:28:36`. They entered their step
  `7574/4869` scheduled callbacks, but neither callback has produced a new
  complete 192-row artifact yet. Pure-GDN POS Stage-A a0 job `1231553`
  remains pending for `(Priority)`. HEX quota is home `52.0%`, scratch
  `38.2%`; there are two running plus one pending owned job and no owned
  A100-80GB or L40S work.
- Scientifically trusted terminal progress remains base `16/16`, NER seed-42
  `9/11`, T2X seed-42 `11/11`, T2X confirmations `3/4`, POS enhanced grid
  `0/11`, frozen winners `0/8`, held-out adapter evaluations `0`, and Mono
  not started. No held-out metric was accessed; Sheet E/F/G remain blank and
  Hugging Face publication remains blocked.

## T2X final confirmation starts; NER b5 closes; POS opens — 20:38 SAST

- Kombuys T2X a2 seed 13 completed all `1932/1932` frozen steps in about
  `1:17:59` and is terminal-valid. Its final validation chrF is
  `50.22442959480154`, retaining checkpoint 1932. The exact 64-row final
  artifact has no literal or whitespace-only empty raw predictions and `62`
  unique outputs. Final debug, retained-state, retained-adapter,
  final-adapter, and adapter-config SHA-256 values are
  `520172d14bdfb76d15108c98d48026e8ae1052e989b80216cffb47e440589420`,
  `440672ab64092f57115032cc82bdde6ee1ec138d647322a819a59ea8af97a64c`,
  `444fb695cb7dfef2a40740c589b020b6bb1d872811db65ed003ec19fa544e8c3`,
  `d85fb21a520e85b26a7ea388790fd0acab1850277eac96e2f04b2bb182ddc133`,
  and `2a439ac30dddba38c6a1ed70b2c1b26fb5f2f93d43ca366ffa1144e9b2cc6960`.
  All `424/424` adapter tensors, totaling `71,762,560` values, exactly match
  between retained checkpoint 1932 and the final adapter serialization.
- After verifying assigned GPU 1 idle, absent output/log/session, frozen
  ranking hash
  `cd532deceac60acef5490a908a1ab9d7fd6522e6b1caf61f30cd3c9af40a2cb7`,
  registry hash
  `8fdd6ea5f192a1d2ef2267353162b9a640ccd09ae0d889cb5a999645198bb726`,
  and frozen pair `b7,a2`, the last T2X confirmation, a2 seed 87, started in
  tmux `sallm-t2x-confirm-a2-s87`. It verified 694 immutable files, exact
  `3,859/460` train/validation rows, and the pure-GDN fast path before its
  1,932-step run on assigned GPU 1 RTX 3080 Ti. Manifest/trial hashes are
  `93e3b4564e4a3768ca4d515ada34ad0ca0600025fdb987118a45b39810fe9159`/
  `4ea85cd4bc26d0adaf01ee306591081c7a9c46b3fc77a162b858e3b7f2ee1c0a`;
  conditional first-artifact/terminal ETAs are about `20:50--21:00` and
  `21:50 SAST`. Foreign GPU 0 RTX 5090 remains occupied and untouched;
  Kombuys root/scratch free were `24 GB/2.1 TB` at launch.
- HEX NER b5 job `1226719` completed `0:0` at `20:06:37 SAST` after
  `18:28:47`, raising trusted NER grid progress to `9/11`. Its final step
  8115 mean validation F1 is `0.5736957518558778`, below retained checkpoint
  7574 best `0.5743710116912446`; Tsn/Xho/Zul final F1 values are
  `0.5655716162942996/0.5679966133981935/0.5875190258751403`. The exact
  192-row, 64/language final artifact has no literal empties or parse
  failures, `60` whitespace-only outputs, and `39/53/36` unique outputs.
  Final-debug, retained-state, retained-adapter, final-adapter, and config
  hashes are
  `7398cecd60ffa3052eadc3c5ce1179ae88ac63fcf744cdb9a6fb4b4828b22656`,
  `c1697ee1e72d35ae065dc962f941906e9341eb3b505a1aa93abafd4f13d6c1b9`,
  `c9ded6d4b8c7af24f5541253e62a9f85b231b4cf52d70439b15292eb4ad6b64a`,
  `c6baf265120b29c4f647d9f37080fca9afe7f7ef9291e040c91074a9925baa9e`,
  and `1726d985400a83dbab377f1d1d6884c4e98ca20bf07735232bb29a91f39c9cb9`.
- NER b6/b7 jobs `1227987/1230086` remain healthy A100-40GB runs on
  `srvrocgpu010`, elapsed `16:33:33/10:03:27` at the audit. B6 improved at
  step 7033 to mean validation F1 `0.43460119481609377`, retaining checkpoint
  7033; b7 declined at step 4328 to `0.6611371591688714` and retains
  checkpoint 3787 best `0.6708899559501509` with its first patience miss.
  Their exact debug/state hashes are
  `fe5e52f2211842d0a4b7fd8f61e13e21906da8b93fbcb94c481bf8581a0814c1`/
  `0d6def1819d5cebc2625dbd8462673e64a4412ed91d9ef49624ebf310866575c`
  and
  `6b498fd260e068992883bb585ddefd58f0a24420115f425b3242315e3b2a330f`/
  `a267169bd15f35fad9c0795f4a41d5bc72c6463f4640f000b0249316510c268e`.
- The freed HEX slot was used for the preregistered pure-GDN POS Stage-A a0
  validation-only lane. Job `1231553` targets
  `adapter_hpo_v3/pure_gdn/pos/stage_a/a0/seed_42` with the required
  `nlpgroup/a100/nlpgroup`, `gpu:ampere:1`, 24-hour, 8-CPU settings and exact
  immutable snapshot launcher. It is pending for `(Priority)`; queue-dependent
  start ETA is unknown. This is pure GDN only; extending this HPO to the other
  architectures remains explicitly deferred.
- HEX has two running plus one pending owned job, quota home `52.0%` and
  scratch `38.2%`, and no owned A100-80GB or L40S work. Trusted progress is
  base `16/16`, NER seed-42 `9/11`, T2X seed-42 `11/11`, T2X confirmations
  `3/4`, POS enhanced grid `0/11` terminal, frozen winners `0/8`, held-out
  adapter evaluations `0`, and Mono not started. No held-out metric was
  accessed; Sheet E/F/G remain blank and Hugging Face publication remains
  blocked.

## T2X a2 seed 13 improves to 47.24 chrF — 20:00 SAST

- Kombuys a2 seed-13 confirmation improved from validation chrF
  `44.201928271802885` at step 483 to `47.23730282734459` at step 966,
  retaining checkpoint 966. The exact 64-row artifact has no literal or
  whitespace-only empty raw predictions and `63` unique outputs. Debug/state
  SHA-256 values are
  `e7b28077908e8008f6bb6109441a92e8422caee22a8283f80c83e80d06a38caa`/
  `ef49422e610f47440ba3cce4d96c6f5105a5c33dc01e131eafdf56e763254ccb`.
  The run reached its step-1449 scheduled callback and remains healthy in
  tmux `sallm-t2x-confirm-a2-s13` on assigned GPU 1 RTX 3080 Ti; conditional
  terminal ETA remains about `20:25 SAST`. Foreign GPU 0 RTX 5090 remains
  occupied and untouched. Kombuys root/scratch free are `22 GB/2.1 TB`.
  This remains validation-only confirmation evidence; no winner is frozen.
- HEX NER jobs `1226719/1227987/1230086` remain exactly three healthy
  A100-40GB jobs on `srvrocgpu010`, elapsed
  `18:21:43/15:58:37/09:28:31`. B5 is still in its final step-8115 callback;
  b6 and b7 are in their step-7033 and step-4328 callbacks. All three logs
  remain fresh without fault markers, but none has produced a newer complete
  192-row artifact since the 19:30 reconciliation. Conditional callback
  completion remains around `20:05--20:40 SAST`. HEX quota is home `52.0%`,
  scratch `38.2%`; no owned A100-80GB or L40S work exists.
- Scientifically trusted terminal progress remains base `16/16`, NER seed-42
  `8/11`, T2X seed-42 `11/11`, T2X confirmations `2/4`, frozen winners
  `0/8`, held-out adapter evaluations `0`, and Mono not started. No held-out
  metric was accessed; Sheet E/F/G remain blank and Hugging Face publication
  remains blocked.

## T2X a2 seed 13 valid first artifact; NER b7 improves — 19:30 SAST

- Kombuys a2 seed-13 confirmation produced a valid epoch-1 step-483
  validation chrF `44.201928271802885`, retaining checkpoint 483. Its exact
  64-row artifact has no literal or whitespace-only empty raw predictions and
  `61` unique outputs; debug/state SHA-256 values are
  `f5f03ae27bea6ee3a1ebf2b8ce5e8bd6369e01b41dbb8978a057df5ae40cea32`/
  `d65ca8da896f7df369fc1c966403d796ef3aa50416c4ae6fd444db76f71683a4`.
  The run resumed and was healthy near step `688/1932` in tmux
  `sallm-t2x-confirm-a2-s13` on assigned GPU 1 RTX 3080 Ti; conditional
  terminal ETA remains about `20:25 SAST`. Foreign GPU 0 RTX 5090 remained
  occupied and untouched. Kombuys root/scratch free remain `23 GB/2.1 TB`.
  This is within-candidate validation evidence only; no confirmation ranking
  occurs before a2 seeds 13 and 87 both terminate validly.
- HEX NER b7 job `1230086` improved at epoch 7 step 3787 to exact mean
  validation F1 `0.6708899559501509`, retaining checkpoint 3787 and resetting
  patience. Tsn/Xho/Zul F1 values are
  `0.6499535747446111/0.6627073301230106/0.7000089629828308`; its exact
  192-row, 64/language artifact has no literal empties or parse failures,
  `56` whitespace-only outputs, and `40/54/37` unique outputs. Debug/state
  SHA-256 values are
  `7455ebbb1263be51656393257762e3f7fcea4dcc55ffb0a741f9931e3883b2f8`/
  `c4ab23637005af27bdcbafbebc1fb6768dcddfa90e885ee6c6a054d2ffe121a3`.
  B7 has entered its step-4328 callback.
- NER b6 job `1227987` declined at epoch 12 step 6492 to exact mean
  validation F1 `0.42658069985026664`, below retained checkpoint-5951 best
  `0.4270927910241913`; this is its first patience miss after that
  improvement. Tsn/Xho/Zul F1 values are
  `0.43762327416168595/0.40615600561454795/0.43596281977456586`; the exact
  artifact has 192 rows, no literal empties or parse failures, `55`
  whitespace-only outputs, and `41/57/36` unique outputs. Debug/state hashes
  are `6e212d0fe4d79105feb10101bb687b2021ce780964394860dc23b0cb7b15e0b1`/
  `2f626d07ef7829585b3a85b5fdb5d3db93e4f9e8f52474ca73bc3050c07e4512`.
  B6 resumed and was near step `7006/8115`.
- NER b5 job `1226719` finished all `8115/8115` training steps and entered
  its final scheduled validation callback; no terminal artifact exists yet.
  Jobs `1226719/1227987/1230086` remain exactly three healthy A100-40GB jobs
  on `srvrocgpu010`, elapsed `17:51:58/15:28:52/08:58:46`. Conditional next
  complete callback/terminal evidence is expected around `20:05--20:40
  SAST`. HEX quota is home `52.0%`, scratch `38.2%`; no owned A100-80GB or
  L40S work exists.
- Scientifically trusted terminal progress remains base `16/16`, NER seed-42
  `8/11`, T2X seed-42 `11/11`, frozen winners `0/8`, held-out adapter
  evaluations `0`, and Mono not started. T2X confirmations remain `2/4`
  terminal-valid, with a2 seed 13 active and seed 87 pending. No held-out
  metric was accessed; Sheet E/F/G remain blank and Hugging Face publication
  remains blocked.

## T2X b7 confirmations complete; a2 seed 13 starts — 19:03 SAST

- Kombuys b7 seed-87 confirmation completed all `1932/1932` frozen steps
  and is terminal-valid. Validation chrF over epochs was
  `44.99226017209242`, `49.92559710705884`, `51.497458946416664`, and
  `51.72122400804512`; checkpoint 1932 is the retained best. The final exact
  64-row artifact has no literal or whitespace-only empty raw predictions and
  `63` unique outputs. Final debug, retained-state, retained-adapter,
  final-adapter, and adapter-config SHA-256 values are
  `17570f3e83ca282e4e1820b231b3022da017380848ed3817ee9273bec14d6e97`,
  `3c7a9c842f82867241069b05f601762b4bc2989bc63f60e4692c7a1b3b02419e`,
  `d3abfa6174ef833121f66ef27ec35ea3191b3f3b65c07e5b595812a14b76bda7`,
  `282bd2e9e75db44bfd4188661f11bbd5db86da5107e330d2cd9f301ab01a53e5`,
  and `086fc720659270d92e46dd77003d20da9bee63907f0081bfc246cca9e0a1a9ba`.
  All `424/424` adapter tensors, totaling `76,410,112` values, exactly match
  between retained checkpoint 1932 and the final adapter serialization. B7's
  three validation-only seed scores are therefore
  `52.26645731208185/51.975103592985164/51.72122400804512`; this does not
  freeze a winner before the two a2 confirmations finish.
- After verifying assigned GPU 1 idle, absent a2 seed-13 output/log/session,
  frozen ranking SHA-256
  `cd532deceac60acef5490a908a1ab9d7fd6522e6b1caf61f30cd3c9af40a2cb7`,
  registry SHA-256
  `8fdd6ea5f192a1d2ef2267353162b9a640ccd09ae0d889cb5a999645198bb726`,
  and confirmation pair `b7,a2`, a2 seed 13 started in tmux
  `sallm-t2x-confirm-a2-s13`. It verified all `694` immutable source/config
  files and exact `3,859/460` train/validation rows, then entered its frozen
  `1,932`-step run on assigned GPU 1 RTX 3080 Ti. Manifest/trial SHA-256
  values are
  `bac8a6850b589bf0741f22813af89e07699daa15963f88fd41990677f847ba46`/
  `696b37f912dcfd3aa6d9921541260540ebfd4a49cf3487837ca9fe4df815c7ca`.
  Conditional first-artifact/terminal ETAs are about `19:30/20:25 SAST`.
  Foreign GPU 0 RTX 5090 remains occupied and untouched; Kombuys root/scratch
  free are `23 GB/2.1 TB`.
- HEX NER b5 job `1226719` improved at epoch 14 step 7574 to exact mean
  validation F1 `0.5743710116912446`, retaining checkpoint 7574 and resetting
  patience. Tsn/Xho/Zul F1 values are
  `0.5659087922196898/0.570158259672845/0.5870459831811989`; its exact 192-row,
  64/language artifact has no literal empties or parse failures, `60`
  whitespace-only outputs, and `39/54/36` unique outputs. Debug/state hashes
  are `cd062606ccf205eeb87f8e612c9f2497f6264af5fc4b162f0b61e187ed7a0038`/
  `c1697ee1e72d35ae065dc962f941906e9341eb3b505a1aa93abafd4f13d6c1b9`.
  Jobs `1226719/1227987/1230086` remain exactly three healthy A100-40GB jobs
  on `srvrocgpu010`, elapsed `17:23:06/15:00:00/08:29:54`; b6/b7 have no
  newer complete callback. HEX quota is home `52.0%`, scratch `38.1%`; no
  owned A100-80GB or L40S work exists.
- Scientifically trusted terminal progress remains base `16/16`, NER seed-42
  `8/11`, T2X seed-42 `11/11`, frozen winners `0/8`, held-out adapter
  evaluations `0`, and Mono not started. T2X confirmations are `2/4`
  terminal-valid with a2 seed 13 active and a2 seed 87 still pending. No
  held-out metric was accessed; Sheet E/F/G remain blank and Hugging Face
  publication remains blocked.

## T2X b7 seed 87 reaches 51.50 chrF — 18:30 SAST

- Kombuys b7 seed-87 confirmation improved from validation chrF
  `44.99226017209242` at step 483 to `49.92559710705884` at step 966 and
  `51.497458946416664` at step 1449, retaining checkpoint 1449. All three
  exact 64-row artifacts have no literal or whitespace-only empty raw
  predictions; step 1449 has `64` unique outputs. Step-1449 debug/state
  SHA-256 values are
  `047d12b3705600ec39f24e654340f67a7f220c5d07ea8fbe792bd7cf81cebb20`/
  `41b89b1c35c309a593760b869867ad3b625556e4caad309462ae729a18c7439f`.
  Epoch-4 training resumed after the complete callback; conditional terminal
  ETA remains about `18:50 SAST`. Assigned GPU 1 RTX 3080 Ti remains active;
  foreign GPU 0 RTX 5090 remains occupied and untouched. Kombuys root/scratch
  free are `23 GB/2.1 TB`. This remains within-candidate validation evidence;
  no confirmation ranking occurs before all four runs terminate validly.
- HEX NER jobs `1226719/1227987/1230086` remain exactly three healthy
  A100-40GB jobs on `srvrocgpu010`, elapsed
  `16:51:42/14:28:36/07:58:30`, with allocation time left
  `07:08:18/09:31:24/16:01:30`. No new complete callback artifact exists
  after the 18:00 reconciliation. HEX quota remains home `52.0%`, scratch
  `38.2%`; no owned A100-80GB or L40S work exists.
- Scientifically trusted terminal progress remains base `16/16`, NER
  seed-42 `8/11`, T2X seed-42 `11/11`, frozen winners `0/8`, held-out adapter
  evaluations `0`, and Mono not started. No held-out metric was accessed;
  Sheet E/F/G remain blank and Hugging Face publication remains blocked.

## NER b6/b7 improve again; T2X seed 87 valid first artifact — 18:00 SAST

- HEX NER b5 job `1226719` declined at epoch 13 step 7033 to exact mean
  validation F1 `0.5722423856201379`, below retained best
  `0.5733473836832604` at checkpoint 6492; this is its first patience miss
  after the improvement. B6 job `1227987` improved at epoch 11 step 5951 to
  `0.4270927910241913`, retaining checkpoint 5951 and resetting patience. B7
  job `1230086` improved at epoch 6 step 3246 to `0.642549691285009`,
  retaining checkpoint 3246 and resetting patience. All three exact 192-row,
  64/language artifacts have no literal empty raw predictions or parse
  failures. Their whitespace-only totals are `60/58/50`; Tsn/Xho/Zul unique
  counts are `39/53/36`, `40/56/36`, and `41/57/38`. Debug/state SHA-256
  pairs are
  `4ef830289a08fe4dc423e5ff45e62c6586d1a0bbd751be1914948fb619476bd1`/
  `d3413b9783fde2c1212e2cbed010abafc5ec1aeaa32272598881f646ff2cb51f`,
  `c65755030066cb88887588702b8a7f9c41222c38b0c18972ab3d49248e57b42f`/
  `eb437fcc8435818ec9c8c6707300ce20f1d5c760740a5a989896c9d50dbcb851`,
  and `77bb4e250548a3afa10e1856185828014576726d05d2e0e3c74c86a2284223fb`/
  `8addc49604b4f2a9ca86c40f1a04d00b08884ed7a79363d6e72f0b24ccf57689`.
- Jobs `1226719/1227987/1230086` remain exactly three healthy A100-40GB jobs
  on `srvrocgpu010`, elapsed `16:21:36/13:58:30/07:28:24`, with allocation
  time left `07:38:24/10:01:30/16:31:36`. HEX quota is home `52.0%`, scratch
  `38.2%`; no owned A100-80GB or L40S work exists. Conditional next complete
  callbacks are around `18:55--19:05 SAST`.
- Kombuys b7 seed-87 confirmation produced a valid epoch-1 step-483
  validation chrF `44.99226017209242`, retaining checkpoint 483. Its exact
  64-row artifact has no literal or whitespace-only empty raw predictions and
  `60` unique outputs; debug/state SHA-256 values are
  `e374ea71e0ba7e4dd2f754c5a5e50919d98a8fb5fe46525dcb09b75b428c1f48`/
  `fecb08f8f18d43ca15da85e29982d3d8ca5fabea1499df245a1207c67f607f5e`.
  The run is healthy near step `732/1932` on assigned GPU 1 RTX 3080 Ti;
  conditional terminal ETA remains about `18:50 SAST`. Foreign GPU 0 RTX
  5090 remains occupied and untouched. Kombuys root free is now `22 GB` and
  scratch free remains `2.1 TB`. This is within-candidate validation evidence
  only; confirmation ranking remains blocked until all four runs finish.
- Scientifically trusted terminal progress remains base `16/16`, NER
  seed-42 `8/11`, T2X seed-42 `11/11`, frozen winners `0/8`, held-out adapter
  evaluations `0`, and Mono not started. No held-out metric was accessed;
  Sheet E/F/G remain blank and Hugging Face publication remains blocked.

## T2X b7 seed 13 terminal-valid; seed 87 starts — 17:31 SAST

- Kombuys b7 seed-13 confirmation completed all `1932/1932` frozen steps and
  is terminal-valid. Validation chrF over epochs was
  `45.60933714692265`, `49.040358075702585`, `51.55113862792683`, and
  `52.26645731208185`; checkpoint 1932 is the retained best. The final exact
  64-row artifact has no literal or whitespace-only empty raw predictions and
  `64` unique outputs. Final debug, retained-state, retained-adapter,
  final-adapter, and adapter-config SHA-256 values are
  `016483a044c8418034723f174103bc4dd27c95c627e1be67a743d8b5c7677217`,
  `d203d8c5fb5418ca044273850cd90b76db41e25aa6b8beee4d941087a40c422d`,
  `fa83fe95bbac224834244355633f2fe61c57d14ddfbf06211b108c4ca2eb409b`,
  `86cd1acd441a0362ffb47ade51bfc8633bd815561274421a047fa2719cf2b51e`,
  and `b62038215db914d91303d4a4b82102248b6937254a68d81c0e459755c76c3776`.
  All `424/424` adapter tensors, totaling `76,410,112` values, exactly match
  between retained checkpoint 1932 and the final adapter serialization.
- After verifying GPU 1 idle, absent seed-87 output/log targets, frozen
  ranking SHA-256 `cd532deceac60acef5490a908a1ab9d7fd6522e6b1caf61f30cd3c9af40a2cb7`,
  registry SHA-256
  `8fdd6ea5f192a1d2ef2267353162b9a640ccd09ae0d889cb5a999645198bb726`,
  and confirmation pair `b7,a2`, b7 seed 87 started in tmux
  `sallm-t2x-confirm-b7-s87`. It verified all `694` immutable source/config
  files, exact `3,859/460` train/validation rows, and entered its frozen
  1,932-step run on GPU 1 RTX 3080 Ti. Execution-manifest/trial SHA-256 values
  are `421e9e2afb24d03c420b1714a8fa4a7509cd06bb567a0210b23b4b9510ac1334`/
  `2d0fd4f13a1c63435f65766dfffb6b7e81ec7fa61aa6dd13f7f6d4a753b6063e`.
  Conditional first-artifact/terminal ETAs are about `17:55/18:50 SAST`.
  Foreign GPU 0 RTX 5090 remains occupied and untouched; Kombuys root/scratch
  free remain `23 GB/2.1 TB`.
- HEX NER jobs `1226719/1227987/1230086` remain exactly three healthy
  A100-40GB jobs on `srvrocgpu010`, elapsed
  `15:51:38/13:28:32/06:58:26`, with allocation time left
  `08:08:22/10:31:28/17:01:34`. No new complete callback artifact exists
  after the 17:00 reconciliation. HEX quota remains home `52.0%`, scratch
  `38.1%`; no owned A100-80GB or L40S work exists.
- Scientifically trusted terminal progress remains base `16/16`, NER
  seed-42 `8/11`, T2X seed-42 `11/11`, frozen winners `0/8`, held-out adapter
  evaluations `0`, and Mono not started. B7 seed 87 plus the two a2
  confirmations and the NER terminals/confirmations remain the immediate
  blockers. No held-out metric was accessed; Sheet E/F/G remain blank and
  Hugging Face publication remains blocked.

## NER b6/b7 improve; T2X confirmation reaches 49.04 chrF — 17:00 SAST

- HEX NER b6 job `1227987` improved at epoch 10 step 5410 to Tsn/Xho/Zul
  validation F1 `0.4356891416192298/0.3938532786022025/0.4304066625320666`,
  exact mean `0.4199830275844996`, retaining checkpoint 5410 and resetting
  patience. B7 job `1230086` improved at epoch 5 step 2705 to
  `0.6332838439734957/0.611708221943319/0.670272252268719`, exact mean
  `0.6384214393951778`, retaining checkpoint 2705 and resetting patience.
  Both exact 192-row, 64/language artifacts have no literal empty raw
  predictions or parse failures. B6/B7 whitespace-only totals are `62/56`;
  Tsn/Xho/Zul unique counts are `36/57/35` and `40/55/37`. Their
  debug/state/adapter SHA-256 triples are
  `e3d52483d68a861bd7dfb1238bcc65190ce8d066f8969c108d0af4738c0c96ff`/
  `e249d695c4c21d44de81dcb5fc78101d05ffbad4759e01eca5bf319b22cfef50`/
  `3f530c521eff89b7bb234606ca52b58f647bae25408c35e4d8005b95b28eea39`
  and `1122afef010484102aa85290a3db3f6686c3e4e536e878c4d4d8a249fadcd34e`/
  `556f31a70d38e00895a68b39e89bff417b7d078e8b1472ac1c592870de0d8523`/
  `4dce00c7342fcf41ff6b5d1e3a8664fd5b4e5702a0f27fe3cc4bd67eedfcae80`.
- Jobs `1226719/1227987/1230086` remain exactly three healthy A100-40GB jobs
  on `srvrocgpu010`, elapsed `15:21:40/12:58:34/06:28:28`, with allocation
  time left `08:38:20/11:01:26/17:31:32`. HEX quota is home `52.0%`, scratch
  `38.1%`; no owned A100-80GB or L40S work exists. Conditional next complete
  artifacts are around `17:40--17:50 SAST`.
- Kombuys b7 seed-13 confirmation improved from validation chrF
  `45.60933714692265` at step 483 to `49.040358075702585` at step 966,
  retaining checkpoint 966. Both exact 64-row artifacts have no empty raw
  predictions; step 966 has `63` unique outputs. Step-966 debug/state
  SHA-256 values are
  `da86b9c46239986d0dbb399a02433ca3acaca215f2b221266a1b2b5a1fe450ec`/
  `a7c383817554bdae9fbf83a9824fe85bbd884954b43180c6ddfbfe84d76afe8a`.
  The run is healthy near step `1393/1932` on assigned GPU 1 RTX 3080 Ti;
  conditional terminal ETA remains about `17:25 SAST`. Foreign GPU 0 RTX
  5090 remains occupied and untouched; Kombuys root/scratch free remain
  `23 GB/2.1 TB`. This is within-candidate validation evidence only; no
  confirmation ranking occurs before all frozen seeds complete.
- Scientifically trusted terminal progress remains base `16/16`, NER
  seed-42 `8/11`, T2X seed-42 `11/11`, frozen winners `0/8`, held-out adapter
  evaluations `0`, and Mono not started. No held-out metric was accessed;
  Sheet E/F/G remain blank and Hugging Face publication remains blocked.

## NER b5 improves; T2X confirmation healthy — 16:30 SAST

- HEX NER b5 job `1226719` improved at epoch 12 step 6492 to Tsn/Xho/Zul
  validation F1 `0.566977363515263/0.567200723904778/0.5858640636297403`,
  exact mean `0.5733473836832604`. It retains checkpoint 6492 and resets
  patience. The exact 192-row, 64/language artifact has no literal empty raw
  predictions or parse failures, `60` whitespace-only predictions, and
  Tsn/Xho/Zul unique counts `40/53/36`. Debug, trainer-state, and retained
  adapter SHA-256 values are
  `054b855c904ea3002876963dfb611c472a1c989de4a11d06d998dba90204613d`,
  `9a8cea1fa2d7261cc9f93c1005f89de2309ab32954d8c5c9b8ef9bce7900cc34`,
  and `3b6c5ebdca0f23b657ce9b282cbb80d92ce92b02e75c9b2fa3ca81f429a8391c`.
  B6/b7 have incomplete step-5410/2705 callbacks, so no metric or checkpoint
  decision is taken from them.
- Jobs `1226719/1227987/1230086` remain exactly three healthy A100-40GB jobs
  on `srvrocgpu010`, elapsed `14:51:44/12:28:38/05:58:32`, with allocation
  time left `09:08:16/11:31:22/18:01:28`. HEX quota remains home `52.0%`,
  scratch `38.2%`; no owned A100-80GB or L40S work exists.
- Kombuys b7 seed-13 confirmation is healthy near step `613/1932` in tmux
  `sallm-t2x-confirm-b7-s13`; no complete validation artifact exists yet, so
  the seed-42 ranking remains the only selection evidence. Conditional first
  artifact/terminal ETAs remain about `16:35/17:25 SAST`. Assigned GPU 1 RTX
  3080 Ti is active; foreign GPU 0 RTX 5090 remains occupied and untouched.
  Kombuys root/scratch free remain `23 GB/2.1 TB`.
- Scientifically trusted terminal progress remains base `16/16`, NER
  seed-42 `8/11`, T2X seed-42 `11/11`, frozen winners `0/8`, held-out adapter
  evaluations `0`, and Mono not started. Four T2X confirmations and the three
  NER terminals/confirmations remain the immediate blockers. No held-out
  metric was accessed; Sheet E/F/G remain blank and Hugging Face publication
  remains blocked.

## T2X seed-42 grid complete; confirmation begins — 16:06 SAST

- Kombuys T2X b7 completed all `1932/1932` frozen steps and is terminal-valid,
  taking the seed-42 grid to `11/11`. Validation chrF over epochs was
  `44.69243255853447`, `49.68565734381204`, `51.925102265094594`, and
  `51.975103592985164`; checkpoint 1932 is the retained best. Its final exact
  64-row Xhosa artifact has no literal or whitespace-only empty raw
  predictions and `63` unique outputs. Final debug, retained-state,
  final-adapter, and adapter-config SHA-256 values are
  `df7320a3538bde5fd6fb40de4e12dccf0aada06f982f45597afdee06f6935ebc`,
  `e5473cb9250952aa8f1f7a67dc18015502745f533c63b0ef03bf63470d81b46a`,
  `f95725b9e5ba175b46a38ad89c61fcbbd459cac2b63339b9d64afd2fb3ff2bc0`,
  and `e21938cac6ebfa81faa7855b72f9d0ad7ce9dd892f08412ccd9ea2bdb77235ea`.
  All `424/424` adapter tensors, totaling `76,410,112` values, exactly match
  between retained checkpoint 1932 and the final adapter serialization.
- The immutable validation-only seed-42 ranking artifact is
  `/scratch/alombard/masters/sallm/results/adapter_hpo_v3/pure_gdn/t2x/seed42_ranking.json`,
  SHA-256 `cd532deceac60acef5490a908a1ab9d7fd6522e6b1caf61f30cd3c9af40a2cb7`.
  It validates all 11 trial manifests against registry SHA-256
  `8fdd6ea5f192a1d2ef2267353162b9a640ccd09ae0d889cb5a999645198bb726`
  and freezes b7 (`51.975103592985164`) and a2
  (`51.306312454497196`) as the top two for seeds 13 and 87. No held-out
  artifact was accessed.
- After confirming GPU 1 idle, absent output/log targets, the ranking and
  registry hashes, and the frozen confirmation pair `b7,a2`, b7 seed 13
  started in tmux `sallm-t2x-confirm-b7-s13`. It verified all `694`
  immutable source/config files, exact `3,859/460` train/validation rows, and
  entered the frozen 1,932-step run on GPU 1 RTX 3080 Ti. Execution
  manifest/trial SHA-256 values are
  `d1286b1219667a9bacc574f2fd6011fd1ac6a99688a07f144f89842887b34aaa`/
  `d103dfb8d54dffc13375f643f9d0530ed5b96e104809b90cb3cd618edf8c6ba4`.
  Conditional first-artifact/terminal ETAs are about `16:35/17:25 SAST`.
  Foreign GPU 0 RTX 5090 remains occupied and untouched; Kombuys root/scratch
  free remain `23 GB/2.1 TB`.
- At 15:59 SAST HEX jobs `1226719/1227987/1230086` remained exactly three
  healthy A100-40GB jobs on `srvrocgpu010`, elapsed
  `14:22:02/11:58:56/05:28:50`, with allocation time left
  `09:37:58/12:01:04/18:31:10`. B5/b6 were in their next corrected callbacks
  at steps 6492/5410; no complete new selection artifact existed. HEX quota
  remained home `52.0%`, scratch `38.2%`, with no owned A100-80GB or L40S.
- Scientifically trusted terminal progress is now base `16/16`, NER seed-42
  `8/11`, and T2X seed-42 `11/11`; frozen task-family winners remain `0/8`,
  held-out adapter evaluations `0`, and Mono not started. Four serial T2X
  confirmation runs and the three NER terminals/confirmations remain the
  immediate blockers. Sheet E/F/G remain blank and Hugging Face publication
  remains blocked.

## NER b5/b7 improve; T2X b7 reaches 49.69 chrF — 15:30 SAST

- HEX NER b5 job `1226719` improved at epoch 11 step 5951 to exact mean
  validation F1 `0.5719458254448195` (Tsn/Xho/Zul
  `0.5667928610058993/0.5692530430585593/0.5797915722699999`), retaining
  checkpoint 5951 and resetting patience. B6 job `1227987` declined at epoch
  9 step 4869 to `0.40108590595274`, its first patience miss after improving,
  and retains checkpoint 4328 best `0.4045046237234104`. B7 job `1230086`
  improved at epoch 4 step 2164 to `0.6165239138215678` (Tsn/Xho/Zul
  `0.6299057532704524/0.5909921530688528/0.6286738351253982`), retaining
  checkpoint 2164 and resetting patience. All three artifacts have exact
  192-row and 64/language coverage, no literal empty predictions or parse
  failures, and debug/state SHA-256 pairs
  `d0979ef3316e72038e717244a85b633d755c6d8fd99050ea7192afddaa62611f`/
  `1d02b2c9e5eb5d2b2f95447deb602fdbe29aaa40f83e6307b7f253b2ca7a782d`,
  `b81996e2678bad8622fd20a0b4f4ec98cd041fd4583da11c988eedfa2f56c496`/
  `3ab69b07d81fe1332c3947f7c1fa9cf6b1f9b8c89c0bc9866aad2b4f198f687e`,
  and `4f240e39a0ba20e92e5ee3a60e8d119c1b7f39d23bab427dbaa0057fe18c8a40`/
  `003799cfe1ec9b62976331424cc905714a00ffdb4c767404d7d9474ce786863c`.
- Jobs `1226719/1227987/1230086` remain exactly three healthy A100-40GB jobs
  on `srvrocgpu010`, elapsed `13:51:54/11:28:48/04:58:42`, with remaining
  allocations `10:08:06/12:31:12/19:01:18`. HEX quota is home `52.0%` and
  scratch `38.2%`; there is no owned A100-80GB or L40S work.
- Kombuys T2X b7 improved at epoch 2 step 966 to validation chrF
  `49.68565734381204`, retaining checkpoint 966. Its exact 64-row Xhosa
  artifact has no literal or whitespace-only empty raw predictions and `62`
  unique outputs; debug/state SHA-256 values are
  `f7bbd6b10cdaa1c49472592695a161e46a34ce60fe3cc01b766a257712127ba6`/
  `7db93b3e13f99ee21e0913919277834ef7bff6b94f24365129a4a6ce1008a33a`.
  Epoch 3 completed training and its corrected step-1449 generation callback
  is active; no incomplete metric is used. Assigned GPU 1 RTX 3080 Ti is
  about `96%` utilized by b7; foreign GPU 0 RTX 5090 is about `96%` utilized
  and remains untouched. Kombuys root has `23 GB` free and scratch `2.1 TB`
  free; conditional terminal ETA remains about `16:00--16:10 SAST`.
- Scientifically trusted terminal progress remains base `16/16`, NER
  seed-42 `8/11`, and T2X seed-42 `10/11`; frozen winners `0/8`, held-out
  adapter evaluations `0`, and Mono not started. No ranking or held-out
  metric access occurred. T2X b7 and the three NER terminal completions,
  followed by preregistered confirmation seeds, remain the blockers. Sheet
  E/F/G remain blank and Hugging Face publication remains blocked.

## T2X b7 first artifact valid; NER callbacks continue — 15:00 SAST

- Kombuys T2X b7 produced a valid epoch-1 step-483 validation chrF
  `44.69243255853447`, retaining checkpoint 483. Its exact 64-row Xhosa
  artifact has no literal or whitespace-only empty raw predictions and `63`
  unique outputs. Debug/state SHA-256 values are
  `48d2f3d5e7fe3a4569078cd556ad8e170c079b166c70e766f286f9f379c6b95f`/
  `47f0421fe2ac0978437f6b0e25885d50923107c74aa761f8c99a252918316cb9`.
  This is within-candidate validation evidence only; no T2X ranking occurred.
  The run is healthy near step `734/1932` in tmux `sallm-t2x-hpo-b7`, with
  conditional terminal ETA about `16:00--16:10 SAST`. Assigned GPU 1 RTX
  3080 Ti remains isolated; foreign GPU 0 RTX 5090 remains occupied at about
  `93%` utilization and was untouched. Kombuys root has `23 GB` free and
  scratch `2.1 TB` free.
- HEX NER b5/b6/b7 jobs `1226719/1227987/1230086` remain exactly three
  healthy A100-40GB jobs on `srvrocgpu010`, elapsed
  `13:21:53/10:58:47/04:28:41`. They are in their epoch-11/9/4 corrected
  callbacks at steps `5951/4869/2164`; no complete new debug artifact exists,
  so no metric, checkpoint, or patience decision was made. HEX quota remains
  home `52.0%`, scratch `38.2%`; no owned A100-80GB or L40S work exists.
- Scientifically trusted terminal progress remains base `16/16`, NER
  seed-42 `8/11`, and T2X seed-42 `10/11`; frozen winners `0/8`, held-out
  adapter evaluations `0`, and Mono not started. T2X b7 terminal completion,
  all three NER terminals, and subsequent preregistered confirmation seeds
  remain the blocker. No held-out metric was accessed; Sheet E/F/G remain
  blank and Hugging Face publication remains blocked.

## T2X b6 terminal-valid and b7 starts; NER artifacts advance — 14:32 SAST

- Kombuys T2X b6 completed all `1932/1932` frozen steps and is
  terminal-valid, taking seed-42 progress to `10/11`. Validation chrF across
  epochs was `31.376884549616697`, `36.85528231891048`,
  `38.35361029340392`, and `38.13150671845184`; checkpoint 1449 remains the
  retained best. The final exact 64-row Xhosa artifact has no literal or
  whitespace-only empty raw predictions and `64` unique outputs. Final debug,
  retained-state, final-adapter, and adapter-config SHA-256 values are
  `32b12f27bfcb344a06b9d141f5c4e50856728343517d74c6db3bbcbe66940940`,
  `9e977c481c045c469ce8808ec427b5765147d720ba3e0d189105a20975e31f47`,
  `3b84725d99758751b0f3aee07002e898cdcf26f3c50a43d5e0166500ef4baa55`,
  and `3ed3dd1f8655fcc7816ef132e8c3b88fac7143873421644949bf298e0c68e800`.
  All `424/424` adapter tensors, totaling `71,762,560` values, exactly match
  retained checkpoint 1449 after loading the two serialization formats.
- After verifying GPU 1 idle, absent b7 output/log, the frozen registry SHA
  `8fdd6ea5f192a1d2ef2267353162b9a640ccd09ae0d889cb5a999645198bb726`,
  and the canonical Kombuys model path, preregistered T2X b7 started in tmux
  `sallm-t2x-hpo-b7`. Its frozen recipe is LR `0.0001223079850011719`,
  rank/alpha `32/64`, dropout `0.008055734634399415`, warmup
  `0.0765941160917282`, seed 42, BF16, and 1932 steps. It verified all `694`
  immutable files, loaded the canonical model, verified exact `3,859/460`
  train/validation rows, and entered training. Manifest/trial hashes are
  `c77101425dee388023ca8e1ff5eeaa35d75a98bdb2aacf762be79050c6acd66b`/
  `ce1e329811408d0857ccbf19e16efcfa503382c20af088cc1fb77e4b16ea18d0`.
  Conditional first-artifact/terminal ETAs are about `14:55/15:55 SAST`.
  Assigned GPU 1 RTX 3080 Ti remains isolated; foreign GPU 0 RTX 5090 is
  occupied and untouched.
- HEX NER b5 job `1226719` declined at epoch 10 step 5410 to exact mean F1
  `0.5525407691385177`, its first patience miss after the preceding
  improvement, and retains checkpoint 4869 best `0.5584894571005082`. B6 job
  `1227987` improved at epoch 8 step 4328 to `0.4045046237234104`, retaining
  checkpoint 4328 and resetting patience. B7 job `1230086` improved at epoch
  3 step 1623 to `0.5725969497508221`, retaining checkpoint 1623 and resetting
  patience. All artifacts have exact 192-row and 64/language coverage, no
  literal empty raw predictions or parse failures, and whitespace-only totals
  `59/59/53`. Their debug/state SHA-256 pairs are
  `87942e71...179c0`/`a2ecc77c...a67dc`,
  `5acc92bd...5fe3e`/`ed39fc3d...93099`, and
  `33a76d22...00c62`/`235a19b1...dad64`.
- Jobs `1226719/1227987/1230086` remain exactly three healthy A100-40GB jobs
  on `srvrocgpu010`, elapsed `12:51:56/10:28:50/03:58:44`. HEX quota is home
  `52.0%`, scratch `38.2%`; no owned A100-80GB or L40S work exists. Trusted
  base remains `16/16`; NER seed-42 terminal-valid `8/11`; T2X seed-42
  terminal-valid `10/11`; frozen winners `0/8`; held-out adapter evaluations
  `0`; Mono not started. Completing b7 and all NER terminals, then performing
  preregistered confirmation seeds, remains the blocker. No held-out metric
  was accessed; Sheet E/F/G remain blank and Hugging Face remains blocked.

## T2X b6 improves; three HEX callbacks remain healthy — 14:00 SAST

- Kombuys T2X b6 improved at epoch 2 step 966 from chrF
  `31.376884549616697` to `36.85528231891048`, retaining checkpoint 966.
  Its exact 64-row Xhosa artifact has no literal or whitespace-only empty raw
  predictions and `64` unique outputs. Debug/state SHA-256 values are
  `d78c929748d66c90f882f4b9b12fab30f72ade2a196510e1690576b3e0a313fc`/
  `c7f22c6be767b2c8e900f41cbab63bff8978290358113f8c0ae846f88fd6b815`.
  The run remains healthy in tmux `sallm-t2x-hpo-b6-r1`; conditional terminal
  ETA is about `15:00 SAST`. Assigned GPU 1 RTX 3080 Ti remains isolated.
  Foreign GPU 0 RTX 5090 is `93%` utilized and was untouched. Kombuys root
  has `23 GB` free and scratch `2.1 TB` free.
- HEX NER b5/b6/b7 jobs `1226719/1227987/1230086` remain exactly three
  healthy A100-40GB jobs on `srvrocgpu010`, elapsed
  `12:22:06/09:59:00/03:28:54`. They are all inside corrected generation
  callbacks for steps `5410/4328/1623`; no complete new artifact exists, so
  no checkpoint or patience decision was made. HEX quota remains home
  `52.0%`, scratch `38.2%`; no owned A100-80GB or L40S work exists.
- Operationally the HPO machinery is behaving as intended: candidates are
  separating on validation, invalid/incomplete artifacts are excluded, and
  retry provenance is isolated. Scientific claims remain deliberately
  narrower: base `16/16` trusted, NER seed-42 terminal-valid `8/11`, T2X
  seed-42 terminal-valid `9/11`, frozen winners `0/8`, and held-out adapter
  evaluations `0`. This is an appropriately rigorous search rather than
  needless complexity; the multi-seed confirmation and one-time test gates
  are what prevent optimistic validation results from being mistaken for
  generalization. No held-out metric was accessed; Sheet E/F/G remain blank
  and Hugging Face publication remains blocked.

## NER b7 improves; T2X b6 first artifact valid — 13:32 SAST

- HEX NER b7 job `1230086` improved at epoch 2 step 1082 from its initial
  mean F1 `0.23712363123925098` to Tsn/Xho/Zul F1
  `0.5019266625232567/0.46150749723251516/0.5087644292432164`, exact mean
  `0.4907328629996628`. It retains checkpoint 1082 and resets patience. The
  artifact has exact `192` rows and `64/language`, no literal empty raw
  predictions or parse failures, `54` whitespace-only raw predictions, and
  Tsn/Xho/Zul unique counts `42/55/36`. Debug/state SHA-256 values are
  `1f347f3ff73d5b714350c67cc312192d13018a043f2cdf1dd25b42a0ca68013f`/
  `a9ca20de82449cb96785d78809548b87a6d8a73e1eca26c537fd416b51d5850e`.
- NER b5/b6 jobs `1226719/1227987` entered their epoch-10/8 corrected
  callbacks at steps `5410/4328`; no complete new debug artifact exists, so
  no checkpoint or patience decision was made. Together with b7 they are
  exactly three healthy jobs on `srvrocgpu010` A100-40GB, elapsed
  `11:52:44/09:29:38/02:59:32` at the status check. HEX quota is home
  `52.0%`, scratch `38.2%`; no owned A100-80GB or L40S work exists.
- Kombuys T2X b6 produced a valid epoch-1 step-483 chrF
  `31.376884549616697`, retaining checkpoint 483. Its exact 64-row Xhosa
  artifact has no literal or whitespace-only empty raw predictions and `63`
  unique outputs. Debug/state SHA-256 values are
  `36b64e6e706ec514cf1d281fd5d415876020e78b500f0955485c90aff41fd91f`/
  `31aeaeffb986a2bc86df5f8b50c04032deb0eff9de1391c4b15d3788619340e5`.
  The environment-only retry is healthy near step `775/1932` in tmux
  `sallm-t2x-hpo-b6-r1`, with conditional terminal ETA about
  `14:25--14:35 SAST`. Assigned GPU 1 RTX 3080 Ti is isolated; foreign GPU 0
  RTX 5090 remains about `92%` utilized and was untouched.
- Operationally both active grids continue to produce valid validation
  artifacts. Scientifically trusted terminal progress remains base `16/16`,
  NER seed-42 `8/11`, and T2X seed-42 `9/11`; frozen winners `0/8`, held-out
  adapter evaluations `0`, and Mono not started. Grid completion followed by
  preregistered multi-seed confirmation remains the blocker. No held-out
  metric was accessed; Sheet E/F/G remain blank and Hugging Face publication
  remains blocked.

## NER b5 improves; T2X b6 path retry enters model load — 13:00 SAST

- HEX NER b5 job `1226719` improved at epoch 9 step 4869 to Tsn/Xho/Zul F1
  `0.5500723779444165/0.5519957424161291/0.5734002509409789`, exact mean
  `0.5584894571005082`. It resets patience and retains checkpoint 4869. The
  exact 192-row, 64/language artifact has no literal empty raw predictions or
  parse failures and `60` whitespace-only raw predictions. Debug/state
  SHA-256 values are
  `ef6b7676a6e785c6a79822a0f462d40346a566d05147a8885d60252cb12db17e`/
  `65394c17f84cd4ff2104c290e60f82658f92e3363d4a223dcf47be35b4d47a4f`.
- NER b6 job `1227987` declined at epoch 7 step 3787 to Tsn/Xho/Zul F1
  `0.3586993725042294/0.31952217466956406/0.3836160782515177`, exact mean
  `0.3539458751417704`. This is the first frozen patience miss; it correctly
  retains checkpoint 3246 best `0.3750359191540918`. Coverage is exact 192
  and 64/language, with no literal empty raw predictions or parse failures and
  `53` whitespace-only raw predictions. Debug/latest-state SHA-256 values are
  `2e696a6a0c02100d9c233df20fccbd62daace76ec286252fc2293cebdbce414e`/
  `e3c7b35cf865b840f9c63eee716dfe4d69156343c15de3be4e190a86a347258c`.
  NER b7 job `1230086` remains in its epoch-2 step-1082 corrected callback;
  no complete artifact exists yet.
- The three jobs remain healthy on `srvrocgpu010` A100-40GB at elapsed
  `11:21:52/08:58:46/02:28:40`. HEX quota is home `52.0%`, scratch `38.2%`;
  exactly three owned A100-40GB jobs remain active and no owned A100-80GB or
  L40S work exists. Conditional next-artifact ETA is about
  `13:15--13:35 SAST`.
- Kombuys T2X b6 attempt 0 failed after immutable-manifest verification but
  before canonical model/data/metric access because the environment resolved
  the model under nonexistent `/scratch/alombard/masters/...` instead of the
  verified Kombuys location. It is preserved unchanged under
  `failure_provenance/t2x_b6_seed42_attempt0_model_path_20260813T123244`;
  manifest/trial/log SHA-256 values are
  `8882d9f0e609cc89173251a414408d45ac7f250f3ba686aefdbbb21bb6f20b1d`,
  `60519a3e77ded53d444dfc18e6e70c9b9a8b66a4e364e2a2e5ffbce706deec59`,
  and `8ff97a5a0a544dd4070d94d324d01caef2599f93a86ab012566df378432d95b0`.
  No selection evidence was exposed.
- The environment-only retry in tmux `sallm-t2x-hpo-b6-r1` sets the verified
  canonical model/tokenizer path and otherwise preserves the candidate. It
  verified all `694` immutable files, passed the GatedDeltaNet kernel gate,
  loaded the canonical BF16 model, verified exact `3,859/460`
  train/validation rows, and entered training on assigned GPU 1. Retry
  manifest/trial SHA-256 values are
  `75340d5f6bda5be3c81efdd2655c0ccc3adc4186ef6801fb7e7b1b2a84ecf104`/
  `7e9be9d1b97e724f04b983a70fbe3b5aacb062dedefb01dd13f50bd0fcd3148e`.
  Conditional first-artifact/terminal ETAs are about `13:25/14:25 SAST`.
  GPU 1 RTX 3080 Ti remains isolated; foreign GPU 0 RTX 5090 remains occupied
  and untouched.
- Trusted base `16/16`; NER seed-42 terminal-valid `8/11`; T2X seed-42
  terminal-valid `9/11`; frozen winners `0/8`; held-out adapter evaluations
  `0`; Mono not started; Sheet E/F/G blank; Hugging Face blocked. Grid
  completion and preregistered multi-seed confirmation remain the blocker;
  no held-out metric was accessed.

## T2X b5 terminal-valid; b6 starts; HEX callbacks healthy — 12:32 SAST

- Kombuys T2X Stage-B `b5` completed all frozen `1932/1932` steps and is
  terminal-valid, taking seed-42 progress to `9/11`. Validation chrF rose
  across epochs to `35.59502883566279`, `43.72808904321918`,
  `44.239883696192486`, and `44.54370114772246`; checkpoint 1932 is the
  retained best. The final exact
  64-row Xhosa artifact has no literal or whitespace-only empty raw
  predictions and `61` unique outputs. Final debug, retained-state,
  final-adapter, and adapter-config SHA-256 values are
  `5d546045e84593bab8612b66e571e5e5b1ed4ac01faa86a0c87bec385c9aed7b`,
  `266e5ff9ce166eaa94596bf7ed93e63110db174a4171e28f205b55cdbaa0aaf2`,
  `dd285ba3f545f0709b47431ac05f6c63314603df87b67bb918a5fa7578d22895`,
  and `13112526d822d8b5eae5f9fabdd2ce2b294ebe54ff70a50bbf5f7159eb243ccb`.
  No cross-candidate ranking occurred.
- After b5 terminated and assigned GPU 1 was verified idle, preregistered T2X
  Stage-B `b6` started in tmux `sallm-t2x-hpo-b6`. It uses LR
  `0.000021146461814662826`, rank/alpha `16/32`, dropout
  `0.05002756714820862`, warmup `0.017216095104813575`, seed 42, BF16, and
  the frozen 1932-step batch semantics. It wrote manifest/trial SHA-256
  `8882d9f0e609cc89173251a414408d45ac7f250f3ba686aefdbbb21bb6f20b1d`/
  `60519a3e77ded53d444dfc18e6e70c9b9a8b66a4e364e2a2e5ffbce706deec59`,
  verified all `694` immutable files, and reached the GatedDeltaNet kernel
  gate. Conditional first-artifact/terminal ETAs are about
  `12:55/13:55 SAST`. Assigned GPU 1 RTX 3080 Ti remains isolated; foreign
  GPU 0 RTX 5090 remains occupied and untouched. Kombuys root has `23 GB`
  free and scratch `2.1 TB` free.
- HEX NER b5/b6/b7 jobs `1226719/1227987/1230086` remain exactly three
  healthy A100-40GB runs on `srvrocgpu010`, elapsed
  `10:51:51/08:28:45/01:58:39`. B5 and b6 are in the next corrected
  callbacks, while b7 entered its epoch-2 step-1082 callback; no new complete
  debug artifact exists, so no metric or patience decision was made. HEX
  quota remains home `52.0%`, scratch `38.1%`; no owned A100-80GB/L40S work
  is active.
- Trusted base `16/16`; NER seed-42 terminal-valid `8/11`; T2X seed-42
  terminal-valid `9/11`; frozen winners `0/8`; held-out adapter evaluations
  `0`; Mono not started; Sheet E/F/G blank; Hugging Face blocked. The current
  blocker remains completion of both seed-42 grids and preregistered
  multi-seed confirmation. All chrF/F1 values above are validation-only, not
  held-out test evidence.

## NER b5/b6/b7 improve; T2X b5 reaches 43.73 chrF — 12:00 SAST

- HEX NER b5/b6/b7 jobs `1226719/1227987/1230086` produced valid corrected
  step `4328/3246/541` artifacts. Exact Tsn/Xho/Zul F1 and their unweighted
  means are
  `0.5478494623655414/0.5299919159255768/0.5734380708804763` ->
  `0.5504264830571982`,
  `0.3919698870764872/0.3485961348126812/0.384541735573107` ->
  `0.3750359191540918`, and
  `0.23150183150178386/0.22900692411352436/0.2508621381024448` ->
  `0.23712363123925098`. B5 and b6 improve over their preceding checkpoints;
  b7 initializes its best, so all three retain the new checkpoints and reset
  patience. Each artifact has exact `192` rows and `64/language`, no literal
  empty raw predictions or parse failures, and whitespace-only raw totals
  `60/56/64`. Debug/state SHA-256 values are
  `1a0c248aed2f2c046bbe031916241699478e9e383a74349ad87712f9d56d0787`/
  `ca384ab351a7ddbccdf2b2e47d75ddc4f3ace94ac99dc631c6c1349101c708d9`,
  `76ebca0ccaaf0298defc7f91e2d8aef3b75ee734ea518cb0897c0c3c7fbf8389`/
  `4c527dc9c282768bb60f2dde1953bed1dd8a27644aeb4c9b1faef67c1a6088ba`,
  and
  `8e41259c9cc5aa8e1a1821c5a7089112d38498fd7818d1268ff76583c1b78347`/
  `182d6451ccad979d2555a33666f2325054dc684a7e2990128e4f61982094f3c7`.
  No cross-candidate ranking occurred.
- The same three jobs remain healthy on `srvrocgpu010` A100-40GB at elapsed
  `10:22:31/07:59:25/01:29:19`. B5 is near `4852/8115` and its epoch-9
  callback is imminent; b6 is already in the next corrected callback; b7 is
  training epoch 2. The next material artifacts are expected roughly
  `12:15--13:00 SAST`, conditional on callback duration. HEX quota is home
  `52.0%`, scratch `38.1%`; exactly three owned A100-40GB jobs are active and
  no owned A100-80GB/L40S work exists.
- Kombuys T2X b5 improved at epoch 2 step 966 from `35.59502883566279` to
  validation chrF `43.72808904321918`, retaining checkpoint 966. Its exact
  64-row Xhosa artifact has no literal or whitespace-only empty raw
  predictions and `63` unique outputs. Debug/state SHA-256 values are
  `b33e068792ec7cf88d3eb6b15eec398cc740caa677954f7286899f05cf181f2e` and
  `16674327d3b858f72d973e76f6b7935c908f10535cdc4201b870a1125b22a911`.
  The offline retry remains healthy in its epoch-3 callback, with terminal
  ETA about `12:25 SAST`. Assigned GPU 1 RTX 3080 Ti remains isolated;
  foreign GPU 0 RTX 5090 is occupied and untouched. This is validation-only
  evidence and does not establish held-out test performance.
- Trusted base `16/16`; NER seed-42 terminal-valid `8/11`; T2X seed-42
  terminal-valid `8/11`; frozen winners `0/8`; held-out adapter evaluations
  `0`; Mono not started; Sheet E/F/G blank; Hugging Face blocked. The blocker
  is completion of both seed-42 grids, followed by preregistered multi-seed
  confirmation; held-out evaluation remains gated.

## T2X b5 first artifact valid; HEX callbacks continue — 11:30 SAST

- Kombuys T2X Stage-B `b5` produced a valid initial epoch-1 step-483 chrF
  `35.59502883566279`, retaining checkpoint 483. Its artifact has exact
  64-row Xhosa coverage, no literal or whitespace-only empty raw predictions,
  and `64` unique outputs. Debug/state SHA-256 values are
  `b06d679e613efdd1e33029caacd58d7b89506363dec3694b9ad037736962841d`
  and `3800f5d2f2298d8abf8f224e3cdb37effe15fc8aa0fdadf65ccf598303b5bfa8`.
  This is within-candidate validation evidence only; no T2X ranking occurred.
  The retry is healthy near `645/1932`, with terminal ETA about `12:23 SAST`.
  Assigned GPU 1 is isolated; foreign GPU 0 remains occupied and untouched.
  Kombuys root has `23 GB` free and scratch `2.1 TB` free.
- HEX NER b5/b6/b7 jobs `1226719/1227987/1230086` remain exactly three
  healthy A100-40GB runs on `srvrocgpu010`, elapsed
  `09:51:41/07:28:35/00:58:29`. B5/b6 remain in epoch-8/6 corrected callbacks
  at steps `4328/3246` after health-only losses
  `0.3844813180235682/0.7173408253042228`. B7 entered its initial step-541
  callback after health-only loss `1.0148706145446096`. None of the three new
  debug artifacts is complete, so no metric, checkpoint, or patience decision
  was made. HEX quota is home `52.0%`, scratch `38.1%`; no owned
  A100-80GB/L40S work is active.
- Trusted base `16/16`; NER seed-42 terminal-valid `8/11`; T2X seed-42
  terminal-valid `8/11`; frozen winners `0/8`; held-out adapter evaluations
  `0`; Mono not started; Sheet E/F/G blank; Hugging Face blocked.

## T2X b4 terminal-valid; b5 starts — 11:01 SAST

- Kombuys T2X Stage-B `b4` completed all frozen `1932/1932` steps and is
  terminal-valid, taking seed-42 progress to `8/11`. Final epoch-4 chrF
  `45.69104321766449` improves over checkpoint 1449 and retains checkpoint
  1932. Its artifact has exact 64-row Xhosa coverage, no literal or
  whitespace-only empty raw predictions, and `62` unique outputs. Final
  debug, state, adapter, and adapter-config SHA-256 values are
  `15f4d344af75e0c749af892eef2b3ddcf300be4a86289248c65388235b8bc317`,
  `cb2174e4d5b56ae439a9c8b6b59a1a6f82d7b81ca69dd4e024ec1f36c3732211`,
  `5eab549815bc062c0ae3ac4fab0318537c8b187b5eee1b64d9b6f55c83755551`,
  and `56c9f34119b06f98dd559d322d3527b59d34a2f9aeaecbefc6ba608776d6e414`.
  No cross-candidate ranking occurred.
- After b4 terminated and assigned GPU 1 was verified idle, preregistered T2X
  Stage-B `b5` started in tmux `sallm-t2x-hpo-b5`. It uses LR
  `0.0000517158347654887`, rank/alpha `16/32`, dropout
  `0.04221685230731964`, warmup `0.0861762648820877`, seed 42, BF16, and
  frozen `1932`-step batch semantics. The first invocation verified all `694`
  immutable files but failed before model/data/metric access because this new
  shell lacked W&B authentication. Its output and log are preserved unchanged
  under `failure_provenance/t2x_b5_seed42_attempt0_wandb_auth_20260813T110116`;
  manifest/trial/log hashes are
  `ba63e13539c0fd77bd3a0f7fbc467d60bd0ce1888e840c12df20129912510e4f`,
  `26c9f8307159bc3c03f98785fc70060ac36350ebd83a8a6cc01af6cb5e410f3b`,
  and `e7a60c6a8724e525100852f2afe67d620f624c3794f3e0a14f0cf976e8430597`.
  This operational failure exposed no selection evidence.
- The environment-only retry adds `WANDB_MODE=offline`, matching the preceding
  Kombuys trials, and otherwise leaves the frozen candidate unchanged. It runs
  in tmux `sallm-t2x-hpo-b5-r1`, verified all `694` immutable files, loaded
  the canonical model, verified exact `3,859/460` train/validation rows, and
  entered training. Retry manifest/trial hashes are
  `181bfc5650c7e31d7675a62e22c88518f195dcbf0b937fff48b8e2cd7f3e3303`
  and `1eeb76c578c853f02b21408a36850a9c7fe917bdd78dcfdfc7e2d1c973ffc46e`.
  Conditional first-artifact and terminal ETAs are about `11:25/12:23 SAST`.
  Assigned GPU 1 remains isolated; foreign GPU 0 remains occupied and
  untouched.
- HEX NER b5/b6/b7 jobs `1226719/1227987/1230086` remain exactly three
  healthy A100-40GB runs on `srvrocgpu010`, elapsed
  `09:21:39/06:58:33/00:28:27`. B5/b6 are in epoch-8/6 corrected callbacks at
  steps `4328/3246` after health-only losses
  `0.3844813180235682/0.7173408253042228`; their artifacts are incomplete, so
  no metric or patience decision was made. B7 is healthy near step `528/8115`.
  HEX quota is home `52.0%`, scratch `38.1%`, with no owned A100-80GB/L40S
  work.
- Trusted base `16/16`; NER seed-42 terminal-valid `8/11`; T2X seed-42
  terminal-valid `8/11`; frozen winners `0/8`; held-out adapter evaluations
  `0`; Mono not started; Sheet E/F/G blank; Hugging Face blocked.

## NER b4 terminal-valid; b7 starts; T2X b4 improves — 10:33 SAST

- NER Stage-B `b4` job `1224858` completed `0:0` after frozen early stopping
  at epoch 11 step 5951. Final Tsn/Xho/Zul F1 is
  `0.5824749163879099/0.581411451398086/0.593915462622224`, exact mean
  `0.5859339434694067`. The gain over epoch 10 is below the frozen `0.001`
  threshold, so it is the second consecutive patience miss and terminates the
  run; Trainer's raw-greater comparison records checkpoint 5951 as the final
  retained best. Coverage is exact `192` and `64/language`, with no literal
  empty raw predictions or parse failures, `64` whitespace-only raw
  predictions, and Tsn/Xho/Zul unique counts `37/54/34`. Final debug,
  retained-state, final-adapter, and adapter-config SHA-256 values are
  `2ecdf302630e207b61f2dc2d324d47e1bb43e0d260f8ef31b9f06ddb3ebaa5af`,
  `a5d566353c70b63ca33850f05d78ad512516771e5ab728721b71f9a4f51c6a60`,
  `4cef1566b5f21eb7cb948fdf14224c0a196f08bb7974992f4266e5d52538d4d4`,
  and `86931f42b6c9987137dec42a6fb8e51595841d93232c45a5dff4f603a03288d8`.
  NER seed-42 terminal-valid progress is now `8/11`; no ranking occurred.
- The released HEX slot started preregistered NER Stage-B `b7` as job
  `1230086`. It uses LR `0.0001223079850011719`, rank/alpha `32/64`, dropout
  `0.008055734634399415`, warmup `0.0765941160917282`, seed 42, and the
  frozen registry SHA-256
  `8fdd6ea5f192a1d2ef2267353162b9a640ccd09ae0d889cb5a999645198bb726`.
  Slurm verifies `nlpgroup/a100/nlpgroup`, `gpu:ampere:1`, `24:00:00`, eight
  CPUs, one node, and the canonical chdir. The run verified all `694`
  immutable files, loaded the canonical model, verified exact `4,323/10,760`
  train/validation rows, and entered training on `srvrocgpu010`; manifest/trial
  hashes are
  `9c35e22ea3544065957230ac1357feddd0e2eb0d437d0b56bf65c4f68cbc3448`
  and `43c483cd5c6b00b1311e53a9c49ae0a96e9f7f07f23b300f751bc546fa966a6c`.
  Conditional first-artifact ETA is about `11:45 SAST`; early-stopped/maximal
  terminal ETA is roughly `23:00--05:30 SAST`.
- NER b5/b6 jobs `1226719/1227987` improved at epoch-7/5 steps `3787/2705`
  to exact mean F1 `0.5352507187972697/0.3471549886942178`. Their respective
  Tsn/Xho/Zul values are
  `0.5326446828234171/0.5120478709194352/0.5610596026489567` and
  `0.3547816623588127/0.3146560735939288/0.372027230129912`. Both reset
  patience and retain their new checkpoints. Coverage is exact `192` and
  `64/language`, with no literal empties or parse failures and whitespace-only
  totals `50/62`. Debug/state hashes are
  `6ed226fcf655d2be68d74fab8460ac30defedb414fac91e4815379632f163562`/
  `98e0bb645ef0b079d891d03d5464151dbc9573ae15fa8c8d99afe6c49d2dcfe2`
  and
  `090b34cb4d2e9da14dbb665a5d7dd749f233716cf8fb31c2c2d1a39986f0404e`/
  `ab3b9385f9e84572ad0f853a4bbef76b3655aec748ad71c2d620179372ff6eb1`.
- Kombuys T2X b4 improved epoch-2/3 chrF to
  `43.83300504241281/45.204596233784045`, retaining checkpoint 1449. Each
  artifact has exact 64-row Xhosa coverage, no empty raw predictions, and
  `62/61` unique outputs. Epoch-2 debug and epoch-3 debug/state hashes are
  `1a86e2612f071fc0bc4366397cdca9b8874463bd5e83f5aec864a713d7ed0995`,
  `295e03fe68b2f9ca3150c96e8d9946db485452a51bf9691455502357e38c48ea`,
  and `e807bf2fcef496145b21d18ea51165533a71896595e175641cbb8e6533f5da87`.
  It is healthy in epoch 4, with terminal ETA about `10:50 SAST`; assigned
  GPU 1 is isolated and foreign GPU 0 remains occupied and untouched.
- HEX has exactly jobs `1226719/1227987/1230086` on three A100-40GB GPUs,
  with no owned A100-80GB/L40S work. Quota is home `52.0%`, scratch `38.1%`.
  Kombuys root has `22 GB` free and scratch `2.1 TB` free. Trusted base
  `16/16`; NER terminal-valid `8/11`; T2X terminal-valid `7/11`; frozen
  winners `0/8`; held-out adapter evaluations `0`; Mono not started; Sheet
  E/F/G blank; Hugging Face blocked.

## T2X b4 first artifact valid; HEX callbacks continue — 10:01 SAST

- Kombuys T2X Stage-B `b4` produced a valid initial epoch-1 step-483 chrF
  `37.277711104727736`, retaining checkpoint 483. Its artifact has exact
  64-row Xhosa coverage, no literal or whitespace-only empty predictions,
  and `63` unique raw outputs. Debug/state SHA-256 values are
  `4227b2c59dd8d9f83f7522916bb090453eafbb244ea4f614dd7436d81d228668`
  and `6ed5208c5089000d0c8a56aa7c61d84558368d91e6d0f95aa8b1066320252b7e`.
  This is within-candidate validation evidence only; no T2X candidate ranking
  occurred. The run is healthy near `718/1932`, with terminal ETA about
  `10:50 SAST`. Assigned Kombuys GPU 1 is isolated; foreign GPU 0 remains
  occupied and untouched. Kombuys root has `23 GB` free and scratch `2.1 TB`
  free.
- HEX NER b4/b5/b6 jobs `1224858/1226719/1227987` remain exactly three
  healthy A100-40GB runs on `srvrocgpu010`, elapsed
  `13:29:01/08:21:39/05:58:33`. They are in epoch-11/7/5 corrected callbacks
  at steps `5951/3787/2705` after health-only losses
  `0.3986847037276371/0.3807423304447897/0.8609939064678206`. Their new
  debug artifacts are incomplete, so no metric, checkpoint, or patience
  decision was made. B7 stays unsubmitted. HEX quota is home `52.0%`,
  scratch `38.0%`; no owned A100-80GB/L40S work is active.
- Trusted base `16/16`; NER seed-42 terminal-valid `7/11`; T2X seed-42
  terminal-valid `7/11`; frozen winners `0/8`; held-out adapter evaluations
  `0`; Mono not started; Sheet E/F/G blank; Hugging Face blocked.

## T2X b3 terminal-valid; b4 starts; NER b5/b6 improve — 09:33 SAST

- Kombuys T2X Stage-B `b3` completed all `1932/1932` frozen steps and is
  terminal-valid, taking seed-42 progress to `7/11`. Epoch-3 step-1449 chrF
  `46.10443672013474` remains the retained best after final epoch-4 chrF
  `46.08058576186493`. Both exact 64-row Xhosa artifacts have no literal or
  whitespace-only empty predictions and `62` unique raw outputs. Epoch-3/
  final debug, retained-state, final-adapter, and adapter-config SHA-256 are
  `eab6ce22023bb95433e28357189e3158499214e39d043adaacfc3b75d67780ea`,
  `4690290f68682224365a94b534dcb90730c1f67bae613610340a08e662e32871`,
  `51a6bb55e7741db21c0f70c41bd59f15da7db43b3bdda58f1f2e3fd5966d6477`,
  `b8d96f0cfa09bea6c2e66a3cbca4cdef485ce0813f336c5a44876d01af0730c7`,
  and `6ea6aea98f48350a5601bcedec34522b34572de5f3eb77a19479c79251e74117`.
  No cross-candidate ranking occurred.
- After b3 terminated and assigned GPU 1 was verified idle, preregistered T2X
  Stage-B `b4` started in tmux `sallm-t2x-hpo-b4`. It uses LR
  `0.00008981661817441004`, rank/alpha `8/16`, dropout
  `0.0971165418624878`, warmup `0.04982422709465027`, seed 42, BF16, and
  frozen `1932`-step batch semantics. It verified all `694` immutable files
  and exact `3,859/460` train/validation rows, then entered training.
  Execution-manifest/trial hashes are
  `1f7dfc9adbea6ecb6b56b83528ca273ba1e722329196f3cb3d5833059796bba0`
  and `82f5e2014fc34aeb369a794b73586743cd44d13d0202a34d6316e0997b52fc27`.
  Conditional first-artifact/terminal ETAs are about `09:52/10:50 SAST`.
  GPU 1 is isolated; foreign GPU 0 remains occupied and untouched.
- HEX NER b5 job `1226719` improved epoch-6 step 3246 to exact mean F1
  `0.5095075141003677`, from Tsn/Xho/Zul
  `0.5004326863641485/0.4958077709610952/0.532282084975859`. NER b6 job
  `1227987` improved epoch-4 step 2164 to `0.2797155538913259`, from
  `0.28363801521512083/0.2518963922293675/0.3036122542294894`. Both gains
  reset patience and retain their new checkpoints. Each artifact has exact
  `192` rows and `64/language`, no literal empties or parse failures, and
  whitespace-only totals `59/55`. Debug/state hashes are
  `0bd8fcdd...8afbf`/`a49dbfea...e812b` and
  `07cd2107...12806`/`dbba0239...ce69e`.
- NER b4 job `1224858` is in its step-5951 corrected callback after
  health-only loss `0.3986847037276371`; its artifact is incomplete, so no
  second patience decision was made. Jobs `1224858/1226719/1227987` remain
  exactly three healthy A100-40GB jobs on `srvrocgpu010`; b7 stays
  unsubmitted. HEX quota is home `52.0%`, scratch `38.0%`; no owned
  A100-80GB/L40S work is active.
- Trusted base `16/16`; NER seed-42 terminal-valid `7/11`; T2X seed-42
  terminal-valid `7/11`; frozen winners `0/8`; held-out adapter evaluations
  `0`; Mono not started; Sheet E/F/G blank; Hugging Face blocked.

## T2X b3 improves; NER b4 records first patience miss — 09:00 SAST

- Kombuys T2X Stage-B `b3` improved epoch-2 step 966 from chrF
  `38.65717411733255` to `45.124063374893765`, retaining checkpoint 966.
  Its exact 64-row Xhosa artifact has no literal or whitespace-only empty
  predictions and `61` unique raw outputs; debug/state hashes are
  `ed8c3f7880a77957f4687a342d3f20bf9492b0bf13a60cad8efd871e0bb78ad0`
  and `8fbb32ef3d0aa14f90ce0633dba3a09b8542d21fdc14477b81ad655536e14d7a`.
  This is within-candidate validation evidence only; no cross-candidate
  ranking occurred. Epoch-3 step-1449 corrected validation is active, so no
  decision was made from partial output. The run is healthy on assigned
  Kombuys GPU 1, with terminal ETA about `09:25 SAST`; foreign GPU 0 remains
  occupied and untouched.
- HEX NER b4 job `1224858` recorded epoch-10 step-5410 exact mean F1
  `0.57713788562457324`, from Tsn/Xho/Zul
  `0.5678358702845298/0.5687995319397408/0.5947782546494493`. This is below
  checkpoint-4869 best `0.5858902577772452`, so frozen patience advances to
  `1/2` and checkpoint 4869 remains retained. Coverage is exact `192` and
  `64/language`, with no literal empties or parse failures, `57`
  whitespace-only predictions, and unique counts `38/55/38`. Debug/state
  hashes are
  `b28e56ad5dc7489ed54d87ae84182e00f313fd845d0376f83d75b25cbecab5aa`
  and `b41cb64e02e506f8c0235b3304f478f21dde3c022030f0a58dd91b4fb49e80af`.
- NER b5/b6 jobs `1226719/1227987` remain in their step-3246/2164 corrected
  callbacks after health-only losses `0.3808601918273699/1.015368629654101`;
  their new debug artifacts are incomplete, so no selection decision was
  made. All three HEX jobs remain healthy A100-40GB runs on `srvrocgpu010`;
  b7 stays unsubmitted. Quota is home `52.0%`, scratch `38.0%`, with no owned
  A100-80GB/L40S work.
- Trusted base `16/16`; NER seed-42 terminal-valid `7/11`; T2X seed-42
  terminal-valid `6/11`; frozen winners `0/8`; held-out adapter evaluations
  `0`; Mono not started; Sheet E/F/G blank; Hugging Face blocked.

## T2X b3 first artifact valid; NER callbacks active — 08:30 SAST

- Kombuys T2X Stage-B `b3` produced a valid initial epoch-1 step-483 chrF
  `38.65717411733255`, retaining checkpoint 483. Its artifact has exact
  64-row Xhosa coverage, no literal or whitespace-only empty predictions,
  and `64` unique raw outputs. Debug/state hashes are
  `453cb3cfc0b5642a32b73217d93919712b5720f1d28b4888a7594f0e8f50cabf`
  and `fdfed85b996c1a6154512fc3f1c7d8236cbae7f9518b569f4b03756f6c0b2558`.
  This is within-candidate validation evidence only; no T2X candidate ranking
  occurred. The run is healthy near `663/1932`, with next-artifact/terminal
  ETAs around `08:45/09:25 SAST`. Assigned GPU 1 is isolated; foreign GPU 0
  remains occupied and untouched. Kombuys root has `23 GB` free and scratch
  `2.1 TB` free.
- HEX NER b4/b5/b6 jobs `1224858/1226719/1227987` remain exactly three
  healthy A100-40GB jobs on `srvrocgpu010`, elapsed
  `11:59:09/06:51:47/04:28:41`. They entered epoch-10/6/4 corrected
  callbacks at steps `5410/3246/2164` after health-only losses
  `0.39288670422862454/0.3808601918273699/1.015368629654101`. Their new debug
  artifacts are incomplete, so no metric, checkpoint, or patience decision
  was made. B7 stays unsubmitted. HEX quota is home `52.0%`, scratch `38.0%`;
  no owned A100-80GB/L40S work is active.
- Trusted base `16/16`; NER seed-42 terminal-valid `7/11`; T2X seed-42
  terminal-valid `6/11`; frozen winners `0/8`; held-out adapter evaluations
  `0`; Mono not started; Sheet E/F/G blank; Hugging Face blocked.

## T2X b2 terminal-valid; b3 starts; NER b4/b5/b6 improve — 08:05 SAST

- Kombuys T2X Stage-B `b2` completed all `1932/1932` frozen steps and is
  terminal-valid, taking seed-42 progress to `6/11`. Final chrF
  `40.32726992706406` improved over epoch 2 and retains checkpoint 1932. The
  final artifact has exact 64-row Xhosa coverage, no literal or
  whitespace-only empty predictions, and `64` unique raw outputs. Final
  debug/state/adapter/config SHA-256 values are
  `3654f374c26cf3e96143dc3ab3398e86ad800afeeb30cb599169321deee2f5e8`,
  `33452a28e48095cc7fe8b786d2763e77b7748629c5c923e3b94dafb3fb991ba1`,
  `86baa82d51c6136e462fb30c1db1d62770b70c12bb151ba15e03aa8bd80e7e02`,
  and `7973974e0fb2c0887a309aef2aac8bb941f0d07e289b6ffe5ddffae2e235409f`.
  No cross-candidate ranking occurred.
- After the assigned GPU was verified idle, preregistered T2X Stage-B `b3`
  started in tmux `sallm-t2x-hpo-b3`. It uses LR
  `0.00007207193245743205`, rank/alpha `16/32`, dropout
  `0.06486568450927735`, warmup `0.04297354072332382`, seed 42, BF16, and
  the frozen `1932`-step batch semantics. It verified all `694` immutable
  source/config files and exact `3,859/460` train/validation rows, then
  entered training. Execution-manifest/trial hashes are
  `99a4d9d8eb7255577728b3ea97205cc668b5a9603b80266561d4787079bf327e`
  and `8156ee051304fb5a6c0334424cdb204f9ebc8b16de7bcfadc33d651d4c045918`.
  Conditional first-artifact/terminal ETAs are about `08:25/09:25 SAST`.
  GPU 1 is isolated; foreign GPU 0 remains occupied and untouched.
- HEX NER b4/b5/b6 jobs `1224858/1226719/1227987` improved at steps
  `4869/2705/1623` to exact mean F1
  `0.5858902577772452/0.4874091148048638/0.2044077624651216`, resetting
  patience and retaining those checkpoints. Per-language Tsn/Xho/Zul F1 is
  `0.585213373199189/0.5702515177796555/0.6022058823528912`,
  `0.48686340430825553/0.46482737201713514/0.5105365680892009`, and
  `0.20095963872419936/0.19265580553395673/0.21960784313720869`.
  Every artifact has exact `192` rows and `64/language`, no literal empties
  or parse failures, and whitespace-only totals `60/56/20`. Debug/state
  hashes are `90daa54f...12a66`/`3b430463...e0d36`,
  `88d199e1...8bea1`/`97ff88f8...6e85`, and
  `12a5f4d7...f7a2a1`/`3fa2e448...bc319`.
- The same three jobs remain healthy on `srvrocgpu010` near
  `5290/2899/1937` steps. B7 stays unsubmitted. HEX quota is home `52.0%`,
  scratch `38.0%`; no owned A100-80GB/L40S work is active. Trusted base
  `16/16`; NER terminal-valid `7/11`; T2X terminal-valid `6/11`; frozen
  winners `0/8`; held-out adapter evaluations `0`; Mono not started; Sheet
  E/F/G blank; Hugging Face blocked.

## T2X b2 improves; three NER callbacks active — 07:30 SAST

- Kombuys T2X `b2` improved epoch-2 step 966 from chrF
  `31.31387618368498` to `37.9190530917995`, retaining checkpoint 966. Its
  exact 64-row Xhosa artifact has no literal or whitespace-only empty
  predictions and `64` unique raw outputs; debug/state hashes are
  `5b2c4078f3a7c7f6f8fb7ddf03c192b1a48f6209bc4f1ec6751357f3049a1102`
  and `7f2a29182a87d0cf0b33dcd064e655f25ddb3987d630989cbffcd6b66f55f2be`.
  This is within-candidate evidence only; no T2X ranking occurred. Epoch-3
  step-1449 corrected generation is active after health-only loss
  `2.5125199027683425`, so no decision was made from that incomplete callback.
  The run is healthy on assigned GPU 1, with terminal ETA about `08:00 SAST`;
  foreign GPU 0 remains occupied by its existing workload and untouched.
- HEX NER b4/b5/b6 jobs `1224858/1226719/1227987` remain exactly three
  healthy A100-40GB jobs on `srvrocgpu010`, elapsed
  `10:58:51/05:51:29/03:28:23`. B4/b5/b6 are in epoch-9/5/3 corrected
  callbacks after health-only losses
  `0.3814328558826092/0.38187587128252787/1.2016226729495818`. Their new
  debug artifacts are incomplete, so no metric or patience decision was made.
  No fault marker is present and b7 remains unsubmitted. HEX quota is home
  `52.0%`, scratch `38.0%`; no owned A100-80GB/L40S work is active.
- Trusted base `16/16`; NER seed-42 terminal-valid `7/11`; T2X seed-42
  terminal-valid `5/11`; frozen winners `0/8`; held-out adapter evaluations
  `0`; Mono not started; Sheet E/F/G blank; Hugging Face blocked.

## NER b4/b5/b6 improve; T2X b2 first artifact valid — 07:00 SAST

- NER `b4` job `1224858` improved epoch-8 step 4328 to exact mean F1
  `0.5711115998283094`, from Tsn/Xho/Zul
  `0.5701800847457127/0.5578402134672238/0.5853145012719916`. Its gain over
  epoch 7 exceeds the frozen threshold, resets patience, and retains
  checkpoint 4328. Coverage is exact `192` and `64/language`, with no literal
  empties or parse failures, `60` whitespace-only predictions, and unique
  counts `38/54/36`. Debug/state hashes are
  `bf22345c17439c0a4c39e26ec6ce22c6114e3e69cc5b3bfb111dc660479b6de8`
  and `40f4b8167ceba9fb4c39ebd9c9678b033c32c748c31667c3b0f2baad84b44362`.
- NER `b5` job `1226719` improved epoch-4 step 2164 to exact mean F1
  `0.4162427285350683`, from Tsn/Xho/Zul
  `0.42235307701184327/0.3977921583554887/0.4285829502378729`. Its gain over
  epoch 3 exceeds the frozen threshold, resets patience, and retains
  checkpoint 2164. Coverage is exact `192` and `64/language`, with no literal
  empties or parse failures, `58` whitespace-only predictions, and unique
  counts `40/57/36`. Debug/state hashes are
  `07b8942a25ca37f320d947c69e5d9492b2053609b1f944ba922767f94882f2db`
  and `f8625b531418f3f478142ba90564efce1b3af4d991e9709294cbe7cb9fa29932`.
- NER `b6` job `1227987` improved epoch-2 step 1082 to exact mean F1
  `0.19493532967233085`, from Tsn/Xho/Zul
  `0.198307134220023/0.1749649199205965/0.21153393487637298`. Its gain over
  epoch 1 exceeds the frozen threshold, resets patience, and retains
  checkpoint 1082. Coverage is exact `192` and `64/language`, with no literal
  empties, `47` whitespace-only predictions, one explicitly retained parse
  failure, and unique counts `49/55/39`. Debug/state hashes are
  `d363a3df42d90f6333f513a281d1f6ec3b53b1c7f0e76d6f52667a7fb50d389a`
  and `3db07dcfdbdb0eccdcf63ee891c26bbbd05e5c436a1bcf3ee272abb4486955b3`.
- Jobs `1224858/1226719/1227987` remain exactly three healthy A100-40GB jobs
  on `srvrocgpu010`, near steps `4328/2546/1602`. No fault marker is present;
  b7 remains unsubmitted. Their next complete validation artifacts are
  conditionally expected around `07:40/07:55/07:45 SAST`. HEX quota is home
  `52.0%`, scratch `38.0%`; no owned A100-80GB/L40S work is active.
- Kombuys T2X `b2` produced a valid initial epoch-1 step-483 chrF
  `31.31387618368498`, retaining checkpoint 483. Its exact 64-row Xhosa
  artifact has no literal or whitespace-only empty predictions and `63`
  unique raw outputs; debug/state hashes are
  `94457434917c85ac75137b4217b0db5f22ace3e9a74790b2418252df76002e18`
  and `43701ea5dbf5930ce53a9de0a8a4c9c3fdb7876cac786453fdd376db9c57d3c3`.
  This is within-candidate evidence only; no T2X ranking occurred. The run is
  healthy near `719/1932`, with terminal ETA about `07:55 SAST`; assigned GPU
  1 is isolated and foreign GPU 0 remains untouched. Kombuys root has `24 GB`
  free and scratch `2.1 TB` free.
- Trusted base `16/16`; NER seed-42 terminal-valid `7/11`; T2X seed-42
  terminal-valid `5/11`; frozen winners `0/8`; held-out adapter evaluations
  `0`; Mono not started; Sheet E/F/G blank; Hugging Face blocked.

## Deferred cross-architecture extension

- After the pure-GDN downstream program is complete and frozen, run this
  scientifically clean validation-only HPO design for every other model
  architecture in scope. The corrected pure-GDN runs show encouraging,
  auditable optimization behavior, so this is a planned follow-up study.
- This extension is not active now. Do not allocate jobs or change the
  pure-GDN schedule for it. Preregister each architecture's search space,
  metric, host/GPU assignment, provenance, and confirmation seeds separately;
  keep held-out tests untouched until that architecture's winner is frozen.
  Do not assume pure-GDN hyperparameters transfer across architectures.

## T2X b1 terminal-valid; b2 starts — 06:31 SAST

- Kombuys T2X Stage-B `b1` completed all `1932/1932` frozen training steps
  and is terminal-valid, taking seed-42 progress to `5/11`. ChrF improved to
  `43.22297087877961` at epoch-3 step 1449 and final-best
  `43.70046004673441` at epoch-4 step 1932, so checkpoint 1932 and the final
  adapter are retained. Both artifacts have exact 64-row Xhosa coverage, no
  literal or whitespace-only empty predictions, and `61` unique raw outputs.
  Epoch-3/final debug, terminal-state, final-adapter, and adapter-config
  SHA-256 values are
  `d91448f636615d8b83f5fae8fae5d4385f9b2ea8259cd18ff2f964acc91c60ca`,
  `c7bc6c5d0543ff69ec645b1170773f7957ff57a6941f801ff9b343a131ab20dd`,
  `61d1efbd395411f9158de3868c05f9571234c9896ac956f2da0c3eb4816269ff`,
  `865465c00be142055ac47758715e6d92a28758770053706a6b3b56eb3ffdf4e3`,
  and `3b3cca8b815b497e06dabb961035e55ef9d023373619e10f18a681aee30102e1`.
  No cross-candidate ranking occurred.
- After b1 terminated and assigned GPU 1 was verified idle, preregistered T2X
  Stage-B `b2` started in tmux `sallm-t2x-hpo-b2`. It uses LR
  `3.617950373518279e-05`, rank/alpha `8/16`, dropout
  `0.019731278717517856`, warmup `0.09582568347454071`, seed 42, frozen batch
  semantics, BF16, and `1932` steps. All `694` immutable files and exact
  `3,859/460` train/validation rows verified; execution-manifest/trial hashes
  are `5929088d56c5172de95b6e4d055af13809c5a6800855e4e11d941b9ebeb039c6`
  and `3616194322c7de88a167881b4397ab42cd96e1c706547705f92f123efe68daba`.
  It entered training at 06:30 on GPU 1; conditional terminal ETA is about
  `07:55 SAST`. Foreign GPU 0 remains occupied by its existing workload and
  untouched. Kombuys root has `23 GB` free and scratch `2.1 TB` free.
- HEX NER jobs `1224858/1226719/1227987` remain exactly three healthy
  A100-40GB jobs on `srvrocgpu010`, elapsed
  `09:58:59/04:51:37/02:28:31`. Their epoch-8/4/2 corrected callbacks remain
  active with no complete new artifacts or fault markers, so no decision was
  made and b7 remains unsubmitted. HEX quota is home `52.0%`, scratch
  `38.0%`; no owned A100-80GB/L40S work is active.
- Trusted base `16/16`; NER seed-42 terminal-valid `7/11`; T2X seed-42
  terminal-valid `5/11`; frozen winners `0/8`; held-out adapter evaluations
  `0`; Mono not started; Sheet E/F/G blank; Hugging Face blocked.

## T2X b1 improves; three NER callbacks active — 05:59 SAST

- Kombuys T2X `b1` improved epoch-2 step 966 from chrF
  `36.10345122148267` to `42.91101817994199`, retaining checkpoint 966. Its
  exact 64-row Xhosa artifact has no literal or whitespace-only empty
  predictions and `60` unique raw outputs; debug/state hashes are
  `cefdf05cca12b709b14f183b5026b2a572f48c45de833ebf38bd5f16c4e8d431`
  and `077383bfafc9238225955bc4ed942e043f9af37bd4647ab3da61f351c0a4a562`.
  This is within-candidate evidence only; no T2X ranking occurred. Epoch-3
  step-1449 corrected generation is active after health-only loss
  `2.355991264011549`, so no decision was made from that incomplete callback.
  The run is healthy on assigned GPU 1, with terminal ETA about `06:25 SAST`;
  foreign GPU 0 remains occupied by its existing workload and untouched.
- HEX NER b4/b5/b6 jobs `1224858/1226719/1227987` remain exactly three
  healthy A100-40GB jobs on `srvrocgpu010`, elapsed
  `09:29:06/04:21:44/01:58:38`. B4 epoch-8 step 4328 and b6 epoch-2 step 1082
  entered corrected callbacks after health-only losses
  `0.37415863944695343/1.3581883540383946`; b5 is likewise in its epoch-4
  callback. Their new debug artifacts are not complete, so no metric or
  patience decision was made. No fault marker is present and b7 remains
  unsubmitted. HEX quota is home `52.0%`, scratch `38.0%`; no owned
  A100-80GB/L40S work is active.
- Trusted base `16/16`; NER seed-42 terminal-valid `7/11`; T2X seed-42
  terminal-valid `4/11`; frozen winners `0/8`; held-out adapter evaluations
  `0`; Mono not started; Sheet E/F/G blank; Hugging Face blocked.

## NER b4/b5/b6 artifacts valid; T2X b1 first artifact valid — 05:31 SAST

- NER `b4` job `1224858` improved epoch-7 step 3787 to exact mean F1
  `0.5692895173118145`, from Tsn/Xho/Zul
  `0.5705770793822312/0.5448822316291699/0.5924092409240425`. The gain over
  epoch 6 exceeds the frozen threshold, resets patience, and retains
  checkpoint 3787. Coverage is exact `192` and `64/language`, with no literal
  empties or parse failures, `53` whitespace-only predictions, and unique
  counts `40/56/38`. Debug/state hashes are
  `d9bc5d6c5c52e5e9f46b4dbab7abf7d7e6f5191de79eb1b6f397d7d060b790c5`
  and `e9fc1493c7aed60387407ceec7bdbcbbf4dc099e7f3eef1a5d425dda744529e4`.
- NER `b5` job `1226719` improved epoch-3 step 1623 to exact mean F1
  `0.3072057532693912`, from Tsn/Xho/Zul
  `0.3088094783106745/0.28816116797841923/0.32464661351907975`. The gain over
  epoch 2 exceeds the frozen threshold, resets patience, and retains
  checkpoint 1623. Coverage is exact `192` and `64/language`, with no literal
  empties or parse failures, `54` whitespace-only predictions, and unique
  counts `44/59/37`. Debug/state hashes are
  `b9640ba079cebcf8e8c9ce296b4b8448806c90b7296df1a38b37a3ab2410f9c7`
  and `c1efb382eb421ab9ceeddfa363bce4cf7edc9ffcafce4e51295394a05338182c`.
- NER `b6` job `1227987` produced its valid initial epoch-1 step-541 exact
  mean F1 `0.1216232572664236`, from Tsn/Xho/Zul
  `0.12503745131324592/0.0906416097190086/0.14919071076701632`.
  Checkpoint 541 is retained and patience is reset. Coverage is exact `192`
  and `64/language`, with no literal empties or parse failures, `61`
  whitespace-only predictions, and unique counts `52/44/33`. Debug/state
  hashes are `02c9efe87834304352650096b9bcc4170b9ba7c03347d4701f3ae3e4c02cb082`
  and `1a277f7d2b85600ae865adab6692f8973a571307168eb8a99b13b593b79d64fc`.
- Jobs `1224858/1226719/1227987` remain exactly three healthy A100-40GB jobs
  on `srvrocgpu010`, near steps `4053/1730/782`. No fault marker is present;
  b7 remains unsubmitted. Their next complete validation artifacts are
  conditionally expected around `06:15/06:30/06:25 SAST`. HEX quota is home
  `52.0%`, scratch `38.0%`; no owned A100-80GB/L40S work is active.
- Kombuys T2X `b1` produced a valid initial epoch-1 step-483 chrF
  `36.10345122148267`, retaining checkpoint 483. Its exact 64-row Xhosa
  artifact has no literal or whitespace-only empty predictions and `64`
  unique raw outputs; debug/state hashes are
  `97d6084ba5a7ee5a29265c75c929167cd4bfd33ee9d44d2590e3796cae8912a1`
  and `e6f2b620f25ca8e273e802ca226b91bc9f5515dab2f2f9b2117ec38596fcaab6`.
  This is within-candidate evidence only; no T2X ranking occurred. The run is
  healthy near `732/1932`, with terminal ETA about `06:30 SAST`; assigned GPU
  1 uses `8,130 MiB`. Foreign GPU 0 remains occupied by its existing workload
  and untouched. Kombuys root has `24 GB` free and scratch `2.1 TB` free.
- Trusted base `16/16`; NER seed-42 terminal-valid `7/11`; T2X seed-42
  terminal-valid `4/11`; frozen winners `0/8`; held-out adapter evaluations
  `0`; Mono not started; Sheet E/F/G blank; Hugging Face blocked.

## T2X b0 terminal-valid; b1 starts — 05:01 SAST

- Kombuys T2X Stage-B `b0` completed all `1932/1932` frozen training steps
  and is terminal-valid, taking seed-42 progress to `4/11`. Epoch-3 step 1449
  is the retained best with chrF `47.56264117884068`; final epoch-4 chrF is
  `47.55999905312901`. Both artifacts have exact 64-row Xhosa coverage and
  no literal or whitespace-only empty predictions; their unique raw-output
  counts are `61/63`. Epoch-3/final debug, retained-state, final-adapter, and
  adapter-config SHA-256 values are
  `ff52d0618a18a100115e87c9c9af23c6c80b78350eaee7f0f2b78dc6f6708a44`,
  `342791db72d816ac54fd97b0457f8bd94e798bf3fc132d3d8d67ac80dee096f5`,
  `5d0d9194c0b7eb1424571779ec5b41fa2959df0e91306016f4002f11c6fca4ac`,
  `705bd44de6c8933d7233fe0a0d261a908f00404670c57ae20bfe189e8baf1b46`,
  and `79146ef4310e3fe50bbb41bfac022db0e5d064056ead960c505cee40e97edb32`.
  No cross-candidate ranking occurred.
- After b0 terminated and assigned GPU 1 was verified idle, preregistered T2X
  Stage-B `b1` started in tmux `sallm-t2x-hpo-b1`. It uses LR
  `3.0329114402234482e-05`, rank/alpha `32/64`, dropout
  `0.08691746592521668`, warmup `0.029702768474817273`, seed 42, frozen
  batch semantics, BF16, and `1932` steps. All `694` immutable files and
  exact `3,859/460` train/validation rows verified; execution-manifest/trial
  hashes are
  `b154bcd57a6a50878082da380bb2bbad0f1471138e796d0ae2f1c07cb3a4b35e`
  and `bfca20710c128245a08da5e6b238e961c21c14e99cc7139efd3047ca391c4aa7`.
  The run entered training at 05:01, with GPU 1 using `2,766 MiB`; conditional
  terminal ETA is about `06:30 SAST`. Foreign GPU 0 remains untouched.
- HEX NER jobs `1224858/1226719/1227987` remain exactly three healthy
  A100-40GB jobs on `srvrocgpu010`, at elapsed times
  `08:31:19/03:23:57/01:00:51`. B4/b5/b6 are in epoch-7/3/1 corrected
  callbacks after health-only losses
  `0.3728755738212273/0.5045523831392309/1.6143718960559945`; their next
  complete artifacts remain absent, so no decisions were made. B7 remains
  unsubmitted. HEX quota is home `52.0%`, scratch `38.0%`; no owned
  A100-80GB/L40S work is active.
- Trusted base `16/16`; NER seed-42 terminal-valid `7/11`; T2X seed-42
  terminal-valid `4/11`; frozen winners `0/8`; held-out adapter evaluations
  `0`; Mono not started; Sheet E/F/G blank; Hugging Face blocked.

## NER b4/b5 improve; b6 running; T2X b0 reaches 46.7406 — 04:29 SAST

- NER `b4` job `1224858` improved epoch-6 step 3246 to exact mean F1
  `0.5391488200979205666666666667`, from Tsn/Xho/Zul
  `0.5218896858767743/0.5322437932649269/0.5633129811520605`. Its
  `0.0043944614492383` gain resets patience and retains checkpoint 3246.
  Coverage is exact `192` and `64/language`, with no literal empties or parse
  failures, `55` whitespace-only predictions, and unique counts `40/55/37`.
  Debug/state hashes are
  `17f0d581103ba653495613ff6577d2665233d10e90ae20b354968c1835541771`
  and `d0d2383046d0be48eb99cca41bae04a6505f49efc3a9d4b200be684692f6aa6c`.
- NER `b5` job `1226719` improved epoch-2 step 1082 to exact mean F1
  `0.2718237001045021`, from Tsn/Xho/Zul
  `0.2738202973496592/0.2471243042671116/0.2945264986967355`. Its
  `0.1429776877644584633333333333` gain resets patience and retains checkpoint
  1082. Coverage is exact `192` and `64/language`, with no literal empties or
  parse failures, `60` whitespace-only predictions, and unique counts
  `40/58/34`. Debug/state hashes are
  `9950be4d87a1909f3e5013d16b41369f7585f3ff59954116aa57b03cf98a29e4`
  and `44fbbf16ae0b2c7d32cfaba29d36c72967cf3b60d2b9db005e181af3e6b3ad75`.
- NER `b6` job `1227987` is now running on `srvrocgpu010`, healthy near step
  531 in epoch 1. Its `694`-file execution manifest and trial hash are
  `83791021555af5e7228094470716dd3b4a667c1712c3ef50db0f266dcb50bdbe`
  and `6fb77d13b3f2b401c8fc478c03c03bfa31699c0b5f509cbd3c9247e29006f074`.
  Jobs `1224858/1226719/1227987` are exactly three healthy A100-40GB jobs;
  b7 remains unsubmitted and no A100-80GB/L40S work is owned. HEX quota is
  home `52.0%`, scratch `38.0%`.
- Kombuys T2X `b0` improved epoch-2 step 966 from chrF
  `40.382906319702634` to `46.74057155779831`, retaining checkpoint 966. Its
  exact 64-row Xhosa artifact has no literal or whitespace-only empty
  predictions and `62` unique raw outputs; debug/state hashes are
  `c5b60c9d71eb9cbb5e53a24b1ec84f8f81fd6d4b5396bfb50923201da8ae6ad5`
  and `37d52f7ee86d17739f295065c54784edd1d47d9b478a1692ef1c941e341f26d1`.
  Epoch-3 step-1449 corrected generation is active, so no further decision
  was made. Conditional terminal ETA is about `05:00 SAST`; assigned GPU 1
  is healthy and foreign GPU 0 remains untouched. Kombuys root has `24 GB`
  free and scratch `2.1 TB` free.
- Trusted base `16/16`; NER seed-42 terminal-valid `7/11`; T2X seed-42
  terminal-valid `3/11`; frozen winners `0/8`; held-out adapter evaluations
  `0`; Mono not started; Sheet E/F/G blank; Hugging Face blocked.

## NER b3 terminal-valid; b6 submitted; T2X b0 first artifact valid — 04:01 SAST

- NER Stage-B `b3` job `1222348` completed `0:0` after frozen early stopping
  at epoch 10 step 5410. Final Tsn/Xho/Zul F1 is
  `0.5968683651804172/0.5964388527561076/0.6111563149805124`, exact mean
  `0.6014878443056790666666666667`. This second consecutive miss correctly
  stops the run and retains epoch-8 checkpoint 4328 best
  `0.6027308342440767666666666667`. Final coverage is exact `192` and
  `64/language`, with no literal empties or parse failures, `57`
  whitespace-only predictions, and unique counts `40/54/37`. Final debug,
  retained-state, final-adapter, and adapter-config hashes are
  `33cb6d3f122303b5c8928afbacee52698066135b53616768f29281614bab7af5`,
  `2a520f87580693f285dccd5d878f8e883bd75126404ad62701a9d01fc76c5160`,
  `ad6ab2aaad2aa9dbad50e6763359ff00eea3c5e72dca3ce13df41215fa2859e5`,
  and `5f72ceb50ea7012db676f5e44f8a7a8a9a52bd8ee4a9bc1afa7db152afdb3182`.
  NER seed-42 terminal-valid progress is now `7/11`; no ranking occurred.
- After verifying two owned running A100-40GB jobs, no A100-80GB/L40S work,
  absent b6 output, and the frozen registry hash, preregistered NER Stage-B
  `b6` was submitted as job `1227987`. Slurm verifies account/partition/QOS
  `nlpgroup/a100/nlpgroup`, `gpu:ampere:1`, `24:00:00`, eight CPUs, one node,
  and the canonical chdir. It is `PENDING (Priority)`; owned HEX state is two
  running plus one pending job, so b7 remains unsubmitted.
- Kombuys T2X Stage-B `b0` produced a valid epoch-1 step-483 chrF
  `40.382906319702634`. Its artifact has exact 64-row Xhosa coverage, no
  literal or whitespace-only empty predictions, and `64` unique raw outputs;
  debug/state hashes are
  `8217da07107de2060bba38baecb745fc06cfdee1d397cbfe5c5456eed9fda1ae`
  and `9b789220e149d1a89096d0b654dca0eec7500e8edfdb26489d3605008f9a675e`.
  Checkpoint 483 is retained. This is within-candidate evidence only; no T2X
  ranking occurred. The run is healthy near `692/1932`, ETA about `04:50
  SAST`; GPU 1 uses about `5,484 MiB`, while foreign GPU 0 remains untouched.
- NER `b4` job `1224858` remains in its epoch-6 corrected callback after
  health-only loss `0.36709431361088524`; `b5` job `1226719` remains in its
  epoch-2 corrected callback after health-only loss `1.0214382469432504`.
  Their step-3246/1082 artifacts are absent, so no decisions were made.
  HEX quota is home `52.0%`, scratch `38.0%`; Kombuys root has `24 GB` free
  and scratch `2.1 TB` free.
- Trusted base `16/16`; NER seed-42 terminal-valid `7/11`; T2X seed-42
  terminal-valid `3/11`; frozen winners `0/8`; held-out adapter evaluations
  `0`; Mono not started; Sheet E/F/G blank; Hugging Face blocked.

## T2X a2 terminal-valid; Stage-B b0 starts — 03:32 SAST

- Kombuys T2X `a2` completed `1932/1932` steps and is terminal-valid. Final
  epoch-4 chrF `51.14499797288408` correctly retains epoch-3 checkpoint 1449
  best `51.306312454497196`. The final artifact has exact 64-row Xhosa
  coverage, no literal or whitespace-only empty predictions, and `64` unique
  raw outputs. Final debug, retained state, final adapter, and adapter-config
  hashes are `bef02bd51253f2c3a3446718a2e5215195c3699875894f59258321c33f3df6c0`,
  `d1f2ff9f48abe937d5a0dfe74649c3a1edefed9dd8600b99cd55d27c6d61867c`,
  `9f4991a1546d2d1ff1a22bde6ceb0a7ac5209509dd1ec8b54d7267d589204d4e`,
  and `edf6d27e52739b7af601f03838a2e9ea03be2aaa14dfcecc5c64971375824e0f`.
  T2X seed-42 terminal-valid progress is now `3/11`; no ranking occurred.
- After verifying GPU 1 idle and registry SHA-256
  `8fdd6ea5f192a1d2ef2267353162b9a640ccd09ae0d889cb5a999645198bb726`,
  preregistered T2X Stage-B `b0` was launched. The first tmux invocation
  exited before a process, output directory, manifest, model/data access, or
  metric because its log directory did not exist. The unchanged directory-only
  retry runs in tmux `sallm-t2x-hpo-b0-r1`; this failure provenance is
  preserved here and cannot influence selection.
- Active b0 uses LR `0.00015156541821567134`, rank/alpha `8/16`, dropout
  `0.028929591178894043`, warmup `0.061286738514900206`, seed 42, frozen
  batch semantics, BF16, and `1932` steps. It verified all `694` immutable
  files and exact `3,859/460` train/validation rows, then entered training on
  assigned GPU 1. Execution-manifest/trial hashes are
  `758fc88a0f1d559fb74938ffe582bc4382e017c9e1736e828582f1999ed4af01`
  and `a28ed57c57ace9ee3bc5629a42ccbf989684dd9cfeb26e8468c2b6cf40031afc`.
  Conditional terminal ETA is about `04:50 SAST`. Foreign GPU 0 remains
  untouched; Kombuys root has `24 GB` free and scratch `2.1 TB` free.
- HEX NER `b3` job `1222348` remains in its epoch-10 step-5410 corrected
  callback with no complete artifact. `b4` job `1224858` entered its epoch-6
  callback after health-only loss `0.36709431361088524`; `b5` job `1226719`
  is in epoch-2 validation. No decision was made from incomplete outputs.
  These are exactly three healthy A100-40GB jobs on `srvrocgpu010`; b6
  remains unsubmitted, quota is home `52.0%`, scratch `37.9%`, and no
  A100-80GB/L40S work is owned.
- Trusted base `16/16`; NER seed-42 terminal-valid `6/11`; T2X seed-42
  terminal-valid `3/11`; frozen winners `0/8`; held-out adapter evaluations
  `0`; Mono not started; Sheet E/F/G blank; Hugging Face blocked.

## T2X a2 reaches 51.3063; NER b4/b5 artifacts valid — 03:00 SAST

- Kombuys T2X `a2` improved from epoch-1 chrF `44.05334578338157` to
  `48.45863225473362` at step 966 and `51.306312454497196` at epoch-3 step
  1449. Both artifacts have exact 64-row Xhosa coverage and no literal or
  whitespace-only empty predictions; step 1449 has `63` unique raw outputs.
  Step-966/1449 debug hashes are
  `2116200b7d688cf31d5cf58464b6755f726926f5b123a62c591a209e5396b413`
  and `89ac3c5d016284995dd42392ffdca65cce7f806d36fe5f2489944ad282290548`;
  retained trainer-state SHA-256 is
  `d1f2ff9f48abe937d5a0dfe74649c3a1edefed9dd8600b99cd55d27c6d61867c`.
  Checkpoint 1449 is retained. This is within-candidate evidence only; no
  cross-candidate ranking occurred. Epoch 4 is healthy near `1492/1932`, with
  terminal ETA about `03:20 SAST`.
- HEX NER `b4` job `1224858` improved epoch-5 step 2705 to exact mean F1
  `0.5347543586486822666666666667`, from Tsn/Xho/Zul
  `0.529746508018574/0.5153574627258339/0.5591591052016389`. Its
  `0.0627028041238621033333333334` gain resets patience and retains checkpoint
  2705. Coverage is exact `192` and `64/language`, with no literal empties or
  parse failures, `52` whitespace-only predictions, and unique counts
  `41/56/38`. Debug/state hashes are
  `f9167b25efc085650b1bec8f0a4f6eba4d194dc7e12d53038b8076699a06e9be`
  and `e6b7c56dc7747fa2db9e6c9b69ecf27a4770e3fb28fb37bfdecf0a32ce7af7d6`.
- NER `b5` job `1226719` produced its valid initial epoch-1 step-541 exact
  mean F1 `0.1288460123400436366666666667`, from Tsn/Xho/Zul
  `0.128840639082296/0.09949179046124791/0.158205607476587`. Checkpoint 541
  is retained and patience is reset. Coverage is exact `192` and
  `64/language`, with no literal empties or parse failures, `46`
  whitespace-only predictions, and unique counts `57/50/38`. Debug/state
  hashes are `686f43356721c7cb22621ee6316deb2c429ce42a7fe92d1362e14666a44ed59c`
  and `a226fc9906c1e841b656bc8645261f50ad35ec93113d4630496e96418d83ffa1`.
- NER `b3` job `1222348` entered its epoch-10 step-5410 corrected callback
  after health-only loss `0.39855622358924836`; its artifact is absent, so no
  decision was made. Jobs `1222348/1224858/1226719` remain exactly three
  healthy A100-40GB jobs on `srvrocgpu010`, and b6 remains unsubmitted. HEX
  quota is home `52.0%`, scratch `37.9%`; no A100-80GB/L40S work is owned.
  Kombuys GPU 1 is healthy; foreign GPU 0 remains untouched. Root has about
  `25 GB` free and scratch `2.1 TB` free.
- Trusted base `16/16`; NER seed-42 terminal-valid `6/11`; T2X seed-42
  terminal-valid `2/11`; frozen winners `0/8`; held-out adapter evaluations
  `0`; Mono not started; Sheet E/F/G blank; Hugging Face blocked.

## T2X a2 first artifact valid; NER b3 first patience miss — 02:29 SAST

- Kombuys T2X `a2` produced a valid epoch-1 step-483 chrF
  `44.05334578338157`. The artifact has exact 64-row Xhosa coverage, no
  literal or whitespace-only empty predictions, and `63` unique raw outputs;
  debug/state hashes are
  `aa96f2a5340cb9fa4d2f5ef2f54012ddc8caf73d6aa6f61d74fd19c7f3a75c7a`
  and `179b9c204e125c7e5979656175a96f95cb9d5b20634aa8cd2c2db32678ff8eba`.
  Checkpoint 483 is retained. This is within-candidate evidence only; no T2X
  ranking occurred. The run is healthy near `730/1932`, ETA about `03:25
  SAST`. Assigned GPU 1 uses `4562 MiB` at `26%`; foreign GPU 0 remains
  untouched. Kombuys root has about `25 GB` free and scratch `2.1 TB` free.
- HEX NER `b3` job `1222348` recorded epoch-9 step-4869 exact mean F1
  `0.6025279148686584666666666667`, from Tsn/Xho/Zul
  `0.5995898838003603/0.5916999891926445/0.6162938716129706`. This is
  `0.0002029193754183` below its best, so frozen patience correctly becomes
  `1/2` and checkpoint 4328 remains retained at F1
  `0.6027308342440767666666666667`. Coverage is exact `192` and
  `64/language`, with no literal empties or parse failures, `58`
  whitespace-only predictions, and unique counts `40/54/37`. Debug/state
  hashes are `37c2651c66a75f0d39984f0304a1e5d02986208fc00fed163df829210d42be97`
  and `b1291b4f00762ec23e2a9de87dd5307cc3e49cb163403c4b315d5fa7cd858944`.
- NER `b4` job `1224858` is in its epoch-5 callback after health-only loss
  `0.3631326838496892`; `b5` job `1226719` is in epoch 1 after health-only
  loss `1.586359073681459`. Their step-2705/541 artifacts are absent, so no
  decision was made from either loss or partial output. Jobs
  `1222348/1224858/1226719` are exactly three healthy A100-40GB jobs on
  `srvrocgpu010`; b6 remains unsubmitted, with no A100-80GB/L40S work owned.
  HEX quota is home `52.0%`, scratch `37.9%`.
- Trusted base `16/16`; NER seed-42 terminal-valid `6/11`; T2X seed-42
  terminal-valid `2/11`; frozen winners `0/8`; held-out adapter evaluations
  `0`; Mono not started; Sheet E/F/G blank; Hugging Face blocked.

## T2X a1 terminal-valid; a2 starts; NER b4 improves — 02:01 SAST

- Kombuys T2X `a1` is terminal-valid after `1932/1932` steps. Epoch-3 step
  1449 chrF `46.33302341833507` remains the best after final epoch-4 chrF
  `46.01656370856435`; checkpoint 1449 and the final exported adapter are
  therefore correct. The final artifact has exact 64-row Xhosa coverage, no
  literal or whitespace-only empty predictions, and `62` unique raw outputs.
  Epoch-3/final debug hashes are `51c41ff7...45755`/
  `65242c33...2fe5f`; trainer-state/final-adapter/config hashes are
  `b5165888...8c65c`/`9024c825...80ef5`/`831260ba...ed0a7`. Manifest/trial
  hashes remain `889e4a62...a28f9`/`901eb352...8afd0`. T2X terminal-valid
  progress is now `2/11`; no cross-candidate ranking occurred.
- After verifying GPU 1 idle, preregistered T2X `a2` started in tmux
  `sallm-t2x-hpo-a2` with explicit canonical Kombuys model/tokenizer path and
  durable logging. Its exact frozen recipe is LR `1.5e-4`, rank/alpha
  `16/32`, dropout `0.05`, warmup `0.03`, seed 42, batch `4/4`, accumulation
  2, BF16, and `1932` steps. The execution manifest verified all `694` files;
  manifest/trial hashes are `0084c85f...d85f8d`/
  `39fc8c5e...1e213`. It loaded the pure-GDN checkpoint, tokenized exact
  `3,859/460` train/validation rows, and is healthy near step 15. Terminal ETA
  is about `03:25 SAST`. GPU 0 remains untouched.
- HEX NER `b4` job `1224858` improved epoch 4 step 2164 to exact mean F1
  `0.4720515545248201633333333333`, from Tsn/Xho/Zul
  `0.46890412606755544/0.4566782583141/0.49057227919280505`. Its
  `0.1012699454318800433333333333` gain resets patience and retains checkpoint
  2164. Coverage is exact `192` and `64/language`, with no literal empty
  predictions or parse failures, `57` whitespace-only predictions, and
  unique counts `41/56/36`. Debug/state hashes are
  `ca54665c745c2ffc57fec350e41cc3723d9d0b31fbae22ab329cce683e03d19b`
  and `fb183bd1a2c87ac94a352a5ac9e7f15a609b735cc965cc0d6ba37e9032ee2346`.
- NER b5 job `1226719` is now running on `srvrocgpu010`; its `694`-file
  execution manifest and trial hash to `70d8469a...faa534`/
  `ed5188e3...238381`. Jobs `1222348/1224858/1226719` are exactly three
  healthy A100-40GB jobs, so b6 remains unsubmitted. No A100-80GB/L40S work is
  owned. HEX quota is home `52.0%`, scratch `37.9%`; Kombuys root has about
  `25 GB` free and scratch `2.1 TB` free.
- Trusted base `16/16`; NER seed-42 terminal-valid `6/11`; T2X seed-42
  terminal-valid `2/11`; frozen winners `0/8`; held-out adapter evaluations
  `0`; Mono not started; Sheet E/F/G blank; Hugging Face blocked.

## T2X a1 and NER b3 improve; b5 remains resource-pending — 01:29 SAST

- Kombuys T2X `a1` improved at epoch 2 step 966 from chrF
  `39.13618399069993` to `45.13571777938961`, a gain of
  `5.99953378868968`, retaining checkpoint 966. The exact 64-row Xhosa
  validation artifact has no literal or whitespace-only empty predictions and
  `61` unique raw predictions. Debug/state SHA-256 are
  `f3e3c2bb580c5506fcd99b756ff964dde11e0116f5ebcc60aec1f66e307e56ad`
  and `0f7599a9e1dddae16bc47e8f0935c68312c88c4c0581d62c8ad33521907aa831`.
  This is within-candidate validation evidence only; no T2X ranking occurred.
  The run is healthy near `1362/1932`, with terminal ETA still about
  `02:10 SAST`. GPU 1 remains the assigned RTX 3080 Ti and foreign GPU 0
  remains untouched. Root has about `25 GB` free at `90%` used; scratch has
  about `2.1 TB` free.
- HEX NER `b3` job `1222348` improved epoch 8 step 4328 to exact mean F1
  `0.6027308342440767666666666667`, from Tsn/Xho/Zul
  `0.6043554713918071/0.5905688622753994/0.6132681690650238`. Its
  `0.0192414694887328` gain resets frozen patience and retains checkpoint
  4328. Coverage is exact `192` and `64/language`, with no literal empty
  predictions or parse failures, `57` whitespace-only predictions, and
  unique counts `41/54/37`. Debug/state hashes are
  `e1795ae60264ed4a852b7490558b7ece5c729f45377ee0a49192f71a4126a47a`
  and `2a520f87580693f285dccd5d878f8e883bd75126404ad62701a9d01fc76c5160`.
- NER `b4` job `1224858` is in its epoch-4 corrected callback after
  health-only loss `0.38768135269335213`; step 2164 is absent, so no decision
  was made from loss or partial output. NER `b5` job `1226719` remains
  `PENDING (Resources)`. Owned HEX state is two running plus one pending
  A100-40GB job, with no A100-80GB/L40S overlap; b6 remains unsubmitted. Quota
  is home `52.0%`, scratch `37.9%`.
- Trusted base remains `16/16`; NER seed-42 terminal-valid `6/11`; T2X
  seed-42 terminal-valid `1/11`; frozen winners `0/8`; held-out adapter
  evaluations `0`; Mono not started. Sheet E/F/G remain blank and Hugging Face
  publication remains blocked.

## NER b2 terminal-valid; b5 submitted; T2X a1 first artifact valid — 01:01 SAST

- NER Stage-B `b2` job `1220056` completed `0:0` after `8115/8115` steps and
  is terminal-valid. Final epoch-15 step-8115 exact mean F1 is
  `0.46707924525954831`, from Tsn/Xho/Zul
  `0.4643221202853732/0.44837267339082043/0.4885429421024513`. Its
  `0.0008733869788212233333333333` gain over epoch 13 is below the early-stop
  threshold but is numerically the trainer best, so the terminal trainer state
  and exported adapter correctly point to checkpoint 8115. Coverage is exact
  `192` and `64/language`, with no literal empties or parse failures, `57`
  whitespace-only predictions, and unique counts `40/54/37`. Debug/state/
  final-adapter hashes are `8d7e836b...36b6df`/`47b7a3ee...df26c6`/
  `d5decfc9...fdce2`; execution-manifest/trial hashes remain
  `00a0a053...54abe`/`e97b1ce2...626e`.
- A shell preflight erroneously invoked the b5 wrapper directly on the login
  node and was killed after about one second. It left no b5 output directory,
  persisted manifest, model, data, or metric artifact. This failed preflight
  is provenance only and did not affect the recipe. After rechecking exactly
  two running A100-40GB jobs and no A100-80GB/L40S work, preregistered NER b5
  was submitted correctly through Slurm as job `1226719`, with account/
  partition/QOS `nlpgroup/a100/nlpgroup`, `gpu:ampere:1`, `24:00:00`, 8 CPUs,
  and the required chdir. It is `PENDING (Resources)`; owned HEX state is two
  running plus one pending A100-40GB job, so b6 remains unsubmitted.
- Kombuys T2X `a1` produced a valid epoch-1 step-483 chrF
  `39.13618399069993`. Its exact 64-row Xhosa validation artifact has no
  literal or whitespace-only empty predictions, `64` unique raw predictions,
  and debug/state hashes `fe0c54ec...2f99b4`/
  `18bb9d8f...03bffb`. This is within-candidate evidence only and no ranking
  occurred. The run is healthy near `630/1932` in tmux
  `sallm-t2x-hpo-a1-r2`; GPU 1 uses about `11,338 MiB`, while foreign GPU 0
  remains untouched. Terminal ETA remains about `02:10 SAST`. Root has about
  `25 GB` free and scratch `2.1 TB` free.
- Scientifically trusted base remains `16/16`; NER seed-42 terminal-valid is
  now `6/11`; T2X seed-42 terminal-valid is `1/11`; frozen Multilingual
  winners `0/8`; held-out adapter evaluations `0`; Mono not started. HEX
  quota is home `52.0%`, scratch `37.9%`. Sheet E/F/G remain blank and
  Hugging Face publication remains blocked.

## T2X a0 terminal-valid; a1 starts after host-path correction — 00:34 SAST

- Kombuys T2X Stage-A `a0` is terminal-valid after `1932/1932` steps. Its
  final epoch-4 chrF `41.01639999061889` is below retained epoch-3 best
  `41.19802690700617`, so checkpoint 1449 and its final exported adapter are
  correct under the frozen rule. The final debug artifact has exact `64`
  Xhosa validation rows, no literal or whitespace-only empty predictions,
  and `64` unique raw predictions. Final debug SHA-256 is
  `7fe475c156ffae62f0519f71791c1e58a47eb96b8d29b564101a56bfeb1f6c71`;
  final adapter/config hashes are
  `9bc4b3f570bbf8a2c55a27af540e4e2a5d831b984e749bbf60248777aa4f84ca`
  and `a2910a54d06522d265de3e94915a53bccd9f65e0f1bda95f4d1b8faedea8a28c`.
  Manifest/trial hashes remain `58e90c6d...8dbe9`/`11e8aef8...bca76`.
  T2X seed-42 terminal-valid progress is now `1/11`; no ranking occurred.
- The first `a1` launch exited after generating only its three manifest files;
  GPU 1 stayed idle and no model, dataset, or metric artifact was produced.
  It is preserved as `seed_42-launch-failure`, with manifest/trial hashes
  `ea20758d...93efb9`/`cd55610d...d5830`. A durable unchanged retry exposed
  the root cause: the generic wrapper derived nonexistent Kombuys model and
  tokenizer path `/scratch/alombard/masters/sallm/checkpoints/...`; it failed
  while opening the tokenizer before dataset or metric access. That attempt is
  preserved as `seed_42-model-path-failure`, with manifest/trial hashes
  `e86342e0...e9189`/`4448ef3b...a9239a86` and its traceback log.
- A host-path-only retry started in tmux `sallm-t2x-hpo-a1-r2` with the same
  frozen `a1`, seed 42, dataset, optimizer, batches, and metric, but explicit
  canonical Kombuys model/tokenizer path
  `/scratch/alombard/sallm/checkpoints/sallm-pure-gdn-125m/a10080-3epoch-20260802/final_model`.
  Its execution manifest verified all `694` files and hashes to
  `889e4a62d5d62b3d0622e6329ea52cb9f3c8640e3b40c7f0689806d04f5a28f9`;
  trial SHA-256 is
  `901eb352e1e164baa8c19488f6c4af1ccbb835a119ac05eecadb7c8ac138afd0`.
  It loaded the pure-GDN checkpoint in BF16, tokenized exact `3,859/460`
  train/validation rows, and began `1932` steps on GPU 1. Foreign RTX 5090 GPU
  0 remains untouched. Based on `a0`, terminal ETA is about `02:10 SAST`.
- HEX NER `b3` job `1222348` improved epoch 7 step 3787 to exact mean F1
  `0.5834893647553439666666666667` from Tsn/Xho/Zul
  `0.5749901587717647/0.5753040224508388/0.6001739130434284`; the
  `0.0182371178617658666666666667` gain resets patience and retains checkpoint
  3787. `b4` job `1224858` improved epoch 3 step 1623 to exact mean F1
  `0.37078160909294012` from
  `0.36881438093309776/0.3573382755272651/0.3861921708184575`; the
  `0.07486671362132525` gain resets patience and retains checkpoint 1623.
  Both have exact `192` rows and `64/language`, no literal empties or parse
  failures, and `54` whitespace-only outputs. b3 debug/state hashes are
  `3f093f94...91bda5`/`90dd1249...a33239`; b4 values are
  `b0ad1b16...5e5006`/`874c77ef...5fddf1`.
- NER `b2` `1220056` remains in its final epoch-15 callback after health-only
  loss `0.4197781133828996`; no terminal step-8115 artifact exists yet, so
  the job is not counted terminal-valid and `b5` is not submitted. Jobs
  `1220056/1222348/1224858` remain healthy on A100-40GB `srvrocgpu010`; no
  A100-80GB/L40S work is owned. HEX quota is home `52.0%`, scratch `37.8%`.
  Trusted base remains `16/16`; NER seed-42 terminal-valid `5/11`; frozen
  winners `0/8`; held-out adapter evaluations `0`; Mono not started; Sheet
  E/F/G blank; Hugging Face blocked.
