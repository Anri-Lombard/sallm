# Pure-GDN corrected validation-only HPO — 2026-08-15

## POS a2 improves again at step 1981 — 22:19 SAST

- POS a2 `1238228` step 1981 improved exact validation accuracy to
  `0.8594919779162872` over all `1,800` declared rows, 12 language/template
  cells, and 17 closed labels. The gain over step 1698 is
  `0.0031854468158129`, above the frozen `0.001` threshold, so checkpoint
  1981 is the new raw best and EarlyStopping patience again reset to `0/2`.
  Artifact, trainer-state, and adapter SHA-256 values are
  `917ff951df18bf22f1fa565d8c43b9b850f39e7dd4023d06afc57f0c437794c3`,
  `829494f8d734a64cefb4720e2aa72298634c02921d0c7eea6d17e07507a1b82d`,
  and
  `9683f8022fe52267b74d2d2f550abb0f197d36743c3128ef22092c8a96a591c0`.
  This remains clean validation-only operational evidence, not a terminal
  trusted candidate.
- The job remains healthy on `srvrocgpu010` A100-40GB and reached step 2264,
  whose operational loss is `0.12269888136121962`; its next frozen
  constrained callback began at `22:16`. The next exact artifact is estimated
  around `00:25--00:40 SAST`. With patience reset, the earliest possible
  terminal decision moves to the following step-2547 artifact, roughly
  `02:55--03:20 SAST`. The `04:11 SAST` 24-hour limit becomes a provenance
  risk if either upcoming artifact resets patience again; no continuation is
  being invented before its need and frozen resume semantics are established.
  Stage B remains blocked and no additional job was submitted.
- Owned state remains exactly one running A100-40GB `gpu:ampere` job,
  `1238228`, with no pending owned job and no A100-80GB or L40S work. HEX
  quota is home `70.3%`, scratch `38.7%`. Kombuys remains read-only and idle:
  foreign GPU 0 RTX 5090 is `10 MiB/0%`, assigned GPU 1 RTX 3080 Ti is
  `1 MiB/0%`, scratch is `60%` used, and only `tailscale-kombuys` tmux exists.
  Trusted terminal counts remain base `16/16`, NER seed-42 `11/11`, NER
  confirmations `2/4`, T2X seed-42 `11/11`, T2X confirmations `4/4`, POS
  seed-42 `2/11`, winners `0/8`, held-out `0`, and Mono not started. Sheet
  E/F/G remain blank, quarantined rows unpublished, and publication blocked.
  Remaining validation grids prevent a fixed full-results ETA.

## POS a2 improves at step 1698; patience resets — 19:52 SAST

- POS a2 `1238228` step 1698 improved exact validation accuracy to
  `0.8563065311004743` over all `1,800` declared rows, 12 language/template
  cells, and 17 closed labels under
  `closed_label_tuple_mean_logprob_v1`. The gain over step 1415 is
  `0.0025767203700779`, above the frozen `0.001` threshold, so checkpoint
  1698 is the new raw best and EarlyStopping patience correctly reset to
  `0/2`. Artifact, trainer-state, and adapter SHA-256 values are
  `aa6aa0f3336af2173b15498226dea58a04fd3609d70875b27518078122485ba8`,
  `bafa8eb9f9bd472ba49e04aad4c60120303297abf18cb61f27f8c1a0664fd9f7`,
  and
  `9327a4044041f605efc2062299cac6a7b2cd5eaf52182c0e9a675a2ff57423b1`.
  This is clean validation-only operational evidence, not a terminal trusted
  candidate.
- The job remains healthy on `srvrocgpu010` A100-40GB and has entered its
  step-1981 constrained callback after operational loss
  `0.11816371493869357`; it had completed `50/1,800` rows at `19:47`. The
  next exact artifact is estimated around `21:55--22:10 SAST`. Because
  patience reset, the earliest possible terminal decision now requires two
  later threshold misses and is no earlier than roughly `00:25--00:55 SAST`.
  Stage B remains blocked and no additional job was submitted.
- Owned state remains exactly one running A100-40GB `gpu:ampere` job,
  `1238228`, with no pending owned job and no A100-80GB or L40S work. HEX
  quota is home `70.3%`, scratch `38.7%`. Kombuys remains read-only and idle:
  foreign GPU 0 RTX 5090 is `10 MiB/0%`, assigned GPU 1 RTX 3080 Ti is
  `1 MiB/0%`, scratch is `60%` used, and only `tailscale-kombuys` tmux exists.
  Trusted terminal counts remain base `16/16`, NER seed-42 `11/11`, NER
  confirmations `2/4`, T2X seed-42 `11/11`, T2X confirmations `4/4`, POS
  seed-42 `2/11`, winners `0/8`, held-out `0`, and Mono not started. Sheet
  E/F/G remain blank, quarantined rows unpublished, and publication blocked.
  Remaining validation grids prevent a fixed full-results ETA.

## NER seed-87 terminal-valid; POS step-1698 halfway — 18:21 SAST

- NER seed-87 confirmation job `1233035` completed `0:0` at `18:07:40 SAST`
  after `17:21:13`. Its final step-8115 artifact scored exact mean validation
  F1 `0.6894898423629795`; Tsn/Xho/Zul F1 values are
  `0.6711970726606924/0.6959178298656337/0.7013546245626127`. Coverage is
  exact `192`, `64/language`, with no literal empty raw outputs or parser
  failures, `54` whitespace-only outputs (`21/6/27`), and `39/56/37` unique
  raw outputs. This is the second consecutive frozen threshold miss, so
  checkpoint 7033 remains the validation-only winner at mean F1
  `0.6904636994389453`. Final debug, retained trainer-state, retained adapter,
  final adapter, and final config SHA-256 values are
  `5b94ebaf532a7b32e5b94ee1f42c5593e4914b07a1304527f506968f1d8bfa8a`,
  `a000611aaf92ac9737d6ab10497234c45031b3455c1f23b6fa915163e00263f9`,
  `107a024229f1ed4c550d7f50ddcc64296af2517bf0e8dd30ab51c938a1ca66c4`,
  `f0c4d157d16c7d2b5065d37543851639480e1d0240b66c291b8a13fdff860c2f`,
  and
  `62af9169f55da9cb4e68254cddc650df484f6954444ca75251e92905e70fc973`.
  A targeted read-only comparison proved exact equality for all `424/424`
  tensors (`76,410,112` values) between checkpoint 7033 and the final
  serialization. The confirmation is therefore terminal-valid and raises
  trusted NER confirmations to `2/4`.
- POS a2 `1238228` remains healthy on `srvrocgpu010` A100-40GB. At `18:18`
  it had completed `900/1,800` rows in the frozen step-1698 constrained
  callback after operational loss `0.11603702757093641`, with no fault marker.
  Its exact artifact and earliest terminal decision remain estimated around
  `19:15--19:30 SAST`; this is operational progress only until the artifact,
  terminal state, and retained/final adapter identity are verified. Stage B
  remains blocked and no third job was submitted.
- Owned state is now exactly one running A100-40GB `gpu:ampere` job,
  `1238228`, with no pending owned job and no A100-80GB or L40S work. HEX
  quota is home `70.3%`, scratch `38.7%`. Kombuys remains read-only and idle:
  foreign GPU 0 RTX 5090 is `10 MiB/0%`, assigned GPU 1 RTX 3080 Ti is
  `1 MiB/0%`, scratch is `60%` used, and only `tailscale-kombuys` tmux exists.
  Trusted terminal counts are base `16/16`, NER seed-42 `11/11`, NER
  confirmations `2/4`, T2X seed-42 `11/11`, T2X confirmations `4/4`, POS
  seed-42 `2/11`, winners `0/8`, held-out `0`, and Mono not started. Sheet
  E/F/G remain blank, quarantined rows unpublished, and publication blocked.
  Remaining validation grids prevent a fixed full-results ETA.

## NER first miss; POS raw-best sub-threshold gain — 17:04 SAST

- NER seed-87 confirmation job `1233035` step 7574 scored exact mean
  validation F1 `0.6898049384622436`; Tsn/Xho/Zul F1 values are
  `0.6729387168430181/0.6964671194650163/0.7000089790786965`. Coverage is
  exact `192`, `64/language`, with no literal empty raw outputs or parser
  failures, `54` whitespace-only outputs (`21/6/27`), and `39/57/37` unique
  raw outputs. It is a frozen first patience miss against retained checkpoint
  7033 mean F1 `0.6904636994389453`, so patience is `1/2`. Debug,
  current-step trainer-state, and current-step adapter SHA-256 values are
  `372b2a48b0b942ba4035df42e1f845e8975fcfd7a2b80fb837f6d6bc10d34204`,
  `77f976d744d95bfd340790d711bc93884c7327950f6e80f77cc39a8843eff0dd`,
  and
  `c5ee12babbfc83320ae26de00a93a3ba5b3d5df6400adc8dd650b26d6f162709`.
  The job resumed healthy near step `7701/8115`. Its final scheduled
  step-8115 artifact and terminal decision are estimated around
  `18:05--18:20 SAST`.
- POS a2 `1238228` step 1415 reached a new raw-best exact accuracy
  `0.8537298107303964` over `1,800` rows, 12 cells, and 17 labels. The gain
  over step 1132 is only `0.0007278971717521`, below the frozen `0.001`
  early-stopping threshold, so the EarlyStopping callback correctly records
  this as patience `1/2` even though Trainer retains checkpoint 1415 as the
  raw best. Artifact, trainer-state, and adapter SHA-256 values are
  `9da1ee04529d2a585b86eb1e810fb1b0121a726c598a4ac6c2328eee17e56a7c`,
  `157bce3464bb05b8e174090bd69856ce0649fcd374c527ddb57f42ed421f1998`,
  and
  `c9170922ab12e61eec892e4448c907f1d9abb67d54399f38a28e06534e379ced`.
  It resumed healthy near step `1598/4245`; its next step-1698 artifact and
  earliest terminal decision are estimated around `19:15--19:30 SAST`.
- Owned state remains exactly two healthy running A100-40GB `gpu:ampere`
  jobs, `1233035` and `1238228`, on `srvrocgpu010`; there is no pending owned
  job and no A100-80GB or L40S work. The third slot remains intentionally
  idle because Stage B waits for a2's terminal validation-only decision. HEX
  quota is home `70.3%`, scratch `38.7%`. Kombuys remains read-only and idle:
  foreign GPU 0 RTX 5090 is `10 MiB/0%`, assigned GPU 1 RTX 3080 Ti is
  `1 MiB/0%`, scratch is `60%` used, and only `tailscale-kombuys` tmux exists.
  Trusted terminal counts remain base `16/16`, NER seed-42 `11/11`, NER
  confirmations `1/4`, T2X seed-42 `11/11`, T2X confirmations `4/4`, POS
  seed-42 `2/11`, winners `0/8`, held-out `0`, and Mono not started. Sheet
  E/F/G remain blank, quarantined rows unpublished, and publication blocked.
  Active grids prevent a fixed full-results ETA.

## NER seed-87 improves to 0.6905; both callbacks converge — 16:04 SAST

- NER seed-87 confirmation job `1233035` improved at step 7033 to exact mean
  validation F1 `0.6904636994389453`; Tsn/Xho/Zul F1 values are
  `0.6727676820498912/0.6974781048854626/0.7011453113814818`. Coverage is
  exact `192`, `64/language`, with no literal empty raw outputs or parser
  failures, `54` whitespace-only outputs (`21/6/27`), and `39/56/37` unique
  raw outputs. Checkpoint 7033 is retained with patience reset to `0/2`.
  Debug, trainer-state, and adapter SHA-256 values are
  `3425049ce98f2d4443593839e963cfe147167a4ff7afcf55c80fcd744a8bdc8a`,
  `a000611aaf92ac9737d6ab10497234c45031b3455c1f23b6fa915163e00263f9`,
  and
  `107a024229f1ed4c550d7f50ddcc64296af2517bf0e8dd30ab51c938a1ca66c4`.
  This remains a scientifically eligible validation-only interim artifact,
  not a terminal confirmation. The job resumed healthy near step
  `7354/8115`; its next exact artifact is estimated around
  `16:55--17:10 SAST`.
- POS a2 `1238228` advanced to `1,100/1,800` rows in its step-1415 frozen
  constrained callback without a fault marker, retaining checkpoint 1132
  accuracy `0.8530019135586443`. Its next exact artifact is likewise
  estimated around `16:50--17:05 SAST`.
- Owned state remains exactly two healthy running A100-40GB `gpu:ampere`
  jobs, `1233035` and `1238228`, on `srvrocgpu010`; there is no pending owned
  job and no A100-80GB or L40S work. The third slot remains intentionally
  idle because Stage B waits for a2's terminal validation-only decision. HEX
  quota is home `70.3%`, scratch `38.7%`. Kombuys remains read-only and is
  now idle: foreign GPU 0 RTX 5090 is `10 MiB/0%`, assigned GPU 1 RTX 3080 Ti
  is `1 MiB/0%`, scratch is `60%` used, and only `tailscale-kombuys` tmux
  exists. Trusted terminal counts remain base `16/16`, NER seed-42 `11/11`,
  NER confirmations `1/4`, T2X seed-42 `11/11`, T2X confirmations `4/4`, POS
  seed-42 `2/11`, winners `0/8`, held-out `0`, and Mono not started. Sheet
  E/F/G remain blank, quarantined rows unpublished, and publication blocked.
  Active grids prevent a fixed full-results ETA.

## NER seed-87 improves to 0.6881; POS callback healthy — 15:05 SAST

- NER seed-87 confirmation job `1233035` improved at step 6492 to exact mean
  validation F1 `0.6880731826771278`; Tsn/Xho/Zul F1 values are
  `0.6716242661447642/0.692505916381754/0.7000893655048651`. Coverage is
  exact `192`, `64/language`, with no literal empty raw outputs or parser
  failures, `54` whitespace-only outputs (`21/6/27`), and `39/56/37` unique
  raw outputs. Checkpoint 6492 is retained with patience reset to `0/2`.
  Debug, trainer-state, and adapter SHA-256 values are
  `49673d05335d26875e248e1b4ccbbfe250aa1d4fafeff627b37b0ea7adf02ef4`,
  `79b25e9c99abdf66261fa94bc701375ab0b315fb0eac1d0e351e821e90a28e7a`,
  and
  `435f118543619996c8f772873f3e3ca98cb15b522c66a1ba4959cd2859125f78`.
  This is a scientifically eligible validation-only interim artifact, not a
  terminal confirmation. The job reached step `7033/8115`; its next exact
  artifact is estimated around `15:45--16:00 SAST`.
- POS a2 `1238228` reached step 1415, completed operational validation loss
  `0.12002206590440538/1,800`, and advanced to `350/1,800` rows in its frozen
  constrained callback without a fault marker. The loss is not a selection
  metric; retained checkpoint 1132 accuracy remains
  `0.8530019135586443`. Its next exact artifact remains estimated around
  `16:50--17:05 SAST`.
- Owned state is exactly two healthy running A100-40GB `gpu:ampere` jobs,
  `1233035` and `1238228`, on `srvrocgpu010`; there is no pending owned job
  and no A100-80GB or L40S work. The third slot remains intentionally idle
  because Stage B waits for a2's terminal validation-only decision. HEX quota
  is home `70.3%`, scratch `38.7%`. Kombuys remains read-only: assigned GPU 1
  RTX 3080 Ti is idle at `1 MiB/0%`; foreign GPU 0 RTX 5090 is active at
  `14,772 MiB/89%` and was untouched; scratch is `60%` used and only
  `tailscale-kombuys` tmux exists. Trusted terminal counts remain base
  `16/16`, NER seed-42 `11/11`, NER confirmations `1/4`, T2X seed-42
  `11/11`, T2X confirmations `4/4`, POS seed-42 `2/11`, winners `0/8`,
  held-out `0`, and Mono not started. Sheet E/F/G remain blank, quarantined
  rows unpublished, and publication blocked. Active grids prevent a fixed
  full-results ETA.

## POS a2 improves to 0.8530; NER callback healthy — 14:35 SAST

- POS a2 `1238228` improved at step 1132 to exact validation accuracy
  `0.8530019135586443` over `1,800` rows, 12 cells, and 17 labels under the
  frozen `closed_label_tuple_mean_logprob_v1` protocol. Checkpoint 1132 is
  retained with patience reset to `0/2`. Artifact, trainer-state, and adapter
  SHA-256 values are
  `572122f79cd4dc46c9702970c67ef17cebce2f9728d0a4fe8a1ec050f8175d87`,
  `60add6732dcb7a91f968cfe4cce0d7b86c399d0d8af36827e8424d35459a9474`,
  and
  `19028705ff0fde37ac280b93d977708561689fac249cc5119d94cf346a239782`.
  This is a valid validation-only interim result, not a terminal Stage-A
  decision. The job resumed healthy, reached step `1411/4245`, and is due to
  enter its step-1415 callback; its next exact artifact is estimated around
  `16:50--17:05 SAST`.
- NER seed-87 `1233035` completed step-6492 operational validation loss
  `0.3785168956203532/10,760` and advanced all three automatic generation
  segments at `14:01:18`, `14:09:05`, and `14:23:42` without a fault marker.
  The loss is not a selection metric; retained checkpoint 5951 mean F1 stays
  `0.6865433259628579`. The step-6492 exact artifact is expected around
  `14:40--14:50 SAST`.
- Owned state remains exactly two healthy running A100-40GB `gpu:ampere`
  jobs, `1233035` and `1238228`, on `srvrocgpu010`; there is no pending owned
  job and no A100-80GB or L40S work. The third allowed slot remains
  intentionally idle because Stage B waits for a2's terminal validation-only
  decision. HEX quota is home `70.3%`, scratch `38.7%`. Kombuys remains
  read-only: assigned GPU 1 RTX 3080 Ti is idle at `1 MiB/0%`; foreign GPU 0
  RTX 5090 is active at `19,318 MiB/97%` and was untouched; scratch is `60%`
  used and only `tailscale-kombuys` tmux exists. Trusted terminal counts stay
  base `16/16`, NER seed-42 `11/11`, NER confirmations `1/4`, T2X seed-42
  `11/11`, T2X confirmations `4/4`, POS seed-42 `2/11`, winners `0/8`,
  held-out `0`, and Mono not started. Sheet E/F/G remain blank, quarantined
  rows unpublished, and publication blocked. Active grids prevent a fixed
  full-results ETA.

## POS a1 terminal-valid; NER improves twice — 14:02 SAST

- POS a1 job `1233034` is now terminal-valid. A local read-only comparison
  proved exact equality for all `424/424` tensors (`71,762,560` values)
  between retained checkpoint 1415 and `final_adapter`. Together with its
  clean `0:0` completion, exact coverage, frozen early-stop decision, and
  recorded hashes, this advances POS seed-42 terminal progress to `2/11`.
  Compute-node verification `1238386` remained priority-pending with no
  allocation or model/data access; after the local proof made it redundant,
  it was cancelled and is preserved as `CANCELLED`, elapsed `00:00:00`.
- NER seed-87 `1233035` step 5410 scored mean validation F1
  `0.6796900433721967` (Tsn/Xho/Zul
  `0.658340767172118/0.6911527871577247/0.6895765757867476`), a frozen
  patience miss against step 4869. Its debug SHA-256 is
  `155245b25c5e9b37a652eac99edc0c117eac0f31adca02f1a8bc39687f437dc3`.
  Step 5951 then improved to retained mean F1 `0.6865433259628579`, with
  Tsn/Xho/Zul
  `0.6641949152541873/0.6956059720524448/0.6998290905819415`, resetting
  patience to `0/2`. Both have exact `192`, `64/language`, no literal empty
  outputs or parser failures. Step 5951 has `54` whitespace-only outputs
  (`21/6/27`) and `39/56/37` unique values. Debug, trainer-state, and adapter
  hashes for retained step 5951 are
  `9d0c12526228ce8c2edea55e25dd9ff3a3dbaf8954b792a110b358bfab90cb46`,
  `73f1dda589bdf2bef149521d729dfeb1f5f9a639486458cf4b9a20dead11b7a5`,
  and
  `86a27eeb67d504e595327e337b8bff3b7befca22bfc61c67f870233834873afa`.
  It reached step `6492/8115` and entered full validation; the next exact
  artifact is estimated around `14:50--15:05 SAST`.
- POS a2 `1238228` remains healthy in its step-1132 constrained callback,
  last observed at `1,450/1,800` rows, retaining checkpoint 849 accuracy
  `0.8411869496098022`. Its next exact artifact is estimated around
  `14:20--14:35 SAST`. Stage B is not eligible to start until a2 reaches a
  terminal validation-only decision, so the freed third A100 slot is
  intentionally unused.
- Owned state is exactly two healthy running A100-40GB `gpu:ampere` jobs,
  `1233035` and `1238228`, with no pending owned job and no A100-80GB or L40S
  work. HEX quota is home `70.3%`, scratch `38.7%`. Kombuys remains
  read-only: assigned GPU 1 RTX 3080 Ti is idle at `1 MiB/0%`; foreign GPU 0
  RTX 5090 is active at `16,992 MiB/96%` and was untouched; scratch is `60%`
  used and only `tailscale-kombuys` tmux exists. Trusted counts otherwise
  remain base `16/16`, NER seed-42 `11/11`, NER confirmations `1/4`, T2X
  seed-42 `11/11`, T2X confirmations `4/4`, winners `0/8`, held-out `0`, and
  Mono not started. Sheet E/F/G remain blank, quarantined rows unpublished,
  and publication blocked. Active grids prevent a fixed full-results ETA.

## POS a1 completes; a2 improves; terminal proof queued — 12:23 SAST

- POS a1 job `1233034` completed cleanly `0:0` at `11:58:10 SAST` after
  `17:53:22`. Step 1981 scored exact accuracy `0.8432799447332778` over
  `1,800` rows, 12 cells, and 17 labels, so retained checkpoint 1415 remains
  the raw best at `0.8474682174611892` after the frozen second miss. The
  artifact, retained trainer-state, retained adapter, final adapter, and
  final config SHA-256 values are
  `a9d43240e93a6b638183ac8d8e05ee0bbcaa535272c6b7a3c08aaaac41e6773f`,
  `118b3055eb64cd1842e8acec515f751b4103ce83a754a18886a37dc266fff324`,
  `9711a463c72df12f7bceb75de6641e11dd0ed80906a5f860d458fef12a1450f1`,
  `14f37519baf8d394a94245f83c7c8622798d133a28480d74baeef96fd7791db5`,
  and
  `d828b30ffe24f8bff31f8c4bf0dcfc6d92c0beebf2534763fbc96e6d42ec399b`.
  A compute-node exact retained/final tensor comparison was submitted as
  verification job `1238386`; it is priority-pending without GPU allocation
  or model/data access. Until that proof completes, a1 is operationally
  complete but not promoted to terminal-valid, so POS terminal progress
  remains `1/11`.
- POS a2 `1238228` improved at step 849 to exact accuracy
  `0.8411869496098022` over `1,800` rows, 12 cells, and 17 labels. Checkpoint
  849 is retained at patience `0/2`. Artifact, trainer-state, and adapter
  hashes are
  `1662c7b20cdf502162cb7b0859ea73dfcc766581b931124b403b6ae9f41cbdf9`,
  `e7b81446d51be2b8a717636c05ad7c184a4b513b17351916626bbd29cba8715d`,
  and
  `5d65fc9a206c6fa55f7597ee3eb6683e0472a5a7f0f4db2d406bacad23d87332`.
  It resumed, reached step 1132, completed operational loss
  `0.11797693888346354/1,800`, and advanced to `100/1,800` constrained rows;
  its next exact artifact is estimated around `14:15--14:30 SAST`.
- NER seed-87 `1233035` remains healthy in its step-5410 generation callback;
  automatic segments advanced at `11:39:46`, `11:47:44`, and `12:02:37`
  without faults. Retained checkpoint 4869 mean F1 remains
  `0.6802812718619565`; the step-5410 exact artifact is estimated around
  `12:25--12:35 SAST`.
- Owned state is two healthy running jobs (`1233035`, `1238228`) plus
  priority-pending verification `1238386`, all A100-40GB `gpu:ampere` on the
  mandated account/partition/QoS; no owned A100-80GB or L40S work exists.
  HEX quota is home `70.3%`, scratch `38.7%`. Kombuys remains read-only:
  assigned GPU 1 RTX 3080 Ti is idle at `1 MiB/0%`; foreign GPU 0 RTX 5090
  is active at `20,144 MiB/99%` and was untouched; scratch is `60%` used and
  only `tailscale-kombuys` tmux exists. Trusted terminal counts otherwise
  remain base `16/16`, NER seed-42 `11/11`, NER confirmations `1/4`, T2X
  seed-42 `11/11`, T2X confirmations `4/4`, winners `0/8`, held-out `0`, and
  Mono not started. Sheet E/F/G remain blank, quarantined rows unpublished,
  and publication blocked. Active grids and the a1 proof prevent a fixed
  full-results ETA.

## POS callbacks nearly complete; tenth NER callback healthy — 11:46 SAST

- POS a1 `1233034` advanced to `1,600/1,800` constrained rows at step 1981,
  retaining checkpoint 1415 accuracy `0.8474682174611892`. POS a2 `1238228`
  advanced to `1,700/1,800` rows at step 849, retaining checkpoint 566
  accuracy `0.817024570725743`. Both logs are fresh and fault-free; exact
  artifacts are now estimated around `11:55--12:05 SAST`.
- NER seed-87 `1233035` reached step `5410/8115`, completed full operational
  validation loss `0.3694716556364719/10,760`, and entered its tenth frozen
  generation callback. Automatic generation began at `11:39:46` without a
  fault marker. The loss is not a selection metric; retained checkpoint 4869
  remains mean F1 `0.6802812718619565`. The next exact NER artifact is
  estimated around `12:15--12:30 SAST`.
- Owned state remains exactly three healthy running A100-40GB `gpu:ampere`
  jobs on `srvrocgpu010`: `1233034`, `1233035`, and `1238228`. There is no
  owned A100-80GB or L40S work. HEX quota is home `70.3%`, scratch `38.7%`.
  Kombuys remains read-only: assigned GPU 1 RTX 3080 Ti is idle at
  `1 MiB/0%`; foreign GPU 0 RTX 5090 is active at `21,508 MiB/99%` and was
  untouched; scratch is `60%` used and only `tailscale-kombuys` tmux exists.
  Scientifically trusted terminal progress is unchanged: base `16/16`, NER
  seed-42 `11/11`, NER confirmations `1/4`, T2X seed-42 `11/11`, T2X
  confirmations `4/4`, POS seed-42 `1/11`, global winners `0/8`, held-out
  `0`, and Mono not started. Sheet E/F/G remain blank, quarantined rows remain
  unpublished, and Hugging Face publication stays blocked. Active grids
  prevent a fixed full-results ETA.

## NER seed-87 improves to 0.6803 mean F1 — 11:11 SAST

- NER seed-87 confirmation `1233035` improved at step 4869 to exact mean
  validation F1 `0.6802812718619565`; Tsn/Xho/Zul F1 values are
  `0.6592776111617704/0.6831280071907713/0.6984381972333281`. Coverage is
  exact `192`, `64/language`, with no literal empty raw outputs, `49`
  whitespace-only values (`18/6/25`), `42/56/39` unique values, and no parser
  failures. Checkpoint 4869 is retained at frozen patience `0/2`. Debug,
  trainer-state, and adapter SHA-256 values are
  `1687c8f73ba29630416b87e901e7e8b62c11823498e9828d224c377852323280`,
  `6f32f792248e07b7694e5389e4d976cb4c230235ca581f9010fa4edc89c2b4d9`,
  and
  `74f79666822a9638b91811032306db19344b3d38ae2376332a0cd77fe37d7d3e`.
  This is a valid interim validation-only artifact, not a terminal
  confirmation result. The job resumed healthy near `4986/8115`; its next
  exact artifact is estimated around `12:10--12:25 SAST`.
- POS a1 `1233034` advanced to `1,150/1,800` constrained rows at step 1981,
  retaining checkpoint 1415 accuracy `0.8474682174611892`. POS a2 `1238228`
  advanced to `1,250/1,800` at step 849, retaining checkpoint 566 accuracy
  `0.817024570725743`. Both logs are fresh and clean; their next exact
  artifacts are estimated around `11:50--12:10 SAST`.
- Owned state remains exactly three healthy running A100-40GB `gpu:ampere`
  jobs on `srvrocgpu010`: `1233034`, `1233035`, and `1238228`. There is no
  owned A100-80GB or L40S work. HEX quota is home `70.3%`, scratch `38.7%`.
  Kombuys remains read-only: assigned GPU 1 RTX 3080 Ti is idle at
  `1 MiB/0%`; foreign GPU 0 RTX 5090 is active at `24,130 MiB/95%` and was
  untouched; scratch is `60%` used and only `tailscale-kombuys` tmux exists.
  Scientifically trusted terminal progress is unchanged: base `16/16`, NER
  seed-42 `11/11`, NER confirmations `1/4`, T2X seed-42 `11/11`, T2X
  confirmations `4/4`, POS seed-42 `1/11`, global winners `0/8`, held-out
  `0`, and Mono not started. Sheet E/F/G remain blank, quarantined rows remain
  unpublished, and Hugging Face publication stays blocked. Active grids
  prevent a fixed full-results ETA.

## Next NER callback entered generation; POS callbacks advancing — 10:41 SAST

- NER seed-87 `1233035` reached step `4869/8115`, completed full operational
  validation loss `0.361292127871602/10,760`, and entered its ninth frozen
  generation callback. Automatic generation advanced at `10:29:38` and
  `10:37:11` without a fault marker. The loss is not a selection metric;
  retained eligible checkpoint 4328 remains mean F1
  `0.6748547444980355`. The next exact artifact is estimated around
  `11:00--11:15 SAST`.
- POS a1 `1233034` reached `750/1,800` rows in its step-1981 constrained
  callback, retaining checkpoint 1415 accuracy `0.8474682174611892`. POS a2
  `1238228` reached `850/1,800` rows in its step-849 callback, retaining
  checkpoint 566 accuracy `0.817024570725743`. Both logs are fresh and clean;
  their next exact artifacts remain estimated around `11:50--12:10 SAST`.
- Owned state remains exactly three healthy running A100-40GB `gpu:ampere`
  jobs on `srvrocgpu010`: `1233034`, `1233035`, and `1238228`. There is no
  owned A100-80GB or L40S work. HEX quota is home `70.3%`, scratch `38.7%`.
  Kombuys remains read-only: assigned GPU 1 RTX 3080 Ti is idle at
  `1 MiB/0%`; foreign GPU 0 RTX 5090 is active at `17,238 MiB/97%` and was
  untouched; scratch is `60%` used and only `tailscale-kombuys` tmux exists.
  Scientifically trusted terminal progress is unchanged: base `16/16`, NER
  seed-42 `11/11`, NER confirmations `1/4`, T2X seed-42 `11/11`, T2X
  confirmations `4/4`, POS seed-42 `1/11`, global winners `0/8`, held-out
  `0`, and Mono not started. Sheet E/F/G remain blank, quarantined rows remain
  unpublished, and Hugging Face publication stays blocked. Active grids
  prevent a fixed full-results ETA.

## Two NER improvements and new POS artifacts reconciled — 10:10 SAST

- NER seed-87 confirmation `1233035` produced two further exact
  validation-only improvements. Step 3787 reached mean F1
  `0.6673562272065903` with Tsn/Xho/Zul
  `0.6463013698629637/0.6633297062023442/0.6924376055544631`; its debug
  artifact SHA-256 is
  `bcf9a682281272819a92238e99a1289768a0c2cce31669f0722007fc22b27e5f`.
  Step 4328 then reached retained mean F1 `0.6748547444980355`, with
  Tsn/Xho/Zul
  `0.6579163248563898/0.6699258445481515/0.6967220640895654`. Both artifacts
  have exact `192`, `64/language`, no literal empty outputs or parser
  failures. Step 4328 has `54` whitespace-only outputs (`21/6/27`) and
  `39/56/37` unique raw outputs. Checkpoint 4328 is retained at patience
  `0/2`; debug, trainer-state, and adapter SHA-256 values are
  `d1324aff2ef368ef291bfd136f317e15f3a46f4c9e340db189d4c8185d3ed14e`,
  `12039d0220291342c53086cc87e3ba456246e2f6af8a441b4a04fc5d6c08fa86`,
  and
  `12d4a3cefdf077c440cdbc5b2ef55ac30ab1d2ad380b3a93bbbe4d2c67aaa6c1`.
  The job resumed healthy near `4586/8115`; its next exact artifact is
  estimated around `11:00--11:15 SAST`.
- POS a1 `1233034` step 1698 scored exact accuracy
  `0.8429252468406908` over `1,800` rows, 12 cells, and 17 labels. It did not
  replace retained checkpoint 1415 (`0.8474682174611892`), and patience is
  `1/2`. Artifact, trainer-state, and current-step adapter hashes are
  `50e7069b9b51bb218db7ca6eb9676f43247e895c1af0fa7a5ad83aa299ff9260`,
  `a7b00a8b8070b6b3be0a4c6f4870efaf63de016f24b1cea064c47ece0741a3a5`,
  and
  `21c0397118392c9238ce9622d208d03e342365d1a6df95b617471ebb30bfc0dc`.
  It advanced to step 1981, full operational validation loss
  `0.12092266082763672/1,800`, and `350/1,800` constrained rows.
- POS a2 `1238228` improved at step 566 to exact accuracy
  `0.817024570725743` over `1,800` rows, 12 cells, and 17 labels; checkpoint
  566 is retained at patience `0/2`. Artifact, trainer-state, and adapter
  hashes are
  `6916f58179b04706bbf0b13319d37ea52a9dd1d5254c8c75b51bf841c4a3793a`,
  `a5a8790ae22651af59498a620a326b53f37ed5fa546b4a146b9648cbe1710d27`,
  and
  `d6e14db8487b41fea63b3fb2063222bffdcbde09fa0840efd44b59f89288a5f6`.
  It advanced to step 849, full operational validation loss
  `0.12573289235432944/1,800`, and `400/1,800` constrained rows. The two POS
  artifacts currently in progress are estimated around `11:50--12:10 SAST`.
- Owned state is exactly three healthy running A100-40GB `gpu:ampere` jobs
  on `srvrocgpu010`: `1233034`, `1233035`, and `1238228`. There is no owned
  A100-80GB or L40S work. HEX quota is home `70.3%`, scratch `38.7%`.
  Kombuys remains read-only: assigned GPU 1 RTX 3080 Ti is idle at
  `1 MiB/0%`; foreign GPU 0 RTX 5090 is active at `16,854 MiB/88%` and was
  untouched; scratch is `60%` used and only `tailscale-kombuys` tmux exists.
  These are eligible interim validation artifacts; scientifically trusted
  terminal progress remains base `16/16`, NER seed-42 `11/11`, NER
  confirmations `1/4`, T2X seed-42 `11/11`, T2X confirmations `4/4`, POS
  seed-42 `1/11`, global winners `0/8`, held-out `0`, and Mono not started.
  Sheet E/F/G remain blank, quarantined rows remain unpublished, and Hugging
  Face publication stays blocked. Active grids prevent a fixed full-results
  ETA.

## NER seed-87 improves again at step 3246 — 07:58 SAST

- NER seed-87 confirmation `1233035` improved at step 3246 to exact mean
  validation F1 `0.6536194168330306`; Tsn/Xho/Zul F1 values are
  `0.6384674723578827/0.6380013420739707/0.6843894360672385`. Coverage is
  exact `192`, `64/language`, with no literal empty raw outputs, `46`
  whitespace-only values (`17/7/22`), `43/57/41` unique values, and no parser
  failures. Checkpoint 3246 is retained and frozen patience is `0/2`. Debug,
  trainer-state, and adapter SHA-256 values are
  `88ac1f966b7be13fa0fad0f4d09d3dcd7ef935cd3fab6aef76bd3e89271f563b`,
  `f0b81e9a2fe0fd634f413d107d4703899d6143cfdf1d3433c6237d91c790f593`,
  and
  `723d528a2acf521a5624a771089286192f041fd38215383b80c7c9e6685e0764`.
  This is a valid validation-only interim artifact, not a terminal
  confirmation result. The job resumed near step `3609/8115`; its next exact
  artifact is estimated around `08:55--09:10 SAST`.
- POS a1 `1233034` and a2 `1238228` remain healthy in their next frozen
  constrained callbacks at `600/1,800` and `700/1,800` rows, respectively.
  Retained eligible accuracies remain `0.8474682174611892` and
  `0.7964588530699107`; the next exact artifacts are estimated around
  `09:10--09:30 SAST`.
- Owned state is exactly three running A100-40GB `gpu:ampere` jobs on
  `srvrocgpu010`: `1233034`, `1233035`, and `1238228`. There is no owned
  A100-80GB or L40S work. HEX quota is home `70.3%`, scratch `38.6%`.
  Kombuys remains read-only: assigned GPU 1 RTX 3080 Ti is idle at
  `1 MiB/0%`; foreign GPU 0 RTX 5090 is active at `25,626 MiB/97%` and was
  untouched; scratch is `60%` used and only `tailscale-kombuys` tmux exists.
  Scientifically trusted terminal progress remains base `16/16`, NER seed-42
  `11/11`, NER confirmations `1/4`, T2X seed-42 `11/11`, T2X confirmations
  `4/4`, POS seed-42 `1/11`, global winners `0/8`, held-out `0`, and Mono not
  started. Sheet E/F/G remain blank, quarantined rows remain unpublished,
  and Hugging Face publication stays blocked. Active validation grids prevent
  a fixed full-results ETA.

## Sixth NER and next POS callbacks healthy — 07:27 SAST

- NER seed-87 `1233035` completed step-3246 full validation loss
  `0.33468470236625814/10,760` and advanced all three automatic generation
  segments at `07:02:52`, `07:10:31`, and `07:25:21` without a fault marker.
  The loss is operational only; retained eligible mean F1 remains
  `0.648578345565961`. The next exact constrained artifact is expected around
  `07:40--07:50 SAST`.
- POS a1 `1233034` reached step `1698/4245`, completed full validation loss
  `0.1237556160820855/1,800`, and advanced to `200/1,800` rows in its next
  frozen constrained callback. POS a2 `1238228` reached step `566/4245`,
  completed full validation loss `0.13736961364746095/1,800`, and advanced
  to `300/1,800` constrained rows. These losses are not selection metrics;
  retained eligible accuracies remain `0.8474682174611892` and
  `0.7964588530699107`. Their next exact artifacts are expected around
  `09:15--09:35 SAST`.
- Owned state remains exactly three running A100-40GB `gpu:ampere` jobs on
  `srvrocgpu010`, with no A100-80GB or L40S work. HEX quota remains home
  `70.3%`, scratch `38.6%`. Kombuys remains read-only: assigned GPU 1 RTX
  3080 Ti is idle at `1 MiB/0%`; foreign GPU 0 RTX 5090 is active at
  `17,290 MiB/95%` and was untouched; scratch is `61%` used and only
  `tailscale-kombuys` tmux exists. Scientifically trusted terminal progress
  remains base `16/16`, NER seed-42 `11/11`, NER confirmations `1/4`, T2X
  seed-42 `11/11`, T2X confirmations `4/4`, POS seed-42 `1/11`, global
  winners `0/8`, held-out `0`, and Mono not started. Sheet E/F/G remain
  blank, quarantined rows remain unpublished, and Hugging Face publication
  stays blocked. Active validation grids prevent a fixed full-results ETA.

## NER and POS a1 improve; POS a2 first exact metric — 06:58 SAST

- NER seed-87 confirmation `1233035` improved at step 2705 to exact mean
  validation F1 `0.648578345565961`; Tsn/Xho/Zul F1 values are
  `0.6576375314157558/0.6211193703541262/0.666978134928001`. Coverage is
  exact `192`, `64/language`, with no literal empty raw outputs, `54`
  whitespace-only values (`21/6/27`), `39/56/37` unique values, and no parser
  failures. Checkpoint 2705 is retained and frozen patience is `0/2`. Debug,
  trainer-state, and adapter SHA-256 values are
  `fbe0c97321145348e41a09dde18267a447273aab89893b88b5b3a295be96ce36`,
  `1bd1b6186d9c5c9a94db6dd4bc2e4bbc8b780a68ab019a34f61375ccfd7d0b23`,
  and
  `150322c4fa6d5028ab35c6860f022c0c15f0d91464257a4e765d307981a7414f`.
  It resumed and reached step `3246/8115`, where its sixth callback began;
  the next exact artifact is expected around `07:40--07:55 SAST`.
- POS a1 `1233034` improved at step 1415 to exact accuracy
  `0.8474682174611892`, covering `1,800` rows, 12 cells, and 17 labels under
  `closed_label_tuple_mean_logprob_v1`. Checkpoint 1415 is retained at
  patience `0/2`. Artifact, trainer-state, and adapter SHA-256 values are
  `f779b24522eedc2d73e00c1025ea09edd198e57f11e3b11ead4212ea8ceb1c48`,
  `118b3055eb64cd1842e8acec515f751b4103ce83a754a18886a37dc266fff324`,
  and
  `9711a463c72df12f7bceb75de6641e11dd0ed80906a5f860d458fef12a1450f1`.
  It resumed healthy near `1492/4245`; its next artifact is expected around
  `09:10--09:25 SAST`.
- POS a2 `1238228` produced its first exact artifact at step 283: accuracy
  `0.7964588530699107`, with exact `1,800` rows, 12 cells, 17 labels, and the
  same frozen constrained protocol. Checkpoint 283 is retained at patience
  `0/2`. Artifact, trainer-state, and adapter SHA-256 values are
  `9f7fd5aca91ce96a50e4ae3c08a8144235cd74829413fc30e14bb9647b0a30cf`,
  `9d341b9f1c03410f08c9096f197e18108d693d190dfb9e1914c4e59ef536bfcf`,
  and
  `75c0c7a51f7c4c664fcc7c42dc557610a6e587f2244061625871840b48176498`.
  It resumed healthy near `495/4245`; its next artifact is expected around
  `09:00--09:15 SAST`.
- Owned state remains exactly three running A100-40GB `gpu:ampere` jobs on
  `srvrocgpu010`, with no A100-80GB or L40S work. HEX quota remains home
  `70.3%`, scratch `38.6%`. Kombuys remains read-only: assigned GPU 1 RTX
  3080 Ti is idle at `1 MiB/0%`; foreign GPU 0 RTX 5090 is active at
  `14,972 MiB/97%` and was untouched; scratch is `60%` used and only
  `tailscale-kombuys` tmux exists. These are valid validation-only interim
  artifacts, but scientifically trusted terminal progress remains base
  `16/16`, NER seed-42 `11/11`, NER confirmations `1/4`, T2X seed-42
  `11/11`, T2X confirmations `4/4`, POS seed-42 `1/11`, global winners `0/8`,
  held-out `0`, and Mono not started. Sheet E/F/G remain blank, quarantined
  rows remain unpublished, and Hugging Face publication stays blocked.
  Active validation grids prevent a fixed full-results ETA.

## All three callbacks near artifact completion — 06:27 SAST

- NER seed-87 `1233035` remains healthy in its step-2705 fifth generation
  callback. Automatic generation segments advanced at `05:55:11`,
  `06:01:06`, and `06:15:44`, with no fault marker. The retained eligible
  artifact remains step-2164 mean F1 `0.5993678536371205`; the step-2705
  exact artifact is expected around `06:30--06:40 SAST`.
- POS a1 `1233034` advanced to `1,450/1,800` frozen constrained rows,
  retaining step-1132 accuracy `0.8268296279293154`. POS a2 `1238228`
  advanced to `1,500/1,800` constrained rows. Both logs are fresh and clean;
  their next/first exact artifacts are expected around `06:45--07:00 SAST`.
- Owned state remains exactly three running A100-40GB `gpu:ampere` jobs on
  `srvrocgpu010`, with no A100-80GB or L40S work. HEX quota remains home
  `70.3%`, scratch `38.6%`. Kombuys remains read-only: assigned GPU 1 RTX
  3080 Ti is idle at `1 MiB/0%`; foreign GPU 0 RTX 5090 is active at
  `15,652 MiB/99%` and was untouched; scratch is `60%` used and only
  `tailscale-kombuys` tmux exists. Scientifically trusted terminal progress
  remains base `16/16`, NER seed-42 `11/11`, NER confirmations `1/4`, T2X
  seed-42 `11/11`, T2X confirmations `4/4`, POS seed-42 `1/11`, global
  winners `0/8`, held-out `0`, and Mono not started. Sheet E/F/G remain
  blank, quarantined rows remain unpublished, and Hugging Face publication
  stays blocked. Active validation grids prevent a fixed full-results ETA.

## Fifth NER callback and both POS callbacks healthy — 05:57 SAST

- NER seed-87 confirmation `1233035` reached step `2705/8115`, completed
  full validation loss `0.331830769932403/10,760`, and entered its fifth
  frozen generation callback without a fault marker. This interim loss is not
  a selection metric; the retained scientifically eligible artifact remains
  step-2164 mean F1 `0.5993678536371205`. Its next exact artifact is expected
  around `06:30--06:45 SAST`.
- POS a1 `1233034` advanced to `1,050/1,800` rows in its step-1415 frozen
  constrained callback, retaining step-1132 accuracy
  `0.8268296279293154`. POS a2 `1238228` advanced to `1,150/1,800` rows in
  its first constrained callback. Both logs are fresh with no fault marker;
  their next/first exact artifacts are expected around `06:45--07:00 SAST`.
- Owned state remains exactly three running A100-40GB `gpu:ampere` jobs on
  `srvrocgpu010`, with no A100-80GB or L40S work. HEX quota remains home
  `70.3%`, scratch `38.6%`. Kombuys remains read-only: assigned GPU 1 RTX
  3080 Ti is idle at `1 MiB/0%`; foreign GPU 0 RTX 5090 is active at
  `20,682 MiB/92%` and was untouched; scratch is `60%` used and only
  `tailscale-kombuys` tmux exists. Scientifically trusted terminal progress
  remains base `16/16`, NER seed-42 `11/11`, NER confirmations `1/4`, T2X
  seed-42 `11/11`, T2X confirmations `4/4`, POS seed-42 `1/11`, global
  winners `0/8`, held-out `0`, and Mono not started. Sheet E/F/G remain
  blank, quarantined rows remain unpublished, and Hugging Face publication
  stays blocked. Active validation grids prevent a fixed full-results ETA.

## NER seed-87 improves to 0.5994 mean F1 — 05:29 SAST

- NER b7 seed-87 confirmation `1233035` improved at step 2164 to exact mean
  validation F1 `0.5993678536371205`; Tsn/Xho/Zul F1 values are
  `0.5816062176165304/0.578320705118243/0.6381766381765882`. Coverage is
  exact `192`, `64/language`, with no literal empty raw outputs, `56`
  whitespace-only values (`21/8/27`), `39/56/37` unique values, and no parser
  failures. Checkpoint 2164 is retained and frozen patience is `0/2`.
  Debug, trainer-state, and adapter SHA-256 values are
  `c28c9781dfbc51c57b9daaec859d498561fe9e4864335643a7cc9872b353331d`,
  `3cafef57a8c262a54ec4f807e696a5ee9a3b0812ebf4a1b733fad1663dad6930`,
  and
  `2eaa8c7db671882267191b35fd52726ace5e71173fd9a150f9e78e119ca3396b`.
  The job resumed healthy near step `2315/8115`; its next artifact is expected
  around `06:30--06:45 SAST`.
- POS a1 `1233034` remains healthy in its step-1415 constrained callback at
  `650/1,800` rows, retaining step-1132 accuracy `0.8268296279293154`; its
  next artifact is expected around `06:50--07:05 SAST`. POS a2 `1238228`
  remains healthy in its first constrained callback at `750/1,800` rows; its
  first exact artifact is expected around `06:40--06:55 SAST`.
- Owned state remains exactly three running A100-40GB `gpu:ampere` jobs on
  `srvrocgpu010`, with no A100-80GB or L40S work. HEX quota remains home
  `70.3%`, scratch `38.6%`. Kombuys remains read-only: assigned GPU 1 RTX
  3080 Ti is idle at `1 MiB/0%`; foreign GPU 0 RTX 5090 is active at
  `14,902 MiB/89%` and was untouched; scratch is `60%` used and only
  `tailscale-kombuys` tmux exists. Scientifically trusted terminal progress
  remains base `16/16`, NER seed-42 `11/11`, NER confirmations `1/4`, T2X
  seed-42 `11/11`, T2X confirmations `4/4`, POS seed-42 `1/11`, global
  winners `0/8`, held-out `0`, and Mono not started. Sheet E/F/G remain
  blank, quarantined rows remain unpublished, and Hugging Face publication
  stays blocked. Active validation grids prevent a fixed full-results ETA.

## Three validation callbacks healthy — 04:57 SAST

- POS a1 `1233034` remains healthy on `srvrocgpu010` A100-40GB. It reached
  step `1415/4245`, completed the full step-1415 validation loss
  `0.12730493757459851/1,800`, and advanced to `250/1,800` rows in the frozen
  constrained metric callback. The retained scientifically eligible artifact
  remains step 1132 accuracy `0.8268296279293154`; the next artifact is
  expected around `06:45--07:00 SAST`.
- NER b7 seed-87 confirmation `1233035` reached step `2164/8115`, completed
  full validation loss `0.32311224387924026/10,760`, and entered its fourth
  frozen generation callback. Automatic generation batching advanced at
  `04:41:28` and `04:50:00` with no fault marker; its next exact constrained
  artifact is expected around `05:15--05:30 SAST`. The retained eligible
  artifact remains step 1623 mean F1 `0.5768414049419194`.
- POS a2 `1238228` remains healthy on the same A100-40GB node. It reached step
  `283/4245`, completed its first full validation loss
  `0.23404678344726562/1,800`, and advanced to `350/1,800` constrained rows.
  Its first exact selection artifact is expected around `06:40--06:55 SAST`.
- Owned state remains exactly three running A100-40GB `gpu:ampere` jobs, with
  no A100-80GB or L40S work. HEX quota remains home `70.3%`, scratch `38.6%`.
  Kombuys remains read-only: assigned GPU 1 RTX 3080 Ti is idle at `1 MiB/0%`;
  foreign GPU 0 RTX 5090 is active at `14,590 MiB/92%` and was untouched;
  scratch is `60%` used and only `tailscale-kombuys` tmux exists. Trusted
  terminal progress remains base `16/16`, NER seed-42 `11/11`, NER
  confirmations `1/4`, T2X seed-42 `11/11`, T2X confirmations `4/4`, POS
  seed-42 `1/11`, global winners `0/8`, held-out `0`, and Mono not started.
  Sheet E/F/G remain blank, quarantined rows remain unpublished, and Hugging
  Face publication stays blocked. Active validation grids prevent a fixed
  full-results ETA.

## POS a1 and NER seed-87 improve; a2 starts — 04:27 SAST

- POS a1 job `1233034` improved at step 1132 to exact
  `all_token_accuracy=0.8268296279293154`. Coverage remains exact `1,800`
  rows, 12 language/template cells, and 17 labels; checkpoint 1132 is
  retained and patience is `0/2`. Artifact, trainer-state, and adapter
  SHA-256 values are
  `60be578c2a31bde2cced01f64712c932a7817572e62a96167845f9465757c743`,
  `609cc077e1d824077ea32cbf32a4444961c0741ab5e271aceb947a9afb8e7dd6`,
  and
  `8d62e2ae8eb2ff5441d1790fea66748e3a2e235c5f66c64327c6ff6785e4746e`.
  It resumed healthy near `1295/4245`; its next artifact is expected around
  `06:40--07:00 SAST`.
- NER b7 seed-87 confirmation `1233035` improved at step 1623 to exact mean
  validation F1 `0.5768414049419194`; Tsn/Xho/Zul F1 values are
  `0.564731240738196/0.557582976880128/0.608209997207434`. Coverage is exact
  `192`, `64/language`, with no literal empty raw outputs, `53`
  whitespace-only values (`20/6/27`), `40/56/37` unique values, and no parser
  failures. Checkpoint 1623 is retained and patience is `0/2`. Debug,
  trainer-state, and adapter SHA-256 values are
  `73228958a51b6eac8f43df61595e4c44fa29cc30aa13d3c019991bc34ef01fa1`,
  `477cb51d6742efa7b313c67a8ed462c7d7266dbdfe5aafea1d9a8a4b8398b725`,
  and
  `cffe3dfa866bdc6d9b3d752b0d2026bfc22ab10b0193e75f7caa056ea7e9c326`.
  It resumed healthy near `2032/8115`; its next artifact is expected around
  `05:10--05:30 SAST`.
- POS a2 job `1238228` started at `04:11:21 SAST` on `srvrocgpu010`
  A100-40GB. It verified all `694` immutable source/config files, exact
  `2,259` train and `1,800` validation rows, and the frozen a2 recipe (LR
  `1.5e-4`, rank/alpha `16/32`, dropout `0.05`, warmup `0.03`), then entered
  training at `04:12:11`. It was healthy near step `275/4245`, with its first
  exact artifact expected around `06:35--06:55 SAST`. Execution-manifest and
  trial SHA-256 values are
  `65b62cb0b2792b7d0c515a4738a6782f15f59917a8c422e61a88263a5e301472`
  and
  `4edadeb3d851456d58043651042f3a4e226f7c7aa59887df6652af6e1c917122`.
- Owned state is exactly three running A100-40GB jobs, with no A100-80GB or
  L40S work. HEX quota is home `70.3%`, scratch `38.6%`. Kombuys remains
  read-only: assigned GPU 1 RTX 3080 Ti is idle at `1 MiB/0%`; foreign GPU 0
  RTX 5090 is active at `18,394 MiB/92%` and was untouched; scratch is `60%`
  used and only `tailscale-kombuys` tmux exists. Trusted terminal progress
  remains base `16/16`, NER seed-42 `11/11`, NER confirmations `1/4`, T2X
  seed-42 `11/11`, T2X confirmations `4/4`, POS seed-42 `1/11`, global
  winners `0/8`, held-out `0`, and Mono not started. Sheet E/F/G remain
  blank, quarantined rows remain unpublished, and Hugging Face publication
  stays blocked. Active validation grids prevent a fixed full-results ETA.

## POS a0 terminal-valid; a2 queued — 03:31 SAST

- POS a0 job `1232181` completed `0:0` at `03:09:18 SAST` after
  `22:49:21`. Its step-2547 exact accuracy is `0.8211477429884227`, a raw
  improvement over step 1981 but only `0.0006173890501758`; because this is
  below the frozen `0.001` early-stopping threshold, patience reached `2/2`
  and training stopped. The trainer's frozen raw-best rule selected
  checkpoint 2547. Coverage is exact `1,800` rows, 12 language/template
  cells, and 17 labels.
- Artifact, trainer-state, current-adapter, final-adapter, and final-config
  SHA-256 values are
  `c1c8b1156a0a48e50c873fb2f3133231453c16dc67894e1d560eba63d49eb837`,
  `833600b5180065f827131db95a2f33f433bf91d249a96a88c4180d3691e58690`,
  `402447d6961949a03f805887e676ec07d5dc6720fd8d8161fdb88997eec38dcd`,
  `a8cf6bc36b090afe4162b33693d860926488b7988e76b2a2364570ce661f2059`,
  and
  `d7682a345e35d2b0488a69ed64635636a3a397a364e9f5fe8572e47758910546`.
  A local read-only comparison proved exact equality for all `424/424`
  tensors (`71,762,560` values) between checkpoint 2547 and final
  serialization. POS seed-42 terminal-valid progress is now `1/11`.
- After verifying the a2 output path was absent and no duplicate job existed,
  preregistered POS a2 seed-42 candidate (LR `1.5e-4`, rank/alpha `16/32`,
  dropout `0.05`, warmup `0.03`) was submitted as job `1238228`. It is
  resource-pending without model/data access. Exact Slurm settings are
  `nlpgroup/a100/nlpgroup`, one `gpu:ampere`, 24 hours, eight CPUs, mandated
  home working directory, immutable POS-source snapshot, and Python 3.12
  runtime.
- POS a1 `1233034` remains healthy at `1150/1800` rows in step-1132
  constrained validation, retaining checkpoint 849 accuracy
  `0.8225303477478206`; its next artifact is expected around
  `04:15--04:30 SAST`. NER seed-87 `1233035` reached step `1623/8115` and
  entered its third validation callback, retaining checkpoint 1082 mean F1
  `0.5257548232368509`; its next artifact is expected around
  `04:00--04:20 SAST`.
- Owned state is two running plus one pending A100-40GB job, with no
  A100-80GB or L40S work. HEX quota is home `70.3%`, scratch `38.6%`.
  Kombuys remains read-only: assigned GPU 1 RTX 3080 Ti is idle at
  `1 MiB/0%`; foreign GPU 0 RTX 5090 is active at `17,084 MiB/97%` and was
  untouched; scratch is `60%` used and only `tailscale-kombuys` tmux exists.
  Trusted progress is base `16/16`, NER seed-42 `11/11`, NER confirmations
  `1/4`, T2X seed-42 `11/11`, T2X confirmations `4/4`, POS seed-42 `1/11`,
  global winners `0/8`, held-out `0`, and Mono not started. Sheet E/F/G
  remain blank, quarantined rows remain unpublished, and Hugging Face
  publication stays blocked. A2 allocation and the remaining validation
  grids prevent a fixed full-results ETA.

## NER seed-87 improves to 0.5258 mean F1 — 02:57 SAST

- NER b7 seed-87 confirmation job `1233035` improved at step 1082 to exact
  mean validation F1 `0.5257548232368509`; Tsn/Xho/Zul F1 values are
  `0.5114958074113647/0.5188186965692855/0.5469499657299023`.
  Coverage is exact `192`, `64/language`, with no literal empty raw outputs,
  `53` whitespace-only raw outputs (`18/7/28`), `42/56/36` unique raw
  outputs, and no parser failures. Checkpoint 1082 is retained and patience
  remains `0/2`. Debug, trainer-state, and adapter SHA-256 values are
  `9af4d0f34ad0cb75348f2aaff3a0647c3e179d7c346352ce34afcc69be560672`,
  `4fbe87a209c86774fde6eea2f68e84caa797e0166c5bfe5beba21bdfe617fab3`,
  and
  `cff24f238012307d23f92aaecfb2b5e01bcdbad905898b8933f8133d10638a88`.
  It resumed healthy near `1102/8115`; the next artifact is expected around
  `04:00--04:20 SAST`.
- POS a0 `1232181` is healthy at `1600/1800` rows in step-2547 constrained
  validation, retaining step-1981 accuracy `0.8205303539382469` at patience
  `1/2`; its next decision is expected around `03:08--03:15 SAST`. POS a1
  `1233034` is healthy at `700/1800` rows in step-1132 validation, retaining
  step-849 accuracy `0.8225303477478206`; its next artifact is expected
  around `04:05--04:25 SAST`.
- All three owned jobs remain on A100-40GB `gpu:ampere`; no A100-80GB or L40S
  work exists. HEX quota is home `70.3%`, scratch `38.6%`. Kombuys remains
  read-only: assigned GPU 1 RTX 3080 Ti is idle at `1 MiB/0%`; foreign GPU 0
  RTX 5090 is active at `23,408 MiB/92%` and was untouched; scratch is `60%`
  used and only `tailscale-kombuys` tmux exists.
- Terminal trusted progress is unchanged: base `16/16`, NER seed-42 `11/11`,
  NER confirmations `1/4`, T2X seed-42 `11/11`, T2X confirmations `4/4`,
  POS `0/11` with 11 valid interim artifacts, global winners `0/8`, held-out
  `0`, and Mono not started. Sheet E/F/G remain blank, quarantined rows remain
  unpublished, and Hugging Face publication stays blocked. Active
  confirmations and the remaining POS, AfriHG, and General grids prevent a
  fixed full-results ETA.

## POS a1 improves to 0.8225; NER seed-87 first metric — 01:58 SAST

- POS a1 job `1233034` improved at step 849 to exact
  `all_token_accuracy=0.8225303477478206`. The frozen constrained artifact
  covers all `1,800` rows, all 12 language/template cells, and all 17 labels;
  checkpoint 849 is retained and patience remains `0/2`. Artifact,
  trainer-state, and adapter SHA-256 values are
  `fd9758086e0b6e8d4bf3d8e7c3d8b6673c3201b9dc2053e082e9f60c39996480`,
  `3a0670b86029fb42a940fea16688c21f33d00182e30ccb27193dba36731c11ba`,
  and
  `051dcb847b878fa7fdb30296765189995115e84ce0520940201659caa03e740a`.
  It resumed healthy near step `1087/4245`; the next exact artifact is
  expected around `04:05--04:25 SAST`.
- NER b7 seed-87 confirmation `1233035` produced its first exact artifact at
  step 541: mean validation F1 `0.32363493061792803`, with Tsn/Xho/Zul F1
  `0.3202108384120759/0.2890758511896485/0.36161810225205965`.
  Coverage is exact `192`, `64/language`; raw output has no literal empties,
  `44` whitespace-only values (`16/4/24`), `44/59/40` unique values, and one
  parser failure. Debug, trainer-state, adapter, execution-manifest, and trial
  SHA-256 values are
  `d510949237df5cecbf31a98b891ddeac5d8ce8a557a24ad38c00d83158b4e206`,
  `fa19ae0ec8a7b9ffe9052ed8ee99b63a14aca442e2a28a28135728de8cb75624`,
  `53d6def63871bed70f18c777d7b24df1566358c1f8427b16bbf90dd342296911`,
  `1004e74dae1e3dd765972e67ef4e8e7f1406b16231623c5bfed35b1c35654f8b`,
  and
  `b68ab4ac552339226bf9d5e5412a2ac75d86ac589514d7152d0da4df3b5c9ec6`.
  Checkpoint 541 is retained; the job resumed healthy near `825/8115`, with
  its next artifact expected around `02:35--02:55 SAST`.
- POS a0 `1232181` remains healthy at `800/1800` rows in step-2547
  constrained validation, retaining step-1981 accuracy
  `0.8205303539382469` at patience `1/2`; its next artifact is expected
  around `03:05--03:20 SAST`. All three owned jobs are running on A100-40GB
  `gpu:ampere`; no A100-80GB or L40S work exists. HEX quota is home `70.3%`,
  scratch `38.6%`.
- Kombuys remains read-only: assigned GPU 1 RTX 3080 Ti is idle at
  `1 MiB/0%`; foreign GPU 0 RTX 5090 is active at `24,334 MiB/94%` and was
  untouched; scratch is `60%` used and only `tailscale-kombuys` tmux exists.
  Terminal trusted progress remains base `16/16`, NER seed-42 `11/11`, NER
  confirmations `1/4`, T2X seed-42 `11/11`, T2X confirmations `4/4`, POS
  `0/11` with 11 valid interim artifacts, global winners `0/8`, held-out `0`,
  and Mono not started. Sheet E/F/G remain blank, quarantined rows remain
  unpublished, and Hugging Face publication stays blocked. Active
  confirmations and the remaining POS, AfriHG, and General grids prevent a
  fixed full-results ETA.

## POS a0 first patience miss; NER seed-87 starts — 00:57 SAST

- Corrected POS a0 job `1232181` completed its step-2264 artifact at exact
  `all_token_accuracy=0.8153398128421713`, below retained step-1981 best
  `0.8205303539382469`; frozen early-stopping patience is now `1/2` and
  checkpoint 1981 remains selected. Coverage is exact `1,800` rows, 12
  language/template cells, and 17 labels under
  `closed_label_tuple_mean_logprob_v1`. Artifact, trainer-state, and current
  adapter SHA-256 values are
  `6d3764841d95823c0b052d69de6230fa9037daf644d95758dffa20510e228af8`,
  `0ae11dc69a8fa243a93fde1b73637f641c6dd68afabcb462a635fa5d55cae872`,
  and
  `8ecc7f65c79e2e06fcc7312286a9e0a5313715d0e580cc343a8939463907193d`.
  It remains healthy at `50/1800` rows in step-2547 constrained validation;
  another frozen patience miss will terminate it, with the next artifact
  expected around `03:05--03:20 SAST`.
- NER b7 seed-87 confirmation job `1233035` started at `00:46:27 SAST` on
  `srvrocgpu010` A100-40GB. It passed immutable startup and exact `4,323`
  training-row tokenization, entered training at `00:47:30`, and was healthy
  near step `191/8115` at about `3.03 s/step`, with no runtime fault marker.
  Its first validation-only confirmation artifact is expected around
  `01:45--02:10 SAST`. POS a1 `1233034` remains healthy at `1150/1800` rows
  in step-849 constrained validation, retaining step-566 accuracy
  `0.7763574498537653`; its next artifact remains expected around
  `01:40--01:55 SAST`.
- Owned state is exactly three running A100-40GB `gpu:ampere` jobs, with no
  A100-80GB or L40S work. HEX quota is home `70.3%`, scratch `38.5%`.
  Kombuys remains read-only: assigned GPU 1 RTX 3080 Ti is idle at
  `1 MiB/0%`; foreign GPU 0 RTX 5090 is active at `19,230 MiB/97%` and was
  untouched; scratch is `60%` used and only `tailscale-kombuys` tmux exists.
- Operational and scientifically trusted terminal progress is unchanged:
  base `16/16`, NER seed-42 `11/11`, NER confirmations `1/4`, T2X seed-42
  `11/11`, T2X confirmations `4/4`, POS `0/11` with ten valid interim
  artifacts, global winners `0/8`, held-out `0`, and Mono not started. Sheet
  E/F/G remain blank, quarantined rows remain unpublished, and Hugging Face
  publication stays blocked. The active NER confirmations and remaining POS,
  AfriHG, and General validation grids still prevent a fixed full-results ETA.
