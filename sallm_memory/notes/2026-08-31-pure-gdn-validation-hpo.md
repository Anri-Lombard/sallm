# Pure-GDN validation HPO — 2026-08-31

## General confirmation wall ceiling raised prospectively - 23:15 SAST

- Before any General Stage-C confirmation was submitted, the user authorized
  a scheduler-only 36-hour ceiling for the four future top-two by seed-13/87
  confirmations. Frozen amendment SHA-256 is
  `4a07b2a27dc996c276c1ad58a47184eb95ca502276b6113b586808d0413d0bf5`.
  Submission will use Slurm's native `--time=36:00:00` override against the
  unchanged immutable scientific launcher. The remaining A100-80GB/eight-CPU
  resource contract and every scientific setting remain unchanged.
- The decision uses only the fixed 13,640-step General schedule and measured
  runtime envelope: about 25.8 hours of raw training at 6.8 seconds per step,
  before exact 22,167-row validation and finalization. No held-out metric,
  candidate score, or interim validation magnitude informed it.
- Current b1/b2/b4 jobs `1279470/1279471/1279652` remain running with their
  original 24-hour limits. B5 no-launch preflight `1279656` remains
  `AssocGrpGRES`-pending with its 10-minute limit. The amendment does not
  authorize another b3 attempt or resolve the blocked all-11 ranking.
- Quota-first HEX verification reports home `88.6%` and scratch `45.3%`.
  Partition `a100` reports `MaxTime=UNLIMITED`, and a no-submit
  `sbatch --test-only` accepted the exact A100-80GB/eight-CPU/36-hour
  envelope. The scheduler-only check emitted planning identifier `1279728`,
  but the subsequent queue readback contained only the unchanged four active
  owned submissions, so no confirmation or probe job was created.
- Independent GPT-5.6 Sol review caught one monitor-prompt contradiction: a
  later sentence could have ordered General ranking despite the explicit b3
  block. The hourly monitor now makes ranking and confirmation submission
  conditional on the user's separate, disclosed protocol resolution and a
  valid frozen ranking artifact. No scientific artifact or live job changed.

## B3 continuation is terminally lost; b4 starts and b5 preflights — 21:47 SAST

- Exact b3 continuation `1279472` passed its 695-file source/config check,
  exact runtime check, and wrote resume manifest SHA-256
  `30bfda0598bd462208411ba332ed0e2c83ea1065ab3e107c3f30d4b4098c028c`.
  It then loaded the frozen model and began dataset processing, but failed
  after `00:02:07` on an SSL `UNEXPECTED_MESSAGE` while reading the unchanged
  AfriHG Zulu validation file from `raw.githubusercontent.com`. It produced no
  resumed training step, validation artifact, or final adapter. Because model
  and data payload had already begun, the uniformly frozen terminality rule
  makes this the one terminal b3 continuation: preserve it and never rerun it.
  The strict all-11 General ranking is now blocked unless the user explicitly
  authorizes a disclosed protocol deviation; it will not be changed silently.
- The released position activated the next frozen candidate-order
  continuation, b4. Its original root is preserved in a read-only
  `195,543,040`-byte archive with SHA-256
  `ab2c93160222ba8bfe0a48e086a67a1548cbcbfcc67ce0977e60aa583a4ac399`.
  Candidate-specific wrapper SHA-256 is
  `770834281b39f36f82ef095d80caa6eaa9c1bb2c3101e152b678307d1d3b249a`;
  prospective implementation-note SHA-256 is
  `1f2f9e811dcb868066d61409d58e14b40c91409a2d2f6a69e616585c92501db8`.
  Compute-node preflight `1279651` completed `0:0`, verified all frozen files,
  runtime, archive, manifest, and checkpoint hashes, and produced durable
  preflight SHA-256
  `64f697b6f839b92ffbd87ee67f80ca2565562b515a23ec91d66d3dec430ad2e3`.
  Exact b4 continuation `1279652` is running on A100-80GB and wrote
  resume-manifest SHA-256
  `dd11698c0340a0e78f2ac84f33000c706d2fa74d9e5de9e7531b7100b44a3dc9`;
  it advanced directly from checkpoint `10912` to steps `10913-10924`,
  proving that it did not restart at step zero.
- B5's original root is now preserved in a read-only `245,811,200`-byte
  archive with SHA-256
  `3f52518bf028a1a921ea7c4718654b5bd21e3f27115aa761c8a689dd92051595`.
  Candidate-specific wrapper and prospective implementation-note SHA-256
  values are
  `40f4ec9323c991d9cbaee19f03324d832b444f474b24f1ebd029c806bb575c27`
  and `84533964b9d8838eaff286a2d5a772ca1e4d2c58a3611169f4af1eb0ffe391ae`.
  Compute-node no-launch preflight `1279656` is `AssocGrpGRES`-pending as the
  fourth active owned submission; no b5 scientific continuation has started.
- B1/b2 `1279470/1279471` remain healthy near steps `11486/11492`, with no
  targeted fault marker. General Stage-B remains `2/8` terminal-valid and the
  global family freeze remains `4/8`. Quota is home `88.6%`, scratch `45.3%`;
  adapter held-out access remains zero and Sheet E/F/G remain blank.

## B7 verifies; corrected b1/b2 recoveries resume exactly — 20:42 SAST

- General b7 `1277552` completed `0:0`. Its terminal step-10912 artifact has
  sidecar-matched SHA-256
  `884417617c465af2b91efb91d4277fdce27681e86a40e5b6e040a139cb6f22cb`
  and exact `22,167`-row six-family coverage. The frozen within-run callback
  retained checkpoint `5456` at validation-only macro NLL
  `0.9166715052168936`; the other scheduled boundaries were
  `0.9391455045913958/0.9417427433072895/0.9895616203288914` at steps
  `2728/8184/10912`. CPU verifier `1279479` completed `0:0` and proved exact
  retained-to-final equality across `424` keys and `76,410,112` values.
  General Stage-B therefore advances to `2/8` terminal-valid. This is
  validation-only recipe evidence, not a held-out result.
- General b6 `1277424` reached the fixed wall and ended `TIMEOUT`/`0:0` with
  no final adapter. Its complete checkpoint-10912 adapter, optimizer,
  scheduler, RNG, and trainer-state SHA-256 values are
  `0cb7799764dcf92c10af4023c4dbb99312aae39d0ef8cb65580509bbe51f9697`,
  `1fd8255bc8d0035a8b787164f8f21474a9c1a83fb19f149fbd76e6f2a8f7303e`,
  `dbdb451011aacdbeedc3b19ff1bddd1d5dc1d10fb96ec8b65ec0be5b0331227c`,
  `3180a03567b7f40c2f694e184313fd4ca7212fda830074f2e9bc3350b97907e5`,
  and `7424d09dfa63eaa2d8d9dd2bbd87bda84e0f68f66d4b36308820e2cb472a86bf`.
  Preserve its root and log; its same-trial continuation remains later in the
  frozen candidate order.
- Initial b1/b2 recoveries `1278130/1278993` failed at zero seconds before
  payload loading because Slurm resolved the wrapper-relative recovery bundle
  under `/var/spool/slurmd.spool/job*`. They produced no model, data,
  training, validation, evaluator, or mutable result work. The prospective
  path-only correction is frozen in
  `2026-08-31-pure-gdn-general-recovery-bundle-path-launch-correction.md`,
  SHA-256 `5248bf2d821457b0b7856142321eff5a7731ff64f7f0c1632ce16520882ddb54`;
  it preserves the wrapper byte-for-byte and explicitly exports the immutable
  bundle path.
- Corrected exact b1/b2 recoveries `1279470/1279471` passed all `695` source
  and config hashes, exact runtime and checkpoint-state checks, created
  resume-specific manifests with SHA-256
  `adb50b6bbe8a006c08467ce40f0d1e594ba9a58871807192dc9645e21334cc78`
  and `6307c9510df01cebb958ca1c04e3063f9ccbb4e2b3625c31a73ee2dc78a3c00a`,
  and proved trainer-level continuation from checkpoint `10912` by advancing
  directly to steps `10913-10919`; neither restarted at step zero. First b3
  recovery `1279472` is `AssocGrpGRES`-pending. Quota is home `88.6%`,
  scratch `45.1%`; adapter held-out access remains zero and Sheet E/F/G stay
  blank.

## B6 reaches a complete step-10912 boundary — 18:35 SAST

- General b6 `1277424` remains healthy on A100-80GB and resumed training after
  writing its complete scheduled checkpoint `10912`. The sidecar-verified
  validation artifact has SHA-256
  `b35cc3f08fd0d50edd7910f6dc98dba504996b02e41aaa92445b7c2f4f018f19`,
  exact `22,167`-row six-family coverage, and validation-only macro NLL
  `1.0334865102881885`. It improves b6's own earlier values
  `1.4567470667356914/1.079590829544548/1.0499393719779446`, so checkpoint
  `10912` is its current within-run retained boundary. This changed no
  cross-candidate ranking, submission order, retry rule, or budget.
- B6/b7 `1277424/1277552` remain running after `22:24/20:44`; exact b1/b2
  recoveries `1278130/1278993` remain `AssocGrpGRES`-pending behind two
  other-user jobs. Quota remains home `88.6%`, scratch `45.1%`; adapter
  held-out access is zero and Sheet E/F/G remain blank.

## Post-hoc LLaMA lane freezes; pure-GDN queue remains saturated — 14:31 SAST

- Pure-GDN General b6/b7 `1277424/1277552` remain healthy on A100-80GB
  after `18:20/16:40`; exact b1/b2 recoveries `1278130/1278993` remain
  `AssocGrpGRES`-pending. Two other-user jobs occupy the other association
  cards and were not modified. Quota remains home `88.6%`, scratch `45.1%`;
  adapter held-out access is zero and Sheet E/F/G remain blank.
- Post-hoc LLaMA-125M T2X a2 seed 87 is terminal-valid. Validation chrF is
  `44.83365297535967`, `47.02310995608528`, `49.100440742466795`, and
  `48.859427683935806` at steps `483/966/1449/1932`; checkpoint `1449` is
  retained. All four artifacts have exact 64-row/index coverage, beam 5, no
  empty predictions, and 64 unique predictions. Exact retained-to-final
  equality passes across `124` keys and `67,929,088` values. Retained/final
  SHA-256 values are
  `28eee496aab7c334150a731c1dcabcaca2badc6c11bb8a8693a972692cf06369`
  and `72445d0f62b0003aee309324bdf083d8e2d07712abc58c26e9d3b9f3e460b74e`.
  Preserve it and never rerun it; confirmation progress is `4/4`.
- The immutable HPO utility applied the frozen arithmetic three-seed mean.
  B7 scores `50.25952890593673/49.89574145828771/49.52933730149589`
  for seeds `13/42/87`, mean `49.89486922190678`, sample SD
  `0.3650965836545177`. A2 scores
  `48.084019151368054/48.52835434271314/49.100440742466795`, mean
  `48.57093807884933`, sample SD `0.5095470965969793`. Post-hoc LLaMA T2X
  therefore freezes b7; its seed-42 checkpoint `1932` is the representative
  adapter, with retained/final SHA-256
  `0b539e0a32cb90e1dc0f89966aa76d565df2beeca37554a6a06f6d0edee51b4b`
  and `855937a56dbcfc575bcf00d31c992dcd482d5882ad78cc91048ded5c780d5374`.
  Read-only confirmation-ranking artifact SHA-256 is
  `6e2245989d95c9dfbe4cb7d3ffcab32b1ae7e1175a18e20ca98616a1f30329a2`.
  This post-hoc freeze used validation only and cannot affect pure-GDN.

## B5 timeout is preserved; b2 recovery queues; final LLaMA confirmation starts — 13:32 SAST

- General b5 job `1277423` reached the fixed 24-hour wall and ended
  `TIMEOUT`/`0:0` after `1-00:00:06`, with no final adapter. Its numerically
  latest complete scheduled state is checkpoint `10912`; adapter, optimizer,
  scheduler, RNG, and trainer-state SHA-256 values are
  `252af47bbbb411a87ed3bc59bec8596a2845d8afc9c0cdfb093759be71434d5f`,
  `7853c01504a5d823055b33eb31bd10400736a1750217f0074b29d2cf04018294`,
  `b403be7999c394122e0db5d991abdd751bfe923ce60b4cb961d4b1de55a58413`,
  `3a5fa191be89653519d53a698bb731547927ec65d27015072e0de1b7e71c27d3`,
  and `51a64b5a7c3f09296dd8cb7b61bd8f4ef54bb0202bae9e695b66fc8f9d695f70`.
  Preserve its root and log; its one allowed same-trial continuation follows
  b1/b2/b3 and eligible b4 under the frozen later-timeout amendment.
- B5's release reduced the active owned count to three. Exact b2 recovery
  passed the frozen wrapper hash, absent-final-output, no-active-duplicate,
  immutable archive, 695-file source/config, runtime, manifest, and all five
  checkpoint-state checks. It was submitted once as `1278993` and is
  `AssocGrpGRES`-pending behind b1 `1278130`; b6/b7 `1277424/1277552` remain
  healthy on A100-80GB. Two other-user jobs hold the remaining association
  cards and were not modified. Quota is home `88.6%`, scratch `45.1%`;
  adapter held-out access is zero and Sheet E/F/G remain blank.
- Post-hoc LLaMA-125M T2X a2 seed 13 is terminal-valid. Validation chrF is
  `43.48261563532691`, `47.025476607703375`, `47.85185955501886`, and
  `48.084019151368054` at steps `483/966/1449/1932`; checkpoint `1932` is
  retained. All four artifacts have exact 64-row/index coverage, beam 5, no
  empty predictions, and 64 unique predictions. Exact retained-to-final
  equality passes across `124` keys and `67,929,088` values. Retained/final
  SHA-256 values are
  `72bd9279f7f8e9e98de77eb3a6e17c47527a3315f3d37f7067b632c1409b0633`
  and `8a8647ecb9553ef3d61122face69a0e413ba3546fdc6979562b34f3684afbce4`.
  Preserve it and never rerun it; confirmation progress is `3/4`.
- Frozen a2 seed 87 passed all absence, ranking, immutable provenance,
  prior-manifest, dry-run, and GPU-isolation checks before starting in tmux
  `sallm-llama125-t2x-a2-s87`. It verified 694 immutable files, loaded exact
  `3,859/460` rows, and entered its `1,932`-step schedule. Execution-manifest
  and HPO-trial SHA-256 values are
  `6be8bfd8f79e9d66f5223b20fbe58b1628d5e2329cde1de4c031d3d9c5d368d3`
  and `2781dcbd1d77e938fbc4a94c67bfc376652030a5ddbcccfe6d28290460ff9f78`.
  GPU 1 remains outside the frozen serial protocol and pure-GDN is unaffected.

## Second post-hoc LLaMA confirmation verifies; third starts — 12:31 SAST

- Pure-GDN General b5/b6/b7 `1277423/1277424/1277552` remain healthy on
  A100-80GB after `23:21/16:20/14:39`; targeted fault scans are empty. Exact
  b1 recovery `1278130` remains `AssocGrpGRES`-pending and will take b5's slot
  when its fixed wall releases. Quota remains home `88.6%`, scratch `45.1%`;
  adapter held-out access is zero and Sheet E/F/G remain blank.
- Post-hoc LLaMA-125M T2X b7 seed 87 is terminal-valid. Validation chrF is
  `45.43849823497772`, `48.33804044559621`, `49.52933730149589`, and
  `49.28995414995973` at steps `483/966/1449/1932`; the frozen rule retains
  checkpoint `1449`. All four artifacts have exact 64-row/index coverage,
  beam 5, no empty predictions, and 64 unique predictions. Retained-to-final
  equality is exact across `124` keys and `68,743,168` values. Retained/final
  SHA-256 values are
  `00689266dbcca3578401b24062052846c77d3e76e3736e9901fb083b7f76f077`
  and `c1140d00df262c02b9d971017095e1f018142a4da4cd370971f22c53f066f2e3`.
  Preserve this confirmation and never rerun it; confirmation progress is
  `2/4`.
- With GPU 0 isolated, frozen a2 seed 13 passed all absence, ranking,
  immutable-source/model/tokenizer/registry, prior-manifest, dry-run, and GPU
  checks before starting in tmux `sallm-llama125-t2x-a2-s13`. It verified all
  694 immutable files, loaded exact `3,859/460` rows, and entered its
  `1,932`-step schedule. Execution-manifest and HPO-trial SHA-256 values are
  `a6d62d02bf033329b95498fed5cfa1e42c15da831699987223d13011e2bfd81b`
  and `3b70a1c796974b28f690f935ae466b47cff2b5c5ffe7235af5456377a5827761`.
  GPU 1 remains outside the frozen serial protocol and pure-GDN is unaffected.

## Frozen second post-hoc LLaMA confirmation starts — 11:32 SAST

- Pure-GDN General b5/b6/b7 `1277423/1277424/1277552` remain healthy on
  A100-80GB after `22:20/15:19/13:39`; targeted fault scans are empty. Exact
  b1 recovery `1278130` remains `AssocGrpGRES`-pending while those three jobs
  and one other user's job hold the four association cards. B5 crossed its
  complete step-10912 state and continues unchanged toward the fixed wall.
  Quota is home `88.6%`, scratch `45.1%`; adapter held-out access is zero and
  Sheet E/F/G remain blank.
- Kombuys GPU 0 became genuinely isolated. Frozen b7 seed 87 passed absent
  output/log/session, ranking-order, registry, deployment-manifest, tokenizer,
  prior execution-manifest, 694-file source/config, model-artifact, and fresh
  GPU-isolation checks. Its dry run created no scientific output.
- B7 seed 87 then started serially in tmux
  `sallm-llama125-t2x-b7-s87`, verified all 694 immutable files, loaded exact
  `3,859/460` train/validation rows, and entered its `1,932`-step schedule on
  GPU 0. Execution-manifest and HPO-trial SHA-256 values are
  `108d532dcd57e757658d2712fb821613c5f958f0bb088248e6d5dcab5f30e465`
  and `d559899dc4fcca428e34e6057b6f04fefdd8997e177b2036541ddd9f07224eb9`.
  GPU 1 remains outside the frozen serial LLaMA protocol and pure-GDN is
  unaffected.

## First post-hoc LLaMA confirmation verifies; next waits for isolation — 10:30 SAST

- Pure-GDN General b5/b6/b7 `1277423/1277424/1277552` remain healthy on
  A100-80GB after `21:22/14:21/12:40`; targeted fault scans are empty. Exact
  b1 recovery `1278130` remains `AssocGrpGRES`-pending while those three jobs
  and one other user's job hold the four association cards. Quota remains home
  `88.6%`, scratch `45.0%`; adapter held-out access is zero and Sheet E/F/G
  remain blank.
- Isolated post-hoc LLaMA-125M T2X b7 seed 13 is terminal-valid. Validation
  chrF is `44.062487091935004`, `48.44505602089041`, `49.17059584733388`, and
  `50.25952890593673` at steps `483/966/1449/1932`; the frozen rule retains
  checkpoint `1932`. All four artifacts have exact 64-row/index coverage,
  beam 5, no empty predictions, and 64 unique predictions. Retained-to-final
  equality is exact across `124` keys and `68,743,168` values. Retained/final
  SHA-256 values are
  `30648a7705a2442385b3852d3ac140ba0eb524d81c3b2a386b906d20e479d25e`
  and `3a50e7ec293f615d49d9074afa5365bd7ef6ccf98f2a99744ea955bcb3682230`;
  execution-manifest SHA-256 remains
  `44a9584669519b84f1cba10c4458347d86ed1d2aa91fa938b864363c5e236d37`.
  Preserve this confirmation and never rerun it.
- Frozen next confirmation b7 seed 87 remains absent and unlaunched because
  another user's process owns Kombuys GPU 0. GPU 1 remains outside this frozen
  serial LLaMA protocol. No foreign process was changed; launch will occur only
  after a fresh isolation and no-duplicate preflight.

## Post-hoc LLaMA seed-42 grid freezes; first confirmation starts — 09:32 SAST

- Pure-GDN General b5/b6/b7 `1277423/1277424/1277552` remain healthy on
  A100-80GB near steps `10259/6708/5808`; targeted fault scans are empty.
  Exact b1 recovery `1278130` remains `AssocGrpGRES`-pending. Quota remains
  home `88.6%`, scratch `45.0%`; adapter held-out access is zero and Sheet
  E/F/G remain blank.
- Isolated LLaMA-125M T2X b7 is terminal-valid. Validation chrF is
  `44.73004698015957/48.33975914732357/49.69324873709872/49.89574145828771`
  at steps `483/966/1449/1932`; the frozen rule retains checkpoint `1932`.
  All four artifacts have exact 64-row/index coverage, beam 5, no empty
  predictions, and 64 unique predictions. Retained-to-final equality is exact
  across `124` keys and `68,743,168` values. Retained/final SHA-256 values are
  `0b539e0a32cb90e1dc0f89966aa76d565df2beeca37554a6a06f6d0edee51b4b`
  and `855937a56dbcfc575bcf00d31c992dcd482d5882ad78cc91048ded5c780d5374`.
  Preserve b7 and never rerun it; the seed-42 LLaMA grid is now `11/11`.
- The complete verified grid was ranked once by validation chrF. B7 ranks
  first at `49.89574145828771` and a2 second at `48.52835434271314`.
  Ranking artifact SHA-256 is
  `deb603fa4e9a40d2b77b326360831cd6baa865c83bdae100afd407582c009fc8`;
  it freezes confirmation order b7 seed 13, b7 seed 87, a2 seed 13, a2 seed
  87. The first confirmation is running in tmux
  `sallm-llama125-t2x-b7-s13` after exact provenance, absence, ranking, and
  GPU-isolation checks. It loaded exact `3,859/460` train/validation rows.
  Execution-manifest and HPO-trial SHA-256 values are
  `44a9584669519b84f1cba10c4458347d86ed1d2aa91fa938b864363c5e236d37`
  and `bb78549de426bf641e110da1ed2e3e653a3a1aac5902e69affbaad1b8f1d3dec`.
  This post-hoc lane remains isolated from pure-GDN.

## General b6/b7 boundaries verify; post-hoc LLaMA b6 verifies and b7 starts — 08:47 SAST

- Pure-GDN General b5/b6/b7 `1277423/1277424/1277552` remain healthy on
  A100-80GB near steps `9885/6336/5456`; targeted fault scans are empty.
  B6 step `5456` and b7 step `2728` each have exact `22,167`-row coverage and
  matching sidecars. Their validation-only macro NLL and artifact SHA-256
  values are `1.079590829544548`/
  `839f4a457ee85a5a5de3794476e246f8458eadfb61177426b776bca0b2281a45`
  and `0.9391455045913958`/
  `5ebcaba9c791e718475f2304272318667742f4eee0d51f0ad35ce257aef9830a`.
  B6 improved only its own retained boundary; b7's artifact is its first
  boundary. Neither affected cross-candidate ranking, scheduling, retry, or
  budget. Exact b1 recovery `1278130` remains `AssocGrpGRES`-pending. Quota
  remains home `88.6%`, scratch `45.0%`; adapter held-out access is zero and
  Sheet E/F/G remain blank.
- Isolated LLaMA-125M T2X b6 is terminal-valid. Validation chrF is
  `34.47308361229159/38.734704730281706/39.75908363480086/39.990824045303356`
  at steps `483/966/1449/1932`; the frozen rule retains checkpoint `1932`.
  All four artifacts have exact 64-row/index coverage, beam 5, no empty
  predictions, and 64 unique predictions. Retained-to-final equality is exact
  across `124` keys and `67,929,088` values. Retained/final SHA-256 values are
  `2e9213125e0002689deaa2fb1bc4cfda4d0d6fd1c13744fc487a65ddbc159f70`
  and `0c38ca3eeb68aef158a45fe6f877905ab86adc8e355892fc7c80eb2cfa48abe3`.
  Preserve b6 and never rerun it; the seed-42 LLaMA grid is now `10/11`.
- With GPU 0 isolated and b7 output/log/session absent, unchanged b7 passed
  frozen registry, 694-file source, six-file base-model, tokenizer, and GPU
  checks, then started serially in tmux `sallm-llama125-t2x-b7`. It loaded
  exact `3,859/460` train/validation rows and entered its `1,932`-step run.
  Execution-manifest and HPO-trial SHA-256 values are
  `e143827b41d93c7be5d7c31257c06ae1677f56b7757839f5d2ae524fec4e697c`
  and `db5e7dcf29982d8078045c6de89ffc5b0162a0f276664193ca221ed622f121bd`.
  This post-hoc lane remains isolated from pure-GDN.

## Post-hoc LLaMA b5 verifies; b6 starts — 07:26 SAST

- Pure-GDN General b5/b6/b7 `1277423/1277424/1277552` remain healthy on
  A100-80GB near steps `9174/5629/4881`; targeted fault scans are empty.
  Exact b1 recovery `1278130` remains `AssocGrpGRES`-pending while those three
  jobs and one other user's job hold the four association cards. Quota remains
  home `88.6%`, scratch `45.0%`; adapter held-out access is zero and Sheet
  E/F/G remain blank.
- Isolated LLaMA-125M T2X b5 is terminal-valid. Validation chrF is
  `39.66137939746889/42.92873157650323/43.29533776349944/43.51713717457478`
  at steps `483/966/1449/1932`; the frozen rule retains checkpoint `1932`.
  All four artifacts have exact 64-row/index coverage, beam 5, no empty
  predictions, and 64 unique predictions. Retained-to-final equality is exact
  across `124` keys and `67,929,088` values. Retained/final SHA-256 values are
  `48d43951fe02539e655d4ec2d386b178bb895356972c1d7f5e602305ad1b01fd`
  and `2b6405c6c7f8fb21bfd485f7a5c6acb12bfa7348d8d4988cbc513fb721d2011c`.
  Preserve b5 and never rerun it; the seed-42 LLaMA grid is now `9/11`.
- With GPU 0 isolated and b6 output/log/session absent, unchanged b6 passed
  frozen registry, 694-file source, six-file base-model, tokenizer, and GPU
  checks, then started serially in tmux `sallm-llama125-t2x-b6`. It loaded
  exact `3,859/460` train/validation rows and entered its `1,932`-step run.
  Execution-manifest and HPO-trial SHA-256 values are
  `b98c410be75da35610669e218ae289aa8d840b1f50b257aa3086c01cb365fd56`
  and `a50fe26f0445c2dacb398a4fea47aa1934d4b3ad8e559c72960e9603048ad345`.
  This post-hoc lane remains isolated from pure-GDN.

## Pure-GDN b5 advances; post-hoc LLaMA b4 verifies and b5 starts — 06:27 SAST

- General b5 job `1277423` remains healthy on A100-80GB and passed its frozen
  step-`8184` validation boundary with exact `22,167`-row six-family coverage,
  macro NLL `0.9589405712228745`, and matching artifact/sidecar SHA-256
  `b7ddf007cd38c92e52cb920ab60f29a17d83aaa305f15ce9d5e343f81f126039`.
  This improves only b5's own retained boundary and did not affect candidate
  ranking, scheduling, retry, or budget. B5/B6/B7 `1277423/1277424/1277552`
  are running near steps `8591/5131/4331`; targeted fault scans are empty.
  Exact b1 recovery `1278130` remains `AssocGrpGRES`-pending behind the three
  running jobs and one other user's card. Quota remains home `88.6%`, scratch
  `45.0%`; adapter held-out access is zero and Sheet E/F/G remain blank.
- Isolated LLaMA-125M T2X b4 is terminal-valid. Validation chrF is
  `41.14859978077697/43.53834092412122/44.57380920361152/44.54892913287072`
  at steps `483/966/1449/1932`; the frozen rule retains checkpoint `1449`.
  All four artifacts have exact 64-row/index coverage, beam 5, no empty
  predictions, and 64 unique predictions. Retained-to-final equality is exact
  across `124` keys and `67,522,048` values. Retained/final SHA-256 values are
  `2f4c1a6844fc902423ccf67a215553b63c9a57a573f431dc5e91691449adec37`
  and `bbb2ccf4a3227c01d31b241c1b192c8c395561d7b05e5887872bcb4aca96c7d3`.
  Preserve b4 and never rerun it; the seed-42 LLaMA grid is now `8/11`.
- With GPU 0 isolated and b5 output/log/session absent, unchanged b5 passed
  frozen registry, 694-file source, six-file base-model, tokenizer, and GPU
  checks, then started serially in tmux `sallm-llama125-t2x-b5`. It loaded
  exact `3,859/460` train/validation rows and entered its `1,932`-step run.
  Execution-manifest and HPO-trial SHA-256 values are
  `50f222365ae659da1d65c163b274b31c606f3fb80020cbfc29696849cc7c2633`
  and `a9994e6e13477211338efa04811104372bae54f63ad216640e366f5f8465efe9`.
  This post-hoc lane remains isolated from pure-GDN.

## Post-hoc LLaMA b3 verifies; b4 starts — 05:31 SAST

- Isolated LLaMA-125M T2X b3 is terminal-valid. Validation chrF is
  `40.69645308307086/44.201515387271336/44.865800837994094/44.89238882495095`
  at steps `483/966/1449/1932`; the frozen rule retains terminal checkpoint
  `1932`. All four artifacts have exact 64-row/index coverage, beam 5, zero
  empty predictions, and 64 unique predictions.
- Retained-to-final equality is exact across `124` keys and `67,929,088`
  values. Retained/final adapter SHA-256 values are
  `c216a948509ea98f3cef4a1d8386c3716e6902666b2a69bbd1fc3bc1c951e115`
  and `fa7b87bfebf506d54f0b5b1000cce435b1fc0d326c6b967c3e183b39dd35c9c0`.
  Preserve b3 and never rerun it; the seed-42 LLaMA grid is now `7/11`.
- With GPU 0 isolated and b4 output/log/session absent, unchanged b4 passed
  frozen registry, 694-file source, six-file base-model, tokenizer, and GPU
  checks, then started serially in tmux `sallm-llama125-t2x-b4`. It loaded
  exact `3,859/460` train/validation rows and entered training. Its
  execution-manifest and HPO-trial SHA-256 values are
  `27e60edb3b86197584fe2038c3ac4cd14a28005af92e39d0df912090a88378e1`
  and `39463eee346bd932d2a80415352cb3f218525cb51212fe0663841ae67e52f39f`.
  This post-hoc lane remains isolated from pure-GDN.

## B4 times out cleanly; exact b1 recovery is queued — 04:25 SAST

- General b4 job `1277379` reached the fixed 24-hour wall at training step
  `12115/13640` and ended `TIMEOUT`, exit `0:0`, after `1-00:00:10`. It
  produced no final adapter. Its numerically latest complete scheduled state,
  checkpoint `10912`, is intact with SHA-256 values
  `edc7cdac66426ac5824903492a65da3987503309b5d4cbdfb06af0406c29be66`,
  `f5cbd3ac2e9d0ab27464fde538bf01e3a2eb75f1652ea2bb260d668fdba04e5d`,
  `ce4fcc97ebd7b7c36ccf7a5faa5df7b8a4794b4841d3f329517b029a0eff92a5`,
  `d31146d7c73d1c62b0541ebcd0c731dcb2a9aa1652731b25333180d24e9d170f`,
  and `13f8c6df16c4ac250b17642868499db350be5a0bf98b0160c8f7c4a52189abac`
  for adapter, optimizer, scheduler, RNG, and trainer state. Preserve its root;
  its prospective same-trial continuation remains after the already frozen
  b1/b2/b3 recovery order.
- B4's release reduced the active owned association count to three. The exact
  b1 checkpoint-10912 recovery passed absent-final-output, no-active-duplicate,
  immutable archive, execution-runtime, and state-hash preflight using the
  unchanged wrapper SHA-256
  `b3cd62c36dd368e4ee90f32324651f7de04dbcce8bc6914a260e1b48c7a730fa`.
  It was submitted once as job `1278130` with the frozen A100-80GB/24-hour/
  eight-CPU request and is `AssocGrpGRES`-pending. B1's recovery manifest is
  still absent because payload execution has not started.
- B5/B6/B7 `1277423/1277424/1277552` remain healthy on A100-80GB near steps
  `7730/4204/3282`; targeted fault scans are empty. Quota is home `88.6%` and
  scratch `45.0%`. Adapter held-out access remains zero and Sheet E/F/G remain
  blank.

## B4 and b6 validation boundaries verify — 02:16 SAST

- Quota-first HEX readback is home `88.6%` and scratch `45.0%`.
  B4/B5/B6/B7 remain healthy on all four A100-80GB cards near steps
  `11187/6606/3075/2310`; targeted fault scans remain empty and no recovery
  slot has released.
- B4 step `10912` has exact `22,167`-row coverage, macro NLL
  `0.9606701593321049`, and matching artifact/sidecar SHA-256
  `a1d589a017078ce78286279006009138d60834e0b8ece07ba069eb15549e4e46`.
  It is a within-run non-improvement relative to b4 step `8184`, so the frozen
  callback retains step `8184`; this does not affect the later continuation
  checkpoint rule, which is based only on the numerically latest complete
  scheduled training state after a 24-hour timeout.
- B6's first frozen boundary at step `2728` has exact `22,167`-row coverage,
  macro NLL `1.4567470667356914`, and matching artifact/sidecar SHA-256
  `6c24d2f8b77ea7c6f5abdb2d7739ff03a809d4159c4235d81109cacb00e4c518`.
  Both artifacts are validation-only within-run retention evidence; neither
  changed candidate ranking, scheduling, retry, or budget.

## Isolated post-hoc LLaMA b2 released — 02:18 SAST

- Kombuys GPU 0 became isolated, and unchanged LLaMA-125M T2X b2 passed the
  frozen absent-output, no-duplicate, 694-file source, six-file model,
  tokenizer, registry, and GPU-isolation preflight. It is running in tmux
  `sallm-llama125-t2x-b2`, loaded exact `3,859/460` train/validation rows,
  and entered training. Execution-manifest SHA-256 is
  `651e627bd83ff23887bd30e867fb66dc9d7f270bfd30ccfa68717790736a8abd`.
  This separate post-hoc run uses no HEX card and does not alter pure-GDN.

## Post-hoc LLaMA b2 verifies; b3 starts — 03:18 SAST

- LLaMA T2X b2 is terminal-valid with validation chrF
  `34.25156415347831/39.492568075259186/40.59239731926782/39.96438880908078`
  at steps `483/966/1449/1932`; checkpoint `1449` is retained. Exact
  retained-to-final equality passed across `124` keys and `67,522,048`
  values. The first boundary has one scored empty prediction; retained and
  terminal boundaries have none. B2 is preserved and will not rerun.
- B3 then passed the frozen isolation and provenance checks and started on
  Kombuys GPU 0 with exact `3,859/460` rows. Execution-manifest SHA-256 is
  `a0d1f5bf5fc1d87acf42d1ec59a5381ababe9f1d630599a6eaca0457ee3e4fda`.
  Pure-GDN HEX work is unchanged.

## General Stage-B remains healthy — 00:16 SAST

- Quota-first HEX readback is home `88.6%` and scratch `44.9%`.
  B4/B5/B6/B7 `1277379/1277423/1277424/1277552` are all running on the
  four A100-80GB cards near steps `10284/5552/2151/1252`. Targeted fault
  scans are empty, so no slot has released for the frozen b1 recovery.
- B5 step `5456` produced a sidecar-verified validation-only General artifact
  with SHA-256
  `6239224a015003aef3f3a7cd1548bf9c06ab176cb70ae74f419e79fe5af0da82`,
  exact `22,167`-row six-family coverage, and macro NLL
  `0.9722614411299868`. It improves b5's own step-2728 value and is only an
  unchanged within-run retention fact; it was not used for cross-candidate
  ranking, scheduling, retry, or budget changes.
- Adapter held-out access remains zero, historical base outputs remain
  quarantined, and Sheet E/F/G remain blank.

## Non-General family-wise results become the priority — 23:27 SAST

- The user prospectively removed General as the release barrier for complete
  non-General Mono/Multi results. The frozen amendment is
  `2026-08-31-pure-gdn-nongeneral-familywise-heldout-acceleration-amendment.md`,
  SHA-256
  `dcfaa036979b6ef02f6d2cefde9b5118a282528d669fc9ac9ab451a7f33f641b`.
  No current pure-GDN adapter official held-out metric had been opened or
  scored before the amendment.
- The already running exact General b1/b2/b4 continuations
  `1279470/1279471/1279652` remain untouched and will be preserved. Pending
  b5 no-launch preflight `1279656` was confirmed `PENDING`, elapsed zero, and
  cancelled before model, data, validation, evaluator, checkpoint, or result.
  No further General work may displace non-General work.
- The official-test barrier is now family-local. A family requires a frozen
  corrected Multi winner, all applicable validation-frozen Mono adapters, exact
  provenance/coverage/roundtrip verification, and the metric-free exact
  checkpoint CUDA gate. That family may then run its applicable official tests
  once and fill only its verified E/F/G cells. Test metrics are report-only and
  cannot influence any unfinished work.
- The exact-checkpoint verifier now exercises the loaded checkpoint with BF16
  forward/backward before deterministic generation. The regression test first
  failed under the old behavior, then the focused test and full verifier suite
  passed; final local result is `10 passed`, with Ruff clean. Verifier and test
  SHA-256 values are
  `0c242dcfa6d127ddff15c34b17e29b4a0bcb920cd9faa8f08a8470c0818879d1`
  and
  `37a4ca1c2457b25d682c6d32146052270f0266e12b424cc889c86cdd6c1cc7cc`.
- Quota at the cancellation readback is home `88.6%` and scratch `45.3%`.
  Adapter held-out access remains zero; no Sheet E/F/G value has changed.
- Metric-free exact-checkpoint CUDA gate `1279742` is submitted once and
  `AssocGrpGRES`-pending behind the three active General continuations. It uses
  immutable snapshot
  `pure-gdn-exact-checkpoint-gate-20260831-0c242dcf`; verifier, wrapper, and
  deployment-manifest SHA-256 values are
  `0c242dcfa6d127ddff15c34b17e29b4a0bcb920cd9faa8f08a8470c0818879d1`,
  `f6bfdf65e941aac336b2d5a5676c01da1e9a88c261e93720adddbc52dd837a5b`,
  and
  `3677977800271391baab9cb0003d3145715d2a1033e34dc3ad1992db051c986a`.
  The manifest verifies `1,424` source/config files and binds the exact frozen
  base checkpoint. Slurm readback is the intended A100-80GB account,
  partition, QOS, one GPU, eight CPUs, one-hour ceiling, and chdir. No duplicate
  gate is authorized.

## Independent review tightens the metric-free gate — 23:50 SAST

- Independent GPT-5.6 Sol review found ambiguous General-only wording and
  three implementation gaps before pending gate `1279742` ran. The job was
  still `PENDING`, elapsed zero, and was cancelled. It and the first two
  immutable gate snapshots are preserved and excluded.
- Prospective correction notes SHA-256
  `02cf0dfc8d6671da04ef4cc27c72ba3d36583b98cf896685447ba9b84425ecda`
  and
  `8c4f010984718303cef1c17e37bab398cad24afe410e8b182a7f2ece975bcbda`
  clarify that only General-scoped work is paused and require exact checkpoint
  path binding, immutable Python/package runtime verification, finite BF16
  loss, gradients for every trainable parameter, and finite gradients.
- Corrected tests first failed for the missing behavior, then all `13` relevant
  tests passed and Ruff was clean. Fresh immutable snapshot
  `pure-gdn-exact-checkpoint-gate-20260831-v3-0c2958dc` verifies `713`
  source/config files plus the exact base-checkpoint artifacts and recorded
  runtime. Deployment-manifest SHA-256 is
  `00c344589f1e728e5c0dd53c1e1338c4d51e2623d188682eac9f2594804288c7`.
- Replacement job `1279752` is submitted once and `AssocGrpGRES`-pending with
  correct A100-80GB, one-hour, eight-CPU, and chdir readback. Adapter held-out
  access remains zero and Sheet E/F/G remain unchanged.
- Follow-up review found that the wrapper-supplied snapshot/checkpoint roots
  were not themselves compared to the manifest roots. Pending v3 gate
  `1279752` was cancelled at elapsed zero. Final prospective correction SHA
  `935ea6dd8f43c0d9ad6dd56bf416ea531829a12dbcab720f77b335d2d76a40fd`
  adds exact non-empty root binding; all `13` tests pass and Ruff is clean.
- Final v4 snapshot verifies `713` source/config files, exact checkpoint
  artifacts, immutable runtime, and both expected roots. Its manifest SHA is
  `f5ec949feef494075e67fe94e55942e9e1a72d943dd24230921b800f937d7c7a`.
  Replacement job `1279771` is submitted once and `AssocGrpGRES`-pending with
  correct Slurm resource readback. Adapter held-out access remains zero.
