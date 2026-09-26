# Pure-GDN validation HPO — 2026-08-30

## All four General candidates now running — 22:12 SAST

- Quota-first HEX readback is home `88.6%` and scratch `44.9%`. B7 job
  `1277552` left `AssocGrpGRES` and started on A100-80GB at `21:49:53`.
  Its immutable 694-file verification passed, the canonical pure-GDN model
  loaded, and training entered the frozen `13,640`-step schedule.
- B4/B5/B6/B7 `1277379/1277423/1277424/1277552` are all running on the four
  A100-80GB cards near steps `9183/4619/1056/162`. Their logs are fresh and
  targeted scans contain no traceback, CUDA OOM, NCCL, segmentation-fault,
  killed, or error marker. All four owned positions remain occupied, so the
  exact b1 recovery is not yet administratively eligible.
- On Kombuys, another user's GPU-0 process holds about `13.6 GiB`; the frozen
  LLaMA b2 output, log, and tmux session remain absent, so b2 was not launched.
  No job or process was modified. Adapter held-out access remains zero and
  Sheet E/F/G remain blank.

## News/SIB/Intent budget reduced prospectively — 21:29 SAST

- The user confirmed that pure-GDN may use A100-40GB or A100-80GB, but never
  both GPU families at once. The active program therefore remains on
  A100-80GB only; no job was moved or cancelled.
- Before any corrected post-BOS News, SIB, or Intent run or metric existed, the
  user approved a uniform budget-limited close-out. The frozen amendment is
  `2026-08-30-pure-gdn-news-sib-intent-budget-limited-closeout-preregistration.md`,
  SHA-256
  `23818b9fbaf064d0e0f131f23678db39701eaf8f6379ffcd203241ebf4e13872`.
- Each family will run `a0/a1/a2` at seed 42, rank by corrected
  validation-only `all_macro_f1`, then run seed 13 once for each top-two
  finalist and select by the two-seed arithmetic mean. Stage-B `b0--b7` and
  seed 87 are uniformly omitted and may not be added later based on results.
- The missing-family budget is now 15 GPU runs rather than 45, reducing the
  post-General training/HPO tail from 69 to 39 launches: four General
  confirmations, 15 missing-family runs, and 20 new Mono adapters. The result
  remains complete but News/SIB/Intent must be disclosed as budget-limited.
- Quota-first live state remained b4/b5/b6 running and b7
  `AssocGrpGRES`-pending on A100-80GB. Adapter held-out access remains zero and
  Sheet E/F/G remain blank.

## Monitor expiry corrected; valid backfill accelerated — 21:23 SAST

- `15 September` is only the heartbeat's automatic safety expiry, not a
  results ETA. The previous wording was misleading. The live pure-GDN
  critical path remains the result program itself.
- Quota-first HEX readback is home `88.6%` and scratch `44.9%`. General
  b4/b5/b6 `1277379/1277423/1277424` are healthy near steps
  `8732/4166/604`; b7 `1277552` remains `AssocGrpGRES`-pending. The fourth
  A100-80GB is held by another user's job, so all four owned submission
  positions remain occupied and the frozen b1 recovery is not yet eligible.
  Slurm's b7 estimate moved from `04:00` to `11:23` on 31 August and is not a
  promise.
- At the observed approximately `5h25m` frozen-boundary cadence, General
  Stage-B has an optimistic lower-bound completion around late afternoon or
  evening on 1 September if each exact continuation succeeds once and every
  released slot is immediately backfilled. This is not the full-results ETA.
- After General seed-42 closes, the remaining program still includes four
  General confirmations, 45 News/SIB/Intent candidate and confirmation runs,
  20 new Mono adapters, the one consolidated eight-family freeze, the final
  one-time adapter evaluations, the metric-free real-FLA CUDA canary, and 14
  corrected raw base lanes. No defensible calendar date exists until the
  three post-BOS a0 canaries provide actual runtimes.
- The active gate monitor now runs every 15 minutes instead of hourly. Its
  prompt explicitly overlaps eligible General confirmations with the three
  missing-family a0 canaries under the four-submission cap, then keeps the
  frozen candidate-major pipeline full and releases each family's
  confirmations as soon as its 11 candidates verify. This is scheduling-only
  acceleration; no candidate, metric, prompt, checkpoint, held-out boundary,
  or scientific order changed.

## Reviewed producer commit preserved locally — final stop

- The reviewed signed producer commit `633a993` is now preserved in the
  persistent repository as local branch `anri/strict-validation-producers`.
  The branch points to the exact reviewed object from the disposable checkout;
  no working-tree switch, push, PR, live job, held-out access, or Sheet change
  occurred.
- Broader repository optimization remains stopped. The hourly monitor owns
  the existing compute gates and will act only when their frozen eligibility
  conditions are met.

## Hourly monitor corrected to reviewed producer commit — 21:20 SAST

- The saved heartbeat still referenced the pre-merge integration commit and
  path. It now binds the signed merged-main producer commit `633a993` at
  `/private/tmp/sallm-merge-GqBPA1/repo`, including the reviewed fail-closed
  General artifact correction and its 131-test verification.
- The hourly schedule, live science rules, user stop boundary, recovery order,
  held-out restrictions, and GPU gates are unchanged. Exact saved TOML
  readback confirms status `ACTIVE` and the existing hourly recurrence.

## Both live execution gates remain closed — 21:15 SAST

- Quota-first HEX readback is home `88.6%` and scratch `44.9%`. General
  b4/b5/b6 `1277379/1277423/1277424` are healthy near steps
  `8634/4071/505`; b7 `1277552` remains `AssocGrpGRES`-pending. Targeted
  fault scans are empty. All four owned submission positions remain occupied,
  so the frozen b1 continuation is not administratively eligible.
- B4's step-8184 artifact and sidecar verify at SHA-256
  `650c10dba6061022009d9c073a9e3017f61e8009aab6bbf621aa1772243bfa60`
  with exact coverage and validation-only macro NLL `0.9568249946661681`.
  B5's step-2728 artifact verifies at SHA-256
  `db9ee9efed937001e1462a6400f53f5b9a8492c0440faa54da38766442d39651`
  with macro NLL `1.0694999094992783`. These are within-run retention facts
  only; no cross-candidate selection occurred.
- Kombuys GPU 0 is not isolated: another user's active ASR process holds about
  `28,038 MiB`. LLaMA b2 still has no output root, log, session, or process,
  and was not launched. GPU 1 is idle, but xLSTM still requires explicit start
  approval; Mamba remains terminally gated. No process, job, held-out artifact,
  or Sheet cell changed.

## Producer review closes with one fail-closed fix — 21:10 SAST

- Independent standards/spec review of the merged-main producer port found
  one substantive issue: General validation JSON and SHA sidecars could be
  replaced at the same global step. `CustomSFTTrainer.evaluate()` now refuses
  different existing artifacts or sidecars; a red-to-green regression test
  proves the public behavior.
- The reported ordinary-classification regression was rejected after direct
  comparison with merged `main`: constrained label scoring already existed
  there, so the port does not change that behavior. Two duplication suggestions
  were deliberately left alone as non-correctness refactors.
- The amended local signed commit is `633a993`. All 131 CPU tests, Ruff lint,
  Ruff formatting, ty, and diff checks pass, and the signature verifies. No
  push, PR, live job, held-out access, or Sheet change occurred.

## Strict producers verify on merged main — 21:00 SAST

- The already-reviewed strict General, POS, News, SIB, and Intent producer
  commit was cherry-picked locally onto merged `origin/main` `94815d5` in the
  disposable branch `producer-port` as signed commit `633a993`. It applied
  without a content conflict.
- The exact merged-main port passes 131 CPU tests, every all-file pre-commit
  hook, Grype ignore-expiry policy, signed-commit verification, diff-check,
  and the exact Grype high-severity gate. No branch was pushed and no PR was
  created, respecting the user's request to stop further repository work.
- The separate overbuilt issue-43 HPO slice remains uncommitted and excluded.
  No live job, held-out artifact, or Sheet cell changed.

## Cleanup stack merged; live science unchanged — 20:55 SAST

- The authorized repository stack merged in order: `#126`, `#127`, `#129`,
  `#130`, `#134`, and `#131`, at merge commits `c3cf6f7`, `c631395`,
  `31ca48f`, `3fc272d`, `ee1d0a6`, and `94815d5`. Every rebased head passed
  the full CPU suite, all-file pre-commit, ignore-expiry policy, signed-commit
  verification, and the exact Grype high-severity gate before merge.
- PR `#118` was closed after `#127` superseded its focused CPU-CI purpose.
  Issue `#125` remains open because `#126` deliberately does not resolve the
  upstream plain-pip/sqlitedict advisory; closing it would lose a real risk
  tracker rather than retire superseded work.
- PR `#131` exposed one Linux-only test-fixture failure after rebase. The
  production code was unchanged; its factory test now pins CPU execution and
  disables bf16/fp16. The corrected head passed 89 CPU tests and all three
  required GitHub checks.
- HEX quota remains home `88.6%`, scratch `44.9%`. General b4/b5/b6
  `1277379/1277423/1277424` are running and b7 `1277552` is
  `AssocGrpGRES`-pending. All four owned submission positions remain occupied,
  so the frozen b1 recovery is still ineligible. No held-out artifact or Sheet
  E/F/G cell was touched.

## Missing-family producers close locally; HPO refactor stopped — 20:00 SAST

- Strict post-BOS News, SIB, and Intent validation producers now exist only on
  the disposable six-PR integration stack. SIB fails closed unless all `2,970`
  rows across six languages and 30 language/prompt cells verify; Intent
  requires all `4,155` rows across four languages and 20 cells. Both require
  constrained-choice scoring and finite validation-only mean-language macro-F1.
  Their exact a0 configs pin the canonical GDN/tokenizer, full target set,
  seed/data-seed 42, optimizer recipe, assistant-only training, and Hub-off
  behavior. Fifty-four focused tests, Ruff, format, and ty pass.
- The proposed issue-43 frozen-trial slice reached 11 focused passing tests,
  but its current implementation is too large for the requested cleanup. It
  remains uncommitted and is explicitly stopped for simplification rather than
  being merged as a framework. No live protocol, job, or artifact depends on
  it.
- General b4/b5 remain healthy and b6/b7 remain pending; no recovery slot has
  opened. Kombuys GPU 0 remains occupied by another user and GPU 1 is idle.
  No job, process, held-out artifact, Sheet, PR, or issue changed.

## Strict producers complete locally; live gates remain closed — 19:47 SAST

- The defensible global pure-GDN freeze remains `4/8`: T2X and NER completed
  their frozen protocols, POS has a prospectively disclosed budget-limited
  closeout, and AfriHG satisfies its original three-seed rule. News, SIB, and
  Intent still require their post-BOS candidate and confirmation protocols;
  General remains incomplete. No complete-table date is evidence-based until
  the first post-BOS News/SIB/Intent runtimes exist.
- On HEX, quota-first readback remained home `88.6%` and scratch `44.9%`.
  General b4/b5 `1277379/1277423` were healthy near steps `7947/3226`;
  b6/b7 `1277424/1277552` remained `AssocGrpGRES`-pending. All four owned
  submission positions remain occupied, so the exact b1 checkpoint-10912
  continuation cannot yet preflight or launch. Other-user jobs were untouched.
- The fresh six-PR integration stack now has strict General, serial POS, and
  post-BOS News validation producers. News requires the exact `3,095` rows,
  Eng/Xho counts `2,360/735`, all ten language/prompt cells, and constrained
  scoring before writing deterministic JSON plus a SHA-256 sidecar. Ordinary
  classification and Mono POS behavior remain unchanged unless their exact
  protocol environment variable is set. The focused suite is `58 passed` and
  Ruff, format, and ty checks pass.
- Provenance and terminal completeness deliberately remain outside the metric
  callbacks. Issue `#43` owns the smallest production HPO slice: immutable
  trial registry/protocol pins, terminal markers written last, exact
  retained-to-final adapter equality, and validation-only ranking that rejects
  partial or timed-out runs. Obsolete dirty-root launcher scripts will not be
  ported.
- On Kombuys, another user's process still occupied frozen LLaMA GPU 0; GPU 1
  remained outside that protocol. LLaMA T2X has `5/11` terminal-valid seed-42
  candidates; b2 remains correctly unlaunched. The separately frozen xLSTM
  checkpoint is locally available and hash-valid, but its GPU-1 structural
  gate still needs explicit start authority. Mamba's frozen target protocol is
  terminally incompatible and may only continue under a separately named,
  prospectively frozen protocol.
- No held-out metric was inspected or used. Adapter held-out access remains
  zero, historical base artifacts remain quarantined, and Sheet E/F/G remain
  blank.

## General queue and artifact integrity remain healthy — 18:01 SAST

- Quota-first HEX readback remains home `88.6%` and scratch `44.9%`. General
  b4/b5 jobs `1277379/1277423` are running on A100-80GB near steps
  `7110/2553`, with fresh logs and no targeted traceback, OOM, or error
  markers. B6/b7 `1277424/1277552` remain `AssocGrpGRES`-pending; their
  output roots and logs are absent.
- B4's execution manifest and step-2728/5456 validation sidecars verify
  exactly. B5's execution-manifest sidecar also verifies. Neither running job
  has a final adapter, terminal result, or retained-to-final roundtrip yet, as
  expected before terminal completion.
- Other-user jobs `1276835/1276836` hold the remaining two association cards
  and were not modified. All four owned submission positions remain occupied,
  so the frozen b1 checkpoint-10912 continuation is scientifically eligible
  but not administratively eligible. No recovery was preflighted or submitted.
  No held-out artifact was accessed; adapter held-out access remains zero and
  Sheet E/F/G remain blank.

## Mono authority clarified; live gates unchanged — 17:35 SAST

- Before any Mono launch, the 9 August authority was made explicit in a
  separately hashed clarification: only the learning-rate component of the
  final hashed family winner transfers to Mono. Rank 16, alpha 32, dropout
  0.05, warmup 0.03, the full target-module set, and the original optimizer
  recipe remain fixed. The enhanced Multilingual winner's other
  hyperparameters do not transfer. Train 20 new language adapters; the frozen
  T2X HPO winner is already the single T2X Xhosa arm and must not be retrained.
- General b1--b3 continuation authority was added after their administrative
  timeouts and interim validation artifacts. The b4--b7 rule was frozen after
  start but before terminal outcomes. Both are metric-independent, same-state
  amendments and must not be reported as unamended preregistered runs.
- At 17:31, quota remains `88.6%/44.9%`; b4/b5 are healthy near steps
  `6845/2288`, and b6/b7 remain association-pending. All four owned submission
  positions remain occupied, so b1 is scientifically eligible but cannot yet
  launch. Kombuys GPU 0 remains occupied by another user, so LLaMA b2 stays
  gated. No held-out access or external state change occurred.

## Both compute lanes remain correctly gated — 17:16 SAST

- Quota-first HEX readback is home `88.6%` and scratch `44.9%`. General b4
  `1277379` and b5 `1277423` are healthy near steps `6717/2157`, with no
  traceback, OOM, or targeted error markers. B6/b7 `1277424/1277552` remain
  `AssocGrpGRES`-pending. All four owned submission positions are occupied.
- The frozen b1 checkpoint-10912 recovery is scientifically eligible but not
  administratively eligible. Submit it only after an owned position releases
  and a fresh absent-output/no-duplicate preflight passes. No job was changed.
- Kombuys GPU 0 remains occupied by another user's ASR process. Frozen LLaMA
  b2 has no output, session, or process, but cannot launch without isolated
  GPU 0. GPU 1 is idle and xLSTM remains blocked on explicit approval for the
  frozen private-checkpoint transfer. No process or file was changed.
- Global freeze remains `4/8`. No held-out access occurred in this action;
  adapter held-out access remains zero, historical base artifacts stay
  preserved, and Sheet E/F/G remain blank.

## Queue remains saturated — 16:58 SAST

- Quota-first readback is home `88.6%` and scratch `44.9%`. B4/B5
  `1277379/1277423` remain running on A100-80GB; b6/b7
  `1277424/1277552` remain `AssocGrpGRES`-pending. All four active owned
  submissions are occupied, so no b1/b2/b3 recovery is eligible.
- No job was changed. Global freeze remains `4/8`, adapter held-out access is
  zero, and Sheet E/F/G remain blank.

## Evidence boundary and prospective timeout correction — 16:30 SAST

- Historical base held-out outputs already exist and remain excluded from all
  model, recipe, checkpoint, retry, and scheduling decisions. The accurate
  current boundary is adapter held-out access zero; Sheet E/F/G remain blank.
  Exactly 14 raw base lanes are quarantined for one post-freeze implementation-
  correction rerun under the separately hashed amendment.
- Before any b4--b7 terminal outcome, a uniform metric-independent rule was frozen:
  an original administrative 24-hour timeout before a final adapter permits
  one same-trial continuation from the latest complete scheduled training-state
  checkpoint. Existing b1--b3 recoveries stay first; later recoveries follow in
  candidate order. No job was submitted or changed.

## Queue remains healthy and saturated — 15:37 SAST

- Quota-first readback remains home `88.6%` and scratch `44.7%`. B4/B5
  `1277379/1277423` are healthy near steps `5772/1222` on A100-80GB;
  b6/b7 `1277424/1277552` remain `AssocGrpGRES`-pending. All four owned
  submission positions therefore remain occupied, so the frozen b1, b2, then
  b3 checkpoint-10912 recoveries are not yet eligible.
- No job was changed. Global freeze remains `4/8`, held-out access remains
  zero, and Sheet E/F/G remain blank.

## B4 second boundary verifies — 15:01 SAST

- B4 `1277379` wrote step-5456 artifact SHA-256
  `0ff8cbfabe4e48b64f6cebd1e2d81ed1dcf3c6441ea1e6f5f1bd50eefa84808e`;
  the adjacent sidecar matches exactly. Coverage remains 22,167 rows across
  all six General validation families and macro NLL is
  `0.9677870068460876`. B4 resumed healthy training after the callback.
- B5 `1277423` remains healthy. B6/B7 `1277424/1277552` remain
  `AssocGrpGRES`-pending, so no exact b1/b2/b3 recovery is eligible yet.
  Global freeze remains `4/8`, held-out access zero, and Sheet E/F/G blank.

## B4 enters its second frozen boundary — 14:48 SAST

- A quota-first readback remains home `88.6%` and scratch `44.7%`. B4
  `1277379` reached step `5456/13640` and entered its second frozen
  validation pass. B5 `1277423` is healthy near step `856/13640`; targeted
  fault scans remain empty.
- B6/B7 `1277424/1277552` remain `AssocGrpGRES`-pending. The four active
  owned submission positions are still b4--b7, so the frozen b1, b2, then b3
  checkpoint-10912 recoveries remain ineligible. No job was changed.
- The reconciled global freeze remains `4/8`: T2X, NER, budget-limited POS,
  and AfriHG are frozen; News, SIB, Intent, and General are not. Held-out
  access remains zero and Sheet E/F/G remain blank.

## B4 verifies its first boundary; queue remains saturated — 14:34 SAST

- A quota-first readback shows home `88.6%` and scratch `44.7%`. B4/B5
  `1277379/1277423` are healthy near steps `5420/719` on two A100-80GB
  cards, with fresh logs and no targeted fault matches. Two other-user jobs
  hold the remaining cards and were not modified.
- B4's step-2728 artifact matches its SHA-256 sidecar and has exact 22,167-row
  coverage across all six General validation families. This is within-run
  structural evidence only; no cross-candidate ranking was performed.
- B6/B7 `1277424/1277552` remain `AssocGrpGRES`-pending with absent output
  roots. Because b4--b7 still occupy the four active owned submission
  positions, the frozen b1, b2, then b3 checkpoint-10912 recoveries are not
  yet eligible. Held-out access remains zero and Sheet E/F/G remain blank.

## B3 times out; b5 starts and b7 is queued — 13:30 SAST

- B3 `1276434` ended `TIMEOUT` at the fixed 24-hour limit at exact step
  `11991/13640`, with no final adapter. Its complete checkpoint-10912 state
  is preserved and hashed. The exact tail disproves the earlier inference
  that b3 had entered terminal validation.
- The prospective same-trial recovery amendment now freezes b1, b2, then b3
  checkpoint-10912 recoveries after candidate submission. No recovery has
  been submitted. Its SHA-256 is
  `48d8881e01c9ee1f7eb17e367a83dc3a99606efe97166e8740fa191e355304b6`.
- B5 `1277423` started on the released A100-80GB card and verified all 694
  immutable files plus exact `43,637/22,167` train/validation rows. B4 is
  healthy near `4872/13640`; b6 remains association-pending.
- After registry, absent-output, and no-duplicate checks, unchanged b7 was
  submitted once as `1277552` and is association-pending. The four active
  owned submissions are now b4/b5/b6/b7. Stage-B remains `1/8`, global freeze
  `7/8`, quota `88.6%/44.7%`, held-out access zero, and Sheet E/F/G blank.

## General Stage-B remains queue-limited — 07:44 SAST

- B3/B4 `1276434/1277379` remain healthy near steps `9310/1957` with empty
  fault scans. B5/B6 `1277423/1277424` remain `AssocGrpGRES`-pending because
  two other-user jobs still hold the remaining A100-80GB cards.
- No owned slot has released, so b7 is not yet eligible for submission.
  Stage-B remains `1/8` terminal-valid, global freeze `7/8`, quota
  `88.6%/44.6%`, held-out access zero, and Sheet E/F/G blank. No intervention
  is due.

## General Stage-B steady; exact recoveries preregistered — 06:44 SAST

- B3/B4 `1276434/1277379` remain healthy near steps `8778/1419` with empty
  fault scans. B5/B6 `1277423/1277424` remain `AssocGrpGRES`-pending behind
  two other-user jobs; no other job was modified.
- The prospective b1/b2 exact same-trial recovery protocol is now recorded in
  `2026-08-30-pure-gdn-general-stage-b-walltime-recovery-preregistration.md`.
  It freezes one checkpoint-10912 resume each, in b1 then b2 order, using the
  already-proven corrected resume propagation only after b7 has been
  submitted and later slots release. No recovery job was submitted.
- Stage-B remains `1/8` terminal-valid, global freeze `7/8`, quota
  `88.6%/44.6%`, held-out access zero, and Sheet E/F/G blank.

## General b3 third boundary verifies — 05:43 SAST

- B3 `1276434` produced a sidecar-matching step-8184 artifact at SHA-256
  `d11f0e9d3a32954618e03b595ecc3fc82917d6121853bc46d4cb48196d78f308`,
  with exact 22,167-row coverage and validation-only macro NLL
  `0.9508656889374341`. This improves its own step-5456 value, so b3 retains
  step 8184. No cross-candidate selection was made.
- B3 and b4 `1277379` remain healthy near steps `8251/887`. B5/B6
  `1277423/1277424` remain `AssocGrpGRES`-pending while two other-user jobs
  hold the remaining cards. All targeted fault scans are empty.
- Stage-B remains `1/8` terminal-valid, global freeze `7/8`, quota
  `88.6%/44.6%`, held-out access zero, and Sheet E/F/G blank. B7 is not yet
  eligible because all four active owned submission slots remain occupied.

## General b1/b2 hit wall time; b5/b6 queued — 04:44 SAST

- B1/B2 `1276432/1276433` reached the fixed 24-hour limit and ended as
  `TIMEOUT` at 04:00 with no final adapters. Both complete step-10912 states
  remain intact. Adapter/optimizer/scheduler/RNG/trainer-state SHA-256 values
  are, respectively:
  - b1: `29aa21d2237b40efffa70f719a364d92addd03042eb3b1dffe8d88a1d12819f0`,
    `8557158f6b2f9580089f6844f4a5b2cc0210cd9e30d5a39d1b69ed700f484f6e`,
    `58d820ba2be4b96b51823238bab995b391147e7c05fc0223f29f94d6a6473407`,
    `0d4e900b49e99f73ab4c969265a4600f448e5236d37bed123a9baa3915e9013c`,
    and `2b5d9b74c8c8ba71ac86bd79dd216edabb6e98316d299944c79fcab5bbabade0`.
  - b2: `cc2a37683387dfc2ebb4969ab8337ce24f42bdf372d568be628545b3f34aedcb`,
    `5a42ee1ef504c3a4e43990f04ad983142debf06691bde0f7ca1d2a6808bcaac8`,
    `8ad971f1950fc2d8cbce77d99f29c9806ee058be8ba844f45d74d9a63fed6707`,
    `3180a03567b7f40c2f694e184313fd4ca7212fda830074f2e9bc3350b97907e5`,
    and `fe862cfbfc8430edd220ec888a7b4ed3dab4151f26e26f0aa090d9341314c804`.
- B4 `1277379` started at 04:00 and is healthy near step 352. B3 `1276434`
  remains healthy near step 7881. Two other-user jobs hold the remaining two
  association cards and were not modified.
- After rechecking registry SHA-256 `8fdd6ea5...bb726`, absent output/logging
  roots, and zero queue/history duplicates, unchanged b5 and b6 were submitted
  once in preregistered order as jobs `1277423/1277424`. Both are pending.
- Stage-B remains `1/8` terminal-valid and global freeze `7/8`. Exact b1/b2
  same-trial recoveries remain required after the preregistered candidate
  queue releases eligible slots. Held-out access remains zero and Sheet E/F/G
  remain blank. The fixed-wall-time failures move the complete-table plan into
  the 5--6 September buffer unless later phases finish faster than expected.

## General b1/b2 approach wall time cleanly — 03:43 SAST

- B1/B2 `1276432/1276433` remain healthy near steps `11829/12014` with empty
  fault scans and about 17 minutes of their fixed 24-hour allocations left.
  Their complete checkpoint-10912 states are preserved; neither has a final
  adapter yet. No cancellation or restart was attempted.
- B3 `1276434` remains healthy near step `7357`. B4 `1277379` remains
  `AssocGrpGRES`-pending with no output or log yet. All four shared
  A100-80GB cards remain allocated across three owned jobs and one other-user
  job.
- Stage-B remains `1/8` terminal-valid, global freeze `7/8`, quota
  `88.6%/44.6%`, held-out access zero, and Sheet E/F/G blank. The next pass
  must preserve the b1/b2 timeout provenance and fill only genuinely released
  slots in the preregistered order.

## General b0 becomes terminal-valid; b1/b2 fourth boundaries verify — 02:43 SAST

- CPU verifier `1277380` completed `0:0` and proved b0's exact retained
  step-5456 to final-adapter equality across 424 keys and 69,438,784 values.
  Retained/final file SHA-256 values are
  `1d8ab35141f3d3c48b30c24b4180ca8b2f84b721f9ed915e5ed0f7115524b4bb`
  and `60bae76b7104bfab77559b15c3a2f4c0d313d4a1836d11c31ff4541264d811b8`.
  General Stage-B advances to `1/8` terminal-valid.
- B1/B2 produced sidecar-matching step-10912 artifacts with exact 22,167-row
  coverage. Artifact SHA-256 values are
  `38d682f9ddcf307e4ce9cedd46d6c89e72acee9e3443b7da1c016d05b979177b`
  and `56f24d191b14f12e6cc1ffa9cdf72168dadb32d84e80f8411c7424b943ef4fce`;
  validation-only macro NLL is `0.9586805432462503/1.004664310741637`.
  B1 records its first within-run non-improvement and retains step 8184; b2
  improves and retains step 10912. No cross-candidate selection was made.
- B1/B2/B3 remain healthy near steps `11301/11476/6827`. B4 `1277379`
  remains `AssocGrpGRES`-pending behind the shared four-card cap. Global
  freeze remains `7/8`, quota is `88.6%/44.6%`, held-out access is zero, and
  Sheet E/F/G remain blank.

## General b0 terminates; b4 and verifier submitted — 01:43 SAST

- B0 `1276431` completed `0:0` at 01:34 after its second consecutive
  non-improvement. Its terminal step-10912 artifact is sidecar-matching at
  SHA-256 `41f7ab1a728bebbbde2f2c511439c3bd1cae773317c82a33132e59cab8bf97b4`,
  with exact 22,167-row coverage and validation-only macro NLL
  `0.9643478094125281`. The run retains step 5456 at macro NLL
  `0.9451928643283297`.
- CPU roundtrip verifier `1277380` was submitted once for retained step 5456
  and is pending. B0 remains scientifically uncounted until it proves exact
  retained-to-final equality.
- After confirming immutable registry SHA-256
  `8fdd6ea5f192a1d2ef2267353162b9a640ccd09ae0d889cb5a999645198bb726`,
  absent b4 output/logging roots, and zero queue/history duplicates, unchanged
  b4 was submitted once as A100-80GB job `1277379`. It is priority-pending
  while the shared four-card association remains saturated.
- B1/B2/B3 remain healthy. Stage-B stays `0/8` terminal-valid, global freeze
  `7/8`, held-out access zero, and Sheet E/F/G blank.

## General b3 second boundary verifies — 00:38 SAST

- B3 `1276434` produced a sidecar-matching step-5456 validation artifact at
  SHA-256 `d868b01a523d2a8e34f8e3da1a17c289004f31a4b46e5db4e6eb27c67735b3a9`.
  Coverage is exactly 22,167 rows across all six families, including all
  3,082 AfriHG rows (`1305/1777` Xho/Zul). Validation-only macro NLL is
  `0.9549178716296577`; this improves b3's own step-2728 value and is
  within-run retention evidence only.
- B0/B1/B2/B3 remain healthy near steps `10564/10374/10533/5740`; all four
  targeted fault scans are empty. B0--b2 remain on course for step-10912
  artifacts around 01:30--02:05.
- All four A100-80GB cards remain occupied by required work. Stage-B remains
  `0/8` terminal-valid, global freeze `7/8`, quota `88.6%/44.5%`, held-out
  access zero, and Sheet E/F/G blank. No cross-candidate selection was made.
