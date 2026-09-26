# Pure-GDN corrected base evaluation — 2026-08-10

## AfriHG ratified and Sheet released — 09:29 SAST

- User explicitly accepted the single whitespace-only Xhosa prediction as a
  valid held-out model miss. It remains scored empty and preserved; no retry
  or protocol change is permitted. Corrected base is scientifically trusted
  `16/16`.
- Canonical Sheet rows 42--43 were reread, updated only in C/D with the
  verified `7.6220/5.6752 chrF` results and exact provenance, then reread.
  Dates show `10 Aug`, wrapping/date formats are preserved, and E/F/G remain
  blank. See `2026-08-10-pure-gdn-validation-hpo.md` for HPO release.

## AfriHG terminal audit; one-empty gate unresolved — 09:03 SAST

- AfriHG `1216495_15` completed `0:0` at `08:33:21 SAST` in `05:43:37`
  on `srvrocgpu010`, one A100-40GB `gpu:ampere`. The log verifies the
  canonical `final_model`, BF16 evaluation, zero-shot, `peft_adapter=None`,
  `merge_lora=False`, and no traceback, OOM, CUDA/NCCL, token-contract, or
  other runtime fault.
- Held-out test coverage is exact for the frozen split: Xhosa `1,305`, Zulu
  `1,776`, total `3,081`. Every prompt has exactly one leading `[BOS]`, no
  terminal `[EOS]`, and no system/user/assistant chat marker. Predictions and
  raw predictions contain no special/chat marker. Zulu has zero empty and
  `1,376` unique predictions; Xhosa has `872` unique predictions but one
  empty prediction at row 1,099. That row's raw decoded prediction is one
  space and normalized prediction is empty; its prompt contract is valid.
- This isolated empty is a model outcome, not evidence of the old
  EOS-after-assistant evaluator bug. Scientifically it must be scored as a
  miss and must not trigger a held-out-driven rerun. It nevertheless fails
  the automation's literal zero-empty acceptance gate, so AfriHG remains
  unratified pending explicit gate adjudication. No retry, HPO run, or Sheet
  write was made.
- Xhosa metrics: chrF `7.622005962640957`, ROUGE-1
  `0.006487162177268816`, ROUGE-2 `0.0008102756899473299`, ROUGE-L
  `0.006457689734003271`, BLEU `0.0`. Zulu metrics: chrF
  `5.675229879488165`, ROUGE-1 `0.006637295971780775`, ROUGE-2
  `0.0018717696789224741`, ROUGE-L `0.006611702196187`, BLEU `0.0`.
- SHA-256: aggregate summary
  `d84cbd390ee62308e07b7113ccd0ca04c6015e5af6a2210f2ed19b817d856ca9`;
  Xhosa examples/metrics/summary
  `33e9969dbd34cdd9b5cb15c21150eceb27c1c2774b7de42a634645b5b5fb6b99`,
  `d7dbcbabe5011b06acef6418e22cc578e3c252944a325165783ff3c7a8d55bcf`,
  `3c9d32e8e8a9df41cf80aacc72da9a10c5bd928bef06e45fb9524e30e56eb2bc`;
  Zulu examples/metrics/summary
  `1c097124fc09eaab56ba38a915ce6ffbbce22379a2ce7cbc04dbb1e914ad30cc`,
  `920d526f0448a7703030328d9766d92140eaafdfeb243e0544116208164d0444`,
  `313f53af12c46dcf520806e3d1e7f0e34838327f595e80974ae4fa16861dcd36`.
- Operational corrected-base completion is `16/16`; scientifically ratified
  trust remains `15/16` pending the one-empty gate decision. HPO remains
  `0/8`, held-out adapter tests `0`, and there are no owned HEX GPU jobs.
  HEX quota is home `33.5%`, scratch `36.7%`; no A100-80GB or L40S work
  exists. Kombuys is read-only and idle: RTX 5090 `10 MiB/0%`, RTX 3080 Ti
  `1 MiB/0%`, scratch `61%`, only the existing `tailscale-kombuys` session.
  Sheet rows 42--43 remain quarantined and E/F/G remain blank.

## AfriHG Xhosa complete; split-count reconciliation — 06:29 SAST

- AfriHG `1216495_15` remains healthy on `srvrocgpu010` after `03:37:56` on
  the sole owned A100-40GB. Xhosa completed all 1,305 test rows and saved its
  artifacts at `06:00:33`; Zulu then started, prepared 1,776 test rows, and
  selected generation batch size 64 at `06:06:33`. No runtime fault marker or
  final summary exists yet. Conditional completion remains near `08:41 SAST`.
- The earlier `1,305 + 1,777 = 3,082` completion gate conflated AfriHG's
  validation and held-out test counts. A read-only CSV parse on HEX proves:
  test is Xhosa `1,305` plus Zulu `1,776` = `3,081`; dev/validation is Xhosa
  `1,305` plus Zulu `1,777` = `3,082`. This agrees with the frozen evaluator's
  prepared counts and the preserved June test-run record. Test-file SHA-256
  values are Xhosa
  `67adfa18ead0d9a39b8fd3f4701da0c121dba7438fd5ea04fa68afc73221b3f6`
  and Zulu
  `9b6feef24111e84ed385ea563728e220ebb99387a9d9af1b77f4fb80d7c0d90b`.
  Therefore 3,081 is the correct frozen held-out coverage gate; this
  count-only reconciliation does not change the job, prompt, recipe, retry,
  or any selection decision.
- HEX quota remains home `33.5%`, scratch `36.7%`; there is no owned
  A100-80GB or L40S work. Corrected-base scientific trust remains `15/16`
  until the terminal two-language artifact audit. HPO remains `0/8`, held-out
  adapter tests `0`, Sheet rows 42--43 remain quarantined, and E/F/G remain
  blank.

## AfriHG monitoring — 04:30 SAST

- AfriHG `1216495_15` remains healthy on `srvrocgpu010` after `01:39:59` on
  the sole owned A100-40GB `gpu:ampere`. It is still generating the 1,305-row
  Xhosa split, with fresh activity at `04:17:13 SAST`. Three long inputs have
  emitted the frozen context-limit truncation warning; there is no traceback,
  OOM, CUDA/NCCL, token-contract, or other runtime fault and no summary exists
  yet.
- The runtime-only conditional ETA remains approximately `08:41 SAST`, based
  on superseded job `1210866`; output-dependent batching can move it.
- HEX quota remains home `33.5%`, scratch `36.7%`. No owned A100-80GB or L40S
  work exists. Kombuys is read-only and idle: RTX 5090 `10 MiB/0%`, RTX 3080
  Ti `1 MiB/0%`, scratch `61%`, and only the existing `tailscale-kombuys`
  tmux session.
- Operational corrected-base state remains 15 complete plus one running;
  scientifically trusted progress is `15/16`. HPO remains `0/8`, held-out
  adapter tests remain `0`, Sheet rows 42--43 remain quarantined, and E/F/G
  remain blank.

## AfriHG monitoring — 03:58 SAST

- AfriHG `1216495_15` remains healthy on `srvrocgpu010` after `01:08:06` on
  the sole owned A100-40GB `gpu:ampere`. Xhosa generation continues with
  fresh log activity at `03:32:27 SAST`; two long inputs have emitted the
  frozen context-limit truncation warning. There is no traceback, OOM,
  CUDA/NCCL, token-contract, or other runtime failure.
- The runtime-only conditional ETA remains approximately `08:41 SAST` based
  on superseded job `1210866`; output-dependent batching can move it.
- No owned A100-80GB or L40S work exists. HEX quota remains home `33.5%`,
  scratch `36.7%`. Kombuys remains read-only and idle: RTX 5090 `10 MiB/0%`,
  RTX 3080 Ti `1 MiB/0%`, scratch `61%`, and only the existing
  `tailscale-kombuys` tmux session.
- Operational corrected-base state remains 15 complete plus one running;
  scientifically trusted progress is `15/16`. HPO remains `0/8`, held-out
  adapter tests remain `0`, Sheet rows 42--43 remain quarantined, and E/F/G
  remain blank.

## AfriHG monitoring — 03:28 SAST

- Final corrected base lane AfriHG `1216495_15` remains healthy on
  `srvrocgpu010` after `00:38:02` on one A100-40GB `gpu:ampere`. It loaded the
  canonical checkpoint with no adapter/merge, prepared all 1,305 Xhosa test
  examples, selected automatic generation batch size 64, and logged fresh
  progress at `03:21:03 SAST`. A context-limit truncation warning for one
  long input is protocol-defined and not a runtime fault. There is no
  traceback, OOM, CUDA/NCCL, or token-contract failure.
- The runtime-only conditional ETA remains approximately `08:41 SAST` based
  on superseded job `1210866`; output-dependent batching can move it.
- This is the only owned job. No A100-80GB or L40S work exists. HEX quota is
  home `33.5%`, scratch `36.7%`. Kombuys remains read-only and idle with RTX
  5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `61%`, and only the
  existing `tailscale-kombuys` tmux session.
- Operational corrected-base state is 15 complete plus one running;
  scientifically trusted progress remains `15/16`. Multilingual HPO remains
  `0/8`, held-out adapter tests remain `0`, Sheet rows 42--43 remain
  quarantined, and E/F/G remain blank.

## T2X verified; AfriHG running — 02:58 SAST

- Corrected T2X Xhosa `1216462_14` completed `0:0` in `00:45:42` on one
  A100-40GB `gpu:ampere`. It used canonical `final_model`, BF16, zero-shot,
  no adapter/merge, raw prompts with exactly one leading BOS, no terminal EOS,
  and no chat markers. All 378 declared test examples were evaluated with zero
  empty predictions and 235 unique predictions; no runtime fault marker exists.
- Corrected metrics are chrF `2.4064287155136577`, ROUGE-1
  `0.007500444490458154`, ROUGE-2 `0.0030431334380027733`, ROUGE-L
  `0.007465116828040053`, and BLEU `0.0006865987846649683`. This valid
  negative result does not trigger a prompt, protocol, or retry change.
- Artifact SHA-256 values are summary
  `ecc78c42cb030ed38d69d377877d4f5d9a21dfa012f19687f9da389148564ab0`,
  metrics
  `67b9fd1bc70c7373a76fd5450f7a7a06c1d5eccee9804b4f5eb95be202f2b5ee`,
  task summary
  `21007e28be58977231e37d9e54993df2849309ec7b7673b4dab17f4c2c4537ff`,
  and 378-row examples
  `24d77bc0446f5c0996e0367335edda3e79f3238dcf4e7496e3e2226dc9998b77`.
- Canonical Sheet row 41 was reread, updated only in C/D, and verified with
  `10 Aug`, `2.4064 chrF`, complete provenance and output checks, preserved
  date/wrapping formats, and blank E/F/G. Rows 42--43 remain quarantined.
  Scientifically trusted corrected-base progress is now `15/16`.
- AfriHG `1216495_15` started at `02:49:44 SAST` on `srvrocgpu010` and is
  healthy at `00:10:40` on A100-40GB after preparing all 1,305 Xhosa test
  examples. A runtime-only ETA based on superseded
  job `1210866` (`05:51:20`) is approximately `08:41 SAST`; output-dependent
  auto batching can move it. It is the only owned job; no A100-80GB or L40S
  work exists.
- HEX quota is home `33.5%`, scratch `36.7%`. Kombuys remains read-only and
  idle: RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `61%`, with only
  the existing `tailscale-kombuys` session. Multilingual HPO remains `0/8`
  and held-out adapter evaluation remains blocked until AfriHG passes.

## Final-lane monitoring — 02:28 SAST

- T2X Xhosa `1216462_14` remains healthy on `srvrocgpu010` after `00:21:48`
  on one A100-40GB `gpu:ampere`. It has prepared all 378 frozen test examples;
  no traceback, OOM, CUDA/NCCL, token-contract, or other fault marker exists.
  Conditional completion remains near `02:56 SAST` based only on the prior
  lane runtime.
- AfriHG `1216495_15` remains resource-pending with unknown start/ETA. It is
  the only other owned job. There is no owned A100-80GB or L40S work.
- HEX quota is home `33.5%` and scratch `36.7%`. Kombuys remains read-only and
  idle: RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `61%`, and only
  the existing `tailscale-kombuys` tmux session.
- Corrected-base trust remains `14/16`; Multilingual HPO remains `0/8`; no
  held-out adapter test has run. Canonical Sheet rows 40--43 remain untouched
  in this pass and E/F/G remain blank.

## Belebele Zulu verified; final two lanes active — 02:07 SAST

- Corrected Belebele Zulu `1216461_13` completed `0:0` in `00:02:06` on
  A100-40GB `gpu:ampere`. Canonical checkpoint, BF16, zero-shot,
  no-adapter/no-merge, raw prompt, `apply_chat_template=false`,
  `add_bos_token=true`, five-task coverage, and fault-marker checks passed.
  Summary SHA-256 is
  `be74eaf014ee859c49ba2f292e284c0fa1689b329107a8b48b2268911b06d77e`;
  raw-result SHA-256 is
  `a7d28b7cf3224f3fd9b569e44af457714a9bc44e311b9908aa7a4673d9a81d37`.
  P1--P5 accuracy is tied at `0.2288888888888889`.
- Canonical Sheet row 21 was reread, updated only in C/D, and verified with
  the corrected value, exact provenance note, preserved date/wrapping formats,
  and blank E/F/G. Trusted corrected-base progress is now `14/16`.
- T2X Xhosa `1216462_14` is healthy on `srvrocgpu010` A100-40GB. It loaded
  canonical `final_model`, BF16, no adapter/merge, and prepared all 378 frozen
  test examples under the raw zero-shot prompt contract with no fault marker.
  Based only on the prior lane runtime, conditional completion is near
  `02:56 SAST`.
- After confirming no corrected summary or active duplicate and fewer than
  three owned jobs, final base lane AfriHG was submitted individually as
  `1216495_15` from the same immutable snapshot and frozen Slurm/protocol
  envelope. It is resource-pending, so ETA is unknown. No A100-80GB or L40S
  work exists. HEX quota is home `33.5%`, scratch `36.7%`; the latest
  read-only Kombuys state remains idle with RTX 5090 untouched. Multilingual
  HPO and all held-out adapter evaluation remain blocked.

## Belebele Xhosa verified — 02:06 SAST

- Corrected Belebele Xhosa `1216460_12` completed `0:0` in `00:02:08` on
  A100-40GB `gpu:ampere`. It passed canonical checkpoint, BF16, zero-shot,
  no-adapter/no-merge, raw prompt, `apply_chat_template=false`,
  `add_bos_token=true`, five-task coverage, and fault-marker checks.
- Summary SHA-256 is
  `c76c07dd04b9a32a00b912dd34013069a0ae944f499f748c03c31f982d304a4d`;
  raw-result SHA-256 is
  `bee8ec8eaee1eeca36d886df818ed2a142094f82ab3d03679a24c559abf2d62c`.
  P1--P5 accuracy is tied at `0.2288888888888889`.
- Canonical Sheet row 20 was reread, updated only in C/D, and verified with
  the corrected value, exact provenance note, preserved date/wrapping formats,
  and blank E/F/G. Trusted corrected-base progress is `13/16`.
- Belebele Zulu `1216461_13` then started on `srvrocgpu010`; T2X Xhosa
  `1216462_14` remains resource-pending. HPO and held-out adapter evaluation
  remain blocked.

## Belebele Swati, Tswana, and Tsonga verified — 02:02 SAST

- Corrected Belebele Swati `1215945_9`, Tswana `1215946_10`, and Tsonga
  `1215947_11` completed `0:0` in `00:02:10`, `00:02:11`, and `00:02:14`
  on A100-40GB `gpu:ampere`. Each used the canonical `final_model`, BF16,
  zero-shot, no adapter/merge, raw prompts, `apply_chat_template=false`,
  `add_bos_token=true`, and exactly five P1--P5 tasks. No runtime fault marker
  was present.
- Summary SHA-256 values are Swati
  `4410575256e606e80a36e9fd18b9a3114afd3c9839ffa80d6a39d2e4a19c166d`,
  Tswana
  `83774b5ff51a543be9c5c49df5bc3e5c3169323b4ec1c079bfe421973ff50f55`,
  and Tsonga
  `e9137225024ed5d266ac3aae2283ebbdb5c9fdd9dbcd6470e23bb5e2606ecf40`.
  Raw-result SHA-256 values are respectively
  `8d2d635f8785d40a13c3b1c39d8aa5b4ca74fb81e73165b95eca258ed8341ce5`,
  `487f4069f2629c160f54b5bad3d6eb82aa7fc61aaf0a1f46b0966a85491903bd`,
  and
  `afdcc4282aeff38eea4042c756bca5eca594649c98830c24e47afb7c4bf0f646`.
  Every lane has P1--P5 accuracy tied at `0.2288888888888889`.
- Canonical Sheet rows 23, 22, and 27 were reread, updated only in C/D, and
  verified with `10 Aug`, corrected values, exact job/path/hash/protocol notes,
  preserved date/wrapping formats, and blank E/F/G cells. Scientifically
  trusted corrected-base progress is now `12/16`; Multilingual HPO remains
  `0/8`, and held-out adapter tests remain `0`.
- After confirming an empty owned queue and no corrected summaries, the next
  frozen lanes were submitted individually from immutable snapshot
  `pure-gdn-prompt-correction-20260809-693fe42c`: Belebele Xhosa
  `1216460_12`, Belebele Zulu `1216461_13`, and T2X Xhosa `1216462_14`.
  All request `nlpgroup/a100/nlpgroup`, one `gpu:ampere`, 24 hours, and eight
  CPUs from `$HOME/masters/sallm`. At 02:02 SAST Xhosa was running on
  `srvrocgpu010`; Zulu was resource-pending and T2X priority-pending.
- HEX quota is home `33.5%` and scratch `36.7%`. There is no owned
  A100-80GB or L40S work. Kombuys remains read-only and idle: RTX 5090
  `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `61%`, with only the existing
  `tailscale-kombuys` tmux session. HPO and held-out adapter evaluation remain
  blocked until all 16 corrected base lanes pass.

## SA-general and Belebele-Afrikaans verified — 00:30 SAST

- Corrected SA-general `1213987_5` completed `0:0` in `02:22:22` on one
  A100-40GB `gpu:ampere`. It passed canonical `final_model`, pure-GDN
  `127,425,448` parameter identity, BF16, zero-shot, no adapter/merge, raw
  prompt, `apply_chat_template=false`, `add_bos_token=true`, exact three-pack
  and 60-task coverage, and fault-marker checks.
- SA-general summary SHA-256 is
  `733a494cf4d30fd9bc923c46bb47f1f1e009529a183e14598b08cf89d45e8d95`.
  Result hashes are AfriMGSM
  `ae374ddaa50267c42b2ba2c3dd4c1b0e1a3644e8d599b799c8b65a336f08a9d6`,
  AfriMMLU
  `e91398f9a3a1865006d22c3136b61738bdea0ccc131dc18976e6267823a1436d`,
  and AfriXNLI
  `f3fd7c7b4a23f4e8d00a80008d0b14edd665fead356bb557b30b409445501e42`.
- Corrected AfriMGSM flexible-extract exact match is `0.0` for every P1--P5
  prompt in English, Xhosa, Zulu, and Southern Sotho. This is a verified
  negative result, not a retry or prompt-change trigger. AfriMMLU accuracy
  best/mean ranges are Xho `0.1920/0.1884`, Zul `0.2260/0.2224`, Sot
  `0.1960/0.1952`, Eng `0.2120/0.2100`. AfriXNLI is near chance: Xho
  best/mean `0.3333/0.3327`, Zul `0.3333/0.3333`, Sot `0.3367/0.3340`, Eng
  `0.3333/0.3333`.
- Corrected Belebele-Afrikaans `1214451_6` completed `0:0` in `00:01:47` on
  A100-40GB. Summary SHA-256 is
  `13efdc7ebeb11c555884b24a47dd7e2d2115637ec487e04a1999d289386b3520`;
  result SHA-256 is
  `a6cca711ed207c3dcd4fc942959fb821e65322a17bb4fc1d51337c301d3a70bd`.
  All P1--P5 accuracies tie at `0.2288888888888889`; protocol and coverage
  checks passed.
- Canonical Sheet row 26 and rows 28--39 were reread, updated, and verified
  with `10 Aug` dates, established task-native headlines, complete prompt
  values/means/ranges, exact job/path/hash/model/protocol notes, preserved
  formatting, and blank E/F/G cells.
- Trusted corrected-base progress is now `6/16`; frozen Multilingual HPO
  winners remain `0/8`; held-out adapter tests remain `0`.
- After summary/duplicate checks, corrected Belebele-English was submitted as
  `1215825_7` and started on `srvrocgpu010`; Belebele-Southern-Sotho was
  submitted as `1215826_8` and remains resource-pending. Both use the immutable
  snapshot plus established mutable runtime and frozen base overrides.
- Corrected NER `1211758_1` remains healthy, last observed at `14201/14980`
  with about `19m` remaining. No owned A100-80GB or L40S work exists. HEX quota
  is home `33.5%`, scratch `36.6%`. Kombuys remains at its last verified
  read-only idle state. HPO and held-out adapter evaluation remain blocked.

## NER, Belebele-English, and Belebele-Sotho verified — 01:00 SAST

- Corrected MasakhaNER `1211758_1` completed `0:0` in `06:16:12` on one
  A100-40GB `gpu:ampere`. Summary SHA-256 is
  `784a155538e47bb7df1752f6097e8c1d60c0d0fa3bef17aa57665183765614df`;
  raw results SHA-256 is
  `d30d7582a5d0e443e14936b6f60af9770de339bb17aea3d990c3ece6fdfa8de4`.
  Canonical model/BF16/zero-shot/no-adapter/raw-prompt/exactly-one-BOS,
  15-task coverage, and fault-marker checks passed. Xho/Zul/Tsn P1--P5
  flexible-extract F1 values are all `0.0`, a verified negative result.
- Corrected Belebele-English `1215825_7` completed `0:0` in `00:01:44`.
  Summary/result SHAs are
  `ebd0ddd24c2c22a123dbabe7715c8111ea8cafa4925e2a10540c1039f0ceb5ae`
  and `f1087d504b559174b3b6e928057d00f9eded1fec2780d46c602f0b3f7117d600`.
  Corrected Belebele-Southern-Sotho `1215826_8` completed `0:0` in `00:01:48`;
  summary/result SHAs are
  `a4d9ebe34463d6e97b9b09af75196aae7e75e4c2ac8349d5c4dd792bdde697ac`
  and `bcbfc913c10bb43ebaf29d8ce4bc0002a74cfe18218a4e87e30dcafaf2ac6660`.
  Both languages have P1--P5 accuracy tied at `0.2288888888888889` and passed
  the corrected protocol/coverage/fault checks.
- Canonical Sheet rows 4--6 and 24--25 were reread, updated, and verified with
  `10 Aug` dates, exact prompt values/means/ranges, model/protocol/job/path/hash
  notes, preserved formatting, and blank E/F/G cells.
- Trusted corrected-base progress is now `9/16`; Multilingual HPO remains
  `0/8`; held-out adapter tests remain `0`.
- After summary/duplicate checks, corrected Belebele-Swati `1215945_9`,
  Tswana `1215946_10`, and Tsonga `1215947_11` were submitted from the
  immutable snapshot with frozen overrides. Swati started on `srvrocgpu010`;
  Tswana is resource-pending and Tsonga priority-pending. No A100-80GB or L40S
  work exists. HEX quota is home `33.5%`, scratch `36.6%`; Kombuys remains at
  its last verified read-only idle state. HPO and held-out evaluation remain
  blocked.
