# Pure-GDN corrected validation-only HPO — 2026-08-14

## POS a1 improves to 0.7764 — 23:27 SAST

- POS a1 `1233034` improved at step 566 to exact
  `all_token_accuracy=0.7763574498537653`. The frozen constrained artifact
  covers all `1,800` rows, all 12 language/template cells, and all 17 labels;
  checkpoint 566 is retained. Artifact, trainer-state, and adapter SHA-256
  values are
  `184caafe263c11039d43e895221cacbccd5aa876433234925a309faea7c3a774`,
  `fe311cc75348d188b06e8437f4d33624fbf722bf46634f27e05e1a6cb2fba3ee`,
  and
  `c630ecf30fc3b71abac8678153ef9f9bde6073a1bc4e588d5265b95095dce107`.
  It remains healthy near the step-849 boundary; its next exact artifact is
  expected around `01:40--01:55 SAST`.
- POS a0 `1232181` remains healthy at `850/1800` rows in step-2264
  validation, retaining checkpoint 1981 accuracy `0.8205303539382469`; its
  next artifact is expected around `00:30--00:45 SAST`. NER b7 seed-87
  `1233035` remains resource-pending without model/data access. Owned state
  is two running plus one pending A100-40GB job, with no A100-80GB or L40S
  work. HEX quota is home `70.3%`, scratch `38.5%`. Kombuys remains
  read-only: assigned GPU 1 RTX 3080 Ti is idle at `1 MiB/0%`; foreign GPU 0
  RTX 5090 is active at `23,040 MiB/96%` and untouched.
- Trusted terminal progress is unchanged: base `16/16`, NER seed-42
  `11/11`, NER confirmations `1/4`, T2X seed-42 `11/11`, T2X confirmations
  `4/4`, POS `0/11` with nine valid interim artifacts, global winners `0/8`,
  held-out `0`, Mono not started. Sheet E/F/G remain blank, quarantined rows
  remain unpublished, and publication stays blocked. Queue time and
  remaining validation grids still prevent a fixed full-results ETA.

## POS a0 improves to 0.8205 — 22:27 SAST

- Corrected POS a0 `1232181` improved at step 1981 to exact
  `all_token_accuracy=0.8205303539382469`. The frozen constrained artifact
  covers all `1,800` rows, all 12 language/template cells, and all 17 labels;
  checkpoint 1981 is retained. Artifact, trainer-state, and adapter SHA-256
  values are
  `64fb032ab16ab9746d5be362c726db8d41da4d2a2e34e3c8335f6646401febd9`,
  `0d3ec38cef2ee56f4afa00ef9782680c787aa526b001071f377f9fb4a6d149a4`,
  and
  `752359968254d1b881aaf7a369f4e32c432aeb9a96a54707316623873f7a960b`.
  It remains healthy at `50/1800` rows in step-2264 validation; the next
  exact artifact is expected around `00:30--00:45 SAST`.
- POS a1 `1233034` remains healthy at `1150/1800` rows in step-566
  validation; its next exact artifact is expected around
  `23:10--23:25 SAST`. NER b7 seed-87 `1233035` remains resource-pending
  without model/data access. Owned state is two running plus one pending
  A100-40GB job, with no A100-80GB or L40S work. HEX quota is home `70.3%`,
  scratch `38.5%`. Kombuys remains read-only: assigned GPU 1 RTX 3080 Ti is
  idle at `1 MiB/0%`; foreign GPU 0 RTX 5090 is active at
  `15,426 MiB/99%` and untouched.
- Trusted terminal progress remains base `16/16`, NER seed-42 `11/11`, NER
  confirmations `1/4`, T2X seed-42 `11/11`, T2X confirmations `4/4`, POS
  `0/11` with eight valid interim artifacts, global winners `0/8`, held-out
  `0`, Mono not started. Sheet E/F/G remain blank, quarantined rows remain
  unpublished, and publication stays blocked. Queue time and remaining
  validation grids still prevent a fixed full-results ETA.

## POS a1 first exact metric — 20:57 SAST

- POS a1 `1233034` completed its first exact constrained artifact at step
  283 with `all_token_accuracy=0.722852150097027`. The frozen protocol covers
  all `1,800` rows, all 12 language/template cells, and all 17 labels;
  checkpoint 283 is retained. Artifact, trainer-state, and adapter SHA-256
  values are
  `511d540b09971d7bdeff5ec6290d1f29fe0a1640549f0b5195e0f87233cb8438`,
  `a97fcf74cd792eda65722c699bc1f286605106a6362f2ee785b33f0b5780d54c`,
  and
  `79e6f10a25eeb71880412436c729305af385526d8aaf14749fe139850b10dce2`.
  The job remains healthy and has reached the next epoch boundary; its next
  exact artifact is expected around `23:10--23:25 SAST`.
- POS a0 `1232181` remains healthy at `850/1800` rows in step-1981
  validation, retaining checkpoint 1698 accuracy `0.8153800322428616`; its
  next artifact is expected around `22:00--22:15 SAST`. NER b7 seed-87
  `1233035` remains resource-pending without model/data access. Owned state
  is two running plus one pending A100-40GB job, with no A100-80GB or L40S
  work. HEX quota is home `70.3%`, scratch `38.5%`. Kombuys remains
  read-only: assigned GPU 1 RTX 3080 Ti is idle at `1 MiB/0%`; foreign GPU 0
  RTX 5090 is active at `17,692 MiB/99%` and untouched.
- Trusted terminal progress is unchanged: base `16/16`, NER seed-42
  `11/11`, NER confirmations `1/4`, T2X seed-42 `11/11`, T2X confirmations
  `4/4`, POS `0/11` with seven valid interim artifacts, global winners
  `0/8`, held-out `0`, Mono not started. Sheet E/F/G remain blank,
  quarantined rows remain unpublished, and publication stays blocked. Queue
  time and remaining validation grids still prevent a fixed full-results
  ETA.

## POS a0 improves to 0.8154 — 19:57 SAST

- Corrected POS a0 `1232181` improved at step 1698 to exact
  `all_token_accuracy=0.8153800322428616`. The frozen constrained artifact
  covers all `1,800` rows, all 12 language/template cells, and all 17 labels;
  checkpoint 1698 is retained. Artifact, trainer-state, and adapter SHA-256
  values are
  `a0e327fc71b46ba3227343bc015a6b9217eb2289f6a5b061ce1c5fb2ad0a05b2`,
  `51bd5edfe7a2e56c9ff1af2f300eb22c2c028922bd898e4a622bdda00f094c7a`,
  and
  `9d34c70a651959a5055fb60988382d956e386c4c026f1b20ae99aab0f3230d8e`.
  It remains healthy at `50/1800` rows in step-1981 validation; the next
  exact artifact is expected around `22:00--22:15 SAST`.
- POS a1 `1233034` remains healthy at `1200/1800` rows in its first exact
  constrained callback; its first metric is expected around
  `20:35--20:50 SAST`. NER b7 seed-87 `1233035` remains resource-pending
  without model/data access. Owned state is two running plus one pending
  A100-40GB job, with no A100-80GB or L40S work. HEX quota remains home
  `70.3%`, scratch `38.4%`. Kombuys is read-only: assigned GPU 1 RTX 3080 Ti
  is idle at `1 MiB/0%`; foreign GPU 0 RTX 5090 is active at
  `17,700 MiB/94%` and untouched.
- Scientifically trusted terminal progress is unchanged: base `16/16`, NER
  seed-42 `11/11`, NER confirmations `1/4`, T2X seed-42 `11/11`, T2X
  confirmations `4/4`, POS `0/11` with six valid interim artifacts, global
  winners `0/8`, held-out `0`, and Mono not started. Sheet E/F/G remain
  blank, quarantined rows remain unpublished, and publication stays blocked.
  Queue time and remaining validation grids still block a fixed full-results
  ETA.

## POS a1 retry starts cleanly — 18:27 SAST

- Environment-only POS a1 retry `1233034` started at `18:04:50 SAST` on
  `srvrocgpu010` A100-40GB and passed the correction gates: Python 3.12
  runtime, `694/694` immutable source/config hashes, pure-GDN fast kernels,
  exactly `2,259` train and `1,800` validation rows, and the frozen a1
  candidate (LR `8e-5`, rank/alpha `16/32`, dropout `0.05`, warmup `0.03`,
  seed 42). It completed step-283 full validation loss
  `0.3147279442681207/1,800` and is entering constrained validation; its
  first exact metric is expected around `20:35--20:50 SAST`.
- The immutable wrapper intentionally owns the canonical run/output names,
  so the submitted `sourcefix2` run/output overrides were replaced by
  `/scratch/lmbanr001/masters/sallm/checkpoints/adapter_hpo_v3/pure_gdn/pos/stage_a/a1/seed_42`.
  This did not overwrite prior artifacts: old a1 `1232086` never ran and
  failed retry `1232226` stopped before manifest creation in its separate
  `sourcefix1` path. Execution-manifest and trial SHA-256 values are
  `ba9623b602850b2f105cb1632d316cf574a94096d6b30bc2d3ecea7278a544f6`
  and
  `8dc2990dd29eb861ce3a29f8dc08fce8c73a8e42ea1685f83bfbd0f547a96a5f`.
- POS a0 `1232181` remains healthy at `900/1800` rows in step-1698
  validation, retaining checkpoint 1415 accuracy `0.8096575744368201`; its
  next artifact is expected around `19:30--19:45 SAST`. NER b7 seed-87
  `1233035` remains resource-pending without model/data access. Owned state
  is two running plus one pending A100-40GB job, with no A100-80GB or L40S
  work. HEX quota is home `70.3%`, scratch `38.4%`. Kombuys remains
  read-only: assigned GPU 1 RTX 3080 Ti is idle at `1 MiB/0%`; foreign GPU 0
  RTX 5090 is active at `17,446 MiB/87%` and untouched. Scientific terminal
  counts, blank Sheet E/F/G, quarantined rows, held-out gate, and publication
  block are unchanged.

## POS a0 improves to 0.8097 — 17:27 SAST

- Corrected POS a0 `1232181` improved at step 1415 to exact
  `all_token_accuracy=0.8096575744368201`. The frozen
  `closed_label_tuple_mean_logprob_v1` artifact covers all `1,800` rows,
  all 12 language/template cells, and all 17 labels; checkpoint 1415 is the
  retained best. Artifact, trainer-state, and adapter SHA-256 values are
  `de039e95e16e44d0b2b00da8bfc32c36d538116607aaaaf770b2576bddc19873`,
  `49e831d13a3789e91ee4bae5547c78fd77397411746755116a310994b7362a2a`,
  and
  `1c018f67f7750412c1186067f737096b0b5d1f43ddd2d61e711e6eb8f6840e3b`.
  It remains healthy at `100/1800` rows in step-1698 validation; the next
  exact artifact is expected around `19:35--19:50 SAST`.
- POS a1 environment-only retry `1233034` remains resource-pending and NER
  b7 seed-87 confirmation `1233035` remains priority-pending, both without
  model/data access. Owned state is one running plus two pending A100-40GB
  jobs, with no A100-80GB or L40S work. HEX quota is home `70.3%`, scratch
  `38.4%`. Kombuys remains read-only: assigned GPU 1 RTX 3080 Ti is idle at
  `1 MiB/0%`; foreign GPU 0 RTX 5090 is active at `17,310 MiB/96%` and
  untouched.
- Operational base artifacts remain `16/16`; scientifically trusted
  terminal progress remains base `16/16`, NER seed-42 `11/11`, NER
  confirmations `1/4`, T2X seed-42 `11/11`, T2X confirmations `4/4`, POS
  `0/11` with five valid interim artifacts, global winners `0/8`, and
  held-out `0`; Mono has not started. Sheet E/F/G remain blank, quarantined
  rows remain unpublished, and Hugging Face publication remains blocked.
  Queue time and the remaining validation grids still prevent a responsible
  fixed full-results ETA.

## NER confirmation terminal-valid; next lanes queued — 15:32 SAST

- NER b7 seed-13 confirmation `1232204` completed `0:0` after `10:47:18`.
  Its step-4869 mean validation F1 is `0.6681165754371922` (Tsn/Xho/Zul
  `0.6494543518764473/0.6725933719094714/0.6823020025256581`), below
  retained checkpoint 3787 best `0.6698278113149699`; this was the frozen
  second patience miss. Coverage is exact `192`, `64/language`, with no
  literal empty raw outputs or parser failures, `57` whitespace-only raw
  outputs, and `39/55/37` unique raw outputs. Debug, retained-state,
  retained-adapter, final-adapter, and final-config SHA-256 values are
  `c5df3f2975ddb2188b03ce98d587ca39512d088583a35054ff5469e60683393a`,
  `b7f838c338eb5d66e7b94746ad22ec751a166937e95f0267e185148386e5752a`,
  `4d52f4f3eaa09cc7a16e2a8653db851bb6f097d800f70394f9b01095b554e675`,
  `91c16983744ebf9af82c504e5a87176575b27137de6aaf0fbe509262bbe605b7`,
  and
  `8d05d65afb869c21f95a77fd186746a7c2e3fe1c98299db1a9aa02f0bab6604c`.
  A local read-only comparison verified identical key sets and exact equality
  for all `424/424` tensors (`76,410,112` values) between checkpoint 3787
  and `final_adapter`. NER confirmation progress is terminal-valid `1/4`.
- Environment-only POS a1 retry `1233034` is resource-pending with no
  model/data access. NER b7 seed-87 confirmation `1233035` is
  priority-pending with no model/data access; it uses the already-frozen top
  two and confirmation seeds, so no new selection decision was made.
  Corrected POS a0 `1232181` remains healthy at `600/1800` rows in its
  step-1415 callback, retaining checkpoint 1132 accuracy
  `0.7947789577505381`; its next artifact is expected around
  `16:50--17:10 SAST`.
- Owned state is exactly one running plus two pending A100-40GB jobs, with no
  A100-80GB or L40S work. HEX quota is home `70.3%`, scratch `38.4%`.
  Kombuys remains read-only: assigned GPU 1 RTX 3080 Ti is idle at
  `1 MiB/0%`; foreign GPU 0 RTX 5090 is active at `21,800 MiB/90%` and
  untouched. Operational base artifacts remain `16/16`; scientifically
  trusted progress is base `16/16`, NER seed-42 `11/11`, NER confirmations
  `1/4`, T2X seed-42 `11/11`, T2X confirmations `4/4`, POS `0/11`, global
  winners `0/8`, and held-out `0`; Mono has not started. Sheet E/F/G remain
  blank, quarantined rows remain unpublished, and Hugging Face publication
  remains blocked. Queue time for `1233034/1233035` and the remaining
  validation grids prevent a responsible fixed full-results ETA.

## POS a1 environment-only retry preregistration — 15:30 SAST

- POS a1 job `1232226` failed `1:0` after `00:00:00` because its submission
  omitted `SALLM_RUNTIME_REPO`; the immutable snapshot has no `.venv`, so
  manifest creation fell back to system Python 3.9 and failed importing
  `datetime.UTC`. It did not create a manifest, load the model or data, or
  access any metric. The job, empty output path, and log are preserved.
- One infrastructure-only retry is preregistered with the identical a1
  candidate, seed 42, frozen validation-only POS protocol, model, tokenizer,
  immutable source snapshot, registry, and Slurm resources. The only changes
  are an explicit `SALLM_RUNTIME_REPO=/home/lmbanr001/masters/sallm` (the
  Python 3.12 runtime already used successfully by a0) and new `sourcefix2`
  run/output/log names. No held-out evidence informed this retry.

- The preregistered retry was submitted as `1233034`; it is resource-pending
  with the required account/partition/QOS, one `gpu:ampere`, 24 hours, eight
  CPUs, and mandated home working directory.

## POS improves to 0.7948; NER next decision in progress — 14:58 SAST

- Corrected POS a0 `1232181` improved at step 1132 to exact
  `all_token_accuracy=0.7947789577505381`. The frozen
  `closed_label_tuple_mean_logprob_v1` protocol covers all `1,800` rows,
  all 12 language/template cells, and all 17 labels; checkpoint 1132 is the
  retained best. Artifact, trainer-state, and adapter SHA-256 values are
  `d072e54d6306244756bec0587db11f2239e8d57b83ec3cf1c07c3b2937b26d82`,
  `78dfca2504af87f245b50e65c7a379ac63d0a6e492ae3474eba710b5d4bcafbc`,
  and
  `e7c2c0493d53ac6ccfbe815e8f0eac78a5ab25f431aa50672d4b1702caa70d56`.
  It remains healthy at `150/1800` rows in its step-1415 callback; the next
  exact artifact is expected around `16:50--17:10 SAST`.
- NER b7 seed-13 confirmation `1232204` completed step 4869 full validation
  loss at `0.36888760137735244` over all `10,760` declared examples and is
  entering its exact generation callback. Checkpoint 3787 mean F1
  `0.6698278113149699` remains retained with frozen patience `1/2`; its next
  decision artifact is expected around `15:10--15:25 SAST`. Corrected POS a1
  `1232226` remains resource-pending without model/data access.
- Owned state is exactly two running plus one pending A100-40GB job, with no
  A100-80GB or L40S work. HEX quota is home `70.3%`, scratch `38.4%`.
  Kombuys remains read-only: assigned GPU 1 RTX 3080 Ti is idle at
  `1 MiB/0%`; foreign GPU 0 RTX 5090 is active at `19,828 MiB/94%` and
  untouched. Operational base artifacts remain `16/16`; scientifically
  trusted progress is base `16/16`, NER seed-42 `11/11`, NER confirmations
  terminal `0/4` with eight valid interim artifacts, T2X seed-42 `11/11`,
  T2X confirmations `4/4`, POS terminal `0/11` with four valid interim
  artifacts, global winners `0/8`, and held-out `0`; Mono has not started.
  Sheet E/F/G remain blank, quarantined rows remain unpublished, and Hugging
  Face publication remains blocked.

## NER first patience miss; POS artifact imminent — 14:29 SAST

- NER b7 seed-13 confirmation `1232204` completed its exact step-4328
  artifact at mean F1 `0.6624468056515672`; Tsn/Xho/Zul values are
  `0.6457120682964189/0.6680659363161167/0.6735624123421661`. This is below
  retained step-3787 best `0.6698278113149699`, so frozen patience advances
  to `1/2`. Exact coverage is `192`, `64/language`, with no literal empty raw
  outputs, `55` whitespace-only raw outputs, `40/56/37` unique raw outputs,
  and no parser failures. Debug, trainer-state, and current-adapter SHA-256
  values are
  `5f0f168f16762fabf4b4c3fa98e40c817a0b874b079f5ffc5aa95a38a3970ab8`,
  `30998847262cf8ae1dc80e04bfa10477b9a95d97f7444aaf9764412aa7640809`,
  and
  `75b77836c8b72dd4cb5166369a731c506e941d1c35e9c8cb4904197778ba4d96`.
  The job resumed healthy near `4787/8115`; its next artifact is expected
  around `15:10--15:25 SAST` and another frozen patience miss will terminate
  the run while preserving checkpoint 3787.
- Corrected POS a0 `1232181` remains healthy at `1750/1800` rows in its
  step-1132 constrained callback, retaining checkpoint 849 accuracy
  `0.7794016731767367`; the next artifact is imminent. Corrected POS a1
  `1232226` remains resource-pending without model/data access.
- Owned state remains two running plus one pending A100-40GB job, with no
  A100-80GB or L40S work. HEX quota is home `70.3%`, scratch `38.4%`.
  Kombuys remains read-only: assigned GPU 1 RTX 3080 Ti is idle at
  `1 MiB/0%`; foreign GPU 0 RTX 5090 is active at `18,112 MiB/96%` and
  untouched. No held-out metric was accessed; trusted terminal counts, blank
  Sheet E/F/G, quarantined rows, and publication gates are unchanged.

## POS and NER callbacks remain healthy — 13:57 SAST

- NER b7 seed-13 confirmation `1232204` completed its step-4328 full
  validation loss at `0.3567892875813197` over all `10,760` examples and
  remains healthy in generation with fresh automatic batch-size probes
  through `13:49:39`. Checkpoint 3787 mean F1 `0.6698278113149699` remains
  retained; the next artifact is expected around `14:00--14:15 SAST`.
- Corrected POS a0 `1232181` remains healthy at `1350/1800` rows in its
  step-1132 constrained callback, retaining checkpoint 849 accuracy
  `0.7794016731767367`; its next artifact is expected around
  `14:25--14:40 SAST`. Corrected POS a1 `1232226` remains resource-pending
  without model/data access.
- Owned state remains two running plus one pending A100-40GB job, with no
  A100-80GB or L40S work. HEX quota is home `70.3%`, scratch `38.4%`.
  Kombuys remains read-only: assigned GPU 1 RTX 3080 Ti is idle at
  `1 MiB/0%`; foreign GPU 0 RTX 5090 is active at `13,494 MiB/95%` and
  untouched. No new selection metric is accepted; trusted counts, blank
  Sheet E/F/G, quarantined rows, and publication gates are unchanged.

## NER seed-13 reaches 0.6698 mean F1 — 13:15 SAST

- NER b7 seed-13 confirmation job `1232204` improved at step 3787 to mean
  validation F1 `0.6698278113149699`; Tsn/Xho/Zul values are
  `0.6403700372606466/0.6753356591605381/0.693777737523725`. Exact coverage
  is `192`, `64/language`, with no literal empty raw outputs, `51`
  whitespace-only raw outputs, `41/57/38` unique raw outputs, and no parser
  failures. Debug, trainer-state, and adapter SHA-256 values are
  `4766e17110423e82708bd965f890a7e3d08ea0ae7ebc1bfb27d8b534812b21e6`,
  `b7f838c338eb5d66e7b94746ad22ec751a166937e95f0267e185148386e5752a`,
  and
  `4d52f4f3eaa09cc7a16e2a8653db851bb6f097d800f70394f9b01095b554e675`.
  Checkpoint 3787 is retained. The job resumed healthy near `4231/8115`;
  its next artifact is expected around `14:00--14:15 SAST`.
- Corrected POS a0 `1232181` remains healthy at `800/1800` rows in its
  step-1132 constrained callback, retaining checkpoint 849 accuracy
  `0.7794016731767367`; its next artifact is expected around
  `14:25--14:40 SAST`. Corrected POS a1 `1232226` remains resource-pending
  without model/data access.
- Owned state remains two running plus one pending A100-40GB job, with no
  A100-80GB or L40S work. HEX quota is home `70.3%`, scratch `38.4%`.
  Kombuys remains read-only: assigned GPU 1 RTX 3080 Ti is idle at
  `1 MiB/0%`; foreign GPU 0 RTX 5090 is active at `13,476 MiB/97%` and
  untouched. Operational base artifacts remain `16/16`; scientifically
  trusted progress remains base `16/16`, NER seed-42 `11/11`, NER
  confirmations terminal `0/4` with seven valid interim artifacts, T2X
  seed-42 `11/11`, T2X confirmations `4/4`, POS terminal `0/11` with three
  valid interim artifacts, global winners `0/8`, held-out `0`, and Mono not
  started. Sheet E/F/G remain blank, quarantined rows remain unpublished,
  and Hugging Face publication remains blocked.

## POS and NER improve again — 12:06 SAST

- Corrected POS a0 job `1232181` improved at step 849 to exact
  `all_token_accuracy=0.7794016731767367`. The frozen closed-label
  tuple-mean-logprob protocol covers all `1,800` rows, all three languages,
  all four templates, and all 17 labels. Artifact, trainer-state, and adapter
  SHA-256 values are
  `6f2c308bdb2ea0a428b8ceba8ecfbd20fa319cc7cad388f17170460d844f8078`,
  `292d91395663f5963587ab85a12762714f5f60716c0e21212d9c58ea8e3e20c1`,
  and
  `d5aeb30c646e76f89d4d9b6a6109bc836cfe8fa2c822a25f00eb0811c0e3a8cf`.
  Checkpoint 849 is retained. The job resumed healthy near `1063/4245`; its
  next artifact is expected around `14:15--14:35 SAST`.
- NER b7 seed-13 confirmation job `1232204` improved at step 3246 to mean
  validation F1 `0.6488434791145067`; Tsn/Xho/Zul values are
  `0.6291104415039458/0.6451244813277509/0.6722955145118235`. Exact coverage
  is `192`, `64/language`, with no literal empty raw outputs, `51`
  whitespace-only raw outputs, `40/57/39` unique raw outputs, and no parser
  failures. Debug, trainer-state, and adapter SHA-256 values are
  `6eeaeaaae084c734d3a86b9996609ef309af1e0fdf5eae3f140b2bb51f844394`,
  `e12e44d64ec2235ef4a50cc9ebd04a8d4f6704803643df2febbc8b9b0428ae45`,
  and
  `508c88d177c2d1549463aaa422be142b2ff1636d80bfcc08124c6b64742a42a8`.
  Checkpoint 3246 is retained. The job resumed healthy near `3735/8115`;
  its next artifact is expected around `12:45--13:00 SAST`.
- Corrected POS a1 `1232226` remains resource-pending without model/data
  access. Owned state remains two running plus one pending A100-40GB job,
  with no A100-80GB or L40S work. HEX quota is home `70.3%`, scratch
  `38.4%`. Kombuys remains read-only: assigned GPU 1 RTX 3080 Ti is idle at
  `1 MiB/0%`; foreign GPU 0 RTX 5090 holds `27,404 MiB` at `0%` and is
  untouched. Operational base artifacts remain `16/16`; scientifically
  trusted progress remains base `16/16`, NER seed-42 `11/11`, NER
  confirmations terminal `0/4` with six valid interim artifacts, T2X
  seed-42 `11/11`, T2X confirmations `4/4`, POS terminal `0/11` with three
  valid interim artifacts, global winners `0/8`, held-out `0`, and Mono not
  started. Sheet E/F/G remain blank, quarantined rows remain unpublished,
  and Hugging Face publication remains blocked.

## POS and NER callbacks near completion — 11:36 SAST

- Corrected POS a0 `1232181` remains healthy at `1500/1800` rows in its
  step-849 constrained callback, retaining checkpoint 566 accuracy
  `0.7194463042526343`; the next artifact is expected around
  `11:55--12:05 SAST`.
- NER b7 seed-13 confirmation `1232204` remains healthy in its step-3246
  generation callback, with fresh automatic batch-size probes through
  `11:28:11`. Checkpoint 2705 mean F1 `0.6363575862728579` remains retained;
  the next artifact is expected around `11:45--12:00 SAST`. Corrected POS a1
  `1232226` remains resource-pending without model/data access.
- Owned state remains two running plus one pending A100-40GB job, with no
  A100-80GB or L40S work. HEX quota is home `70.3%`, scratch `38.4%`.
  Kombuys remains read-only: assigned GPU 1 RTX 3080 Ti is idle at
  `1 MiB/0%`; foreign GPU 0 RTX 5090 is active at `15,978 MiB/91%` and
  untouched. No new selection metric is accepted; trusted counts, blank
  Sheet E/F/G, quarantined rows, and publication gates are unchanged.

## NER step-3246 loss completes — 11:06 SAST

- NER b7 seed-13 confirmation `1232204` completed its step-3246 full
  validation loss at `0.3455629483474675` over all `10,760` examples and
  entered generation. Checkpoint 2705 mean F1 `0.6363575862728579` remains
  retained; no new 192-row metric is accepted yet.
- Corrected POS a0 `1232181` remains healthy at `1100/1800` rows in its
  step-849 constrained callback, retaining checkpoint 566 accuracy
  `0.7194463042526343`. Corrected POS a1 `1232226` remains resource-pending
  without model/data access. Owned state remains two running plus one pending
  A100-40GB job, with no A100-80GB or L40S work. HEX quota is home `70.3%`,
  scratch `38.4%`. Kombuys remains read-only: assigned GPU 1 RTX 3080 Ti is
  idle at `1 MiB/0%`; foreign GPU 0 RTX 5090 is active at
  `18,698 MiB/97%` and untouched. Trusted counts, blank Sheet E/F/G,
  quarantined rows, and publication gates are unchanged.

## NER seed-13 improves again; POS callback advances — 11:05 SAST

- NER b7 seed-13 confirmation job `1232204` improved at step 2705 to mean
  validation F1 `0.6363575862728579`; Tsn/Xho/Zul values are
  `0.6111820270614768/0.6341540737609139/0.6637366579961829`. Exact coverage
  is `192`, `64/language`, with no literal empty raw outputs, `57`
  whitespace-only raw outputs, `41/55/36` unique raw outputs, and no parser
  failures. Debug, trainer-state, and adapter SHA-256 values are
  `b5dce35780f05c482a42685df6a23815a035d3670468c542648b7d62a8ebe6ca`,
  `680bb667cfdd8b6288df3c88145419c7056e30bc9bc8b9ee2792188eda19bf14`,
  and
  `6f567a9a2dd38036e2c3a35ca5cdbb17626186f064f624ec4e81d023988e7f48`.
  Checkpoint 2705 is retained. The job reached step `3246/8115` and entered
  its next callback; its next artifact is expected around
  `11:45--12:00 SAST`.
- Corrected POS a0 job `1232181` remains healthy in its step-849 constrained
  callback, reaching `1100/1800` rows at `11:02:06`. Checkpoint 566
  `all_token_accuracy=0.7194463042526343` remains retained; the next complete
  artifact is expected around `11:50--12:05 SAST`. Corrected POS a1
  `1232226` remains resource-pending without model/data access.
- Owned state remains two running plus one pending A100-40GB job, with no
  A100-80GB or L40S work. HEX quota is home `70.3%`, scratch `38.4%`.
  Kombuys remains read-only: assigned GPU 1 RTX 3080 Ti is idle at
  `1 MiB/0%`; foreign GPU 0 RTX 5090 is active at `17,178 MiB/99%` and
  untouched. Operational base artifacts remain `16/16`; scientifically
  trusted progress remains base `16/16`, NER seed-42 `11/11`, NER
  confirmations terminal `0/4` with five valid interim artifacts, T2X
  seed-42 `11/11`, T2X confirmations `4/4`, POS terminal `0/11` with two
  valid interim artifacts, global winners `0/8`, held-out `0`, and Mono not
  started. Sheet E/F/G remain blank, quarantined rows remain unpublished,
  and Hugging Face publication remains blocked.

## POS and NER improve at their next callbacks — 09:35 SAST

- Corrected POS a0 job `1232181` produced its step-566 artifact with exact
  `all_token_accuracy=0.7194463042526343`. The frozen closed-label
  tuple-mean-logprob protocol covers all `1,800` rows, all three languages,
  all four templates, and all 17 labels. Artifact, trainer-state, and adapter
  SHA-256 values are
  `b053ef9c5f08eb5897333978201db02a39c3fbfe645c32a6bda5e6c1f78bea59`,
  `c7e4c20ce11b53572515f78a14b3d752fbb87b93b6bc2b30a79efade2673a21c`,
  and
  `28e8ed0cce62a70f2b6feaf3c09615843bfec15d56135aec0d1fb70dbd9f2e5d`.
  Checkpoint 566 is retained. The job resumed through step `849/4245` and
  entered its next callback; the next complete metric is expected around
  `11:50--12:10 SAST`.
- NER b7 seed-13 confirmation job `1232204` improved at step 2164 to mean
  validation F1 `0.6007840028814361`; Tsn/Xho/Zul values are
  `0.5953342890655046/0.5901889092016562/0.6168288103771477`. Exact coverage
  is `192`, `64/language`, with no literal empty raw outputs, `56`
  whitespace-only raw outputs, `39/56/38` unique raw outputs, and no parser
  failures. Debug, trainer-state, and adapter SHA-256 values are
  `cb97b672fdcdaee61432dc69d525c1f3f7266ed0629f15c90d43f4b5e6242a8e`,
  `b1a398dfb0b1c5e8023075126231a296270cf60913e43a418cd118423692c6c0`,
  and
  `242064cf84fe8d7c4cfacd0ae717e444ed1170eaf391c254cef1397d10512f81`.
  Checkpoint 2164 is retained. The job resumed healthy near `2505/8115`;
  its next artifact is expected around `10:25--10:40 SAST`.
- Corrected POS a1 `1232226` remains resource-pending without model/data
  access. Owned state remains two running plus one pending A100-40GB job,
  with no A100-80GB or L40S work. HEX quota is home `70.3%`, scratch
  `38.4%`. Kombuys remains read-only: assigned GPU 1 RTX 3080 Ti is idle at
  `1 MiB/0%`; foreign GPU 0 RTX 5090 is active at `15,534 MiB/98%` and
  untouched. Operational base artifacts remain `16/16`; scientifically
  trusted progress remains base `16/16`, NER seed-42 `11/11`, NER
  confirmations terminal `0/4` with four valid interim artifacts, T2X
  seed-42 `11/11`, T2X confirmations `4/4`, POS terminal `0/11` with two
  valid interim artifacts, global winners `0/8`, held-out `0`, and Mono not
  started. Sheet E/F/G remain blank, quarantined rows remain unpublished,
  and Hugging Face publication remains blocked.

## POS near callback completion; NER generation healthy — 09:05 SAST

- Corrected POS a0 job `1232181` remains healthy on `srvrocgpu010`
  A100-40GB and reached `1500/1800` rows in its step-566 constrained callback
  at `09:01:49`. Checkpoint 283 `all_token_accuracy=0.49199491017457003`
  remains retained; the complete artifact is expected around
  `09:25--09:30 SAST`.
- NER b7 seed-13 confirmation job `1232204` completed its step-2164 epoch-4
  full validation loss at `0.33058257262502905` over all `10,760` examples.
  Its generation callback remains healthy with fresh automatic batch-size
  probes through `09:05:22`; checkpoint 1623 mean F1
  `0.5765106149612969` remains retained and the next 192-row artifact is
  expected around `09:20--09:35 SAST`. Corrected POS a1 `1232226` remains
  resource-pending without model/data access.
- Owned state remains two running plus one pending A100-40GB job, with no
  A100-80GB or L40S work. HEX quota is home `70.3%`, scratch `38.4%`.
  Kombuys remains read-only: assigned GPU 1 RTX 3080 Ti is idle at
  `1 MiB/0%`; foreign GPU 0 RTX 5090 is active at `20,590 MiB/96%` and
  untouched. No new selection metric is accepted. Operational base artifacts
  remain `16/16`; scientifically trusted progress remains base `16/16`, NER
  seed-42 `11/11`, NER confirmations terminal `0/4` with three valid interim
  artifacts, T2X seed-42 `11/11`, T2X confirmations `4/4`, POS terminal
  `0/11` with one valid interim artifact, global winners `0/8`, held-out
  `0`, and Mono not started. Sheet E/F/G remain blank, quarantined rows
  remain unpublished, and Hugging Face publication remains blocked.

## POS callback advances; NER enters epoch-4 validation — 08:35 SAST

- Corrected POS a0 job `1232181` remains healthy on `srvrocgpu010`
  A100-40GB and reached `1150/1800` rows in its step-566 constrained callback
  at `08:34:16`. Checkpoint 283 `all_token_accuracy=0.49199491017457003`
  remains retained; the next complete artifact is expected around
  `09:20--09:30 SAST`.
- NER b7 seed-13 confirmation job `1232204` trained healthily through the
  step-2164 epoch-4 boundary and entered its next validation pass. Checkpoint
  1623 mean F1 `0.5765106149612969` remains the accepted best; the next
  complete 192-row artifact is expected around `09:10--09:25 SAST`.
  Corrected POS a1 `1232226` remains resource-pending without model/data
  access, with Slurm's dynamic projection still `2026-08-15 00:45:59`.
- Owned state remains two running plus one pending A100-40GB job, with no
  A100-80GB or L40S work. HEX quota is home `70.3%`, scratch `38.4%`.
  Kombuys remains read-only: assigned GPU 1 RTX 3080 Ti is idle at
  `1 MiB/0%`; foreign GPU 0 RTX 5090 is active at `16,892 MiB/95%` and
  untouched. No new metric is accepted. Operational base artifacts remain
  `16/16`; scientifically trusted progress remains base `16/16`, NER
  seed-42 `11/11`, NER confirmations terminal `0/4` with three valid interim
  artifacts, T2X seed-42 `11/11`, T2X confirmations `4/4`, POS terminal
  `0/11` with one valid interim artifact, global winners `0/8`, held-out
  `0`, and Mono not started. Sheet E/F/G remain blank, quarantined rows
  remain unpublished, and Hugging Face publication remains blocked.

## NER seed-13 improves at step 1623 — 08:20 SAST

- NER b7 seed-13 confirmation job `1232204` improved at step `1623` to mean
  validation F1 `0.5765106149612969`; Tsn/Xho/Zul values are
  `0.582905544147794/0.5562117696687964/0.5904145310673001`. Exact coverage
  is `192`, `64/language`, with no literal empty raw outputs, `57`
  whitespace-only raw outputs (`21/8/28`), `39/54/36` unique raw outputs,
  and no parser failures. The metric exactly matches the retained state.
  Debug, state, adapter, and config SHA-256 values are
  `ad3c06ee64a4cf7fd55ebb9a1b59c6982a334cfcc47c90dd0c7f34acab2b3cc4`,
  `53266a58a45822cd7f132dc12ecd391e829c7fffc336f1a16638a108a372fb67`,
  `91f9bc667b6435a123a6a3ec01156c4ff56c5c2de928683e9c3bc8cbf073a6dc`,
  and `8d05d65afb869c21f95a77fd186746a7c2e3fe1c98299db1a9aa02f0bab6604c`.
  The job resumed healthy near `1883/8115`; its next artifact is expected
  about `09:05--09:20 SAST`.
- Corrected POS a0 job `1232181` remains healthy in its step-566 epoch-2
  constrained callback, reaching `950/1800` rows at `08:19:05`. Checkpoint
  283 accuracy `0.49199491017457003` remains retained; the next complete POS
  artifact is expected around `09:15--09:30 SAST`. Corrected POS a1
  `1232226` remains resource-pending without model/data access.
- Owned state remains two running plus one pending A100-40GB job, with no
  A100-80GB or L40S work. HEX quota is home `70.3%`, scratch `38.4%`.
  Kombuys remains read-only: assigned GPU 1 RTX 3080 Ti is idle at
  `1 MiB/0%`; foreign GPU 0 RTX 5090 is active at `16,734 MiB/97%` and
  untouched. Operational base artifacts remain `16/16`; scientifically
  trusted progress is base `16/16`, NER seed-42 `11/11`, NER confirmations
  terminal `0/4` with three valid interim artifacts, T2X seed-42 `11/11`,
  T2X confirmations `4/4`, POS terminal `0/11` with one valid interim
  artifact, global winners `0/8`, held-out `0`, and Mono not started. Sheet
  E/F/G remain blank, quarantined rows remain unpublished, and Hugging Face
  publication remains blocked.

## Next POS/NER callbacks active — 07:30 SAST

- Corrected POS a0 job `1232181` remains healthy after `03:09:38` on
  `srvrocgpu010` A100-40GB. It reached step `566/4245`, completed epoch-2
  full validation loss `0.32744540744357636` over all `1,800` examples, and
  advanced to `250/1800` constrained rows at `07:27:26`. Checkpoint 283
  `all_token_accuracy=0.49199491017457003` remains its only accepted interim
  metric. Based on observed callback throughput, the next complete artifact
  is expected around `09:10--09:25 SAST`.
- NER b7 seed-13 confirmation job `1232204` remains healthy after
  `03:03:27`. It reached step `1623/8115` and entered its epoch-3 full
  `10,760`-example validation-loss pass; per-language accounting was fresh
  through `07:27:22`, with no fault marker. Checkpoint 1082 mean F1
  `0.49773337486849173` remains retained. The next complete 192-row artifact
  is expected about `08:10--08:25 SAST`.
- Corrected POS a1 `1232226` remains resource-pending without model/data
  access, with Slurm's dynamic projection still
  `2026-08-15 00:45:59 SAST`. Owned state is two running plus one pending
  A100-40GB job and no A100-80GB/L40S work. HEX quota remains home `70.3%`,
  scratch `38.4%`. Kombuys remains read-only: assigned GPU 1 RTX 3080 Ti is
  idle at `1 MiB/0%`; foreign GPU 0 RTX 5090 is active at
  `24,304 MiB/99%` and untouched.
- No new selection metric is accepted. Operational base artifacts remain
  `16/16`; scientifically trusted progress remains base `16/16`, NER
  seed-42 `11/11`, NER confirmations terminal `0/4` with two valid interim
  artifacts, T2X seed-42 `11/11`, T2X confirmations `4/4`, POS terminal
  `0/11` with one valid interim artifact, global winners `0/8`, held-out
  `0`, and Mono not started. Sheet E/F/G remain blank, quarantined rows
  remain unpublished, and Hugging Face publication remains blocked.

## POS first metric and NER epoch-2 improvement accepted — 07:00 SAST

- Corrected POS a0 job `1232181` produced its first complete validation-only
  artifact at step `283` with `all_token_accuracy`
  `0.49199491017457003`. The closed-label tuple mean-logprob protocol covers
  exactly `1,800` rows, all three languages, all four prompt templates, and
  all 17 registered labels; every cell records explicit correct/total
  counts. The metric exactly matches `trainer_state.json`, which retains
  checkpoint 283. Artifact, state, adapter, and adapter-config SHA-256 values
  are
  `2b083e77969acaf0647b62408e1edc5adedc9e15ad4baa3f71d267f5428a1669`,
  `b4950070d75d659e834b87f8efef2cac37e9cada1ba53d28521f192d6f7b7980`,
  `9eb41ef557661ddd3feb7520b0be93096d423c3ff4b099185ba34cc71ed3685a`,
  and `d7682a345e35d2b0488a69ed64635636a3a397a364e9f5fe8572e47758910546`.
  The job resumed healthy near `434/4245`; its next metric is expected about
  `09:15--09:30 SAST`.
- NER b7 seed-13 confirmation job `1232204` improved at step `1082` to mean
  F1 `0.49773337486849173`; Tsn/Xho/Zul values are
  `0.4795533845079752/0.4802905110256856/0.5333562290718142`. Exact coverage
  is `192`, `64/language`, with no literal empty raw outputs, `54`
  whitespace-only raw outputs (`20/7/27`), `41/56/37` unique raw outputs,
  and no parser failures. The metric exactly matches the retained state.
  Debug, state, adapter, and config SHA-256 values are
  `dd6f8159ded2a0d39b3e350153b944cddc78b2540f5deab44341f30fc2b48fb9`,
  `bc32d5a9dec51344289f6d50983710a89a7e909b7c14205efc8d1436b7610246`,
  `c18da68d112412b751185d0b5ffaf10c7da930859a6f26a01331ed39e080cff4`,
  and `8d05d65afb869c21f95a77fd186746a7c2e3fe1c98299db1a9aa02f0bab6604c`.
  It resumed healthy near `1189/8115`; its next artifact is expected about
  `08:05--08:20 SAST`.
- These are accepted interim validation artifacts, not terminal trials or
  frozen selection decisions. Corrected POS a1 `1232226` remains
  resource-pending without model/data access under the required A100-40GB
  envelope. Owned state is two running plus one pending A100-40GB job and no
  A100-80GB/L40S work. HEX quota is home `70.3%`, scratch `38.4%` (`115 GB`
  shown used). Kombuys remains read-only: assigned GPU 1 RTX 3080 Ti is idle
  at `1 MiB/0%`; foreign GPU 0 RTX 5090 is active at `20,590 MiB/98%` and
  untouched.
- Operational base artifacts remain `16/16`; scientifically trusted progress
  is base `16/16`, NER seed-42 `11/11`, NER confirmations terminal `0/4`
  with two valid interim artifacts, T2X seed-42 `11/11`, T2X confirmations
  `4/4`, POS terminal `0/11` with one valid interim artifact, global winners
  `0/8`, held-out `0`, and Mono not started. Sheet E/F/G remain blank,
  quarantined rows remain unpublished, and Hugging Face publication remains
  blocked.

## POS nears first metric; NER epoch 2 generating — 06:30 SAST

- Corrected POS a0 job `1232181` remains healthy after `02:09:38` on
  `srvrocgpu010` A100-40GB. Its first constrained callback reached
  `1450/1800` rows at `06:26:53 SAST`; at observed throughput, the complete
  validation artifact remains due about `06:50--07:00`. No POS metric has
  been accepted yet.
- NER b7 seed-13 confirmation job `1232204` remains healthy after
  `02:03:27`. It reached step `1082/8115`, completed epoch-2 full validation
  loss `0.36973950127243554` over all `10,760` examples, and is generating
  the 192-row metric artifact with fresh automatic batch-size probes through
  `06:24:26`. Checkpoint 541 mean F1 `0.30889695092008335` remains the only
  accepted interim metric; the step-1082 artifact is expected about
  `06:55--07:10 SAST`.
- Corrected POS a1 `1232226` remains resource-pending without model/data
  access, with Slurm's current dynamic projection still
  `2026-08-15 00:45:59 SAST`. Owned state remains two running plus one
  pending A100-40GB job and no A100-80GB/L40S work. HEX quota is home
  `70.3%`, scratch `38.3%` (`115 GB` shown used). Kombuys remains read-only:
  assigned GPU 1 RTX 3080 Ti is idle at `1 MiB/0%`; foreign GPU 0 RTX 5090
  is active at `15,542 MiB/94%` and untouched.
- Operational base artifacts remain `16/16`; scientifically trusted progress
  remains base `16/16`, NER seed-42 `11/11`, NER confirmations terminal
  `0/4` with one valid interim artifact, T2X seed-42 `11/11`, T2X
  confirmations `4/4`, POS `0/11`, global winners `0/8`, held-out `0`, and
  Mono not started. Sheet E/F/G remain blank, quarantined rows remain
  unpublished, and Hugging Face publication remains blocked.

## NER seed-13 first artifact accepted — 06:00 SAST

- NER b7 seed-13 confirmation job `1232204` produced its first complete
  validation-only artifact at step `541` with mean F1
  `0.30889695092008335`; Tsn/Xho/Zul F1 values are
  `0.2943698990177948/0.2845754318153887/0.3477455219270665`. The artifact
  has exactly `192` rows and `64/language`, no literal empty raw outputs,
  `55` whitespace-only raw outputs (`21/6/28`), `41/57/36` unique raw
  outputs, and two parser failures preserved as model behavior. The metric
  exactly matches `trainer_state.json`, which retains checkpoint 541.
- Debug, trainer-state, adapter, and adapter-config SHA-256 values are
  `ff0cc3f15dc728e7fa6931efed2ab338baffaff1b3c93664323a35856d29efb8`,
  `6c9fa18de42ffcc24eb787c75a61c227e0681b1b1b3771ef3a2c17e191a3faad`,
  `7de1b895d28414281b2a8ff46095432c0408dfbc98581506293a338d77d90d90`,
  and `8d05d65afb869c21f95a77fd186746a7c2e3fe1c98299db1a9aa02f0bab6604c`.
  The job resumed healthy near step `936/8115`; its next epoch artifact is
  expected about `06:45--07:05 SAST`. This is accepted interim confirmation
  evidence, not a terminal seed result or a frozen selection decision.
- Corrected POS a0 job `1232181` remains healthy in its first constrained
  callback, reaching `1100/1800` rows at `05:59:10`. Its first complete
  validation metric remains expected about `06:45--07:00 SAST`. Corrected
  POS a1 `1232226` remains resource-pending without model/data access under
  the required A100-40GB envelope.
- Owned state remains two running plus one pending A100-40GB job, with no
  A100-80GB or L40S work. HEX quota is home `70.3%`, scratch `38.3%` (`115
  GB` shown used). Kombuys remains read-only: assigned GPU 1 RTX 3080 Ti is
  idle at `1 MiB/0%`; foreign GPU 0 RTX 5090 is active at
  `21,376 MiB/95%` and untouched. Operational base artifacts remain `16/16`;
  scientifically trusted progress is base `16/16`, NER seed-42 `11/11`, NER
  confirmations terminal `0/4` with one valid interim artifact, T2X seed-42
  `11/11`, T2X confirmations `4/4`, POS `0/11`, global winners `0/8`,
  held-out `0`, and Mono not started. Sheet E/F/G remain blank, quarantined
  rows remain unpublished, and Hugging Face publication remains blocked.

## Validation callbacks remain healthy — 05:30 SAST

- Corrected POS a0 job `1232181` remains healthy after `01:09:41` on
  `srvrocgpu010` A100-40GB. Its epoch-1 constrained callback advanced to
  `650/1800` rows at `05:26:13 SAST`, with a fresh log and no fault marker.
  The accepted metric remains pending; observed throughput still projects
  the first complete validation artifact around `06:45--07:00 SAST`.
- NER b7 seed-13 confirmation job `1232204` remains healthy after
  `01:03:30` on the same GPU family. Its epoch-1 full validation loss is now
  complete at `1.0411281429702022` over all `10,760` examples, and the
  generation callback is active with fresh automatic batch-size probes
  through `05:26:50`. No complete 192-row artifact exists yet, so no
  confirmation metric is accepted; the updated evidence window is about
  `05:40--06:00 SAST`.
- Corrected POS a1 job `1232226` remains resource-pending without model/data
  access; Slurm's dynamic projection is `2026-08-15 00:45:59 SAST` and may
  move. Owned state is two running plus one pending A100-40GB job, exactly
  the cap of three, with no A100-80GB or L40S work. HEX quota is home
  `70.3%`, scratch `38.3%`. Kombuys remains read-only: assigned GPU 1 RTX
  3080 Ti is idle at `1 MiB/0%`; foreign GPU 0 RTX 5090 is active at
  `17,478 MiB/98%` and untouched.
- Operational base artifacts remain `16/16`; scientifically trusted progress
  remains base `16/16`, NER seed-42 `11/11`, NER confirmations `0/4`, T2X
  seed-42 `11/11`, T2X confirmations `4/4`, POS `0/11`, global winners
  `0/8`, held-out `0`, and Mono not started. The immediate blockers are the
  two active callbacks and corrected POS allocation/completion; full-result
  timing is not yet responsibly fixed. Sheet E/F/G remain blank,
  quarantined rows remain unpublished, and Hugging Face publication remains
  blocked.

## First callbacks active; corrected POS a1 queued — 05:00 SAST

- Corrected POS a0 job `1232181` remains healthy on `srvrocgpu010`
  A100-40GB. It reached the frozen epoch-1 boundary at step `283/4245`,
  completed full validation loss `0.7721000501844618` over all `1,800`
  rows, and entered constrained POS validation. The callback advanced from
  `50/1800` at `04:42:09` to `250/1800` at `04:56:51` with no fault marker.
  Observed callback throughput moves the first complete validation artifact
  ETA to about `06:45--07:00 SAST`; no POS selection metric is accepted yet.
- NER b7 seed-13 confirmation job `1232204` remains healthy on the same
  A100-40GB node. It reached its epoch-1 boundary at step `541/8115` at
  `04:54:39` and entered the full `10,760`-example validation callback. The
  log remains fresh with per-language loss accounting and no fault marker;
  no complete 192-row generation artifact exists yet. The first scientific
  confirmation metric remains expected about `05:25--05:50 SAST`.
- POS a0 has now satisfied the preregistered source/count/startup gate, so
  the never-started old-source a1 job `1232086` was cancelled at
  `05:01:37 SAST` after zero run time and without model/data access. Its
  Slurm provenance is preserved and there were no artifacts to remove.
  Corrected immutable-snapshot a1 replacement job `1232226` was submitted
  with account/partition/QOS `nlpgroup/a100/nlpgroup`, one `gpu:ampere`, 24
  hours, eight CPUs, and the mandated home working directory. It is pending
  for resources and will use only validation data.
- Owned state is two running plus one pending A100-40GB job, exactly the cap
  of three, with no A100-80GB or L40S work. HEX quota remains home `70.3%`,
  scratch `38.3%`. Kombuys remains read-only: assigned GPU 1 RTX 3080 Ti is
  idle at `1 MiB/0%`; foreign GPU 0 RTX 5090 is active at
  `23,148 MiB/97%` and untouched. Operational base artifacts remain `16/16`;
  scientifically trusted progress remains base `16/16`, NER seed-42
  `11/11`, NER confirmations `0/4`, T2X seed-42 `11/11`, T2X confirmations
  `4/4`, POS `0/11`, global winners `0/8`, held-out `0`, and Mono not
  started. Full downstream timing remains gated by these callbacks and the
  remaining validation grids. Sheet E/F/G remain blank, quarantined rows
  remain unpublished, and Hugging Face publication remains blocked.

## Corrected POS and NER confirmation start healthy — 04:30 SAST

- Corrected POS a0 job `1232181` started at `04:19:57 SAST` on
  `srvrocgpu010` A100-40GB and is healthy at step `188/4245`, about
  `3.06 s/step`. It verified all `694` immutable source/config files, loaded
  exactly `2,259` train and `1,800` validation rows, tokenized both complete
  datasets, and entered training at `04:20:59`. This satisfies the frozen
  implementation-correction startup gate. Its execution-manifest and trial
  hashes are
  `da37437b99112dbb7eb205abcffa80ec04bb59dc2eb89eb2792a514203974eb6`
  and `17059aaae8b11b1f450b4bb6913cc5e6330478cf9ed46f89000372b007e59c22`.
  The first complete validation artifact is expected about
  `05:00--05:30 SAST`; no POS metric is accepted yet. Old-source a1 job
  `1232086` remains safely user-held and will not be released.
- Frozen NER b7 seed-13 confirmation job `1232204` started at
  `04:26:08 SAST` on the same A100-40GB node and is healthy at step
  `67/8115`, about `3.06 s/step`. It verified all `694` immutable files,
  loaded `4,323` training examples, and entered training at `04:27:11` with
  the validation-only b7 recipe and seed 13. Its execution-manifest and trial
  hashes are
  `12bd5577de9fa956157e4c83cfe709b12a3e68dadd27d8c901a7fe9d4e95d60d`
  and `c503772202bcb691ae5a2711fe15dcfe1929c17c864d2815f42df994c561010e`.
  The first complete validation artifact is expected about
  `05:25--05:50 SAST`; no confirmation metric is accepted yet.
- Owned state is exactly two running plus one held A100-40GB job, the cap of
  three, with no A100-80GB or L40S work. HEX quota is home `70.3%`, scratch
  `38.3%`. Kombuys remains read-only: assigned GPU 1 RTX 3080 Ti is idle at
  `1 MiB/0%`, while foreign GPU 0 RTX 5090 is active at `19,790 MiB/91%` and
  untouched. Operational base artifacts remain `16/16`; scientifically
  trusted progress remains base `16/16`, NER seed-42 `11/11`, NER
  confirmations `0/4`, T2X seed-42 `11/11`, T2X confirmations `4/4`, POS
  `0/11`, global winners `0/8`, held-out `0`, and Mono not started. The full
  downstream-result ETA remains gated by corrected POS plus the remaining
  validation grids and cannot yet be stated responsibly. Sheet E/F/G remain
  blank, quarantined rows remain unpublished, and Hugging Face publication
  remains blocked.

## NER seed-42 grid completes; top two frozen for confirmation — 03:30 SAST

- NER Stage-B b7 job `1230086` completed cleanly `0:0` at `03:06:51 SAST`
  after `16:35:49`. Its step-7574 epoch-14 mean validation F1 is
  `0.6848459167143336` (Tsn/Xho/Zul
  `0.6742444503877542/0.6776541095889914/0.7026391901662552`), a decline of
  `0.002412595141534246` from retained checkpoint 6492 best
  `0.6872585118558678`. This was the frozen second patience miss, so training
  stopped and restored checkpoint 6492. The exact artifact has `192` rows,
  `64/language`, no literal empty predictions or parse failures, `55`
  whitespace-only raw outputs (`21/7/27`), and `40/55/37` unique raw outputs.
- Final-debug, retained-state, retained-adapter, final-adapter, and config
  SHA-256 values are
  `e9aa0af1342a49625622500d51b202273fe3703ad2c4363396bebc64720f0eec`,
  `b2ac7c4acc0d2304e983411b87bb0c5832ccadd2d41894005a2ca9e435ad0d29`,
  `a40cd263f28256fcf86b9eaecf15bdf01e4735e352f43d3bfe025108ff4ae034`,
  `06b4685fcdb43089ea2de5ad4be1f476b9397cbd9cc57467f291534b1a5309f5`,
  and `e823b4136c4b02e2a0287056c687245ff53c1dd6d7c315181c9acd5b7937ca4e`.
  A local streamed read-only comparison verified identical key sets and all
  `424/424` tensors (`76,410,112` values) exactly equal between retained and
  final serialization. NER seed-42 progress is therefore terminal-valid
  `11/11`.
- The validation-only reconciliation over all three Stage-A and eight
  Stage-B candidates freezes b7 and a2 as the top two seed-confirmation
  candidates at `0.6872585118558678/0.6795225766333571`. Its reproducible
  artifact is
  `sallm_memory/artifacts/2026-08-14-pure-gdn-ner-seed42-ranking.json`,
  SHA-256
  `f51a8a3ba4947d8f6752a202bbf34d68e34b9a370978add5f62959d299f9b1e2`;
  it records all 11 validation metrics, configs, jobs, retained-state paths
  and hashes, the frozen tie-break, and that no test metric was consulted.
- The first frozen confirmation, b7 seed 13, was submitted as job `1232204`.
  It is priority-pending with no model/data access and no scheduler ETA.
  Verified settings are `nlpgroup/a100/nlpgroup`, one `gpu:ampere`, 24 hours,
  eight CPUs, the immutable HPO snapshot, and the mandated home working
  directory. Corrected POS a0 `1232181` is resource-pending with dynamic
  projection `06:12:27 SAST`; old a1 `1232086` remains safely held. This is
  exactly three owned A100-40GB jobs, two schedulable and one held, with no
  A100-80GB or L40S work.
- HEX quota remains home `70.3%`, scratch `38.3%`. Kombuys GPU 1 RTX 3080 Ti
  is idle at `1 MiB/0%`; foreign GPU 0 RTX 5090 is active at
  `16,908 MiB/91%` and untouched. Operational base artifacts remain `16/16`;
  scientifically trusted progress is base `16/16`, NER seed-42 `11/11`, NER
  confirmations `0/4`, T2X seed-42 `11/11`, T2X confirmations `4/4`, POS
  `0/11`, global winners `0/8`, held-out `0`, and Mono not started. Sheet
  E/F/G remain blank and Hugging Face publication remains blocked.

## NER step-7574 loss complete; generation still healthy — 03:00 SAST

- NER b7 job `1230086` remains healthy after `16:29:51` on
  `srvrocgpu010` A100-40GB. Its step-7574 epoch-14 full validation-loss pass
  completed at `0.39439457889826324` over all `10,760` examples, and the
  generation callback remains active with fresh automatic batch-size probes
  through `02:54:47 SAST`. No complete 192-row step-7574 artifact exists yet,
  so no selection metric is accepted and retained checkpoint 6492 at mean F1
  `0.6872585118558678` remains unchanged. Based on the observed segmented
  callback, the next scientific decision is expected around `03:20--03:40`;
  a second frozen patience miss will terminate the run, while a qualifying
  improvement will continue to the final epoch near `04:35--04:50`.
- Corrected POS a0 retry `1232181` remains priority-pending without model or
  data access; Slurm's current dynamic projection is `14:39 SAST`. POS a1
  `1232086` remains safely user-held and failed a0 `1231553` remains
  preserved. No job uses A100-80GB or L40S.
- HEX quota is home `70.3%`, scratch `38.3%`. Kombuys GPU 1 (RTX 3080 Ti) is
  idle at `1 MiB/0%`; foreign GPU 0 (RTX 5090) is active at
  `15,504 MiB/92%` and untouched. Operational base artifacts remain `16/16`,
  while scientifically trusted progress remains base `16/16`, NER `10/11`,
  T2X `11/11`, confirmations `4/4`, POS `0/11`, global winners `0/8`,
  held-out `0`, and Mono not started. Sheet E/F/G remain blank, Hugging Face
  publication remains blocked, and the immediate blockers are terminal NER
  evidence plus corrected POS allocation and completion.

## NER b7 enters step-7574 callback; POS retry queued — 02:30 SAST

- NER b7 job `1230086` reached step `7574/8115` and entered its epoch-14
  validation callback at about `02:25 SAST` on `srvrocgpu010` A100-40GB. Its
  log remains fresh with no fault marker. No complete step-7574 192-row
  selection artifact exists yet, so retained checkpoint 6492 and mean F1
  `0.6872585118558678` remain unchanged. The artifact is expected around
  `02:55--03:10`; a second frozen patience miss would terminate the run,
  while a qualifying improvement would continue to the final epoch near
  `04:15--04:20`.
- Corrected POS a0 retry `1232181` remains priority-pending with no log or
  model/data access. Slurm's dynamic projection is `14:39 SAST`, but it may
  move after NER releases its A100. POS a1 `1232086` remains safely held and
  failed a0 `1231553` remains preserved.
- HEX quota is home `70.3%`, scratch `38.3%`; owned state is one running, one
  priority-pending, and one held A100-40GB job, with no A100-80GB or L40S
  work. Kombuys GPU 1 remains idle; foreign GPU 0 is active at
  `23460 MiB/98%` and untouched. Trusted progress remains base `16/16`, NER
  `10/11`, T2X `11/11`, confirmations `4/4`, POS `0/11`, global winners
  `0/8`, held-out `0`, and Mono not started. Sheet E/F/G remain blank and
  Hugging Face publication remains blocked.

## POS source fix frozen/deployed; a0 retry queued — 02:00 SAST

- The implementation-driven POS source correction was preregistered before
  rerun at SHA-256
  `0f2544b9825081e5430767d276d0ac4e5c16523e007c000eb46ffe3f9933f48b`.
  It pins upstream MasakhaPOS commit
  `376f4161f0425584d4bd7664122b56fa026926d3`, uses its commit-addressed
  GitHub Contents API raw response, and retries only transient transport/HTTP
  failures. It changes no data content, HPO point, prompt, metric, seed,
  checkpoint, or early-stopping rule. Local gates passed `117` tests, Ruff,
  `git diff --check`, and exact train/validation row and token counts for
  Tsn/Xho/Zul.
- New read-only HEX snapshot
  `/home/lmbanr001/masters/sallm_snapshots/pure-gdn-hpo-possourcefix-20260814-d9501087`
  verified all `694/694` source/config hashes. Source-set SHA-256 is
  `59459749600ea24e19411eb44ddff884711587ada81594f8a2e34955d0d34f60`;
  deployment-manifest SHA-256 is
  `d996a19bb3441471f4a13e70c14acc2615a16d502cea3f0ee2a5f04cc6308c0a`;
  deployed loader SHA-256 is
  `d9501087564982e5e4b07c4b3aada876d5625fa24560024a96002a4ee9ee4006`.
- The one preregistered a0 infrastructure retry is job `1232181`, pending for
  `(Priority)`. Verified Slurm settings are account/partition/QOS
  `nlpgroup/a100/nlpgroup`, one `gpu:ampere`, 24 hours, eight CPUs, and the
  mandated home working directory. It writes to a new `sourcefix1` output
  path, preserving failed a0 job `1231553` and its artifacts. Structurally
  identical a1 job `1232086` remains on reversible user hold until a0 passes
  source/count/startup gates.
- NER b7 job `1230086` completed its exact step-7033 artifact at mean
  validation F1 `0.6827893429348121`, below step-6492 best
  `0.6872585118558678`, so checkpoint 6492 correctly remains retained and
  frozen patience advances to `1/2`. Tsn/Xho/Zul F1 values are
  `0.6698692287162564/0.6785981408269628/0.699900659261217`. Coverage is
  exactly `192`, `64/language`, with no literal empty predictions, `55`
  whitespace-only raw outputs, and `40/55/37` unique outputs. Debug/state
  hashes are
  `feb153c8c67b90e6e19d48d7ecbd4baa6207f29bc9366f569daf180f5e0aa5ce`/
  `4b03ebb79c17dc5ef7357e870213936af3419188cd59ccb6e1b6723401b8f66a`.
  It resumed healthy; conditional terminal ETA is about `03:05 SAST` if the
  next callback exhausts patience, or `04:15--04:20` if it reaches epoch 15.
- HEX quota after the immutable snapshot is home `70.3%`, scratch `38.3%`;
  owned state is one running, one
  priority-pending, and one held A100-40GB job, with no A100-80GB or L40S
  work. Kombuys assigned GPU 1 is idle; foreign GPU 0 is active at
  `19934 MiB/98%` and remains untouched. Trusted progress remains base
  `16/16`, NER `10/11`, T2X `11/11`, confirmations `4/4`, POS `0/11`, global
  winners `0/8`, held-out `0`, and Mono not started. Sheet E/F/G remain blank
  and Hugging Face publication remains blocked.

## POS a0 fails before metrics; a1 held — 01:30 SAST

- POS a0 job `1231553` failed `1:0` after `00:00:53` at `01:06:48 SAST`.
  It verified all 694 immutable-snapshot files, loaded the canonical model,
  and resized embeddings, then failed in the MasakhaPOS source loader with
  `urllib.error.HTTPError: HTTP Error 404: Not Found` before loading the
  train/validation datasets, training, or producing any selection metric.
  The failed job and output remain preserved as infrastructure provenance.
  Current direct checks show the configured files still exist, but the
  GitHub raw endpoint is intermittently returning `503` for Xho/Zul train
  files. No dataset, recipe, or metric correction has been inferred from
  held-out evidence.
- POS a1 job `1232086` uses the same immutable loader and was therefore put
  on reversible user hold before allocation (`JobHeldUser`) rather than
  repeat the known pre-metric failure. It must remain held until the source
  availability/root cause is resolved and the correction is documented and
  deployed immutably.
- NER b7 job `1230086` remains healthy on `srvrocgpu010` A100-40GB and is in
  its step-7033 generation callback. The callback has complete validation
  loss `0.39061575396796583`, but no complete new 192-row metric artifact
  yet; the latest accepted retained checkpoint remains step 6492 at mean
  validation F1 `0.6872585118558678`. Conditional terminal ETA remains about
  `03:30--04:00 SAST`.
- HEX quota is home `52.0%`, scratch `38.2%`; owned state is one running and
  one held A100-40GB job, with no A100-80GB or L40S work. Kombuys assigned
  GPU 1 is idle; foreign GPU 0 is active at `16710 MiB/93%` and was left
  untouched. Trusted progress remains base `16/16`, NER `10/11`, T2X
  `11/11`, confirmations `4/4`, POS `0/11`, global winners `0/8`, held-out
  `0`, and Mono not started. Sheet E/F/G remain blank and Hugging Face
  publication remains blocked.

## NER b7 improves at step 6492 — 01:00 SAST

- NER b7 job `1230086` improved at epoch 12 step 6492 to exact mean
  validation F1 `0.6872585118558678`, retaining checkpoint 6492 and resetting
  patience. Tsn/Xho/Zul F1 values are
  `0.6754666666666168/0.6833671616279814/0.7029417072730054`. Its exact
  192-row, 64/language artifact has no literal empties or parse failures,
  `55` whitespace-only outputs, and `40/55/37` unique outputs. Debug/state
  SHA-256 values are
  `b7631734de99dd2eeeea2ade9b94e91d6f72cdf21264c60036b28142d31e545c`/
  `b2ac7c4acc0d2304e983411b87bb0c5832ccadd2d41894005a2ca9e435ad0d29`.
  B7 resumed near step `6711/8115` on `srvrocgpu010` A100-40GB with no fault
  marker; conditional terminal ETA remains about `03:30--04:00 SAST`.
- POS a0/a1 jobs `1231553/1232086` remain pending for
  `(Resources)/(Priority)`. HEX has one running plus two pending owned
  A100-40GB jobs, quota home `52.0%` and scratch `38.2%`, and no A100-80GB or
  L40S work. The continued late NER improvement supports the preregistered
  decision not to introduce low-fidelity pruning.
- Trusted progress remains base `16/16`, NER `10/11`, T2X `11/11`, T2X
  confirmations `4/4`, POS `0/11`, global frozen winners `0/8`, held-out
  adapter evaluations `0`, and Mono not started. No held-out metric was
  accessed; Sheet E/F/G remain blank and Hugging Face publication remains
  blocked.

## NER b7 reaches step-6492 callback — 00:30 SAST

- NER b7 job `1230086` remains healthy on `srvrocgpu010` A100-40GB and has
  entered its step-6492 validation callback after `13:58:26` elapsed. Its log
  was fresh at `00:22:13 SAST`, with no fault marker; no complete new 192-row
  artifact exists yet, so no new metric is accepted.
- POS a0/a1 jobs `1231553/1232086` remain pending for
  `(Resources)/(Priority)`. HEX has one running plus two pending owned
  A100-40GB jobs, quota home `52.0%` and scratch `38.2%`, and no A100-80GB or
  L40S work. Kombuys assigned GPU 1 remains idle, root/scratch free are
  `23 GB/2.1 TB`, and foreign GPU 0 remains occupied and untouched.
- Trusted progress remains base `16/16`, NER `10/11`, T2X `11/11`, T2X
  confirmations `4/4`, POS `0/11`, global frozen winners `0/8`, held-out
  adapter evaluations `0`, and Mono not started. No held-out metric was
  accessed; Sheet E/F/G remain blank and Hugging Face publication remains
  blocked.

## NER b7 improves again at step 5951 — 00:00 SAST

- NER b7 job `1230086` improved at epoch 11 step 5951 to exact mean
  validation F1 `0.6854031406360509`, retaining checkpoint 5951 and resetting
  patience. Tsn/Xho/Zul F1 values are
  `0.671124782666795/0.6839622641508937/0.7011223750904642`. Its exact
  192-row, 64/language artifact has no literal empties or parse failures,
  `54` whitespace-only outputs, and `40/56/37` unique outputs. Debug/state
  SHA-256 values are
  `648083079f465e0ba5ded42ee8d6686a5211214d9ab8fddae3f59c1281479adc`/
  `70ec959351b01a0d8a8595c8292ef80a219511527f3ba0e80f68f69e7fabf0dc`.
  B7 resumed near step `6340/8115` on `srvrocgpu010` A100-40GB with no fault
  marker; conditional terminal ETA is about `03:30--04:00 SAST`.
- POS a0/a1 jobs `1231553/1232086` remain pending for
  `(Resources)/(Priority)`. HEX has one running plus two pending owned
  A100-40GB jobs, quota home `52.0%` and scratch `38.2%`, and no owned
  A100-80GB or L40S work. Kombuys assigned GPU 1 remains idle and foreign GPU
  0 remains untouched.
- Trusted progress remains base `16/16`, NER `10/11`, T2X `11/11`, T2X
  confirmations `4/4`, POS `0/11`, global frozen winners `0/8`, held-out
  adapter evaluations `0`, and Mono not started. No held-out metric was
  accessed; Sheet E/F/G remain blank and Hugging Face publication remains
  blocked.
