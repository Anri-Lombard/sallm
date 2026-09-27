# Pure-GDN corrected validation-only HPO — 2026-08-11

## Stage-B b0 first validation F1 artifact reconciles cleanly — 23:40 SAST

- Enhanced NER Stage-B `b0` job `1218997` completed a scientifically valid
  epoch-1 step-`541` artifact. Tsn/Xho/Zul span micro-F1 is
  `0.11904761904757113/0.12050286181515853/0.07635291304689416`; arithmetic
  mean is `0.10530113130320794`, matching the retained checkpoint metric
  `0.10530113130320795` to floating-point representation. This is the first
  within-candidate validation point only. The protocol does not prune on
  low-fidelity rank, so no Stage-B selection or rejection is permitted.
- Independent checks found exactly `192` rows, `64` per language, and all raw
  predictions nonempty. Unique raw-output counts are `46/51/35`. Debug and
  trainer-state SHA-256 values are
  `d9385ecdb4eb242972d23b3e6ccd6868f09f5a54a8e842714f9acaccdbaddfe8` and
  `31722fd494c0aa18ba6d780efe70b47384f734930dd737083b26d063d819380a`.
- At `23:40`, `1218997` had resumed healthy epoch-2 training. Its next
  selection artifact is tentatively due around `00:45--01:05`. Stage-A LR
  `1.5e-4` job `1218345` remained healthy in epoch-11 full validation, with
  retained provisional best `0.6737384326410746`; its next artifact is
  tentatively due around `00:15--00:35`. Both logs had zero targeted faults.
- Owned state remains two running A100-40GB `gpu:ampere` jobs, with no owned
  A100-80GB or L40S work. Quota is home `52.0%` and scratch `37.4%`. Kombuys
  was not accessed and remains read-only with RTX 5090 untouched. Trusted base
  remains `16/16`, frozen Multilingual winners `0/8`, held-out adapter tests
  `0`, and Monolingual not started. The verified Sheet labels remain `Qwen
  Results`/`GDN Results`, GDN E/F/G remain blank, and no Sheet or Hugging Face
  write occurred.

## Stage-A LR-2 improves at epoch 10; Stage-B enters first generation — 23:10 SAST

- Stage-A LR `1.5e-4` job `1218345` completed a scientifically valid epoch-10
  step-`5410` artifact. Tsn/Xho/Zul span micro-F1 is
  `0.6492476060191019/0.6827980168139187/0.6891696750902028`; exact mean
  `0.6737384326410746` matches newly retained `checkpoint-5410` exactly. The
  gain over epoch 9 is `0.0053915640314053`, above the frozen `0.001`
  threshold, so patience resets and continuing is correct. This remains a
  provisional Stage-A result; NER is not frozen.
- Independent checks found exactly `192` rows, `64` per language, and all raw
  predictions nonempty. Unique raw-output counts are `40/54/37`. Debug and
  trainer-state SHA-256 values are
  `2d5fe174ae2046dba7ea51eeaf259cb2319666f8a48950ba56a2f41d9b1846eb` and
  `b3785a9fd5232eb383ac84589524db516d78ba06be705a420562d69dfb0b4988`.
  At `23:10`, `1218345` had resumed healthy epoch-11 training near step
  `5451/8115`; its next artifact is tentatively due around `00:15--00:35`.
- Enhanced Stage-B `b0` job `1218997` remained healthy. It completed its
  first exact 10,760-row validation loss pass at epoch 1, loss
  `0.9400531300824814`, and entered corrected generation. No step-`541`
  selection artifact existed yet, so no Stage-B metric has been selected or
  compared. Its first artifact is tentatively due around `23:35--23:55`.
- Both logs had zero targeted fault markers. Owned state is two running
  A100-40GB `gpu:ampere` jobs, with no owned A100-80GB or L40S work. Quota is
  home `52.0%` and scratch `37.4%`. Kombuys was not accessed and remains
  read-only with RTX 5090 untouched. Trusted base remains `16/16`, frozen
  Multilingual winners `0/8`, held-out adapter tests `0`, and Monolingual not
  started. The verified Sheet labels remain `Qwen Results`/`GDN Results`, GDN
  E/F/G remain blank, and no Sheet or Hugging Face write occurred.

## Enhanced Stage-B b0 starts and passes startup verification — 22:41 SAST

- Enhanced NER Stage-B candidate `b0` job `1218997` started at
  `22:25:31 SAST` on `srvrocgpu010` with the required
  `nlpgroup/a100/nlpgroup`, `gpu:ampere:1`, 24-hour, 8-CPU envelope. It runs
  concurrently with Stage-A `1218345`, leaving two owned A100-40GB jobs and
  remaining below the three-job cap. There is no owned A100-80GB or L40S
  work.
- Startup verified all `694` immutable source/config files. The execution
  manifest SHA-256 is
  `f9f0c20312dfdb5c3cd61195ce7266039d3e99905e7a9e53940a9ff3765c14f2`.
  The immutable source snapshot is
  `/home/lmbanr001/masters/sallm_snapshots/uniform-adapter-hpo-20260811-6aabf717`.
  The run loaded the canonical local pure `GatedDeltaNetForCausalLM` path in
  BF16 with backend dispatch disabled and one visible A100-PCIE-40GB.
- The hashed `hpo_trial.json` SHA-256 is
  `9072fba1d3206550fd29a7a7ea3e8f5cefa6eb38044cfcf4324ec3d95d436aa8`.
  It reconciles exactly to registry SHA-256
  `8fdd6ea5f192a1d2ef2267353162b9a640ccd09ae0d889cb5a999645198bb726`:
  validation-only seed `42`, LR `0.00015156541821567134`, LoRA rank/alpha
  `8/16`, dropout `0.028929591178894043`, and warmup
  `0.061286738514900206`. The frozen pure-GDN wrapper supplies architecture-
  complete targets `[q_proj,k_proj,v_proj,a_proj,b_proj,g_proj,o_proj,
  gate_proj,up_proj,down_proj]`; runtime reports `2,325,312` trainable of
  `129,752,296` post-resize parameters. Training coverage is `4,323` rows and
  declared validation coverage is `10,760`; no held-out split was accessed.
- At `22:41`, `1218997` was healthy near step `268/8115` with zero targeted
  fault markers. Its first epoch validation artifact is tentatively due around
  `23:30--23:50`. Stage-A LR `1.5e-4` job `1218345` remained healthy in
  epoch-10 corrected generation after exact 10,760-row validation loss
  `0.36745544206697256`; its next selection artifact is tentatively due around
  `23:05--23:25`.
- Quota remains home `52.0%` and scratch `37.4%`. Kombuys was not accessed
  and remains read-only with RTX 5090 untouched. Trusted base remains `16/16`,
  frozen Multilingual winners `0/8`, held-out adapter tests `0`, and
  Monolingual not started. The verified Sheet labels remain `Qwen Results`/
  `GDN Results`, GDN E/F/G remain blank, and no Sheet or Hugging Face write
  occurred.

## LR-2 materially improves at epoch 9 — 22:10 SAST

- Stage-A LR `1.5e-4` job `1218345` completed a scientifically valid epoch-9
  step-`4869` artifact. Tsn/Xho/Zul span micro-F1 is
  `0.6594493450948444/0.6606450241750736/0.6849462365590899`; exact mean
  `0.6683468686096693` matches newly retained `checkpoint-4869` exactly. The
  gain over epoch 8 is `0.0085636587751213`, above the frozen `0.001`
  threshold, so the preregistered patience counter resets and continuing is
  correct. This remains a provisional Stage-A result; NER is not frozen.
- Independent checks found exactly `192` rows, `64` per language, and all raw
  predictions nonempty. Unique raw-output counts are `41/56/38`. Debug and
  trainer-state SHA-256 values are
  `1db3d3a3061e1fa9cd253966690be461e9a373f85e760d0dad9753702f3545f2` and
  `7cdfff25a0a058047b75a74ff2da6a277c998581c8ca9296bccd592a7a95e641`.
- At `22:10`, job `1218345` had resumed healthy epoch-10 training near step
  `5085/8115`, with zero targeted fault markers. Its next validation artifact
  is tentatively due around `23:05--23:25`; terminal state remains dependent
  on the frozen improvement and patience rules. Enhanced Stage-B `b0` job
  `1218997` remains `PENDING (Resources)`, has no output or metric, and keeps
  the dynamic projected start `2026-08-12 00:48:20 SAST`.
- Owned state remains one running and one pending A100-40GB `gpu:ampere` job,
  with no owned A100-80GB or L40S work. Quota is home `52.0%` and scratch
  `37.4%`. Kombuys was not accessed and remains read-only with RTX 5090
  untouched. Trusted base remains `16/16`, frozen Multilingual winners
  `0/8`, held-out adapter tests `0`, and Monolingual not started. The verified
  Sheet labels remain `Qwen Results`/`GDN Results`, GDN E/F/G remain blank,
  and no Sheet or Hugging Face write occurred.

## LR-2 materially improves at epoch 8 — 21:10 SAST

- Stage-A LR `1.5e-4` job `1218345` completed a scientifically valid epoch-8
  step-`4328` artifact. Tsn/Xho/Zul span micro-F1 is
  `0.6519292604501108/0.6511328250831673/0.6762875439703658`; exact mean
  `0.659783209834548` matches newly retained `checkpoint-4328` exactly. The
  gain over epoch 7 is `0.0073393242457308`, above the frozen `0.001`
  threshold, so the preregistered patience counter resets and continuing is
  correct. This remains a provisional Stage-A result; NER is not frozen.
- Independent checks found exactly `192` rows, `64` per language, and all raw
  predictions nonempty. Unique raw-output counts are `41/53/37`. Debug and
  trainer-state SHA-256 values are
  `0c441991f7a5d964886497e91ef18fbe4713e0b61d0fad9002ffb60da347cd93` and
  `868ad3b3537e4c6e5412085213584ad5bd6586534488dd21d664dbb80afd2cab`.
- At `21:10`, job `1218345` had resumed healthy epoch-9 training near step
  `4736/8115`, with zero targeted fault markers. Its next validation artifact
  is tentatively due around `21:55--22:15`; terminal state remains dependent
  on the frozen improvement and patience rules. Enhanced Stage-B `b0` job
  `1218997` remains `PENDING (Resources)`, has no output or metric, and keeps
  the dynamic projected start `2026-08-12 00:48:20 SAST`.
- Owned state remains one running and one pending A100-40GB `gpu:ampere` job,
  with no owned A100-80GB or L40S work. Quota is home `52.0%` and scratch
  `37.4%`. Kombuys was not accessed and remains read-only with RTX 5090
  untouched. Trusted base remains `16/16`, frozen Multilingual winners
  `0/8`, held-out adapter tests `0`, and Monolingual not started. The verified
  Sheet labels remain `Qwen Results`/`GDN Results`, GDN E/F/G remain blank,
  and no Sheet or Hugging Face write occurred.

## LR-2 epoch 8 validation in progress; Stage B projection advances — 20:10 SAST

- Stage-A LR `1.5e-4` job `1218345` remained healthy on A100-40GB in its
  epoch-8 full 10,760-row validation/generation callback. The log was active
  at `20:09`, with per-language validation coverage progressing and zero
  targeted fault markers. No step-`4328` selection artifact existed, so the
  retained provisional best remains epoch 7 mean F1 `0.6524438855888172`.
  The next artifact is tentatively due around `20:45--21:05`.
- Enhanced Stage-B `b0` job `1218997` remains pending with no output or
  metric, but Slurm changed its reason from `Priority` to `Resources` and
  advanced the dynamic projected start from `2026-08-12 07:07` to
  `2026-08-12 00:48:20 SAST`. No submission or intervention was made.
- Owned state remains one running and one pending A100-40GB `gpu:ampere` job,
  with no owned A100-80GB or L40S work. Quota is home `52.0%` and scratch
  `37.4%`. Kombuys was not accessed and remains read-only with RTX 5090
  untouched. Trusted base remains `16/16`, frozen Multilingual winners
  `0/8`, held-out adapter tests `0`, and Monolingual not started. The verified
  Sheet labels remain `Qwen Results`/`GDN Results`, GDN E/F/G remain blank,
  and no Sheet or Hugging Face write occurred.

## LR-2 materially improves again at epoch 7 — 19:40 SAST

- Stage-A LR `1.5e-4` job `1218345` completed a scientifically valid epoch-7
  step-`3787` artifact. Tsn/Xho/Zul span micro-F1 is
  `0.6412354804645751/0.6379301203541202/0.678166055947756`; exact mean
  `0.6524438855888172` matches newly retained `checkpoint-3787` exactly.
  The gain over epoch 6 is `0.0161418127602831`, above the frozen `0.001`
  threshold, so the preregistered patience counter resets and continuing is
  correct. This is the provisional Stage-A leader only; NER remains unfrozen.
- Independent checks found exactly `192` rows, `64` per language, and all raw
  predictions nonempty. Unique raw-output counts are `41/55/38`. Debug and
  trainer-state SHA-256 values are
  `79770d80307796c32b169b7de60cb512ecb29db0b40583fffe500c1dc56b355d` and
  `ed2174acabfb35092ffab49335b472ca1a981a69bca397a72fa727790fca1252`.
- At `19:40`, job `1218345` had resumed healthy epoch-8 training near step
  `3818/8115`, with zero targeted fault markers. Its next validation artifact
  is tentatively due around `20:45--21:05`; terminal state remains dependent
  on the frozen improvement and patience rules. Enhanced Stage-B `b0` job
  `1218997` remains `PENDING (Priority)`, has no output or metric, and retains
  the dynamic projected start `2026-08-12 07:07 SAST`.
- Owned state remains one running and one pending A100-40GB `gpu:ampere` job,
  with no owned A100-80GB or L40S work. Quota is home `52.0%` and scratch
  `37.4%`. Kombuys was not accessed and remains read-only with RTX 5090
  untouched. Trusted base is `16/16`, frozen Multilingual winners `0/8`,
  held-out adapter tests `0`, and Monolingual not started. The verified Sheet
  labels remain `Qwen Results`/`GDN Results`, GDN E/F/G remain blank, and no
  Sheet or Hugging Face write occurred.

## LR-2 epoch 6 gives a sub-threshold retained improvement — 18:40 SAST

- Stage-A LR `1.5e-4` job `1218345` completed a scientifically valid epoch-6
  step-`3246` artifact. Tsn/Xho/Zul span micro-F1 is
  `0.6250316696224484/0.6259589467136139/0.6579156021495404`; exact mean
  `0.6363020728285341` matches newly retained `checkpoint-3246` exactly.
  The numerical gain over epoch 5 is only `0.0001544210587123`, below the
  frozen `0.001` threshold, so continuing under the preregistered patience
  rule rather than intervening is correct. It remains the provisional Stage-A
  leader, but NER is not frozen.
- Independent checks found exactly `192` rows, `64` per language, and all raw
  predictions nonempty. Unique raw-output counts are `41/56/38`. Debug and
  trainer-state SHA-256 values are
  `2b53aee1c30b8df1db0361e11fc0dbf1f0507e0167c64b2c723af906df534a42` and
  `fd3e87b4341bdeb88f1a51cff2fe583728d0e8dc0fe2b69fd8a35ded93909e14`.
- At `18:40`, job `1218345` had resumed healthy epoch-7 training near step
  `3493/8115`, with zero targeted fault markers. Its next validation artifact
  is tentatively due around `19:20--19:40`; terminal state depends on the
  frozen improvement and patience rules. Enhanced Stage-B `b0` job `1218997`
  remains `PENDING (Priority)`, has no output or metric, and retains the
  dynamic projected start `2026-08-12 07:07 SAST`.
- Owned state remains one running and one pending A100-40GB `gpu:ampere` job,
  with no owned A100-80GB or L40S work. Quota is home `52.0%` and scratch
  `37.4%`. Kombuys was not accessed and remains read-only with RTX 5090
  untouched. Trusted base is `16/16`, frozen Multilingual winners `0/8`,
  held-out adapter tests `0`, and Monolingual not started. The verified Sheet
  labels remain `Qwen Results`/`GDN Results`, GDN E/F/G remain blank, and no
  Sheet or Hugging Face write occurred.

## LR-2 becomes provisional Stage-A leader at epoch 5 — 18:12 SAST

- Stage-A LR `1.5e-4` job `1218345` produced a scientifically valid epoch-5
  step-`2705` validation artifact. Tsn/Xho/Zul span micro-F1 is
  `0.631073144687617/0.6197408414653157/0.6576289691565328`; the exact
  arithmetic mean `0.6361476517698218` matches the retained
  `checkpoint-2705` `best_metric` exactly. This provisionally exceeds
  completed LR `8e-5` job `1218344` at `0.6173130857642662`, but it does not
  freeze NER because `1218345` is still running and enhanced Stage B/C remain
  incomplete.
- Independent checks found exactly `192` rows, `64` per language, all raw
  predictions nonempty, and unique raw-output counts `41/57/37`. Debug and
  trainer-state SHA-256 values are
  `f43d713808a41470691a4446a1724b9607eb1a0916ab914fda7dee5e29e3436c` and
  `556649389b5f6f657c5bba5c714dfbb2b45ad5816ec4d51dd8dd04ad69357977`.
- At `18:12`, job `1218345` was healthy in epoch-6 corrected generation after
  exact 10,760-row validation loss `0.34308718429622154`, with zero targeted
  fault markers. Its next F1 artifact is tentatively due around
  `18:20--18:40`; terminal time remains output- and early-stopping-dependent.
  Enhanced NER Stage-B `b0` job `1218997` remains `PENDING (Priority)`, has
  produced no output or metric, and has a dynamic projected start of
  `2026-08-12 07:07 SAST`.
- Owned state is one running and one pending A100-40GB `gpu:ampere` job, with
  no owned A100-80GB or L40S work. Quota is home `52.0%` and scratch `37.4%`.
  Kombuys was not accessed and remains read-only with RTX 5090 untouched.
  Trusted base remains `16/16`, frozen Multilingual winners `0/8`, held-out
  adapter tests `0`, and Monolingual not started. The workbook remains at its
  verified `Qwen Results`/`GDN Results` labels with GDN E/F/G blank; no Sheet
  or Hugging Face write occurred.

## NER LR-0 terminal; enhanced HPO remains queued — 17:10 SAST

- Stage-A LR `3e-5` job `1218343` completed `0:0` at `17:07:40 SAST`
  after `18:35:25` on one A100-40GB. It completed all 15 epochs with zero
  targeted fault markers. The retained validation-only winner is epoch 13,
  step `7033`, exact mean NER F1 `0.500502987595103`; trainer state SHA-256 is
  `a483590e2d9858c5f77f8838ada946e63085f39e1c2bb4d3d3bd391212f1ae04`.
- The restored `final_adapter` exists. Its weight/config SHA-256 values are
  `ed8d5c1c0cd4944f71bb4e2cdedc54ba398cf936c15308c7bacd74c093e97c50`
  and `ccedbaf68858080e8255e272dcf6dd605ac9d1780fe9895e6017ada7526a3c5f`;
  retained/final adapter configs are byte-identical. The checkpoint uses
  legacy `adapter_model.bin` while the final adapter uses safetensors, so an
  exact tensor comparison remains unclaimed until a compute-side read-only
  check is available. Execution-manifest SHA-256 remains
  `cadacc9f02a3d1f68d0c7f34599d3c2db979559291a78d918631594d2f6b61ff`.
- LR `1.5e-4` job `1218345` remained healthy in epoch-5 validation/generation
  with zero targeted faults. Enhanced Stage-B `b0` job `1218997` remained
  `PENDING (Priority)` despite the released owned GPU; two other-user jobs
  still occupy the remaining A100-40GBs and Slurm's dynamic projection moved
  to `2026-08-12 07:07 SAST`. It has created no output or metric.
- Home/scratch quota is `52.0%/37.4%`. The only owned GPU family remains
  A100-40GB; Kombuys was not accessed and RTX 5090 remains untouched. Trusted
  base stays `16/16`, frozen Multilingual winners `0/8`, held-out adapter
  tests `0`, and Mono not started. `GDN Results` E/F/G remain blank; no Sheet
  or Hugging Face write occurred. Immediate blockers are LR-2 terminal state
  and scheduler start of `1218997`.

## Uniform enhanced HPO frozen and first Stage-B trial submitted — 14:24 SAST

- The enhanced process is now a single architecture-neutral validation-only
  registry, launcher, trial record, and ranker. Built-in profiles cover pure
  GDN, LLaMA-125M, Mamba-125M, xLSTM-125M, and the historical Qwen-GDN
  hybrid; a custom profile accepts an explicit base config, architecture, and
  checkpoint. Every profile dry-resolved the exact same `b0` point. Model
  results remain independent and no historical metric entered the search.
- The frozen preregistration SHA-256 is
  `b7d50b8d20a2ea09699ec343cccbda438e05c088e6656dea1bbf62df03135c3c`.
  The generic 11-candidate registry SHA-256 is
  `8fdd6ea5f192a1d2ef2267353162b9a640ccd09ae0d889cb5a999645198bb726`.
  Pure-GDN NER `b0` is LR `0.00015156541821567134`, rank `8`, alpha `16`,
  dropout `0.028929591178894043`, warmup `0.061286738514900206`, seed/data
  seed `42`.
- Shell syntax, focused Ruff, all five profile dry runs, and the complete
  local suite pass (`115 passed`). Uniform wrapper, pure-GDN backend,
  protocol/ranker, manifest builder, and focused-test SHA-256 values are
  `f92329a9f510edddf7a5cf521cd52808309511d8a67d942838dc3d7bf44b4e59`,
  `eeb88f17ff5d2cb4a2d4b16851ab0ddbc1c2c667f3cd4d34213d880dc88eaa53`,
  `edf5638cae9d783b840088a16079770a907208d518602bad10e210829ed53568`,
  `55095f225dba0710b23831fb306ca3e471cbb1bbaaeec48529dd080fec235024`,
  and `fd039ada35691fdb095f2ef2b0703f4612ba821580e44c12d8d3e09706a5fcde`.
- The read-only HEX snapshot is
  `/home/lmbanr001/masters/sallm_snapshots/uniform-adapter-hpo-20260811-6aabf717`.
  All `694/694` local/HEX source/config hashes match at source-set SHA-256
  `6aabf717b808b4bbe1471593d68febcebb5694940b605efea3f027b05fe6b914`;
  deployment-manifest SHA-256 is
  `da1c456774b789578bc6d25f816e627fa0d6677de928802a89fb5a226557b285`.
- Pure-GDN NER Stage-B `b0` was submitted as job `1218997`. Verified Slurm
  settings are account/partition/QOS `nlpgroup/a100/nlpgroup`, exactly one
  `gpu:ampere`, 24 hours, eight CPUs, one node, and working directory
  `/home/lmbanr001/masters/sallm`. At 14:24 it was `PENDING (Priority)`;
  all four physical A100-40GB devices were occupied by owned jobs
  `1218343/1218345` and other-user jobs `1217866/1215880`. Slurm's dynamic
  projection was `2026-08-12 12:32 SAST`, so startup manifest/model/LoRA
  verification remains pending rather than failed.
- Owned Stage-A jobs `1218343/1218345` remained healthy in corrected
  generation with zero targeted fault markers. The only owned GPU family is
  A100-40GB; Kombuys was not accessed and remains read-only with RTX 5090
  untouched. Quota after the immutable snapshot is home `52.0%`, scratch
  `37.3%`. Trusted base remains `16/16`, frozen Multilingual winners `0/8`,
  held-out adapter tests `0`, and Monolingual not started. `GDN Results`
  E/F/G remain blank; no Sheet or Hugging Face write occurred. The immediate
  blocker is scheduler start of `1218997`, followed by its startup checks.

## LR-2 improves sharply at epoch 2; LR-0 enters epoch-13 generation — 14:04 SAST

- LR `1.5e-4` job `1218345` completed its scientifically valid epoch-2
  step-1082 artifact. Tsn/Xho/Zul span micro-F1 is
  `0.46153846153841205/0.44188790560466973/0.5015956141068156`; exact mean
  `0.46834066041663247` exactly matches retained `checkpoint-1082`
  `best_metric`. This materially improves its epoch-1 mean
  `0.255946479788738`, but remains below LR-0's current best
  `0.491048079892295` and the completed LR-1 best `0.6173130857642662`.
- The debug artifact has exactly `192` rows and `64` per language. All
  `192/192` raw predictions are nonempty; normalized nonempty counts are
  `46/57/35`, with unique raw-output counts `43/56/35`. Debug and trainer
  state SHA-256 values are
  `4f91866166291dd283b46f5c9237cf135cb0327c921a7453cdcda647917651b9`
  and
  `8de32147b6ed3cad615d28ce150f31e95107177773df54007caf047b270b3804`.
- Job `1218345` resumed healthy epoch-3 training near `1576/8115`. LR
  `3e-5` job `1218343` reached epoch-13 step `7033`, completed exact
  10,760-row health-only validation loss `0.40935960507304253`, and entered
  corrected generation; no epoch-13 F1 artifact existed at the check. Its
  next artifact is tentatively due around `14:35--14:55`, output-dependent.
- At `14:04:24`, jobs `1218343/1218345` were the only owned active jobs, both
  healthy on `srvrocgpu010` with one A100-40GB `gpu:ampere` each. There was
  no owned A100-80GB/L40S work. Home/scratch quota was `33.7%/37.3%`.
  Kombuys remained read-only and untouched. Trusted base remains `16/16`,
  frozen Multilingual winners `0/8`, held-out adapter tests `0`, Monolingual
  not started, and `GDN Results` adapter columns E/F/G blank. NER remains
  unfrozen until both active trials terminate.

## LR-0 improves at epoch 12; LR-2 is in epoch-2 generation — 13:31 SAST

- LR `3e-5` job `1218343` completed a scientifically valid epoch-12
  step-6492 artifact. Tsn/Xho/Zul span micro-F1 is
  `0.4837698959768896/0.47802141764400913/0.5113529260559865`; exact mean
  `0.491048079892295` matches `checkpoint-6492` `best_metric` to displayed
  precision. This improves epoch 11 by `0.0033917618186639`, above the frozen
  `0.001` threshold, so continuing into epoch 13 is required by the
  preregistered patience rule.
- The debug artifact has exactly `192` rows and `64` per language. All
  `192/192` raw predictions are nonempty; normalized nonempty counts are
  `44/57/37` and unique raw-output counts are `42/56/37`. Debug and trainer
  state SHA-256 values are
  `858611a1c60f12b2ed25de7c3545c1c42c186b1c21b620d6ebc5e3d0df914892`
  and
  `50da38ed8e2a4dd53594cba421c6635da9a562f9815187b6a20650446834629f`.
- LR `1.5e-4` job `1218345` completed its exact 10,760-row epoch-2
  health-only validation loss `0.3859550674608649` and remained in corrected
  generation. Its step-1082 F1 artifact was absent at the check, so no new
  within-LR or cross-LR comparison was made. Its next artifact is tentatively
  due around `13:40--14:00`; LR-0's next epoch artifact is tentatively due
  around `14:35--15:00`, both output-dependent.
- At `13:31:06`, jobs `1218343/1218345` were the only owned active jobs and
  were healthy on `srvrocgpu010`, one A100-40GB `gpu:ampere` each. There was
  no owned A100-80GB or L40S work. Home/scratch quota was `33.7%/37.3%`.
  Kombuys remained read-only and untouched. Trusted base remains `16/16`,
  frozen Multilingual winners `0/8`, held-out adapter tests `0`, Monolingual
  not started, and publication blocked.
- The results workbook label correction is verified: historical hybrid data
  is now `Qwen Results` and `Qwen` in the comparison dashboard, while the
  actual pure model tab is `GDN Results`. Metrics and formatting were
  preserved; pure-GDN adapter columns E/F/G remain blank.

## LR-0 epoch 11 and LR-2 epoch 1 reconcile cleanly — 12:31 SAST

- LR `3e-5` job `1218343` completed its scientifically valid epoch-11
  step-5951 artifact. Tsn/Xho/Zul span micro-F1 is
  `0.48576558091813415/0.4750207296848588/0.5021826436179002`; exact mean
  `0.4876563180736311` matches `checkpoint-5951` `best_metric` exactly. The
  numerical gain over epoch-10 `0.4874799449169605` is only
  `0.0001763731566706`, below the frozen early-stopping threshold `0.001`, so
  continuing under the preregistered patience logic rather than intervening
  is correct. The trainer still retains the numerically best checkpoint.
- LR `1.5e-4` job `1218345` completed its first scientifically valid epoch-1
  step-541 artifact. Tsn/Xho/Zul span micro-F1 is
  `0.21524997672465432/0.25427872860630824/0.29831073403525155`; exact mean
  `0.255946479788738` matches retained `checkpoint-541` exactly. This is an
  initial within-LR metric only; no cross-LR winner is frozen.
- Each debug artifact has exactly `192` rows and `64` per language, all raw
  predictions nonempty, and one constant F1 per language. Unique raw-output
  counts are `40/55/37` for LR-0 and `40/56/35` for LR-2. LR-0 debug/state
  SHA-256 values are
  `830075fff380b3295f97c9b2dad92607b74925fc303d3b9f13624d12bf0a2e30`/
  `6f60156dbfb4d9dd5950e043aa7ca853a7ef9b7076f0e3a97470e11ae268a9bf`;
  LR-2 values are
  `bfbb57972a95db33d362210fb191c1bc9fadb7878481b49ebb51be5fb6285ac1`/
  `753e3ed5d4231890e69fed7b7bf45116df34a4c105d5fc1d10187e83cd0c3ffc`.
- Jobs `1218343/1218345` resumed healthy epoch-12/epoch-2 training near
  `6304/8115` and `673/8115`; targeted fault counts were zero. Their next
  artifacts are tentatively due around `13:20--14:00`, output-dependent.
- Exactly two A100-40GB `gpu:ampere` jobs are active, with no A100-80GB/L40S
  overlap. Quota is home `33.7%`, scratch `37.3%`; Kombuys remains read-only
  and untouched. NER remains unfrozen until both terminate. Trusted base is
  `16/16`, frozen Multilingual winners `0/8`, held-out adapter tests `0`,
  Monolingual not started, Sheet E/F/G blank, and publication blocked.

## NER LR-1 terminates cleanly and LR-2 starts — 11:22 SAST

- LR `8e-5` job `1218344` completed `0:0` at `11:07:19 SAST` after
  `12:06:17`. Its epoch-10 step-5410 artifact has Tsn/Xho/Zul span micro-F1
  `0.5899299247339236/0.6182086718007731/0.624611283873784`; exact mean
  `0.6109166268028269`. This is its second consecutive non-improvement after
  the epoch-8 best `0.6173130857642662`, so the frozen patience-2 rule stopped
  training at epoch 10 and retained `checkpoint-4328`.
- Independent checks found exactly `192` rows, `64` per language, all raw
  predictions nonempty, one constant F1 per language, and unique raw-output
  counts `40/55/38`. Debug SHA-256 is
  `523954f78c8a40b86e8be0086c53964b9d170efd9fdc8c5866bbcd8772a9260a`;
  retained state SHA-256 remains
  `dabc4dd4d1146cf3fd88d9e83b6e658aa9abd2fdd5c30314efbf09f198433935`.
- A read-only comparison on the active compute allocation verified all
  `424/424` final-adapter tensors exactly equal the retained checkpoint, with
  zero missing, extra, or mismatched keys. Final adapter/config, retained
  checkpoint weights/config/state, and final README SHA-256 values are
  `3bcf812b57d4ad5197adf138211fd2daf3504c35bd768fc99b7f0457ede91e40`,
  `4e6bd64ed9b81586cd1cd90051f151781a5fad98e37682b8f03dec19b7887be2`,
  `abbcb4d5d2861629b02b49e692227f39fc6878bbe9e209e4eea3c63e97cfde41`,
  `4e6bd64ed9b81586cd1cd90051f151781a5fad98e37682b8f03dec19b7887be2`,
  `dabc4dd4d1146cf3fd88d9e83b6e658aa9abd2fdd5c30314efbf09f198433935`,
  and `c099b29dcf34b38e481827dbfc9912967019aeed44c084e5b207f6160addf7e1`.
- LR `1.5e-4` job `1218345` started immediately at `11:07:19 SAST` on the
  released A100-40GB. It verified all `691` immutable source/config files;
  execution-manifest SHA-256 is
  `3181df56896df904fc81808eaaff9cdbd75d410d93b59343f3dd20f2e08ebeb1`.
  Startup confirmed canonical `GatedDeltaNetForCausalLM`, the frozen
  label-smoothed causal-shift trainer, `4,649,088` trainable LoRA parameters,
  15-epoch cap, and no startup fault. It was healthy near `278/8115`.
- LR `3e-5` job `1218343` remained healthy near `5854/8115`. Exactly two
  A100-40GB `gpu:ampere` jobs are active with no A100-80GB/L40S overlap.
  Quota is home `33.7%`, scratch `37.3%`; Kombuys remains read-only and
  untouched. NER cannot be frozen until jobs `1218343/1218345` terminate.
  Trusted base remains `16/16`, frozen Multilingual winners `0/8`, held-out
  adapter tests `0`, Monolingual not started, Sheet E/F/G blank, and
  publication blocked.

## NER LR-1 declines slightly at epoch 9 — 10:22 SAST

- LR `8e-5` job `1218344` completed a scientifically valid step-4869
  artifact. Tsn/Xho/Zul span micro-F1 is
  `0.5981203534857126/0.6170811049235874/0.623181604880288`; exact mean
  `0.6127943544298625`. This is below its epoch-8 best
  `0.6173130857642662`, so the state correctly retains `checkpoint-4328` and
  records one non-improving validation under the frozen patience rule.
- Independent checks found exactly `192` rows, `64` per language, all raw
  predictions nonempty, one constant F1 per language, and unique raw-output
  counts `37/54/37`. Debug/state SHA-256 values are
  `2f3022f36a45bc08cd20a28783ae0ed8071a01bac8da515e5f3c450ac0ee6098`
  and `ea32813ef7e40b5e341ca9ff8e3784dbf0463833c0a1808a7231b6197430b3ee`.
- Jobs `1218343/1218344` remained healthy at the epoch-10 boundary near
  `5410/8115` and `5394/8115`; neither step-5410 artifact existed and
  targeted fault counts were zero. LR `1.5e-4` job `1218345` remains
  resource-pending with the dynamic Slurm projection around `22:32 SAST`.
- Owned state remains two running plus one pending A100-40GB `gpu:ampere`
  jobs, with no A100-80GB/L40S overlap. Quota is home `33.7%`, scratch
  `37.3%`; Kombuys remains read-only and untouched. Trusted base is `16/16`,
  frozen Multilingual winners `0/8`, held-out adapter tests `0`, Monolingual
  not started, Sheet E/F/G blank, and publication blocked.

## NER LR-0 improves at epoch 9 — 09:52 SAST

- LR `3e-5` job `1218343` completed a scientifically valid step-4869
  artifact. Tsn/Xho/Zul span micro-F1 is
  `0.4719526030504985/0.4591701498623213/0.4890492141200223`; exact mean
  `0.473390655677614` matches `checkpoint-4869` `best_metric` exactly and
  improves its epoch-8 best `0.46612916638577717`.
- Independent checks found exactly `192` rows, `64` per language, all raw
  predictions nonempty, one constant F1 per language, and unique raw-output
  counts `39/55/37`. Debug/state SHA-256 values are
  `542aebb4443c1ade6d27295011b26d9a1d0d0017a95c155c8e93d47f7c23380a`
  and `ca09f509edf9e41fd76723f818a00674387a7c24cbcb7766bbed9de820ffb2ff`.
- Job `1218343` resumed epoch 10 near `4994/8115`. LR `8e-5` job `1218344`
  remained healthy in epoch-9 generation at step `4869`; its step-4869
  artifact did not yet exist, so its provisional best remains
  `0.6173130857642662`. Targeted fault counts were zero. LR `1.5e-4` job
  `1218345` remains resource-pending with the dynamic Slurm projection around
  `22:32 SAST`.
- Owned state remains two running plus one pending A100-40GB `gpu:ampere`
  jobs, with no A100-80GB/L40S overlap. Quota is home `33.7%`, scratch
  `37.2%`; Kombuys remains read-only and untouched. Trusted base is `16/16`,
  frozen Multilingual winners `0/8`, held-out adapter tests `0`, Monolingual
  not started, Sheet E/F/G blank, and publication blocked.

## Both active NER trials improve at epoch 8 — 08:50 SAST

- LR `3e-5` job `1218343` completed a valid step-4328 artifact with
  Tsn/Xho/Zul span micro-F1
  `0.47139287123498885/0.4315817838856953/0.49541284403664726`; exact mean
  `0.46612916638577717` matches `checkpoint-4328` `best_metric` exactly and
  improves its epoch-7 best `0.43220752568933674`.
- LR `8e-5` job `1218344` completed a valid step-4328 artifact with
  Tsn/Xho/Zul span micro-F1
  `0.617786829599407/0.6111986096778952/0.6229538180154965`; exact mean
  `0.6173130857642662` matches `checkpoint-4328` `best_metric` exactly and
  improves its epoch-7 best `0.6093127253514629`. It remains the provisional
  cross-LR leader, but no winner is frozen.
- Each debug artifact has exactly `192` rows and `64` per language, all raw
  predictions nonempty, one constant F1 per language, and unique raw-output
  counts `40/55/36` for LR-0 and `41/53/37` for LR-1. Debug/state SHA-256
  values are
  `1468ea5f59e18ce46c2c4f1b33e19976ef2ab9d4100b19c2a4fa6aaf4bad7002`/
  `6aa18335580f4e13b6626d1a042b7145092a206b323607a342f8d327227b0236`
  and
  `ac1f0ac2ea13cf7e0e6910c8fd3c06cd3fd3398819a6cb997405951ade6b89c3`/
  `dabc4dd4d1146cf3fd88d9e83b6e658aa9abd2fdd5c30314efbf09f198433935`.
- Both runs resumed epoch 9 and were healthy at `4699/8115` and `4406/8115`;
  targeted fault counts were zero. LR `1.5e-4` job `1218345` remains
  resource-pending with a dynamic Slurm projection around `22:32 SAST`.
  Epoch-9 artifacts are tentatively due around `09:40--10:15`, but terminal
  timing remains output-dependent.
- Owned state remains two running plus one pending A100-40GB `gpu:ampere`
  jobs, with no A100-80GB/L40S overlap. Quota is home `33.7%`, scratch
  `37.2%`; Kombuys remains read-only and untouched. Trusted base is `16/16`,
  frozen Multilingual winners `0/8`, held-out adapter tests `0`, Monolingual
  not started, Sheet E/F/G blank, and publication blocked.

## Both active NER trials in epoch-8 generation — 08:20 SAST

- Jobs `1218343/1218344` remain healthy on two A100-40GB `gpu:ampere`
  allocations and both reached epoch-8 step `4328`. LR `3e-5` completed
  exact 10,760-row health-only validation loss `0.413688801478276`; LR
  `8e-5` completed `0.37547330608155205`. Both entered generation.
- LR `8e-5` reported epoch-8 causal training token accuracy `1.0`. This is
  training-health/possible-saturation evidence only; the preregistered
  validation mean per-language span F1 remains the stopping and selection
  metric. Neither step-4328 F1 artifact existed, so no scientific ordering
  changed.
- Targeted logs showed no traceback, CUDA OOM/error, NCCL, non-finite,
  manifest, or coverage fault. LR `1.5e-4` job `1218345` remains
  resource-pending with no released slot. Next LR-0/LR-1 F1 artifacts are
  tentatively due around `08:50--09:10` and `09:15--09:35`.
- Owned state remains two running plus one pending A100-40GB jobs, no
  A100-80GB/L40S overlap. Quota home `33.7%`, scratch `37.2%`; Kombuys
  read-only and untouched. Base `16/16` trusted, frozen Multilingual winners
  `0/8`, held-out `0`, Mono not started, Sheet E/F/G blank.

## NER LR-1 exceeds 0.60 mean F1 at epoch 7 — 07:50 SAST

- LR `8e-5` job `1218344` completed its scientifically valid step-3787
  artifact. Tsn/Xho/Zul span micro-F1 is
  `0.6016624884548913/0.6030139935413927/0.6232616940581044`; exact mean
  `0.6093127253514629` matches `checkpoint-3787` `best_metric` exactly.
  This materially improves epoch-6 best `0.5833389060125157` and preserves
  its provisional lead over LR `3e-5` best `0.43220752568933674`. The frozen
  stopping condition has not been met; no cross-LR winner is frozen.
- Independent checks found exactly `192` rows, `64` per language, all raw
  predictions nonempty, unique raw-prediction counts `42/57/39`, and one
  constant stored metric per language. Debug/state SHA-256 values are
  `762e48ddd79d6adae8694c90a7326a7261eff33a70ba65de2f6f7fe4361beee4`
  and `f58a7bea3bf0361afa62b91c4fff2cba151d96a71d30eb153f926974ab139522`.
- Job `1218344` resumed healthy epoch-8 training near step `4044/8115`.
  Job `1218343` reached epoch-8 step `4328` and entered validation. LR
  `1.5e-4` `1218345` remains resource-pending. Targeted fault scans remained
  empty.
- Owned state remains two running plus one pending A100-40GB jobs, no
  A100-80GB/L40S overlap. Quota home `33.7%`, scratch `37.2%`; Kombuys
  read-only and untouched. Base `16/16` trusted, frozen Multilingual winners
  `0/8`, held-out `0`, Mono not started, Sheet E/F/G blank. Next artifacts
  are tentatively due around `08:30--08:50` for LR-0 and `08:55--09:15` for
  LR-1; terminal and LR-2 start times remain output/scheduler dependent.

## NER LR-0 improves again at epoch 7 — 07:20 SAST

- LR `3e-5` job `1218343` completed its scientifically valid step-3787
  artifact. Tsn/Xho/Zul span micro-F1 is
  `0.45303867403309933/0.37934005778290913/0.4642438452520018`; exact mean
  `0.43220752568933674` matches `checkpoint-3787` `best_metric` exactly.
  This improves epoch-6 best `0.42604881329910455`, so the frozen stopping
  rule keeps it active. It remains below LR `8e-5`'s provisional best
  `0.5833389060125157`.
- Independent checks found exactly `192` rows, `64` per language, all raw
  predictions nonempty, unique raw-prediction counts `43/57/37`, and one
  constant stored metric per language. Debug/state SHA-256 values are
  `f96f7d5d180c4740259f8d4ab917f59afe129c738d251d8383ae24dd1f5a3fb9`
  and `8ee69cba2c05719a21411b1cad79a534521207c98b2bbe3f78cf7534dfa2e56e`.
- Job `1218343` resumed healthy epoch-8 training near step `3830/8115`.
  Job `1218344` reached epoch-7 step `3787`, completed exact 10,760-row
  health-only loss `0.3714326199102579`, and entered generation; its
  step-3787 artifact was not present. LR `1.5e-4` `1218345` remains
  resource-pending. Targeted fault scans remained empty.
- Owned state remains two running plus one pending A100-40GB jobs, no
  A100-80GB/L40S overlap. Quota home `33.7%`, scratch `37.2%`; Kombuys
  read-only and untouched. Base `16/16` trusted, frozen Multilingual winners
  `0/8`, held-out `0`, Mono not started, Sheet E/F/G blank. LR-1's epoch-7
  artifact is tentatively due around `07:45--08:05`; terminal and LR-2 start
  times remain output/scheduler dependent.

## NER LR-1 improves modestly at epoch 6 — 06:50 SAST

- LR `8e-5` job `1218344` completed its scientifically valid step-3246
  artifact. Tsn/Xho/Zul span micro-F1 is
  `0.569631400176073/0.5751914241959685/0.6051938936655055`; exact mean
  `0.5833389060125157` matches `checkpoint-3246` `best_metric` exactly.
  This improves epoch-5 best `0.5766600986590078` and preserves its
  provisional lead over LR `3e-5` best `0.42604881329910455`. The frozen
  stopping condition has not been met; no cross-LR winner is frozen.
- Independent checks found exactly `192` rows, `64` per language, all raw
  predictions nonempty, unique raw-prediction counts `41/57/37`, and one
  constant stored metric per language. Debug/state SHA-256 values are
  `f41dac64297ceb730a0e2ef0e6cb20f3f98a126f8305fbb3a68007daf9f4bbcb`
  and `66cf56740fa30ab12c74f48aadf56529fb51d50db8800cc8fef7580bd1c8d762`.
- Job `1218344` resumed healthy epoch-7 training near step `3723/8115`.
  Job `1218343` reached epoch-7 step `3787`, completed exact 10,760-row
  health-only loss `0.4281238314830681`, and entered generation; its
  step-3787 F1 artifact was not present. LR `1.5e-4` `1218345` remains
  resource-pending. Targeted fault scans remained empty.
- Owned state remains two running plus one pending A100-40GB jobs, no
  A100-80GB/L40S overlap. Quota home `33.7%`, scratch `37.2%`; Kombuys
  read-only and untouched. Base `16/16` trusted, frozen Multilingual winners
  `0/8`, held-out `0`, Mono not started, Sheet E/F/G blank. Next artifacts
  are tentatively due around `07:15--07:35` for LR-0 and `07:40--08:00` for
  LR-1; terminal and LR-2 start times remain output/scheduler dependent.

## NER LR-0 improves modestly at epoch 6 — 06:20 SAST

- LR `3e-5` job `1218343` completed its scientifically valid step-3246
  artifact. Tsn/Xho/Zul span micro-F1 is
  `0.4330256921925169/0.4061216105176164/0.4389991371871805`; exact mean
  `0.42604881329910455` matches `checkpoint-3246` `best_metric` exactly.
  This modestly improves epoch-5 best `0.41520939686372715`, so the frozen
  rule correctly keeps it active. It remains below LR `8e-5`'s provisional
  best `0.5766600986590078`.
- Independent checks found exactly `192` rows, `64` per language, all raw
  predictions nonempty, unique raw-prediction counts `42/56/36`, and one
  constant stored metric per language. Debug/state SHA-256 values are
  `518cd110f032c046f5a8985853910c838235de1ed461d8230d9a8d69efed400f`
  and `8d197eeb184ea6252a5db389feacae26f8d979a59aa087665fcb7321f57b931e`.
- Job `1218343` resumed healthy epoch-7 training near step `3570/8115`.
  Job `1218344` remained in epoch-6 generation; its step-3246 artifact had
  not appeared. LR `1.5e-4` `1218345` remains resource-pending. Targeted
  fault scans remained empty.
- Owned state remains two running plus one pending A100-40GB jobs, no
  A100-80GB/L40S overlap. Quota home `33.7%`, scratch `37.2%`; Kombuys
  read-only and untouched. Base `16/16` trusted, frozen Multilingual winners
  `0/8`, held-out `0`, Mono not started, Sheet E/F/G blank. LR-1's epoch-6
  artifact remains tentatively due around `06:30--06:50`; terminal and LR-2
  start times remain output/scheduler dependent.

## Both active NER trials in epoch-6 generation — 05:50 SAST

- Jobs `1218343/1218344` remain healthy on two A100-40GB `gpu:ampere`
  allocations and both reached epoch-6 step `3246`. LR `3e-5` completed
  exact 10,760-row health-only validation loss `0.4369589653156947`; LR
  `8e-5` completed `0.36010172110064764`. Both then entered generation.
- Neither step-3246 F1 artifact existed, so retained bests and provisional
  ordering remain unchanged. Targeted fault scans showed no traceback, CUDA
  OOM/error, NCCL, non-finite, manifest, or coverage fault. LR `1.5e-4` job
  `1218345` remains resource-pending with no released slot.
- The next LR-0/LR-1 F1 artifacts are tentatively due around `06:05--06:20`
  and `06:25--06:45`, respectively. Owned state remains two running plus one
  pending A100-40GB jobs, no A100-80GB/L40S overlap. Quota home `33.7%`,
  scratch `37.2%`; Kombuys read-only and untouched. Base `16/16` trusted,
  frozen Multilingual winners `0/8`, held-out `0`, Mono not started, Sheet
  E/F/G blank.

## NER LR-1 reaches 0.5767 mean F1 at epoch 5 — 05:20 SAST

- LR `8e-5` job `1218344` completed its scientifically valid step-2705
  artifact. Tsn/Xho/Zul span micro-F1 is
  `0.5732620320855114/0.5689924990555869/0.5877257648359251`; exact mean
  `0.5766600986590078` matches `checkpoint-2705` `best_metric` exactly.
  This replaces epoch-4 best `0.5335878171485439` and continues its
  provisional lead over LR `3e-5` best `0.41520939686372715`. The coherent,
  monotonic repaired learning curve is strong evidence against the previous
  low NER being a valid model-quality result, but no cross-LR winner is frozen.
- Independent checks found exactly `192` rows, `64` per language, all raw
  predictions nonempty, unique raw-prediction counts `38/56/38`, and one
  constant stored metric per language. Debug/state SHA-256 values are
  `0b0eec7f9bedd2d9472e5f801c3919bc4a32d6a383191f109e268845d36657d8`
  and `a9ae28ab8377bea02e7409ea1c9d1802c100f8adfa074ff728a3b398b3d383ed`.
- Job `1218344` resumed healthy epoch-6 training near step `2858/8115`.
  Job `1218343` reached epoch-6 step `3246` and entered its validation
  callback. LR `1.5e-4` `1218345` remains resource-pending. Targeted fault
  scans remained empty.
- Owned state remains two running plus one pending A100-40GB jobs, no
  A100-80GB/L40S overlap. Quota home `33.7%`, scratch `37.2%`; Kombuys
  read-only and untouched. Base `16/16` trusted, frozen Multilingual winners
  `0/8`, held-out `0`, Mono not started, Sheet E/F/G blank. Next LR-0/LR-1
  artifacts are tentatively due around `06:10` and `06:35--06:55`; terminal
  and LR-2 start times remain output/scheduler dependent.

## NER LR-0 improves at epoch 5 — 04:50 SAST

- LR `3e-5` job `1218343` completed its scientifically valid step-2705
  artifact. Tsn/Xho/Zul span micro-F1 is
  `0.42706244150282474/0.3864130146785575/0.43215273440979923`; exact mean
  `0.41520939686372715` matches `checkpoint-2705` `best_metric` exactly.
  This replaces epoch-4 best `0.34181699883258126` and keeps the run active
  under the frozen rule, while still trailing LR `8e-5`'s provisional best
  `0.5335878171485439`.
- Independent checks found exactly `192` rows, `64` per language, all raw
  predictions nonempty, unique raw-prediction counts `38/56/36`, and one
  constant stored metric per language. Debug/state SHA-256 values are
  `d5c7a903dee7a4ef4bdf417a03f8f34c89e893b932814a2d77f44dd0895b3afd`
  and `fb0d7b8a1670667ed2cdf47ebc38cb23e4ebad4364eb0a93ff76889448971338`.
- Job `1218343` resumed healthy epoch-6 training near step `2734/8115`.
  Job `1218344` reached epoch-5 step `2705`, completed exact 10,760-row
  health-only loss `0.34560057345819295`, and entered generation; no
  step-2705 F1 artifact existed yet. LR `1.5e-4` `1218345` remains
  resource-pending. Targeted fault scans remained empty.
- Owned state remains two running plus one pending A100-40GB jobs, no
  A100-80GB/L40S overlap. Quota home `33.7%`, scratch `37.3%`; Kombuys
  read-only and untouched. Base `16/16` trusted, frozen Multilingual winners
  `0/8`, held-out `0`, Mono not started, Sheet E/F/G blank. LR-1's epoch-5
  artifact is tentatively due around `05:20--05:40`; terminal and LR-2 start
  times remain output/scheduler dependent.

## NER LR-1 improves again at epoch 4 — 04:20 SAST

- LR `8e-5` job `1218344` completed its scientifically valid step-2164
  artifact. Tsn/Xho/Zul span micro-F1 is
  `0.5317041800642588/0.5205521316150878/0.5485071397662851`; exact mean
  `0.5335878171485439` matches `checkpoint-2164` `best_metric` exactly.
  This replaces its epoch-3 best `0.4557537887462699` and continues its
  provisional lead over LR `3e-5` best `0.34181699883258126`. The sustained
  improvement explains why the frozen early-stopping rule correctly keeps
  the trial active; no cross-LR winner is frozen.
- Independent checks found exactly `192` rows, `64` per language, all raw
  predictions nonempty, unique raw-prediction counts `40/56/38`, and one
  constant stored metric per language. Debug/state SHA-256 values are
  `73e758f05b0e5bec3789d72a6f6a298e3ac91145733fc18ca9ea669f795d9c76`
  and `7c3910ab5e0236db8bd32d4b4b203da9226e989ce2f3871b4cf19180690d1bd2`.
- Job `1218344` resumed healthy epoch-5 training near step `2547/8115`.
  Job `1218343` reached epoch-5 step `2705`, completed exact 10,760-row
  health-only loss `0.4703660702616752`, and entered generation; no
  step-2705 F1 artifact existed yet. LR `1.5e-4` `1218345` remains
  resource-pending. Targeted fault scans remained empty.
- Owned state remains two running plus one pending A100-40GB jobs, no
  A100-80GB/L40S overlap. Quota home `33.7%`, scratch `37.2%`; Kombuys
  read-only and untouched. Base `16/16` trusted, frozen Multilingual winners
  `0/8`, held-out `0`, Mono not started, Sheet E/F/G blank. LR-0's epoch-5
  artifact is tentatively due around `04:50--05:10`; terminal and LR-2 start
  times remain output/scheduler dependent.

## NER LR-0 materially improves at epoch 4 — 03:50 SAST

- LR `3e-5` job `1218343` completed its scientifically valid step-2164
  artifact. Tsn/Xho/Zul span micro-F1 is
  `0.332561906965931/0.32743742550650556/0.3654516640253073`; exact mean
  `0.34181699883258126` matches `checkpoint-2164` `best_metric` exactly.
  This materially improves its epoch-3 best `0.22690209232181432`, so
  continuing under the frozen patience rule is correct. It remains below LR
  `8e-5`'s provisional best `0.4557537887462699`.
- Independent checks found exactly `192` rows, `64` per language, all raw
  predictions nonempty, unique raw-prediction counts `41/57/36`, and one
  constant stored metric per language. Debug/state SHA-256 values are
  `7860c239fa39b0356f8f67eafb199b09c63aa277efe71778bfdaa77041f69fe8`
  and `e68b452e04df15ffb72647a47b81e95b735dd9f9132c3bc29d101b61f6bce15e`.
- Job `1218343` resumed healthy epoch-5 training near step `2478/8115`.
  Job `1218344` completed exact 10,760-row epoch-4 health-only validation
  loss `0.35859204260390043` and remained in generation; its step-2164 F1
  artifact was not yet present. LR `1.5e-4` `1218345` remains
  resource-pending. Targeted fault scans remained empty.
- Owned state remains two running plus one pending A100-40GB jobs, no
  A100-80GB/L40S overlap. Quota home `33.7%`, scratch `37.2%`; Kombuys
  read-only and untouched. Base `16/16` trusted, frozen Multilingual winners
  `0/8`, held-out `0`, Mono not started, Sheet E/F/G blank. LR-1's epoch-4
  artifact is tentatively due around `04:10--04:30`; terminal times remain
  output-dependent and no NER winner is frozen.

## Both active NER trials in epoch-4 callbacks — 03:20 SAST

- Jobs `1218343/1218344` remain healthy on `srvrocgpu010`, each on one
  A100-40GB `gpu:ampere`. Both reached epoch-4 step `2164`. LR `3e-5`
  completed its exact 10,760-row health-only validation loss
  `0.6467588289962826`; LR `8e-5` reached the same boundary with causal token
  accuracy `0.9624315708875656`, while its validation loss was still running.
- Neither step-2164 generation artifact existed, so retained bests and the
  provisional ordering are unchanged. Targeted logs showed no traceback,
  CUDA OOM/error, NCCL, non-finite, provenance, or coverage fault. LR
  `1.5e-4` job `1218345` remains resource-pending with no released slot.
- The next F1 artifacts are tentatively due around `03:50--04:20`, but the
  frozen early-stopping behavior and terminal time remain output-dependent.
  Owned state remains two running plus one pending A100-40GB jobs, no
  A100-80GB/L40S overlap. Quota is home `33.7%`, scratch `37.2%`. Kombuys
  remains read-only and untouched. Base `16/16` trusted, frozen Multilingual
  winners `0/8`, held-out `0`, Mono not started, Sheet E/F/G blank.

## NER LR-1 improves again at epoch 3 — 02:50 SAST

- LR `8e-5` job `1218344` completed its scientifically valid step-1623
  artifact. Tsn/Xho/Zul span micro-F1 is
  `0.4305254016499734/0.45789998575290636/0.4788359788359298`; exact mean
  `0.4557537887462699` matches `checkpoint-1623` `best_metric` exactly.
  `checkpoint-1623` replaces epoch-2 `checkpoint-1082` as its retained best
  and extends its provisional lead over LR `3e-5` best
  `0.22690209232181432`. No cross-LR winner is frozen.
- Independent checks found exactly `192` rows, `64` per language, all raw
  predictions nonempty, unique raw-prediction counts `38/57/35`, and one
  constant stored metric per language. Debug/state SHA-256 values are
  `3a7c37e15b5f6004533329ff5a817466ce85a162269a5d10b6cfd184e8bdfde5`
  and `46884992cfd637c1b060a67f824acc9fe424f9c9ff0c2c723cdcce990591a23e`.
- Job `1218344` resumed healthy epoch-4 training near step `1687/8115`.
  Job `1218343` reached epoch-4 step `2164` and entered its next callback.
  LR `1.5e-4` `1218345` remains resource-pending, so no slot has yet been
  released. Targeted fault scans remained empty.
- Owned state remains two running plus one pending A100-40GB jobs, no
  A100-80GB/L40S overlap. Quota is home `33.7%`, scratch `37.2%`. Kombuys
  remains read-only and untouched. Base is trusted `16/16`, frozen
  Multilingual winners `0/8`, held-out `0`, Monolingual not started, and
  Sheet E/F/G blank. Terminal times remain output-dependent; NER cannot be
  frozen before all three LRs terminate under the preregistered rules.

## NER LR-0 epoch 3 gives a marginal retained improvement — 02:20 SAST

- LR `3e-5` job `1218343` completed its scientifically valid step-1623
  generation artifact. Tsn/Xho/Zul span micro-F1 is
  `0.21940016211830168/0.2188571906633832/0.2424489241837581`; exact mean
  `0.22690209232181432` matches `checkpoint-1623` `best_metric` exactly.
  This is a marginal `0.00037506808298442` improvement over epoch 2 and
  `checkpoint-1623` is now the retained within-LR best. It remains well below
  LR `8e-5`'s provisional epoch-2 mean `0.34910683094139167`.
- Independent checks found exactly `192` rows, `64` per language, all raw
  predictions nonempty, unique prediction counts `47/60/41`, and one
  constant stored metric per language. Debug/state SHA-256 values are
  `95b59051c50209f72811cc703efeb80256c0b4819d8d8fd5204d4475a12c0519`
  and `5e6e5d5ec55cdf7b5ae359edfee0e3e5f0646677da1fe4a7de7e8528e4f497a9`.
- The improvement is below the frozen early-stopping threshold `0.001`, so
  the run continued under the preregistered patience rule rather than being
  manually stopped. At `02:20`, job `1218343` was healthy near step
  `1660/8115`. Job `1218344` reached epoch-3 step `1623`, completed its
  health-only validation loss `0.3914661166393181`, and entered generation;
  no step-1623 F1 artifact existed yet. LR `1.5e-4` `1218345` remains
  resource-pending. Targeted fault scans were empty.
- Owned state remains two running plus one pending A100-40GB jobs, no
  A100-80GB/L40S overlap. Quota is home `33.7%`, scratch `37.3%`. Kombuys
  remains read-only and untouched. Base is trusted `16/16`, frozen
  Multilingual winners `0/8`, held-out `0`, Monolingual not started, and
  Sheet E/F/G blank. LR-1's epoch-3 artifact is tentatively due near
  `02:50--03:10`; terminal and LR-2 start times remain output/scheduler
  dependent.

## NER LR-1 improves strongly at epoch 2 — 01:50 SAST

- LR `8e-5` job `1218344` completed its scientifically valid step-1082
  generation artifact. Full-grid per-language span micro-F1 is Tsn
  `0.34223399918462466`, Xho `0.3469783855314902`, and Zul
  `0.35810810810806015`; exact arithmetic mean `0.34910683094139167`
  matches `checkpoint-1082/trainer_state.json` `best_metric` exactly.
  `checkpoint-1082` replaced its epoch-1 `checkpoint-541` mean
  `0.15912753902811083` as the retained within-LR best. It also provisionally
  exceeds LR `3e-5`'s epoch-2 best `0.2265270242388299`, but cross-LR
  selection remains unfrozen.
- Independent checks found exactly `192` rows, `64` per Tsn/Xho/Zul, all raw
  predictions nonempty, unique raw-prediction counts `39/55/36`, and one
  constant stored metric per language. Debug/state SHA-256 values are
  `753d0d91bccbbe4120b61d3ac955bfd64f85d1fa576244e2a6f237f5281c9922`
  and `e2cfaff095a48e21d83cbb11f4259ec44ef6606e48218549fc6d66b2775bfdb5`.
- Job `1218344` resumed healthy epoch-3 training near step `1428/8115`.
  LR `3e-5` job `1218343` reached epoch-3 step `1623` and completed its
  exact 10,760-row health-only validation loss `0.9596275811744889`; its
  terminal generation artifact was still pending. LR `1.5e-4` `1218345`
  remains resource-pending. No targeted fault marker appeared.
- The next terminal LR-0/LR-1 F1 artifacts are tentatively due around
  `02:20` and `02:40--03:00`, output-dependent. Owned state remains two
  running plus one pending A100-40GB jobs, with no A100-80GB/L40S overlap.
  HEX quota is home `33.7%`, scratch `37.2%`; Kombuys remains read-only and
  untouched. Trusted base is `16/16`, frozen Multilingual winners `0/8`,
  held-out adapter tests `0`, Monolingual not started, and Sheet E/F/G blank.

## NER LR-0 improves at epoch 2 — 01:20 SAST

- LR `3e-5` job `1218343` completed its scientifically valid step-1082
  generation artifact. Full-grid per-language span micro-F1 is Tsn
  `0.24126268320175404`, Xho `0.20350841614272025`, and Zul
  `0.23480997337201542`; exact arithmetic mean `0.2265270242388299` matches
  `checkpoint-1082/trainer_state.json` `best_metric` exactly. This improves
  its own epoch-1 mean `0.12351486799421617`, and `checkpoint-1082` replaced
  `checkpoint-541` as the retained within-LR best. It is provisional only;
  cross-LR selection is not frozen.
- Independent checks found exactly `192` debug rows, `64` per language, all
  raw predictions nonempty. Tsn/Xho/Zul unique raw-prediction counts are
  `41/56/36`, and every stored per-language metric is constant across its
  64 debug rows. Debug/state SHA-256 values are
  `5700d025b808244de67e75238b0e3b5f0b79bb704f1bb6fa0bbb236f9be41da3`
  and `14cab0168ea2f0cedf4545842d5f59ff40e01ed41feb7734e370d12b85468502`.
- Job `1218343` resumed healthy epoch-3 training near step `1400/8115`.
  LR `8e-5` job `1218344` completed its exact 10,760-row epoch-2 health-only
  validation loss `0.45399808635498956` and remained in its generation
  callback; no step-1082 F1 artifact existed yet. LR `1.5e-4` job `1218345`
  remains resource-pending. Targeted fault scans remained empty.
- Owned state is two running plus one pending A100-40GB jobs, no
  A100-80GB/L40S overlap. HEX quota remains home `33.7%`, scratch `37.2%`.
  Kombuys remains read-only and untouched. Trusted base is `16/16`, frozen
  Multilingual winners `0/8`, held-out adapter tests `0`, Monolingual not
  started, and Sheet E/F/G blank. LR-1's next F1 artifact is tentatively due
  around `01:45--02:00`; NER cannot freeze before every LR terminates.

## Both running NER trials reached epoch-2 validation — 00:50 SAST

- LR `3e-5` job `1218343` and LR `8e-5` job `1218344` remain healthy on
  `srvrocgpu010`, each using one required A100-40GB `gpu:ampere`. Both reached
  step `1082` and entered their epoch-2 validation callbacks. LR-0 completed
  exact 10,760-row validation loss `1.221553873395388`; LR-1 reached the same
  boundary with causal training token accuracy `0.8979236841201782`, while
  its loss callback was still running. Loss is health evidence only; NER
  selection remains the preregistered mean per-language span micro-F1.
- No step-1082 generation artifact was complete yet, so epoch-1 provisional
  ordering is unchanged and no winner is frozen. Targeted logs showed no
  traceback, CUDA OOM/error, NCCL, non-finite, manifest, or coverage fault.
  The next LR-0/LR-1 F1 artifacts are tentatively due around
  `01:20--01:45`, output-dependent.
- LR `1.5e-4` job `1218345` remains resource-pending. Owned state remains two
  running plus one pending A100-40GB jobs, with no A100-80GB/L40S overlap.
  HEX quota remains home `33.7%`, scratch `37.2%`. Kombuys remains read-only
  and untouched. Trusted base is `16/16`, frozen Multilingual winners `0/8`,
  held-out adapter tests `0`, Monolingual not started, and Sheet E/F/G blank.

## NER LR-1 epoch-1 metric independently reconciled — 00:22 SAST

- Clean LR `8e-5` job `1218344` produced its first scientifically valid
  shifted-training NER selection artifact at step `541`. The exact full-grid
  per-language span micro-F1 values recorded in the 192-row debug artifact
  are Tsn `0.13289389751217834`, Xho `0.16886087531129043`, and Zul
  `0.17562784426086375`; their arithmetic mean is
  `0.15912753902811083`, exactly matching `checkpoint-541` `best_metric`.
  This provisionally exceeds LR `3e-5` job `1218343`'s epoch-1 mean
  `0.12351486799421617`, but no cross-LR or within-LR winner is frozen.
- Independent artifact checks found exactly `192` debug rows, `64` for each
  language, all raw predictions nonempty. Debug/state SHA-256 values are
  `1b6c90553acb82160a4af4e12cd51ef6c8c7c8ad5760a40b2a0fd32120f21647`
  and `ae6e68cb045b56ea99c768a78d00497c5a8520e3c097ea458a0c7800a1f91c9a`.
  The retained checkpoint path is
  `/scratch/lmbanr001/masters/sallm/checkpoints/pure_gdn_adapter_hpo_r2/ner/lr_1-causalshift1/checkpoint-541`.
- At `00:22`, job `1218344` had resumed healthy epoch-2 training near step
  `634/8115`; job `1218343` reached the epoch-2 boundary at step `1082` and
  entered its next validation callback. Batch CPU accounting remained live
  and targeted logs showed no traceback, CUDA OOM/error, NCCL, non-finite,
  provenance, or coverage failure. The next LR-0/LR-1 metric artifacts are
  tentatively due around `01:00` and `01:15--01:30`, respectively, but
  terminal time remains output-dependent.
- LR `1.5e-4` job `1218345` remains resource-pending; the controller's current
  `22:32:15` projected start is dynamic and not a scientific ETA. Owned state
  is two running plus one pending A100-40GB `gpu:ampere` jobs, with no owned
  A100-80GB or L40S work. HEX quota is home `33.7%` and scratch `37.2%`.
  Kombuys was not accessed and remains read-only with the RTX 5090 untouched.
- Operational and scientifically trusted base progress remains `16/16`.
  Frozen Multilingual winners remain `0/8`, held-out adapter evaluations `0`,
  and Monolingual training has not started. Sheet column D remains populated;
  E/F/G remain blank. No Sheet write or Hugging Face publication occurred.
  NER must reach terminal validation-only comparison across all three LRs,
  followed by corrected POS and the other invalidated family grids, before
  Monolingual training or any held-out adapter access.
