# Pure-GDN corrected validation-only HPO — 2026-08-10

## First scientifically valid repaired NER metric — 23:50 SAST

- Clean LR `3e-5` job `1218343` completed its epoch-1 corrected generation
  callback and saved exact `192`-row coverage (`64` each Tsn/Xho/Zul). The
  per-language span micro-F1 values are
  `0.12548262548258135/0.095311562526736/0.14975041597333114`; their exact
  arithmetic mean is `0.12351486799421617`, matching
  `checkpoint-541/trainer_state.json` `best_metric` bit-for-bit. This is the
  first scientifically valid NER selection metric after the causal-loss fix,
  but it is provisional within LR-0 and does not freeze the cross-LR winner.
- The 192-row debug artifact SHA-256 is
  `d938dce7d5cf29b626cb80ce81c6f2ad8400808b215cfa90ae1abbccfd849d80`;
  checkpoint state SHA-256 is
  `86dbed28f7b48742a760263c0e4893ecb5bd1ae3031082c94c5dcd36b45ca1e8`.
  LR-0 retained `checkpoint-541` and resumed healthy epoch-2 training near
  step `574/8115`.
- LR `8e-5` job `1218344` completed epoch-1 shifted training at causal token
  accuracy `0.7557807087898254`, loss `2.1272`, finite gradient norm `4.2731`,
  and exact 10,760-row validation loss `1.04413329624332`. It is healthy in
  corrected NER generation at batch size 64; first span-F1 is expected around
  `00:05--00:20 SAST`.
- LR `1.5e-4` job `1218345` remains resource-pending. Owned hardware is two
  A100-40GB `gpu:ampere`; no A100-80GB or L40S job exists. HEX quota is home
  `33.7%`, scratch `37.2%`. Kombuys remains read-only idle and was not
  accessed; RTX 5090 untouched. Base trust `16/16`, frozen Multilingual
  winners `0/8`, held-out tests `0`, Sheet E/F/G blank.

## Repaired causal objective empirically confirmed — 23:20 SAST

- Clean NER LR `3e-5` job `1218343` reached epoch-1 step `541`. Its final
  training log reports causal next-token `mean_token_accuracy=0.7241757959`,
  loss `2.5004`, and finite gradient norm `7.2572`. This is direct runtime
  evidence that the external FLA model now follows the shifted causal training
  path rather than the quarantined current-token reconstruction path.
- The complete 10,760-row epoch-1 loss evaluation is finite and covers exact
  Tsn/Xho/Zul counts `2495/4085/4180`. Per-language losses are
  `1.5848/1.3959/1.6249`; overall validation loss is
  `1.528678082888011`. The job then entered the corrected NER generation
  callback with automatic batch size 64. No primary span micro-F1 artifact
  exists yet, so no NER winner or checkpoint is frozen. First selection metric
  remains tentatively due around `23:30--23:45 SAST`.
- NER LR `8e-5` job `1218344` started at `23:01:02` on a second A100-40GB and
  is healthy near step `338/8115`. Its job-created 691-file execution manifest
  verifies at SHA-256
  `e24806a4bacf835bedeeaf9f4e02500dd32f59e9a9a623942fc8a7a40776027c`
  and records corrected trainer SHA-256 `921327f8...97b1`, canonical pure-GDN,
  label smoothing `0.05`, and `4,649,088` trainable LoRA parameters. Its first
  full metric is tentatively due around `00:00--00:20 SAST`.
- LR `1.5e-4` job `1218345` remains resource-pending with a non-binding
  controller estimate of `22:32:15` on 11 August. Owned hardware is exactly
  two A100-40GB `gpu:ampere`; no A100-80GB or L40S job exists.
- HEX quota is home `33.7%`, scratch `37.2%`. Kombuys was not accessed and
  remains read-only idle with RTX 5090 untouched. Base trust is `16/16`,
  scientifically frozen Multilingual winners `0/8`, held-out tests `0`, and
  Sheet E/F/G remain blank.

## First causal-shift runtime active — 22:50 SAST

- Clean NER LR `3e-5` job `1218343` started at `22:32:15` on
  `srvrocgpu010`, using exactly one A100-40GB `gpu:ampere`. At `22:49` it was
  healthy at approximately step `325/8115` with finite progress near
  `3.02 s/step`; no traceback, CUDA, NCCL, non-finite, provenance, or coverage
  fault marker exists. Epoch-1 training reaches step `541`; the first complete
  validation loss/generation artifact is tentatively expected around
  `23:30--23:50 SAST`, output-dependent.
- Its job-created execution manifest verifies all `691` source/config files at
  SHA-256
  `cadacc9f02a3d1f68d0c7f34599d3c2db979559291a78d918631594d2f6b61ff`.
  It records the corrected trainer SHA-256
  `921327f8dbde191cd645e4beaa581c0541a1f0b41f450b9dd3081d7b5dda97b1`,
  Transformers `4.57.3`, TRL `0.26.2`, and FLA `0.5.1`. Runtime logs confirm
  the exact canonical `GatedDeltaNetForCausalLM`, label smoothing `0.05`,
  assistant-only loss, and `4,649,088` trainable LoRA parameters.
- LR `8e-5` job `1218344` remains resource-pending with dynamic projected
  start `02:53:11 SAST` on 11 August; LR `1.5e-4` job `1218345` remains
  priority-pending without ETA. Only `1218343` owns a GPU. There is no owned
  A100-80GB or L40S work.
- HEX quota is home `33.7%`, scratch `37.2%`. Kombuys was not accessed and
  remains at its last read-only idle state with RTX 5090 untouched. Base trust
  is `16/16`, scientifically frozen Multilingual winners `0/8`, held-out
  adapter tests `0`, and Sheet E/F/G remain blank.

## Causal-shift correction deployed; clean NER grid submitted — 19:56 SAST

- The prospective correction protocol was frozen before implementation at
  `sallm_memory/notes/2026-08-10-pure-gdn-causal-label-shift-correction-preregistration.md`,
  SHA-256
  `29ddcaa029197b84dd8b31425a04f596be3b86aa4383b2ec9be2cc86a5b7fc13`.
  It preserves label smoothing `0.05` and every existing recipe/selection
  rule; only the missing external causal-LM registration is corrected.
- The new regression test exercised Transformers `Trainer.compute_loss` with
  a dummy class named `GatedDeltaNetForCausalLM`. Before the correction it
  failed with actual unshifted loss `7.9213` versus shifted reference
  `0.3213`. Registering `gated_deltanet -> GatedDeltaNetForCausalLM` in the
  shared trainer makes the inherited Transformers/TRL loss path select
  `shift_labels=True`. The focused test and Ruff pass; the complete suite is
  `113 passed`. Trainer/test SHA-256 values are
  `921327f8dbde191cd645e4beaa581c0541a1f0b41f450b9dd3081d7b5dda97b1`
  and `54dd93c27a99720bcafb9ef221c247d5d223befac432155a0ed060bf4af4223a`.
- POS jobs `1217737/1217738/1217739` were intentionally cancelled at
  `19:53:09` after `04:47:13/03:53:19/03:51:34`. All existing logs,
  checkpoints, manifests, and epoch-1 validation artifacts remain untouched.
  The jobs are scientifically unusable because of the wrong training
  objective, so further epochs had no selection value.
- The new read-only runtime snapshot is
  `/home/lmbanr001/masters/sallm_snapshots/pure-gdn-hpo-causalshift-20260810-921327f8`.
  Its complete `691`-file source/config manifest verifies at SHA-256
  `821d9b613916c0d68441d6e91f3c9be997d4ee17af3bb68d21bea11d6ea7aff6`.
  Launcher SHA-256 remains
  `20dd9439b329d5ce2811e0913620e4f4cda0d889fce3f07630b79ccf93d0fbbe`.
- Clean validation-only NER replacements from the canonical base were
  submitted with new `-causalshift1` output paths: LR `3e-5` job `1218343`,
  LR `8e-5` job `1218344`, and LR `1.5e-4` job `1218345`. All request exactly
  one A100-40GB `gpu:ampere`, account/partition/QOS
  `nlpgroup/a100/nlpgroup`, 24 hours, and eight CPUs. At submission all three
  were pending. At `19:55:58`, Slurm projected LR-0 `1218343` at
  `02:53:11 SAST` on 11 August due resources; LR-1/LR-2 remained
  priority-pending without start estimates. These estimates are dynamic.
- HEX quota is home `33.6%`, scratch `37.1%`. There is no owned A100-80GB or
  L40S job. Kombuys was not accessed and remains at its last verified
  read-only idle state with RTX 5090 untouched. Base trust is `16/16`,
  scientifically frozen Multilingual winners `0/8`, held-out adapter tests
  `0`, and Sheet E/F/G remain blank.

## Critical training-objective quarantine — 19:24 SAST

- The low corrected NER/POS metrics are not currently evidence of a pure-GDN
  architecture limit. The immutable shared launcher forces
  `label_smoothing_factor=0.05` for every validation family. Execution
  manifests confirm that these jobs use Transformers `4.57.3`, TRL `0.26.2`,
  FLA `0.5.1`, and the exact external class
  `GatedDeltaNetForCausalLM` through PEFT.
- In Transformers `4.57.3`, `Trainer.compute_loss` removes `labels` before
  model forward whenever label smoothing is enabled. It passes
  `shift_labels=True` to `LabelSmoother` only when the unwrapped class name is
  present in `MODEL_FOR_CAUSAL_LM_MAPPING_NAMES`. FLA's
  `GatedDeltaNetForCausalLM` is absent from that mapping, so the actual jobs
  compare position-*t* logits with the position-*t* label instead of the
  causal position-*t+1* label. The model therefore optimizes an unintended
  current-token reconstruction/copy objective on assistant tokens.
- The corrected NER generation and POS closed-label evaluators remain valid
  measurements of the resulting adapters: they use causal continuation or
  explicitly shifted continuation scoring. Their low values are consequently
  compatible with the training-objective defect and cannot ratify a recipe or
  characterize the base architecture. Falling training loss is also not
  reassuring because it is the misaligned loss.
- This finding prospectively quarantines NER jobs `1217444/1217445/1217446`,
  withdraws the previously frozen NER winner, and quarantines active POS jobs
  `1217737/1217738/1217739`. Scientifically frozen Multilingual winners return
  to `0/8`; held-out adapter tests remain `0`. No Monolingual run is active or
  authorized. All stored artifacts remain provenance and must not be deleted
  or overwritten.
- At `19:21:10`, the three POS jobs were still healthy on `srvrocgpu010`, one
  A100-40GB `gpu:ampere` each. Their current constrained callbacks were
  `1050/1800`, `350/1800`, and `300/1800`; no A100-80GB or L40S job was owned.
  HEX quota was home `33.6%`, scratch `37.1%`. Kombuys was not accessed in
  this pass; its last verified state remains read-only idle with the RTX 5090
  untouched. Sheet E/F/G remain blank.
- Before any replacement HPO, preregister the implementation correction,
  force causal shifting for label-smoothed external causal-LM classes in the
  shared trainer, add a regression test that distinguishes shifted from
  unshifted loss, deploy a new immutable snapshot, and rerun NER/POS
  validation-only without consulting held-out results.

## All POS epoch-1 metrics available — 18:52 SAST

- All three corrected POS epoch-1 validation artifacts now cover exactly
  1,800 rows and the preregistered 12 Tsn/Xho/Zul by P1--P4 cells under
  `closed_label_tuple_mean_logprob_v1`. Primary arithmetic-mean token
  accuracies are LR-0 `0.10956738155126096`, LR-1
  `0.23358808107576032`, and LR-2 `0.23815571879370404`. Independent means
  from the stored cell correct/total pairs are `0.10956738155126099`,
  `0.23358808107576035`, and `0.238155718793704`, respectively.
- LR-2 job `1217739` is the provisional leader only. Its epoch-1 artifact
  SHA-256 is `4eda17d51b11def85a207acf3a4cd4a08faf9ff0e1d944f7b75be73ca67a65c7`;
  `checkpoint-283/trainer_state.json` SHA-256 is
  `f25164ace4ac1f39fb42637972206e400a27f7f672fbf402a2de7204610efaea`.
  The state records the same best metric and retained checkpoint. LR-1
  artifact/state SHA-256 values are
  `51e12eac34d771603a815b55e215ccb923e7b6d93e3930484fadd4f416f2daf8`
  and `dcf5b84871b511270e131e5dfd06d235601afe2a7a09da629c9b235e2fada3e5`.
- All three jobs remain healthy on `srvrocgpu010`. LR-0 `1217737` reached
  `700/1800` in epoch-2 constrained validation; LR-1/LR-2 `1217738/1217739`
  resumed training after saving checkpoint 283. No cross-epoch or cross-LR
  POS winner is frozen; patience-2 stopping still requires later complete
  validation callbacks.
- Owned hardware remains exactly three A100-40GB `gpu:ampere` jobs with no
  A100-80GB/L40S overlap. HEX quota is home `33.6%`, scratch `37.1%`.
  Kombuys remains read-only and idle: RTX 5090 `10 MiB/0%`, RTX 3080 Ti
  `1 MiB/0%`, scratch `62%`, only the existing Tailscale tmux. No held-out or
  Sheet access occurred; base trust is `16/16`, frozen winners `1/8`,
  held-out tests `0`, and Sheet E/F/G remain blank.

## POS callbacks advancing — 18:20 SAST

- POS LR-1 `1217738` and LR-2 `1217739` remain healthy at `1550/1800` and
  `1500/1800` in epoch-1 constrained validation. Their first primary metrics
  are expected around `18:40--18:45 SAST`.
- LR-0 `1217737` completed epoch-2 teacher-forced loss evaluation at
  `12.60320095486111` and entered its second constrained callback, now
  `250/1800`. Current throughput places its second token-accuracy artifact
  near `20:15--20:25 SAST`.
- No targeted fault, held-out access, or Sheet access occurred. Owned hardware
  remains exactly three A100-40GB jobs; no A100-80GB/L40S. HEX quota is home
  `33.6%`, scratch `37.0%`; latest Kombuys read-only state remains idle. Base
  trust `16/16`, frozen winners `1/8`, held-out `0`, Sheet E/F/G blank.

## First trusted POS metric — 17:49 SAST

- POS LR-0 `1217737` completed epoch-1 constrained validation at `17:42:10`.
  The artifact covers exactly `1,800` rows and all 12 Tsn/Xho/Zul by P1--P4
  cells under `closed_label_tuple_mean_logprob_v1`. Its primary arithmetic
  mean token accuracy is `0.10956738155126096`; independent recomputation from
  the 12 stored correct/total pairs gives `0.10956738155126099` (floating-point
  equivalent). Cell accuracies range `0.09533073929961089` to
  `0.13386487620646245`.
- `checkpoint-283/trainer_state.json` records `best_metric=0.10956738155126096`
  and `best_model_checkpoint=.../checkpoint-283`. The validation artifact
  SHA-256 is `e39dbac83b74687061462b597dd9c60657ca77ce02759d3e4ef3f7c9c6895773`;
  trainer-state SHA-256 is
  `6d17c62be182b5d902b9c33c322f81525f80f973f90694ef2688ee9923ecf5ea`.
  LR-0 resumed epoch-2 training normally.
- LR-1 `1217738` and LR-2 `1217739` remain healthy in their first constrained
  callbacks at `1150/1800` and `1100/1800`. First metrics are expected around
  `18:40--18:50 SAST`. No cross-LR winner can be frozen yet.
- All three owned jobs remain on A100-40GB `gpu:ampere`; no A100-80GB/L40S,
  held-out, or Sheet access occurred. HEX quota is home `33.6%`, scratch
  `37.0%`. Kombuys remains read-only idle (RTX 5090 `10 MiB/0%`, RTX 3080 Ti
  `1 MiB/0%`, scratch `61%`). Base trust is `16/16`, frozen winners `1/8`,
  held-out tests `0`, and Sheet E/F/G remain blank.

## POS constrained callback progress — 17:19 SAST

- All three POS jobs remain healthy in epoch-1 constrained tuple validation:
  LR-0 `1217737` is `1450/1800`, LR-1 `1217738` is `750/1800`, and LR-2
  `1217739` is `700/1800`. No primary token-accuracy metric or checkpoint has
  been emitted yet, and targeted fault scans remain empty.
- Current throughput places the first complete metrics near `17:45`, `18:35`,
  and `18:40 SAST`, respectively. Terminal ETAs remain output-dependent under
  frozen patience-2 early stopping.
- Owned hardware remains exactly three A100-40GB `gpu:ampere` jobs; no
  A100-80GB/L40S or held-out work. HEX quota is home `33.6%`, scratch `37.0%`.
  Latest Kombuys read-only state remains idle. Base trust is `16/16`, frozen
  Multilingual winners `1/8`, held-out tests `0`, and Sheet E/F/G remain blank.

## Full corrected POS grid in constrained validation — 16:49 SAST

- All three artifact-preserving POS retries are running on `srvrocgpu010`:
  `1217737` LR `3e-5` since `15:05:56`, `1217738` LR `8e-5` since
  `15:59:50`, and `1217739` LR `1.5e-4` since `16:01:35`. Each owns one
  A100-40GB `gpu:ampere`; this is exactly the three-job cap.
- Every trial passed immutable-manifest verification, canonical pure-GDN
  loading, Hydra composition, complete 2,259-row train preparation, and exact
  1,800-row prompt-expanded validation preparation. Their execution-manifest
  SHA-256 values are LR-0
  `a33ee8637a893aee365f02ab96ee4720f3beeb65f08434cb6cb1982c62ca1316`,
  LR-1 `019358491d5d77b7cef9c08fd62ec9901435ddd0eae11764dcc00997680f5e28`,
  and LR-2 `3e6106f053693d1a6468d117faf51d668b28b7daf618bbd1f61d35a9430c1280`.
- All three completed epoch-1 teacher-forced loss evaluation and entered the
  intended `closed_label_tuple_mean_logprob_v1` constrained token-accuracy
  callback. LR-0 validation loss is `11.04969970703125`; its constrained
  callback reached `1100/1800`. LR-1 and LR-2 reached `350/1800`. There is no
  traceback, OOM, NCCL, manifest mismatch, coverage failure, or contract fault.
- Current throughput suggests first epoch selection metrics near `17:40 SAST`
  for LR-0 and `18:35--18:50` for LR-1/LR-2. Terminal ETAs remain
  output-dependent because early stopping patience 2 only advances after each
  complete 1,800-row callback.
- No A100-80GB/L40S or held-out work exists. HEX quota is home `33.6%`,
  scratch `37.0%`. Kombuys remains read-only idle at RTX 5090 `10 MiB/0%`,
  RTX 3080 Ti `1 MiB/0%`, scratch `61%`, only existing Tailscale tmux.
  Corrected base is `16/16`, frozen Multilingual winners `1/8`, held-out
  adapter tests `0`, and Sheet E/F/G remain blank.

## NER frozen; POS sparse-config correction and clean retries — 15:07 SAST

- NER LR-2 `1217446` completed `0:0` at `14:42:33 SAST` in `03:50:04`.
  Its execution manifest reverified, no targeted runtime/provenance/coverage
  fault exists, and it retained `checkpoint-541` at corrected mean
  per-language span micro-F1 `0.0`. The epoch-3 debug artifact has exact
  Tsn/Xho/Zul coverage `64/64/64`; all `192/192` sampled raw predictions are
  empty and all per-language F1 values are `0.0`.
- LR-2 SHA-256 values: execution manifest
  `c212b9b8fea653d52c033cfa6b91d81e3770b734ce65a567a7f6fd0d5695248e`,
  epoch-3 debug
  `d16e5d40bdab1c8f824d0da647287f2198d57a34ee4fd75ea1fc9722e80782dd`,
  final adapter config
  `f936ec26df69cb0a0af007a116ceea6487d89fb53c6e8ca346dda4cdc8a1289e`,
  and weights
  `b211b7e97d710e42c9846421789fd12048a871d842238cdeaa88ebe569e5d629`.
- All three NER learning rates and retained epoch-1 checkpoints tie at `0.0`.
  The preregistered exact-tie rules therefore freeze LR `3e-5`, job `1217444`,
  checkpoint 541. The complete validation-only selection record is
  `sallm_memory/artifacts/2026-08-10-pure-gdn-ner-validation-selection.json`,
  SHA-256 `dfbbae17d7a5f79bed113e1d0f64e298decd691503aa65797383fce45071d199`.
  Frozen Multilingual winners are now `1/8`; held-out adapter evaluations
  remain `0`.
- POS LR-0/LR-1 jobs `1217704/1217705` started immediately after NER released
  capacity but failed before model or data access in `00:00:45/00:00:39`.
  Both hit the same Hydra composition error: sparse POS training config has no
  `adam_beta2`, while the shared launcher used a strict override. They created
  no metric, checkpoint, or adapter and remain preserved unchanged.
- Root correction is confined to the shared launcher: the five optional
  training fields now use Hydra `++`, and a validated `SALLM_ATTEMPT_TAG`
  gives retries new output/log/run paths rather than overwriting failed
  artifacts. `bash -n`, all 21 family/trial dry-runs, invalid-tag rejection,
  and direct POS Hydra composition pass. Launcher SHA-256 is
  `20dd9439b329d5ce2811e0913620e4f4cda0d889fce3f07630b79ccf93d0fbbe`.
- Final read-only snapshot is
  `/home/lmbanr001/masters/sallm_snapshots/pure-gdn-hpo-posretryfix-20260810-20dd9439`.
  Its 691-file deployment manifest verifies at SHA-256
  `8bb57c68f9a3f81a6d255bee5593076f8c5b2df0904ff362c9e9e53d15ecb2c3`.
  The earlier unused intermediate snapshot `pure-gdn-hpo-poshydrafix-20260810-ad2d85c5`
  remains preserved and executed no experiment.
- Clean unchanged POS validation retries use new `-retry1` artifact paths:
  `1217737` LR `3e-5`, `1217738` LR `8e-5`, and `1217739` LR `1.5e-4`.
  LR-0 `1217737` started at `15:05:56`, passed Hydra composition, prepared all
  2,259 train and 1,800 prompt-expanded validation rows, and entered healthy
  training. Its 691-file execution manifest reverified at SHA-256
  `a33ee8637a893aee365f02ab96ee4720f3beeb65f08434cb6cb1982c62ca1316`.
  LR-1/LR-2 remain resource/priority pending. These are the only
  three owned jobs, all A100-40GB `gpu:ampere:1`; no A100-80GB/L40S work.
- HEX quota is home `33.6%`, scratch `37.0%`. Kombuys remains read-only idle:
  RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `61%`, only existing
  Tailscale tmux. Corrected base remains `16/16`, held-out adapter tests `0`,
  and Sheet E/F/G remain blank.

## NER LR-2 epoch-3 generation — 14:31 SAST

- NER LR-2 `1217446` remains healthy in the corrected full-generation stage
  after step `1623`. Its complete 10,760-row epoch-3 loss pass is
  `13.713622139289033`; Tsn, Xho, and Zulu automatic generation batch-size
  probes all resolved to 64 at `14:02:54`, `14:12:13`, and `14:27:15`.
- The epoch-3 192-row debug artifact, primary mean per-language span micro-F1,
  early-stop decision, and terminal adapter are not yet saved, so no NER
  winner is frozen. Conditional terminal ETA is `14:45--15:00 SAST`.
- POS `1217704/1217705` remain pending. These plus `1217446` are the only
  owned jobs and all request A100-40GB `gpu:ampere:1`; no A100-80GB/L40S or
  held-out work exists. HEX quota is home `33.6%`, scratch `37.0%`. Kombuys
  remains read-only and idle at RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`,
  scratch `61%`; Sheet E/F/G remain blank.

## NER LR-2 epoch-3 callback — 14:02 SAST

- NER LR-2 `1217446` reached step `1623/8115` and is healthy inside its
  epoch-3 full validation callback. The loss pass completed Tsn and Xho and
  is advancing through Zulu; there is no traceback, OOM, manifest mismatch,
  coverage failure, or terminal selection artifact yet. Epochs 1--2 remain
  corrected mean per-language span micro-F1 `0.0`.
- No cross-LR NER winner can be frozen until this callback and terminal save
  complete. Conditional terminal ETA is approximately `14:40--15:00 SAST`.
  POS `1217704/1217705` remain resource/priority pending, and the third POS
  trial is not submitted while the three-owned-job cap remains occupied.
- Owned state is exactly one running plus two pending jobs, all A100-40GB
  `gpu:ampere:1` under `nlpgroup/a100`; no A100-80GB or L40S work exists.
  HEX quota is home `33.6%`, scratch `37.0%`. Kombuys remains read-only and
  idle: RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `61%`, only
  the existing Tailscale tmux. Base trust is `16/16`, Multilingual winners
  `0/8`, held-out adapter tests `0`, and Sheet E/F/G remain blank.

## NER user-facing status check — 13:43 SAST

- NER LR-2 `1217446` remains healthy on `srvrocgpu010` after `02:50:30` and
  is training toward step 1,623 / the epoch-3 validation callback. Its epoch-2
  corrected debug artifact was saved at step 1,082, resolving the earlier
  artifact-visibility audit item. Epochs 1--2 remain mean per-language span
  micro-F1 `0.0`; diagnostic generations are empty or repetitive assistant
  markers rather than parseable `LABEL: entity` spans.
- LR-0/LR-1 `1217444/1217445` remain completed at `0.0`. LR-2 can still
  improve at epoch 3, but the two zero-score epochs make that unlikely;
  conditional terminal ETA remains `14:35--14:50 SAST`. If all three tie,
  the frozen tie-break selects LR `3e-5`, job `1217444`, checkpoint 541.
- POS `1217704/1217705` remain pending. Owned state is one running plus two
  pending A100-40GB jobs; HEX quota is home `33.6%`, scratch `37.0%`. No
  held-out access or Sheet write occurred.

## NER LR-0/1 terminal; POS LR-0/1 submitted — 13:35 SAST

- NER LR-0 job `1217444` completed `0:0` at `13:23:01 SAST` in `03:54:28`;
  LR-1 job `1217445` completed `0:0` at `13:21:54` in `03:49:07`. Both stopped
  after epoch 3 under the frozen patience-2 rule, loaded retained
  `checkpoint-541`, and saved `final_adapter`. No runtime/provenance/coverage
  fault marker exists.
- Both retained validation-only best metrics are mean per-language span
  micro-F1 `0.0`, so the within-LR exact tie correctly retains the earlier
  epoch-1 checkpoint. Their epoch-3 debug artifacts each contain exact
  Tsn/Xho/Zul coverage `64/64/64`, all `192/192` raw outputs nonempty, only
  `2/2/2` unique predictions by language, and per-language F1 `0.0`.
- LR-0 terminal SHA-256 values: execution manifest
  `f74086fc8d43d0c93e1523e92d3c0cc5fec30cd42399fc9a387a0a8e5ff01d97`,
  epoch-3 debug
  `2748222730cd7fbedbe5df148da1ae5d65506838e80dd02440fdb456aeeeccd7`,
  adapter config
  `e395e69cee36cdc166d139399d5efa82579deb5539f7acca148bddc9db28d8a8`,
  and adapter weights
  `fffb3129951eaee3e576ade9977d33f10f897999a06c5d93cf6456b38ad83fb5`.
- LR-1 terminal SHA-256 values: execution manifest
  `7309aba178d96b731330acd2970f676f3bff7f7e2c1f606279c93651ce797b4e`,
  epoch-3 debug
  `36ddb397eccbcf65b16416f9cd5b49d8a408fce4659c3e52e07fedeeafaaf20a`,
  adapter config
  `bdf0e1054e14af12b4ce184c43f5f13befa06d76619f2fb6231f5e36f7f668a1`,
  and adapter weights
  `0057b7fab79cb8f10c59b81fece6aa19f90f7cde1e7df8f32b6bc01d93d1e39f`.
- NER LR-2 `1217446` remains healthy and is projected to finish near
  `14:35--14:50 SAST` if epoch-3 F1 remains non-improving. NER cannot yet be
  frozen across LRs; operational grid progress is two terminal plus one
  running, while scientifically frozen Multilingual winners remain `0/8`.
- After verifying only `1217446` remained owned, no prior POS-r2 target
  directories existed, and immutable dry runs resolved the correct POS
  validation-only family, two independent POS trials were submitted from the
  same read-only snapshot. POS LR `3e-5` is job `1217704`; POS LR `8e-5` is
  job `1217705`. Both request `nlpgroup/a100/nlpgroup`, one `gpu:ampere`, 24
  hours, eight CPUs, immutable source plus the established runtime venv. Both
  are currently priority-pending, so no POS model/data access has begun.
- Owned state is one running plus two pending A100-40GB jobs, with no
  A100-80GB/L40S overlap. HEX quota is home `33.6%`, scratch `37.0%`.
  Kombuys remains read-only and idle at 13:35 (RTX 5090 `10 MiB/0%`, RTX 3080
  Ti `1 MiB/0%`, scratch `61%`). Base trust remains `16/16`; held-out adapter
  tests remain `0`; Sheet E/F/G remain blank.

## NER epoch-3 callbacks and ETA correction — 13:01 SAST

- All three NER jobs remain healthy on `srvrocgpu010` with live batch
  processes and no targeted fault. LR-0 `1217444` and LR-1 `1217445` reached
  step `1623/8115` and entered epoch-3 full metric callbacks. Their
  validation-only losses are `13.51897326614777` and `13.434167707365242`.
  LR-2 `1217446` reached step `1082/8115` and entered its epoch-2 callback
  with loss `13.502328502555763`. Primary F1 for these callbacks is not yet
  artifact-complete; loss is not used to override the preregistered selector.
- The frozen launcher was reread and confirms early-stopping patience is `2`
  with threshold `0.001`, rather than the previously assumed patience of `3`.
  If LR-0/LR-1 epoch-3 F1 remains `0.0`, these callbacks are the second
  consecutive non-improvement and should end both jobs near `13:20--13:30
  SAST`. Under the same condition LR-2 should finish after epoch 3 near
  `14:35--14:50 SAST`. These remain output-dependent estimates.
- No epoch-3 LR-0/LR-1 or epoch-2 LR-2 debug artifacts/checkpoint states exist
  yet, so neither terminal audit nor winner selection is possible. POS remains
  blocked by the three-job cap. Owned GPU state is exactly three A100-40GB
  jobs with no A100-80GB/L40S overlap. HEX quota is home `33.6%`, scratch
  `36.9%`; Kombuys remains read-only and idle at 13:01 (RTX 5090
  `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `61%`). Base trust remains
  `16/16`, Multilingual winners `0/8`, held-out adapter tests `0`, and Sheet
  E/F/G remain blank.

## NER callbacks complete through LR-0/1 epoch 2 — 12:31 SAST

- All three jobs remain healthy and resumed training after their latest full
  callbacks. Current progress is LR-0 `1217444` step `1568/8115`, LR-1
  `1217445` step `1583/8115`, and LR-2 `1217446` step `953/8115`.
  Targeted fault scans are empty and all logs are fresh.
- LR-0/LR-1 both saved `checkpoint-1082`, while their best checkpoint remains
  the earlier `checkpoint-541` at `best_metric=0.0`; equal epoch-2 F1 did not
  replace the earlier checkpoint. The step-1082 trainer-state SHA-256 values
  are LR-0
  `1d163216189d75d45252950e3358fa1a0f5e4201de9f1a844bfd8dcaddd871fa`
  and LR-1
  `4618bc51a887095cfcda47a6febec8d3799bbb2ce7ca00a7af1f9826fba68541`.
- Both epoch-2 debug artifacts have exact `192`-row coverage, every raw
  prediction nonempty, and all per-language F1 values `0.0`. LR-0 Tsn/Xho/Zul
  unique prediction counts are `3/3/2`; LR-1 counts are `2/2/3`. Their
  SHA-256 values are LR-0
  `2a1d971f94a5adc192d833902504291fa1171001da7bc3df45ca68af19323cf0`
  and LR-1
  `fdca738e058e939211be54972fa42eafdd8f0a646c4b8ac53b858b9614aeaeac`.
- LR-2 saved `checkpoint-541`, with state SHA-256
  `587e5bfee55993e5e9573a7e2233c42d05bd053dc89e4a9d473c7adbe8987a06`,
  `best_metric=0.0`, and that checkpoint retained as best. Its exact 192-row
  epoch-1 artifact SHA-256 is
  `064c0e86d8ae614407eb520293bdb5bbf84720a043da14764cab19eccd389c10`;
  all `192/192` predictions are empty and F1 is `0.0`. Because LR-0/LR-1 use
  the identical corrected evaluator and produce nonempty outputs, the LR-2
  emptiness is a high-LR model outcome, not recurrence of the shared terminal-
  EOS implementation defect. It does not authorize retry or recipe change.
- If F1 remains non-improving, the frozen early-stopping rule suggests LR-0/1
  may finish near `15:00 SAST` and LR-2 near `16:20 SAST`; these are conditional
  estimates, not selection decisions. The three-job cap still blocks POS.
  Owned hardware is A100-40GB only with no A100-80GB/L40S overlap. HEX quota
  is home `33.6%`, scratch `36.9%`; Kombuys remains read-only and idle at 12:31
  (RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `61%`). Base trust
  remains `16/16`, Multilingual winners `0/8`, held-out adapter tests `0`, and
  Sheet E/F/G remain blank.

## NER corrected-generation artifact audit — 12:02 SAST

- Jobs `1217444/1217445/1217446` remain `RUNNING` on three A100-40GB GPUs
  after `02:32:21`, `02:28:07`, and `01:08:25`. LR-0/LR-1 remain inside the
  epoch-2 full metric-generation callback after step `1082`; LR-2 remains
  inside its epoch-1 callback after step `541`. Slurm batch CPU time continues
  to track wall time, and targeted fault scans remain empty.
- The completed epoch-1 debug artifacts for LR-0/LR-1 were found and audited.
  Each contains the frozen `192` rows: exactly `64` each for Tsn, Xho, and Zul.
  All `192/192` raw predictions in both artifacts are nonempty, proving the
  corrected generation path no longer reproduces the old terminal-EOS empty
  output failure. However all normalize to no parseable NER spans, giving
  per-language and mean selection F1 `0.0`. This is a valid negative model
  result and does not trigger a prompt, retry, or recipe change.
- LR-0 unique predictions in the 64-row debug samples are Tsn/Xho/Zul
  `9/8/8`; LR-1 unique counts are `2/1/2`. Artifact SHA-256 values are LR-0
  `c48a7628e2ad0e42bbb309910d8d29d64829614beaa0583cf07183a319e1126e`
  and LR-1
  `970e76c7c5dff6f65bd7f8f0ae4fdd1808c3fb0a03d57a83ae31d582f585d395`.
  Original prediction text and every debug record remain preserved.
- No epoch-2 retained checkpoint or LR-2 debug artifact exists yet, so no
  cross-LR selection is possible. Callback completion remains the ETA blocker,
  and POS remains capacity-blocked by the three-job cap. Owned GPU state is
  A100-40GB only; no A100-80GB/L40S overlap exists. HEX quota is home `33.6%`,
  scratch `36.8%`. Kombuys remains read-only and idle at 12:02 (RTX 5090
  `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `61%`). Base trust stays
  `16/16`, Multilingual winners `0/8`, held-out adapter tests `0`, and Sheet
  E/F/G remain blank.

## NER epoch-2 validation and metric wiring audit — 11:32 SAST

- All three NER jobs remain healthy on `srvrocgpu010`. LR `3e-5` job
  `1217444` and LR `8e-5` job `1217445` reached step `1082/8115` and entered
  their second corrected validation callbacks. Their epoch-2 validation-only
  losses are `13.459356485246282` and `13.291437107922862`, respectively.
  LR `1.5e-4` job `1217446` reached step `541/8115` and entered its first
  callback with validation-only loss `13.726716426579925`.
- The retained epoch-1 checkpoints for `1217444/1217445` are both
  `checkpoint-541`; their `trainer_state.json` records `best_metric=0.0` and
  `metric_for_best_model=eval_all_f1`. The state-file SHA-256 values are LR-0
  `7e5ff5b50ca51e5bf139bb0642b341db8bdb67b3c193abe66b8f3476b421183e`
  and LR-1
  `246e6aea75031d8533ef34d30763700fb82e485734428f457d4804e738fcb821`.
  This is provisional validation evidence only; no cross-LR winner is frozen.
- Source wiring was re-audited against the 691-file-equivalent local tree.
  `GenerationEvaluator` computes span micro-F1 independently within each
  language using aggregate TP/FP/FN, then defines `eval/all_f1` as the
  arithmetic mean across language values; `GenerationMetricsCallback` exposes
  the exact `eval_all_f1` underscore alias consumed by Trainer. Thus the
  configured selection field implements the preregistered mean per-language
  span micro-F1 rather than a support-weighted aggregate.
- All three batch processes and logs remain live, with no targeted fault.
  Full metric-generation callback time remains the dominant ETA uncertainty.
  The job cap stays occupied, so POS remains capacity-blocked. Owned hardware
  is exactly three A100-40GB jobs with no A100-80GB/L40S overlap. HEX quota is
  home `33.6%`, scratch `36.8%`; Kombuys remains read-only and idle at 11:32
  (RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `61%`). Base trust
  is `16/16`, Multilingual winners `0/8`, held-out adapter tests `0`, and Sheet
  E/F/G remain blank.

## Full NER grid running — 11:02 SAST

- LR `1.5e-4` NER job `1217446` received the third A100-40GB at
  `10:52:29 SAST`; all three preregistered NER trials now run together on
  `srvrocgpu010`. It verified all `691` immutable source/config hashes,
  loaded the canonical `GatedDeltaNetForCausalLM` checkpoint, and exposed
  exactly `4,649,088/132,076,072` trainable/total parameters. Its execution
  manifest SHA-256 is
  `c212b9b8fea653d52c033cfa6b91d81e3770b734ce65a567a7f6fd0d5695248e`.
- The corresponding verified execution-manifest SHA-256 values are
  `f74086fc8d43d0c93e1523e92d3c0cc5fec30cd42399fc9a387a0a8e5ff01d97`
  for LR `3e-5` job `1217444` and
  `7309aba178d96b731330acd2970f676f3bff7f7e2c1f606279c93651ce797b4e`
  for LR `8e-5` job `1217445`.
- Jobs `1217444/1217445` completed their first corrected validation callbacks
  and resumed training, reaching steps `787/8115` and `766/8115`; `1217446`
  reached `147/8115`. All logs were fresh at `11:01:04`, all three jobs use
  `NVIDIA A100-PCIE-40GB`, and targeted fault scans remained empty.
- Tqdm's training-only remaining projections are approximately `6h15--6h50`
  (roughly `17:15--17:55 SAST`), but repeated full validation callbacks add
  material overhead and early stopping is output-dependent. Terminal ETA is
  therefore not yet reliable; hard Slurm deadlines are the corresponding
  start times on 11 August plus 24 hours.
- The owned-job cap is exactly three running A100-40GB jobs, so POS remains
  blocked on capacity. There is no A100-80GB or L40S work. HEX quota is home
  `33.6%`, scratch `36.8%`. Kombuys remains read-only and idle: RTX 5090
  `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `61%`, only the existing
  `tailscale-kombuys` tmux session. Base trust remains `16/16`, Multilingual
  winners `0/8`, held-out adapter tests `0`, and Sheet E/F/G remain blank.

## NER first-epoch validation — 10:32 SAST

- NER LR `3e-5` job `1217444` and LR `8e-5` job `1217445` remain `RUNNING`
  on `srvrocgpu010` after `01:03:13` and `00:58:59`. Each completed step
  `541/8115` and its first loss pass over all `10,760` validation rows. LR
  `3e-5` reported per-language losses Tsn `13.0049`, Xho `12.6342`, Zul
  `13.0717`, and overall `12.890100415311338`. LR `8e-5` reported Tsn
  `13.3197`, Xho `13.4696`, Zul `13.5577`, and overall
  `13.469093793564127`. These are validation-only intermediate values, not
  winner selection or held-out evidence.
- Both jobs then entered the corrected full NER validation metric/generation
  stage. No final mean per-language span micro-F1, retained checkpoint, or
  winner exists yet. Slurm and batch processes remain live, with accumulated
  CPU time tracking wall time and no traceback, CUDA OOM/error, NCCL error,
  manifest mismatch, token-contract failure, or coverage failure.
- LR `1.5e-4` job `1217446` remains resource-pending; Slurm's dynamic,
  non-binding start estimate is still `13:55:01 SAST`. Because validation
  callback duration is now the dominant unknown, terminal ETA for the two
  running jobs is not scientifically reliable yet. No POS trial was submitted
  while the three-job cap remains occupied.
- Owned GPU state is two running plus one pending A100-40GB `gpu:ampere` jobs,
  with no A100-80GB/L40S overlap. HEX quota remains home `33.6%`, scratch
  `36.7%`. Kombuys remains read-only and idle at 10:32: RTX 5090
  `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `61%`, only the existing
  `tailscale-kombuys` tmux session. Base trust remains `16/16`, Multilingual
  winners `0/8`, held-out adapter tests `0`, and Sheet E/F/G remain blank.

## NER grid monitoring — 10:03 SAST

- NER LR `3e-5` job `1217444` and LR `8e-5` job `1217445` remain healthy
  on `srvrocgpu010`, each using one A100-40GB `gpu:ampere`. At the check they
  had run for `00:34:28` and `00:30:14`, respectively. Both reached the first
  epoch validation callback after step `541/8115`; logged training losses and
  gradients are finite, and targeted scans found no traceback, CUDA OOM/error,
  NCCL error, manifest mismatch, token-contract failure, or coverage failure.
- Repetitive validation samples (`s s s ...` for `1217444` and
  `outhouth ...` for `1217445`) are retained model outputs. They do not change
  the frozen validation protocol and do not justify a retry or prompt change.
  Job `1217444` completed its five generated diagnostic samples and began the
  all-row/language validation aggregation; its first reported Tsn slice covers
  `2,495` rows with finite average loss `13.0049`. Job `1217445` was producing
  its five diagnostics when checked.
- LR `1.5e-4` job `1217446` remains resource-pending. Slurm's current
  non-binding start estimate is `13:55:01 SAST`; its scientific ETA remains
  unknown until allocation. These are the only three owned jobs, all on the
  required A100-40GB family. No A100-80GB or L40S work exists.
- Training progress projects roughly another `5.5--6.5` hours per running
  trial before validation-callback overhead and possible early stopping, so a
  reliable terminal ETA is not available yet. No POS job was submitted while
  the three-job ownership cap is occupied.
- HEX quota is home `33.6%` and scratch `36.7%`. Kombuys was checked read-only:
  RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `61%`, and only the
  existing `tailscale-kombuys` tmux session. Corrected base remains trusted
  `16/16`; Multilingual winners remain `0/8`; held-out adapter tests remain
  `0`; Sheet base column D is released, while E/F/G remain blank.

## Base gate ratified and HPO released — 09:29 SAST

- User explicitly authorized treating AfriHG Xhosa's single whitespace-only
  output as a valid model miss. It remains scored empty and preserved; there
  is no held-out-driven retry or protocol change. Corrected base is now
  operationally and scientifically trusted `16/16`.
- Canonical Sheet `Pure GatedDeltaNet Results` rows 42--43 were reread, then
  only C/D were replaced with the verified 10 August Xhosa/Zulu AfriHG
  results and full provenance notes. The values are `7.6220 chrF` and
  `5.6752 chrF`; formatting and wrapping are preserved and E/F/G remain
  blank. The hybrid tab was untouched.
- The frozen validation-only launcher passed `bash -n` and all 15 family/LR
  dry-run mappings. Initial NER submissions `1217441`, `1217442`, and
  `1217443` failed before training in `00:00:00/00:00:01`: immutable source
  correctly excluded `.venv`, but manifest creation fell back to system
  Python 3.9, where `datetime.UTC` is unavailable. No execution manifest,
  checkpoint, adapter, validation metric, or selection evidence was created.
- Root cause was corrected once in the shared launcher: manifest creation now
  uses `${SALLM_RUNTIME_REPO:-$repo}/.venv/bin/python`. No scientific
  protocol, data, prompt, recipe, learning rate, seed, metric, or output
  selection rule changed.
- New read-only snapshot:
  `/home/lmbanr001/masters/sallm_snapshots/pure-gdn-hpo-runtimefix-20260810-05a4a619`.
  Launcher SHA-256 is
  `05a4a6197387bde7721c0d633c093f5d9400a29f01ad16b8418c99f54a5c60fe`.
  Its deployment manifest verifies all `691/691` source/config files and has
  SHA-256
  `a234fe873b665acafae3a0eba9fec087d1cb1cf0b6d0027bb1651d583b3c4c38`.
- Unchanged NER validation-only retries are `1217444` (LR `3e-5`), `1217445`
  (LR `8e-5`), and `1217446` (LR `1.5e-4`). All request
  `nlpgroup/a100/nlpgroup`, one `gpu:ampere`, 24 hours, and eight CPUs.
  `1217444` started at `09:28:33` on `srvrocgpu010`, wrote execution-manifest
  SHA-256
  `f74086fc8d43d0c93e1523e92d3c0cc5fec30cd42399fc9a387a0a8e5ff01d97`,
  verified all 691 files, and passed the exact A100-40GB/GDN-kernel startup
  path. `1217445/1217446` remain resource/priority pending. No A100-80GB or
  L40S owned job exists.
- At `09:29:30`, `1217444` additionally verified the canonical
  `GatedDeltaNetForCausalLM` checkpoint in BF16, installed the frozen chat
  template, resized embeddings, and exposed exactly `4,649,088` trainable
  LoRA parameters of `132,076,072` total. It entered dataset preparation with
  no fault marker.
- Multilingual winners remain `0/8`; this is the first corrected
  validation-only grid and no held-out adapter evaluation is authorized yet.
  HEX quota is home `33.5%`, scratch `36.7%`. Latest read-only Kombuys state
  is idle: RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `61%`, only
  the existing `tailscale-kombuys` session.
