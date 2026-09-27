# Cross-architecture Kombuys HPO activation — 2026-08-30

Preregistered at 08:49 SAST before any new cross-architecture validation
result was produced.

## Scientific status

- The user has explicitly activated the previously deferred
  cross-architecture HPO work while pure-GDN continues independently on HEX.
- This is a uniform post-hoc reproduction, not a fully prospective
  architecture comparison: held-out results for the older architectures have
  already been observed. Those results must not influence candidates,
  checkpoints, retries, prompts, ordering, or selection in this workstream.
- The pure-GDN program, its A100-80GB allocation, its freeze state, and Sheet
  E/F/G remain unchanged. No official held-out evaluation is authorized here.
- Each architecture receives a separate frozen protocol and output root. Do
  not assume that pure-GDN target modules or rank ranges transfer.

## First active lane: LLaMA-125M T2X

- Host/GPU: `kombuys`, GPU 0 only, NVIDIA GeForce RTX 5090 32 GB. Keep every
  LLaMA-125M T2X candidate and confirmation on this GPU. GPU 1 remains unused
  until a separate architecture-specific protocol and fit check are frozen.
- Base model:
  `/scratch/alombard/masters/sallm/checkpoints/sallm-llama-125m/final_model`.
  Frozen SHA-256 values:
  - `config.json`: `38c277e5b5258ec80260c6411b537c311b43b53f19a48752454510386c5883ab`
  - `pytorch_model.bin`: `7388b67c8fe73a1bb8f80a97b8b756de4a110e914ac8e1dfcaf0d06697a61273`
  - `tokenizer.json`: `446895905ea9b20c746317eefd0c6a3b097bcbbef71e8e44b0bf9772d664782a`
- Immutable source snapshot:
  `/scratch/alombard/sallm_snapshots/uniform-adapter-hpo-20260811-6aabf717`.
  Deployment manifest SHA-256:
  `da1c456774b789578bc6d25f816e627fa0d6677de928802a89fb5a226557b285`.
- Candidate registry:
  `src/conf/hpo/pure_gdn_enhanced_v1.json`, SHA-256
  `8fdd6ea5f192a1d2ef2267353162b9a640ccd09ae0d889cb5a999645198bb726`.
  Reuse its fixed three Stage-A and eight Stage-B points only as a common
  numeric search budget. This does not transfer pure-GDN module choices.
- LLaMA target modules are frozen to the architecture's existing `q_proj` and
  `v_proj` LoRA path. The candidate supplies rank, alpha, dropout, learning
  rate, and warmup. Alpha remains twice rank.
- Data and metric: T2X Xhosa train/validation only, four epochs, seed 42,
  validation chrF, greater is better, frozen prompt/template and beam-decoding
  settings from `llama_t2x_xho`.
- Order: run Stage-A `a0`, `a1`, `a2`, then Stage-B `b0` through `b7`, all at
  seed 42. After all eleven terminal artifacts and adapter roundtrips verify,
  rank by validation chrF, freeze the top two, and run each at seeds 13 and
  87. Select the final recipe from seeds 13/42/87 only.
- `a0` is the execution canary and a real preregistered candidate, not an
  extra trial. Continue only after its startup manifest, data counts, first
  validation artifact, and GPU isolation verify. Never alter or retry from
  its metric.
- Output root:
  `/scratch/alombard/masters/sallm/checkpoints/adapter_hpo_v3/llama125/t2x`.
  No output existed at preregistration.

## Later architectures

- Mamba-2 and xLSTM may run on Kombuys only after separate target-module,
  rank-range, base-model, runtime, and GPU assignments are frozen before any
  new validation result. Their historical test exposure must remain disclosed.
- The Qwen3Next GDN-attention hybrid is not pure GDN and must remain a
  separately labelled hybrid workstream if activated.

## 09:52 SAST execution update

- LLaMA-125M T2X Stage-A `a0` completed cleanly at all four frozen
  validation boundaries. Validation chrF was `36.22638504340215`,
  `41.095135134876884`, `42.29958392023858`, and `41.89502053184986` at
  steps 483, 966, 1449, and 1932. The within-run rule retains checkpoint
  1449. Each artifact has exactly 64 rows and indices, no empty predictions,
  and 64 unique predictions.
- Exact retained-to-final equality was verified across 124 keys and
  67,929,088 values with maximum difference zero. Checkpoint/final adapter
  SHA-256 values are `b449e367e2f3e7d75b5e1ea78a09a6082c475ae71f41d5d3cf187c32d34ea696`
  and `2bedf1c9d34a57469d354289fc69634e8dc2456fc8d63c1ed351588ad7a5b674`.
  Preserve `a0`; never rerun it.
- Absent-output, no-session, no-process, and GPU-isolation checks passed for
  `a1`. It is now running on Kombuys GPU 0 in tmux
  `sallm-llama125-t2x-a1`. Its immutable execution-manifest SHA-256 is
  `0b77f5498160bb2a336607ff804a90eee3b6e1f11015404d9647200b611dd195`.
  GPU 1 remains idle.

## 11:00 SAST execution update

- LLaMA-125M T2X Stage-A `a1` completed cleanly. Validation chrF was
  `42.475449108280195`, `44.75867788907608`, `45.53530032884151`, and
  `45.25749369948135` at steps 483, 966, 1449, and 1932. Every frozen
  artifact has exact 64-row/index coverage, no empty prediction, and 64
  unique predictions. The within-run rule retains checkpoint 1449.
- Exact retained-to-final equality was verified across 124 keys and
  67,929,088 values. Retained/final adapter SHA-256 values are
  `bcdf60a98cf80366973772ec54fb678993d729df5f1b0254e29f88826c9cb3fa`
  and `0577ec3c67213ca5c689e790f9ae71d41c3f73f5cb75fe363eef0d4e0233fe05`.
  Preserve `a1`; never rerun it.
- The first `a2` launcher omitted `WANDB_MODE=offline` and exited during W&B
  initialization before model/data loading. Preserve its metadata-only root
  and log. The prospective one-line correction is frozen in
  `2026-08-30-cross-architecture-kombuys-a2-wandb-launch-correction.md`,
  SHA-256 `346a023b4983a529e004f9fff1f2521a2f568e2e45194c06cda1da56cc61f398`.
- The one-time corrected `a2` is running in isolated tmux session
  `sallm-llama125-t2x-a2-corrected` on GPU 0. It verified all 694 immutable
  files, loaded exact 3,859/460 train/validation rows, and entered training.
  Its execution-manifest and HPO-trial SHA-256 values are
  `77d2de4be0a74a6befe3493d98c5d71ad8c93c4c1aaa8115eb838aee89194be2`
  and `b04b39e9ba24cdc00bd620a8dc55b894c3199a909d7b5e25384e987a6e85e55f`.
  GPU 1 remains idle.

## 13:30 SAST execution update

- Corrected Stage-A `a2` completed cleanly. Validation chrF was
  `44.58453982372142`, `47.69833691068496`, `48.216235845955254`, and
  `48.52835434271314` at steps 483, 966, 1449, and 1932. Each frozen
  artifact has exact 64-row/index coverage, no empty prediction, and 64
  unique predictions. The within-run rule retains terminal checkpoint 1932.
- Exact retained-to-final equality was verified across 124 keys and
  67,929,088 values. Retained/final SHA-256 values are
  `7694e79bff624d0c0a7db1d830eba3e3a2b79968b02a87abb334d5e9995e85a9`
  and `38241f5d1ab1428e04fdd7a60355cea8d57b1e593eb79ac143d3d0eb7facf77a`.
  Preserve the corrected a2 and never rerun either a2 attempt.
- After absent-output, no-session, no-process, registry, and GPU-isolation
  checks, Stage-B `b0` started serially on GPU 0 in tmux
  `sallm-llama125-t2x-b0`. It verified all 694 immutable files, loaded exact
  3,859/460 train/validation rows, and entered training. Its execution
  manifest and HPO-trial SHA-256 values are
  `ea1a72d9bcc9d7a83731ff5ded94ff669e42596177d209ce0d3047fc561b8c0f`
  and `1e237728d1f4db7e8aa5c525296edca48bac377b7ed7fe822118e1560f3f8f3c`.
  GPU 1 remains idle.

## 14:18 SAST execution update

- Stage-B `b0` completed cleanly. Validation chrF was
  `43.74339218622562`, `45.126837292667624`, `46.47335462047807`, and
  `46.98977851830069` at steps 483, 966, 1449, and 1932. Each frozen
  artifact has exactly 64 rows and indices, no empty predictions, 64 unique
  predictions, and beam 5. The within-run rule retains terminal checkpoint
  1932.
- Exact retained-to-final equality was verified across 124 keys and
  67,522,048 values. Retained/final adapter SHA-256 values are
  `ec57f1a89ae7160ee0919c6caa3806089d83c623521576d329d81b17361d4b12`
  and `22d6fbcd7f126b7779188d9c6e3a60c666c2e4615c6b9b0266cc517b02a21d6a`.
  Preserve b0 and never rerun it.
- After absent-output, no-session, no-process, registry, and GPU-isolation
  checks, unchanged Stage-B `b1` started serially on GPU 0 in tmux
  `sallm-llama125-t2x-b1`. It verified all 694 immutable files, loaded exact
  3,859/460 train/validation rows, and entered training. Its execution
  manifest and HPO-trial SHA-256 values are
  `0c6432f33ca6fd52d51be1512c271278eb2ad49f69baf38490903f2e9b6963a5`
  and `8f1653994446f769a0f6d3a4abc3959e2dd975de92d94e73f1ec28a4edbb469a`.
- The separately frozen Mamba activation on GPU 1 is terminally blocked by
  PEFT `0.18.1` rejecting the required `out_proj` target for Mamba-2. No data
  or task metric was used. The preserved failure is documented in
  `2026-08-30-cross-architecture-kombuys-mamba2-activation-failure.md`; GPU 1
  remains idle and the gate must not be retried.

## 15:12 SAST execution update

- Stage-B `b1` completed cleanly. Validation chrF was
  `39.14468356048833`, `41.87082610042089`, `42.756910256699435`, and
  `42.81002752398118` at steps 483, 966, 1449, and 1932. Every frozen
  artifact has exactly 64 rows and indices 0--63, no empty prediction, and
  beam 5. Step 483 has 63 unique predictions; the remaining artifacts have
  64. The within-run rule retains terminal checkpoint 1932.
- Exact retained-to-final equality was verified across 124 keys and
  68,743,168 values. Retained/final adapter SHA-256 values are
  `4db781e6e301b7ae731cbeb34cba1ba8abc13c688d63a970204b07d591edd8ab`
  and `17ac9c9593e5e9bb20f6b3c2a091884f9a0d6362f50fbbbaa2834e2a69d65e92`.
  The execution-manifest sidecar verifies. Preserve `b1`; never rerun it.
- `b2` remains unlaunched. Another user's two processes currently occupy
  Kombuys GPU 0, and the frozen lane requires the same isolated RTX 5090.
  No process was changed. Release `b2` only after absent-output,
  no-duplicate, registry, process, and GPU-isolation checks all pass.

## 18:00 SAST execution update

- GPU 0 remains unavailable for the frozen LLaMA lane. It has `28,062 MiB`
  allocated, including a `28,044 MiB` ASR process (`PID 1966755`) owned by
  `csikasote`; the process started at `17:45:47 SAST`. No process was changed.
- The required b2 pre-launch absence checks remain clean: no b2 output path,
  log, tmux session, or `alombard` b2 process exists. The only visible
  `alombard` tmux session is the unrelated long-lived Tailscale session.
- Because GPU 0 is not isolated, Stage-B `b2` was not launched. GPU 1 remains
  idle and is outside this frozen LLaMA protocol. No held-out artifact was
  opened or used.

## 31 August 02:18 SAST execution update

- Kombuys GPU 0 became genuinely isolated at `10 MiB/0%`; b2's output, log,
  tmux session, and process were all absent. A single fail-closed preflight
  verified the deployment-manifest sidecar, all `694` immutable source/config
  hashes, all six frozen LLaMA base-model artifact hashes, the tokenizer hash,
  the unchanged registry recipe, and GPU isolation immediately before launch.
- Stage-B `b2` is now running serially on GPU 0 in tmux
  `sallm-llama125-t2x-b2`. Its frozen recipe is seed `42`, LR
  `3.617950373518279e-05`, rank `8`, alpha `16`, dropout
  `0.019731278717517856`, and warmup `0.09582568347454071` with
  `q_proj/v_proj` targets. It loaded exact `3,859/460` train/validation rows
  and entered the `1,932`-step training schedule.
- Execution-manifest and HPO-trial SHA-256 values are
  `651e627bd83ff23887bd30e867fb66dc9d7f270bfd30ccfa68717790736a8abd` and
  `abd9d67657720c371165f4052f2da6d7419dc527d78ca3aa89299003bf80cc58`.
  This post-hoc lane remains isolated from pure-GDN and cannot affect its
  selection, scheduling, held-out boundary, or ETA.

## 31 August 03:18 SAST execution update

- Stage-B `b2` completed cleanly. Validation chrF was
  `34.25156415347831`, `39.492568075259186`, `40.59239731926782`, and
  `39.96438880908078` at steps `483/966/1449/1932`; the frozen within-run rule
  retains checkpoint `1449`. Every artifact has exact 64-row/index coverage
  and beam 5. The step-483 artifact contains one empty prediction, which is
  included as ordinary scored validation evidence; the later three artifacts,
  including the retained and terminal boundaries, contain none. Unique
  prediction counts are `64/63/64/64`.
- Exact retained-to-final equality passed across `124` keys and `67,522,048`
  values. Retained/final adapter SHA-256 values are
  `9c8898e0f7e607f54d93222691c3eb5ee3cdc5f669b6c2ed23405ca6a62ea165` and
  `434be4b41f9a81ddf433c1d8bcd33d04e6aa0674861fba0d22b47cbf6a99ed84`.
  Preserve b2 and never rerun it. The LLaMA seed-42 grid is now `6/11`
  terminal-valid.
- After fresh absent-output, no-session, no-process, registry, 694-file source,
  six-file model, tokenizer, and GPU-isolation checks, unchanged Stage-B `b3`
  started serially on GPU 0 in tmux `sallm-llama125-t2x-b3`. It loaded exact
  `3,859/460` train/validation rows and entered training. Its execution-manifest
  and HPO-trial SHA-256 values are
  `a0d1f5bf5fc1d87acf42d1ec59a5381ababe9f1d630599a6eaca0457ee3e4fda` and
  `45cb3badfa9598f20d9717bbe5f820f1649f38ff8f3ca308ca1d75e738fdac50`.

## 31 August 05:31 SAST execution update

- Stage-B `b3` completed cleanly. Validation chrF is
  `40.69645308307086`, `44.201515387271336`, `44.865800837994094`, and
  `44.89238882495095` at steps `483/966/1449/1932`; the frozen within-run
  rule retains checkpoint `1932`. Every artifact has exact 64-row/index
  coverage, beam 5, zero empty predictions, and 64 unique predictions.
- Exact retained-to-final equality passed across `124` keys and `67,929,088`
  values. Retained/final adapter SHA-256 values are
  `c216a948509ea98f3cef4a1d8386c3716e6902666b2a69bbd1fc3bc1c951e115`
  and `fa7b87bfebf506d54f0b5b1000cce435b1fc0d326c6b967c3e183b39dd35c9c0`.
  Preserve b3 and never rerun it. The LLaMA seed-42 grid is now `7/11`.
- After fresh absent-output, no-session, frozen registry, 694-file source,
  six-file model, tokenizer, and GPU-isolation checks, unchanged Stage-B `b4`
  started serially on GPU 0 in tmux `sallm-llama125-t2x-b4`. It loaded exact
  `3,859/460` train/validation rows and entered training. Its execution-manifest
  and HPO-trial SHA-256 values are
  `27e60edb3b86197584fe2038c3ac4cd14a28005af92e39d0df912090a88378e1`
  and `39463eee346bd932d2a80415352cb3f218525cb51212fe0663841ae67e52f39f`.
  GPU 1 remains idle and pure-GDN is unaffected.

## 31 August 06:27 SAST execution update

- Stage-B `b4` completed cleanly. Validation chrF was
  `41.14859978077697`, `43.53834092412122`, `44.57380920361152`, and
  `44.54892913287072` at steps `483/966/1449/1932`; the frozen within-run
  rule retains checkpoint `1449`. Every artifact has exact 64-row/index
  coverage, beam 5, no empty prediction, and 64 unique predictions.
- Exact retained-to-final equality passed across `124` keys and `67,522,048`
  values. Retained/final adapter SHA-256 values are
  `2f4c1a6844fc902423ccf67a215553b63c9a57a573f431dc5e91691449adec37`
  and `bbb2ccf4a3227c01d31b241c1b192c8c395561d7b05e5887872bcb4aca96c7d3`.
  Preserve b4 and never rerun it. The LLaMA seed-42 grid is now `8/11`.
- After fresh absent-output, no-session, frozen registry, 694-file source,
  six-file model, tokenizer, and GPU-isolation checks, unchanged Stage-B `b5`
  started serially on GPU 0 in tmux `sallm-llama125-t2x-b5`. It loaded exact
  `3,859/460` train/validation rows and entered training. Its execution-manifest
  and HPO-trial SHA-256 values are
  `50f222365ae659da1d65c163b274b31c606f3fb80020cbfc29696849cc7c2633` and
  `a9994e6e13477211338efa04811104372bae54f63ad216640e366f5f8465efe9`.
  GPU 1 remains idle and pure-GDN is unaffected.

## 31 August 07:26 SAST execution update

- Stage-B `b5` completed cleanly. Validation chrF was
  `39.66137939746889`, `42.92873157650323`, `43.29533776349944`, and
  `43.51713717457478` at steps `483/966/1449/1932`; the frozen within-run
  rule retains checkpoint `1932`. Every artifact has exact 64-row/index
  coverage, beam 5, no empty prediction, and 64 unique predictions.
- Exact retained-to-final equality passed across `124` keys and `67,929,088`
  values. Retained/final adapter SHA-256 values are
  `48d43951fe02539e655d4ec2d386b178bb895356972c1d7f5e602305ad1b01fd`
  and `2b6405c6c7f8fb21bfd485f7a5c6acb12bfa7348d8d4988cbc513fb721d2011c`.
  Preserve b5 and never rerun it. The LLaMA seed-42 grid is now `9/11`.
- After fresh absent-output, no-session, frozen registry, 694-file source,
  six-file model, tokenizer, and GPU-isolation checks, unchanged Stage-B `b6`
  started serially on GPU 0 in tmux `sallm-llama125-t2x-b6`. It loaded exact
  `3,859/460` train/validation rows and entered training. Its execution-manifest
  and HPO-trial SHA-256 values are
  `b98c410be75da35610669e218ae289aa8d840b1f50b257aa3086c01cb365fd56` and
  `a50fe26f0445c2dacb398a4fea47aa1934d4b3ad8e559c72960e9603048ad345`.
  GPU 1 remains idle and pure-GDN is unaffected.

## 31 August 08:47 SAST execution update

- Stage-B `b6` completed cleanly. Validation chrF was
  `34.47308361229159`, `38.734704730281706`, `39.75908363480086`, and
  `39.990824045303356` at steps `483/966/1449/1932`; the frozen within-run
  rule retains checkpoint `1932`. Every artifact has exact 64-row/index
  coverage, beam 5, no empty prediction, and 64 unique predictions.
- Exact retained-to-final equality passed across `124` keys and `67,929,088`
  values. Retained/final adapter SHA-256 values are
  `2e9213125e0002689deaa2fb1bc4cfda4d0d6fd1c13744fc487a65ddbc159f70`
  and `0c38ca3eeb68aef158a45fe6f877905ab86adc8e355892fc7c80eb2cfa48abe3`.
  Preserve b6 and never rerun it. The LLaMA seed-42 grid is now `10/11`.
- After fresh absent-output, no-session, frozen registry, 694-file source,
  six-file model, tokenizer, and GPU-isolation checks, unchanged Stage-B `b7`
  started serially on GPU 0 in tmux `sallm-llama125-t2x-b7`. It loaded exact
  `3,859/460` train/validation rows and entered training. Its execution-manifest
  and HPO-trial SHA-256 values are
  `e143827b41d93c7be5d7c31257c06ae1677f56b7757839f5d2ae524fec4e697c` and
  `db5e7dcf29982d8078045c6de89ffc5b0162a0f276664193ca221ed622f121bd`.
  GPU 1 remains idle and pure-GDN is unaffected.

## 31 August 09:32 SAST execution update

- Stage-B `b7` completed cleanly. Validation chrF was
  `44.73004698015957`, `48.33975914732357`, `49.69324873709872`, and
  `49.89574145828771` at steps `483/966/1449/1932`; the frozen within-run
  rule retains checkpoint `1932`. Every artifact has exact 64-row/index
  coverage, beam 5, no empty prediction, and 64 unique predictions.
- Exact retained-to-final equality passed across `124` keys and `68,743,168`
  values. Retained/final adapter SHA-256 values are
  `0b539e0a32cb90e1dc0f89966aa76d565df2beeca37554a6a06f6d0edee51b4b`
  and `855937a56dbcfc575bcf00d31c992dcd482d5882ad78cc91048ded5c780d5374`.
  Preserve b7 and never rerun it. All eleven seed-42 candidates are now
  terminal-valid.
- The complete grid was ranked once by validation chrF and frozen in
  `seed42_validation_ranking.json`, SHA-256
  `deb603fa4e9a40d2b77b326360831cd6baa865c83bdae100afd407582c009fc8`.
  Ranking is b7, a2, b0, a1, b3, b4, b5, b1, a0, b2, b6. The frozen top two
  are b7 (`49.89574145828771`) and a2 (`48.52835434271314`), with
  confirmation order b7 seed 13, b7 seed 87, a2 seed 13, a2 seed 87.
- The first confirmation dry-run initially passed the frozen provenance checks
  but used `stage_b`; the immutable wrapper rejected that operator invocation
  before creating an output, log, session, or scientific payload because
  confirmations require stage `confirm`. The corrected existing path required
  no implementation change.
- B7 seed 13 then passed absent-output, frozen-ranking, source/model/tokenizer/
  registry, and GPU-isolation checks and started in tmux
  `sallm-llama125-t2x-b7-s13`. It loaded exact `3,859/460`
  train/validation rows. Execution-manifest and HPO-trial SHA-256 values are
  `44a9584669519b84f1cba10c4458347d86ed1d2aa91fa938b864363c5e236d37` and
  `bb78549de426bf641e110da1ed2e3e653a3a1aac5902e69affbaad1b8f1d3dec`.
  GPU 1 remains idle and pure-GDN is unaffected.

## 31 August 10:30 SAST execution update

- B7 seed 13 completed cleanly. Validation chrF was
  `44.062487091935004`, `48.44505602089041`, `49.17059584733388`, and
  `50.25952890593673` at steps `483/966/1449/1932`; the frozen rule retains
  checkpoint `1932`. Every artifact has exact 64-row/index coverage, beam 5,
  no empty prediction, and 64 unique predictions.
- Exact retained-to-final equality passed across `124` keys and `68,743,168`
  values. Retained/final adapter SHA-256 values are
  `30648a7705a2442385b3852d3ac140ba0eb524d81c3b2a386b906d20e479d25e`
  and `3a50e7ec293f615d49d9074afa5365bd7ef6ccf98f2a99744ea955bcb3682230`.
  Preserve b7 seed 13 and never rerun it. Confirmation progress is `1/4`.
- The frozen next run, b7 seed 87, remains absent and unlaunched because a
  different user's process owns Kombuys GPU 0. GPU 1 is not part of this
  frozen serial LLaMA protocol. No foreign process was changed; the next run
  will start only after fresh isolation and no-duplicate checks. Pure-GDN is
  unaffected.

## 31 August 11:32 SAST execution update

- GPU 0 became genuinely isolated. Frozen b7 seed 87 passed absent
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

## 31 August 12:31 SAST execution update

- B7 seed 87 completed cleanly. Validation chrF was `45.43849823497772`,
  `48.33804044559621`, `49.52933730149589`, and `49.28995414995973` at
  steps `483/966/1449/1932`; the frozen rule retains checkpoint `1449`.
  Every artifact has exact 64-row/index coverage, beam 5, no empty prediction,
  and 64 unique predictions.
- Exact retained-to-final equality passed across `124` keys and `68,743,168`
  values. Retained/final adapter SHA-256 values are
  `00689266dbcca3578401b24062052846c77d3e76e3736e9901fb083b7f76f077`
  and `c1140d00df262c02b9d971017095e1f018142a4da4cd370971f22c53f066f2e3`.
  Preserve b7 seed 87 and never rerun it. Confirmation progress is `2/4`.
- With GPU 0 isolated, frozen a2 seed 13 passed all absence, ranking,
  immutable-source/model/tokenizer/registry, prior-manifest, dry-run, and GPU
  checks before starting in tmux `sallm-llama125-t2x-a2-s13`. It verified all
  694 immutable files, loaded exact `3,859/460` rows, and entered its
  `1,932`-step schedule. Execution-manifest and HPO-trial SHA-256 values are
  `a6d62d02bf033329b95498fed5cfa1e42c15da831699987223d13011e2bfd81b`
  and `3b70a1c796974b28f690f935ae466b47cff2b5c5ffe7235af5456377a5827761`.
  GPU 1 remains outside the frozen serial protocol and pure-GDN is unaffected.

## 31 August 13:32 SAST execution update

- A2 seed 13 completed cleanly. Validation chrF was `43.48261563532691`,
  `47.025476607703375`, `47.85185955501886`, and `48.084019151368054` at
  steps `483/966/1449/1932`; the frozen rule retains checkpoint `1932`.
  Every artifact has exact 64-row/index coverage, beam 5, no empty prediction,
  and 64 unique predictions.
- Exact retained-to-final equality passed across `124` keys and `67,929,088`
  values. Retained/final adapter SHA-256 values are
  `72bd9279f7f8e9e98de77eb3a6e17c47527a3315f3d37f7067b632c1409b0633`
  and `8a8647ecb9553ef3d61122face69a0e413ba3546fdc6979562b34f3684afbce4`.
  Preserve a2 seed 13 and never rerun it. Confirmation progress is `3/4`.
- Frozen a2 seed 87 passed all absence, ranking, immutable provenance,
  prior-manifest, dry-run, and GPU-isolation checks before starting in tmux
  `sallm-llama125-t2x-a2-s87`. It verified 694 immutable files, loaded exact
  `3,859/460` rows, and entered its `1,932`-step schedule. Execution-manifest
  and HPO-trial SHA-256 values are
  `6be8bfd8f79e9d66f5223b20fbe58b1628d5e2329cde1de4c031d3d9c5d368d3`
  and `2781dcbd1d77e938fbc4a94c67bfc376652030a5ddbcccfe6d28290460ff9f78`.
  GPU 1 remains outside the frozen serial protocol and pure-GDN is unaffected.

## 31 August 14:31 SAST execution update

- A2 seed 87 completed cleanly. Validation chrF was `44.83365297535967`,
  `47.02310995608528`, `49.100440742466795`, and `48.859427683935806` at
  steps `483/966/1449/1932`; the frozen rule retains checkpoint `1449`.
  Every artifact has exact 64-row/index coverage, beam 5, no empty prediction,
  and 64 unique predictions. Exact retained-to-final equality passed across
  `124` keys and `67,929,088` values. Retained/final adapter SHA-256 values
  are `28eee496aab7c334150a731c1dcabcaca2badc6c11bb8a8693a972692cf06369`
  and `72445d0f62b0003aee309324bdf083d8e2d07712abc58c26e9d3b9f3e460b74e`.
  Preserve a2 seed 87 and never rerun it. Confirmations are `4/4`.
- The immutable HPO utility applied the frozen arithmetic three-seed mean.
  B7 scores `50.25952890593673/49.89574145828771/49.52933730149589`
  for seeds `13/42/87`, mean `49.89486922190678`, sample SD
  `0.3650965836545177`. A2 scores
  `48.084019151368054/48.52835434271314/49.100440742466795`, mean
  `48.57093807884933`, sample SD `0.5095470965969793`. B7 wins; its seed-42
  checkpoint `1932` is the representative adapter. The read-only
  confirmation-ranking artifact SHA-256 is
  `6e2245989d95c9dfbe4cb7d3ffcab32b1ae7e1175a18e20ca98616a1f30329a2`.
  The LLaMA T2X post-hoc lane is frozen and no official held-out metric was
  accessed.
