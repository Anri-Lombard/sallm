# Pure-GDN downstream validation work — 2026-08-05

## Preregistered first lane

- Base artifact is canonical job `1179847`'s hash-verified `final_model`, not
  the two-step readiness canary.
- First task is MasakhaNews joint English/Xhosa classification using the
  existing five-prompt cycle, official train split for fitting, and official
  validation split for checkpoint and recipe selection. Held-out test remains
  untouched.
- The historical `gdn_news_all_hpo_r1` config is excluded because it targets
  the Qwen3Next GDN–Attention Hybrid.
- Pure-GDN LoRA targets are fixed before metrics are observed to all recurrent
  mixing projections (`q_proj`, `k_proj`, `v_proj`, `a_proj`, `b_proj`,
  `g_proj`, `o_proj`) and all MLP projections (`gate_proj`, `up_proj`,
  `down_proj`). This architecture-complete set avoids importing an
  attention-derived `q_proj`/`v_proj` smoke recipe.
- LoRA is fixed at rank 16, alpha 32, dropout 0.05. The existing four-point
  learning-rate grid is `3e-5`, `8e-5`, `1.5e-4`, `3e-4`; all other optimizer,
  batching, prompt, and early-stopping settings are shared. Selection metric
  is validation `eval_classification/all_f1` only.
- Trials run serially on Kombuys RTX 3080 Ti GPU 1. RTX 5090 remains out of
  scope. The smallest initial gate is trial 0; proceed through the frozen grid
  only after confirming real nonzero loss/gradients, validation callbacks,
  adapter persistence, and no traceback/OOM.

## Implementation

- Added `src/conf/finetune/gdn_pure_news_all_hpo_r1.yaml` and reused
  `scripts/run_news_hpo_trial_kombuys.sh` with a `pure_gdn` architecture case.
- Hydra composition passed locally against the canonical artifact path.
- Canonical HEX artifact was staged locally and all six SHA256 values exactly
  match the final manifest. Transfer to Kombuys was not performed because the
  private-checkpoint export requires explicit approval; the run stays on HEX.
- Initial HEX job `1183133` failed before model or data loading because
  `hub.base_model_id: null` violated the typed string schema. No training or
  metric occurred; the field now resolves to the same canonical model path.
- Replacement job `1183134` runs trial 0 (`3e-5`) on one A100-80GB. It loaded
  pure `GatedDeltaNetForCausalLM`, attached `4,649,088` trainable LoRA
  parameters, prepared 4,341 train and 3,095 validation rows, and entered the
  TileLang backward path. Early variable-shape compilation amortized from
  about 93 seconds on step 1 to about 2 seconds for steps 5--10; no traceback
  or CUDA OOM observed.
- Trial 0 reached steady state near `1.73 s/step` by step 61, giving roughly
  2.6--3 hours including validation. Remaining frozen-grid trials are a strict
  same-family serial chain: `1183160` (`8e-5`) afterok `1183134`, `1183161`
  (`1.5e-4`) afterok `1183160`, and `1183162` (`3e-4`) afterok `1183161`.
  No held-out evaluation is queued; it waits for validation macro-F1 selection.
- Job `1183134` completed epoch-1 trainer validation with finite
  `eval_loss=13.105342222839257` over 3,095 rows; training loss fell from about
  2.48 at epoch 0.50 to 1.92 at epoch 0.99 with finite gradient norms near
  3.6--4.8. The separate classification callback evaluates up to 256 examples
  per language by label-choice scoring and is taking about 14 minutes per
  language. Macro-F1 is not yet available and no held-out metric was touched.
  This revises the full-grid worst-case ETA upward to roughly 36--40 hours if
  all ten epochs run; early stopping after the minimum useful epochs could
  still finish the grid in roughly 11--15 hours.
- W&B run `ldtnnu63` records identical validation-only classification results
  at epochs 1 and 2: macro-F1 `0.10811573554007083`, English F1
  `0.043573575434573804`, and Xhosa F1 `0.17265789564556785`. Epoch-3
  classification scoring is in progress. With patience 2 and threshold 0.001,
  another non-improvement should stop trial `1183134` after epoch 3, which
  would reduce the serial-grid ETA to about 7--9 hours from the 00:04 SAST
  check. These are validation-selection metrics, not held-out results.
- At the 2026-08-06 01:06 SAST monitoring pass, trial `1183134` had completed
  `0:0` at 00:09:36 after `01:47:47`. Epoch 3 did not improve the
  preregistered validation macro-F1: W&B summary `ldtnnu63` remained
  `0.10811573554007083` (English `0.043573575434573804`, Xhosa
  `0.17265789564556785`). The job saved a PEFT adapter containing
  `adapter_model.safetensors` (`152,880,608` bytes); its log contains no
  traceback, CUDA OOM, or NCCL timeout.
- Strict `afterok` successor `1183160` (LR `8e-5`, W&B `d46ssqps`) released at
  00:09:36 on one `ampere80` GPU and was healthy through epoch-2 validation:
  trainer validation loss improved from `12.95649043315832` to
  `12.631155025747173`; epoch-2 classification scoring was in progress. Jobs
  `1183161` and `1183162` remained dependency-pending. If all remaining trials
  stop after three epochs like trial 0, the grid should finish around
  05:30--06:30 SAST; the conservative all-ten-epoch bound remains roughly
  34--38 hours from this check. Held-out test remains untouched.
- HEX quota was `/scratch` `91/300 GB` (`30.6%`). The only owned active or
  schedulable GPU jobs were `1183160--1183162`, all explicitly
  `ampere80`; no owned A100-40GB or L40S job was active/schedulable. Kombuys
  showed both RTX 5090 and RTX 3080 Ti idle; the completed readiness artifacts
  remain diagnostic-only and no checkpoint transfer or new Kombuys work was
  started.
- At the 2026-08-06 02:06 SAST pass, LR `8e-5` trial `1183160` had completed
  `0:0` at 01:45:44 after `01:36:08`, saved its adapter, and contained no
  traceback, CUDA OOM, or NCCL error. Its epoch-3 trainer validation loss was
  `12.653601133380452`; W&B `d46ssqps` finalized validation macro-F1
  `0.06244142349681912` (English `0.051428746252218084`, Xhosa
  `0.07345410074142017`). It therefore does not displace trial `1183134`'s
  current validation leader `0.10811573554007083`.
- Trial `1183161` (LR `1.5e-4`) cleared its dependency but is
  priority-pending; Slurm's current estimated start is 2026-08-08 00:54:36
  SAST. All four A100-80GB GPUs on `srvrocgpu011` are occupied by another
  user's jobs `1183614--1183617`; they were not modified. Trial `1183162`
  remains dependency-pending behind `1183161`. There is no owned active
  A100-40GB or L40S job. If the scheduler estimate holds and both remaining
  trials early-stop after three epochs, grid completion moves to roughly
  04:30--05:30 SAST on 2026-08-08; queue estimates are not guarantees.
- HEX scratch remains `91/300 GB` (`30.6%`). Kombuys RTX 3080 Ti is idle and
  mechanically ready, but the canonical private-checkpoint transfer remains
  outside the existing authorization; no duplicate job or transfer was
  started. This queue wait is the only current blocker. Held-out test remains
  untouched.
- At 2026-08-06 04:08 SAST, Slurm refined `1183161`'s pending reason from
  generic `Priority` to `AssocGrpGRES`, identifying the A100-80GB association
  GPU limit as the explicit blocker. Its estimated start remained
  2026-08-08 00:54:36 SAST; `1183162` remained dependency-pending. Other-user
  jobs `1183614--1183617` still occupied the four A100-80GB GPUs, while the
  owned queue contained no A100-40GB or L40S work. HEX scratch remained
  `91/300 GB` (`30.6%`), and both Kombuys GPUs remained idle. No action was
  taken and held-out test remained untouched.
- By the 2026-08-06 06:10 SAST check, `1183161` had started on
  `srvrocgpu011` using one `ampere80` GPU. It reached epoch `0.77` at about
  `1.73 s/step` with finite loss and gradients, the expected `4,649,088`
  trainable LoRA parameters, and no traceback/OOM/NCCL fault; W&B run is
  `y4go6aeh`. Because the frozen job was already running, no hardware-family
  migration was performed. `1183162` remains its sole dependency successor.
- Read-only capacity inspection showed idle A100-40GB capacity on
  `srvrocgpu009` and available capacity across the L40S partition. The 125M
  single-GPU BF16 LoRA workload should fit either 40GB A100 or 48GB L40S, but
  moving a running trial would waste progress and is unnecessary now.
- An authenticated read-only Hugging Face inventory query found no repository
  matching `sallm-pure-gdn` or `pure-gdn` under `anrilombard`. The visible
  `sallm-gated-deltanet-*` repositories are the older GDN--Attention Hybrid
  lineage and must not be substituted for the canonical pure-GDN artifact.
  Uploading the hash-verified `final_model` to a new private repository remains
  pending explicit approval of the external write and repository identity.

## Base-model evaluation gate — preregistered 2026-08-06

- User changed the hardware rule: an A100-80GB lane or an A100-40GB lane is
  permitted, but owned work must not use both memory classes concurrently.
  L40S is not selected for this gate. Current LR `1.5e-4` fine-tuning job
  `1183161` is allowed to finish alone on A100-80GB; pending LR `3e-4`
  successor `1183162` was cancelled before it started so no further
  fine-tuning can auto-release.
- Base-model evaluation precedes adapter evaluation. The canonical
  hash-verified `final_model` is evaluated directly with `peft_adapter=null`,
  `merge_lora=false`, BF16, and zero-shot prompts. This is a one-time held-out
  characterization, not a selection loop; no metric may change prompts,
  checkpoints, or recipes.
- Reuse the established 16-lane architecture-comparison matrix exactly:
  MasakhaNews all, MasakhaNER all, MasakhaPOS all, SIB-200 all,
  InjongoIntent all, SA-general (AfriMGSM/AfriMMLU/AfriXNLI), Belebele
  Afrikaans/English/Sesotho/siSwati/Setswana/Xitsonga/Xhosa/Zulu, T2X Xhosa,
  and AfriHG all. The existing five-prompt task packs and official test splits
  remain unchanged. Generation tasks are zero-shot.
- Reuse `scripts/launch_llama_252m_base_eval_array.sh` through environment
  overrides rather than creating a duplicate evaluator. All 16 dry-run routes
  resolved to the canonical local checkpoint and unique
  `pure_gdn_125m_base_0shot_*_r1` result directories. Schedule the array
  `afterok:1183161` on A100-40GB GRES `amperemk`, maximum concurrency 3; this
  prevents overlap with the current A100-80GB job.
- Publish the six canonical files to the new private repository
  `anrilombard/sallm-pure-gdn-125m` from an A100-40GB compute job after
  `1183161`. Publication must add the scientific model card and exact SHA-256
  manifest, then freshly download and verify all six files before success.
- Execution update: Slurm rejected the attempted `afterok:1183161`
  A100-40GB evaluation array at submission with `AssocGrpGRES`; this
  association counts the future GRES while the A100-80GB job is active. No
  evaluation job ID was created and no held-out example was touched. Submit
  the already dry-run-verified array only after `1183161` exits, when the
  owned A100-80GB lane is empty.
- At 07:10 SAST, `1183161` was healthy in epoch-1 classification scoring after
  finite trainer validation loss `13.113272730714863`; no traceback/OOM/NCCL
  fault was present. Based on the two completed trials, expected completion is
  approximately 08:20--08:40 SAST, after which the A100-40GB base array can be
  submitted.
- The private Hub publisher is implemented, syntax-checked, and staged on HEX,
  but the combined upload/evaluation submission was rejected at the external
  write approval boundary. No Hugging Face repository or commit was created.
  Publication remains blocked pending explicit approval of the exact private
  destination `anrilombard/sallm-pure-gdn-125m`, the six canonical model files,
  `README.md`, and `artifact_sha256.json`.

## Base-model evaluation execution — 2026-08-06

- Final allowed fine-tuning job `1183161` completed `0:0` at 08:25:29 SAST
  after `01:35:51`. Its validation-only macro-F1 was
  `0.06244142349681912` (English `0.051428746252218084`, Xhosa
  `0.07345410074142017`); the adapter was saved and the log contained no
  traceback, CUDA OOM, or NCCL failure. Job `1183162` remains cancelled and
  no more fine-tuning or held-out adapter evaluation is authorized.
- The apparent A100-40GB `AssocGrpGRES` blocker was a GRES-label mismatch:
  the `nlpgroup` QOS grants `gres/gpu:ampere=4`, not `amperemk`. Switching
  only the request label to `gpu:ampere:1` selects idle A100-40GB node
  `srvrocgpu010` without changing the scientific protocol or mixing GPU
  memory classes.
- Provenance-only job `1184499` allocated the correct A100-40GB but failed
  before model/data access because Slurm started it in `$HOME` and the array
  launcher uses a repo-relative child script. Adding
  `--chdir=$HOME/masters/sallm` corrected the submission environment.
- Jobs `1184500` and `1184502` then failed before inference, and `1184501`
  was cancelled before the same failure, because `_prepare_include_paths`
  hard-coded the repo `.venv` task root while `lm_eval` was imported from the
  scratch environment. The shared harness now derives the task root from the
  imported `lm_eval.tasks` package; the focused evaluator test passes and the
  one-file fix is synced to HEX.
- Corrected base-evaluation jobs `1184503` (MasakhaNews), `1184504`
  (MasakhaNER), and `1184505` (MasakhaPOS) are running concurrently on three
  `gpu:ampere:1` A100-40GB allocations on `srvrocgpu010`. All passed the old
  failure point and entered real request execution using the canonical local
  pure `GatedDeltaNetForCausalLM` (`attn=null`), BF16, zero-shot,
  `peft_adapter=None`, and `merge_lora=False`. No result is interpreted or
  used to change the frozen protocol.
- HEX scratch is `92/300 GB` (`30.8%`). Kombuys RTX 5090 and RTX 3080 Ti are
  idle and untouched; `/scratch` is 61% used. The current tranche should take
  several hours, but a defensible full-suite ETA awaits steady request rates.
  Hub publication remains the only approval blocker.

## Full evaluation mission and result publication — 2026-08-06

- User authorized the complete evaluation sequence: finish the frozen 16-lane
  base matrix first, then preregister and execute validation-selected
  Monolingual, Multilingual, and General variants. Held-out metrics must never
  select recipes, checkpoints, prompts, or reruns.
- Base lane 0, job `1184503_0`, completed `0:0` in `00:08:32`. The verified
  MasakhaNews zero-shot held-out F1 headlines are English `0.2727` (P2; mean
  `0.2498`; range `0.2220–0.2727`) and Xhosa `0.3098` (P5; mean `0.2761`;
  range `0.2540–0.3098`). These best-of-five prompt values are descriptive.
  Summary SHA-256 is
  `0300b678248cfdcaac6a24efc20737d57cf436439f99ca2e43dcb4d4533b6046`.
- Jobs `1184504_1` (MasakhaNER) and `1184505_2` (MasakhaPOS) remain healthy
  on A100-40GB at about 3% and 5%, with scheduler-side request ETAs of roughly
  6.5 and 3 hours. Freed slot 0 was filled by job `1184508_3` for SIB-200;
  the owned concurrency cap remains three `gpu:ampere` jobs.
- The canonical results workbook now has a separate visible tab
  `Pure GatedDeltaNet Results` (`sheetId=202608060`). It duplicates the
  established row/layout conventions but all inherited hybrid metrics and
  provenance were cleared. Only completed lane-0 Base cells `C2:D3` and
  `I2:J3` are populated, with exact model/job/protocol/path/hash notes. The
  historical `GatedDeltaNet Results` hybrid tab is untouched.
- The hourly monitor now runs through 2026-08-31 and must update the pure-GDN
  tab incrementally from verified artifacts. Mono/Multi/General fine-tuning
  stays blocked until all base lanes complete and their protocol is
  preregistered.
- Base lane 3, job `1184508_3`, completed `0:0` in `00:04:11`. Its SIB-200
  summary SHA-256 is
  `0aabb81c1a2319bee42b3e159859d8b05f9824e67cc762b3955c5c2edf2249da`.
  Zero-shot best-prompt F1 is Zulu `0.1642`, Xhosa `0.1942`, Southern Sotho
  `0.3303`, Northern Sotho `0.2445`, English `0.3522`, and Afrikaans `0.4576`.
  These are descriptive best-of-five held-out values and did not change the
  frozen protocol. Rows 10--15 of `Pure GatedDeltaNet Results` now contain the
  exact prompt means/ranges, job, artifact paths, and hash.
- At the 09:06 SAST pass, jobs `1184504_1` (MasakhaNER), `1184505_2`
  (MasakhaPOS), and `1184513_4` (InjongoIntent) were all running on
  `srvrocgpu010`, each using one A100-40GB `gpu:ampere`. NER and POS had run
  for about 24 minutes and were executing `generate_until`; Intent had run
  for about four minutes and entered `loglikelihood`. Targeted logs contained
  no traceback, CUDA OOM, NCCL failure, or runtime error. Scratch was
  `92/300 GB` (`30.8%`).
- Base lane 4, job `1184513_4`, completed `0:0` in `01:04:45`. Its
  InjongoIntent summary SHA-256 is
  `126d38f265220b86d4a583cb361f524547e20fc41c8dad5ff3fd9a28ef3ba9e0`.
  All five frozen prompts tied within each language: English macro-F1
  `0.001290205525708353`; Xhosa, Zulu, and Southern Sotho each
  `0.0012195121951219512`. These held-out values were recorded without
  changing the protocol. `Pure GatedDeltaNet Results!C16:D19,I16:J19` now
  contains exact job, prompt, artifact, model, and hash provenance.
- The freed third A100-40GB slot was filled with frozen lane 5 SA-general job
  `1185142_5`. It was submitted with the canonical local final model, no
  adapter, BF16, zero-shot, and the same launcher/environment overrides as the
  accepted base jobs. It was initially priority-pending while `1184504_1`
  and `1184505_2` continued; no other lane was duplicated.
- Base lane 2, job `1184505_2`, completed `0:0` in `03:06:22`. Its
  MasakhaPOS summary SHA-256 is
  `c9dc80d5d7aa67d77aebc1e9b5e44422b69b3507baca2ed9973a17d417be69f`.
  Xhosa, Zulu, and Tswana flexible-extract token accuracy is exactly `0.0000`
  for all four frozen prompts. This is a verified negative base result, not a
  missing run. `Pure GatedDeltaNet Results!C7:D9,I7:J9` now records the full
  job, metric, protocol, artifact, and hash provenance.
- With POS complete, frozen Belebele Afrikaans lane 6 was submitted as
  `1185270_6`. SA-general `1185142_5` was resource-pending and lane 6 was
  priority-pending; together with active NER `1184504_1`, the owned base gate
  remains capped at three A100-40GB active-or-schedulable lanes.
- Base lane 6, job `1185270_6`, completed `0:0` in `00:01:53`. All five
  frozen Belebele Afrikaans prompts returned accuracy
  `0.2733333333333333`; summary SHA-256 is
  `2155482d50d16ec429c6e392044cb3455b1d44a3b38d637c69abe62a22768b87`.
  The tied held-out values were recorded descriptively and did not alter the
  protocol. `Pure GatedDeltaNet Results!C26:D26,I26:J26` now contains the
  exact job, prompt, artifact, model, and hash provenance.
- At the 13:09 SAST pass, base progress was `5/16` completed. NER
  `1184504_1` and SA-general `1185142_5` were running on A100-40GB
  `gpu:ampere`; Belebele English `1185322_7` was priority-pending as the third
  owned lane. HEX scratch was `93/300 GB` (`31.0%`). No A100-80GB or L40S
  work was active or schedulable, and no additional lane was submitted.
- Targeted progress at 13:11 SAST was NER `10,449/14,980` generation requests
  (`70%`, counter ETA about `1:54`) and SA-general's current request block
  `2,057/5,000` (`41%`, counter ETA about `1:16`); neither log contained a
  traceback, CUDA OOM, NCCL failure, runtime error, or generic error marker.
  These counters are request-block estimates, not guaranteed whole-lane ETAs.
  Kombuys remained read-only and idle: RTX 5090 `10 MiB/0%`, RTX 3080 Ti
  `1 MiB/0%`, scratch `61%`; no tmux evaluation or fine-tuning work was
  started there.
- Base lane 7, job `1185322_7`, completed `0:0` in `00:01:58`. All five
  frozen Belebele English prompts returned accuracy
  `0.2733333333333333`; summary SHA-256 is
  `ce1f3dcfd8f7bb062b0a96f4f114b4089febc1435d651dbe9d43f229bd17bbbf`.
  The tied held-out values were recorded descriptively without changing the
  protocol. `Pure GatedDeltaNet Results!C25:D25,I25:J25` now contains the
  exact job, prompt, artifact, model, and hash provenance.
- After verifying no lane-8 summary or active duplicate, frozen Belebele
  Sesotho job `1185855_8` was submitted into the free third A100-40GB slot.
  Early health confirmed the canonical local `final_model`,
  `peft_adapter=None`, `merge_lora=False`, BF16, and no fault marker.
- At 14:10 SAST, base progress was `6/16`. NER `1184504_1` was at
  `12,761/14,980` generation requests (`85%`, counter ETA about `57m`) and
  SA-general `1185142_5` was at `4,329/5,000` in its current request block
  (`87%`, counter ETA about `17m`). All three owned jobs were A100-40GB
  `gpu:ampere`; scratch remained `93/300 GB` (`31.0%`). Kombuys remained
  read-only and idle on both GPUs with scratch `61%` used.
- Base lane 8, job `1185855_8`, completed `0:0` in `00:02:04`. All five
  frozen Belebele Sesotho prompts tied at accuracy
  `0.2733333333333333`; summary SHA-256 is
  `0a4892d547184febc8830f7a5304108bb62cd927717dce6cfe7f836cf28d5dcb`.
  `Pure GatedDeltaNet Results!C24:D24,I24:J24` now records the verified
  descriptive result and full provenance. With no lane-9 summary or duplicate,
  frozen siSwati Belebele job `1185870_9` was submitted into the freed
  A100-40GB slot.
- Base lane 9, job `1185870_9`, completed `0:0` in `00:01:59`; all five
  siSwati prompts tied at accuracy `0.2733333333333333` and summary SHA-256
  `4348955a09cf8ea1627dab36631012bef76d4372cd10de8d5db1ad2432a45fdd`.
  Verified provenance is in `Pure GatedDeltaNet Results!C23:D23,I23:J23`.
- Base lane 10, job `1185875_10`, completed `0:0` in `00:01:59`; all five
  Setswana prompts tied at accuracy `0.2733333333333333` and summary SHA-256
  `c894d458cd3b1940204e368eefbe04d5d94a3f158479800411f47143b32cc15d`.
  Verified provenance is in `Pure GatedDeltaNet Results!C22:D22,I22:J22`.
  Frozen Xitsonga Belebele lane 11 was submitted as `1185878_11` after
  verifying no summary or active duplicate.
- Base lane 11, job `1185878_11`, completed `0:0` in `00:02:01`; all five
  Xitsonga prompts tied at accuracy `0.2733333333333333` and summary SHA-256
  `ac57800e3c1c67ee471c0217e2500d232a5d110aeb359c4ddc36af4676adb47e`.
  The inherited workbook omitted the eighth Belebele language label; its
  reserved blank row 27 was minimally completed with `B27=Tso` and the
  verified result/provenance in `C27:D27,I27:J27`. Old hybrid and backup tabs
  remain untouched. Frozen Xhosa Belebele lane 12 was submitted as
  `1185883_12` after verifying no summary or active duplicate.
- Base lanes 12 and 13 completed cleanly. Xhosa job `1185883_12` and Zulu job
  `1185884_13` each returned accuracy `0.2733333333333333` for all five frozen
  Belebele prompts. Their summary SHA-256 values are respectively
  `6f3b0b9c9b8ff4ec582c964623f063e98540488d75c6d682e2c927e4907e12fc`
  and `a958533479fa44eb3d0fb90ac6e7be4889d7d20a2daac0bdaaa9d6358ffdb3ae`.
  The verified results and full provenance are published in rows 21 and 20 of
  `Pure GatedDeltaNet Results`.
- Base lane 5 SA-general job `1185142_5` completed `0:0`; summary SHA-256 is
  `7cf80bf4d15b98aaef71a35db3a28057542f0fd370d36dabe8ec26e2197e05fd`.
  Rows 28--39 of `Pure GatedDeltaNet Results` now contain every frozen prompt
  value and exact provenance. Descriptive best-prompt accuracy is AfriXNLI
  Xho/Zul/Sot/Eng `0.3567/0.3667/0.3667/0.3517`, AfriMMLU
  `0.2200/0.2560/0.2360/0.2380`, and flexible-extract AfriMGSM exact match
  `0.0120/0.0080/0.0040/0.0040`. These held-out values did not select or
  alter the frozen protocol.
- Initial generation jobs `1185900_14` (T2X Xhosa) and `1185901_15`
  (AfriHG) failed before producing a summary or metric. Both loaded the
  canonical local pure-GDN model and then failed on their first five-beam
  generation with `AttributeError: 'NoneType' object has no attribute
  'index_select'` while Transformers reordered the FLA cache. The evaluator
  had overridden the canonical model's saved `use_cache=false` with a
  hard-coded `true`. The shared generation path now respects
  `model.config.use_cache`; its focused regression test passes.
- After syncing that one-file implementation correction and confirming no
  completed summary or active duplicate, replacement jobs `1185914_14` and
  `1185915_15` started together on A100-40GB `gpu:ampere` at 14:50 SAST.
  Both remained running without a fault after passing the former failure
  window; logs confirmed canonical checkpoint loading and real task datasets.
  NER `1184504_1` remained healthy at `14,393/14,980` requests (96%, current
  block ETA about 15 minutes). The only three owned GPU lanes are these three
  A100-40GB jobs; scratch is `93/300 GB` (31.2%).
- Base lane 1 MasakhaNER job `1184504_1` completed `0:0` at 15:05:43 SAST
  after `06:22:45`. All five frozen prompts returned flexible-extract F1
  `0.000000` for Xhosa, Zulu, and Setswana. This verified negative result is
  not a missing run and did not change the protocol. Summary SHA-256 is
  `d01df9bcd264e398ebfdf5cc57f9af9d6e5e1d7e35e9f4577a2e3698c4cc18d3`.
  `Pure GatedDeltaNet Results!C4:D6,I4:J6` now records the exact task-native
  result, job, model/protocol identity, raw/summary paths, and hash; the write
  and preserved formatting were re-read successfully.
- Base progress is therefore `14/16`. Replacement T2X `1185914_14` and
  AfriHG `1185915_15` remained healthy on A100-40GB after about 20 minutes,
  with canonical local model loading, real held-out task preparation, and no
  traceback/OOM/NCCL/runtime fault. Their generation evaluator emits no
  per-request counter, so a defensible completion ETA is not yet available.
  HEX scratch is `94/300 GB` (31.4%); no A100-80GB or L40S work is active or
  schedulable.
- Kombuys remained read-only and idle: RTX 5090 `10 MiB/0%`, RTX 3080 Ti
  `1 MiB/0%`, and `/scratch` 61% used. No evaluation or fine-tuning process was
  started there.
- At 16:11 SAST, T2X `1185914_14` and AfriHG `1185915_15` had each run
  `01:20:10` on A100-40GB with no traceback, CUDA OOM, NCCL, runtime, or cache
  fault. AfriHG selected automatic generation batch size 64 at 15:56:13,
  about 65 minutes after entering its first generation probe. T2X had not yet
  completed its 512-token probe. This establishes a material 24-hour wall-time
  risk from uncached five-beam generation, but not a failure or a defensible
  whole-job ETA yet. The frozen beam configuration was not changed and neither
  job was duplicated or cancelled. HEX scratch was `96/300 GB` (32.1%); only
  these two A100-40GB jobs were active.
- Planning deadline from the user: finish all experiments and evaluations by
  31 August 2026. Paper drafting begins afterward, targeting a first advisor
  draft around mid-September; no paper writing is requested during this gate.
- At 17:13 SAST, both remaining jobs were still healthy after `02:22:06`.
  T2X `1185914_14` completed its automatic batch-size-64 probe at 16:43:49,
  roughly 1h52m after generation preparation, and entered real generation.
  AfriHG `1185915_15` continued real Xhosa generation after its 65-minute
  probe. Neither log contained a traceback, OOM, NCCL/runtime/cache fault or
  summary yet. If probe time were representative, T2X's six real batches plus
  probe would fit within 24 hours, whereas AfriHG's roughly 21 batches per
  language would not; compilation and early EOS can materially change that
  extrapolation. Preserve the current jobs until actual task-completion or
  terminal evidence is available. Scratch was `98/300 GB` (32.7%), and only
  these two A100-40GB jobs were active.
- Base lane 14 T2X Xhosa job `1185914_14` completed `0:0` at 17:35:31 SAST
  after `02:44:37`. The frozen zero-shot, five-beam result is chrF
  `3.1401008038579232`, ROUGE-1/ROUGE-L `0.000816014651532441`, ROUGE-2
  `0.0`, and BLEU `0.0`. Summary SHA-256 is
  `542a9729e92bd7afe44f68aa85c74d3fab218e1c6f9e662e82c78a98c34667d0`.
  `Pure GatedDeltaNet Results!C41:D41,I41:J41` now records the task-native
  chrF headline and exact job/model/protocol/artifact provenance; the cells,
  note, hash, and preserved wrapping/date formatting were re-read after the
  write. There was one frozen template, so no held-out prompt selection.
- Base progress is `15/16`. At 18:14 SAST, AfriHG `1185915_15` remained
  healthy after `03:23:06`; it logged real input-truncation progress at
  17:19 and 17:44 with no traceback, OOM, NCCL/runtime/cache fault. T2X's
  post-probe work finished in only 52 minutes, showing probe-time
  extrapolation was overly pessimistic; AfriHG completion ETA remains
  uncertain but its 24-hour risk is reduced. HEX scratch was `99/300 GB`
  (33.0%), with only AfriHG on A100-40GB active. Kombuys remained read-only
  idle: RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch 61%.
- At 19:42 SAST, AfriHG `1185915_15` remained the sole owned GPU job and was
  healthy after `04:51:31` on one A100-40GB `gpu:ampere`. New real Xhosa
  generation progress was logged at 18:48 and 18:59; there was still no
  summary or traceback/OOM/NCCL/runtime/cache fault. Its 24-hour limit leaves
  about 19 hours, but a completion ETA remains uncertain. HEX scratch was
  `99/300 GB` (`33.2%`). Kombuys remained read-only and idle: RTX 5090
  `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch 61%.
- At 21:12 SAST, AfriHG `1185915_15` had completed and saved the frozen Xhosa
  task, then started Zulu. Xhosa chrF is `4.079834809867061`, ROUGE-1 and
  ROUGE-L are `0.00012029488297117726`, and ROUGE-2/BLEU are `0.0`; the
  task metrics SHA-256 is
  `9cbb9f7572794838c9cb2b53104ff814bd688bdf225d1b22ed7897b15001f198`.
  Zulu prepared `1,776` examples and selected batch size 64 at 21:02 after a
  short warm probe. The combined `evaluation_summary.json` does not exist
  yet, so the sheet remains unchanged and Xhosa must not be repeated. The
  job remained healthy after `06:21:05` on the sole A100-40GB lane; HEX
  scratch was `99/300 GB` (`33.2%`). Kombuys remained read-only and idle.
- At 22:12 SAST, `1185915_15` logged its first post-probe Zulu generation
  progress at 22:08, confirming the task is advancing rather than stalled.
  It remained healthy after `07:21:04` with no traceback/OOM/NCCL/runtime
  fault or combined summary. HEX scratch was `100/300 GB` (`33.4%`); only
  the A100-40GB lane was active. Kombuys remained read-only and idle.
