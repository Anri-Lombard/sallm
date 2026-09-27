# SALLM advisor meeting update - 2026-07-02

Coverage window: Thursday 2026-06-25 through Wednesday 2026-07-01.

## Short version

This week changed the interpretation of the xLSTM downstream failures. The xLSTM HPO-selected generative/eval wave is mixed: News and Intent are strong, NER is nonzero but modest, SIB is very weak, and generation still has source-binding problems. But the new task-head diagnostics show that xLSTM representations are not empty or inherently unusable for structured tasks. Under supervised heads, xLSTM is the strongest architecture on NER and competitive or strongest on Intent/POS. That means the likely failure is the decoder-only generative adapter/evaluation formulation, not the base representation.

GatedDeltaNet has not produced a SALLM training result yet. Most work there was launch/config/speed triage: fixing wrapper/env issues, getting a 125M config to smoke, installing fast kernels, finding that the 57-layer shape is too slow, then moving to a shallower/wider full-run plan that is still queued. Do not present GDN as a result yet.

## Completed xLSTM HPO-selected held-out wave

Held-out test jobs `952247`-`952252` completed cleanly on 2026-06-25:

| Job | Task | Status | Main result |
| --- | --- | --- | --- |
| `952247` | SIB all | completed `00:05:00` | weak; best per-language F1 only `0.1260`/`0.1415`/`0.0800`/`0.1032`/`0.1175`/`0.0888` |
| `952248` | MasakhaNews all | completed `00:14:12` | strong; best F1 Eng `0.8686`, Xho `0.9238` |
| `952249` | InjonGoIntent all | completed `00:39:59` | strong Eng/Xho, weaker Sotho/Zulu; best F1 Eng `0.9064`, Sot `0.6998`, Xho `0.8463`, Zul `0.6226` |
| `952250` | AfriHG all | completed `08:35:14` | Xho chrF `15.1960`, Zul chrF `17.0986`; BLEU `0.0`, ROUGE-L near `0.013` |
| `952251` | T2X Xho | completed `00:11:56` | chrF `30.8090`, BLEU `0.0325`, ROUGE-L `0.2644` |
| `952252` | MasakhaNER all | completed `00:41:55` | nonzero but modest; best F1 Tsn `0.3174`, Xho `0.2538`, Zul `0.2545` |

Artifacts:

- Scratch root: `/scratch/lmbanr001/masters/sallm/results/eval/xlstm_hpo_test_20260624`
- Local compact summaries were pulled to `/tmp/sallm_eval_summaries/`

NER caveat: the stock MasakhaNER task naming still surfaced `_val` names in the summary, but the submitted manifest row was `masakhaner_all_test_tmp`, and the temporary YAML set `validation_split: test`. Treat as held-out/test with that metadata caveat recorded.

Interpretation:

- HPO selection helped task-specific classification/extraction enough to keep News, Intent, and NER alive.
- HPO did not rescue SIB.
- HPO did not solve generation. T2X HPO-selected chrF `30.81` is below the previous xLSTM mixed-source held-out T2X result `34.1600`.
- AfriHG is still far below strong transformer references despite nonzero chrF.

## Root-cause examples from the HPO-selected wave

SIB failure:

- Best prompt predicted mostly `geography` and `sports`.
- Example prediction counts on prompt 2:
  - Afrikaans: `geography` 131, `sports` 68 out of 204.
  - English: `geography` 129, `sports` 68.
  - Xhosa: `geography` 132, `sports` 66.
- Hypothesis: raw continuation scoring over labels with different lengths biases toward short/high-prior labels. Character normalization helped a bit but then overpredicted `science/technology`, so the deeper issue is not solved by one normalization tweak.

NER failure:

- The HPO-selected adapter has extraction signal but ugly generations.
- In best prompts, raw outputs often start with junk such as `god` or `godition*`.
- Tswana best prompt: 996 docs, 1506 gold entities, 685 predicted entities, exact doc-level entity hit 129; predicted `date` 457 times versus gold `date` 179.
- Interpretation: not pure no-signal failure. It is a noisy-generation, calibration, boundary, and label-skew problem.

T2X failure:

- Not an empty-output or length issue: mean length ratio about `1.06`, median about `1.00`.
- Source binding is the problem: only `28.8%` of predictions contain the subject string, `25.7%` contain the object/value string, and only `9.0%` contain both.
- Examples showed fluent-ish but wrong substitutions, e.g. source entities/values get replaced by unrelated names or nationalities.
- Next useful direction is delexicalization/placeholders/copy-aware formatting, not another broad LR/epoch sweep.

## Task-head pivot: important new diagnostic result

The key new result is that supervised task heads recover much stronger structured-task signal than the generative adapters.

Initial held-out trainable task-head jobs `955145`-`955150` completed cleanly:

| Architecture | SIB macro F1 | SIB accuracy | NER entity F1 | NER precision | NER recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| LLaMA | `0.7369` | `0.7533` | `0.3198` | `0.2794` | `0.3739` |
| xLSTM | `0.7022` | `0.7173` | `0.4155` | `0.3800` | `0.4583` |
| Mamba | `0.6361` | `0.6495` | `0.2333` | `0.2076` | `0.2662` |

This already suggested:

- xLSTM is best on NER under task heads.
- LLaMA is best on SIB.
- Mamba trails both.
- Base retraining is not the next move, because representations are recoverable.

Then the broader task-head matrix and repeat seeds made this more defensible.

Completed task-head matrix:

- Missing `llama_pos_zul` row was rerun as `966424_27`; completed in `00:01:53`.
- Full matrix reached 27/27 artifacts.
- LLaMA POS Zulu row: macro F1 `0.4616`, token accuracy `0.6675`, weighted F1 `0.6620`, exact match `0.0137`, n=`7871` tokens.

Intent/POS one-seed diagnostic summary:

- xLSTM best on every Intent/POS row in the diagnostic matrix.
- Average macro F1:
  - Intent: xLSTM `0.7543`, LLaMA `0.7080`, Mamba `0.5302`.
  - POS: xLSTM `0.4672`, LLaMA `0.4601`, Mamba `0.3520`.

SIB/NER repeat-seed diagnostic:

- Kombuys task-head SIB/NER matrix completed 33/33 JSON artifacts.
- NER F1 winners: xLSTM wins all rows.
  - NER all: xLSTM `0.4224`; LLaMA/Mamba lower.
  - NER Tsn `0.4655`, Xho `0.3764`, Zul `0.3450`.
  - Mean across NER rows: xLSTM `0.4023`, LLaMA `0.3359`, Mamba `0.3055`.
- SIB macro-F1 winners: LLaMA wins all/afr/eng/nso/xho/zul; xLSTM wins Southern Sotho.
  - Mean across SIB rows: LLaMA `0.7207`, xLSTM `0.6871`, Mamba `0.6135`.

Three-seed core result after fixing Mamba runtime and rerunning Mamba seed 13:

| Task | Winner | xLSTM | LLaMA | Mamba |
| --- | --- | ---: | ---: | ---: |
| NER all F1 | xLSTM | `0.4032 +/- 0.0236` | `0.3553 +/- 0.0397` | `0.2491 +/- 0.0105` |
| NER Tsn F1 | xLSTM | `0.4592 +/- 0.0092` | `0.4229 +/- 0.0210` | `0.3210 +/- 0.0073` |
| NER Xho F1 | xLSTM | `0.3642 +/- 0.0154` | `0.3214 +/- 0.0239` | `0.2017 +/- 0.0253` |
| NER Zul F1 | xLSTM | `0.3568 +/- 0.0147` | `0.3299 +/- 0.0295` | `0.2038 +/- 0.0589` |
| SIB all macro F1 | LLaMA | `0.7030 +/- 0.0032` | `0.7355 +/- 0.0149` | `0.6205 +/- 0.0115` |

This is the cleanest research conclusion of the week:

The structured-task weakness is not simply "xLSTM cannot represent NER/POS/Intent/SIB." The base representations are useful under the right objective. The failure is more likely in decoder-only generative task formulation, LoRA/adaptation dynamics, label verbalization/scoring, or output constraints.

## Frozen-backbone control

A small frozen-backbone control was run on Kombuys on 2026-06-30:

- Matrix: 12 rows, xLSTM/Mamba/LLaMA x SIB all, NER all, Intent all, POS all.
- Seed: 13.
- Train/eval cap: 512.
- Eval split: test.
- Output: `sallm_memory/artifacts/2026-06-30/task_head_frozen_probe_20260630/summary.md`

Frozen winners match trainable-backbone winners:

| Architecture | Task | Frozen | Trainable matched setting | Gain |
| --- | --- | ---: | ---: | ---: |
| xLSTM | SIB | `0.6491` | `0.7860` | `+0.1369` |
| xLSTM | NER | `0.3670` | `0.4936` | `+0.1266` |
| xLSTM | Intent | `0.6577` | `0.7959` | `+0.1381` |
| xLSTM | POS | `0.6440` | `0.7385` | `+0.0945` |
| Mamba | SIB | `0.6232` | `0.6819` | `+0.0587` |
| Mamba | NER | `0.2414` | `0.3092` | `+0.0678` |
| Mamba | Intent | `0.4835` | `0.5572` | `+0.0737` |
| Mamba | POS | `0.5844` | `0.6266` | `+0.0422` |
| LLaMA | SIB | `0.6876` | `0.7963` | `+0.1086` |
| LLaMA | NER | `0.2526` | `0.4309` | `+0.1783` |
| LLaMA | Intent | `0.4677` | `0.7641` | `+0.2964` |
| LLaMA | POS | `0.6210` | `0.7330` | `+0.1120` |

Interpretation:

- Frozen representations already show the same architecture pattern.
- Backbone adaptation amplifies the signal but does not flip the winners.
- Mamba remains behind in both frozen and trainable settings, so its gap is not just a task-head optimizer issue.

## Mamba runtime audit

Kombuys initially lacked Mamba fast-path dependencies:

- `mamba_ssm` missing.
- `causal_conv1d` missing.
- `selective_state_update`, `causal_conv1d_fn`, and `causal_conv1d_update` unavailable.

Installed into the Kombuys venv:

- `causal-conv1d 1.6.2.post1`
- `mamba-ssm 2.3.2.post1`

Smoke passed:

- `MAMBA_FORWARD_OK (1, 10, 65536) torch.float32`

Then Mamba seed-13 rows were rerun under the fixed runtime. Scores did not improve:

- SIB all: `0.6653 -> 0.6334`
- NER all: `0.2839 -> 0.2407`
- NER Tsn: `0.3770 -> 0.3268`
- NER Xho: `0.2610 -> 0.1776`
- NER Zul: `0.3002 -> 0.2496`

Interpretation: the Mamba task-head gap is unlikely to be explained by the missing fast-path runtime alone.

## GatedDeltaNet status

No completed SALLM GatedDeltaNet result exists yet.

What happened:

- Original 4x L40S job `956550` was queue-bound.
- A 2x L40S replacement `961688` failed immediately from Slurm spool-path `SCRIPT_DIR`.
- Fixed replacement `961893` failed because `sallm-uv` conda env was absent.
- No-conda replacement `964130` was queued, then moved to A100 as `964131`.
- A100 job `964131` exposed a real config issue: the first `gated_deltanet_125m.yaml` produced only `91.23M` params, below the 120M-130M validation gate.
- Created 57-layer config `gated_deltanet_125m_57l.yaml`; smoke job `964133` passed at `125.00M` params.
- Training jobs then hit practical issues:
  - `964134`: W&B symlink target missing.
  - `964135`: Hugging Face dataset cache hit scratch quota.
  - `964136`: reached training start but failed on Triton/TorchInductor cache quota.
  - `964143`: with cache dirs fixed, reached training but ran at about `40s/step`, implying about 95 days.
  - Fast kernels were installed and verified: `causal-conv1d==1.6.2.post1`, `flash-linear-attention==0.5.1`, `FAST_KERNEL_SMOKE_OK`, logs reported FLA fast path available.
  - `964155`: fast-kernel 57-layer run improved to about `4.5-4.7s/step`, but full 5-epoch plan was still 205395 steps, around 10-11 days.
  - Packed/checkpointing probes still did not make the 57-layer shape viable under 48h.

Current replacement direction:

- Moved away from the 57-layer shape.
- Submitted shallow/wider GDN full run `965310` (`gdn33w-b6-full`) on 4x A100 40GB.
- `965310` failed before training because W&B init timed out after 90s.
- Replacement `973395` (`gdn33w-b6-wbfix`) was submitted with:
  - `WANDB_INIT_TIMEOUT=300`
  - `WANDB__SERVICE_WAIT=300`
- Last successful check on 2026-06-30 21:05:
  - `973395` pending on `a100`, reason `AssocGrpGRES`
  - requested `gres/gpu:ampere:4`
  - projected start `2026-07-02T13:52:11`
  - no log/checkpoint yet
- On 2026-07-01, monitoring was blind due repeated VPN/HEX SSH timeouts, so there is no fresh confirmed Slurm state.

Interpretation:

GDN is still a feasibility experiment, not a result. The useful technical finding is that the deep 57-layer 125M config is too slow; the better direction is the shallow/wider 125M config, but it still needs a real training run.

## BabyLM GDN side finding

Separate from SALLM, BabyLM GDN work produced a useful diagnostic:

- 2k synthetic entity examples with 50% empty state improved official-style entity scores.
- Seed 13:
  - BLiMP `67.14`, Supplement `55.20`, EWoK `50.55`, Entity `44.15`
  - Entity split: all `41.91`, gold-empty `99.45`, non-empty `18.41`
- Seed 29:
  - BLiMP `67.87`, Supplement `59.60`, EWoK `48.45`, Entity `39.92`
  - Entity split: all `38.86`, gold-empty `87.96`, non-empty `18.81`
- 25% empty variant:
  - Entity `37.99`
  - all `36.14`, gold-empty `77.35`, non-empty `19.30`

Interpretation:

The gain is real but narrow: it improves empty-state calibration, not robust object tracking. Custom balanced probes still show weak out-of-template non-empty tracking.

## Operational notes

- Scratch quota was repeatedly a blocker during GDN setup. At points the run hit disk quota during HF dataset cache generation and during Triton/TorchInductor cache writes.
- Approved cleanup removed old downstream/checkpoint/result/cache artifacts, but not active evidence without approval.
- HEX/VPN/DNS flapping repeatedly blocked monitoring. On 2026-07-01 the GDN replacement could not be refreshed for most of the day because SSH timed out before the required quota-first pass.
- Kombuys/Jan is now useful for small single-GPU task-head/eval/debug jobs. It is not being used for full SALLM pretraining.

## Recommended discussion

1. Treat task heads as a separate dissertation branch: decoder-only generation tests instruction/output formulation; task heads test representation quality.
2. Do not retrain xLSTM base because SIB/NER/Intent/POS signal is recoverable under supervised heads.
3. For NER/POS/Intent, xLSTM may be more promising than the generative results suggested.
4. For SIB, LLaMA remains strongest, but xLSTM is close enough under task heads to be worth discussing.
5. Stop small prompt-only T2X rescues; the problem is source binding/copying.
6. Keep GDN framed as pending until `973395` or a successor produces a checkpoint and downstream evidence.

## Immediate next steps

1. Restore HEX connectivity and check `973395`.
2. If `973395` runs, monitor first training steps, step speed, checkpoint creation, and W&B timeout behavior.
3. If `973395` fails before training again, fix only the first launch blocker; do not change model science at the same time.
4. Decide whether the task-head branch should become a formal comparison table in the dissertation.
5. Keep task-head rows separate from the original generative benchmark in the Google Sheet and write-up.
