# GDN anomaly root-cause and recovery track — 2026-07-27

## Decision summary

- **AfriHG generation is recovered and closed.** A shared evaluator context-window
  bug was fixed, then a validation-only rank/LR/seed search selected one checkpoint.
  The frozen held-out test now gives Xhosa/Zulu chrF `23.8442/24.4012`, above the
  original monolingual GDN `14.8387/14.2960` and the original multilingual
  `21.1288/18.8708`. No further AfriHG HPO is justified.
- **INJOngo Intent is a genuine class-prior/optimization collapse, not a row,
  label-order, split, truncation, or aggregation bug.** Both mono and multilingual
  task adapters choose `text` (label index 30) almost universally. The general
  adapter instead collapses to `alarm`. The apparent multilingual advantage is
  negligible at chance scale.
- **Monolingual NER underperformance is genuine under the same corrected r3
  evaluator.** Xhosa and Tswana mono are `0.0` F1; multilingual is
  `0.3154/0.4157`. The strongest actionable recipe confound is effective batch:
  mono Xhosa uses `32 x 8 = 256`, while multilingual uses `32 x 2 = 64`. With
  1,441 mono training rows, Xhosa gets only about six optimizer steps per epoch
  and early-stops after four epochs.
- **One focused validation-only NER canary is justified but was not launched.**
  The exact collision-free Kombuys GPU-1 launch was blocked because fresh
  confirmation is required for the specific external GPU run and output writes.
  Do not replace it with a broader sweep.

## Canonical sheet mapping and live state

Spreadsheet:
`https://docs.google.com/spreadsheets/d/1Ph_zVcSuLZy0dUBnDPF4tkLqAfEVCwK9JVybB2_8x6U/edit`

Tab `GatedDeltaNet Results` (`sheetId=1825869548`):

- MasakhaNER: rows `4-6`
- MasakhaPOS: rows `7-9`
- INJOngo Intent: rows `16-19`
- T2X: row `41` (no multilingual task adapter; not the generation anomaly)
- AfriHG: rows `42-43`

Live-read findings:

- NER rows contain the corrected r3 mono/multilingual means, but the
  general-adapter cells are stale (`Running ... 85.6%`).
- Intent rows still show the older r2 values and queued base/general text; the
  corrected r3 and completed general-adapter artifacts are not reflected.
- AfriHG rows already contain the recovered validation-selected frozen-test
  results.
- No sheet write was made in this track.

## 1. AfriHG generation anomaly

### Root causes

1. `src/main/sallm/evaluation/generation_metrics.py` truncated long prompts to
   `model_ctx_limit - 1`, leaving only one token for generation. The shared fix
   reserves `max_new_tokens`:
   `window_len = max(1, model_ctx_limit - self.max_new_tokens - 1)`.
2. After that repair, the original rank-16/LR-`8e-5` recipe still showed genuine
   rank/optimization sensitivity.

### Validation-only recovery and frozen test

- Selected without test: LR `2e-4`, rank/alpha `32/64`, seed `43`,
  checkpoint `1158`.
- Corrected validation chrF: Xho `23.6045`, Zul `24.9915`.
- One-time frozen test chrF: Xho `23.8442`, Zul `24.4012`.
- Test integrity: Xho `1305` rows, Zul `1776`; zero empty or one-token outputs;
  `1301/1751` unique predictions.

Artifacts:

- `/scratch/alombard/masters/sallm/results/final/gdn_afrihg_r32_s43_test_xho_ctxreserve/evaluation_summary.json`
- `/scratch/alombard/masters/sallm/results/final/gdn_afrihg_r32_s43_test_zul_ctxreserve/evaluation_summary.json`

Held duplicates `1117467/1117468` remain user-held and must not be released or
deleted without fresh approval.

Controls from the live sheet are not fully recipe-matched. Existing xLSTM
multilingual AfriHG is Xho/Zul `17.0169/18.8984`; the old Transformer
multilingual cells are `16.9779/17.4815` chrF, while older local LLaMA bests in
the progress log are about `20.22/23.00`. The recovered GDN result therefore
removes the stated generation deficit; it does not support a remaining
architecture-level failure claim.

## 2. INJOngo Intent

### Evaluator, labels, splits, and data

- Corrected evaluation is 40-way log-likelihood multiple choice across five
  prompts, with complete rows and no generation truncation.
- Training-template label mapping and lm-eval choice order match exactly:
  `alarm ... text (index 30) ... weather (index 39)`.
- Eng: train `1779`, test `622`, no upstream dev/validation. The loader uses a
  deterministic per-class hash holdout from train.
- Sot/Xho/Zul: train `2240`, validation `320`, test `640` each; all have 40
  labels, exactly `56/8/16` examples per class in train/validation/test.
- Multilingual log: `8309` training rows and `1150` raw validation rows,
  expanded across five prompts to `5750`.
- Zulu few-shot evaluation is quarantined because the source validation split
  contains shifted/corrupt label-text pairs. Example:
  `Ungangiphakamisela ukudla okuvela eGhana?` is labelled `car_rental`.

### Failure shape

Corrected task-specific artifacts:

- `/scratch/alombard/sallm/results/eval/gdn_125m_adapter_injongointent_all_r3/injongointent_all/results.json`
- `/scratch/alombard/sallm/results/eval/gdn_125m_adapter_injongointent_xho_r3/injongointent_xho/results.json`

For Xhosa, task-specific GDN predicts `text`:

- multilingual prompt 1: `639/640`; prompts 4/5: `640/640`
- monolingual prompt 1: `637/640`; prompts 4/5: `640/640`

The same collapse occurs in Eng/Sot/Zul. Representative Xhosa `alarm` and
`balance` examples both choose index 30. Five-prompt accuracy remains near the
`1/40 = 0.025` chance rate, with F1 around `0.001-0.0056`.

The completed general adapter collapses to `alarm` instead:

- `/scratch/alombard/sallm/results/final/gdn_general_adapter_full_matrix_20260726/injongointent_all/results.json`
- Xhosa prompt 1: all 16 inspected `alarm` rows are correct; following
  `balance` rows are also predicted `alarm`.

This rules out the hypothesis that the corrected multilingual score is a real
language-transfer breakthrough. It is a tiny chance-level difference between
two different class-prior collapses.

### Training and checkpoint selection

GDN task adapters:

- LoRA rank/alpha `16/32`, about `2.66M` trainable parameters.
- Validated target modules:
  `in_proj_qkvz,in_proj_ba,out_proj,q_proj,k_proj,v_proj,o_proj`.
- LR `8e-5`.
- Multilingual effective batch `64`, configured 10 epochs, stopped after epoch
  4; validation loss improved `1.9488 -> 1.5931`.
- Mono Xhosa effective batch `32`, configured 20 epochs, stopped after epoch 4;
  validation loss improved `2.2312 -> 1.6269`.
- The configured selector is validation
  `eval_classification/all_f1`; generated validation completions are
  empty/immediate-EOS even as teacher-forced loss improves.

Historical logs:

- `/scratch/lmbanr001/masters/sallm/logs/jobs/ft-injongointent_all-1020316.out`
- `/scratch/lmbanr001/masters/sallm/logs/jobs/ft-injongointent_xho-1020319.out`

### Controls and interpretation

- Transformer sheet rows `17-20` are also near chance: multilingual F1
  Eng/Xho/Zul/Sot `0.029/0.010/0.007/0.007`.
- xLSTM sheet rows `16-19` are strong after HPO: Eng/Xho/Zul/Sot
  `0.9064/0.8463/0.6226/0.6998`.
- xLSTM selected adapter `imrufs4l` used rank/alpha `256/512`, targets
  `q,k,v,out_proj,embeddings`, `32.8M` trainable parameters (`20.5%`), LR
  `2.2056e-4`, effective batch `32`, eight epochs, constant-with-warmup.
- Artifact:
  `/scratch/lmbanr001/masters/sallm/results/eval/xlstm_hpo_test_20260624/injongointent_all/evaluation_summary.json`
  (job `952249`).

The xLSTM comparison is therefore confounded by roughly 12x adapter capacity,
HPO, higher LR, and the multilingual data volume. It proves the dataset and
40-way evaluator are learnable; it does not isolate architecture.

Mean-normalized choice scoring may reduce multi-token-label bias, but it cannot
be the sole cause because the xLSTM control succeeds on the same sum-scored
test pack. It is lower priority than testing capacity/LR.

## 3. MasakhaNER monolingual versus multilingual

### Corrected held-out results

Five-prompt F1 means:

| Language | Mono | Multilingual |
| --- | ---: | ---: |
| Xho | `0.0000` | `0.3154` |
| Zul | `0.1279` | `0.3689` |
| Tsn | `0.0000` | `0.4157` |

Multilingual prompt ranges:

- Xho `0.2289-0.3594`
- Zul `0.2654-0.4374`
- Tsn `0.3060-0.4510`

Artifacts:

- `/scratch/alombard/sallm/results/eval/gdn_125m_adapter_masakhaner_all_r3/masakhaner_all/results.json`
- `/scratch/alombard/sallm/results/eval/gdn_125m_adapter_masakhaner_xho_r3/masakhaner_xho/results.json`
- `/scratch/alombard/sallm/results/eval/gdn_125m_adapter_masakhaner_zul_r3/masakhaner_zul/results.json`
- `/scratch/alombard/sallm/results/eval/gdn_125m_adapter_masakhaner_tsn_r3/masakhaner_tsn/results.json`

Representative matched Xhosa row:

- Target:
  `DATE: ngomhla weshumi elinesibhozo kule nyanga yesiLimela $$ ORG: yiPirates $$ DATE: ngale njikalanga`
- Mono raw response copies/continues the input; the flexible extractor returns
  empty.
- Multilingual raw response:
  `ORG: yiPirates $ ORG: yiPirates`; it is incomplete/duplicated but contains a
  real entity signal.

Representative mono Zulu row has target `LOC: India` and returns whitespace.

### Exact volume and optimization burden

Historical logs give:

- multilingual: train `4323`, validation `2152`
- mono Xho: train `1441`, validation `817`
- mono Zul: train `1441`, validation `836`
- mono Tsn: train `1441`, validation `499`

All use the same five templates in cycle mode, assistant-only loss, max length
`2048`, LR `8e-5`, rank/alpha `16/32`, and the validated GDN target modules.

The material mismatch:

- multilingual: batch `32`, accumulation `2`, effective batch `64`,
  `4323/64 ~= 68` optimizer steps per epoch; ten epochs completed; validation
  loss `1.7159 -> 1.2223`.
- mono Xho: batch `32`, accumulation `8`, effective batch `256`,
  `1441/256 ~= 6` optimizer steps per epoch; early-stopped after epoch 4
  (`24` steps); validation loss `4.6728 -> 2.5563`.
- mono Tsn also uses accumulation `8`; mono Zul uses accumulation `4`.

Historical logs:

- `/scratch/lmbanr001/masters/sallm/logs/jobs/ft-ner_all-1020328.out`
- `/scratch/lmbanr001/masters/sallm/logs/jobs/ft-ner_xho-1020337.out`
- `/scratch/lmbanr001/masters/sallm/logs/jobs/ft-ner_tsn-1020330.out`
- `/scratch/lmbanr001/masters/sallm/logs/jobs/ft-ner_zul-1020339.out`

### Tokenization/truncation caveat

- r3 result prompts are short in character terms: maximum prompt lengths are
  multilingual `1135`, mono Xho `1090`, mono Zul `1079`, mono Tsn `1135`
  characters; no truncation warning appears in the evaluated artifacts/logs.
- This makes 2048-token truncation an implausible primary cause, but exact
  training token-length/truncation counts were not recomputed. A proposed
  read-only audit script could not be copied to Kombuys because new remote
  writes require fresh confirmation. Do not state a measured zero truncation
  rate without that audit.

### Controls

- Live Transformer rows `4-6`: mono/multilingual F1
  Xho `0.720/0.693`, Zul `0.670/0.702`, Tsn `0.772/0.779`.
  These are not recipe-matched: the sheet records 50 epochs, v5-only template,
  `apply_chat_template=true`, and the ByteLevel decoder repair.
- Live xLSTM rows `4-6`: mono/multilingual F1
  Xho `0.2065/0.2538`, Zul `0.1473/0.2545`, Tsn `0.2363/0.3174`.
  The multilingual values are HPO-selected test results from job `952252`.

The matched lesson is narrower than “GDN cannot do NER”: multilingual GDN has
real signal and exceeds the current xLSTM multilingual cells for all three
languages, while mono GDN is under-optimized and often copies or emits empty
spans.

## Proposed single canary (requires fresh confirmation)

Run on idle Kombuys GPU 1 only:

- task/data: Xhosa MasakhaNER, train `1441`, validation `817`; **no test**
- base:
  `anrilombard/sallm-gated-deltanet-125m-shallowwide-4x40-20260707`
- hold fixed: seed 42, LR `8e-5`, rank/alpha `16/32`, five cycle templates,
  assistant-only loss, max length 2048, original validated GDN target modules
- change only effective batch `256 -> 64` using microbatch `4`,
  accumulation `16` on the 12GB RTX 3080 Ti
- selector: validation `eval_all_f1`; 15-epoch ceiling, patience 3
- disable Hub push and online reporting
- unique output:
  `/scratch/alombard/masters/sallm/checkpoints/gdn_ner_xho_bs64_canary_20260727`
- unique log:
  `/scratch/alombard/masters/sallm/logs/gdn_ner_xho_bs64_canary_20260727.log`
- tmux:
  `gdn-ner-xho-bs64-canary`
- estimated runtime: approximately `1-3h` on the RTX 3080 Ti; uncertainty is
  dominated by full validation generation each epoch.

Promotion rule: compare validation F1 and output shape against the original
mono Xho checkpoint. Do not touch test unless this validation canary is selected
as the final recipe. Do not launch the Intent rank-32/LR-`2e-4` canary until the
NER result resolves; that avoids two simultaneous poorly identified changes.

## Live jobs, dependencies, and quota

Quota-first HEX check at the end of diagnosis:

- home `32.1%`; scratch `88.3%`
- running A100 work includes:
  - `1118020_1`, `1118021_1`: two/three-shot base NER
  - `1118021_5`: three-shot SA-general
  - `1118020_12`: two-shot array continuation
- remaining `1118020/1118021` cells are pending behind array `%2` limits.
- dependent xLSTM controls:
  - `1118353` NER test
  - `1118354` POS test
  - `1118355` Intent test
  all wait for both base arrays.
- held AfriHG duplicates `1117467/1117468` remain untouched.
- no L40S job was submitted because A100 work is active.

Kombuys:

- GPU 0 continues `gdn-pos-general-base-final`.
- GPU 1 is idle.
- the proposed NER canary was **not** started because fresh confirmation is
  required for the specific GPU consumption and external output writes.

ETA:

- Active few-shot NER cells were roughly four hours into execution at the last
  check and likely have a few hours remaining, but the full arrays and dependent
  xLSTM jobs have no reliable scheduler ETA.
- The proposed NER canary would provide the next causal answer about `1-3h`
  after explicit launch confirmation.
