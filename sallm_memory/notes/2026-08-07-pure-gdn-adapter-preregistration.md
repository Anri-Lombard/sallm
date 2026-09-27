# Pure-GDN Mono/Multi/General preregistration — 2026-08-07

Status: **frozen before any new fine-tuning submission or adapter held-out
evaluation**. Base held-out results and sheet values are excluded from every
selection below. This document may be amended only for an implementation or
infrastructure failure discovered before a validation metric exists; any such
amendment must be dated and must not use held-out results.

## Fixed model and LoRA contract

- Base artifact:
  `/scratch/lmbanr001/masters/sallm/checkpoints/sallm-pure-gdn-125m/a10080-3epoch-20260802/final_model`.
- Identity: pure FLA `GatedDeltaNetForCausalLM`, `attn=None`, exactly
  `127,425,448` parameters. No Qwen3Next or historical GDN--Attention Hybrid
  artifact is eligible.
- LoRA targets are exactly `q_proj`, `k_proj`, `v_proj`, `a_proj`, `b_proj`,
  `g_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`; rank 16, alpha 32,
  dropout 0.05. The diagnostic-only `q_proj`/`v_proj` recipe is forbidden.
- BF16, seed 42, AdamW betas 0.9/0.95, weight decay 0.01, max gradient norm
  1.0, cosine schedule, warmup ratio 0.03, assistant-only loss, no packing.
  Label smoothing is 0.05. Training prompt choice cycles through the frozen
  established prompt pack.
- Hardware is A100-40GB `gpu:ampere` only, at most three owned jobs active or
  schedulable. Do not overlap A100-40GB with A100-80GB or L40S work. Each job
  requests one GPU, eight CPUs, and at most 24 hours. Kombuys remains read-only.
- Hub publication and adapter pushing are disabled. Local validation artifacts
  and checkpoints are preserved until the official-test phase is complete.

## Dataset and validation contract

Every arm uses the established training and validation splits; no official test
row enters training, early stopping, checkpoint selection, recipe selection, or
rerun decisions.

| Family | Source and scope | Validation selection metric | Max epochs | Max length |
| --- | --- | --- | ---: | ---: |
| MasakhaNews | `masakhane/masakhanews`; Eng/Xho | mean macro-F1 over languages and P1--P5 | 10 | 1024 |
| MasakhaNER | `masakhane/masakhaner2`; Tsn/Xho/Zul | mean span F1 over languages and P1--P5 | 15 | 2048 |
| MasakhaPOS | `masakhane/masakhapos`; Tsn/Xho/Zul | mean token accuracy over languages and P1--P5 | 15 | 2048 |
| SIB-200 | `Davlan/sib200`; Afr/Eng/Nso/Sot/Xho/Zul | mean macro-F1 over languages and P1--P5 | 10 | 1024 |
| InjongoIntent | `masakhane/InjongoIntent`; Eng/Sot/Xho/Zul | mean macro-F1 over languages and P1--P5 | 10 | 2048 |
| T2X | `github:francois-meyer/t2x`; Xho only | validation chrF, frozen `t2x_verbalisation/v1` | 4 | 1024 |
| AfriHG | `github:dadelani/AfriHG`; Xho/Zul | mean validation chrF, frozen `afrihg_headline/v1` | 5 | 1024 |
| General | established `mix:sa_general` six-family mix | validation loss only | 5 | 2048 |

- Classification validation already expands every frozen prompt. NER and POS
  validation must explicitly use `eval_template_choice=ALL`, so checkpoint
  selection averages all five prompts rather than a cycling subset.
- T2X uses its repository `validation` split. AfriHG uses `dev`/`validation` as
  validation; the loader's test fallback is not allowed if that split is absent.
- Task-native validation callbacks evaluate the full validation split. No
  validation sampling cap may vary between recipes.
- Early stopping patience is 2 with threshold 0.001. Save every epoch, load the
  best checkpoint at end, and retain the best adapter. Exact ties select the
  lower learning rate; checkpoint ties select the earlier epoch.

## Recipe selection

- Frozen LR grid for each task family is `3e-5`, `8e-5`, `1.5e-4`. No other
  hyperparameter is screened. `3e-4` is excluded and cancelled job `1183162`
  must not be restarted.
- A single multilingual/pool arm selects the shared task-family LR for both the
  Multilingual and Monolingual variants. This deliberately gives Mono and Multi
  the same HPO budget and prevents per-language recipe cherry-picking. Each
  Monolingual adapter is then trained once at that frozen family LR and selects
  only its own checkpoint on its own validation metric.
- The multilingual HPO winner itself becomes the frozen Multilingual adapter;
  do not retrain it. T2X has only Xhosa, so its Xhosa validation selects its LR
  and checkpoint and there is no separate Multilingual arm.
- MasakhaNews may reuse the completed pure-GDN validation-only screen:
  `1183134` (`3e-5`, macro-F1 `0.10811573554007083`), `1183160` (`8e-5`,
  `0.06244142349681912`), and `1183161` (`1.5e-4`, `0.06244142349681912`).
  Therefore the frozen News family LR is `3e-5`; no News HPO rerun is allowed.
- General uses the established token-balanced component weights
  `sib=1.6069283613929293`, `news=1.701448369209409`,
  `ner=1.0316789717478076`, `pos=0.08797885310286906`,
  `afrihg=0.8588232839160144`, `t2x=0.7131421606309706`, temperature 0,
  without min/max clipping. These weights were derived from inverse mean
  assistant-token counts, not downstream test performance. General screens only
  the same three LRs and selects the lowest validation loss; no mixture search
  is permitted. The limitation that concatenated validation loss is
  example-weighted must be reported.

## Structurally applicable sheet variants

- Monolingual: News Eng/Xho; NER Tsn/Xho/Zul; POS Tsn/Xho/Zul; SIB
  Afr/Eng/Nso/Sot/Xho/Zul; Intent Eng/Sot/Xho/Zul; T2X Xho; AfriHG Xho/Zul.
- Multilingual: News, NER, POS, SIB, Intent, and AfriHG languages above.
- No Mono/Multi result is forced for Belebele or SA-general because no matching
  task-specific training arm exists. T2X has no distinct Multilingual arm.
  AfriHG English is inapplicable because no English AfriHG task exists.
- The General adapter is evaluated once on the full frozen 16-lane matrix,
  including zero-shot transfer to Intent, Belebele, and SA-general. AfriHG
  English remains inapplicable because the task itself does not exist.

## One-time official held-out evaluation

- Phase 1 contains training and validation selection only. Historical bulk
  launchers that automatically submit held-out evaluation are not used.
- After every winning recipe and checkpoint is frozen in a manifest, phase 2
  touches each applicable official test exactly once. It reuses the base gate's
  prompt packs, scoring, BF16 model loading, and output conventions. Generation
  reuses five beams, T2X length penalty 1.0, AfriHG length penalty 0.7, early
  stopping, and the model's saved `use_cache=false`.
- All prompt values are retained. A best-prompt headline is descriptive only;
  prompt means and ranges are mandatory and cannot trigger a rerun.
- An infrastructure failure may be rerun identically only when no valid summary
  exists. OOM handling may reduce batch size without changing examples, order,
  decoding, checkpoint, or metric. Scientific/model-quality failures are final.
- Publish each verified artifact incrementally only to `Pure GatedDeltaNet
  Results` columns E/F/G and re-read the exact cells. Historical hybrid tabs are
  never edited.

## Execution order and deadline

1. Implement and dry-run the validation-only launcher; prove it cannot submit
   held-out evaluation.
2. Run multilingual family LR screens and the T2X/General screens, at most three
   A100-40GB jobs at a time.
3. Freeze winners in a manifest, then run the applicable Monolingual adapters.
4. Freeze all checkpoints, run the one-time held-out matrix, and publish verified
   rows incrementally.

All experiments and evaluations must finish by 31 August 2026. Paper drafting
starts afterward; the first advisor draft is targeted for mid-September.

