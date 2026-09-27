# Advisor Meeting Weekly Update: 2026-05-28

Prepared Wednesday night, 2026-05-27, for the Thursday advisor meeting.

Coverage window: Thursday 2026-05-21 through Wednesday 2026-05-27.

## One-Sentence Summary

This week closed most of the Mamba rescue/root-cause TODOs with concrete
negative or caveated evidence, then moved the main comparison forward to xLSTM:
the current HF pure-Mamba2 path remains weak on source-conditioned generation
and POS/NER despite several rescue attempts, while a fair strict-125M xLSTM base
is clearly better than the current Mamba base on clean likelihood audits but
still behind the LLaMA Transformer baseline.

## What Changed Since Last Meeting

### 1. Mamba T2X weakness is now much more concrete

T2X is not only "low chrF". The current evidence points to source/entity/value
preservation and relation verbalisation failures.

Key diagnostics:

| Diagnostic | Result | Interpretation |
|---|---:|---|
| B1 teacher-forced token-class audit | Entity NLL gap `+2.7656`, PPL ratio `15.89x`; value NLL gap `+2.3065`, PPL ratio `10.04x`; other-token PPL ratio `3.25x` | Mamba is disproportionately worse on exact source entity/value tokens, not just all language modelling. |
| B2 placeholder/reinsertion | Mamba generated zero expected placeholders in `216` examples where placeholders were expected; reinserted chrF `8.2455` | Removing the direct copy burden did not rescue Mamba. It also failed to follow placeholder abstractions. |
| B3 source-preservation decoding | Mamba baseline chrF `28.064`; source-checklist chrF `28.148`; repetition-control chrF `20.141` / `21.737` | Prompt/checklist and repetition controls do not fix the gap. |
| B3 Mamba vs LLaMA preservation | Mamba entity coverage `0.410`, value coverage `0.320`; LLaMA entity coverage `0.763`, value coverage `0.594` | The same evaluation path shows LLaMA preserves source information much better. |

Concrete example to discuss:

| Field | Text |
|---|---|
| Triple | `Abilene Regional Airport` / `cityServed` / `Abilene, Texas` |
| Mamba | `Abasebenzi beAbakwa-Abilene, e-Abilene, eAbilene.` |
| LLaMA | `Abilene Regional Airport ifumaneka e Abilene, Texas.` |
| Reference | `Isikhululo seenqwelo moya sase-Abilene sisebenzela isixeko sase-Abilene eTexas.` |
| Read | Mamba repeats Abilene-like fragments and loses the airport/relation structure. LLaMA is not perfect, but it preserves the entity/value much better. |

Another useful example:

- `Afonso Pena International Airport / cityServed / Curitiba` led Mamba toward
  `iCuritiba yeCuritiba yiCuritiba`, preserving the value but collapsing into
  repetition instead of verbalising the relation.

Meeting interpretation:

- T2X looks like a real source-conditioned generation/copying/planning problem.
- It is not solved by longer decoding, source-checklist prompting, repetition
  penalties, or placeholder reinsertion.

### 2. AfriHG has one real Mamba decode improvement, but focus remains weak

C1 official Mamba AfriHG beam5/lp1.2 was positive:

| Task | Prior Mamba checkpoint-selected | C1 beam5/lp1.2 | Delta |
|---|---:|---:|---:|
| AfriHG Xho chrF | `11.5335` | `15.1702` | `+3.6367` |
| AfriHG Zul chrF | `13.5726` | `17.1245` | `+3.5519` |

But the matched LLaMA C1b check showed this is not a shared protocol
improvement:

- LLaMA Xho beam5/lp1.2: chrF `13.1149`, below tracked LLaMA parity `14.2017`.
- LLaMA Zul beam5/lp1.2: chrF `20.8709`, below tracked LLaMA parity `21.5552`
  and older best around `23.0041`.
- Carry-forward tag: `Mamba-only`.

C2 AfriHG forensics showed the residual Mamba errors are mostly focus/content
selection problems:

| Language | Main residual errors |
|---|---|
| Xho | Off-topic/generic `37.2%`; too-short/generic `31.9%` |
| Zul | Off-topic/generic `54.7%`; too-short/generic `14.9%` |

Concrete example:

| Article hook | Mamba | LLaMA | Reference | Read |
|---|---|---|---|---|
| Proteas batting issues and AB de Villiers absence after England series | `UMzantsi Afrika ifikile` | `UDe Villiers ubethe iNgilane ngo'3-1` | `Ingxubakaxaka ngo-AB de Villiers` | Mamba is grammatical-ish but generic; it misses the AB de Villiers hook. |

C3 focus-prompt validation looked promising, but C4 official test confirmation
was ambiguous-to-negative:

| Task | C1 beam5/lp1.2 | C4 focus_v1 official test | Read |
|---|---:|---:|---|
| AfriHG Xho chrF | `15.1702` | `14.4983` | regressed |
| AfriHG Zul chrF | `17.1245` | `17.3358` | tiny improvement only |

Meeting interpretation:

- Keep C1 beam5/lp1.2 as the current Mamba AfriHG recipe.
- Do not adopt `focus_v1` as final. Treat it as a validation-only or
  language-specific idea.
- AfriHG is partially decode-rescuable, unlike T2X, but the remaining issue is
  semantic focus/headline planning.

### 3. POS/NER decoder-only rescue is now mostly closed negative

Earlier constrained-output experiments showed that output shape alone was not
the whole problem:

- A2 prefix-constrained generation forced exact-length, parseable POS/NER
  outputs (`1.0` parseable), but Mamba POS collapsed mostly to `cconj`, and
  Mamba NER flooded labels like `b-date`, `i-date`, and `i-org`.
- LLaMA controls were also weak under the same decoder-only formulation.

D4 tested whether more task data and the best atomic tag-sequence recipes could
rescue Mamba POS/NER. It closed negative.

| D4 run | Validation result | Interpretation |
|---|---:|---|
| POS `lr3e-4/e80` | token accuracy `0.0970`, exact length `0.0333` | poor |
| POS `lr2e-4/e120` | token accuracy `0.1960`, exact length `0.0400`, overgeneration `1.1486` | best D4 POS, still below prior atomic canary and shape-poor |
| NER `lr1e-4/e60` | token accuracy `0.4252`, non-`O` recall `0.1772`, BIO F1 `0.1769`, all-`O` rate `0.9375` | reproduces weak old signal, does not materially improve |

Concrete NER example from the prior inspection:

| Target | Raw Mamba output | Read |
|---|---|---|
| Text mentions `uShauwn`; target should include `PER: uShauwn` | `hofoza ... DATE PER MONEY` | Emits stray labels/words rather than the required entity line; parser returns empty. |

Meeting interpretation:

- For the current pure decoder-only Mamba path, POS/NER is not rescued by
  format constraints, label prior bias, constrained tag generation, or full-data
  atomic tag-sequence LoRA.
- Still not done: the English POS/NER control requested in the meeting notes.
  We have Xho/Zul/Tsn-style evidence, but not the clean English-data control.

### 4. Pure-Mamba base rescue screens are negative

The week added more base-rescue evidence against cheap HF pure-Mamba fixes.

| Run | Result |
|---|---|
| D1 wide pure-Mamba2 20k with torch eval repair | Clean-loss PPLs: T2X `8470`, AfriHG Xho `11315`, AfriHG Zul `11720`; far worse than current Mamba base `348`/`439`/`606`. |
| D2 current-shape lower-LR/warmup 10k | Clean-loss PPLs: T2X `11925.1`, AfriHG Xho `26758.1`, AfriHG Zul `28854.2`; negative. |
| D2 wide lower-LR/warmup 10k | Best wide checkpoint PPLs: T2X `9983.6`, AfriHG Xho `21768.7`, AfriHG Zul `20378.4`; negative. |
| D5 shallow Mamba-2 hybrid screen | Held-out eval improved by checkpoint-10000, but clean-loss PPLs were still T2X `12364.7`, AfriHG Xho `14020.5`, AfriHG Zul `14634.8`; negative as a base candidate. |

Meeting interpretation:

- Cheap fresh-base rescue is not working.
- Do not claim "Mamba is fully optimized and failed." The stronger claim is:
  cheap HF-Mamba rescue arms, decoder-only task formulation fixes, and shallow
  hybrid screening failed to produce a credible base candidate.
- A serious Mamba follow-up would need either official `mamba_ssm` path
  validation or a longer current-base continued-pretraining/task-shaped branch,
  not another small HF-Mamba 10k screen.

### 5. Implementation caveats became clearer

Mamba caveats:

- E1 HF-vs-official Mamba logits parity failed.
- Parameter counts matched and state load had no missing/unexpected keys, but
  logits were not close: max absolute logit deltas were about `9.31`-`16.01`,
  and greedy next-token choices differed on two of three prompts.
- This does not invalidate HF-Mamba vs HF-LLaMA comparisons, because the
  experiments are on the HF path, but it weakens broad claims about official
  Mamba2 as an architecture family.

Generation hygiene caveat:

- E2 found Mamba generation is not perfectly invariant to cache/batch settings.
- T2X exact match to batch1/cache-off was `7/8` under cache-on or batch4
  variants; AfriHG Xho was `6/8` to `7/8`.
- Final Mamba generation reporting should pin and document cache/batch settings.

xLSTM caveats:

- Stock HF xLSTM generation/eval hits chunk-size assumptions unless padded to
  multiples of `64`.
- The evaluation path now has xLSTM-specific padding and chunked generation
  repairs, but this should be documented as part of the architecture comparison.

### 6. xLSTM became the strongest new architecture signal

The strict-125M xLSTM path is now viable after installing the `xlstm` stack and
adding chunked eval/generation handling.

Strict full-base run:

- Run id: `xlstm_h736_ctx2048_native_4gpu_ddp_llama_budget_20260524`.
- Architecture: HF xLSTM `h736_l12_h4_chunk64`.
- Parameters: `126,901,952`.
- Training job: `861849`, completed in `1-05:59:05`.
- Effective token-slot budget: `4,758,208,512`.
- Final reported epoch: `2.1513`.
- Trainer loss logs are inflated by a Transformers/xLSTM 4-GPU DDP loss-scaling
  issue, so clean audits should be used instead of raw Trainer losses.

Clean generation-loss audit:

| Model | T2X Xho NLL/PPL | AfriHG Xho NLL/PPL | AfriHG Zul NLL/PPL |
|---|---:|---:|---:|
| xLSTM final | `5.6584` / `286.70` | `5.7048` / `300.30` | `5.8647` / `352.39` |
| Current Mamba base | `5.8515` / `347.77` | `6.0834` / `438.50` | `6.4066` / `605.85` |
| LLaMA base | `4.1620` / `64.20` | `5.3254` / `205.48` | `5.5885` / `267.32` |

Repaired pretrain-loss audit:

| Model | Weighted NLL/PPL |
|---|---:|
| xLSTM final | `2.9250` / `18.63` |
| Mamba base | `3.8065` / `44.99` |
| LLaMA base | `2.3010` / `9.98` |
| xLSTM checkpoint-30000 probe | `2.6888` / `14.71` |

Meeting interpretation:

- xLSTM is positive versus current Mamba base and negative versus LLaMA base.
- This is the first non-Transformer alternative this week that looks
  promising enough for downstream fine-tuning/evaluation.
- It does not yet beat LLaMA, so the thesis story is not "xLSTM wins"; it is
  "xLSTM is a credible recurrent/efficient architecture candidate, while the
  current HF-Mamba path is weak/fragile."

### 7. xLSTM downstream evaluation is in progress, but not complete

Prepared xLSTM downstream matrix:

- Added missing xLSTM eval configs for `xlstm_afrihg_all`,
  `xlstm_masakhaner_all`, `xlstm_masakhapos_all`, and
  `xlstm_sa_general_all`.
- Added phased submitter for base, monolingual, multilingual, and
  general-instruction phases.
- Validated references to `40` eval configs and `28` fine-tune configs.
- Uploaded fresh xLSTM full-base checkpoint to private HF repo
  `anrilombard/sallm-xlstm-125m`, commit
  `ec4166b4d0fee57e33c6893ebf808d9d3342ac03`.

Base xLSTM downstream read:

- `39` base evaluation summaries were pulled; `afrihg_eng` is invalid/skipped.
- Base xLSTM is not downstream-strong without fine-tuning:
  - Belebele roughly `0.23`-`0.25` acc_norm by language.
  - AfriXNLI average around `0.334` accuracy.
  - AfrMMLU around `0.232` accuracy.
  - AfriMGSM exact match around `0.005`.
  - MasakhaNER F1 `0.0` across `tsn/xho/zul/all`.
  - MasakhaPOS token accuracy `0.0` across `tsn/xho/zul/all`.
  - AfriHG BLEU `0.0`; T2X BLEU about `0.0003`.

Current operational blocker:

- Base phase needed two repair passes:
  - first for xLSTM chunk-size assertion in forward/loglikelihood;
  - then for generation path chunking.
- Base retry1 eventually completed cleanly (`867562`-`867572`, all `0:0`);
  the long-pole `masakhaner_all` job `867570` took `12:47:13`.
- Mono phase first failed systemically because Hydra rejected
  `finetune.training.save_total_limit=1`; it needed
  `+finetune.training.save_total_limit=1`.
- The submitter was patched and a repaired partial mono wave was submitted
  (`868578`-`868606`), but further submissions hit
  `QOSMaxSubmitJobPerUserLimit`.
- As of Wednesday evening, VPN/HEX connectivity is blocked
  (`Cisco Secure Client` disconnected), so repaired mono job state is
  unverified.

Meeting interpretation:

- xLSTM downstream is underway but not yet a result.
- The base model alone is weak downstream, which is expected for many tasks.
  The real comparison needs the monolingual/multilingual/general fine-tuned
  phases.
- Runtime fragility is now mostly engineering/config, not a scientific
  negative result for xLSTM.

## Specific Advisor Discussion Points

1. Mamba claim wording:
   - Recommended wording: "Our current HF pure-Mamba2 SALLM path is weak and
     fragile for source-conditioned generation and POS/NER under these
     decoder-only formulations."
   - Avoid: "Mamba as an architecture is bad."
   - Reason: E1 showed HF-vs-official parity failure, and the official
     `mamba_ssm` codepath has not yet been trained/evaluated.

2. Whether Mamba is exhausted:
   - Cheap HF-Mamba rescue is exhausted enough for now.
   - A serious Mamba branch would be different: official-codepath canary,
     current-base continued pretraining, task-shaped continued pretraining, or
     a much more deliberate 125M recipe/shape run.
   - Recommended next step is not another cheap HF-Mamba screen.

3. Whether to continue xLSTM as the mainline:
   - Yes, unless advisors think the thesis must first close the official-Mamba
     confounder.
   - xLSTM gives a cleaner architecture-comparison signal now and already beats
     the current Mamba base on clean likelihood audits.

4. Whether to add HGRN2/BabyHGRN later:
   - HGRN2/BabyHGRN is the best next candidate if compute/time allow another
     architecture slot.
   - Rationale from the literature pass: it is directly framed around
     sample-efficient low-resource language modelling and compares against
     Transformer, LSTM, xLSTM, and Mamba.

5. What to do about the remaining POS/NER English-control TODO:
   - Still open. The notes now close several Xho/Zul/Tsn-style POS/NER rescue
     attempts, but the English POS/NER control has not been run.
   - Ask whether it is necessary for the next advisor-facing claim, or whether
     the current matched Mamba/LLaMA constrained/tag-sequence evidence is enough
     to move on.

## Open Questions For Advisors

- Is the current Mamba evidence sufficient to pause HF-Mamba rescue and proceed
  with xLSTM downstream, while keeping official Mamba as a caveat/future
  control?
- Should the official `mamba_ssm` control be a small viability canary only, or
  does the thesis need a serious official-codepath base run before final Mamba
  claims?
- For T2X, do the B1/B2/B3 results justify saying the problem is
  source-conditioned preservation/planning rather than just decoding?
- For AfriHG, should we keep the Mamba-only beam5/lp1.2 recipe despite it not
  transferring to LLaMA?
- For POS/NER, should we spend on the English control, classifier-head/CRF-style
  formulation, or just report decoder-only sequence labelling as weak for this
  setup?
- If xLSTM downstream results are competitive, should the next architecture be
  HGRN2/BabyHGRN, or should effort go back into official Mamba?

## Recommended Next Work

1. Restore VPN/HEX access and finish the repaired xLSTM mono phase.
2. Continue xLSTM downstream phases in order: mono -> multi -> general, only
   advancing when the current phase is terminal and no systemic failure appears.
3. Pull and summarize xLSTM downstream results into the same sheet structure as
   LLaMA/Mamba: base, monolingual, multilingual, general.
4. Keep Mamba official-codepath work bounded: instantiate/save-load/tiny-overfit
   canary first, not a blind full training run.
5. Decide with advisors whether the English POS/NER control is required before
   writing the POS/NER conclusion.

## Files To Point To

- Main progress tracker: `sallm_memory/sallm_progress.md`
- Prior advisor note:
  `sallm_memory/notes/2026-05-21-advisor-meeting-mamba-classification-generation-base.md`
- Supervisor TODO note:
  `sallm_memory/notes/2026-05-21-supervisor-todos-mamba-output-forensics.md`
- D2/D4/xLSTM daily evidence:
  `sallm_memory/notes/2026-05-24.md`,
  `sallm_memory/notes/2026-05-25.md`,
  `sallm_memory/notes/2026-05-26.md`,
  `sallm_memory/notes/2026-05-27.md`
- xLSTM literature notes:
  `sallm_memory/xlstm_architecture_literature.md`
- Architecture candidate review:
  `sallm_memory/architecture_candidate_review_2026-05-25.md`
- Current evidence matrix:
  `sallm_memory/mamba_root_cause_rescue_evidence_matrix.md`
