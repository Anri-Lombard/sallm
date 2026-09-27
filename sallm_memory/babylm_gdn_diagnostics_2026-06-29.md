# BabyLM GDN diagnostics - 2026-06-29

## Scope

Compared the completed fast-eval runs:

| Run | BLiMP | Supplement | EWoK | Entity | Eye / self-paced |
|---|---:|---:|---:|---:|---:|
| GPT-2 baseline | 65.35 | 59.60 | 49.09 | 21.68 | 9.63 / 1.64 |
| GDN 12L seq256 | 66.12 | 57.20 | 52.27 | 15.55 | 0.71 / 0.00 |
| GDN 12L seq512 | 65.63 | 58.00 | 47.64 | 11.87 | 0.36 / 0.10 |
| GDN 24L seq256 | 69.17 | 54.40 | 48.27 | 16.46 | 0.36 / 0.00 |

Source files inspected:

- `/private/tmp/babylm_gdn_results`
- `/private/tmp/babylm_fast_eval_data`
- `/scratch/alombard/babylm-gdn/repos/babylm-eval/strict/evaluation_pipeline/sentence_zero_shot/compute_results.py`
- `/scratch/alombard/babylm-gdn/repos/babylm-eval/strict/evaluation_pipeline/reading/run.py`

## Root-cause findings

### 1. Context length is not the problem

The matched seq512 run is worse than the original 12L seq256 run on BLiMP, EWoK, entity tracking, and eye-tracking score. Do not spend more runs on longer context until another diagnostic points there.

### 2. Depth mainly buys BLiMP, not broad BabyLM performance

GDN 24L seq256 improves BLiMP by +3.82 over GPT-2 and +3.05 over GDN 12L seq256.

The biggest GDN 24L BLiMP gains over GPT-2 are:

| Category | GDN 24L - GPT-2 |
|---|---:|
| quantifiers | +13.50 |
| npi_licensing | +10.93 |
| island_effects | +7.56 |
| determiner_noun_agreement | +5.38 |
| filler_gap_dependency | +5.28 |

The biggest BLiMP losses are:

| Category | GDN 24L - GPT-2 |
|---|---:|
| irregular_forms | -12.50 |
| s-selection | -3.25 |
| anaphor_agreement | -2.50 |

Interpretation: depth helps structured syntactic/semantic acceptability judgments, especially NPI/quantifier/island/filler-gap cases. It does not generally improve lexical memorization or commonsense.

### 3. Entity tracking failure is mostly an empty-box / `nothing` calibration issue

Entity official scores are low for all models, but the GDN deficit is concentrated on examples where the correct answer is `nothing.`.

Item-weighted diagnostic split:

| Subset | n | GPT-2 | GDN 12L seq256 | GDN 12L seq512 | GDN 24L seq256 |
|---|---:|---:|---:|---:|---:|
| all entity examples | 3152 | 21.10 | 15.13 | 12.66 | 16.37 |
| gold answer is `nothing.` | 914 | 26.91 | 4.92 | 0.11 | 13.89 |
| gold answer is non-empty | 2238 | 18.72 | 19.30 | 17.78 | 17.38 |

Predicted empty-answer counts:

| Run | Predicted `nothing.` | Gold empty examples |
|---|---:|---:|
| GPT-2 | 246 | 914 |
| GDN 12L seq256 | 45 | 914 |
| GDN 12L seq512 | 1 | 914 |
| GDN 24L seq256 | 127 | 914 |

The evaluator ranks candidate completions by summed completion log probability and exact-matches the selected option against `options[0]`. So this is not output-format failure, and it is not caused by length normalization punishing `nothing.`. GDN is assigning too little probability to the empty-box answer after prompts like `Box 6 contains ...`.

Interpretation: entity tracking is not primarily a long-context state-update failure. On non-empty boxes, GDN 12L seq256 is slightly ahead of GPT-2. The big fix target is the no-object / empty-state prior.

### 4. Supplement weakness is lexical/commonsense, not syntax

GDNs do well on subject-aux inversion but trail GPT-2 on hypernym and QA congruence:

| Supplement UID | GPT-2 | GDN 12L seq256 | GDN 12L seq512 | GDN 24L seq256 |
|---|---:|---:|---:|---:|
| subject_aux_inversion | 80.0 | 94.0 | 92.0 | 84.0 |
| hypernym | 58.0 | 50.0 | 48.0 | 46.0 |
| qa_congruence_easy | 58.0 | 52.0 | 50.0 | 48.0 |
| qa_congruence_tricky | 36.0 | 30.0 | 32.0 | 32.0 |
| turn_taking | 66.0 | 60.0 | 68.0 | 62.0 |

Interpretation: the current GDN recipe can learn syntactic form, but struggles with lexical entailment and discourse/QA plausibility.

### 5. EWoK prefers the smaller 12L seq256 recipe

GDN 12L seq256 beats GPT-2 on EWoK (+3.18), especially spatial relations, physical dynamics, physical relations/interactions, social interactions, and concept-swap contrasts.

GDN 24L seq256 loses most of that advantage. The biggest 24L-vs-12L EWoK drops are active-passive contrast, social-properties, physical-interactions, social-interactions, direct context, and negation/antonym contrasts.

Interpretation: depth is not a free improvement. It improves BLiMP while damaging the best EWoK signal.

### 6. Reading score is weak after baseline controls

Raw reading correlations are not zero, but BabyLM's headline reading score comes from incremental predictive power after controls for frequency, length, and context length.

Normalized eye-tracking contribution:

| Variable | GPT-2 | GDN 12L seq256 | GDN 12L seq512 | GDN 24L seq256 |
|---|---:|---:|---:|---:|
| RTfirstfix | 2.39 | 0.15 | 0.06 | 0.00 |
| RTfirstpass | 4.55 | 0.35 | 0.05 | 0.01 |
| RTgopast | 23.72 | 1.64 | 1.01 | 1.20 |
| RTrightbound | 7.85 | 0.69 | 0.33 | 0.22 |
| mean eye score | 9.63 | 0.71 | 0.36 | 0.36 |

Interpretation: GDN surprisal is only weakly useful beyond simple lexical/context controls. This looks like a training/objective or calibration issue, not a context-length issue.

## Recommendation

Do not make architectural changes yet.

Run the next work in this order:

1. Eval-only score audit for entity `nothing.` examples. Add score export or a tiny scorer so we can see whether `nothing.` is consistently low-margin or catastrophically low-probability.
2. One training recipe run, not a sweep: GDN 12L seq256 with more frequent full attention, e.g. `FULL_ATTENTION_EVERY=2`, same token budget. This tests whether retrieval improves without losing the 12L EWoK signal.
3. Only if the score audit says the empty-state problem is trainable rather than pure eval prior: try a lower-LR 24L run to keep BLiMP gains while reducing supplement/EWoK damage.

Skipped for now:

- More seq512 runs.
- Larger depth-only runs.
- New architecture variants.
- Full HPO.

Paper angle if this holds: GDN is interesting for efficient syntax/acceptability under BabyLM, but its current causal scoring recipe has clear weaknesses on empty-state entity tracking, lexical entailment, and human-reading predictive power.

## Follow-up score audit: entity `nothing.` margins

Ran `/scratch/alombard/babylm-gdn/scripts/babylm_entity_score_audit.py` on Kombuys GPU1. Output:

- Remote JSON: `/scratch/alombard/babylm-gdn/diagnostics/entity_nothing_score_audit_20260629.json`
- Local JSON copy: `/private/tmp/entity_nothing_score_audit_20260629.json`
- Local script: `scripts/babylm_entity_score_audit.py`

The audit reuses BabyLM's entity-tracking tokenizer/collate logic and dumps summed completion log-probability margins for each option.

### Margin summary

| Run | Entity acc | Empty acc | Non-empty acc | Predicted empty | Empty gold margin median | Empty close-miss rate | Empty catastrophic-miss rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| GPT-2 | 21.10 | 26.91 | 18.72 | 246 | -1.45 | 15.86 | 5.47 |
| GDN 12L seq256 | 15.13 | 4.92 | 19.30 | 45 | -1.67 | 18.60 | 0.22 |
| GDN 12L seq512 | 12.66 | 0.11 | 17.78 | 1 | -3.16 | 1.64 | 4.27 |
| GDN 24L seq256 | 16.37 | 13.89 | 17.38 | 127 | -1.24 | 28.45 | 2.41 |

Definitions:

- `Empty acc`: accuracy when the gold answer is `nothing.`
- `Predicted empty`: number of examples where the selected option was `nothing.`
- `Empty gold margin median`: median score gap between the gold `nothing.` option and the best non-empty distractor on gold-empty examples.
- `Empty close-miss rate`: wrong gold-empty examples where margin is above -1.0.
- `Empty catastrophic-miss rate`: wrong gold-empty examples where margin is below -5.0.

Interpretation:

- GDN 12L seq256 is not catastrophically bad on `nothing.`. It is usually close but consistently below the best non-empty distractor.
- GDN 24L is even more suggestive of a calibration problem: empty median margin is only -1.24 and 28.45% of wrong empty examples are within 1 log-prob point.
- Seq512 is the bad outlier. It nearly never selects `nothing.` and has a much worse empty median margin.

### Post-hoc calibration simulation

Adding a constant boost to the `nothing.` option is not a fair BabyLM system, but it is a useful diagnostic. In this fast-eval set, `nothing.` appears only when it is the gold option, so boosting it does not reduce non-empty accuracy.

| Run | Raw acc | Acc with +1 empty boost | Acc with +2 empty boost | Acc with +3 empty boost |
|---|---:|---:|---:|---:|
| GPT-2 | 21.10 | 25.70 | 29.70 | 33.82 |
| GDN 12L seq256 | 15.13 | 20.53 | 31.88 | 39.09 |
| GDN 12L seq512 | 12.66 | 13.13 | 17.23 | 25.67 |
| GDN 24L seq256 | 16.37 | 24.62 | 32.71 | 37.60 |

Decision:

- Skip the `FULL_ATTENTION_EVERY=2` run for now. The score audit weakens the retrieval-attention hypothesis.
- The immediate root cause is an empty-answer prior/calibration issue, not context length and not general non-empty state retrieval.
- Do not submit or optimize with an eval-specific `nothing.` boost. Treat it as a diagnostic showing what kind of training/objective/calibration weakness to investigate.

## Targeted empty-state emphasis run

Added a corpus-audit script and a training knob to repeat BabyLM training documents containing empty-state language.

Local scripts:

- `scripts/babylm_empty_state_corpus_audit.py`
- `scripts/babylm_gdn_smoke_train.py`
- `scripts/run_babylm_gdn_pilot_kombuys.sh`

Remote outputs:

- Corpus audit: `/scratch/alombard/babylm-gdn/diagnostics/empty_state_corpus_audit_20260629.json`
- Run log: `/scratch/alombard/babylm-gdn/logs/gdn-h512l12-s256-emptyrep2-strict-small-20260629.log`
- Checkpoint: `/scratch/alombard/babylm-gdn/outputs/gdn-h512l12-s256-emptyrep2-strict-small-20260629`
- Entity audit: `/scratch/alombard/babylm-gdn/diagnostics/entity_nothing_score_audit_emptyrep2_20260629.json`

Corpus audit using regex `\b(nothing|empty|emptied|none|nobody|no one|without)\b`:

| Measure | Value |
|---|---:|
| documents | 1,104,106 |
| words | 9,956,033 |
| matched documents | 10,900 |
| matched words | 564,724 |
| matches | 11,785 |
| matched document rate | 0.987% |
| matched word rate | 5.672% |
| matches per million words | 1,183.70 |

The run used `EMPTY_STATE_EMPHASIS_REPEAT=2`, which repeated the 10,900 matching documents two extra times. Packed training metadata reported 1,125,906 emitted texts.

Fast eval comparison:

| Run | BLiMP | Supplement | EWoK | Entity | Eye / self-paced |
|---|---:|---:|---:|---:|---:|
| GDN 12L seq256 | 66.12 | 57.20 | 52.27 | 15.55 | 0.71 / 0.00 |
| Empty-state repeat-2 | 66.99 | 58.00 | 48.73 | 16.80 | 0.38 / 0.01 |
| GPT-2 baseline | 65.35 | 59.60 | 49.09 | 21.68 | 9.63 / 1.64 |

Entity score-audit comparison:

| Run | Entity acc | Empty acc | Non-empty acc | Predicted empty rate |
|---|---:|---:|---:|---:|
| GPT-2 | 21.10 | 26.91 | 18.72 | 7.80 |
| GDN 12L seq256 | 15.13 | 4.92 | 19.30 | 1.43 |
| Empty-state repeat-2 | 17.23 | 16.96 | 17.34 | 4.92 |

Interpretation:

- The root-cause diagnosis was real: GDN underpredicted `nothing.`, and natural empty-state emphasis substantially improved the empty subset.
- The fix is too blunt by itself: non-empty entity accuracy dropped from 19.30 to 17.34, EWoK dropped from 52.27 to 48.73, and reading remained weak.
- The next method should not be "repeat more empty documents" as the main paper contribution. Use the result as evidence that GDN has an empty-state calibration/representation weakness under BabyLM causal scoring.

Updated decision:

- Keep the empty-state emphasis knob for diagnostics, but do not submit this run as the final recipe.
- If we continue BabyLM, the next credible fix is a controlled objective/architecture probe: auxiliary state-change supervision, contrastive empty-vs-object calibration, or an architecture-matched baseline that tests whether delta/gating state decay causes empty-state under-selection.
- A BabyLM paper remains plausible only if framed as a diagnostic study of efficient recurrent/linear architectures: strong BLiMP syntax signal, decent EWoK in the smaller recipe, and a clear failure mode on entity empty-state calibration and reading surprisal.

## Synthetic entity-state probe

Added a minimal opt-in synthetic entity-state generator to `scripts/babylm_gdn_smoke_train.py` and launcher env knobs in `scripts/run_babylm_gdn_pilot_kombuys.sh`.

Run:

- Name: `gdn-h512l12-s256-synthentity10k-strict-small-20260629`
- Config: GDN 12L, hidden 512, seq256, 6500 steps, batch 8, packed Strict-Small, no empty-state corpus repeat.
- Synthetic examples: 10,000 short `Box ... contains ...` state examples, 50% empty, appended to the training stream.
- Checkpoint: `/scratch/alombard/babylm-gdn/outputs/gdn-h512l12-s256-synthentity10k-strict-small-20260629`
- Log: `/scratch/alombard/babylm-gdn/logs/gdn-h512l12-s256-synthentity10k-strict-small-20260629.log`
- Entity audit: `/scratch/alombard/babylm-gdn/diagnostics/entity_nothing_score_audit_synthentity10k_20260629.json`

Fast eval:

| Run | BLiMP | Supplement | EWoK | Entity | Eye / self-paced |
|---|---:|---:|---:|---:|---:|
| GDN 12L seq256 | 66.12 | 57.20 | 52.27 | 15.55 | 0.71 / 0.00 |
| Empty-state repeat-2 | 66.99 | 58.00 | 48.73 | 16.80 | 0.38 / 0.01 |
| Synthetic entity 10k | 66.72 | 56.00 | 50.82 | 33.17 | 0.11 / 0.04 |
| GPT-2 baseline | 65.35 | 59.60 | 49.09 | 21.68 | 9.63 / 1.64 |

Entity score-audit split:

| Run | Entity acc | Empty acc | Non-empty acc | Predicted empty rate | Empty median margin |
|---|---:|---:|---:|---:|---:|
| GPT-2 | 21.10 | 26.91 | 18.72 | 7.80 | -1.45 |
| GDN 12L seq256 | 15.13 | 4.92 | 19.30 | 1.43 | -1.67 |
| Empty-state repeat-2 | 17.23 | 16.96 | 17.34 | 4.92 | -2.05 |
| Synthetic entity 10k | 31.98 | 61.38 | 19.97 | 17.80 | 1.21 |

Interpretation:

- This is a much better causal test than corpus oversampling: the empty-state failure is fixable without sacrificing non-empty entity tracking.
- The method is too close to the eval task format to call it a final BabyLM recipe without careful framing, but it is strong diagnostic evidence for a state-update/calibration failure rather than a broad GDN incapacity.
- The broader BabyLM tradeoff remains: Entity becomes excellent, BLiMP stays above GPT-2, EWoK stays above GPT-2 but below the original GDN12, Supplement falls, and reading remains poor.

Next lazy experiment:

- One ablation only: synthetic non-empty-only or lower synthetic count, to check whether the win comes from explicit empty-state examples, entity-task format exposure, or both. Do not launch a broad sweep yet.

## Synthetic entity-state ablation: non-empty only

Run:

- Name: `gdn-h512l12-s256-synthentity10k-nonempty-strict-small-20260629`
- Config: same as the synthetic entity 10k run, but `SYNTHETIC_ENTITY_EMPTY_RATE=0.0`.
- Checkpoint: `/scratch/alombard/babylm-gdn/outputs/gdn-h512l12-s256-synthentity10k-nonempty-strict-small-20260629`
- Log: `/scratch/alombard/babylm-gdn/logs/gdn-h512l12-s256-synthentity10k-nonempty-strict-small-20260629.log`
- Entity audit: `/scratch/alombard/babylm-gdn/diagnostics/entity_nothing_score_audit_synthentity10k_nonempty_20260629.json`

Fast eval:

| Run | BLiMP | Supplement | EWoK | Entity | Eye / self-paced |
|---|---:|---:|---:|---:|---:|
| GDN 12L seq256 | 66.12 | 57.20 | 52.27 | 15.55 | 0.71 / 0.00 |
| Synthetic entity 10k, 50% empty | 66.72 | 56.00 | 50.82 | 33.17 | 0.11 / 0.04 |
| Synthetic entity 10k, non-empty only | 65.66 | 54.40 | 48.64 | 13.54 | 0.77 / 0.05 |

Entity score-audit split:

| Run | Entity acc | Empty acc | Non-empty acc | Predicted empty rate | Empty median margin |
|---|---:|---:|---:|---:|---:|
| GDN 12L seq256 | 15.13 | 4.92 | 19.30 | 1.43 | -1.67 |
| Synthetic entity 10k, 50% empty | 31.98 | 61.38 | 19.97 | 17.80 | 1.21 |
| Synthetic entity 10k, non-empty only | 13.64 | 0.00 | 19.21 | 0.00 | -9.26 |

Interpretation:

- The improvement is not generic entity-format exposure. Non-empty-only examples preserved non-empty accuracy but made empty-state selection disappear entirely.
- The result strengthens the specific empty-state calibration hypothesis: GDN needs explicit no-object/state-deletion evidence, not just more box/object examples.
- For a paper, this is now a clean causal ablation: baseline failure, natural-language oversampling weak fix, explicit empty-state synthetic supervision strong fix, non-empty synthetic control no fix.

## Synthetic entity-state dose probe: 2k examples

Run:

- Name: `gdn-h512l12-s256-synthentity2k-strict-small-20260630`
- Config: GDN 12L, hidden 512, seq256, 6500 steps, batch 8, packed Strict-Small, no empty-state corpus repeat.
- Synthetic examples: 2,000 short `Box ... contains ...` state examples, 50% empty, appended to the training stream.
- Checkpoint: `/scratch/alombard/babylm-gdn/outputs/gdn-h512l12-s256-synthentity2k-strict-small-20260630`
- Log: `/scratch/alombard/babylm-gdn/logs/gdn-h512l12-s256-synthentity2k-strict-small-20260630.log`
- Entity audit: `/scratch/alombard/babylm-gdn/diagnostics/entity_nothing_score_audit_synthentity2k_20260630.json`

Fast eval:

| Run | BLiMP | Supplement | EWoK | Entity | Eye / self-paced |
|---|---:|---:|---:|---:|---:|
| GDN 12L seq256 | 66.12 | 57.20 | 52.27 | 15.55 | 0.71 / 0.00 |
| Synthetic entity 10k, 50% empty | 66.72 | 56.00 | 50.82 | 33.17 | 0.11 / 0.04 |
| Synthetic entity 10k, non-empty only | 65.66 | 54.40 | 48.64 | 13.54 | 0.77 / 0.05 |
| Synthetic entity 2k, 50% empty | 67.14 | 55.20 | 50.55 | 44.15 | 0.83 / 0.01 |

Entity score-audit split:

| Run | Entity acc | Empty acc | Non-empty acc | Predicted empty rate | Empty median margin |
|---|---:|---:|---:|---:|---:|
| GDN 12L seq256 | 15.13 | 4.92 | 19.30 | 1.43 | -1.67 |
| Synthetic entity 10k, 50% empty | 31.98 | 61.38 | 19.97 | 17.80 | 1.21 |
| Synthetic entity 10k, non-empty only | 13.64 | 0.00 | 19.21 | 0.00 | -9.26 |
| Synthetic entity 2k, 50% empty | 41.91 | 99.45 | 18.41 | 28.84 | 5.47 |

Interpretation:

- The 2k dose is the strongest pilot result so far: it improves the fast Entity score to 44.15 while keeping BLiMP above the original GDN12 and GPT-2 baselines.
- The gain is almost entirely from gold-empty examples. Non-empty entity accuracy remains roughly flat or slightly lower than the original GDN12.
- This clears the paper-gate for continuing, but it should not be framed as a clean leaderboard recipe yet. The result is deliberately eval-shaped and now risks overcorrecting toward `nothing.`.
- The next defensible step is a repeat seed or an out-of-template entity probe, not a larger synthetic sweep.

### Repeat seed and out-of-template probe

Repeat run:

- Name: `gdn-h512l12-s256-synthentity2k-seed29-strict-small-20260630`
- Config: same as the 2k run, but `--seed 29`.
- Checkpoint: `/scratch/alombard/babylm-gdn/outputs/gdn-h512l12-s256-synthentity2k-seed29-strict-small-20260630`
- Log: `/scratch/alombard/babylm-gdn/logs/gdn-h512l12-s256-synthentity2k-seed29-strict-small-20260630.log`
- Entity audit: `/scratch/alombard/babylm-gdn/diagnostics/entity_nothing_score_audit_synthentity2k_seed29_20260630.json`

Fast eval:

| Run | BLiMP | Supplement | EWoK | Entity | Eye / self-paced |
|---|---:|---:|---:|---:|---:|
| Synthetic entity 2k, seed 13 | 67.14 | 55.20 | 50.55 | 44.15 | 0.83 / 0.01 |
| Synthetic entity 2k, seed 29 | 67.87 | 59.60 | 48.45 | 39.92 | 0.59 / 0.04 |

Entity score-audit split:

| Run | Entity acc | Empty acc | Non-empty acc | Predicted empty rate | Empty median margin |
|---|---:|---:|---:|---:|---:|
| Synthetic entity 2k, seed 13 | 41.91 | 99.45 | 18.41 | 28.84 | 5.47 |
| Synthetic entity 2k, seed 29 | 38.86 | 87.96 | 18.81 | 25.51 | 4.02 |

Out-of-template forced-choice probe:

- Script: `scripts/babylm_custom_entity_probe.py`
- Remote JSON: `/scratch/alombard/babylm-gdn/diagnostics/entity_custom_state_probe_synthentity2k_repeat_20260630.json`
- Probe size: 6 paraphrased crate/bag/shelf examples, 3 empty and 3 non-empty.

| Run | Sum acc | Sum empty | Sum non-empty | Sum predicted empty | Mean acc | Mean empty | Mean non-empty | Mean predicted empty |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GDN 12L seq256 | 50.00 | 100.00 | 0.00 | 100.00 | 33.33 | 33.33 | 33.33 | 33.33 |
| Synthetic entity 2k, seed 13 | 33.33 | 66.67 | 0.00 | 83.33 | 33.33 | 33.33 | 33.33 | 33.33 |
| Synthetic entity 2k, seed 29 | 50.00 | 100.00 | 0.00 | 100.00 | 33.33 | 66.67 | 0.00 | 66.67 |

Interpretation:

- The official-style Entity gain is seed-stable enough to continue: both 2k seeds substantially improve Entity over the original GDN12.
- The improvement remains concentrated in empty-state calibration. Non-empty entity accuracy stayed around 18-19 in both seeds.
- The out-of-template probe does not yet validate generalization. Under summed completion scoring, even the original GDN12 predicts `nothing.` for every custom item, so the probe is dominated by option/prior/length effects.
- Do not launch more synthetic dose sweeps yet. The next useful experiment is a better matched out-of-template entity set or a training variant that reduces empty overprediction while preserving empty recall.

### Balanced out-of-template probe and 25% empty training variant

Updated `scripts/babylm_custom_entity_probe.py` so all custom probe options are one token under the GPT-2 tokenizer, e.g. `nothing.`, `apple.`, `key.`, `book.`. This removes the obvious summed-score length advantage from the first custom probe.

Balanced custom probe:

- Remote JSON: `/scratch/alombard/babylm-gdn/diagnostics/entity_custom_state_probe_balanced_20260630.json`

| Run | Sum acc | Sum empty | Sum non-empty | Sum predicted empty |
|---|---:|---:|---:|---:|
| GDN 12L seq256 | 33.33 | 66.67 | 0.00 | 66.67 |
| Synthetic entity 2k, seed 13 | 16.67 | 33.33 | 0.00 | 33.33 |
| Synthetic entity 2k, seed 29 | 33.33 | 66.67 | 0.00 | 66.67 |

Training variant:

- Name: `gdn-h512l12-s256-synthentity2k-empty25-strict-small-20260630`
- Config: same as the 2k seed-13 run, but `SYNTHETIC_ENTITY_EMPTY_RATE=0.25`.
- Checkpoint: `/scratch/alombard/babylm-gdn/outputs/gdn-h512l12-s256-synthentity2k-empty25-strict-small-20260630`
- Log: `/scratch/alombard/babylm-gdn/logs/gdn-h512l12-s256-synthentity2k-empty25-strict-small-20260630.log`
- Entity audit: `/scratch/alombard/babylm-gdn/diagnostics/entity_nothing_score_audit_synthentity2k_empty25_20260630.json`
- Balanced custom probe: `/scratch/alombard/babylm-gdn/diagnostics/entity_custom_state_probe_empty25_20260630.json`

Fast eval:

| Run | BLiMP | Supplement | EWoK | Entity | Eye / self-paced |
|---|---:|---:|---:|---:|---:|
| Synthetic entity 2k, 50% empty seed 13 | 67.14 | 55.20 | 50.55 | 44.15 | 0.83 / 0.01 |
| Synthetic entity 2k, 50% empty seed 29 | 67.87 | 59.60 | 48.45 | 39.92 | 0.59 / 0.04 |
| Synthetic entity 2k, 25% empty seed 13 | 65.41 | 58.00 | 51.00 | 37.99 | 0.42 / 0.03 |

Entity score-audit split:

| Run | Entity acc | Empty acc | Non-empty acc | Predicted empty rate | Empty median margin |
|---|---:|---:|---:|---:|---:|
| Synthetic entity 2k, 50% empty seed 13 | 41.91 | 99.45 | 18.41 | 28.84 | 5.47 |
| Synthetic entity 2k, 50% empty seed 29 | 38.86 | 87.96 | 18.81 | 25.51 | 4.02 |
| Synthetic entity 2k, 25% empty seed 13 | 36.14 | 77.35 | 19.30 | 22.43 | 2.37 |

Balanced custom probe for 25% empty:

| Run | Sum acc | Sum empty | Sum non-empty | Sum predicted empty |
|---|---:|---:|---:|---:|
| Synthetic entity 2k, 25% empty seed 13 | 0.00 | 0.00 | 0.00 | 0.00 |

Interpretation:

- Lowering empty synthetic rate from 50% to 25% reduces official empty overprediction and recovers non-empty entity accuracy to the original GDN12 level.
- It also reduces the headline Entity gain: 37.99 vs 39.92-44.15 for the 50% empty seeds.
- The balanced out-of-template probe remains poor. The 25% model stops picking `nothing.`, but picks wrong concrete objects on all six custom items.
- Current conclusion: the synthetic intervention is an official-style Entity calibration fix, not yet evidence of robust state-tracking generalization. Stop dose sweeps here unless the paper target becomes leaderboard submission rather than diagnostic science.
