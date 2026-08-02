# SALLM Architecture Crossover Root-Cause Analysis

Date: 2026-07-16

## Executive conclusion

The observed crossovers do not have one shared explanation.

- xLSTM NER versus POS is substantially explained: xLSTM contains useful
  token-level entity information, but the official free-generation NER
  formulation fails when converting that information into copied,
  correctly-formatted spans. POS succeeds under constrained per-token tag
  scoring, which removes that generation bottleneck.
- xLSTM Injongo multilingual performance is plausibly real, but the current
  generative comparison against LLaMA and Mamba is not protocol matched.
- The AfriHG xLSTM/LLaMA crossover is weak and not robust enough to claim as
  an architecture effect.
- xLSTM T2X is limited mainly by source-entity/value copying. Ordinary HPO is
  unlikely to close the gap without a structural copy intervention.
- Corrected GDN results do not show a general multilingual advantage. They
  show strong multi-over-mono gains on AfriHG and generative NER, but chance
  collapse on both multi and mono Injongo and identical collapsed
  MasakhaNews results.

## xLSTM NER versus POS

### Evidence

Controlled task-head NER diagnostics:

| Architecture | NER F1, 3 seeds |
|---|---:|
| xLSTM | 0.4032 +/- 0.0236 |
| LLaMA | 0.3553 +/- 0.0397 |
| Mamba | 0.2491 +/- 0.0105 |

The best nearby xLSTM task-head setting reached 0.4936 F1. In the seed-13
audit, xLSTM entity-token recall was 0.5512 and its predicted entity density
was close to gold (0.1335 versus 0.1372).

### Exact protocol and raw runs

The three architectures used the same diagnostic setup:

- a single linear 9-way BIO head over each model's token hidden states;
- 512 training and 512 test documents per language for Setswana, isiXhosa,
  and isiZulu;
- three epochs, batch size 2, maximum length 256;
- head learning rate 0.002 and backbone learning rate 0.00002;
- seeds 13, 21, and 34;
- entity-level F1 from `seqeval`.

The seed-level all-language scores were:

| Architecture | Seed 13 | Seed 21 | Seed 34 | Mean +/- sample SD |
|---|---:|---:|---:|---:|
| xLSTM | 0.422421 | 0.376841 | 0.410267 | 0.403176 +/- 0.023603 |
| LLaMA | 0.309500 | 0.378910 | 0.377551 | 0.355320 +/- 0.039687 |
| Mamba | 0.240659 | 0.260870 | 0.245846 | 0.249125 +/- 0.010497 |

Seeds alter both initialization and the shuffled 512-document samples. The
result is therefore a robustness diagnostic over training and data sampling,
not three evaluations of one fixed test subset.

The cleanest representation control froze every backbone parameter and trained
only the linear BIO head, using seed 13:

| Architecture | Frozen-backbone entity F1 |
|---|---:|
| xLSTM | 0.366987 |
| LLaMA | 0.252587 |
| Mamba | 0.241356 |

This is the strongest evidence for the narrow claim that the pretrained xLSTM
hidden states expose more linearly recoverable NER information under this
probe. It does not prove that xLSTM is universally best at NER.

The reported 0.493559 score is a separate seed-13 sensitivity run that changed
only the xLSTM backbone learning rate from 0.00002 to 0.00005. It produced
precision 0.444203 and recall 0.555254. This was a nearby robustness check on
the test split, not validation-selected HPO, so it must not be presented as
the primary benchmark score.

### Aligned seed-13 examples

The following examples use the same sampled documents for all three models.

1. `Ngethuba egaleleka uYanga ... uLinda Sobetwa ... uAyanda Sobetwa`
   - Gold: three complete person spans: `uYanga`, `uLinda Sobetwa`, and
     `uAyanda Sobetwa`.
   - xLSTM: all three complete spans exactly.
   - LLaMA: split each two-token surname span into separate person entities.
   - Mamba: split names, omitted part of the last name, and marked
     `seenqwelomoya` as a location.
2. `Isithethi samapolisa uNoloyiso Rwexana uthe:`
   - Gold: one person span, `uNoloyiso Rwexana`.
   - xLSTM: one correct two-token person span.
   - LLaMA: two separate person spans, `uNoloyiso` and `Rwexana`.
   - Mamba: correct person span plus false-positive person `uthe:`.
3. `... umzila kaNkosi uMaqoma ... utsho uSotyu.`
   - Gold: `uMaqoma` and `uSotyu`.
   - xLSTM: both exact spans.
   - LLaMA: both spans plus false-positive person `utsho`.
   - Mamba: both exact spans.

Across all 1,536 seed-13 documents, xLSTM had 606 exact-document predictions,
including 193 exact predictions among the 1,047 documents containing an
entity. LLaMA had 555 and 143 respectively; Mamba had 483 and 88.

### Generative counterexamples

The free-generation adapter often located useful entity text but failed the
required output channel:

- `Izivakashi ... e-India.` Gold: `LOC: India`; raw xLSTM:
  `god (LOC: e-India`. The location is found but the prefix and boundary fail.
- `... uMnuz Muzi Mahlambi ...` Gold: `PER: Muzi Mahlambi`; raw xLSTM:
  `god name output? PER: Muzi Mahlambi`. The entity is exact behind junk text.
- `NgoMgqibelo, iPirates ... e-Orlando Stadium.` xLSTM recovered several
  organizations and the date but changed the location into the hallucinated
  organization `Orlando City`.

This is why token-head success and generative failure can coexist: the head
chooses one BIO label per observed source token, while the generative task must
also reproduce text boundaries, labels, separators, ordering, and stopping.

### Raw evidence locations

- Probe implementation:
  `/scratch/alombard/sallm/scripts/run_task_head_matrix_2026_06_26.py`
- Seed-13 baseline:
  `results/diagnostics/task_head_sib_ner_test_20260629/*_ner_all.json`
- Seeds 21 and 34:
  `results/diagnostics/task_head_core_seeds_test_20260629/*_ner_all_seed*.json`
- Corrected Mamba seed 13:
  `results/diagnostics/task_head_mamba_fast_seed13_test_20260629/mamba_ner_all_seed13_fast.json`
- Frozen probes:
  `results/diagnostics/task_head_frozen_probe_20260630/*_ner_all_frozen.json`
- Learning-rate sensitivity:
  `results/diagnostics/task_head_hpo_sanity_20260629/*_ner_all_bbhi.json`
- Generative examples:
  `sallm_memory/xlstm_hpo_debug_findings_2026-06-25.html`

By contrast, official free-generation NER repeatedly produced a dominant
`god...` prefix, under-extracted spans, and drifted labels. A BIO/tag-sequence
rescue reached 0.4934 training BIO F1 but only 0.1539 validation BIO F1 and
0.0587 official span-generation F1.

### Interpretation

This is a readout/formulation failure, not evidence that xLSTM lacks entity
representations. POS is evaluated through constrained token-level label
scoring; NER asks the model to copy boundaries and entity strings into a
fragile free-form output. The defensible result is to report both:

1. official generative NER as a failure of the requested formulation; and
2. controlled BIO/task-head NER as evidence about representational quality.

## Injongo multilingual advantage

xLSTM multilingual training pooled English, Sesotho, isiXhosa, and isiZulu,
cycled five prompt templates, used assistant-only loss, and used a much larger
LoRA configuration (rank 256, alpha 512, including embeddings and projection
modules). It was then selected by a Bayesian HPO run optimizing held-out
classification F1.

The later held-out multilingual adapter reached approximately:

| Language | Mean F1 | Best-prompt F1 |
|---|---:|---:|
| English | 0.8806 | 0.9064 |
| isiXhosa | 0.8030 | 0.8463 |
| Sesotho | 0.6424 | 0.6998 |
| isiZulu | 0.6049 | 0.6226 |

Earlier diagnostics also showed the multilingual isiXhosa slice far above the
monolingual adapter, supporting a real label-space/calibration benefit from
pooled training. Controlled task-head intent results also favored xLSTM over
LLaMA and Mamba.

However, the old cross-architecture adapters are not comparable: chat-template
use, LoRA rank/targets, epochs, and checkpoint selection differ materially.
Therefore Mamba and LLaMA should receive the same *experimental opportunity*,
not blindly identical LoRA modules:

- first run the same evaluation protocol on existing adapters;
- then use the same HPO trial budget and validation objective;
- match effective training examples/tokens and approximately match trainable
  parameter counts with architecture-appropriate target modules.

## AfriHG crossover

The old mono/multi table does not show a clean crossover:

- mono isiXhosa: xLSTM 16.594 chrF versus LLaMA 14.283;
- mono isiZulu: xLSTM 17.270 versus LLaMA 19.048;
- multi isiXhosa: xLSTM 17.017 versus LLaMA 16.978, only +0.039;
- multi isiZulu: xLSTM 18.898 versus LLaMA 17.481, +1.417.

The HPO-selected xLSTM multilingual adapter later scored only 15.196 isiXhosa
and 17.099 isiZulu, below the earlier multilingual xLSTM artifact. Validation
decoding could raise scores, but generations remained semantically noisy.

The likely mechanisms are positive Nguni-language transfer and decoding
sensitivity, but the result is not robust enough to call an architecture win.
Run a matched exposure and validation-decoding experiment with two final
seeds, then freeze decoding for test.

## T2X

xLSTM T2X diagnostics found no empty-output or gross length problem. Instead,
subject preservation was 28.8%, object/value preservation 25.7%, and both
were preserved in only 9.0% of examples. This is a source-binding/copy
failure. The useful next experiment is a structural rescue:

- delexicalize entities and values into placeholders;
- generate the template;
- deterministically reinsert source values;
- report subject, object, and joint preservation with chrF;
- run the same intervention as a LLaMA control.

This must be reported as a formulation rescue/ablation, not silently replace
the baseline.

## Corrected GDN mono versus multilingual findings

Only `_r2` corrected adapter evaluations should be treated as valid.

- Injongo: all multilingual and monolingual adapters are at chance
  (accuracy about 0.025, F1 about 0.0012). There is no valid multilingual
  advantage.
- MasakhaNews: multilingual and corresponding monolingual results are exactly
  identical and badly collapsed, suggesting invariant predictions or an
  ineffective adapter path.
- MasakhaNER: multilingual reaches useful prompt-dependent scores while
  Tswana and isiXhosa mono are zero and isiZulu mono is weak. This needs an
  adapter-loading/output audit before architectural interpretation.
- AfriHG: multilingual is clearly above mono:
  isiXhosa 21.1288 versus 14.8387 chrF, and isiZulu 18.8708 versus 14.2960.

GDN has not received task HPO. Before HPO, audit whether multilingual jobs saw
more examples, optimizer steps, or target tokens. A fixed epoch count on
pooled data is not a fair transfer test because it gives the multilingual
adapter more updates.

## Compute-efficient causal experiment order

1. Finish and freeze corrected GDN evaluations.
2. Audit adapter provenance, chat templates, scoring mode, training examples,
   optimizer steps, target tokens, LoRA trainable parameters, and checkpoint
   selection for every mono/multi pair.
3. Complete the already staged matched-protocol Mamba and LLaMA Injongo
   controls.
4. If the Injongo gap survives, run an 8-12 trial matched HPO budget per
   architecture, followed by two or three seeds only for final configs.
5. For GDN AfriHG and NER, compare full multilingual, multilingual downsampled
   to mono exposure, and mono oversampled to equal updates.
6. Run validation-selected decoding for AfriHG and freeze it before test.
7. Run the T2X placeholder/copy rescue with a LLaMA control.

Do not launch broad GDN HPO across every task until the collapsed classification
results and training-exposure confound are resolved.
