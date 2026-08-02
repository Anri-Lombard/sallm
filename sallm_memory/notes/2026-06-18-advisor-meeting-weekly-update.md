# Advisor meeting update - 2026-06-18

Coverage window: Thursday 2026-06-11 through Wednesday 2026-06-17.

## Executive summary

This week was mostly evaluation hygiene and targeted root-cause work, not a broad new leaderboard push.

1. The old xLSTM POS result around `0.93` token accuracy was invalid. A corrected constrained POS test now gives the defensible comparison: LLaMA is best, xLSTM is moderate, Mamba is very weak.
2. xLSTM T2X mixed-source improved validation, but did not transfer to a strong official test result. Test chrF is only `34.1600`, still far below LLaMA around `53.976`.
3. NER remains bad even after reformulating as constrained BIO tag scoring. This is important: the problem is not only span-generation parsing/copying; all architectures show label-prior collapse under legal BIO labels.
4. Mamba POS/NER failure is now better supported: it survives closed-label/constrained scoring, so it is not just malformed free generation.
5. Base 0-shot checks filled some missing evidence, but NER/POS/Injongo base rows remain provenance-sensitive and should stay out of final sheet cells unless the artifacts prove test split.
6. xLSTM POS HPO is running correctly now. The 8-run validation sweep completed; trial 8 is best validation so far, and a held-out test eval job was submitted but is still pending/running as of the last note.
7. GatedDeltaNet remains next after xLSTM HPO and cleanup; do not start it until the current HPO/test gate lands.

## Corrected POS result

The biggest result this week is a correction to the POS story.

Earlier POS results were inflated because the old scorer treated list-valued targets as multiple acceptable answers. The corrected protocol evaluates MasakhaPOS on the test split as closed-label token tagging:

- Dataset: `masakhane/masakhapos`
- Split: `test`
- Languages: `tsn`, `xho`, `zul`
- Contract: tuple output
- Score mode: mean logprob over legal UPOS labels
- Primary metric: token accuracy

Final corrected multilingual POS test jobs:

- xLSTM `928753`: completed cleanly in `02:34:03`.
- Mamba `928754`: completed cleanly in `02:54:19`.
- LLaMA `928755`: completed cleanly in `01:33:09`.

Best per-language constrained POS test token accuracy:

| Architecture | Tsn | Xho | Zul |
|---|---:|---:|---:|
| LLaMA | `0.8460` | `0.8220` | `0.8409` |
| xLSTM | `0.7245` | `0.6933` | `0.7313` |
| Mamba | `0.2340` | `0.0674` | `0.0402` |

Weighted all-language token accuracy by best template family:

- LLaMA: around `0.836`.
- xLSTM: around `0.718`.
- Mamba: around `0.129` at best.

Interpretation for advisors:

- xLSTM POS is not near-`0.93`; that was a scoring bug.
- xLSTM is still meaningfully competent under the corrected protocol.
- LLaMA remains clearly strongest.
- Mamba remains very weak even when we remove free-generation formatting as the bottleneck.

Sheet/progress status:

- Google Sheet multilingual tuned POS cells were updated and verified:
  - Transformer `F7:F9`: Xho/Zul/Tsn `0.822`/`0.841`/`0.846`.
  - Mamba `F7:F9`: Xho/Zul/Tsn `0.067`/`0.040`/`0.234`.
  - xLSTM `F7:F9`: Xho/Zul/Tsn `0.693`/`0.731`/`0.724`.
- `sallm_memory/sallm_progress.md` was updated because this changes the high-level POS interpretation.

## POS failure examples

Corrected xLSTM POS is moderate because the model often learns the broad tuple/tag shape but loses alignment.

Best corrected generative-tag-sequence xLSTM POS prompt before the final constrained evaluator:

- Tswana token accuracy `0.3279`, exact length `0.1362`.
- Xhosa token accuracy `0.3567`, exact length `0.1897`.
- Zulu token accuracy `0.3904`, exact length `0.1963`.

Failure pattern:

- Under-generates on long sentences.
- Skips initial tokens.
- Duplicates or reorders later tokens.
- Confuses open-class tags such as `NOUN`/`VERB`/`PROPN`.
- Proper-noun or punctuation-heavy rows can collapse into repeated fragments.

Concrete examples from the note:

- Tswana doc 22: 53 gold tags collapsed to one invalid tag after hallucinated fragments around `Steenhuisen`; score `0.0`.
- Xhosa doc 32: 41 gold tags collapsed to repeated `Soul of the HIV-1` fragments; score `0.0`.
- Zulu doc 110: 12 gold tags collapsed to repeated `lizo` tags and a final punctuation; score `0.0`.
- Short Xhosa doc 60 matched exactly, showing the task is possible when length/alignment stays simple.

Mamba POS root cause under corrected constrained scoring:

- The evaluator forces each token to one legal UPOS label.
- Mamba still collapses to a few label priors, mainly `DET`, `NOUN`, and sometimes `PART`.
- Example: Xhosa target `VERB NOUN PRON VERB NOUN PUNCT` was predicted as all `NOUN`.
- Punctuation often maps to `DET`/`NOUN`, so this is not just morphology difficulty.

## T2X mixed-source official test

The T2X mixed-source rescue looked promising on validation but did not become a strong official test result.

Job:

- `921485` `eval-xlstm-t2x-mixed-test`
- Completed cleanly.
- Local artifacts: `outputs/hex_results/xlstm_t2x_mixed_source_test_20260611`

Official T2X Xhosa test metrics:

- chrF `34.1600`
- BLEU `0.0454`
- ROUGE-L `0.2974`

Interpretation:

- Validation improved to chrF about `39.47`, but test only reached `34.1600`.
- This is only a small improvement over the prior xLSTM mono T2X official result around chrF `33.719`.
- It remains far below LLaMA T2X around chrF `53.976`.
- The next useful T2X direction is structural delexicalization or source-placeholder reinsertion, not broad decoding HPO.

`sallm_progress.md` was updated for this because it changes the high-level T2X rescue story from promising validation canary to limited/negative official test transfer.

## Base 0-shot wave

A Mamba/xLSTM base 0-shot wave was submitted to fill missing base rows.

Tag:

- `base_zeroshot_mamba_xlstm_20260611`

Mamba jobs:

- `921453`-`921468`

xLSTM jobs:

- `921469`-`921484`

Important outcomes:

- Mamba News 0-shot best F1: Eng `0.2691`, Xho `0.1933`.
- xLSTM News 0-shot best F1: Eng `0.2593`, Xho `0.2323`.
- Mamba SIB 0-shot best F1: Afr `0.2101`, Eng `0.2235`, Nso `0.1061`, Sot `0.0962`, Xho `0.1327`, Zul `0.1639`.
- xLSTM SIB 0-shot best F1: Afr `0.2041`, Eng `0.2165`, Nso `0.1122`, Sot `0.1363`, Xho `0.1730`, Zul `0.1853`.
- Mamba T2X Xho 0-shot chrF: `0.57`.
- Mamba AfriHG 0-shot chrF: Xho `2.00`, Zul `1.24`.
- xLSTM base 0-shot T2X chrF: `4.0312`, BLEU `0.0`, ROUGE-L `0.0664`.
- xLSTM base 0-shot AfriHG fast retry:
  - Xho chrF `7.8187`, BLEU `0.0`, ROUGE-L `0.00155`.
  - Zulu later completed, but the note emphasizes the result remains weak.
- xLSTM base 0-shot MasakhaNER F1 is `0.0` for Tswana/Xhosa/Zulu across prompts, but these are validation-style task names and should not be final test sheet rows.
- Mamba and xLSTM base InjongoIntent 0-shot are essentially collapsed, around F1 `0.001`.

Interpretation:

- Base 0-shot confirms neither Mamba nor xLSTM solves these downstream tasks without adaptation.
- Test-only provenance remains important. NER/POS/Injongo base rows should stay pending unless raw artifacts prove the exact test split/current protocol.

## NER reformulation wave

We ran a diagnostic reformulation: MasakhaNER as constrained BIO tag-sequence scoring rather than official span generation.

This was intentionally diagnostic only. It does not replace official span-generation NER rows.

Submitted jobs:

- xLSTM finetune/eval: `933254` / `933255`
- Mamba finetune/eval: `933256` / original failed eval `933257`, repair eval `933266`
- LLaMA finetune/eval: `933258` / `933259`

Evaluator:

- `scripts/run_constrained_ner_eval.py`
- Dataset: `masakhane/masakhaner2`
- Mirror: `anrilombard/masakhaner-x-parquet`
- Split: `test`
- Languages: `tsn`, `xho`, `zul`
- Label set: `o b-per i-per b-org i-org b-loc i-loc b-date i-date`
- Prompt: `masakhane_named_entity_recognition_tag_sequence_lower/lm_eval_p1`

Final macro diagnostic results:

| Architecture | Token accuracy | Non-O recall | Entity F1 | Exact sequence |
|---|---:|---:|---:|---:|
| Mamba | `0.0103` | `0.0722` | `0.0119` | `0.0000` |
| LLaMA | `0.0381` | `0.2596` | `0.0353` | `0.0007` |
| xLSTM | `0.0274` | `0.1687` | `0.0367` | `0.0000` |

xLSTM per-language:

- Tswana: token accuracy `0.0175`, non-O recall `0.1495`, entity F1 `0.0288`.
- Xhosa: token accuracy `0.0410`, non-O recall `0.1814`, entity F1 `0.0451`.
- Zulu: token accuracy `0.0236`, non-O recall `0.1753`, entity F1 `0.0362`.

Root-cause examples:

- Mamba predicts mostly `b-date`/`i-date`.
  - Tswana top confusion: `o->b-date` with `21446` cases.
  - Example: `Lazarious Ramolotja` is gold `b-per i-per`, but Mamba predicts `b-date` for every token.
- LLaMA predicts mostly `b-per`.
  - Tswana top confusion: `o->b-per` with `21895` cases.
  - Example: a sentence with org/location/date labels is predicted as `b-per` across ordinary tokens.
- xLSTM predicts repeated inside labels such as `i-loc`, `i-per`, `i-org`.
  - Example: `Muzi Mahlambi` is gold `b-per i-per`, but xLSTM predicts `b-per i-per i-per ...` across the full sentence.

Interpretation:

- Constrained BIO tagging did not rescue NER for any architecture.
- This rules out the idea that NER failure is only about span-copying or malformed free generation.
- The deeper issue is label-prior/calibration collapse even when legal BIO labels are forced.
- xLSTM narrowly has the best entity F1, but all scores are near-zero.
- Diagnostic rows were added to the Google Sheet `Eval Provenance` tab as `diagnostic_only`; official NER benchmark rows were not overwritten.

## xLSTM POS HPO

After the corrected POS audit, xLSTM POS constrained-validation HPO was started.

Purpose:

- Improve xLSTM POS under the corrected closed-label tuple contract.
- Validation only; not a final test result.

Implementation:

- Config: `src/conf/finetune/xlstm_pos_all_constrained_hpo.yaml`
- Sweep: `src/conf/sweeps/xlstm_pos_all_constrained_screen.yaml`
- Trial runner: `src/main/sallm/hpo/constrained_pos_trial.py`
- Submitter: `scripts/submit_xlstm_pos_constrained_hpo_2026_06_16.sh`
- Job: `933481`
- W&B sweep: `gyhw0qhd`

Initial failures and repairs:

- Job `933476` failed before sweep creation because launch output was hidden.
- Job `933477` created a sweep but all early trials failed because custom key `training.save_final_adapter_for_hpo` leaked into `SFTConfig`.
- Repair moved adapter saving behind env var `SALLM_SAVE_FINAL_ADAPTER_FOR_HPO=1`.
- Job `933481` then ran actual trials successfully.

Validation results:

- Trial 1: macro token accuracy `0.7600502135`.
- Trial 2: `0.7549175548`.
- Trial 3: `0.7455481754`.
- Trial 4: W&B summary only, about `0.7508`; full JSON was overwritten during an SSH outage.
- Trial 5: `0.7054`.
- Trial 6: `0.7906031986`.
- Trial 7: `0.7743050048`.
- Trial 8: best, `0.7939101624`.

Best trial 8 config:

- W&B run `ywpp4w9r`, `eternal-sweep-8`
- LR `0.00013922147464227958`
- Scheduler `cosine`
- Epochs `12`
- Warmup `0.05`
- Weight decay `0.1`
- LoRA `r=128`, `alpha=256`, dropout `0.1`

Trial 8 validation metrics:

- Macro token accuracy `0.7939101624`.
- Micro token accuracy `0.7974779959`.
- Exact sequence accuracy macro `0.0616666667`.
- Tswana `0.8120136187`, Xhosa `0.7740811153`, Zulu `0.7956357533`.

Interpretation:

- HPO materially improved validation over the corrected xLSTM test baseline region.
- It is still validation-only and must not be written to main result cells yet.
- A held-out constrained POS test eval was submitted as job `937140` under tag `xlstm_pos_hpo_best_test_20260617`; it was `PENDING (Resources)` at submission.
- The next advisor-facing decision is whether to treat this as the chosen xLSTM POS recipe if the held-out test result improves over `0.7245/0.6933/0.7313`.

## Advisor-facing corrections

The previous 2026-06-11 advisor note said "POS is very strong." That should now be corrected:

- Old claim: xLSTM POS is very strong around `0.93`.
- Corrected claim: xLSTM POS is moderate and real under closed-label scoring, around `0.69-0.73` test token accuracy; LLaMA is still clearly better around `0.82-0.85`; Mamba is very weak around `0.04-0.23`.

The NER explanation also changed:

- Old likely explanation: NER is hard mostly because generated span output requires copying/boundaries/formatting.
- Updated explanation: span generation is still hard, but constrained BIO test diagnostics show even legal tag-sequence scoring collapses. So NER has a deeper label-prior/calibration problem across architectures.

## Open questions for advisors

1. Should the thesis report corrected POS via constrained token-label scoring, while explicitly invalidating the old list-target lm-eval POS rows?
2. If the xLSTM POS HPO test job improves, should we rerun matching LLaMA/Mamba HPO or report xLSTM as individually optimized while keeping protocol provenance explicit?
3. For NER, is it worth another formulation experiment, or should we declare decoder-only NER weak under both span-generation and constrained BIO diagnostics?
4. For T2X, should we pursue structural delexicalization/source-placeholder reinsertion, or stop after the mixed-source official test failed to transfer strongly?
5. Given scratch pressure around `91-93%`, what can be safely cleaned before the next architecture phase?
6. Should GatedDeltaNet start only after the POS HPO held-out test lands, or can setup begin while waiting?

## Recommended next work

1. Monitor job `937140` and pull `xlstm_pos_hpo_best_test_20260617/xlstm_pos_hpo_best_test.json` when terminal.
2. If the HPO POS test improves, update the sheet/progress only after proving `split == test` and matching the corrected constrained POS protocol.
3. Do not run more broad NER HPO now. The constrained BIO diagnostic shows the bottleneck is deeper than output formatting.
4. Do not spend broad T2X decoding HPO. If continuing T2X, use a structural source-placeholder/delexicalization experiment.
5. Do scratch cleanup before GatedDeltaNet or any next architecture wave.

## Files and artifacts

- Progress tracker: `sallm_memory/sallm_progress.md`
- Corrected POS notes: `sallm_memory/notes/2026-06-12.md`, `sallm_memory/notes/2026-06-16.md`
- NER reformulation notes: `sallm_memory/notes/2026-06-16.md`
- xLSTM POS HPO notes: `sallm_memory/notes/2026-06-17.md`
- T2X official test notes: `sallm_memory/notes/2026-06-11.md`
- Advisor/root-cause brief: `sallm_memory/notes/2026-06-11-advisor-root-cause-brief.html`
- Research-group table artifact: `sallm_memory/notes/2026-06-11-research-group-update.html`
- Corrected POS artifacts:
  - `outputs/hex_results/pos_constrained_test_20260612/`
- NER tag-sequence diagnostic artifacts:
  - `outputs/hex_results/diagnostics/ner_tagseq_reformulation_20260616/`
  - `outputs/staging/ner_tagseq_reformulation_20260616/summary.md`
- xLSTM POS HPO artifacts:
  - `outputs/hex_results/hpo/xlstm_pos_constrained_hpo_20260616/`
- xLSTM T2X mixed-source test:
  - `outputs/hex_results/xlstm_t2x_mixed_source_test_20260611`
