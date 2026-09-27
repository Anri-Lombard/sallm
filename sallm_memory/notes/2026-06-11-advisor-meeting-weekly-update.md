# Advisor meeting update - 2026-06-11

Coverage window: Thursday 2026-06-04 through Wednesday 2026-06-10.

## Executive summary

This week produced a much clearer xLSTM story.

1. Task-specific xLSTM adapters are alive: POS is very strong, News is strong after the corrected chat/test path, and NER is nonzero but still weak.
2. A single broad xLSTM general adapter is not a good universal adapter for this setup. It mostly collapses structured tasks and should not be used as xLSTM's best downstream representation.
3. Base Mamba/xLSTM 2-shot and 3-shot results were completed, repaired where needed, and written to the shared results sheet.
4. SIB is not a simple xLSTM-only failure. It is highly sensitive to task formulation, adapter scope, and scoring protocol.
5. NER remains the clearest hard failure for generative xLSTM: token/tag diagnostics show learning, but official span generation is still weak because of boundary/copying/format stability.
6. T2X has the only positive generation-rescue signal this week. A mixed-source xLSTM canary improved validation metrics, but examples still show repetition/source-preservation errors, so it needs bounded official/test follow-up before being promoted.
7. AfriHG remains weak despite focus-prompt canaries. Outputs are non-empty, but semantically poor.

## xLSTM downstream gate completed

The xLSTM NER/POS gate and monolingual downstream phase reached terminal, usable evidence.

Terminal POS gate result:

- Multi POS all eval job `881664` completed cleanly.
- Tswana token accuracy by prompt: `0.9133`, `0.9067`, `0.9400`, `0.9467`.
- Xhosa token accuracy by prompt: `0.9200`, `0.9200`, `0.9267`, `0.9333`.
- Zulu token accuracy by prompt: `0.9133`, `0.9267`, `0.9533`, `0.9333`.

Interpretation:

- xLSTM can do dense, token-aligned, closed-label sequence labelling well.
- This is the strongest evidence that xLSTM is not simply broken as a downstream adapter.
- But POS uses parsed token/tag outputs, so we should report it as strong parsed token accuracy, not as proof that the raw generations are always clean.

NER gate:

- Earlier mono Xhosa NER xLSTM F1 was nonzero across prompts, roughly `0.226-0.254`.
- This is meaningfully better than all-zero base prompting and avoids the worst Mamba collapse pattern.
- But it remains far below the transformer/LLaMA story and still shows sparse-span copying/format failures.

News:

- Corrected chat/test evaluation recovered xLSTM News strongly.
- Root-cause diagnostics found News Xhosa as a positive control:
  - mono Xho sum/mean accuracy/F1 about `0.8966`/`0.8935`.
  - multi all sum/mean accuracy/F1 about `0.8820`/`0.8807`.
  - Xho slice in multi accuracy/F1 about `0.8875`/`0.8754`.
- Interpretation: when the task is short-label classification and the eval template path is correct, xLSTM is healthy.

## General adapter result

The general xLSTM adapter completed, but mostly failed as a universal adapter.

General fine-tune/eval:

- Fine-tune job `896551` completed cleanly in `15:55:01`.
- 39/40 general eval jobs completed cleanly.
- Failed eval `896566` was AfriHG English and failed before inference because no AfriHG English CSV was available; this is a dataset/config issue, not model behavior.

Selected general adapter results:

- News: English best F1 `0.6582`, Xhosa best F1 `0.1929`.
- NER: Tswana best F1 `0.0094`, Xhosa `0.0038`, Zulu `0.0035`.
- POS: Tswana token accuracy `0.0933`, Xhosa `0.0133`, Zulu `0.0200`.
- SIB all: best language-level F1s roughly `0.1160-0.1621`.
- InjongoIntent all: best F1s near zero, English `0.0188`, Sotho `0.0097`, Xhosa `0.0080`, Zulu `0.0074`.
- Belebele: near chance, about `0.26-0.30` accuracy.
- AfriXNLI: around chance, about `0.333`.
- AfriMMLU: low, about `0.2200-0.2560`.
- AfriGSM: flexible exact match about `0.0080-0.0200`.
- Generation:
  - AfriHG Xhosa chrF `1.3879`.
  - AfriHG Zulu chrF `2.4220`.
  - T2X Xhosa chrF `22.0024`.

Interpretation:

- The broad general adapter mixes too many output formats under one LoRA adapter.
- It is not xLSTM's best downstream form and should not be used as the main architecture comparison point.
- The thesis comparison should emphasize task-specific mono/multi adapters, then discuss the general adapter as a negative scalability/format-mixing result.

## Multilingual wave and task scope

After the mono/gate evidence, the multilingual wave was submitted:

- Manifest: `outputs/final_submissions/xlstm_downstream_3epoch_20260602_pad64_multi.csv`
- Jobs:
  - `892240/892241` MasakhaNews all.
  - `892242/892243` MasakhaNER all.
  - `892244/892245` MasakhaPOS all.
  - `892246/892247` SIB all.
  - `892248/892249` InjongoIntent all.
  - `892250/892251` AfriHG all.

Important qualitative result:

- Multilingual training helped some label tasks, especially POS and InjongoIntent.
- For InjongoIntent, diagnostics showed mono Xhosa was calibration-collapsed, while multi Xhosa was much healthier:
  - mono Xho validation mean forced-choice accuracy/F1: `0.1547`/`0.1087`.
  - multi all validation mean accuracy/F1: `0.4867`/`0.4883`.
  - multi Xho slice mean accuracy/F1: `0.6906`/`0.6848`.
- Interpretation: the mono-vs-multi gap is not just scoring-mode artifact. Multilingual training appears to stabilize the label space.

## Base few-shot comparison completed

The user requested Mamba and xLSTM base-model 2-shot/3-shot evaluations to match LLaMA few-shot checks.

Submission:

- Submitter: `scripts/submit_base_fewshot_mamba_xlstm_2026_06_06.sh`
- Tag: `base_fewshot_mamba_xlstm_20260606`
- Manifest: `outputs/final_submissions/base_fewshot_mamba_xlstm_20260606.csv`
- Models:
  - Mamba: `anrilombard/sallm-mamba-125m`
  - xLSTM: `anrilombard/sallm-xlstm-125m-native-3epoch-20260531`
- Shots: `2` and `3`.

Operational issues and repairs:

- HEX accepted only 50 jobs initially due `QOSMaxSubmitJobPerUserLimit`; the xLSTM 3-shot tail was later submitted as `899418`-`899431`.
- Mamba 3-shot InjongoIntent job `898500` failed with CUDA OOM during lm-eval loglikelihood after auto batch selected 40 and attempted a huge allocation. Repair `899432` capped batch size to 1 and completed cleanly.
- Mamba 2-shot InjongoIntent job `898468` also failed and was repaired with batch-1 job `903631`, which completed cleanly in `4h37m13s`.
- xLSTM 3-shot MasakhaNER all `898513` was very slow but completed; the slowness was expected because it expands to 10,760 generative requests.

Sheet status:

- Local summary helper: `scripts/summarize_base_fewshot_sheet.py`.
- Sheet update helper: `scripts/prepare_base_fewshot_sheet_updates.py`.
- Final local staging wrote `164` completed rows.
- Google Sheet base cells are complete for Mamba and xLSTM 2-shot/3-shot rows.
- Mamba InjongoIntent repaired readback:
  - English: `2-shot 0.0024 F1`, `3-shot 0.0021 F1`.
  - Xhosa: `2-shot 0.0024 F1`, `3-shot 0.0039 F1`.
  - Zulu: `2-shot 0.0013 F1`, `3-shot 0.0038 F1`.
  - Sotho: `2-shot 0.0036 F1`, `3-shot 0.0069 F1`.

Interpretation:

- Base few-shot rows are now recorded, but several results are weak because these are base-model prompting/loglikelihood checks, not task-specific adapters.
- For advisors, the important point is that missing base few-shot cells are no longer a blocker.

## SIB root-cause work

SIB initially looked like an xLSTM weakness, but the week showed the story is more complicated.

First finding:

- Old official SIB artifacts contained literal `\\category:` prompts.
- A first "corrected" rerun was contaminated because final/test SIB task packs resolved stale lm-eval task definitions.
- A clean retry added explicit SIB test task definitions and verified no literal `\\category` remained.

Clean xLSTM official-style SIB retry:

- Jobs `905498`, `905499`, `905500` completed cleanly.
- Mono SIB Xho best accuracy/F1: `0.2598`/`0.1645`.
- General SIB Xho best F1: `0.1056`.
- Multi SIB all best F1 by language:
  - Afrikaans `0.1112`
  - English `0.0912`
  - Northern Sotho `0.1099`
  - Southern Sotho `0.1194`
  - Xhosa `0.1325`
  - Zulu `0.1121`

This means the stale prompt path was a real hygiene issue, but not the main cause of low official-style SIB scores.

Forced-choice split/formulation audit:

- xLSTM mono SIB Xho test summed accuracy/F1: `0.5843`/`0.5732`.
- xLSTM multi SIB all test summed accuracy/F1: `0.5615`/`0.5260`.
- xLSTM general SIB Xho test summed accuracy/F1: `0.2255`/`0.1478`.
- Mamba SIB all test summed accuracy/F1: `0.6911`/`0.6816`.
- Mamba SIB Xho test summed accuracy/F1: `0.4373`/`0.4024`.
- SIB scoring mode sensitivity:
  - Mamba SIB all prefers summed scoring: sum accuracy `0.6911`, mean accuracy `0.4401`.
  - xLSTM multi SIB all prefers mean scoring: sum accuracy `0.5615`, mean accuracy `0.6964`.

Interpretation:

- SIB is sensitive to scoring aggregation, task scope, and official lm-eval formulation.
- Mamba all is healthy, Mamba Xho-only is weaker, xLSTM forced-choice is plausible, but official-style xLSTM remains low.
- Next SIB work should be a reporting/formulation audit, not broad HPO.

## NER diagnostics

NER remains the most important unresolved xLSTM weakness.

Why POS is strong but NER is weak:

- POS is dense, closed-label, token-aligned, and scored by token accuracy.
- NER span generation is sparse and requires entity detection, exact span copying, entity typing, delimiter formatting, and stopping.
- xLSTM can learn labels, but struggles to convert that into faithful copied generative spans.

Diagnostic and HPO results:

- Atomic tag-sequence diagnostic was used to test whether xLSTM can learn token-aligned entity labels.
- Official span-generation HPO variants completed but were weaker than the earlier mono Xhosa NER:
  - `901226`: best F1 about `0.1391`.
  - `901230`: best F1 about `0.1413`.
  - `901232`: best F1 about `0.1271`.
  - Earlier mono Xhosa NER: about `0.226-0.254`.
- Timed-out HPO cell `901228` was not rerun because it is not blocking the main story and scratch was high.

Lower-label canary:

- Finetune `905623`, eval repair `905662`.
- Train F1 `0.4348`.
- Validation F1 `0.1141`.
- Predictions are non-empty and label-shaped, but overgenerate, repeat spans, hallucinate labels, and make boundary/type errors.
- Example failure: gold `PER: kukaMali`, `PER: Fete`, `LOC: asePunzana`; prediction `per: kukamali $ date: ku0784661235`, missing the person/location boundaries and turning a phone number into a date.

Decode-control sweep:

| Variant | Validation F1 | Empty filtered preds | Raw preds >50 tokens |
|---|---:|---:|---:|
| baseline lower-label | `0.1141` | `59` | `49` |
| max64 + rp1.1 + no-repeat-3 | `0.0534` | `83` | `0` |
| max48 + rp1.2 + no-repeat-3 | `0.0425` | `86` | `0` |
| max32 + rp1.2 + no-repeat-3 | `0.0429` | `89` | `0` |

Interpretation:

- Decode controls successfully removed long generations, but made F1 and target-token recall worse.
- NER is not mainly a decoding-length/repetition problem.
- The next NER work should focus on objective/formulation and span-copy supervision, not decoding HPO.

## T2X and AfriHG generation

T2X:

- Existing source-preservation diagnostic showed xLSTM T2X has entity/value copying problems.
- Baseline validation: ROUGE-L `0.2949`, BLEU `0.0712`, chrF `34.38`, entity coverage `0.6357`, value coverage `0.6798`, repetition `0.2149`.
- Greedy with repetition/no-repeat reduced repetition but hurt metrics badly.
- Source-checklist prompt increased entity coverage to `0.7066`, but reduced core metrics and increased repetition.

Positive rescue signal:

- Mixed-source T2X canary jobs `905684`/`905685` completed cleanly.
- Base prompt after mixed-source training:
  - ROUGE-L `0.3587`
  - BLEU `0.0826`
  - chrF `39.47`
- Previous validation baseline:
  - ROUGE-L `0.2949`
  - BLEU `0.0712`
  - chrF `34.38`
- Interpretation: this is the clearest positive generation-rescue signal this week. It should be promoted to a bounded official/test run plus example inspection.
- Caution: examples still show repetition/source errors, such as an airport/elevation triple producing a long repeated `i507 y elevation` sequence.

AfriHG:

- Focus canary jobs `905686`-`905688` completed cleanly.
- Focus-prompt validation:
  - Xho ROUGE-L `0.0189`, BLEU `0.0`, chrF `13.82`.
  - Zul ROUGE-L `0.0134`, BLEU `0.0`, chrF `14.16`.
- Base-prompt validation using focus adapter:
  - Xho ROUGE-L `0.0110`, BLEU `0.0`, chrF `14.84`.
  - Zul ROUGE-L `0.0114`, BLEU `0.0`, chrF `14.43`.
- Example failures remain semantically broken:
  - Xho focus predicted `12 - 25` for a cricket headline.
  - Xho focus predicted date/noise text for a Chiefs/Ngcobo article.
  - Zulu focus produced noisy mixed headline text for ARV theft and decuplet stories.

Interpretation:

- AfriHG is not rescued.
- The problem is not empty output; it is weak semantic headline conditioning.
- Do not spend broad HPO on AfriHG until there is a stronger formulation/data intervention.

## Operational notes

- Scratch repeatedly crossed the watch threshold during the week, reaching about `93.2%`.
- After base few-shot results were staged and sheet-verified, the completed base few-shot eval tree was removed from scratch:
  - removed `/scratch/lmbanr001/masters/sallm/results/eval/base_fewshot_mamba_xlstm_20260606`.
  - scratch dropped to about `81.5%`.
- Several HEX/VPN outages caused monitoring blind spots, but there were no unresolved active jobs at the end of the recorded diagnostics except any future optional follow-ups.

## Concrete examples for the meeting

Example 1: POS proves xLSTM can do token-aligned sequence labelling.

- Multi POS all Xhosa token accuracy was `0.9200-0.9333`; Zulu reached up to `0.9533`.
- This contrasts with NER, where official span-generation remains around `0.226-0.254` at best and HPO/decoding did not improve it.
- Explanation: POS is dense and closed-class; NER needs sparse copied spans and strict formatting.

Example 2: decoding control did not fix NER.

- Lower-label NER baseline F1 was `0.1141`.
- Adding max-token limits, repetition penalty, and no-repeat ngram controls reduced F1 to `0.0425-0.0534`.
- The controls removed long outputs but increased empty predictions and reduced target-token recall.
- Conclusion: NER weakness is copy/boundary/formulation, not just runaway generation.

Example 3: SIB is formulation-sensitive, not an xLSTM-only collapse.

- Clean official-style xLSTM multi SIB Xho F1 was only `0.1325`.
- Forced-choice xLSTM multi SIB Xho was much healthier, around `0.7187` mean accuracy/F1 in diagnostics and `0.5615`/`0.5260` under summed all-test aggregation.
- Mamba SIB all clean control was strong, with Xhosa F1 `0.5195`, while Mamba Xho-only stayed weak at best F1 `0.1060`.
- Conclusion: task scope/scoring/formulation matters as much as architecture here.

Example 4: T2X has one real positive rescue signal.

- Mixed-source T2X improved validation chrF from `34.38` to `39.47`, BLEU from `0.0712` to `0.0826`, and ROUGE-L from `0.2949` to `0.3587`.
- But examples still show repetition and source-copy problems.
- Conclusion: worth a bounded official/test run, not a claim that generation is solved.

Example 5: AfriHG remains weak.

- Focus-prompt AfriHG stayed around chrF `13.82-14.16`, BLEU `0.0`, with semantically broken headlines.
- Outputs are non-empty, so the failure is not plumbing; it is weak headline conditioning.

## Recommended next research effort

1. Promote only the T2X mixed-source canary to a bounded official/test follow-up, with example inspection and one cheap repetition-controlled decode comparison if needed.
2. Do not run broad NER HPO. Instead, design a formulation experiment around span copying/boundary control, possibly using tag-sequence diagnostics as an intermediate bridge.
3. For SIB, pause GPU-heavy work and decide the reporting protocol: official lm-eval generation/loglikelihood rows versus forced-choice diagnostic rows, and how to compare architectures fairly.
4. Keep task-specific mono/multi xLSTM as the main downstream comparison. Treat the general adapter as a negative finding about mixed-format universal adaptation.
5. For AfriHG, do not spend broad compute until we have a better formulation or data intervention. Current evidence says focus prompting is not enough.

## Files and artifacts to mention

- This advisor note: `sallm_memory/notes/2026-06-11-advisor-meeting-weekly-update.md`
- xLSTM downstream/gate logs: `sallm_memory/notes/2026-06-04.md`
- General adapter and base few-shot notes: `sallm_memory/notes/2026-06-05.md`, `sallm_memory/notes/2026-06-06.md`, `sallm_memory/notes/2026-06-07.md`, `sallm_memory/notes/2026-06-08.md`
- Generation rescue result: `sallm_memory/notes/2026-06-09.md`
- Base few-shot staging:
  - `outputs/analysis/base_fewshot_mamba_xlstm_20260606_sheet_ready.csv`
  - `outputs/analysis/base_fewshot_mamba_xlstm_20260606_sheet_updates.json`
- xLSTM root-cause analysis:
  - `outputs/analysis/xlstm_rootcause_20260608/analysis.md`
  - `outputs/analysis/xlstm_rootcause_20260608/analysis.json`
- SIB formulation audit:
  - `outputs/analysis/sib_formulation_audit_20260608.json`
- NER decode controls:
  - `outputs/analysis/xlstm_ner_decode_controls_20260608.json`
- Generation example audit:
  - `outputs/analysis/xlstm_generation_example_audit_20260608.json`
- T2X/AfriHG rescue artifacts:
  - `outputs/hex_results/diagnostics/xlstm_generation_rescue_20260608/`
