# SALLM advisor meeting update - 2026-07-16

Coverage window: Thursday 2026-07-09 through Wednesday 2026-07-15.

## Short version

The main completed result this week is that the 125M shallow/wide GatedDeltaNet base run finished, published privately, and the base evaluation stack was repaired enough to cover the intended base packs. The downstream base scores are weak so far, which is expected for a raw untuned 125M base model, but the architecture is no longer just a queued experiment: it trained to completion and can be evaluated.

What is not final yet: GDN fine-tuned adapter evaluation. Most adapter fine-tunes completed, but the first adapter eval pass was invalid because the launcher accidentally retained old Mamba `peft_adapter` settings. Corrected adapter evals were still being rerun/moved to Kombuys in the notes, so do not report adapter numbers as final.

## Completed GDN base pretraining

The full GDN pretraining run completed successfully:

- Job: `997801` (`gdn33w-b6-l40s-r4`)
- Status: `COMPLETED`, exit `0:0`
- Completion: `2026-07-08T22:54:32`
- Runtime: `1-19:05:30`
- Final artifacts: `final_model`, `checkpoint-37500`, `checkpoint-38590`
- Final logged test loss: `3.3288254737854004`
- Test runtime: `74.4044s`
- Samples/s: `259.944`
- Private Hub model: `anrilombard/sallm-gated-deltanet-125m-shallowwide-4x40-20260707`

Interpretation: this is now a completed base-pretraining milestone. It does not yet prove downstream competitiveness, but it removes the earlier uncertainty about whether the GDN config could train and checkpoint at SALLM scale.

## Base model evaluation status

The private GDN Hub model loaded successfully in downstream evaluation jobs. Early raw base generation scores are weak:

| Job | Task | Metric |
| --- | --- | --- |
| `1007349` | T2X Xhosa | chrF `6.2571`, BLEU `0.0`, ROUGE-L `0.00335` |
| `1007351` | AfriHG Xhosa | chrF `9.6993`, BLEU `0.0`, ROUGE-L `0.00159` |
| `1007352` | AfriHG Zulu | chrF `9.5554`, BLEU `0.0`, ROUGE-L `0.00140` |

This is not surprising: these are zero-shot/base-model evaluations, not fine-tuned adapters. The correct interpretation is that the generation path works, but raw base generation is not competitive.

By 2026-07-14, the base GDN evaluation directory contained results for all seven packed base evaluation packs:

- AfriGSM
- AfriMMLU
- AfriXNLI
- AfriSenti
- InJoGoIntent
- MasakhaNER
- MasakhaNEWS

Earlier monolingual/base coverage also included:

- T2X Xhosa
- AfriHG Xhosa/Zulu
- MasakhaNER Xhosa/Zulu/Setswana
- MasakhaNEWS Xhosa/English
- MasakhaPOS Xhosa/Zulu/Setswana/all
- SIB-all

Metric consolidation is still a gap. The notes record coverage and some raw generation numbers, but not a clean final table for every base pack.

## Evaluation repairs made this week

Several failures were evaluator/launcher issues, not GDN model failures.

1. HF publish job used the wrong execution environment
   - `1006692` failed because batch env could not find conda env `sallm-uv`.
   - The upload was moved off GPU to CPU-only `ada`/`maths`.
   - `1007348` completed in `00:00:27` and uploaded the private model.

2. Missing lm-eval passthrough field
   - Jobs `1007353`-`1007361` failed early with:
     - `AttributeError: 'ModelEvalConfig' object has no attribute 'lm_eval_model_args'`
   - Fixed by restoring `lm_eval_model_args` in `ModelEvalConfig` and runner formatting.
   - Canary `1011723` passed the former crash point, loaded the model on CUDA, selected five MasakhaNER prompt tasks, built 4,085 requests, and entered generation.

3. POS result serialization
   - POS jobs `1011728`-`1011731` completed generation but failed writing `results.json`.
   - Error: `TypeError: Object of type function is not JSON serializable`
   - Fixed `_to_serializable` to encode callables as stable `module.qualname`.
   - POS reruns `1015856`-`1015859` then ran through successfully.

4. Slurm submit-directory assumptions
   - Multilingual jobs `1016375` and `1016630` failed immediately because the launcher was submitted from `/home` and could not resolve repo paths/env.
   - Repaired `launch_evaluation.sh` and `launch_finetune.sh` to locate `~/masters/sallm` independently of `SLURM_SUBMIT_DIR`, normalize Hydra overrides, and fall back to repo `.venv`.

5. FLA TileLang backend issue
   - Base multilingual eval `1020196` failed after model/data load with:
     - `AssertionError: tilelang lib root do not exists: []`
   - Fixed by setting `FLA_DISABLE_BACKEND_DISPATCH=1` for GDN evaluation configs.

6. AfriMMLU patched import issue
   - `1020440` saved AfriGSM, then failed loading AfriMMLU because patched `utils.py` omitted `weighted_f1_score`.
   - Restored the import for direct prompts 1-5.

7. Non-instruction tokenizer chat-template issue
   - `1029326` saved AfriGSM, AfriMMLU, AfriXNLI, and AfriSenti, then failed when InJoGoIntent requested a chat template that the base tokenizer does not have.
   - Remainder job `1032641` disabled chat templating for InJoGoIntent, MasakhaNER, and MasakhaNEWS.
   - `1032641` completed in `1h03m`.

Interpretation: the evaluation stack is now much more robust for GDN/Qwen3Next-style models. The repeated failures were useful because they exposed concrete infrastructure assumptions inherited from earlier LLaMA/Mamba/xLSTM paths.

## GDN adapter fine-tuning and invalidated evals

Canonical GDN adapter fine-tunes:

- 29 planned adapter fine-tunes.
- 28 completed successfully by 2026-07-12.
- The failed one was Tsonga sentiment job `1020358`; it failed before training because config template ID `afrisenti_v1` did not match the registry path-based ID.
- Fixed template ID to `afrisenti_sentiment_classification/v1`.
- Replacement fine-tune `1029325` completed in `27m14s`, pushed private model `anrilombard/sallm-gated_deltanet-masakhane-afrisenti-tso`, and removed its local checkpoint.

First adapter evaluation attempt:

- Array: `1029327_[0-28%4]`
- Result: 28 completed tasks and one failed task (`_2`, AfriHG Xhosa).
- Inspection showed the launcher replaced only `eval_model.checkpoint` while retaining each Mamba config's old `peft_adapter`.
- Therefore all first-pass adapter eval results are invalid.

Fix:

- Updated `scripts/launch_gdn_adapter_eval_array.sh` to load:
  - the GDN base checkpoint
  - the matching GDN PEFT adapter
- Corrected outputs use `_r2` directories.
- Old invalid outputs were preserved initially, then explicitly invalid first-pass outputs were removed later after user approval.

Second adapter evaluation issue:

- Corrected array `1032642` exposed a Hydra struct-mode override issue.
- Task 2 completed because its config already defined `eval_model.peft_adapter`.
- The other 28 tasks rejected the override because the key was absent under struct mode.
- Fix: use `++eval.eval_model.peft_adapter=...` so the field can be added or replaced.
- Retry array `1038759_[0-1,3-28%4]` was submitted, then later cancelled and moved to Kombuys.

Current adapter-eval state in notes:

- Adapter evals were moved to Kombuys tmux session `gdn-adapter-evals` on RTX 5090 GPU 0.
- First SIB run loaded and merged successfully and entered evaluation context construction.
- By the last note, it was still progressing through multilingual generation tasks, with automatic batch-size fallback handling a batch-64 OOM by retrying at 32.

Interpretation: adapter training is mostly complete, but adapter evaluation is not final. Do not use first-pass adapter eval results.

## Scratch and cleanup

Scratch became a material operational risk:

- 2026-07-13: scratch reached `97.3%`.
- 2026-07-14: scratch reached `97.7%`.
- Main consumers included HF datasets around `49G`, HF hub cache around `5.2G`, and checkpoints around `33G`.

Cleanup with explicit approval:

- Removed 46G cached completed-pretraining dataset:
  - `hf/datasets/anrilombard___mzansi-text-tokenized`
- Removed 1.4G of explicitly invalid first-pass GDN adapter evaluation outputs.
- Removed disposable general/Triton/W&B caches.
- Preserved all checkpoints, valid base results, corrected `_r2` results, and HF model cache.
- Freed approximately `48G`.

Interpretation: cleanup was evidence-preserving. It removed caches and known-invalid results, not useful checkpoints or valid evaluation artifacts.

## New 252M LLaMA canary

This is not the main GDN result, but it matters for next architecture/baseline planning.

Motivation:

- After GDN base completion, a 252M LLaMA baseline is being tested as a larger transformer comparison.
- No 400M run yet.
- Recommended allocation was HEX A100 for 252M only after canary, with Kombuys used for serial fine-tuning/evaluation.

Canary result:

- Job: `1039108`
- Model size: `251,708,160` parameters
- Hardware: 3x A100 40GB
- Steps: 300
- Throughput after startup: about `1.13`-`1.15s/step`
- Loss: `11.2471` in first logged window to `8.4301` final window
- `checkpoint-250` saved successfully, 1.5G, including model, optimizer, scheduler, RNG states, tokenizer, and trainer state.
- Final canary model saved successfully.
- Slurm recorded `COMPLETED`, exit `0`.

Full 252M run:

- Four-GPU pending job `1039160` was cancelled to avoid waiting.
- Submitted 3x A100 full job `1039185`.
- Started: `2026-07-14 20:36:39` SAST.
- Runtime override: `64,538` steps at global batch `36`.
- Token slots: `4,758,257,664`, only `49,152` above the four-GPU target.
- First 38 steps reproduced stable `1.13`-`1.14s/step`.
- Projected runtime: about `20.3`-`20.6h` plus checkpoints.

Interpretation: the 252M LLaMA path is operationally promising, but it is in flight and should not be reported as a result yet.

## Data/source planning

MzansiText already includes:

- WURA
- mC4
- CulturaX
- Glot500-c
- NCHLT Text
- CC100
- WMT2022 ParaCrawl monolingual data
- Inkuba-Mono

New source candidates:

- FineWeb2
- HPLT v2
- Autshumato
- Vuk'uzenzele / government text

FineWeb2 metadata for weak target languages:

- nbl: 5 MB
- nso: 12 MB
- ssw: 4 MB
- tsn: 11 MB
- tso: 13 MB
- ven: 6 MB
- xho: 289 MB
- zul: 219 MB
- Total: about 559 MB
- Weakest six: about 51 MB

Interpretation: FineWeb2 can help balance underrepresented languages, but it is not enough alone for model-scale growth.

## Current recommendations

1. Consolidate GDN base metrics into a single table before making claims beyond "base trained and eval stack works."
2. Treat base GDN generation scores as weak zero-shot baselines, not final downstream performance.
3. Finish corrected adapter evaluations before comparing GDN to xLSTM/Mamba/LLaMA downstream results.
4. Keep invalid adapter-eval pass out of the results sheet.
5. If the 252M LLaMA full run completes, compare it as a scaling baseline, not as a replacement for the 125M architecture comparison.
6. Do FineWeb2 audit first for target-language balancing; HPLT v2 second; Vuk'uzenzele is cleaner but likely narrower.

## Immediate next steps

1. Pull/consolidate all GDN base eval metrics from the completed result directories.
2. Monitor Kombuys `gdn-adapter-evals` until all corrected adapter evals finish.
3. Update Google Sheet only with corrected GDN base/adapted rows, never first-pass invalid adapter rows.
4. Monitor 252M LLaMA `1039185`; if complete, audit checkpoint/save and only then plan evals.
5. Keep `sallm_progress.md` updated only for completed defensible milestones.
