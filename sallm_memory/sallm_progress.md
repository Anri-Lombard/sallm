# SALLM Progress

## Durable update — GDN Xhosa NER HPO final (2026-08-01)

- Validation-only 24-trial screen `1150968` plus three-seed finalist jobs `1151254/1151255` selected `tyvese0z` (`eval/all_f1` mean `0.6085`, sample SD `0.0097`, range `0.5978-0.6165`) over `b888dh4l` (`0.5858`, SD `0.0336`). Representative seed-42 `checkpoint-552` was frozen rather than selecting the lucky highest seed.
- Official held-out job `1151321` returned P1-P5 F1 `0.5192/0.6269/0.6239/0.6156/0.6287`; canonical headline explicitly **best prompt P5 `0.6287`**, mean `0.6029`, range `0.5192-0.6287`, labelled descriptive rather than unbiased. Result and full provenance are promoted to `GatedDeltaNet Results!E4`; `Comparison Data!I44=0.6287`.
- This strongly supports optimization/capacity as a major confound in the former mono-versus-multi NER gap, but matched Mono/Multi exposure controls across Xhosa/Zulu/Tswana remain required before attributing the residual pattern to architecture.

## Durable update — GDN General interpretation (2026-08-02)

- GDN General is not broadly superior: its matched advantage over Multi is confined to MasakhaPOS (`~0.015-0.019`), while it is worse on NER, SIB, Intent, AfriHG, and corrected News. The current General loader samples all six task families equally (`1/6` each) regardless of dataset size, a direct mixture/exposure confound that can oversample small structured tasks such as POS. Literature supports mixture balancing, capacity, and negative-transfer controls, but does not establish a GDN-specific General advantage. Do not make an architecture claim until update/token-matched mixture ablations and a matched non-GDN control are complete.
- The historical GDN General recipe is recovered from HEX job `1020357`, W&B
  `i77vz2jm`, and Hub commit `4c3635d3c15fac6d7cacc8904f389c17bbc3ab5f`:
  equal `1/6` task sampling, 43,637 examples, one epoch / 2,728 updates, max
  length 1024, batch 4 with accumulation 4, LR `8e-5`, cosine, warmup `0.03`,
  weight decay `0.01`, seed 42, and rank-16 / alpha-32 / dropout-0.05 LoRA on
  all seven GDN/attention projection targets. This is the reproducible anchor
  for the controlled mixture study, not evidence of a GDN-specific advantage.

## Purpose

Compare decoder-only architecture families for South African low-resource
language modelling and downstream task performance. The central question is not
only which model gets the highest score, but what each architecture costs and
where it fails under realistic low-resource constraints.

The comprehensive final registry for optimized results is the Google Sheet:
[SALLM results sheet](https://docs.google.com/spreadsheets/d/1Ph_zVcSuLZy0dUBnDPF4tkLqAfEVCwK9JVybB2_8x6U/edit?ouid=101226728847680131120&usp=sheets_home&ths=true).
Use this sheet as the source of final best-result rows. Use this Obsidian vault
for daily notes, experimental context, failure analysis, and defensibility
decisions around those final rows.

## Fair Comparison Policy

- Mamba rescue experiments must be interpreted against matched LLaMA controls
  whenever formulation, decoding, scoring, or result-selection changes.
- Each rescue must also be compared against the current best Mamba result for
  that task, so we can separate "Mamba got better" from "Mamba caught up to
  LLaMA" and from "both models benefit from a cleaner protocol".
- If an intervention improves both Mamba and LLaMA, record it as a shared
  decoder-only protocol improvement, not as a Mamba-only rescue. Shared
  improvements should be considered for later LLaMA reruns as well as Mamba.
- Keep a carry-forward backlog for shared improvements discovered during Mamba
  debugging, so useful protocol changes can be applied to LLaMA after the
  defensible Mamba recipe is selected.
- Matched xLSTM Base AfriXNLI/AfriMMLU 3-shot task-native accuracy is final
  from job `1139425` (`0:0`). Raw results verify `num_fewshot=3`, validation
  demonstrations, official test evaluation, and five complete prompts.
  Best-prompt headlines are AfriXNLI Eng/Xho/Zul/Sot
  `0.3433/0.3433/0.3517/0.3433` and AfriMMLU
  `0.2160/0.2440/0.2260/0.2180`; full provenance is in
  `XLSTM Results!D28:D35` notes. Legacy F1 rows are excluded. The shared
  summary writer now records effective/raw `num_fewshot` rather than the pack
  default.
- When a Mamba-focused experiment reveals a method that improves both Mamba and
  LLaMA, record that explicitly and plan to apply it to the final LLaMA rerun;
  those are protocol improvements, not evidence that Mamba alone was fixed.
- Corrected final Intent reporting now uses mean continuation-token
  log-probability, validation-F1 checkpoint selection, and one frozen official
  held-out test. Final clean multilingual-adapter best-prompt F1 values are:
  LLaMA Eng/Xho/Zul/Sot `0.0039/0.0033/0.0043/0.0043`, Mamba
  `0.0191/0.0096/0.0199/0.0250`, GDN `0.0058/0.0012/0.0091/0.0093`, and xLSTM
  `0.0733/0.1232/0.1025/0.1374`. These best-of-test-prompts values are
  descriptive, not unbiased; prompt means, ranges, winners, and all prompt
  values are retained in the canonical sheet provenance.
- Matched LLaMA and Mamba Base standardized suites are final from jobs
  `1139430/1139432` (`0:0`). Raw results verify exact base-checkpoint identity,
  effective `num_fewshot=3`, official test evaluation, and five complete
  prompts. Best-prompt AfriXNLI Accuracy for Xho/Zul/Sot/Eng is LLaMA
  `0.3533/0.3417/0.3767/0.3467` and Mamba
  `0.3567/0.3517/0.3400/0.3400`; AfriMMLU Accuracy is LLaMA
  `0.2320/0.2360/0.2560/0.2640` and Mamba
  `0.2140/0.2420/0.2380/0.2440`; AfriMGSM flexible exact match is LLaMA
  `0.0160/0.0080/0.0160/0.0120` and Mamba
  `0.0080/0.0120/0.0120/0.0160`. AfriXNLI/AfriMMLU demonstrations use
  validation. AfriMGSM exposes no separate few-shot split, so its three-shot
  results are matched benchmark-style values rather than clean independent
  train-demo estimates. Full prompt provenance is in the source-cell notes.
- Corrected General-adapter Intent evaluation is final for Mamba and GDN under
  the same mean continuation-token scorer and official held-out split. Mamba
  Eng/Xho/Zul/Sot best-prompt F1 is
  `0.0009909/0.0012195/0.0012195/0.0012195`; GDN is
  `0.0019991/0.0035620/0.0027329/0.0042349`. These are valid near-chance
  results, not missing runs. The clean xLSTM General replacement remains in
  training and is not promoted.
- Mamba General AfriMGSM is final under the standardized held-out
  task-native flexible-exact-match protocol. Xho/Zul/Sot/Eng best-prompt
  values are `0.0040/0.0080/0.0040/0.0040` from job `1135843`; prompt means,
  ranges, tied winners, all prompt values, artifact, and SHA are retained in
  `Mamba Results!G36:G39`.
- Corrected GDN monolingual Intent is final for Eng/Xho/Zul/Sot under the same
  official held-out mean continuation-token scorer. Best-prompt F1 is
  `0.0017254/0.0074846/0.0012214/0.0017021` from jobs
  `1137019/1137020/1137021/1137022`.
- Corrected xLSTM monolingual Intent is final for Eng/Xho/Zul/Sot under the
  same official held-out mean continuation-token scorer. Best-prompt F1 is
  `0.0016476/0.0053704/0.0057437/0.0106330` from jobs
  `1137487/1137489/1137491/1137493`. Validation-only selection froze
  checkpoints `165/252/63/504`. These are valid near-chance results, not
  missing runs.
- Mamba base POS is complete on the official held-out test under constrained
  mean label-token scoring: Xho `0.0360`, Zul `0.0211`, and Tsn `0.2157`
  best-prompt token accuracy (job `1131802`).
- Corrected Mamba monolingual Xhosa, Zulu, and Tswana POS are complete under
  the same constrained official-test protocol: best-prompt token accuracy is
  `0.0000`/`0.0421`/`0.0000` (array `1142391`; sacct components
  `1142392/1142393/1142391`). All three are verified negative results rather
  than missing runs: predictions collapse almost entirely to the legal `X`
  label.
- Mamba base MasakhaNER is complete on the corrected parquet-backed official
  held-out test: Xho, Zul, and Tsn are all `0.0000` F1 for every one of five
  prompts (jobs `1131946`-`1131948`). These are valid negative results, not
  missing runs: generation overgenerates/repeats and the entity extractor
  usually returns empty. The verified artifacts are under
  `/scratch/lmbanr001/masters/sallm/results/eval/mamba_base_closeout_20260729/`.
- The canonical four-architecture tabs contain no live `Queued`, `Running`,
  `Pending`, or `TODO` result statuses. Remaining unavailable values are
  explicitly labelled as quarantined or not applicable. Every populated
  task-language row now has an explicit Base/Mono/Multi/General entry rather
  than a blank cell. The legacy Transformer SIB200 Tswana row is not a missing
  run: the matched SIB200 suite intentionally covers only Afrikaans, English,
  Northern Sotho, Southern Sotho, Xhosa, and Zulu.
- If an intervention helps only one architecture, keep it clearly labelled as
  model-specific in the final comparison.
- Once the Mamba recipe is defensible, rerun or consolidate downstream
  evaluations so the final Mamba-vs-LLaMA comparison uses fair matched splits,
  prompts, metric scripts, decoding/result-selection rules, and documented
  model-specific exceptions.
- The final downstream suite should be treated as a confirmation phase: Mamba
  must be compared against both the previous best Mamba result and the matched
  best LLaMA result under fair conditions.
- Once the defensible Mamba recipe is selected, apply any confirmed shared
  decoder-only improvements to the LLaMA rerun where compatible, so the final
  result compares optimized recipes rather than a rescued Mamba against a stale
  transformer baseline.
- A fair final comparison must not compare a rescued Mamba recipe against a
  stale LLaMA result that lacks shared improvements found during Mamba rescue.
- Do not treat diagnostic improvements as final leaderboard claims until the
  matched final downstream evaluation has been run or an explicit exception is
  documented.

## Architecture Roadmap

1. **Transformer baseline**
   - First milestone: train and evaluate a transformer baseline.
   - This is the anchor for downstream metrics, training stability, and compute
     expectations.
   - Current comparison reference is the LLaMA-style decoder-only transformer
     baseline.

2. **Mamba**
   - Current milestone: train and evaluate a Mamba decoder-only model across
     the full downstream evaluation suite.
   - Mamba is being tested with the same broad downstream families where
     possible: classification, sequence labelling, headline generation, and
     translation.
   - The current work is still within decoder-only fine-tuning, decoding,
     formulation, metric-hygiene, and recipe-recovery gates. Architecture or
     pretraining changes should come only after these are exhausted.

3. **Future xLSTM / ExcelSTM-style model**
   - Planned future milestone: train an xLSTM-style model for the same
     downstream suite.
   - This should be compared against both the transformer baseline and Mamba.

4. **Other candidate architectures**
   - Add only when they answer a concrete comparison question: quality,
     data-efficiency, compute efficiency, memory footprint, multilingual
     transfer, or robustness under low-resource data.

## Metrics to Track Across Architectures

For every model family, record:

- Parameter count and trainable parameter count.
- Pretraining data size, language mixture, and training-token budget.
- Fine-tuning method: full fine-tune, LoRA, adapter, or other.
- Fine-tuning hyperparameters: learning rate, weight decay, schedule, warmup,
  batch size, gradient accumulation, epochs, label smoothing, sequence length,
  checkpoint-selection metric.
- Hardware and runtime: GPU type, number of GPUs, wall time, memory pressure,
  batch-size fallback, and failures.
- Evaluation harness version and task pack.
- Decoding strategy: greedy, beam, sampling, repetition penalty, length penalty,
  max tokens, and whether any constrained or guided output method was used.
- Task metrics:
  - Classification: accuracy, macro/weighted F1, per-language F1 where useful.
  - Sequence labelling: token accuracy, strict/length-penalized token accuracy,
    parseable-label rate, BIO/entity F1, non-O recall.
  - Generation: chrF, BLEU, ROUGE-L, plus raw-output sanity checks.
- Error profile: empty outputs, overgeneration, repetition, malformed labels,
  copied input, wrong language, no parseable tags, or task-template mismatch.
- Defensibility status: final, needs rerun, confounded, diagnostic only, or
  superseded.

## Current High-Level Status

### Public Release Usability

- The public `anrilombard/mzansilm-125m` Hugging Face model now has a repaired
  `model.safetensors` file and updated model-card usage instructions as of Hub
  commit `7f017bc71c53c19c1fd122e773ad1f60c5d30826`.
- The default Transformers 4.x load path was verified with the repaired
  safetensors file: no `meta` parameters remain after load and generation works.
- Transformers 5 still rejects the LLaMA config because the model uses explicit
  `head_dim=56` with `hidden_size=512` and `num_attention_heads=9`; the public
  instructions therefore pin `transformers>=4.52.4,<5`.

### Transformer Baseline

- Transformer/LLaMA baseline has already produced strong reference results for
  several generation tasks.
- Known references from the current local notes:
  - T2X Xho: LLaMA chrF about `53.98` best, with current-stack parity recheck
    around `53.48`.
  - AfriHG Xho: older local best chrF about `20.22`; current parity recheck
    `14.20`.
  - AfriHG Zul: older local best chrF about `23.00`; current parity recheck
    `21.56`.
- Exact final POS/NER baseline entries still need to be consolidated from the
  benchmark recordings/sheet before final writing.

### Mamba

- Mamba is not uniformly broken:
  - MasakhaNews monolingual parity recheck is strong after the task-template
    bug fix: Eng best F1 `0.773`, Xho best F1 `0.757`.
  - T2X is recipe-sensitive: recovered bqueawk-style full fine-tune reached
    chrF `32.18`, much better than weak current reproductions around chrF
    `20`, but still far below LLaMA around chrF `54`.
- Mamba remains weak or fragile on:
  - POS under official free-generation-style scoring and tag-sequence variants.
  - NER under free-generation BIO/tag-sequence scoring, despite signs of
    teacher-forced learning.
  - AfriHG, where recovered recipes improve over old weak runs but remain well
    below LLaMA.
- The final pre-hybrid POS/NER decoder-only rescue gate is now negative:
  D4 full-data atomic tag-sequence LoRA did not rescue POS or NER. Best D4 POS
  validation token accuracy is `0.1960` with exact length `0.0400` and high
  overgeneration on the better nonempty arm. D4 NER validation remains near the
  old narrow entity signal with non-`O` recall `0.1772`, BIO F1 `0.1769`, and
  all-`O` rate `0.9375`.

### xLSTM

- Strict-125M HF xLSTM is now viable in this repo after installing the official
  `xlstm` stack and adding xLSTM-specific chunked generation/eval handling.
  The current strict shape is `h736_l12_h4_chunk64` with `126,901,952`
  parameters.
- Corrected base MasakhaNER held-out test job `1126106` completed after the
  exact-head-dimension inference repair. Xhosa, Zulu, and Tswana are all
  `0.0000` F1 on every one of five prompts (best prompt = mean = range
  `0.0000`). Complete test aliases contain 1,000 Xhosa/Zulu and 996 Tswana
  rows per prompt. Treat this as a valid negative base result; generated
  outputs copy/continue input or extract no entities.
- Corrected POS test scoring has overturned the earlier "very strong POS"
  interpretation. The old POS scorer treated list-valued targets as multiple
  acceptable answers and inflated token accuracy. Under the patched serialized
  tag-sequence test pack, xLSTM multi-POS test is only moderate: best token
  accuracy is Tswana `0.328`, Xhosa `0.357`, and Zulu `0.390`, with exact
  length match below `0.20`. Treat earlier POS values around `0.93` as invalid.
- The fair full-base xLSTM run
  `xlstm_h736_ctx2048_native_4gpu_ddp_llama_budget_20260524` completed
  successfully as job `861849` in `1-05:59:05`, using a LLaMA-budget-style
  2048-context token-slot target of `4.758B` and reaching reported epoch
  `2.1513`. Its Trainer loss logs are affected by a Transformers/xLSTM
  loss-scaling issue under 4-GPU DDP, so use the clean/pretrain audit scripts
  rather than raw Trainer loss for final comparisons.
- Full xLSTM clean generation-loss validation is positive against the current
  Mamba base on all three audited generation gates, but still behind LLaMA:
  xLSTM final weighted NLL/PPL is T2X Xho `5.6584` / `286.70`, AfriHG Xho
  `5.7048` / `300.30`, and AfriHG Zul `5.8647` / `352.39`. Current Mamba base
  is `5.8515` / `347.77`, `6.0834` / `438.50`, and `6.4066` / `605.85`;
  LLaMA base is `4.1620` / `64.20`, `5.3254` / `205.48`, and `5.5885` /
  `267.32`.
- Full xLSTM status is now promising-positive as a Mamba replacement candidate
  for downstream evaluation. Repaired pretrain-loss audit confirms the same
  story: xLSTM final weighted NLL/PPL is `2.9250` / `18.63`, better than the
  current Mamba base at `3.8065` / `44.99` but behind LLaMA base at `2.3010` /
  `9.98`. The checkpoint-30000 loss-scale probe gives weighted NLL/PPL
  `2.6888` / `14.71`, confirming the raw 4-GPU Trainer eval-loss values were
  inflated and should not be read as perplexity. Final base-gate classification:
  positive versus Mamba, negative versus LLaMA; proceed to downstream xLSTM
  fine-tuning/evaluation if compute allows.
- The first strict-125M xLSTM 10k streaming base screen completed as jobs
  `861747` -> `861748`. It trained cleanly in `02:09:31`; the clean generation
  loss audit completed in `00:04:53`.
- xLSTM checkpoint-10000 has held-out trainer eval loss `3.310756`, down from
  `3.607671` at checkpoint-5000. On clean generation-loss validation it beats
  the current Mamba base on T2X Xho (`283.95` vs `347.77` PPL), but is worse
  than the current Mamba base on AfriHG Xho (`627.67` vs `438.50`) and AfriHG
  Zul (`852.06` vs `605.85`). LLaMA base remains substantially better on all
  three tasks (`64.20`, `205.48`, `267.32` PPL).
- Current xLSTM classification: ambiguous/promising. It is not a failed base
  screen like the recent fresh-Mamba and shallow-hybrid runs, but it is not yet
  a base-wide replacement for the current Mamba base or LLaMA. Prioritize
  xLSTM downstream adaptation or a longer xLSTM base screen before spending
  more compute on low-probability cheap Mamba rescue arms.

Current open gate:

- AfriHG lower-LR/no-smoothing is closed as a negative HPO gate: Xho chrF
  `3.19`, Zul chrF `4.87`, both below the recovered monolingual Mamba bests.
- Current best Mamba AfriHG is now the checkpoint-selected rerun: Xho chrF
  `11.53` and Zul chrF `13.57`.
- Checkpoint-selection-by-validation-generation-metric is producing useful
  signal. T2X Xho checkpoint-244 beat final on validation chrF (`32.68` vs
  `31.40`), AfriHG Xho checkpoint-656 beat final on validation chrF (`10.68`
  vs `9.54`), and AfriHG Zul checkpoint-892 beat final on validation chrF
  (`14.07` vs `13.09`). These three checkpoints should be promoted to official
  test-set lm-eval reruns. The official reruns are now submitted as serial
  L40S jobs `843499`-`843501`.
- Longer training remains a plausible next gate for Mamba generation if the
  checkpoint-selected test reruns improve, because all three validation wins
  came from later saved checkpoints rather than final/best-loss checkpoints.
- Official checkpoint-selected test reruns are complete and improve all three
  generation tasks checked:
  - T2X Xho checkpoint-244: chrF `32.61`, above prior Mamba best `32.18`.
  - AfriHG Xho checkpoint-656: chrF `11.53`, above prior Mamba best `9.90`.
  - AfriHG Zul checkpoint-892: chrF `13.57`, above prior Mamba best `12.09`.
- This closes the current saved-checkpoint selection gate, but it does not close
  the Mamba-vs-LLaMA generation gap. A small longer-training/continuation gate
  with validation chrF/ROUGE checkpoint selection is now justified before
  architecture-level claims.
- Root-cause forensics show Mamba generation is systematically shorter than
  LLaMA/reference outputs on T2X and AfriHG, with more repetition on T2X. The
  first longer-training gate has been submitted. Initial jobs `843512` ->
  `843513` and `843518` -> `843519` were pre-training Hydra override errors and
  were repaired; `843520` started training but failed at epoch eval because the
  inherited best-model metric still expected disabled trainer chrF metrics. The
  corrected T2X checkpoint-244 continuation jobs `843526` -> `843527`
  completed successfully. Official test chrF improved again to `33.1243`
  (BLEU `0.0577`, ROUGE-L `0.3015`), but output forensics still show Mamba
  under-generates: mean prediction length is about `10.22` tokens versus
  `27.37` reference tokens and about `12.74` for LLaMA.
- User approved deleting the old diagnostics checkpoint directory; it was
  removed, and `purequota` refreshed to scratch `72.7%`.
- The corrected serial AfriHG continuation gate completed on L40S one GPU:
  `843532` Xho finetune, `843533` Xho eval, `843534` Zul finetune, and repaired
  Zul eval `843571` after original eval `843535` failed before scoring due an
  `eval_model.adapter=null` config override. The continuation gate is negative:
  Xho chrF `10.3470` versus checkpoint-selected `11.5335`; Zul chrF `12.6551`
  versus checkpoint-selected `13.5726`. Both continuation runs shortened the
  outputs relative to their selected checkpoints. Do not promote AfriHG
  continuation; keep the checkpoint-selected AfriHG results as current Mamba
  best.
- Base-model lineage audit now has a matched held-out validation loss result:
  using the same tokenizer, same validation prompts, and same target token
  burden, Mamba base has substantially worse teacher-forced target likelihood
  than LLaMA base. Validation PPLs are T2X Xho `331.24` vs LLaMA `62.26`,
  AfriHG Xho `301.75` vs LLaMA `110.41`, and AfriHG Zul `401.35` vs LLaMA
  `157.94`. This makes base-model quality/lineage a real contributor to the
  generation gap, while tokenizer mismatch is ruled out as the architectures
  share the tokenizer.
- Decoder-only output-shape rescue is now checked on validation for current
  best Mamba generation checkpoints. T2X did not improve: stronger length
  controls caused severe overgeneration/repetition and lower chrF. AfriHG is
  more promising: beam5/lp1.2 improves Xho validation chrF from `10.95` to
  `12.81` and Zul validation chrF from `14.21` to `17.00`; Zul also improves
  ROUGE-L and length ratio cleanly. These AfriHG beam5/lp1.2 settings are
  validation-selected diagnostics and should be considered for official test
  reruns before changing any final Google Sheet rows.
- Base-issue validation control wave completed as jobs `848475` -> `848476`
  -> `848477`, all exit `0:0`, with artifacts pulled locally to
  `outputs/eval/diagnostics/mamba_base_issue_validation_20260519/`.
  The cleaner base generation-loss audit confirms the gap: Mamba base PPL is
  `347.81` vs LLaMA `64.20` on T2X Xho, `438.73` vs `205.48` on AfriHG Xho,
  and `606.04` vs `267.32` on AfriHG Zul. The selected Mamba fine-tuned
  checkpoints improve massively but remain behind task-specific LLaMA:
  T2X `18.73` vs LLaMA `5.31`, AfriHG Xho `21.04` vs `13.28`, AfriHG Zul
  `19.28` vs `10.79`; Mamba is close to SA-general LLaMA on AfriHG but not
  the task-specific LLaMA references. The held-out pretraining-style audit is
  the strongest root-cause signal: on the same `302278` validation tokens,
  Mamba base PPL is `45.00` versus LLaMA base `9.98`. Current recommendation:
  treat base checkpoint quality/training recipe as the main remaining root
  cause; run a short continued-pretraining recovery from the existing Mamba
  base before committing to a full new base pretrain.
- Scratch is now a blocker for any new HEX submission: quota reached `98.9%`
  after the HF dataset cache materialized. Top consumers are
  `/scratch/lmbanr001/masters` `64G` and `/scratch/lmbanr001/hf` `26G`. No
  cleanup was performed; deletion/cleanup approval is needed before further
  experiments.
- User approved proceeding; `/scratch/lmbanr001/hf` was deleted and scratch
  quota refreshed to `73.2%`. A bounded streaming continued-pretraining
  recovery wave is now active as jobs `848482` -> `848483` -> `848484` ->
  `848485` on L40S one GPU. The three recovery jobs run `300` steps from
  `anrilombard/sallm-mamba-125m` with LRs `5e-5`, `1e-4`, and `2e-4`, using HF
  streaming mode to avoid recreating the full dataset cache. The final job runs
  clean generation-loss audit over the three recovered checkpoints plus LLaMA
  base. This is a diagnostic/recovery gate, not a final fair-parity replacement
  for a fresh optimized base run.
- Base-model recommendation: if the short continued-pretraining wave improves
  Mamba base/conditional losses substantially, proceed to a fresh Mamba base
  HPO/retrain for the final architecture comparison. If it does not improve,
  skip longer continuation and move directly to fresh-base recipe search. A
  final defensible architecture comparison should include base validation PPL,
  downstream conditional PPL, task metrics, output shape metrics, token budget,
  GPU hours, parameter count, and inference throughput/memory.
- First continued-pretraining chain `848482` -> `848485` failed at eval due
  Mamba cache output hygiene (`Mamba2Cache` could not be fp32-cast by
  Accelerate), not due training divergence. The script was patched to disable
  `use_cache`, stranded jobs were cancelled, and the repaired chain is now
  `848486` -> `848487` -> `848488` -> `848489`; heartbeat updated accordingly.
- Second continued-pretraining chain `848486` -> `848489` passed step-100 eval
  (`eval_loss=3.8641`) but failed at checkpoint save because tied embeddings
  required `save_safetensors=false`. The script was patched with
  `save_safetensors=False`, stranded jobs were cancelled, and the active chain
  is now `848490` -> `848491` -> `848492` -> `848493`.
- Continued-pretraining recovery wave completed successfully as `848490` ->
  `848493`. On the 512-sample held-out pretraining slice, LR `5e-5` was best
  (`eval_loss=3.8625`), LR `1e-4` was nearly tied (`3.8635`), and LR `2e-4`
  was worse (`3.8902`). The downstream clean generation-loss audit did not
  improve over the starting Mamba base: LR `5e-5` PPLs were T2X `351.80`,
  AfriHG Xho `446.38`, AfriHG Zul `618.10`, versus the prior Mamba base
  `347.81` / `438.73` / `606.04`; higher LRs were worse. Conclusion: cheap
  300-step continued pretraining does not rescue the Mamba base. The next
  serious base step should be a fresh Mamba base recipe/HPO rather than longer
  blind continuation.
- Fresh Mamba base probe is active as `848494` -> `848495` -> `848496`, serial
  L40S one-GPU. It trains two fresh-from-config 3k-step Mamba probes
  (`lr2e-4_wu200` and `lr4e-4_wu200`) with streaming data and then runs the
  clean generation-loss audit against current Mamba and LLaMA bases. This is a
  bounded recipe diagnostic, not the final full Mamba base retrain.
- Fresh Mamba base probe completed successfully as `848494` -> `848496`. LR
  `4e-4` was clearly better than LR `2e-4` on held-out pretraining loss
  (`6.3838` vs `7.2619`), but both 3k-step fresh-from-random checkpoints were
  far worse than the current Mamba base on downstream clean generation loss
  (PPLs in the ~9.6k-17.3k range vs current Mamba base `347.81`/`438.73`/
  `606.04`). Interpretation: the short probe is useful for recipe direction
  but not a replacement base; a serious fresh Mamba base run needs a much
  longer token budget and should start from the faster-learning `4e-4` recipe
  direction unless longer-run stability argues otherwise.
- Longer fresh Mamba base pilot submitted as `850743` -> `850744`, serial L40S
  one-GPU. It trains fresh Mamba at LR `4e-4`, warmup `1000`, `20k` steps,
  saves/evals every `5k`, then audits clean generation loss for checkpoints
  `5k/10k/15k/20k/final` against current Mamba and LLaMA bases.
- Longer fresh Mamba base pilot completed successfully as `850743` -> `850744`
  and is negative for this exact pure-Mamba recipe. Best held-out pretraining
  loss was already at `checkpoint-5000` (`5.9383`), then worsened at `10k`
  (`6.0468`), `15k` (`6.1576`), and `20k` (`6.1647`). Clean generation-loss
  remains orders of magnitude worse than the current Mamba base and LLaMA base:
  best fresh PPLs are roughly T2X `15052`, AfriHG Xho `12102`, AfriHG Zul
  `11898`, versus current Mamba base `348`/`439`/`606` and LLaMA base
  `64`/`205`/`267`. This argues against simply extending the current
  fresh-from-random recipe; next pure-Mamba gate should be shape/recipe HPO in
  the same 120-130M band.
- The next pure-Mamba gate first ran as `852296` -> `852297`, serial one-GPU
  L40S. It keeps the parameter count matched but changes the base shape to
  `hidden_size=704`, `num_hidden_layers=24`, `expand=2`, `state_size=128`
  (`126,811,888` trainable params) and reruns the same 20k pretraining/audit
  structure. The first attempt reached the step-5000 evaluation point and then
  failed in the Mamba2 fast causal-conv kernel because the dynamic eval tensor
  layout did not satisfy the stride multiple-of-8 requirement; this is an
  implementation/kernel-layout issue, not a model-quality result. The repaired
  pad-to-multiple-of-8 chain ran as `854071` -> `854072` under run id
  `mamba_fresh_base_wide_e2_s128_pad8_20k_20260521`, but it failed at the same
  first-eval `causal_conv1d` stride constraint and the dependent audit was
  cancelled. Padding sequence length is therefore not enough; this is now an
  HF/Mamba2 fast-kernel layout blocker for the wide shape rather than a
  scientific loss result. The stale heartbeat was deleted.
- Mamba architecture research refresh on 2026-05-20 found hybrid Mamba2 to be
  the strongest current SSM-family literature direction, but the user wants to
  stay on pure Mamba/Mamba2 ("PMamba") for now before changing architecture.
  Immediate next gates should therefore be pure-Mamba base recipe/shape HPO,
  not hybrid. Mamba-3 is noted as a later research/feasibility item rather
  than an immediate pretraining-quality rescue.
- The wide pure-Mamba2 shape gate is now being retried with eval forced through
  the HF torch path while keeping fast fused CUDA training. The first
  torch-eval canary exposed a local patch bug; the second proved the binding
  fix but OOMed in torch eval at batch size `8` and context `2048`. The active
  corrected chain is now `854987` -> `854988` -> `854989` under run id
  `mamba_fresh_base_wide_e2_s128_torcheval3_20k_20260521`, using
  `eval_batch_size=1` and `eval_max_length=1024`, watched by heartbeat
  `sallm-mamba-wide-torch-eval-gate-watch`. This is still an
  implementation-repair gate, not yet a model-quality result.
- Root-cause rescue checklist has started under an active Codex goal. A1
  decoder-only POS/NER constrained label scoring completed as jobs `855022`-
  `855025`. It proves output shape alone is not enough: exact-length parseable
  outputs are `100%`, but POS remains very weak for both Mamba and LLaMA under
  this scoring prompt. Mamba NER sum scoring is a confirmed all-`O` collapse
  despite high token accuracy (`0.7420`): non-`O` recall and global entity F1
  are `0.0`. Mean scoring avoids all-`O` by overpredicting non-entity labels,
  not by recovering spans. Next narrow gate is A3 NER label-prior calibration.
- A3 Mamba NER scalar `O`-bias calibration completed as jobs `855026`-
  `855030` and is negative. Small penalties leave validation pure all-`O`;
  stronger penalties break all-`O` but flood the output with mostly `i-date`/
  `i-org` labels. Best global entity F1 is only `0.0019` at `o=-4.0` with
  `1` true-positive entity and `411` false-positive entities. Conclusion:
  NER is not rescued by a simple label-prior bias; next POS/NER gates should
  test constrained generation or formulation/training controls.
- A2 true prefix-constrained tag-sequence generation ran as jobs
  `855032`-`855035`, matched across Mamba/LLaMA and POS/NER. This tested the
  stricter version of the output-shape hypothesis by using actual
  decoder-only `model.generate` constrained to legal tag-label paths.
- A2 completed successfully and is negative as a rescue. It guarantees
  exact-length, parseable POS/NER outputs (`1.0` for all four runs), but Mamba
  POS still collapses mostly to `cconj`, Mamba NER floods `b-date`/`i-date`/
  `i-org`, and LLaMA controls are also weak under the same formulation.
  Conclusion: POS/NER failure is not simply malformed free generation; next
  POS/NER work should test formulation/training controls, with any shared
  improvements carried forward to both Mamba and LLaMA.
- C1 AfriHG official test rerun is active, testing whether validation-selected
  beam5/lp1.2 improves official AfriHG test metrics over the current
  checkpoint-selected beam5/lp0.7 baselines (Xho chrF `11.5335`, Zul chrF
  `13.5726`). If it improves Mamba, a matched LLaMA decode-variant check
  should be considered before final fair comparison.
- C1 is already positive for Xho: checkpoint-656 official test with beam5/lp1.2
  reached chrF `15.1702`, up from the prior checkpoint-selected Xho baseline
  `11.5335`. The first Zul job `855037` failed due Hydra override hygiene and
  was repaired/resubmitted as `855040`.
- C1 completed and is positive for Mamba AfriHG in both languages. Official
  test beam5/lp1.2 gives Xho chrF `15.1702` and Zul chrF `17.1245`, improving
  over checkpoint-selected baselines by about `+3.6` chrF each. Output length
  moves closer to references with low empty/repetition rates, but examples
  still show generic or wrong-focus headlines. Keep beam5/lp1.2 as the current
  Mamba AfriHG decoding recipe and run/plan a matched LLaMA decode check before
  final fair claims.
- C1b matched LLaMA AfriHG beam5/lp1.2 check is complete. The first attempt
  failed due stale LLaMA config checkpoint paths, but repaired durable-path jobs
  `855054` -> `855055` completed and were pulled locally. Beam5/lp1.2 is not a
  shared AfriHG protocol: it improves Mamba, but LLaMA Xho chrF `13.1149` and
  Zul chrF `20.8709` are below tracked LLaMA baselines and over-generate
  relative to concise headlines. Carry-forward tag: `Mamba-only`.
- C2 AfriHG hook/focus forensics is complete for the current C1/C1b evidence.
  Residual Mamba errors are mostly generic/off-topic and wrong-focus headline
  selection, not empty output: Xho off-topic/generic `37.2%` and too
  short/generic `31.9%`; Zul off-topic/generic `54.7%` and too short/generic
  `14.9%`. Matched LLaMA examples show stronger content preservation but
  over-generation under the Mamba-rescue decode setting, supporting semantic
  focus/headline planning and base quality as the remaining AfriHG issues.
- B3 T2X source-preservation forensics has started from current pulled
  outputs. Mamba checkpoint-selected T2X test has chrF `32.6125` versus LLaMA
  current-stack parity chrF `53.4796`. Crude exact source-token preservation is
  much lower for Mamba: entity coverage `0.241` vs LLaMA `0.511`, value
  coverage `0.202` vs `0.437`, and repetition `0.082` vs `0.034`. This
  supports entity/value preservation as a real T2X gap, while B1
  teacher-forced token-class loss remains needed for stronger causal evidence.
- B1 T2X teacher-forced token-class loss audit ran as job `855076` on one L40S
  GPU. The new script scores assistant/reference target
  tokens by exact source-token class (`source_entity`, `source_value`,
  `source_relation`, `other`) and compares current Mamba T2X continuation
  against LLaMA T2X opt-chrF on validation.
- B1 completed and strengthens the T2X root-cause story. On validation
  teacher-forced scoring, Mamba is worse than LLaMA on all target-token
  classes, but the gap is much larger for exact source entity/value tokens:
  entity NLL gap `+2.7656` with `15.89x` PPL ratio, value NLL gap `+2.3065`
  with `10.04x` PPL ratio, versus `other` token gap `+1.1783` with `3.25x`
  PPL ratio. This supports B2 placeholder/reinsertion as the next T2X rescue
  gate.
- D1 wide pure-Mamba2 base gate completed pretraining as job `854988`, but the
  dependent clean-loss audit `854989` failed before producing metrics. Final
  pretraining remains weak for the "train longer fixes it" hypothesis:
  checkpoint-5000 is still best with eval loss `5.6133`, while checkpoint-10000
  worsened to `5.6522`, checkpoint-15000 to `5.6954`, and checkpoint-20000 to
  `5.6942` despite improved training loss. The audit failed with the Mamba2
  fast-kernel channel-last stride error
  `causal_conv1d with channel last layout requires strides ... to be multiples
  of 8`, leaving the downstream clean generation-loss evidence missing. D1 is
  therefore incomplete/failed, not a defensible base-candidate decision yet.
  Next step is to repair and rerun only the clean-loss audit path once scratch
  pressure is safe; scratch was `89.3%` at the failure check.
- D1 clean-loss audit repair is now active as job `855987`. The local repair
  adds `--mamba-torch-forward` to `run_generation_loss_audit_clean.py`, using
  HF's torch Mamba2 eval path to avoid the fused causal-conv stride-layout
  failure. Local syntax/Ruff/help checks passed, the repair-only submitter was
  synced to HEX, and the job started on one L40S GPU with scratch still high
  at `89.3%`. A fresh 30-minute heartbeat
  `sallm-mamba-d1-clean-loss-repair-watch` is watching `855987` and should
  pull/summarize/classify D1 when `summary.json` is available.
- D1 wide pure-Mamba2 torch-eval base gate is now closed negative. Repaired
  audit job `855987` completed `0:0`, artifacts were pulled, and
  `base_gate_summary.md` was generated. The repair fixed the audit path, but
  the model-quality result is poor: best D1 fresh-wide clean-loss PPLs are T2X
  `8470`, AfriHG Xho `11315`, AfriHG Zul `11720`, versus current Mamba base
  `348`/`439`/`606` and LLaMA base `64`/`206`/`268`. D1 should not feed D3;
  no new base candidate exists. The D1-negative branch is now active: run E1
  implementation parity and B3/C3 formulation/decoding gates before designing
  another D2 base HPO matrix. Scratch remains high at `89.3%`, so no more HEX
  submissions without explicit user approval or cleanup.
- Post-D1 readiness audit: E1/B3/C3 submitters still pass local syntax checks,
  and E1/B3 diagnostic scripts pass local py_compile/Ruff. HEX queue is empty,
  but scratch remains `89.3%`; no job was submitted. Recommended next move is
  E1 first, because it is the lowest-footprint remaining gate and directly
  tests the HF-vs-official Mamba implementation-parity confounder. B3/C3 should
  wait until E1 closes or scratch pressure is reduced.
- E1 HF-vs-official Mamba logits parity completed as job `856253` and is a
  parity failure/caveat. HF and official `mamba_ssm` parameter counts match and
  the state load has no missing/unexpected keys, but logits are not close
  (`all_logits_close_at_1e-4=false`), greedy next token differs on two of three
  prompts, and max absolute logit deltas are `9.31`-`16.01`. This does not
  invalidate HF-Mamba vs HF-LLaMA downstream comparisons, because SALLM used
  the HF path, but it weakens broad architecture claims about official Mamba2.
  Continue B3/C3 as practical HF-Mamba rescue gates and keep final wording
  scoped to the implementation/config path unless official-path parity is
  repaired later.
- B3 T2X source-preservation decoding diagnostic completed as repaired job
  `856269` after two immediate structured-config failures were fixed. It is
  negative as a rescue. Mamba baseline validation chrF is `28.06`; the
  source-checklist prompt is only `28.15` and lowers entity/value coverage.
  Repetition-control variants reduce repetition but collapse chrF to `20.14`
  and `21.74`. LLaMA baseline remains much stronger: chrF `45.89`, entity
  coverage `0.763`, value coverage `0.594`, versus Mamba `0.410`/`0.320`.
  Conclusion: T2X source entity/value preservation is a real Mamba gap, but
  simple decoder-only prompt/checklist or repetition-control decoding does not
  fix it. Do not carry B3 prompt/decoding variants into final reruns.
- C3 AfriHG focus-prompt validation completed as repaired serial L40S chain
  `856290` -> `856293`. It is a positive Mamba validation candidate, especially
  for Zul: Mamba Xho improved from validation chrF `12.8140` to `13.1258`, and
  Mamba Zul improved from `17.0020` to `18.4924`. Matched focus-prompt LLaMA
  controls scored Xho chrF `13.2699` and Zul chrF `19.2154`, but the current
  pulled evidence lacks a matched LLaMA non-focus validation baseline, so this
  is not yet a shared protocol improvement. Carry-forward tag:
  `Mamba validation candidate / shared status ambiguous`; confirm on official
  Mamba test before final recipe adoption.
- C4 AfriHG focus-prompt official Mamba test confirmation is active as jobs
  `856306` -> `856307`, watched by heartbeat
  `sallm-c4-mamba-afrihg-focus-test-watch`. Scratch was `89.3%` at submission;
  top consumers were checked first and no deletion was performed. C4 compares
  focus-prompt test metrics against C1 official beam5/lp1.2 baselines: Xho chrF
  `15.1702`, Zul chrF `17.1245`.
- C4 completed and is ambiguous-to-negative as a final AfriHG recipe gate. Xho
  regressed from C1 chrF `15.1702` to `14.4983` and ROUGE-L `0.0572` to
  `0.0553`; Zul improved only slightly from chrF `17.1245` to `17.3358` and
  ROUGE-L `0.0736` to `0.0782`. Do not adopt `focus_v1` as final. Keep C1
  beam5/lp1.2 as current Mamba AfriHG official-test recipe; record focus_v1 as
  a validation-only/language-specific prompt idea.
- D2 pure-Mamba base HPO is now locally prepared but not submitted. Added
  `scripts/submit_d2_mamba_base_hpo_screen_2026_05_22.sh`, a one-arm-at-a-time
  screen with canary -> 10k train -> clean generation-loss audit. Local `bash -n`
  and usage checks pass. Do not submit while scratch is `89.4%`; cleanup/approval
  is needed before adding checkpoint-heavy base runs. Preferred first arm after
  cleanup is `current_lr2e4_wu2000_10k`.
- Cleanup audit for D2 is ready. Queue is empty; scratch is still `89.4%`.
  Four audited base-HPO checkpoint dirs could free roughly `10.6G`:
  old 3k probe, old expand4/state64 20k, D1 wide 20k, and the D1 canary. The
  three scientific runs have local summaries/loss rows and recorded negative
  decisions; the canary is not a scientific result. No deletion has been
  performed.
- D2 was patched to reduce scratch footprint before submission:
  `run_mamba_fresh_pretrain_streaming.py` now supports `--skip-final-save`, and
  the D2 submitter uses it so screening arms do not duplicate `final_model`.
  D2 will audit `checkpoint-5000` and `checkpoint-10000` only. Local validation
  passes (`bash -n`, `py_compile`, Python Ruff, and help output). Still do not
  submit at scratch `89.4%` without cleanup/approval.
- Lower-footprint D2 code is now also staged on HEX and passes remote
  `bash -n`/`py_compile`; no job submitted. Scratch remains `89.4%`, queue is
  empty, and deletion still requires explicit approval.
- Added an advisor-ready current evidence matrix:
  `sallm_memory/mamba_root_cause_rescue_evidence_matrix.md`. D2 submitter now
  supports `--dry-run`, and the remote dry run for first arm
  `current_lr2e4_wu2000_10k` works without submitting jobs.
- User approved cleanup for D2. Deleted only the audited old base-HPO checkpoint
  dirs, repaired the D2 scratch guard to use `/scratch/slurm/bin/purequota`,
  and submitted first D2 arm `current_lr2e4_wu2000_10k` as jobs `856320` ->
  `856322`. Heartbeat `sallm-d2-mamba-base-hpo-screen-watch` is active. Latest
  light status: canary `856320` completed, train `856321` running, clean-loss
  audit `856322` dependency-pending, scratch about `79.6%`. No D2 result yet.
- D2 first arm has a negative checkpoint-5000 signal: train loss `6.3609`,
  eval loss `6.5037`, eval PPL about `667.6`, worse than D1 wide checkpoint-5000
  eval loss `5.6133`. Keep waiting for checkpoint-10000 and clean-loss audit
  before final D2 classification.
- D2 checkpoint-10000 remains weak: train loss `6.2177`, eval loss `6.4936`,
  eval PPL about `660.9`. It is only slightly better than checkpoint-5000, so
  the current-shape lower-LR/longer-warmup arm is unlikely to be a credible
  base candidate unless the clean-loss audit surprisingly disagrees.
- D2 first arm `current_lr2e4_wu2000_10k` closed negative. Clean generation-loss
  checkpoint-10000 PPLs are far worse than the current Mamba base: T2X Xho
  `11925.1` vs `347.8`, AfriHG Xho `26758.1` vs `438.5`, AfriHG Zul
  `28854.2` vs `605.9`; LLaMA base remains lower still (`64.0`, `205.6`,
  `267.5`). Do not use this D2 base for D3. Lower LR/longer warmup alone does
  not rescue the current pure-Mamba shape.
- D2 second arm `wide_lr2e4_wu2000_10k` was submitted as jobs `861386` ->
  `861388`.
  This tests the wide expand2/state128 shape with lower LR `2e-4` and warmup
  `2000`, because the current-shape LR/warmup rescue failed. Scratch was
  `81.0%` at submission. This is retained as historical submission context;
  the arm has since closed below.
- D2 second arm `wide_lr2e4_wu2000_10k` closed negative. Jobs `861386` ->
  `861388` all completed `0:0`; artifacts were pulled and summarized locally.
  Trainer eval improved mildly by checkpoint-10000 (`eval_loss=6.4619`, PPL
  about `640.3`), but clean generation-loss rejected the arm: best wide
  checkpoint is checkpoint-5000 with PPLs T2X Xho `9983.6`, AfriHG Xho
  `21768.7`, AfriHG Zul `20378.4`, still far worse than current Mamba base
  (`347.8`, `438.5`, `605.9`) and LLaMA base (`64.0`, `205.6`, `267.5`).
  Do not use the wide D2 base for D3. Recommended base follow-up decision:
  stop pure-Mamba base HPO and report the base-rescue path as negative; the
  remaining conservative wide arm (`wide_lr1e4_wu1000_10k`) is optional only if
  an exhaustive ablation table is worth another low-probability run.
- B2 T2X placeholder/reinsertion diagnostic is now running as job `855304` on
  one L40S GPU. It replaces exact source entity/value strings with
  placeholders, generates with matched Mamba/LLaMA checkpoints, deterministically
  reinserts the original strings, then reports placeholder preservation,
  entity/value preservation, and final chrF/BLEU/ROUGE. This directly tests
  whether the B1/B3 T2X source-token gap is fixable by removing copy burden.
- B2 completed and is negative as a Mamba rescue. Mamba generated zero expected
  placeholders in the `216` examples where the placeholdered reference expected
  at least one placeholder, giving reinserted chrF `8.2455`. LLaMA generated at
  least one expected placeholder in `87.5%` of those examples, exact placeholder
  set `29.6%`, and reinserted chrF `27.2327`, but value preservation remains
  weak. Carry-forward tag: `LLaMA-only` diagnostic improvement / `Negative`
  Mamba rescue.
- B3 is now prepared but not submitted. The local diagnostic script
  `scripts/run_t2x_source_preservation_decoding_diagnostic.py` and submitter
  `scripts/submit_b3_t2x_source_preservation_decode_2026_05_21.sh` passed
  syntax/help/ruff checks. It will run matched Mamba/LLaMA validation variants
  for baseline beam3/lp1.2, greedy repetition control, conservative beam
  repetition control, and a source-preservation checklist prompt, then score
  chrF/BLEU/ROUGE plus entity/value preservation and repetition. The two B3
  scripts were narrowly rsynced to HEX. Hold submission while D1 is active and
  scratch remains above `85%`.
- Latest live state at 2026-05-21 23:15 SAST: C1b is closed; D1 job `854988`
  is still running around step `9140/20000` on L40S and D1 audit `854989` is
  dependency-pending. Scratch remains `86.9%`, with checkpoints `54G`, results
  `15G`, and logs `92M`; there is still no D1 artifact beyond
  `checkpoint-5000/trainer_state.json`.
- Follow-up live state at 2026-05-21 23:22 SAST: D1 job `854988` remains
  healthy around step `9528/20000`; `854989` is still dependency-pending.
  Scratch/top consumers are unchanged, and no new D1 artifact is available.
- Follow-up live state at 2026-05-21 23:25 SAST: D1 job `854988` remains
  healthy around step `9670/20000`; `854989` is still dependency-pending.
  Scratch/top consumers are unchanged, and no new D1 artifact is available.
- Follow-up live state at 2026-05-21 23:27 SAST: D1 job `854988` remains
  healthy around step `9750/20000`; `854989` is still dependency-pending.
  Scratch/top consumers are unchanged, and no new D1 artifact is available.
- Follow-up live state at 2026-05-21 23:29 SAST: D1 job `854988` remains
  healthy around step `9834/20000`; `854989` is still dependency-pending.
  Scratch/top consumers are unchanged, and no new D1 artifact is available.
- Follow-up live state at 2026-05-21 23:33 SAST: D1 job `854988` is still
  `RUNNING` on L40S `srvrocgpu012` after about `03:20:48`; `854989` remains
  dependency-pending. Scratch is still `86.9%` with top consumers checkpoints
  `54G`, results `15G`, and logs `92M`. The log tail is active inside an eval
  loop with no traceback, but the only discoverable D1 artifact remains
  `checkpoint-5000/trainer_state.json`, so no new local summary or decision is
  available yet.
- Follow-up live state at 2026-05-21 23:35 SAST: checkpoint-10000
  `trainer_state.json` appeared and was pulled locally. Checkpoint-10000 train
  loss improved to `5.2401`, but eval loss worsened to `5.6522` from
  checkpoint-5000's `5.6133`; trainer best still points to checkpoint-5000.
  This remains better than the earlier failed/current expand4/state64 20k
  run's 5k eval loss `5.9383`, but D1 is no longer a clean monotonic-loss
  story. Wait for checkpoint-15000/20000 and the clean generation-loss audit
  before deciding whether this is a real base-candidate improvement.
- Follow-up live state at 2026-05-21 23:38 SAST: D1 job `854988` is still
  running around step `10212/20000` (`51%`) after the checkpoint-10000 eval.
  Scratch remains `87.6%` with checkpoints `55G`, results `15G`, and logs
  `92M`; `854989` is still dependency-pending. No new D1 artifact is available
  beyond checkpoint-5000/checkpoint-10000 trainer states.
- Follow-up live state at 2026-05-21 23:39 SAST: D1 job `854988` remains
  healthy around step `10298/20000` (`51%`); `854989` is still
  dependency-pending. Scratch/top consumers are unchanged at `87.6%`,
  checkpoints `55G`, results `15G`, logs `92M`. No new artifact is available.
- Follow-up live state at 2026-05-21 23:41 SAST: D1 job `854988` remains
  healthy around step `10382/20000` (`52%`); `854989` is still
  dependency-pending. Scratch/top consumers are unchanged at `87.6%`,
  checkpoints `55G`, results `15G`, logs `92M`. No new artifact is available,
  so wait for the heartbeat rather than continuing manual minute-by-minute
  checks.
- Follow-up live state at 2026-05-21 23:43 SAST: D1 job `854988` remains
  healthy around step `10470/20000` (`52%`); `854989` is still
  dependency-pending. Scratch/top consumers are unchanged at `87.6%`,
  checkpoints `55G`, results `15G`, logs `92M`. No new artifact is available;
  leave the next check to the existing heartbeat unless a user decision is
  needed.
- Follow-up live state at 2026-05-21 23:48 SAST: D1 job `854988` remains
  healthy around step `10736/20000` (`54%`) on L40S `srvrocgpu012`; `854989`
  is still dependency-pending. Scratch/top consumers remain `87.6%`,
  checkpoints `55G`, results `15G`, logs `92M`. Artifact search still finds
  only checkpoint-5000/checkpoint-10000 trainer states, so no new D1 summary or
  recipe decision is available yet.
- Follow-up live state at 2026-05-21 23:51 SAST: D1 job `854988` remains
  healthy around step `10885/20000` (`54%`) on L40S `srvrocgpu012`; `854989`
  is still dependency-pending. Scratch/top consumers remain unchanged at
  `87.6%`, checkpoints `55G`, results `15G`, logs `92M`. There is still no
  checkpoint-15000, final pretrain summary, or clean-loss diagnostic, so D1
  remains active with no new decision.
- Follow-up live state at 2026-05-21 23:52 SAST: D1 job `854988` remains
  healthy around step `10972/20000` (`55%`) on L40S `srvrocgpu012`; `854989`
  is still dependency-pending. Scratch/top consumers remain unchanged at
  `87.6%`, checkpoints `55G`, results `15G`, logs `92M`. There is still no
  checkpoint-15000, final pretrain summary, or clean-loss diagnostic.
- Follow-up live state at 2026-05-21 23:54 SAST: D1 job `854988` remains
  healthy around step `11050/20000` (`55%`) on L40S `srvrocgpu012`; `854989`
  is still dependency-pending. Scratch/top consumers remain unchanged at
  `87.6%`, checkpoints `55G`, results `15G`, logs `92M`. There is still no
  checkpoint-15000, final pretrain summary, or clean-loss diagnostic.
- Follow-up live state at 2026-05-21 23:55 SAST: D1 job `854988` remains
  healthy around step `11124/20000` (`56%`) on L40S `srvrocgpu012`; `854989`
  is still dependency-pending. Scratch/top consumers remain unchanged at
  `87.6%`, checkpoints `55G`, results `15G`, logs `92M`. There is still no
  checkpoint-15000, final pretrain summary, or clean-loss diagnostic. Leave
  further watching to the heartbeat until a new artifact or failure appears.
- E2 Mamba generation parity smoke is now running as job `855312`. It checks
  representative Mamba T2X and AfriHG Xho generation under cache on/off and
  batch1/batch4 greedy settings to rule out evaluation-setting instability as
  a confounder. Immediate log tail showed Mamba CUDA kernels available.
- E2 completed and found an implementation-hygiene issue, not a rescue:
  Mamba generation did not crash, but outputs were not perfectly invariant to
  cache/batch settings. T2X exact match to batch1/cache-off was `7/8` under
  cache-on or batch4 variants; AfriHG Xho was `6/8` for cache-on variants and
  `7/8` for batch4/cache-off. Final Mamba generation evaluations should pin
  and document cache/batch settings, preferably with conservative
  batch1/cache-off reruns or explicit harness-setting documentation.
- Fair-comparison guardrail: any decoder-only improvement that helps both
  Mamba and LLaMA should be recorded as a shared protocol improvement and
  carried into later optimized LLaMA reruns where compatible, not treated as a
  Mamba-only win. Likewise, a LLaMA-only improvement should stay in the notes
  as a useful later LLaMA recipe candidate and as evidence about which fixes do
  or do not transfer to Mamba. Once the Mamba recipe is defensible enough for
  the downstream suite, final Mamba-vs-LLaMA reporting must rerun or align both
  architectures under the same splits, prompts, metrics, decoding/selection
  rules, and documented shared improvements before updating the comprehensive
  Google Sheet. Each checklist result should now carry a `Mamba-only`,
  `LLaMA-only`, `Shared`, or `Negative` tag so later recipe carry-forward is
  auditable. The final comparison has two distinct questions: did the chosen
  recipe improve over the previous best Mamba result, and is the optimized
  Mamba recipe fairly competitive with the optimized LLaMA transformer
  baseline?
- D5 Mamba-2 hybrid base screen completed as `861602` -> `861604` and is
  negative. The `hybrid_126m_waleffe8attn` arm (`126357846` params, LR `4e-4`,
  warmup `2000`, `10000` steps, 2 attention layers out of 24 / 8.3%
  attention) improved held-out pretraining eval from `5.7042` at
  checkpoint-5000 to `5.6123` at checkpoint-10000, which is better than recent
  pure-Mamba fresh HPO arms. However, clean generation-loss remains orders of
  magnitude worse than the current Mamba and LLaMA bases:
  checkpoint-10000 PPLs are T2X Xho `12364.7`, AfriHG Xho `14020.5`, AfriHG
  Zul `14634.8`, versus current Mamba base `347.8`/`438.5`/`605.9` and LLaMA
  base `64.0`/`205.6`/`267.5`. Do not use D5 as a base candidate; record it
  as architecture-follow-up evidence that shallow sparse attention improves
  training loss shape but does not rescue downstream conditional likelihood at
  this budget. Do not generalize this to all hybrids until the custom hybrid
  attention implementation is audited for positional encoding/RoPE and intended
  block structure.

### xLSTM

- xLSTM literature and citation notes are recorded in
  `sallm_memory/xlstm_architecture_literature.md`. The strongest directly
  relevant recipe signal is Beck et al. 2024: 125M-ish xLSTM uses embedding
  dim `768`, `24` blocks, `4` heads/head dim `384`, context `2048`, AdamW
  betas `(0.9, 0.95)`, eps `1e-5`, grad clip `1.0`, warmup `750`, cosine
  decay to 10% peak LR, weight decay `0.1`, and no positional encoding.
- HF/official xLSTM viability probes are complete enough to justify a strict
  125M xLSTM screen. HF xLSTM is viable for training on HEX only when
  `xlstm`/`mlstm-kernels` are installed. The repo-compatible `hidden=768`,
  12-layer HF shape has `135,368,544` params and passed forward/backward,
  finite gradients, save/load, and a 6-step overfit smoke on L40S. After the
  kernels were installed, the stricter `hidden=736`, 12-layer HF shape also
  passed forward/backward/save-load and tiny overfit at `126,901,952` params,
  so it is now the preferred xLSTM base-screen shape under the user's
  125M-parameter constraint.
- Official NX-AI xLSTM vanilla backend can instantiate and forward/backward
  under a comparable 12-layer config, but it is larger (`143,818,064` params)
  and not parameter-matched to the HF baseline. Official CUDA sLSTM backend
  failed extension build on HEX, so it is not the next mainline path.
- Current xLSTM generation status: stock HF `generate()` fails chunk-size
  assertions after the prompt chunk (`65` or `1` tokens not divisible by chunk
  size `64` depending on cache mode), but the SALLM evaluation path now has an
  xLSTM-only chunked generation helper and a passing smoke test. For base
  screening, use the streaming HF `hidden=736`, 12-layer path rather than TRL's
  full-dataset materialization path.
- Final xLSTM 3-epoch base retrain `xlstm_h736_ctx2048_native_4gpu_ddp_3epoch_resume_20260531`
  completed cleanly on L40S. The main train `880318` reached `67498/67498`
  steps in `1-17:39:46` with `train_loss=3.85037` and trainer-state epoch
  `3.0871`; the afterany resume job `880319` resumed from `checkpoint-67498`
  and no-opped cleanly; audits `880320` and `880321` completed cleanly. The
  held-out pretrain-style audit is positive versus Mamba but still behind
  LLaMA: xLSTM final mean NLL/token `2.97127`, PPL `17.48`; Mamba base PPL
  `44.99`; LLaMA base PPL `9.98`. The clean generation-loss audit is not
  competitive with LLaMA: xLSTM final PPLs are T2X Xho `479.63`, AfriHG Xho
  `497.88`, AfriHG Zul `568.79`, versus LLaMA `64.20`/`205.48`/`267.32`.
  Against current Mamba, xLSTM is worse on T2X Xho and AfriHG Xho but slightly
  better on AfriHG Zul (`568.79` vs `605.85`). Classify this run as useful
  architecture evidence, not a downstream base replacement for LLaMA.
- The completed xLSTM base export was published privately to Hugging Face as
  `anrilombard/sallm-xlstm-125m-native-3epoch-20260531` at commit
  `ba2ff845335c8cbf750f8f6f3ebc09468008fef9`. After publication, approved
  pretraining scratch cleanup removed old xLSTM canaries/intermediate
  checkpoints and the `anrilombard___mzansi-text-tokenized` cache; the retained
  current-run scratch payload is `final_model`, `fresh_pretrain_summary.json`,
  and best checkpoint `checkpoint-60000`.
- xLSTM downstream evaluation is complete through mono, multilingual, and
  general waves. The downstream story is split: task-specific mono/multi
  adapters are alive, but the single general adapter is not a good universal
  adapter. Corrected News evals recovered strongly for task-specific adapters
  (mono English/Xhosa best F1 about `0.913`/`0.920`; multilingual English/Xhosa
  about `0.881`/`0.919`). POS originally looked very strong in the multi gate,
  but the 2026-06-11 audit found that the old lm-eval target-list setup
  inflated the reported token accuracies. The final corrected constrained
  test-split POS result now shows a more defensible moderate xLSTM score, not
  the earlier near-`0.93` story: multilingual xLSTM best token accuracy is
  Tswana/Xhosa/Zulu `0.7245`/`0.6933`/`0.7313`.
  NER is modest but nonzero for task-specific adapters. In contrast, the general adapter largely
  collapses structured extraction/classification: News drops to English/Xhosa
  best F1 `0.6582`/`0.1929`, NER best F1s are near zero
  (`0.0094`/`0.0038`/`0.0035`), and POS best token accuracies are only
  `0.0933`/`0.0133`/`0.0200`. Treat task-specific mono/multi results as the
  defensible xLSTM downstream comparison; do not use the general adapter as
  xLSTM's best downstream form. One general eval, `afrihg_eng`, failed because
  no AFriHG English CSV exists on GitHub, so that is a dataset/config issue
  rather than model behavior.
- The xLSTM T2X mixed-source rescue did not transfer from validation to a
  strong official test result. Job `921485` completed cleanly on the official
  T2X Xhosa test split with chrF `34.1600`, BLEU `0.0454`, and ROUGE-L
  `0.2974`: only a small improvement over the prior xLSTM mono T2X official
  result around chrF `33.719`, and still far below LLaMA around chrF `53.976`.
  Treat T2X as an unresolved source-binding/structure problem for xLSTM; the
  next defensible rescue direction is structural delexicalization or
  source-placeholder reinsertion rather than broad decoding HPO.
- Final constrained POS audit update, 2026-06-12: evaluate MasakhaPOS as a
  closed-label token-tagging task on the test split by scoring every token
  against the fixed UPOS label set under the same tuple prompts. This confirms
  the old around-`0.93` xLSTM POS values were inflated, but POS is not a
  complete xLSTM failure under the correct protocol. Best multilingual
  constrained token accuracy by language is:
  - LLaMA: Tswana/Xhosa/Zulu `0.8460`/`0.8220`/`0.8409`.
  - xLSTM: Tswana/Xhosa/Zulu `0.7245`/`0.6933`/`0.7313`.
  - Mamba: Tswana/Xhosa/Zulu `0.2340`/`0.0674`/`0.0402`.
  Interpretation: LLaMA remains clearly strongest for POS under closed-label
  scoring; xLSTM is meaningfully competent but behind LLaMA; Mamba remains
  very weak even when free-generation formatting is removed. The Google Sheet
  multilingual tuned POS cells were updated and verified on 2026-06-12; base,
  monolingual, and general POS cells should not be reinterpreted from this
  constrained multilingual wave.
- xLSTM POS HPO held-out update, 2026-06-18: the HPO-selected multilingual
  xLSTM POS adapter passed the corrected constrained MasakhaPOS `test` gate.
  It uses the same closed-label token-logprob protocol, tuple prompts, and
  languages (`tsn`, `xho`, `zul`). Held-out test token accuracy is:
  - xLSTM HPO-selected multilingual POS: Tswana/Xhosa/Zulu
    `0.7906`/`0.7666`/`0.7855`; macro `0.7809`, micro `0.7824`.
  This supersedes the earlier corrected xLSTM multilingual POS baseline
  (`0.7245`/`0.6933`/`0.7313`) for the tuned xLSTM POS comparison, but it does
  not invalidate the LLaMA/Mamba corrected constrained POS results. Running
  mono/multi/general xLSTM HPO wave2 jobs are validation-only until their own
  selected adapters pass held-out test.
- xLSTM POS wave2 mono held-out update, 2026-06-19: the selected monolingual
  wave2 adapters for Tswana, Xhosa, and Zulu passed the corrected constrained
  MasakhaPOS `test` gate. They use the same closed-label token-logprob
  protocol, tuple prompts, and test split. Held-out token accuracy is Tswana
  `0.7650`, Xhosa `0.7839`, and Zulu `0.8201`; exact sequence accuracy is
  Tswana `0.0228`, Xhosa `0.0611`, and Zulu `0.1090`. These supersede the
  previous unproven monolingual POS sheet cells. Xhosa and Zulu improve over
  their promoted multilingual HPO-selected scores (`0.7666` and `0.7855`);
  Tswana does not improve over the promoted multilingual score `0.7906`.
  Wave2 multilingual held-out POS is defensible but lower than the already
  promoted multilingual HPO result: macro token accuracy `0.7676`, micro
  `0.7638`, with Tswana/Xhosa/Zulu `0.7462`/`0.7758`/`0.7808`, so it was
  recorded but not promoted. General remains HPO validation-only until a
  complete validation artifact and selected held-out eval exist.
- xLSTM HPO-selected held-out downstream update, 2026-06-25: the L40S
  post-HPO test wave completed cleanly for SIB, MasakhaNews, InjonGoIntent,
  AfriHG, T2X Xho, and MasakhaNER as jobs `952247`-`952252`. The held-out
  story is mixed rather than a broad HPO win. MasakhaNews remains strong
  (best F1 English/Xhosa `0.8686`/`0.9238`) and InjonGoIntent is strong for
  English/Xhosa but weaker for Sotho/Zulu (best F1
  `0.9064`/`0.8463` versus `0.6998`/`0.6226`). MasakhaNER is nonzero but still
  modest under flexible extraction (best F1 Tswana/Xhosa/Zulu
  `0.3174`/`0.2538`/`0.2545`). Generation remains below the strongest
  transformer references: T2X Xho chrF `30.81`, AfriHG Xho chrF `15.20`, and
  AfriHG Zul chrF `17.10`. SIB is not rescued by this HPO selection: best
  per-language F1 is only Afrikaans/English/Northern Sotho/Southern
  Sotho/Xhosa/Zulu `0.1260`/`0.1415`/`0.0800`/`0.1032`/`0.1175`/`0.0888`.
  Use the detailed 2026-06-25 note for run provenance and prompt means before
  making sheet-level leaderboard updates.
- xLSTM MasakhaNews protocol correction, 2026-07-28: historical News adapters
  and headline values predated selective training of newly added chat-role
  tokens and classification left truncation, so they are quarantined rather
  than treated as final architecture evidence. The corrected monolingual Xhosa
  recipe selected checkpoint-17 on validation F1 (`0.4095883`) and evaluated
  held-out test exactly once in job `1128440`. Weighted F1 across prompts
  P1-P5 is `0.2758567`/`0.2687654`/`0.1953027`/`0.1916278`/`0.2376085`.
  The canonical headline is explicitly **best prompt** P1 `0.2758567`; mean
  `0.2338322`, range `0.1916278-0.2758567`. This best-of-five test-prompt
  result is descriptive, not an unbiased estimate. Corrected multilingual
  validation selected epoch 4 / checkpoint-272 with
  `eval_classification/all_f1=0.6941930`; its single frozen held-out test job
  `1128810` is queued and must complete before multilingual News promotion.
- Task-head diagnostic update, 2026-06-29/30: held-out supervised task-head
  probes show that the poor generative adapter results do not mean the base
  representations lack task signal. In the matched trainable task-head setting,
  xLSTM is best on NER, Intent, and POS, while LLaMA is best on SIB; the same
  winner pattern appears in the frozen-backbone control. Three-seed core
  results after rerunning Mamba seed 13 with fast-path dependencies are:
  NER-all F1 xLSTM `0.4032 +/- 0.0236`, LLaMA `0.3553 +/- 0.0397`, Mamba
  `0.2491 +/- 0.0105`; SIB-all macro-F1 LLaMA `0.7355 +/- 0.0149`, xLSTM
  `0.7030 +/- 0.0032`, Mamba `0.6205 +/- 0.0115`. The frozen control summary
  confirms the pattern before backbone adaptation: xLSTM beats LLaMA/Mamba on
  NER (`0.3670` vs `0.2526`/`0.2414`), Intent (`0.6577` vs
  `0.4677`/`0.4835`), and POS (`0.6440` vs `0.6210`/`0.5844`), while LLaMA
  wins SIB (`0.6876` vs xLSTM `0.6491`, Mamba `0.6232`). Keep these results
  separate from the original decoder-only generative benchmark: they support a
  "formulation/adaptation failure, not base representation failure" conclusion
  and make task heads a defensible dissertation branch.
- GatedDeltaNet base update, 2026-07-09/14: the shallow/wide 125M GDN base
  pretraining run completed successfully and was published privately as
  `anrilombard/sallm-gated-deltanet-125m-shallowwide-4x40-20260707`. Final
  training job `997801` completed with exit `0:0` after `1-19:05:30`; final
  artifacts include `final_model`, `checkpoint-37500`, and `checkpoint-38590`.
  Final logged test loss was `3.3288254737854004`. Base zero-shot evaluation
  is now operationally covered across the intended packs after several launcher
  and evaluator repairs: generation path, lm-eval schema, POS JSON
  serialization, FLA TileLang dispatch, AfriMMLU import patch, and
  non-instruction tokenizer chat-template handling. The completed base eval
  directory contains results for AfriGSM, AfriMMLU, AfriXNLI, AfriSenti,
  InJoGoIntent, MasakhaNER, and MasakhaNEWS, plus earlier monolingual/base
  packs for T2X, AfriHG, MasakhaNER, MasakhaNEWS, MasakhaPOS, and SIB. Early
  base generation metrics are weak, as expected for an untuned 125M base:
  T2X Xhosa chrF `6.2571`, AfriHG Xhosa chrF `9.6993`, and AfriHG Zulu chrF
  `9.5554`. Treat GDN as a completed base-pretraining and evaluation-stack
  milestone, not yet as a competitive downstream result. Fine-tuned adapter
  evaluations remain pending because the first adapter eval pass retained old
  Mamba `peft_adapter` settings and was invalidated; corrected adapter evals
  are being rerun separately.
- GatedDeltaNet corrected POS update, 2026-07-27: the held-out constrained
  MasakhaPOS matrix now has defensible task-adapter results under the same
  closed-label token-logprob protocol used for the architecture comparison.
  Best-prompt multilingual Tswana/Xhosa/Zulu token accuracy is
  `0.8033`/`0.7855`/`0.8050`; best-prompt monolingual results are
  `0.8112`/`0.2361`/`0.7303`, where Xhosa is an audited genuine `NOUN`
  collapse; best-prompt general-adapter results are
  `0.8181`/`0.8102`/`0.8226`. The prompt means, ranges, and winning prompt IDs
  remain in `notes/2026-07-27-best-prompt-reporting.md`.
  The historical list-target POS scores remain invalid and quarantined.
  Corrected base POS is also complete and promoted. Best-prompt base
  Tswana/Xhosa/Zulu token accuracy is `0.2005`/`0.2217`/`0.2427`; all three
  genuinely collapse toward `NOUN` (`82.5%`/`89.0%`/`89.3%` of best-prompt
  predictions) and have zero exact-sequence accuracy.
- INJOngo Intent protocol audit, 2026-07-27: existing Mamba and LLaMA
  decoder-only Intent rows are not defensible architecture evidence. Summed
  continuation log-probability structurally favours short verbalizers; the old
  validation set included all 622 English test rows; those English test texts
  also occur in train; and the Zulu dev split contains corrupted labels. The
  shared local root fix now uses mean token log-probability, balanced
  prompt/label validation caps, test-text decontamination, and deterministic
  stratified validation derived from the remaining train rows. Focused tests
  pass, but retraining and final promotion remain pending.
- GatedDeltaNet 2/3-shot base closeout, 2026-07-27: all 23 completed non-NER
  artifacts from arrays `1118020/1118021` passed structural audit and were
  promoted using best-prompt headlines. Three-shot News F1 is Eng `0.1592`
  and Xho `0.2083`; SIB best-prompt F1 spans `0.1695-0.1893` (2-shot) and
  `0.1676-0.2094` (3-shot). AfriXNLI reaches `0.3317/0.3567`, AfriMMLU
  `0.2720/0.2360`, and AfriMGSM `0.0120/0.0160` for 2/3-shot respectively.
  Zulu few-shot Intent remains quarantined because its demonstration labels
  are corrupt. Full prompt-level provenance is in
  `notes/2026-07-27-gdn-base-fewshot-closeout.md`.
- GatedDeltaNet base NER few-shot closeout, 2026-07-27: two/three-shot arrays
  completed and were recorded as held-out validation diagnostics, not test
  results. Best-prompt F1 remains near zero: Xhosa `0.0132/0.0032`, Zulu
  `0.0021/0.0021`, and Tswana `0.0027/0.0022`. Outputs are complete but
  frequently copy or repeat prompts and fail extraction. Few-shot prompting
  therefore does not rescue base GDN NER.
- GatedDeltaNet monolingual Xhosa NER recovery, 2026-07-28: reducing effective
  batch from `256` to `64` produced a validation-selected checkpoint at epoch
  14 (`eval_all_f1=0.1343`). Its single frozen held-out test reached
  **best-prompt F1 `0.2780` on prompt 1**; five-prompt mean `0.1818`, range
  `0.0976-0.2780`. This replaces the prior monolingual zero and shows genuine
  extraction signal, although label confusion and repetition remain.
- GatedDeltaNet Xhosa NER HPO screen, 2026-08-01: the validation-only
  24-trial Bayesian/Hyperband screen completed cleanly in HEX job `1150968`.
  The top two recipes are `tyvese0z` (`eval/all_f1=0.6113`) and `b888dh4l`
  (`0.6086`), far above the earlier narrow recovery validation result.
  Three-seed stability is final: `tyvese0z` mean `0.6085`, sample SD `0.0097`,
  range `0.5978-0.6165`; `b888dh4l` mean `0.5858`, sample SD `0.0336`, range
  `0.5472-0.6086`. The validation winner is therefore `tyvese0z`; its
  representative seed-42 `checkpoint-552` is frozen rather than selecting the
  lucky highest seed. This strongly supports optimization/capacity as a major
  confound in the prior mono-versus-multi NER gap, though matched exposure
  controls are still required. Official frozen held-out job `1151321`
  completed: explicitly labelled best-prompt P5 F1 is `0.6287`, prompt mean
  `0.6029`, and range `0.5192-0.6287`. The verified result is promoted in
  `GatedDeltaNet Results!E4` and `Comparison Data!I44`; full prompt-level
  provenance and the descriptive-not-unbiased warning are retained.
- xLSTM base closeout update, 2026-07-27: POS job `1118354` is running.
  NER `1118353` and Intent `1118355` failed before scoring because the
  adapter-free xLSTM path left the model in chunkwise training mode and hit
  non-64-divisible sequence lengths. The shared runner fix is implemented
  locally with a focused passing regression test; cluster sync and isolated
  NER/Intent retries remain pending.
- xLSTM base POS closeout, 2026-07-27: job `1118354` completed exit `0:0` and
  passed the matched held-out constrained POS audit. Best-prompt base
  Xhosa/Zulu/Tswana token accuracy is `0.1536`/`0.1456`/`0.0746`; predictions
  overwhelmingly collapse to `PUNCT` and exact-sequence accuracy is zero.
  These are valid negative base-model results, now promoted with full
  four-prompt provenance.
- xLSTM corrected Xhosa News recovery, 2026-07-28: validation-only selection
  froze epoch-1 checkpoint `17` at validation F1 `0.4096`; its one held-out
  test reached **best-prompt F1 `0.2759` on P1**, mean `0.2338`, range
  `0.1916-0.2759`. This replaces the historical monolingual `0.9197` row,
  which is quarantined because it predates the role-token selective-training
  and left-truncation repairs. The corrected value and full provenance are
  promoted in `XLSTM Results!C3,E3,I3:J3`.
- xLSTM corrected multilingual News recovery, 2026-07-28: validation-only
  selection froze epoch-4 checkpoint `272` at validation F1 `0.6942`. Its
  single held-out test reached English **best-prompt F1 `0.2786` on P1**
  (mean `0.2586`, range `0.2458-0.2786`) and Xhosa **best-prompt F1 `0.5114`
  on P4** (mean `0.4910`, range `0.4674-0.5114`). These replace the historical
  multilingual `0.8810/0.9238` values, which predate the role-token and
  truncation repairs. Corrected values and provenance are promoted in
  `XLSTM Results!F2:F3,I2:J3`.
- xLSTM corrected monolingual English News recovery, 2026-07-28:
  validation-only selection froze epoch-2 checkpoint `104` at validation F1
  `0.5921`. Its single held-out test reached **best-prompt F1 `0.1700` on
  P2**, mean `0.1608`, range `0.1480-0.1700`. This replaces the historical
  monolingual `0.9126` value, which predates the role-token and truncation
  repairs. The corrected value and provenance are promoted in
  `XLSTM Results!E2,I2:J2`.
- xLSTM corrected base Intent closeout, 2026-07-29: adapter-free job `1130553`
  completed the official held-out test with mean continuation token
  log-probability scoring. Best-prompt F1 is English `0.0032`, Xhosa `0.0022`,
  Zulu `0.0037`, and Sotho `0.0042`. This is a defensible near-chance negative
  base result, now promoted in `XLSTM Results!D16:D19` with full prompt
  provenance. Historical tuned Intent cells remain quarantined pending clean
  validation-selected recovery.
- xLSTM corrected monolingual Xhosa AfriHG, 2026-07-30: validation-only job
  `1139443` selected checkpoint-328; frozen held-out job `1139444` produced
  chrF `17.9540` from one task-native prompt on 1,305 official test rows.
  This replaces the quarantined target-truncation-exposed historical value and
  is promoted with artifact hashes and output-quality counts.
- xLSTM corrected monolingual Zulu AfriHG, 2026-07-30: validation-only job
  `1139445` selected checkpoint-892 at validation chrF `19.5712`; frozen
  held-out job `1139446` produced chrF `19.6499` from one task-native prompt
  on 1,776 official test rows. The artifact has 36 empty predictions, 1,739
  unique predictions, and zero null predictions. This replaces the
  quarantined target-truncation-exposed historical value and is promoted with
  artifact and adapter hashes.

- Storage/phase decision, 2026-07-30: scratch capacity is expected to expand
  to approximately `300 GB`, and the user authorizes retaining the best
  validation-selected fine-tuned adapters/checkpoints for later analysis. The
  current deadline phase is therefore **benchmark-only**: finish already
  running training/evaluation dependencies, preserve their best artifacts,
  and do not launch new fine-tuning or broad HPO now. A later phase may rerun
  the full fine-tuning matrix from the retained models/configurations; this
  does not change the current held-out-test, provenance, or promotion rules.
- GatedDeltaNet corrected monolingual MasakhaNews closeout, 2026-07-30:
  validation-selected training array `1142362` and frozen held-out test array
  `1142368` completed under the matched seven-label, five-prompt News contract.
  Promoted best-prompt weighted F1 is English `0.2094` (P1; mean `0.1853`,
  range `0.1613–0.2094`) and Xhosa `0.5621` (P5; mean `0.4717`, range
  `0.3248–0.5621`). These replace the quarantined legacy mono-News zeros;
  full prompt-level provenance, artifacts, hashes, and the shared 2048-token
  truncation caveat are in `notes/2026-07-30.md` and the source-cell notes.
- xLSTM General matched-suite closeout, 2026-07-31: jobs `1137033–1137036`,
  isolated replacements `1145318/1145456`, and matched constrained POS
  `1145480` now provide the complete 41-row General matrix under the corrected
  protocol. Canonical `XLSTM Results` General cells are promoted with
  task-native held-out metrics and full source-cell provenance. Matched POS
  best-prompt token accuracy is Xhosa `0.7481`, Zulu `0.7685`, and Tswana
  `0.7699`; corrected mean-token Intent remains near chance
  (`0.0045–0.0086` F1); T2X Xhosa chrF is `18.9044`; AfriHG Xhosa/Zulu is
  `8.8127/7.6026`. Best-of-test-prompts is explicitly descriptive and not an
  unbiased estimate. The legacy free-generation POS artifact remains
  diagnostic-only and the stale validation-split POS artifact remains
  quarantined.
- Four-architecture Base matrix closure, 2026-07-31: the final genuine
  coverage gap, LLaMA/Transformer Base MasakhaNER Xhosa, is complete from job
  `1145524_1` and promoted in `Transformer Results!D4`. All five official-test
  prompts are `0.0000` F1 on 1,000 Xhosa documents, so the explicitly labelled
  best-prompt headline, mean, and range are all `0.0000`. Raw output collapses
  into repetitive malformed role/special-token strings and produces no
  extractable entities, making this a verified negative rather than a missing
  run. `Comparison Data!F4/O4` now resolve to `0.0000/4/4`; no `3/4` coverage
  row remains. Structural N/A rows remain non-experiments.

## Durable update — architecture label correction and pure-GDN priority (2026-08-02)

- The verified existing checkpoint is `Qwen3NextForCausalLM`, trained from
  scratch with 33 layers total: 25 `linear_attention` GDN layers and 8
  `full_attention` layers, hidden size 640, and 10 heads. The exact label is
  **GDN–Attention Hybrid (Qwen3Next implementation)**. Its results remain a
  valid secondary hybrid arm only and cannot fill the pure-GDN comparison slot;
  this hybrid implementation was deliberate in the early notes, while the
  pure-GDN wording was later reporting drift, not a new implementation.
- Pure GDN is installed on HEX through `flash-linear-attention==0.5.1` and
  `fla-core==0.5.1`, with `GatedDeltaNetConfig`, `GatedDeltaNetModel`, and
  `GatedDeltaNetForCausalLM` under
  `/home/lmbanr001/masters/sallm/.venv/lib/python3.12/site-packages/fla/models/gated_deltanet/`.
  With `attn=None`, every block is GatedDeltaNet; no official pretrained
  weights exist, so matched pretraining from scratch is required. The current
  `gated_deltanet` -> Qwen3Next mapping in `src/main/sallm/models/registry.py`
  is the integration defect to correct.
- The next architecture priority is pure GDN integration, not broad new hybrid
  HPO: preserve hybrid General equal-mixture control `1160839`, verify BF16
  forward/backward, FLA fast path, packed context, DDP, save/load, and
  generation, then run one A100-80GB canary before resumable full pretraining.
  Retain A100-80GB exclusivity and never tune on test.
- The parameter-matched pure shape is now frozen from CPU-only HEX job
  `1164029`: 21 all-GDN layers (`attn=None`), hidden size 512, intermediate
  size 1536, four 128-wide heads, `expand_v=2`, tied 65,536-token embeddings,
  and 2,048-token context, for exactly `127,425,448` parameters. This preserves
  the tied-embedding convention of the other 125M baselines. It is a config
  selection only until the required BF16/DDP/FLA-kernel/save-load/generation
  A100-80GB canary passes.
- Hybrid General equal-mixture control `1160839` completed cleanly `0:0` after
  2,728 updates with final validation loss `0.2754413566702204`. Preserve it as
  a **GDN--Attention Hybrid (Qwen3Next implementation)** control; do not use it
  as evidence for the pure-GDN architecture.
- Pure-GDN integration is locally accepted on
  `research/pure-gdn-baseline-20260802` at signed commit `4feaadf`: 80 tests
  and a third fresh Sol/high `ship` review cover pure/hybrid routing, wrapped
  2,048-token streaming canary batches, the FLA kernel probe, and DDP-safe
  saves. No HEX sync or GPU canary occurred because external-write approval is
  still required.
- The broad-pretraining budget is not yet scientifically frozen. The common
  tokenizer/optimizer/context intentions are aligned, but executed histories
  differ or are incomplete (LLaMA `48,403` steps; xLSTM `67,498` steps at
  epoch `3.0871`; current Mamba Hub lineage not fully recovered). Treat the
  pure config's five epochs as provisional; recover executed token/update
  contracts and pre-register one explicit token budget before full launch.

## Update Rule

- Update this progress note only when a result changes the high-level story.
- Put run-by-run details, logs, failures, and hypotheses into
  `notes/YYYY-MM-DD.md`.
- Current working comparison view: `final_results.html`.
- Current unresolved Mamba gap plan:
  `notes/2026-05-18-mamba-gap-investigation.md`.
- Advisor-ready Mamba classification/generation/base-model meeting note:
  `notes/2026-05-21-advisor-meeting-mamba-classification-generation-base.md`.
- Supervisor follow-up to-do list for Mamba output/root-cause forensics:
  `notes/2026-05-21-supervisor-todos-mamba-output-forensics.md`.
- Current post-advisor priorities from 2026-06-11: verify suspiciously strong
  xLSTM POS and InjongoIntent evals end to end; run focused xLSTM HPO for the
  weakest defensible tasks; train/evaluate GatedDeltaNet under the same
  downstream/test-set protocol for the next architecture comparison.
- 2026-06-16 sequencing update: do not start xLSTM HPO until a light
  evaluation/provenance cleanup is done. The corrected constrained POS result
  explains Mamba's updated POS failure as label-prior collapse under
  closed-label scoring (`DET`/`NOUN`/`PART`), not a free-generation formatting
  issue. HPO is now underway; GatedDeltaNet follows the xLSTM HPO/test gate.
- 2026-06-16 NER reformulation closeout: constrained BIO tag-sequence
  MasakhaNER test diagnostics did not rescue NER for any architecture and do
  not replace the official span-generation NER rows. Macro entity F1 remains
  near zero: Mamba `0.0119`, LLaMA `0.0353`, xLSTM `0.0367`. The models fail
  differently: Mamba overpredicts date labels, LLaMA overpredicts `b-per`, and
  xLSTM overpredicts repeated inside labels such as `i-loc`/`i-per`/`i-org`.
  This changes the root-cause story: poor NER is not only a span-copy or
  free-generation parsing problem; even legal closed-label BIO scoring exposes
  severe label-prior/calibration collapse. Diagnostic rows were written to the
  Google Sheet `Eval Provenance` tab only, marked `diagnostic_only`; official
  benchmark rows were left untouched.
- 2026-07-29 advisor comparison correction: General is a first-class variant,
  never an auxiliary ranking. The canonical `Variant Comparison` sheet now
  uses task-native held-out metrics and normalizes each cell against the best
  architecture × variant on the same task-language. Headline variant averages
  use only fully verified 12/12 Mono/Multi/General rows; exact scores and
  incomplete coverage remain visible without promotion.
## Pure-GDN runtime gate status — 2026-08-02 evening

- Pure arm remains FLA `GatedDeltaNetForCausalLM`, `attn=None`, exact `127,425,448` parameters; prior Qwen3Next results remain explicitly **GDN--Attention Hybrid**, not pure GDN.
- A100-80GB jobs `1164608`, `1164702`, and `1164809` collectively verify BF16 forward/backward, direct FLA chunk-kernel backward, two-rank startup, Hub streaming with `datasets 4.x`, exact model-size validation, a real wrapped `2048`-token batch, TileLang backend selection, and one optimizer step. They do not yet verify a complete canary/save/reload/generation chain.
- Runtime defects found and fixed on `research/pure-gdn-baseline-20260802`: Slurm submit-dir resolution (`8521c6ec...`), grouped Hydra overrides (`3b50f218...`), Hub `List` metadata compatibility (`581e95cf...`). A reviewed but currently uncommitted trainer fix marks only pure GDN `accepts_loss_kwargs=False` and defaults resolved iterable `dispatch_batches=False` when unset; affected suite `16` tests plus fresh Sol verdict `ship`.
- Decisive two-step canary rerun is pending because the external-write approval service rejected both staging and HEX sync with `unknown_parameter: input[6].namespace`. Exact-file sync must be explicitly re-authorized; do not bypass the approval boundary.
- Full pretraining remains gated after canary by storage (`~89.9/100 GB` used) and a preregistered matched token/update budget. Selection is validation loss/validation metrics only; never tune on held-out test.
- Fresh-lane update: reviewed trainer/provenance changes are committed on `research/pure-gdn-baseline-20260802` as `6812806bb92928cb78a912eb8ce2c67f86317e77`. Exact HEX rsync was separately rejected as remote source export, so no decisive canary exists yet. Latest quota is `89/100 GB` (`90.0%`), owned queue is empty, and three A100-80GB GPUs are nominally free. The prior `input[n].namespace` approval failure is publicly tracked in open `openai/codex#31754`; do not conflate it with the later rsync policy rejection.

## Pure-GDN hardware gate accepted — 2026-08-02

- Decisive two-GPU A100-80GB canary `1165989` completed `0:0` in `00:08:01`
  using pure FLA `GatedDeltaNetForCausalLM` with `attn=None`, exactly
  `127,425,448` parameters, and no concurrent owned A100-40GB or L40S work.
  Both wrapped 2,048-token optimizer steps completed with ordinary-scale
  losses `11.1940/11.1951`; both iterable validation passes completed at
  `eval_loss=11.1919603348`; the TileLang FLA backward path was exercised.
- Both step checkpoints and `final_model` saved without rank collisions.
  AutoModel reload, exact state-dict roundtrip, and deterministic greedy
  generation passed. Accepted artifact root:
  `/scratch/lmbanr001/masters/sallm/checkpoints/sallm-pure-gdn-125m-canary/a10080-1165989`
  (`1.7G`; final config SHA256
  `114fd68f5f522b3627aa10fc34d029e9e200adb5ebb228e8f826c64b93eddf5b`).
- The implementation/hardware gate is complete. Full pure-GDN pretraining is
  still not authorized to start: current scratch is `91/100 GB` (`91.6%`),
  and the executed LLaMA/Mamba/xLSTM token/update histories must be recovered
  and one matched token budget preregistered. Pretraining recipe selection is
  validation-only and held-out tests remain untouched.

## Pure-GDN matched pretraining contract frozen — 2026-08-02

- Full pure-GDN pretraining is authorized and frozen to `48,403` optimizer
  steps at global sequence batch `48` and context `2,048`, exactly
  `4,758,208,512` token slots matching the executed LLaMA/xLSTM anchor. Hub
  streaming is mandatory. Tokenizer vocabulary is `65,536`; optimizer is LR
  `4e-4`, cosine, `2,000` warmup steps, weight decay `0.01`, Adam betas
  `0.9/0.95`, and max gradient norm `1.0`.
- Model identity remains pure FLA `GatedDeltaNetForCausalLM`, `attn=None`,
  exactly `127,425,448` parameters. Historical Qwen3Next results remain
  explicitly **GDN--Attention Hybrid**, never evidence for this pure arm.
- Pretraining selection and monitoring use validation loss only. Downstream
  recipe/checkpoint selection must also remain validation-only before any
  one-time official held-out test.
- Verified cold archives now preserve `gdn_afrihg_hpo_r1` and `news_hpo_r1`
  on Kombuys with exact per-file SHA256 source/destination manifests (`119`
  and `126` files). CPU job `1166480` removed only those verified HEX roots
  plus the reproducible `655M` Triton cache; accepted canary `1165989` and all
  canonical winners remain untouched.
