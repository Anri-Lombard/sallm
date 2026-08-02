# GDN HPO and multilingual-advantage audit — 2026-07-31

## Current workstream

- Working target for Thursday, 6 August 2026: comprehensive, time-efficient,
  validation-only GDN HPO across applicable downstream task families and
  Mono/Multi/General regimes, followed by stable validation selection, one
  frozen official-test evaluation per selected recipe, retained best
  checkpoints, complete prompt provenance, and verified canonical-sheet
  promotion.
- This is the current research workstream, not a formal Codex goal or an
  assertion that an exhaustive global optimum can be proven.

## Conclusion

- GDN has **not** received full, comparable HPO across the downstream suite.
  Only News and AfriHG received bounded searches; NER had one effective-batch
  recovery canary; Intent had one focused rank/LR recovery; POS, SIB, General,
  and most monolingual adapters were not systematically searched.
- The current implementation is not a time-optimal HPO system. GDN searches
  are manual Slurm grids with per-trial early stopping, not an asynchronous
  multi-fidelity scheduler. There is no GDN W&B sweep config and no active
  cross-trial pruning callback in the current `training/factory.py`.
- The canonical 20-row matched comparison has GDN winners: Multi 13, Mono 4,
  General 3. All three General wins are MasakhaPOS. The anomaly is therefore
  mainly multilingual, not a broad General-adapter advantage.
- A sheet/evaluator mapping mistake is unlikely: the promoted artifacts use
  the corrected official held-out contracts and were audited for rows, labels,
  prompts, adapter identity, and scoring. The causes differ by task.

## What existing evidence establishes

### AfriHG

The apparent multilingual advantage is confounded by unequal optimization.
The selected multilingual result uses rank/alpha 32/64, LR 2e-4, seed 43,
checkpoint 1158 after a validation-only rank/LR/seed search and an evaluator
context-reservation fix. Held-out chrF is Xho/Zul 23.8442/24.4012. The mono
adapters remain rank 16 / LR 8e-5 and score 14.8387/14.2960. This is evidence
for HPO/capacity sensitivity, not clean architecture-level multilingual gain.

### MasakhaNER

The original mono recipe was under-trained. Multilingual used effective batch
64 and about 68 optimizer steps/epoch; mono Xhosa used effective batch 256 and
about six steps/epoch, early-stopping after roughly 24 optimizer steps. Reducing
mono Xhosa effective batch to 64 raised held-out best-prompt F1 from 0 to
0.2780, while multilingual remains 0.3594. This is direct causal evidence that
HPO/training-exposure deficiency explains a large part, but not necessarily all,
of the gap. Zulu and Tswana still need the same matched recovery control.

### MasakhaPOS

Multilingual and General are stable around 0.80-0.82 token accuracy. Mono
Tswana/Zulu are 0.8112/0.7303, while mono Xhosa genuinely collapses to NOUN at
0.2361 despite correct loading and scoring. General beats Multi by only
0.0188/0.0176/0.0148 for Xho/Zul/Tsn. The evidence supports beneficial pooled
task exposure plus a mono-Xhosa optimization collapse; it does not isolate an
architecture-specific General advantage.

### SIB200

Multilingual has higher prompt-mean F1 for all six languages, although the
best-prompt Zulu headline slightly favors mono. The same official test and
seven-label scorer were used. This is consistent with multilingual transfer
and greater pooled exposure, but no step/token-matched mono control or HPO was
run, so architecture and optimization are not separated.

### INJOngo Intent

All GDN regimes remain near chance and collapse to a dominant class. Small
Multi-over-Mono differences are not meaningful multilingual superiority. A
rank-32 validation-selected recovery did not solve it. The remaining suspects
are adapter capacity/target modules, optimization, and formulation interaction;
ordinary row/scoring bugs were audited out.

### News and General

Corrected monolingual News is stronger than Multi and General for both Eng and
Xho. General is also worse than Mono/Multi on NER, SIB, Intent, and AfriHG.
General's only matched wins are POS; T2X General beats Mono but has no Multi
task adapter and is outside the 12/12 comparison.

## Time-efficient causal/HPO design for the later storage phase

1. Freeze evaluator, split, prompt set, metric, role-token handling, and
   adapter-loading checks before search. One canary per task family.
2. Search architecture-level knobs first on validation: log-uniform LR,
   effective batch/optimizer-step budget, LoRA rank/alpha, target-module set,
   warmup/schedule, and dropout. Keep data regime fixed.
3. Use asynchronous successive halving: broad short-budget screen, promote the
   best fraction to medium budget, then fully train only the top two. Current
   manual grids and per-trial early stopping do not provide this.
4. Verify the GDN FLA fast path and use the largest safe microbatch while
   preserving the candidate's effective batch. Prior measurements showed about
   a 15x speed difference versus the Torch fallback.
5. For Mono versus Multi, match optimizer updates and target-language exposure:
   mono repeated to the same update budget; proportional and language-balanced
   Multi; then a leave-target-language-out transfer control if useful.
6. Re-run the top two validation configurations across three seeds and select
   by mean validation metric with stability considered, not a lucky single
   seed. Freeze one recipe/checkpoint before one held-out test evaluation.
7. Test prompts remain fixed. Canonical test reporting uses explicitly labelled
   best prompt plus mean/range/all prompt values; it is descriptive, not an
   unbiased estimate and must not drive HPO.

## Minimal causal queue after scratch expands

1. NER: apply the effective-batch-64 recovery to Zul/Tsn; compare equal update
   budgets against Multi.
2. POS: rank/LR/batch control for mono Xho plus equal-exposure Mono/Multi.
3. SIB: equal-update and language-balanced Mono/Multi controls before broad HPO.
4. Intent: bounded capacity/target-module/LR search using corrected mean-token
   validation scoring; stop if class collapse persists.
5. AfriHG: run a rank-32/LR-2e-4 mono control before claiming multilingual
   transfer. No broader AfriHG search is needed yet.

No GPU work was launched. HEX scratch remains 100 GB / 91% used; the requested
300 GB quota is not active.

## Checkpoint-retention decision and verified archive

- Retain only the validation-selected best fine-tuned checkpoint for each run,
  together with its trainer state/config/tokenizer and final adapter when
  present. Raw evaluation artifacts and provenance remain separate. Intermediate
  and losing checkpoints are not long-term retention targets.
- This retention is specifically to permit later model analysis and exact reuse
  without retraining. Selection remains validation-only; held-out test metrics
  never determine which checkpoint is retained.
- Archived selected GDN monolingual News models from HEX to Kombuys at
  `/scratch/alombard/masters/sallm/archive/hex_20260731/checkpoints/gdn_news_mono_repair_20260730`:
  English `checkpoint-312` plus final adapter and Xhosa `checkpoint-33` plus
  final adapter.
- Source and destination both measured `692,840,264` bytes and `38` files.
  SHA-256 matched for both checkpoint weights, both `trainer_state.json` files,
  and both final-adapter weights. Only after this verification, the exact HEX
  source `/scratch/lmbanr001/masters/sallm/checkpoints/gdn_news_mono_repair_20260730`
  was removed. The Kombuys copy is recoverable and remains intact.
- HEX quota reporting is refreshed only every few minutes, so the immediate
  post-removal `purequota` output still reported 100 GB quota / 91 GB used.

## 22:22 SAST HPO implementation and canary

- Cancelled only the four stale `DependencyNeverSatisfied` jobs authorized by
  the user: `1137026`, `1137028`, `1137030`, and `1137032`. Slurm re-read shows
  all four `CANCELLED` with exit code `0:0`. Held jobs `1117467/1117468` were
  not touched. Refreshed HEX quota is 100 GB total / 90.4 GB used.
- Fixed shared HPO trial isolation for Hydra `DictConfig`, added derived sweep
  controls for LoRA rank/alpha and effective batch size, and forced Hub pushes
  off during sweeps. Local checks: `tests/hpo/test_trial.py` is `2 passed`;
  shell syntax and sampled GDN sweep composition also pass.
- Added the first validation-only GDN sweep at
  `src/conf/sweeps/gdn_ner_xho.yaml`: Bayesian search capped at 24 runs with
  Hyperband pruning, LR `3e-5–4e-4`, ranks `16/32/64`, effective batches
  `32/64`, dropout `0/0.05/0.1`, warmup `0.03/0.05/0.1`, cosine versus
  constant-with-warmup, 15-epoch ceiling, epoch validation, patience 3, and
  selection metric `eval/all_f1`. Test data is not referenced.
- `scripts/launch_hpo.sh` now recognizes GDN sweep names and aborts before W&B
  sweep creation unless the `causal_conv1d`/FLA GatedDeltaNet fast path imports.
- Synced only `launch_hpo.sh`, `trial.py`, and `gdn_ner_xho.yaml` to Kombuys.
  One offline validation canary is running as tmux task
  `gdn-ner-hpo-canary` on Kombuys GPU 0 (RTX 5090), with log
  `/scratch/alombard/masters/sallm/logs/gdn_hpo_20260731_ner_xho_canary.log`
  and checkpoint root
  `/scratch/alombard/masters/sallm/checkpoints/gdn_hpo_20260731/ner_xho_canary`.
  It loaded the intended GDN checkpoint, exposed 2,660,128 trainable LoRA
  parameters, trained one step, completed validation loss (`5.400584` on all
  817 Xhosa validation rows), and confirmed the FLA TileLang backward fast
  path. It completed in `159.66s`; Trainer selected `checkpoint-1` with
  validation `best_metric=0.0`, matching the debug artifact's aggregate NER F1
  after the deliberately minimal one-step train. The malformed repetitive
  outputs are expected at this budget and are not a scientific result. No test
  split, traceback, CUDA OOM, sheet cell, or HEX GPU job was involved.
- The production sweep sets `save_only_model=true` and `save_total_limit=1` so
  each trial keeps only its current validation-selected model rather than
  optimizer states or intermediate checkpoints. After the sweep, only the
  stable validation winner(s) will be retained for later analysis.
- First HEX submission was rejected before job creation because 16 requested
  CPUs exceeded `AssocMaxCpuPerJobLimit`; reduced to the documented 8 CPUs.
  Job `1150963` then failed before Python because Slurm's spool copy could not
  locate `scripts/lib/env.sh`. Job `1150964` reached four A100s but failed
  before Python because the launcher unconditionally activated a missing
  `sallm-uv` Conda environment. The shared launcher now uses the same
  Slurm-script path fallback and optional-Conda/repository-venv path as the
  working fine-tune launcher.
- Resubmitted only the missing sweep as HEX A100 job `1150968`: four
  `gpu:ampere` A100-40GB GPUs, 8 CPUs, 48-hour ceiling, 24 validation-only
  trials. It entered `RUNNING` on `srvrocgpu010`; no L40S job is active.
- Job `1150968` passed the compute-node fast-path guard and created W&B sweep
  `yrhn2494` (`anri-lombard/sallm-ft`). Four agents started with six-run caps
  each. Initial run IDs are `5rneugxh`, `mb1gmc5p`, `ikbh3jki`, and
  `5e2399ui`; all four loaded the correct GDN checkpoint and created isolated
  scratch roots under
  `/scratch/lmbanr001/masters/sallm/checkpoints/gdn_hpo_20260731/ner_xho/`.
  Current progress is 0/24 complete and 4/24 initializing/training. No
  traceback, CUDA OOM, or disk error is present.

## 23:05 SAST HPO monitor

- HEX A100 job `1150968` remains healthy on `srvrocgpu010` after `00:36:16`.
  It is the only active partition family. Slurm reports `RUNNING`, exit status
  `0:0`, and batch peak RSS about `11.2 GiB`; no traceback, CUDA OOM, or disk
  error is present.
- W&B sweep `yrhn2494` is `2/24` completed/pruned and `4/24` active. Hyperband
  cleaned up `5rneugxh` and `mb1gmc5p` after epoch 3, then started
  `by1rha7z` and `oxlksi6t`. The six artifact roots are isolated under
  `/scratch/lmbanr001/masters/sallm/checkpoints/gdn_hpo_20260731/ner_xho/`.
- Current validation best metrics from checkpoint trainer states are:
  `ikbh3jki` F1 `0.1951779564` at epoch 5/checkpoint 230;
  `5e2399ui` `0.0924735642` at epoch 4/checkpoint 92;
  stopped `5rneugxh` `0.0325467860` at epoch 3/checkpoint 69;
  stopped `mb1gmc5p` `0.0144167759` at epoch 3/checkpoint 69;
  new arms `by1rha7z` `0.0171358629` and `oxlksi6t` `0.0147601476` after
  epoch 1/checkpoint 23. These are validation-only provisional values, not
  test results or promotion candidates.
- Sweep checkpoints currently occupy `643 MiB`. Refreshed quota is 100 GB
  total / 91 GB used (`91.0%`); the 300 GB increase is still not active. No
  checkpoint was moved or removed this cycle because active/pruned sweep state
  remains useful until final ranking and provenance capture.
- Metric-selection rule remains task-native validation NER F1 (`eval/all_f1`),
  with the top two later re-run across three seeds and selected by mean and
  stability before freezing one checkpoint for a single official held-out
  test. The canonical sheet remains unchanged.
