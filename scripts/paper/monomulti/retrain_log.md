# Phase 3 retrain log (in progress)

Roots: HEX `/scratch/lmbanr001/masters/sallm/results/monomulti_retrain_20260924/`, Kombuys `/scratch/alombard/sallm/results/monomulti_retrain_20260924/`. Scripts mirrored in `monomulti/retrain/`. Status per cell: `retrain_scope.csv`. Results: `retrain_results.csv`.

## Mamba Mono SIB-200 eng: 14.44 (why it is low)

- Recipe `finetune/mamba_sib_eng` has effective batch 256 (Kombuys run: 2 x GA 128) and 10 epochs. With 701 training rows that is 3 optimizer steps per epoch, 30 steps in total. The other Mamba Mono SIB recipes had 6-22 steps per epoch (afr 90 steps, nso 220, sot 110, xho 165, zul 330).
- Validation loss was still falling at the end (8.92 -> 7.14; afr reached 5.14). The model is undertrained. This is the same failure as the frozen Mamba Mono NER recipe.
- Validation F1 per epoch (checkpoint step: F1): 3: 0.015, 6: 0.059, 9: 0.116, 12: 0.109, 15: 0.122, 18: 0.126 (selected), 21: 0.120, 24: 0.118, 27: 0.115, 30: 0.120.
- Test predictions collapse onto two labels: health 154/204 and travel 47/204 (entertainment 2, sports 1). Gold is spread over 7 classes. This is label collapse from undertraining, not a scorer fault.
- Diagnostic that confirms the cause: the same recipe (lr 8e-5, 10 epochs, cosine, r16) at effective batch 64 (2 x 32; 11 steps/epoch, 110 steps). Kombuys 5090 for training, Kombuys 3080 Ti for scoring (the scorer and GPU used for the other Mono SIB cells). Label `mamba2_sib_mono_eng_eb64`.
  - Validation F1 by step: 11: 0.118, 22: 0.144, 33: 0.279, 44: 0.168, 55: 0.238, 66: 0.418, 77: 0.514, 88: 0.598, 99: 0.598, 110: 0.612 (selected).
  - Test F1 67.57, against 14.44 for the frozen recipe. Predictions cover all 7 labels (science/technology 76, travel 45, politics 31, health 17, sports 16, entertainment 11, geography 8), which is close to the gold spread.
  - Conclusion: the 14.44 comes from the recipe's optimizer-step budget, not from the data or the scorer.
- Both variants are in `retrain_results.csv` (column `variant`). Only the frozen-recipe row follows the approved scope. Using the eff-batch-64 row needs the author's approval, the same deviation that was approved for Mamba Mono NER.

## Incidents (25 Sep)

- 07:55 Kombuys GPU1 (3080 Ti) fell off the bus; CUDA failed on both GPUs. Runs killed: Mamba SIB Multi (80%), Mamba NER Mono xho/zul, Mamba POS Multi. Archived under `aborted/`, restarted on HEX L40S. The six Mamba Mono SIB runs and their selection + test had already finished on Kombuys.
- A trio job (POS Multi + NER xho + NER zul on one L40S) ran about 3.5 min/step and was cancelled. NER xho and zul were rerun as a pair (job 1370346).
- 10:08 Mamba SIB eng eff-batch-64 diagnostic, run as an overlap step inside job 1370235: CUDA OOM at the first eval (the GPU was shared with other steps). Requeued as its own job (1370772).
- 13:22-14:20 The tsn NER validation selection ran as an overlap step inside the Mamba POS Multi job (1370349). Two concurrent Mamba evals pushed that GPU over 44 GB, and the POS training OOMed at step 36. My error: no more overlap steps next to training. POS Multi restarted (1371341); tsn selection resubmitted as its own job (1371340). The 9 validation outputs already written are reused.
- 14:35-16:35 SSH to HEX was down: the `hex` alias now jumps through Kombuys, which was offline. Switched to `hex-direct`. Slurm jobs were not affected.

## Rule: fewer than 10 optimizer steps per epoch -> effective batch 64 (author, 25 Sep)

Steps per epoch for all 144 Mono/Multi cells are in `steps_per_epoch.csv`. Where a trainer_state or checkpoint exists, the value comes from max_steps / num_train_epochs or from a checkpoint's step/epoch. Otherwise it is ceil(train rows / effective batch), using the recipe's batch and the logged `Samples: train=` counts. Train rows: News eng 3309, xho 1032; SIB 701 per language; Intent eng 1779, others 2240; NER 1441 per language; POS tsn 754, xho 752, zul 753. MzansiLM historical hex_v3 runs used 2 GPUs, so their effective batch is 32.

Cells under the rule:
- Mamba Mono SIB-200 afr: effective batch 128, 6 steps/epoch. Rerun at effective batch 64 (`mamba2_sib_mono_afr_eb64`, Kombuys 5090). The frozen-recipe result (test 50.33, selected epoch 12) is superseded.
- Mamba Mono SIB-200 eng: effective batch 256, 3 steps/epoch. The eff-64 run is now the selected variant (test 67.57). The frozen-recipe row (14.44) is dropped from `retrain_results.csv`.
- Mamba Mono NER tsn/xho/zul: frozen recipe had effective batch 256/256/128 (6/6/12 steps/epoch). Already retrained at effective batch 64 (23 steps/epoch).

Mamba Mono POS zul is not under the rule, contrary to the audit. Its v15 trainer_state shows max_steps 180 over 15 epochs (753 rows, effective batch 64), which is 12 steps/epoch. Not retrained.

No other cell in any architecture is under 10 steps/epoch. The lowest others are xLSTM Mono News xho (17) and Mamba Mono SIB nso/sot/xho (11).
- 16:50-19:39 Mamba POS Multi on the Kombuys 5090 OOMed at step 180/540. The GPU was shared with another user's ASR job (9.5 GB) and my Mamba SIB afr eb64 run. A second attempt could not fit either: another agent's MzansiLM Intent training (8 GB) had also started on the 5090. POS Multi moved to HEX L40S (job 1371685; selection pipeline 1371686). Mamba SIB afr eb64 reruns alone on the 5090 (about 3 GB).

## 20:35: supervision handed over after a session restart
- `mmr-pl-pos` resubmitted as 1371715 without `TOP=4`. The old job would have validated only the 4 checkpoints ranked best by the in-training metric. That pre-filter is selection on a non-protocol metric, so all 15 Mamba POS Multi epochs are now scored on validation (PAR 3).
- `seqsel.py prune` no longer keeps the checkpoint ranked best by the in-training metric; it keeps only the validation-selected one (prune rule). Backup: `jobs/seqsel.py.bak_2040`.
- Step-rule scan finished for all 144 cells in `steps_per_epoch.csv` (see reselect_log.md, 20:35). For this effort:
  - Mamba Mono NER tsn/xho (frozen eff 256, 6 steps/epoch) are under the rule.
  - Mamba Mono NER zul (eff 128, 12 steps/epoch) is not, but it trains at 64 under the earlier approval.
  - Mamba Mono POS zul (12 steps/epoch, already at eff 64) is not under the rule.
- `retrain_scope.csv` now has a `variant` column.
- Two earlier cells do not fully follow "every epoch scored on validation". Recorded here, not rerun:
  - xLSTM NER Multi: 12 of 40 epoch checkpoints scored.
  - MzansiLM POS Mono tsn: no selection; final adapter of the v9 xho/zul recipe, `save_strategy no`.

## 21:15-21:40: effort stopped (author decision: superseded by full fine-tuning with one shared recipe)
- Harvested: **Mamba Mono SIB-200 afr at effective batch 64** (`mamba2_sib_mono_afr_eb64`). Training ended 21:10; validation over all 15 epochs selected checkpoint-121 (epoch 11, val 76.96); test **81.94** (frozen recipe was 50.33). Scored on the Kombuys 3080 Ti, the only job on that card.
- Cancelled on HEX: mmr-mb2 1370346 (Mamba NER Mono xho/zul, 80-90% trained), mmr-mbsib 1370318 (Mamba SIB Multi, 52%), mmr-pl-tsn 1371340 (Mamba NER Mono tsn, training done but 6 of 15 validation epochs plus test left, with no free GPU: over 30 min), mmr-mbpos 1371685, mmr-pl-pos 1371715, mmr-pl-sib 1370457, mmr-pl-xz 1370456. Kombuys tmux `mmr-afr64` exited by itself after the afr test.
- Final state: **16 cells done, 12 stopped** (Mamba SIB-200 Multi ×6, Mamba NER Mono tsn/xho/zul, Mamba POS Multi ×3).
- Cleanup: every checkpoint, final_adapter and partial validation output of the stopped runs deleted, as were the aborted tsn run's checkpoints (freed space counted in reselect_log.md; logs kept).
