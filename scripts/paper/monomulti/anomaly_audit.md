# Mono/Multi rescore: anomaly audit (25 Sep)

Check artefacts (new dir, HEX): `/scratch/lmbanr001/masters/sallm/results/anomaly_audit_20260925/{scripts,templates,out,slurm,newskit}`. All GPU checks: HEX L40S, one job at a time, 1-12 min each (jobs 1370221, 1370223, 1370224, 1370226, 1370231). Kombuys was not used: both GPUs were busy with other agents' Mamba SIB/POS training.

Legend: (a) genuine, (b) checkpoint-selection artefact, (c) prompt/template mismatch, (d) loading/merge/implementation, (e) other (recipe or provenance).

## Summary

| Cell | Cause | Evidence in one line |
|---|---|---|
| MzansiLM Multi NER 12.4/9.3/7.1 | **(a) + (e)**: the adapter is genuinely weak, because the Multi recipe differs from the Mono one | The in-training validation span F1 plateaued at 0.25; the training prompts match the eval prompts; loops appear under both prompt renderings |
| MzansiLM Mono NER (merged) | **(c)**, mainly tsn; **not (d)** | tsn +10 to 13 pts with the v5 training template; xho +1; zul +7 (noise is about ±3) |
| GDN Mono SIB (all 6) | **(b)**, made worse because early stopping cut training short | A flat metric (0.0576) kept epoch 1; eval loss was still falling (2.92→2.15→2.02) when early stopping ended the run at epoch 3; only one checkpoint exists |
| xLSTM Mono Intent zul / xho | zul **(b)** + **(e)**; xho **(e)** | zul kept epoch 1 (eval loss 3.63). The r=4 Mono recipe is weak even at its best epoch: sot at epoch 8 scores 3.6 F1 |
| Mamba Mono News eng 29.5 | **(e)**: the hub adapter is itself degenerate, and its selection and provenance are unknown | It collapses to "health" under all 5 prompts, with or without BOS, and at either answer position. The template, BOS and position all match training |

The results point to one systemic issue: the **in-training classification metric (News/SIB/Intent) carries no information in any pipeline**. Keep-best on that metric, with `save_total_limit=1`, left one checkpoint per run, so no run can be re-selected from what is on disk.

## 1. MzansiLM NER

### Multi (`v9 .../mzansilm_ner_multi/final_adapter`, `llama_ner_all`)
- Recipe: r16 q/v LoRA with trainable chat tokens, lr 3e-5, 10 epochs, `save_strategy: no`, so the final epoch is used and selection plays no part.
- Training templates: the sallm `lm_eval_p1..p5` family, cycled. These are the same prompt family as the shared P2/P5, so there is no prompt mismatch.
- Training-time validation span F1 on 192 debug generations, by epoch 1→10: 0.00, 0.15, 0.15, 0.15, 0.18, 0.23, 0.23, 0.25, 0.25, 0.25. The adapter never learned the task well.
- Check: whitespace variant of the prompt, 200 items per language. Training-style multi-line sallm P2/P5 gives 7.5/15.2/9.8; the protocol single-line lm-eval yaml gives 8.0/10.3/6.7 on the same items. Loops occur in 7-16% of rows under both. Not (c).
- Loading is not the cause (not (d)). The unmerged PEFT path resizes the base to the adapter tokenizer (65,539 tokens, `<|user|>` etc.) and loads the trainable-token deltas. The prompts carry `[BOS]` exactly as in training.
- The old value of 77.9 comes from a different adapter: the hub `sallm-llama-masakhane-masakhaner2-tsn-xho-zul`, which is not on either remote.
- The real anomaly is recipe asymmetry (e). MzansiLM Mono NER uses the reconstructed historical recipe (r128 + embed/lm_head, 20-50 epochs, v5 template). Multi uses the repo default (r16 q/v, 10 epochs). That is why Multi (7-12) sits far below Mono (38-62).

### Mono (historical `hex_v3` adapters, merge_lora=true)
- Trained on the `masakhane_named_entity_recognition/v5` template, not on the `lm_eval_p*` family. The prompts:
  - lm-eval P5 (used for xho and zul) has almost the same wording as v5. It differs in line breaks and has a blank line before "Text:".
  - P2 (used for tsn) is a different, longer prompt.
- Check: same runner, same adapter, only `doc_to_text` swapped, 200 test items. The control rerun of P2 matched the official rows on 168 of 200 items (22.6 vs the official 25.7 on those items), so batch composition adds about ±3 pts of noise.

| lang | shared prompt (official, same 200 items) | v5 training template | Δ |
|---|---|---|---|
| tsn (P2) | 25.7 (22.6 in the control rerun) | **35.2** | +10 to 13 |
| xho (P5) | 67.1 | 68.3 | +1 |
| zul (P5) | 48.6 | 55.6 | +7 (noisy) |

- merge_lora is not a problem (not (d)):
  - Merging is exact for `modules_to_save` (embeddings and lm_head are replaced whole) and adds only bf16 rounding for the LoRA delta.
  - Test scores under v5 (68/56) match the in-training validation F1 at the last epochs (xho 0.47-0.52, zul 0.67, tsn 0.70).
  - The unmerged crash is lm-eval's `tie_weights` failing on a modules_to_save adapter. It does not indicate wrong weights.

## 2. GDN Mono SIB (and every GDN task adapter selected on the classification metric)
- Run: `pure_gdn_mono_familywise_v1`, lr 3e-5, r16, 10 epochs, `metric_for_best_model=eval_classification/all_macro_f1`, early-stopping patience 2, `save_total_limit=1`.
- The metric was 0.0576 at every epoch in every arm. Early stopping therefore ended every run after epoch 3, and keep-best returned epoch 1 (step 88).
- Log `logs/pure_gdn_mono_familywise_v1/sib/nso/seed_42/1288220.out`:
  - eval loss 2.92 → 2.15 → 2.02; train loss at step 80 was 3.9 and falling fast.
  - Total runtime 22 min, of a planned 880 steps.
- Checkpoints on disk: exactly one, `checkpoint-88` (= final_adapter), for all six languages. Re-selection is impossible; fixing this needs a retrain.
- Predictions: 192 of 204 nso rows are `science/technology`, the majority training class, which is a prior-only collapse.
- The metric carries no information. GDN Multi SIB also had 0.0576 at epoch 1, yet scores 57-77 on test under the protocol.
- The same pattern holds for Intent (in-training 0.0009-0.005 vs test 11-89) and News (0.27-0.47 vs 78-93).
- Same artefact in the other GDN cells:
  - Intent Mono eng/sot/xho: epoch 1, scores 11-13.
  - Intent Mono zul stopped at epoch 3 (eval loss 1.69→0.66→0.28) and scores **62.8**. This is the natural control: same recipe, just more epochs.
  - News Mono xho: epoch 1, 78.1 vs Multi 93.5.

## 3. xLSTM Mono Intent (`intent_recovery_20260729/xlstm_mono_*_r4_r2`)
- Recipe: r=4 / alpha=8 LoRA (Multi uses r256), lm_eval p1-5 cycle (matches eval), 8 epochs, keep-best on `eval_classification/all_f1` (0.0036-0.0085, no information), `save_total_limit=1`.
- One checkpoint per language. Selected epoch and eval loss per language:

| lang | selected epoch | eval loss per epoch |
|---|---|---|
| eng | 5 | 5.28 → 2.19 |
| sot | 8 | 3.66 → 2.09 |
| xho | 4 | 3.60 → 2.15 |
| zul | **1** | **3.63** |

- zul (one label only) is (b). The same recipe is weak even when selection is not early: sot at epoch 8 scores 3.6 F1 and xho at epoch 4 puts 91.5% of rows on one label. So xho is (e), recipe capacity; the eval loss plateaus around 2.1, against 1.5 for the xLSTM Multi adapter.
- Not (c): the training prompts are the same family as the eval prompts.
- Not (d): the chat template is ef7dead4 with no BOS, and the eval context has no BOS either (see §5). The same loader scores xLSTM Multi at 63-85.

## 4. Mamba Mono News eng (hub `sallm-mamba2-masakhane-masakhanews-eng@3ea3e4f`)
- The config (`mamba_news_eng`) trains on the lm_eval p1-5 cycle with keep-best on `eval_classification/all_f1`, patience 3. The hub copy has no trainer state, so the selected epoch is unknown. On disk there is no original checkpoint dir, only `*_fullft`.
- Checks on 200 test rows (official first-token F1 on the same rows: 15.6):
  - Parity: normalized template, no BOS (the official setting) reproduces 15.6 exactly.
  - Adding `[BOS]`: 15.2.
  - Prompts p1/p3/p4/p5 in place of p2: 27.0/10.7/11.5/9.7, all dominated by "health".
  - Full-label loglik with the answer at `<|assistant|>\n` + 8 spaces: 23.8. With the answer directly after `<|assistant|>`: 20.0.
- So the template, BOS, prompt and answer position are all ruled out. The adapter itself is degenerate.
- The old 77.3 cannot be reproduced under any of these interfaces. The likely source is a different hub revision or a pipeline difference.
- Classification (e) with a probable (b): an unknown hub selection on the same broken metric. The fix is a retrain under Rule 1.

## 5. Interface cross-checks (to rule out (d) globally)
- **Answer position.** `<|assistant|>\n` + 8 spaces is correct for all ef7dead4 and hub adapters tested. Scoring directly after `<|assistant|>` collapses them:

| adapter | at `<|assistant|>\n` + 8 spaces | directly after `<|assistant|>` |
|---|---|---|
| MzansiLM Mono Intent eng | 49.2 | 0.7 |
| MzansiLM Multi Intent | 32.8 | 4.5 |
| Mamba Mono Intent eng | 12.1 | 0.0 (1 label) |

- **BOS.** Adapters with template ef7dead4 (July, pre-canonical-template) are evaluated without `[BOS]`, and that matches their training: adding BOS drops MzansiLM Multi Intent from 36.6/32.8 to 15.8/10.0, and Mamba Multi from 65.6 to 58.5. Adapters with c681a1d8 (GDN) get BOS.
- Exception: the hub MzansiLM Mono Intent eng gains with BOS, 49.2 → 59.1 on 100 items. Hub-era adapters may therefore have been trained with BOS. This is minor, but it applies only to hub-provenance adapters (MzansiLM and Mamba Mono Intent ×4 each, Mamba Mono News eng).

## Other anomalies in rescore_results.csv

| Cell | Flag | Cause |
|---|---|---|
| GDN Mono SIB afr/eng | 71-73% top label | (b), as in §2 |
| GDN Multi SIB nso/sot/xho 57-64 (General 78-80) | far below General | (b): epoch 1 of 10 (526 steps), same flat metric |
| GDN Mono Intent eng | 67% top label | (b), epoch 1 |
| GDN Mono News xho 78.1 (General 96.2) | far below General | (b), epoch 1 |
| xLSTM Mono News xho 58.5 | 64% top label | (b): epoch 1 of 10 (17 steps/epoch); eval loss 3.98 |
| Mamba Mono News xho 77.8 (General 94.5) | far below General | (b): v15 retrain kept epoch 1 of 15 (65 steps) |
| Mamba NER Multi 2.5 | 13-19% loops, 47-87% blank | (d), known EOS issue; handled in `mamba_eos_fix_20260924` |
| Mamba Mono POS zul 41.1 (General 65.0) | far below General | (e): frozen recipe gives 12 optimizer steps/epoch (eff. batch 64); eval loss 3.33 vs tsn 2.60. Same class as Mamba Mono NER |
| MzansiLM Multi Intent 14-29 (other archs' Multi 51-89) | none (by the criteria) | Most likely (a). Interface verified: no BOS and position `\n` + 8 spaces are both correct; eval loss 0.33 at epoch 4 |

## Recommended fair fixes (one rule each, applied to all four architectures)

### Rule 1: re-select checkpoints with the protocol scorer on validation
**Scope:** every adapter whose keep-best used the in-training classification metric, or whose selection is unknown.
- Nothing can be re-selected from disk (one checkpoint per run). The procedure is:
  - retrain with the same frozen per-cell recipe;
  - no early stopping on the classification metric;
  - save every epoch;
  - score each epoch on the **validation** split with the protocol scorer and shared prompt (SIB: `retrain/sibkit/score_sib_split_hex.py`; News: the v15 driver on validation TSVs; Intent: `run_prefix_eval.py` restricted to the protocol prompt);
  - take the best epoch and rescore it on test.
- Adapters touched:

| arch | adapters | count |
|---|---|---|
| GDN | News Mono eng/xho + Multi; SIB Mono ×6 + Multi; Intent Mono ×4 + Multi | 15 |
| xLSTM | News Mono eng/xho + Multi (rolefix_val_r2); Intent Mono ×4 + Multi (clean_r4) | 8 |
| Mamba | News Mono xho + Multi (v15); Intent Multi (clean_r2); hub News Mono eng; hub Intent Mono ×4 | 8 |
| MzansiLM | Intent Multi (clean_r2); hub Intent Mono ×4 | 5 |

- Not touched: MzansiLM News/SIB and xLSTM SIB Mono (these used `save_strategy: no`, so the final epoch was used) and all NER/POS (their metrics carry information, e.g. GDN NER picks epoch 9-12).
- **Urgent:** the Mamba SIB retrains running now on Kombuys (v15 runner, `mamba_sib_*`) keep-best on the same broken metric, so they will reproduce the epoch-1 artefact. Patch them now to save every epoch and select on the protocol validation score.
- Cost (estimates; GDN measured at 3.3 s/step on HEX):
  - GDN: ~36 GPU-h at 10 epochs (SIB Mono 6×0.8 h, SIB Multi 4.8 h, Intent Mono 8 h, Intent Multi 8 h, News 10 h); ~20 GPU-h with eval-loss early stopping.
  - xLSTM: ~5 GPU-h. Mamba: ~7. MzansiLM: ~3.5.
  - Validation scoring: ~6-8 GPU-h. Test rescore: ~3-5 GPU-h.
  - **Total ~40-60 GPU-h; ~12-16 h wall on 4-6 GPUs** (long pole: GDN Multi Intent/SIB).
- Minimal subset if budget binds: only adapters where epoch 1 was kept, since those are certain artefacts. That is GDN SIB ×7, GDN Intent Mono ×3, GDN/xLSTM/Mamba News Mono xho, Mamba News Multi and xLSTM Intent zul: **15 adapters, ~20 GPU-h + ~3 h scoring, ~8 h wall**. The cost of the subset is that the method section has to say selection was repaired only where it had demonstrably failed.

### Rule 2: prompt mismatch for task adapters not trained on the lm_eval family
- Keep the shared prompt as the primary number (the protocol's whole point). For every task adapter whose training templates exclude the shared prompt family, also report its **own training template** as a sensitivity column.
- In practice this touches only MzansiLM historical-recipe adapters (News Mono ×2 + Multi, SIB Mono ×6 + Multi, NER Mono ×3 = 13). All other architectures train on the `lm_eval_p1..p5` cycle, so for them the training template equals the shared prompt and nothing changes.
- Measured effect: NER tsn +10-13, zul ~+7, xho +1. MzansiLM News/SIB Mono/Multi already score at or above General, so nothing there is anomalous.
- Cost: <1 GPU-h, ~1 h wall (patched runner and templates already in `anomaly_audit_20260925/{scripts,templates}`).
- A fully symmetric alternative is to retrain these 13 on the lm_eval family: ~40+ GPU-h (historical runs took 1-11 h each). Not recommended.

### Rule 3: one provenance class per architecture and task across regimes
- MzansiLM NER Mono (historical recipe: r128 + embed, 20-50 epochs) and Multi (repo default: r16, 10 epochs) come from different recipes. That is what makes Multi 7-12 against Mono 38-62.
- Fix: score the hub original `sallm-llama-masakhane-masakhaner2-tsn-xho-zul` (the adapter behind the old 77.9). This needs `hf auth login` on HEX by the user; ~15 min GPU.
- Otherwise retrain Multi with the historical recipe (~6-9 GPU-h on L40S).
- The same rule covers Mamba Mono NER/POS (frozen recipes give 6-24 optimizer steps per epoch), already queued as R4.

### No action needed
- MzansiLM Mono NER merge_lora: not a defect.
- BOS and answer position: protocol-correct for all tested vintages. Optional sensitivity: BOS for the 9 hub-provenance Intent/News adapters, <0.5 GPU-h.
