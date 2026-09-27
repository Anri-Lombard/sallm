# Plan: rescore every Mono/Multi adapter with the General protocol

Inventory: 144 rows (4 architectures x {Mono 18, Multi 18 language rows}; the 20 Multi adapters appear once per language). Details and statuses are in manifest.csv; the per-task commands are in protocol.md.

| status | rows | adapters |
|---|---|---|
| OK | 116 | exist on HEX or Kombuys with known provenance |
| SUSPECT | 13 | Mamba Mono NER x3 (6-12 optimizer steps per epoch, epoch-1 weights: untrained); xLSTM Multi NER (sweep run picked on test, other base repo) = 3 rows; xLSTM Multi SIB (1 surviving of 3 ambiguous candidates, other base repo) = 6 rows; MzansiLM Mono POS tsn (different recipe, unsealed) |
| MISSING | 15 | Mamba SIB Mono x6 + Multi (7 adapters: hub-only, no HF token on remotes, v15 arms 10-16 never ran); Mamba POS Multi (v15 run died at step ~389/540, no final adapter) |

Already done with the General protocol, so no run is needed: GDN Mono/Multi NER and POS (units 22-24, 27-29, 78, 80), 12 rows.

## A. Evaluation jobs (score everything that exists)

Keep one GPU type per host for all regimes (Kombuys GPU1 = RTX 3080 Ti, which ran General News/SIB; HEX L40S, which ran General NER/POS).

| batch | host | runs | per-run time | GPU-h |
|---|---|---|---|---|
| E1 News | Kombuys, one GPU, sequential | Mono 8 + Multi 4 (= 12). Optional: General News for mzansilm, mamba, xlstm with the BOS template (3) | 1-5 min | ~0.7 |
| E2 SIB | Kombuys, one GPU, sequential | Mono 18 (6 Mamba follow after retraining) + Multi 3 (+ Mamba, xLSTM retrain) | 1-2 min | ~0.6 |
| E3 Intent | Kombuys (or HEX after one parity rerun of a General unit) | Mono 16 + Multi 4 = 20 | ~20 min (mamba2 ~45) for all 4 langs x 5 prompts; restrict Mono to one language's pack if possible | 4-9 |
| E4 NER | HEX Slurm array, L40S | Mono 9 (mzansilm, mamba2, xlstm) + Multi 3 (mzansilm, mamba2, xlstm); later + Mamba Mono x3 retrains + xLSTM Multi retrain | 10-15 min (always 3 languages) | ~3 (+1) |
| E5 POS | HEX Slurm array, L40S | Mono 9 with the bounded single-language runner + Multi 2 (mzansilm, xlstm; Mamba Multi after retraining) + mzansilm tsn retrain | Mono 15-35 min; Multi 45-125 min | ~8 |

Per-unit prerequisites:
- NER and POS: generate one protocol JSON per adapter from the General protocol, changing only `models.<arch>` (adapter_path, adapter_files, adapter_tree_sha256), and check with `jq -S 'del(.models)'`. For mamba2 NER, start from the EOS-fix protocol.
- Adapters are split across hosts. Copy Kombuys-only adapters (xLSTM POS x4, Mamba POS Mono x3, Mamba News xho/Multi) to a new HEX directory, and HEX-only News/SIB/Intent adapters to a new Kombuys directory. Verify tree_sha256 after copying.
- Intent: build `spec.json` units with base and adapter tree shas; key rows on unit_id.
- For each Mono cell, keep only the matching language.

## B. Retraining (MISSING and SUSPECT)

No shared epoch budget exists across architectures: each architecture has its own per-cell recipe (8-50 epochs; effective batch 8-256). "Same epoch budget as the other architectures" therefore means the architecture's own frozen recipe for that cell, with checkpoint selection on validation. That is how every OK adapter of the same architecture was produced. Only GDN Multi has real HPO (adapter_hpo_v3). No HPO exists for Mamba SIB, POS or NER; the only Mamba HPO sweeps are news_xho, t2x and afrihg (reports/mamba_opt_*).

| # | cell(s) | config | where / how | est. time |
|---|---|---|---|---|
| R1 | Mamba SIB Mono afr, eng, nso, sot, xho, zul | `finetune/mamba_sib_<lang>` (lr 8e-5, r16/a32 in_proj+x_proj, cosine, warmup 0.03, wd 0.01, 10-20 ep, keep-best, patience 3; eff batch as YAML via `mamba_l40s_gradient_accumulation`) | Kombuys v15 arms 10-15: `bash .../full_matrix_targeted_recovery_20260922_kombuys_v15/audit/run_v15_kombuys_recovery_arm_20260922_v2.sh <id>`. The arm runner refuses an existing output dir, so point `root` at a new v16 directory (copy of the script) | ~0.3-0.5 h each |
| R2 | Mamba SIB Multi | `finetune/mamba_sib_all` | v15 arm 16 (same runner) | ~2 h |
| R3 | Mamba POS Multi | `finetune/mamba_pos_all` (15 ep, eff 64) | v15 arm 9, new output dir (the old one has checkpoints 324/360 but no final) | ~4 h (at 72% after 3.2 h) |
| R4 | Mamba NER Mono tsn, xho, zul | The frozen `mamba_ner_<lang>` (eff 256/256/128) gives only 6-12 steps per epoch; retraining with it reproduces the same untrained adapter. Recommendation: eff batch 64 (same as Mamba NER Multi, which trained 680 steps), lr 8e-5, 15 ep, keep-best on validation span F1. Declare this as a deviation. | Kombuys v15 runner with `gradient_accumulation_steps` override, or HEX | ~2-3 h each |
| R5 | xLSTM SIB Multi | `finetune/xlstm_sib_all` (r256/a512 incl. embeddings, 15 ep, 16x4) with pad64 and the PEFT embedding-fix snapshot (`pos-generation-recovery-20260915-v2-peft-embedding-fix`, run.py 3541cfba); validation keep-best | Kombuys (`run_xlstm_sib_mono_recovery_jbuys_v2_20260915.sh` pattern) | ~2.5 h |
| R6 | xLSTM NER Multi | `finetune/xlstm_ner_all` (r128/a256 incl. embeddings, 40 ep, 16x4), embedding-fix snapshot, validation keep-best. Rescore the existing 4vfuoadl in parallel and swap only if the retrain passes validation | HEX A100 | ~6-10 h (long pole) |
| R7 | MzansiLM POS Mono tsn | `finetune/llama_pos_tsn` (lr 3e-5, r16/a32 q,v, 20 ep, 4x4) = the recipe of the v9 xho/zul retrains | HEX or Kombuys generic v15 runner | ~0.7-2.3 h |

Alternative to R1/R2 with no GPU: the hub originals `anrilombard/sallm-mamba2-davlan-sib200-<lang>_latn` and `...-afr_latn-eng_latn-nso_latn-sot_latn-xho_latn-zul_latn` (Feb 2026, r16 in_proj/x_proj, the same pipeline as the other Mamba hub adapters). This needs either `hf auth login` on Kombuys/HEX, done by the user, or permission to download about 1 GB locally and scp it. The Jan `sallm-mamba-sib_*` family (r256 + embeddings) is the other ambiguous candidate. Pick by rule (the Feb mamba2 family), not by test score.

The HEX generic recovery sbatch (`run_targeted_recovery_a10080_v15.sbatch`) has never trained anything: its canary was cancelled. Kombuys v15 is the proven route for Mamba.

## C. Decisions needed before Phase 2

1. News chat template:
   - (b) Recommended: use the BOS template that matches training for every Mono/Multi adapter, and rerun General News for mzansilm, mamba and xlstm (about 10 GPU-min).
   - (a) Otherwise: mirror General's per-architecture template, knowing General Mamba News lacked the training [BOS].
2. Mamba NER Mono: R4 deviation (eff batch 64), or rescore the frozen-recipe adapters as they are and report ~0.
3. Mamba SIB: retrain (R1/R2) or fetch the hub originals (needs HF auth on a remote).
4. SIB and Intent validation metrics are degenerate in every training pipeline (SIB macro-F1 0.0576 for every arm; Intent 0.001-0.01), so keep-best keeps the epoch-1 weights. This hits all GDN SIB/Intent adapters (Mono and Multi) and probably others. Rescoring fixes the evaluation, not this. Fixing it would mean retraining with a fixed validation metric, about 26 more adapters. Out of scope unless requested; flagged in manifest notes.
5. MzansiLM News/NER/SIB task adapters were trained on the long v5/v1 training templates (r128 + embed/lm_head) and are scored on the lm_eval_p* prompts, as the paper's appendix states. This is a prompt-transfer handicap that the other architectures do not have. Keep it and state it, or retrain with lm_eval prompts; the latter is a large job.

## D. ETA

- Evaluation of the existing adapters: ~17-21 GPU-h.
  - Kombuys: News + SIB ~1.5 h on one GPU; Intent 4-9 h.
  - HEX: NER + POS arrays ~11 GPU-h, about 2-3 h wall if L40S nodes are free.
  - Can start tonight: Kombuys GPU0 was idle at 22:40, GPU1 busy until ~midnight. HEX is available now.
- Retraining: ~28-35 GPU-h.
  - Kombuys: R1-R5 ~16 h sequential on one GPU, ~9 h on two.
  - HEX: R6 ~6-10 h, R7 ~1-2 h.
- Rescoring the retrained adapters: ~2 GPU-h.
- Wall clock with Kombuys (2 GPUs after midnight) and HEX in parallel: about 1 day (~24 h), gated by R6 and the Kombuys Mamba queue. All OK cells can be final within ~10 h.
