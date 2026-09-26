# Architecture-comparison paper: evaluation and training runners

Verbatim copies of the runners that produced the numbers in the architecture-comparison paper
(`sa-architecture-comparison-paper`, results built from `data/*.csv` by `scripts/build_results.py`).
Files were copied from the directories they ran from on HEX (`/scratch/lmbanr001/masters/...`) and
Kombuys (`/scratch/alombard/...`), not from working drafts. Paths inside them are the cluster paths
they ran with; they are kept as provenance, not as portable tools.

The commit that added this directory holds the exact as-run bytes. Later commits changed only the
NFC normalisation of scored text (see "Known issues").

## Library version

Every runner puts a frozen snapshot of this repo's `src/main` on `PYTHONPATH`:

| family | library snapshot |
|---|---|
| generation, Intent/Belebele, NER/POS | `sallm_snapshots/downstream-generation-20260914-v8` |
| News | `sallm_snapshots/news-general-corrected-official-20260914-v15` |
| SIB-200 | `snapshots/retained-sib-evaluator-20260913-v6` with the frozen kit in `monomulti/retrain/sibkit/` |

`src/main/sallm` in this repo is a superset of v8. It adds batch-1 label scoring for Mamba
(`classification_metrics.py`, `constrained_label_scoring.py`), `use_cache=False` for Mamba in
`harness.py` (v9 mamba-cache-fix), the PEFT embedding de-duplication in `fine_tune/run.py`, and
xLSTM inference mode inside the training callback.

## Result family to runner

| paper result | runner (entry point) | launcher |
|---|---|---|
| Base / General / Mono / Multi T2X and AfriHG generation (greedy, no system prompt) | `generation/run_generation_protocol.py` (inventory units), `generation/run_generation_direct.py` (extra units) | `generation/v3.sbatch` |
| xLSTM generation (batch 1, recurrent cache, no padding) | `generation/run_generation_direct_bs1.py` | `generation/run_xlstm_bs1.sbatch` |
| General InjongoIntent and Belebele at the training answer position | `prompt/run_prefix_eval.py --mode train` | `prompt/run_all.sh`, `prompt/run_control.sh`; `prompt/build_csv.py` |
| General News | `scripts/run_news_natural_first_token_official.py` + `scripts/run_mamba_news_interface_diagnostic.py` (identical to the v15 snapshot) | Kombuys |
| General SIB-200 | `scripts/run_sib_natural_first_token_official_replacement.py`; scorer `monomulti/retrain/sibkit/run_sib_natural_first_token_eval.py` (v5, sha b247505a) | Kombuys |
| General NER (BOS-repair) and POS | `sequence/ner_bos_v2/run_general_sequence_eval_20260917_v2.py` | `sequence/ner_bos_v2/run_general_ner_bos_repair_v2.sbatch`; integration `integrate_general_sequence_official_test_20260917_v2.py` |
| General POS test release and integration | `sequence/general_20260916/` | `run_general_sequence_official_test_hex_20260916_v1.sbatch` |
| xLSTM NER/POS/lm-eval reruns at batch 1 | `sequence/xlstm_bs1/*_bs1.py` | `sequence/xlstm_bs1/run_remaining_l40s.sbatch`, `run_other_bs1.sh` |
| Mamba EOS/PAD-fixed base and its NER/AfriMGSM reruns | `sequence/mamba_eos_fix/make_base.sh`, `make_protocols.py` | `sequence/mamba_eos_fix/*.sbatch` |
| Mono POS (single-language bounded runner) | `sequence/mono_pos/run_gdn_mono_pos.py` | `sequence/mono_pos/run.sbatch` |
| Full-matrix bindings, inventory and official release gates | `full_matrix/` (`prepare_bindings.tree_sha256` is imported by most runners) | `full_matrix/*.sbatch` |
| Mono/Multi rescoring of existing adapters with the General protocol | `monomulti/rescore/score_news_trainpos.py`, `score_sib_trainpos.py`, Intent via `prompt/run_prefix_eval.py`, NER/POS via `sequence/` | `monomulti/rescore/run_unit.sbatch`, `run_array.sbatch`; `parse_hex.py`, `parse_kom.py`, `assemble.py` build `rescore_results.csv` |
| Mono/Multi retrains (fewer than 10 optimizer steps per epoch, missing or suspect adapters) | `monomulti/retrain/hex/train_hex_v2.sbatch`, `kombuys/run_mamba_retrain_v3.sh`; selection `hex/score_*_tt.py`, `hex/seqsel.py` | `monomulti/retrain/build_results.py` builds `retrain_results.csv` |
| Mono/Multi reselection (every epoch saved, validation selection with the protocol scorer) | `monomulti/reselect/hex/*_train.sh`, `kit/reselect.py`, `kit/*_score.py`, `kit/intent_eval.py` | `monomulti/reselect/hex/lanes.sbatch`, `valpass.sh`; `hex/collect.py` builds `reselect_results.csv` |
| Equal-recipe full fine-tuning (T2X pilot, lr sweep, seeds) | `full_ft/train_fft.py`, `full_ft/fft_tools.py`; evaluation `full_ft/run_generation_direct_fullft.py` | `full_ft/sweep_arch.sbatch`, `full_ft/final.sh`; recipe in `full_ft/recipe.md`, hashes in `full_ft/SCRIPTS.sha256` |

The Mono/Multi protocol, plan and decision logs are in `monomulti/*.md`. The paper repo's
`scripts/collect_monomulti.py` merges the rescore, retrain and reselect CSVs.

## Known issues

- The tokenizer normaliser is NFD, so decoded outputs are decomposed while references are NFC.
  The runners above scored without NFC normalisation; chrF and NER span F1 are underestimated
  wherever outputs contain diacritics. The shared scorers now normalise both sides to NFC.
- xLSTM ignores the attention mask under left padding, hence the batch-1 runners.
- Mamba evaluation must use FLA's Mamba-2 class (transformers normalises the gated RMSNorm over
  all channels instead of per group).
