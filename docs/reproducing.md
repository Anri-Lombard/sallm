# Reproducing the Papers

`main` is maintained code. Paper numbers come from frozen snapshots of the code
that produced them; this page says where each snapshot lives and in which order
to run it.

## MzansiLM (LREC 2026)

Check out the tag `mzansitext-mzansilm-lrec2026-v1`; its README covers the
whole pipeline.

## Architecture comparison (Transformer, Mamba-2, xLSTM, Gated DeltaNet)

Code: branch `paper/architecture-comparison-2026-09`, tagged once the runs
finish. It pins the environment (`uv.lock`, transformers 4.57.3) and keeps the
as-run code under `scripts/paper/`; `scripts/paper/README.md` maps each result
to its runner. Tables, figures and statistics are built in the paper repository
[Anri-Lombard/sa-architecture-comparison-paper](https://github.com/Anri-Lombard/sa-architecture-comparison-paper)
(private).

Pipeline, in order:

1. **Pretraining.** `scripts/paper/matched_pretrain/` trains the four
   matched-budget base models (configs in `configs/`, launchers in `code/`).
2. **Fine-tuning and scoring.** `scripts/paper/fft_rollout/` runs, per
   architecture, the learning-rate sweeps, Mono, Multi and Multitask training,
   test scoring and the extra seeds as one dependency graph. Launch and monitor
   it with `RUNBOOK.md` sections 0-3; the `collect` unit writes
   `results/cells.csv` per architecture.
3. **Tables.** In the paper repository, fetch the per-architecture results and
   run `scripts/collect_fft.py` and then `scripts/build_results.py`
   (`RUNBOOK.md` section 4 has the exact commands and file mapping).
4. **Statistics and item-level analyses.** `scripts/analysis/fetch_items.sh`,
   then `scripts/analysis/item_analyses.py` (bootstrap intervals and per-item
   comparisons).
5. **Figures.** `scripts/make_lm_figures.py` and `scripts/make_ft_figures.py`.

The fine-tuning protocol on `main` (`src/conf/finetune/defaults/common.yaml`)
follows the paper's, but `main` runs transformers 5.x, so rerunning on `main` is
a new measurement rather than a reproduction.
