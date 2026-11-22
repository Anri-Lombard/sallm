# Configuration

Pretraining reads JSON configs under `src/conf/pretrain/` via `python -m sallm.pretrain`. Fine-tuning and evaluation use Hydra configs under `src/conf/` via `python -m sallm.main --config-name <target>`, with any key overridable on the command line.

## Pretraining configs

Five matched ~125M WSD JSON configs live in `src/conf/pretrain/`:

- `mzansilm_wsd.json`
- `mamba2_wsd.json`
- `xlstm_wsd.json`
- `gdn_wsd.json`
- `hybrid_wsd.json`

## Fine-tuning configs

`finetune/run` composes any architecture and dataset dynamically from three shared building blocks:

1. `finetune/defaults/common`: full fine-tuning protocol (fused AdamW, cosine schedule, 10% warmup, effective batch 16, bf16 autocast, TF32, Hub push disabled by default).
2. `finetune/defaults/datasets/<task>`: dataset source, prompt templates, splits, and task-specific selection metrics (`sib`, `injongointent`, `masakhanews`, `masakhaner`, `nchlt_ner`, `masakhapos`, `afrihg`, `t2x`, `sa_multitask`).
3. `finetune/defaults/arch/<arch>`: model architecture and default base checkpoint (`llama`, `mamba2`, `xlstm`, `gated_deltanet`, `hybrid`).

Example:

```bash
uv run python -m sallm.main --config-name finetune/run \
  finetune/defaults/arch@_global_=hybrid \
  finetune/defaults/datasets@_global_=sib \
  dataset.subset=xho_Latn \
  model.init_checkpoint=/path/to/hybrid \
  wandb.name=hybrid-sib-xho
```

Fine-tuning updates all weights from float32 checkpoints under bf16 autocast. `num_train_epochs: auto` runs 10 epochs for datasets below 5,000 examples and 4 otherwise, with early stopping on the task's validation metric. Two SALLM-only keys under `training` control validation behavior:

- `task_metrics` (default `true`): runs the task evaluator on validation each evaluation step. When `false`, early stopping falls back to `eval_loss` unless `metric_for_best_model` is explicitly set.
- `general_selection` (default `false`): selects multitask checkpoints by equal-family assistant-token NLL.

## Evaluation configs

`eval/run` evaluates any full checkpoint (Hub id or local directory) on one or more `lm-eval` task packs from `src/conf/eval/tasks/`:

```bash
uv run python -m sallm.main --config-name eval/run \
  eval_model.checkpoint=uctnlp/mzansilm-125m \
  'evaluation.task_packs=[sib_xho]' \
  wandb.name=eval-mzansilm-sib-xho
```

Validation task packs (`*_val.yaml`) are blocked under the default `eval` scope to avoid mixing validation and test runs; pass `evaluation.task_pack_scope=validation` to evaluate them explicitly. Custom `lm-eval` task definitions live under `src/conf/eval/lm_eval_tasks/`.

For generative benchmarks, pass one of four configs:

- `eval/generate_t2x` and `eval/generate_t2x_base` (T2X isiXhosa)
- `eval/generate_afrihg` and `eval/generate_afrihg_base` (AfriHG headline generation)

## Sweeps

`src/conf/sweeps/lr_grid.yaml` provides a W&B learning-rate sweep template built on `finetune/run` with `--override` arguments forwarded to `sallm.hpo.run`. `ops/slurm/launch_hpo.sh` registers and launches it on SLURM.

## Checking changes

```bash
uv run pytest tests/test_config_imports.py
```

This verifies all 45 `(architecture x task)` combinations of `finetune/run`, every entry YAML under `finetune/` and `eval/`, and every sweep config against `ExperimentConfig`.

## Environment variables

Paths use `${oc.env:SCRATCH,outputs}` and `${oc.env:TOKENIZER_PATH,tokenizer/sallm_bpe_tokenizer}`. On SLURM, `ops/slurm/lib/env.sh` derives cluster paths from `SALLM_SCRATCH_DIR`, `SALLM_HOME_DIR`, and `SALLM_REPO_DIR`. Dataset download caches can be redirected via `SALLM_SOURCE_CACHE_DIR`, `SALLM_AFRIHG_CACHE_DIR`, and `SALLM_T2X_CACHE_DIR`; `SALLM_AFRIHG_CACHE_ONLY=1` disables network downloads for AfriHG.
