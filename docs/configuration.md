# Configuration

Hydra configs live under `src/conf`. Pass any target path without `.yaml` to
`python -m sallm.main --config-name <target>`, and override any key on the
command line.

## Pretraining configs

- **Matched-budget pretraining (`src/conf/pretrain/`):** Five matched ~125M WSD
  JSON configs (`mzansilm_wsd.json`, `mamba2_wsd.json`, `xlstm_wsd.json`,
  `gdn_wsd.json`, `hybrid_wsd.json`) consumed by `python -m sallm.pretrain`.
- **Legacy Hydra pretraining (`src/conf/base/`):** Older per-architecture YAML
  configs (`llama_125m`, `llama_400m`, `mamba_125m`, `xlstm_125m`,
  `gated_deltanet_125m`) reading `uctnlp/mzansi-text-tokenized`.

## Fine-tuning configs

`finetune/run` composes any architecture and dataset dynamically from three
shared building blocks:

1. `finetune/defaults/common`: full fine-tuning protocol (fused AdamW, cosine
   schedule, 10% warmup, effective batch 16, bf16 autocast, TF32, Hub push
   disabled by default).
2. `finetune/defaults/datasets/<task>`: dataset source, prompt templates,
   splits, and task-specific selection metrics (`sib`, `injongointent`,
   `masakhanews`, `masakhaner`, `nchlt_ner`, `masakhapos`, `afrihg`, `t2x`,
   `sa_multitask`).
3. `finetune/defaults/arch/<arch>`: model architecture and default base
   checkpoint (`llama`, `mamba2`, `xlstm`, `gated_deltanet`, `hybrid`).

Example:

```bash
uv run python -m sallm.main --config-name finetune/run \
  finetune/defaults/arch@_global_=hybrid \
  finetune/defaults/datasets@_global_=sib \
  dataset.subset=xho_Latn \
  model.init_checkpoint=/path/to/hybrid \
  wandb.name=hybrid-sib-xho
```

Three standalone recipe targets (`finetune/llama_t2x_xho`,
`finetune/mamba_news_xho`, `finetune/xlstm_sib_xho`) remain for `recipes/registry.yaml`
and `sweeps/llama_t2x_xho.yaml`.

Fine-tuning updates all weights from float32 checkpoints under bf16 autocast.
`num_train_epochs: auto` runs 10 epochs for datasets below 5,000 examples and 4
otherwise, with early stopping on the task's validation metric. Two SALLM-only
keys under `training` control validation behavior:

- `task_metrics` (default `true`): runs the task evaluator on validation each
  evaluation step. When `false`, early stopping falls back to `eval_loss` unless
  `metric_for_best_model` is explicitly set.
- `general_selection` (default `false`): selects multitask checkpoints by
  equal-family assistant-token NLL.

## Evaluation configs

`eval/run` evaluates any full checkpoint (Hub id or local directory) on one or
more `lm-eval` task packs:

```bash
uv run python -m sallm.main --config-name eval/run \
  eval_model.checkpoint=uctnlp/mzansilm-125m \
  'evaluation.task_packs=[sib_xho]' \
  wandb.name=eval-mzansilm-sib-xho
```

`eval/generate_<task>` runs generation tasks (AfriHG, T2X); the `_base` variants
use base-model decoding settings. Task packs live in `eval/tasks/` and
`rerank/tasks/`, with custom `lm-eval` task definitions under
`eval/lm_eval_tasks/` and `rerank/lm_eval_tasks/`.

## Sweeps

`sweeps/llama_t2x_xho.yaml` provides a W&B sweep template. Pass a base
fine-tuning config via `--base-config`; `ops/slurm/launch_hpo.sh` registers and
launches it.

## Checking changes

```bash
uv run pytest tests/test_config_imports.py
```

This verifies all 45 `(architecture x task)` combinations of `finetune/run`,
every entry YAML under `base/`, `finetune/`, and `eval/`, and every sweep's base
config against `ExperimentConfig`.

## Environment variables

Paths use `${oc.env:SCRATCH}` and `${oc.env:HOME}`. On SLURM,
`ops/slurm/lib/env.sh` derives them from `SALLM_SCRATCH_DIR`, `SALLM_HOME_DIR`,
and `SALLM_REPO_DIR`. Dataset download caches can be redirected via
`SALLM_SOURCE_CACHE_DIR`, `SALLM_AFRIHG_CACHE_DIR`, and `SALLM_T2X_CACHE_DIR`;
`SALLM_AFRIHG_CACHE_ONLY=1` disables network downloads for AfriHG.
