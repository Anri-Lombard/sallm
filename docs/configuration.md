# Configuration

Runs are Hydra configs under `src/conf`. A target is a path there without
`.yaml`, for example `finetune/mamba_sib_xho`, and runs with
`python -m sallm.main --config-name <target>`.

## Fine-tuning configs

Each `finetune/<arch>_<task>_<lang>.yaml` composes, in order:

1. `finetune/defaults/datasets/<task>`: dataset, templates and splits.
2. `finetune/defaults/common`: settings shared by every fine-tuning run.
3. `finetune/defaults/arch/<arch>`: model, LoRA and training defaults for one
   architecture.
4. The file itself: run names and paths, plus any value that differs from the
   defaults (usually the tuned hyperparameters).

Later entries win. To change a default for every run of an architecture, edit
its `arch/` file; to change one run, set the value in that run's file.

Two `training` options are handled by SALLM rather than passed to the
Transformers trainer: `task_metrics` (default `true`) runs the task scorer on
the validation set each evaluation, and `general_selection` (default `false`)
selects checkpoints by equal-family assistant-token loss on General data.

## Evaluation configs

`eval/run_*.yaml` compose `eval/defaults/run` and set the checkpoint, task packs,
output directory and W&B name. A task pack in `eval/tasks/<name>.yaml` lists
lm-eval tasks and their settings; packs for validation reranking live in
`rerank/tasks/`. Task definitions maintained in this repository live in
`eval/lm_eval_tasks/` and `rerank/lm_eval_tasks/`; a pack that uses them lists
the directory under `task_manager_kwargs.include_path`.

## Sweeps

`sweeps/*.yaml` are W&B sweep files. Each names its base fine-tuning config
with `--base-config`; `ops/slurm/launch_hpo.sh` registers and runs them.

## Checking changes

`uv run pytest tests/test_config_imports.py` composes every base, fine-tuning
and evaluation config against the schema and resolves every sweep's base
config.

## Environment

Paths use `${oc.env:SCRATCH}` and `${oc.env:HOME}`. On SLURM, `ops/slurm/lib/env.sh`
sets them from `SALLM_SCRATCH_DIR`, `SALLM_HOME_DIR` and `SALLM_REPO_DIR`;
`SALLM_SLURM_ACCOUNT` and `SALLM_SLURM_MAIL_USER` are added to the `sbatch` calls the launchers make.
`SALLM_SOURCE_CACHE_DIR`, `SALLM_AFRIHG_CACHE_DIR` and `SALLM_T2X_CACHE_DIR` move
the dataset source caches, and `SALLM_AFRIHG_CACHE_ONLY=1` forbids downloads.
