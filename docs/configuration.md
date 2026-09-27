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

`eval/run` evaluates one checkpoint on lm-eval task packs; name the checkpoint,
the packs and the run on the command line:

```bash
uv run python -m sallm.main --config-name eval/run \
  eval_model.checkpoint=anrilombard/sallm-mamba-125m \
  'evaluation.task_packs=[sib_xho]' wandb.name=eval-mamba-sib-xho
```

Results go to `$SCRATCH/masters/sallm/results/eval/<wandb.name>`. Add
`eval_model.merge_lora=true` or `eval_model.peft_adapter=<path>` for adapters.

`eval/generate_<task>` runs the generation tasks (AfriHG, T2X) the same way;
the `_base` variants use the decoding settings for base models. The
`eval/run_<recipe>` files are the evaluation targets of the recipes.

A task pack in `eval/tasks/<name>.yaml` lists lm-eval tasks and their settings;
packs for validation reranking live in `rerank/tasks/`. Task definitions
maintained in this repository live in `eval/lm_eval_tasks/` and
`rerank/lm_eval_tasks/`; a pack that uses them lists the directory under
`task_manager_kwargs.include_path`.

## Sweeps

`sweeps/llama_t2x_xho.yaml` is a W&B sweep template. A sweep names its base
fine-tuning config with `--base-config`; `ops/slurm/launch_hpo.sh` registers and
runs it. The sweeps behind the paper results are on the paper branch.

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
