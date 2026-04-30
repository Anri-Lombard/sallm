# Cluster Scripts

The public entrypoint is the recipe CLI in [Quickstart](quickstart.md). The
`ops/slurm/` directory contains the small set of supported SLURM entrypoints for
cluster execution.

Single-config runners take a Hydra config target without the `.yaml` suffix:

```bash
sbatch ops/slurm/launch_finetune.sh finetune/llama_t2x_xho
sbatch ops/slurm/launch_evaluation.sh eval/run_llama_t2x_xho
```

HPO runners take a sweep config name or W&B sweep path:

```bash
sbatch ops/slurm/launch_hpo.sh llama_t2x_xho 10
sbatch ops/slurm/resume_hpo.sh anri-lombard/sallm-ft/<sweep-id> 43
```

The shared defaults live in `ops/slurm/lib/env.sh`. Override them when your
cluster paths differ:

```bash
export SALLM_HOME_DIR="$HOME"
export SALLM_SCRATCH_DIR="/scratch/$USER"
export SALLM_REPO_DIR="$HOME/masters/sallm"
export SALLM_SLURM_USER="$USER"
```

Scripts that call `sbatch` also respect `SALLM_SLURM_ACCOUNT` and
`SALLM_SLURM_MAIL_USER`. One-off batch launchers and architecture-specific final
training scripts are intentionally not part of the supported public surface.
