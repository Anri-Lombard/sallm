# Cluster Scripts

The public entrypoint is the recipe CLI in [Quickstart](quickstart.md). The
`scripts/` directory is for existing SLURM workflows that need repository-
specific setup before calling `sallm.main`.

Single-config runners take a Hydra config target without the `.yaml` suffix:

```bash
sbatch scripts/launch_finetune.sh finetune/llama_t2x_xho
sbatch scripts/launch_evaluation.sh eval/run_llama_t2x_xho
```

Batch helpers submit or resume groups of jobs:

- `scripts/submit_finetune_eval.sh`
- `scripts/submit_llama_finetune_eval.sh`
- `scripts/launch_all_xlstm_finetunes.sh`
- `scripts/launch_hpo.sh`
- `scripts/resume_hpo.sh`

The shared defaults live in `scripts/lib/env.sh`. Override them when your
cluster paths differ:

```bash
export SALLM_HOME_DIR="$HOME"
export SALLM_SCRATCH_DIR="/scratch/$USER"
export SALLM_REPO_DIR="$HOME/masters/sallm"
export SALLM_SLURM_USER="$USER"
```

Wrapper scripts that call `sbatch` also respect `SALLM_SLURM_ACCOUNT` and
`SALLM_SLURM_MAIL_USER`.
