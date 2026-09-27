# Archive

Files kept for provenance but no longer part of the maintained workflow.
Nothing in `src/`, `ops/`, `tests/` or CI reads them.

- `envs/`: the 2025 conda environment (`environment.yml`) and a `pip freeze`
  from that environment (`req.txt`). The maintained environment is
  `pyproject.toml` plus `uv.lock`.
- `scripts/get_wandb_sweep_info.py`: one-off W&B sweep summary.
- `scripts/generate_xlstm_configs.py`: one-off generator that produced the
  `src/conf/finetune/xlstm_*.yaml` configs from the Mamba ones.
- `docs/hpo_mamba_sweeps_last_14_days.md`: generated report of the Mamba HPO
  sweeps, 3 February 2026.
- `docs/masakhaner-x-issue.md`: draft upstream request to convert
  `masakhane/masakhaner-x` to Parquet.
