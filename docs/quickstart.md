# Quickstart

This guide uses the public `sallm` CLI. The CLI is intentionally small: it
lists known recipes, shows their config targets, and runs those targets through
the existing Hydra entrypoint.

## Install

From the repository root:

```bash
uv sync
```

For local checks:

```bash
uv sync --extra dev
```

## Inspect Recipes

List the recipe IDs that are currently registered:

```bash
uv run sallm recipes list
```

Inspect one recipe before launching anything:

```bash
uv run sallm recipe show llama_t2x_xho
```

The recipe output shows the config targets used for each action. For example,
`llama_t2x_xho` maps fine-tuning to `finetune/llama_t2x_xho` and evaluation to
`eval/run_llama_t2x_xho`.

## Dry-Run Fine-Tuning and Evaluation

Use `--dry-run` first. It prints the resolved recipe, action, config target,
and underlying Python command without launching the run.

```bash
uv run sallm finetune llama_t2x_xho --dry-run
uv run sallm evaluate llama_t2x_xho --dry-run
```

When you are ready to launch the resolved config, remove `--dry-run`:

```bash
uv run sallm finetune llama_t2x_xho
uv run sallm evaluate llama_t2x_xho
```

Runs may require local Hugging Face, Weights & Biases, dataset, GPU, or cluster
setup depending on the config being launched.

## Direct Hydra Entry Point

Recipes are a thin public layer over Hydra configs. If you already know the
config target, you can call the Hydra entrypoint directly:

```bash
uv run python -m sallm.main --config-name <config-target>
```

Examples:

```bash
uv run python -m sallm.main --config-name finetune/llama_t2x_xho
uv run python -m sallm.main --config-name eval/run_llama_t2x_xho
```

Use recipe IDs for documented public workflows. Use direct Hydra targets when
you are working with the config tree itself.

## Next Steps

- See [Recipes](recipes.md) for the current public recipe registry.
- See [Configuration](configuration.md) for how recipe targets map to YAML.
- See [SLURM](slurm.md) if you need the advanced cluster scripts.
