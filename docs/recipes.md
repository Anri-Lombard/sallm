# Recipes

Recipes are named public entrypoints for known fine-tuning and evaluation
configs. They live in `recipes/registry.yaml` and are loaded by
`src/main/sallm/recipes.py`.

The registry maps a stable recipe ID to one or more Hydra config targets. The
recipe ID chooses the workflow; the YAML config owns the hyperparameters.

## CLI

List available recipes:

```bash
uv run sallm recipes list
```

Show one recipe:

```bash
uv run sallm recipe show llama_t2x_xho
```

Dry-run a recipe action:

```bash
uv run sallm finetune llama_t2x_xho --dry-run
uv run sallm evaluate llama_t2x_xho --dry-run
```

The dry-run output includes the resolved Hydra config target and the underlying
`python -m sallm.main --config-name ...` command.

## Current Registry

| Recipe ID | Fine-Tune Config | Evaluation Config | Tags |
| --- | --- | --- | --- |
| `llama_t2x_xho` | `finetune/llama_t2x_xho` | `eval/run_llama_t2x_xho` | `llama`, `t2x`, `xho` |
| `mamba_news_xho` | `finetune/mamba_news_xho` | `eval/run_mamba_masakhanews_xho` | `mamba`, `masakhanews`, `xho` |
| `xlstm_sib_xho` | `finetune/xlstm_sib_xho` | `eval/run_xlstm_sib_xho` | `xlstm`, `sib`, `xho` |

## Adding or Updating a Recipe

1. Add or update an entry in `recipes/registry.yaml`.
2. Point each action at an existing config target under `src/conf`, without the
   `.yaml` suffix.
3. Keep hyperparameters in the YAML config, not in the recipe entry.
4. Run the focused recipe checks:

```bash
uv run pytest tests/test_cli.py tests/test_recipes.py
```

The loader validates that each recipe config target stays inside the Hydra
config tree and resolves to an existing YAML file.

## What Recipes Do Not Do

Recipes do not provide model, task, or language flag sugar. Use one of the
registered recipe IDs, or call the direct Hydra entrypoint with an explicit
config target:

```bash
uv run python -m sallm.main --config-name <config-target>
```
