# Configuration

SALLM runs are configured with Hydra YAML files under `src/conf`. The public
recipe CLI resolves recipe IDs to those config targets; it does not duplicate
or override the hyperparameters.

## Config Targets

A config target is a path under `src/conf` without the `.yaml` suffix.

Examples:

```text
finetune/llama_t2x_xho
eval/run_llama_t2x_xho
base/llama_125m
```

The direct Hydra command shape is:

```bash
uv run python -m sallm.main --config-name <config-target>
```

For example:

```bash
uv run python -m sallm.main --config-name finetune/llama_t2x_xho
```

## Recipe Mapping

Recipe IDs are defined in `recipes/registry.yaml`:

```yaml
- id: llama_t2x_xho
  description: Fine-tune and evaluate Llama on isiXhosa T2X data-to-text.
  configs:
    finetune: finetune/llama_t2x_xho
    evaluate: eval/run_llama_t2x_xho
  tags:
    - llama
    - t2x
    - xho
```

The recipe loader validates that each target resolves to an existing YAML file
inside `src/conf`. For the example above, the resolved files are:

```text
src/conf/finetune/llama_t2x_xho.yaml
src/conf/eval/run_llama_t2x_xho.yaml
```

## Source of Truth

Keep run behavior in the config files:

- model, tokenizer, dataset, and training settings belong in YAML.
- recipe entries should stay small and point to known configs.
- CLI commands should use recipe IDs or exact Hydra config targets.
- cluster-specific runtime settings belong in the SLURM environment or script
  arguments, not in public CLI flag shortcuts.

## Config Areas

- `src/conf/base`: base model and pretraining configs.
- `src/conf/finetune`: fine-tuning configs.
- `src/conf/eval`: evaluation configs and task definitions.
- `src/conf/sweeps`: HPO sweep configs.
- `src/conf/templates`: prompt and formatting templates.
- `src/conf/datasets`: dataset config.
- `src/conf/tokenizers`: tokenizer config.

## Checking Config Changes

For recipe-facing config changes, run:

```bash
uv run pytest tests/test_cli.py tests/test_recipes.py
```

For a launch preview, prefer a recipe dry-run when the config is registered:

```bash
uv run sallm finetune llama_t2x_xho --dry-run
uv run sallm evaluate llama_t2x_xho --dry-run
```
