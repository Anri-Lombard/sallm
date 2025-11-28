# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SALLM is a modular framework for training small language models for South African languages. The framework supports:

- Foundation model pretraining (Llama, Mamba architectures)
- Fine-tuning for downstream tasks (classification, NER, POS tagging, instruction following)
- Comprehensive evaluation using lm-evaluation-harness and custom generation metrics
- Hyperparameter optimization via WandB sweeps
- Distributed training with Accelerate

## Architecture

### Core Components

**Entry Point & Configuration**

- `src/main/sallm/main.py`: Main entry point using Hydra for config composition
- `src/main/sallm/config.py`: Pydantic dataclasses defining all configuration schemas
- `src/conf/`: YAML configuration files organized by purpose (base/, finetune/, eval/, sweeps/, templates/)
- Hydra composes configs from multiple files; use `--config-name` to specify the primary config

**Three Run Modes**

1. `TRAIN`: Foundation model pretraining (`src/main/sallm/training/run.py`)
2. `FINETUNE`: Task-specific fine-tuning with optional PEFT (`src/main/sallm/fine_tune/run.py`)
3. `EVALUATE`: Model evaluation with lm-eval and generation tasks (`src/main/sallm/evaluation/run.py`)

**Factory Pattern**

- `src/main/sallm/models/factory.py`: Model and tokenizer instantiation
- `src/main/sallm/data/factory.py`: Dataset builders for pretraining and fine-tuning
- `src/main/sallm/training/factory.py`: Trainer configuration and instantiation

**Model Registry**

- `src/main/sallm/models/registry.py`: Maps architecture names to HuggingFace model classes
- `src/main/sallm/evaluation/registry.py`: Registers custom lm-eval tasks

**Data Pipelines**

- `src/main/sallm/data/t2x.py`: T2X pretokenized dataset format (used for foundation training)
- `src/main/sallm/data/afrihg.py`: AfriHG dataset loader
- `src/main/sallm/data/multitask.py`: Multitask mixture datasets with temperature-based sampling

**Templates & Evaluation**

- `src/main/sallm/templates/registry.py`: Chat templates and prompt formatting
- `src/main/sallm/evaluation/harness.py`: Integration with lm-evaluation-harness
- `src/main/sallm/evaluation/generation_metrics.py`: Custom metrics (BLEU, ROUGE, chrF++)

## Development Commands

### Setup

```bash
poetry install
```

### Code Quality

```bash
make format    # Black, Ruff format, Ruff autofix
make lint      # Ruff + yamllint on src/conf/**/*.yaml
make type      # mypy on src/ and tokenizer/
make test      # pytest -q
make check     # All of the above + pre-commit hooks
```

### Single Test

```bash
poetry run pytest -q tests/<path_to_test.py> -k test_function_name
```

### Dependency Management

```bash
make deps-outdated  # Check for outdated packages
make deps-update    # Update dependencies
make export-req     # Export requirements.txt
```

### Pre-commit

```bash
make pre-commit-install  # Install hooks
make pre-commit-check    # Run all hooks
```

## Running Experiments

### Local Execution

```bash
# Foundation training
poetry run python -m sallm.main --config-name base/llama_125m

# Fine-tuning
poetry run python -m sallm.main --config-name finetune/llama_afrihg_all

# Evaluation
poetry run python -m sallm.main --config-name eval/run_llama_belebele_eng
```

### Distributed Training (Accelerate)

```bash
accelerate launch --num_processes 4 --mixed_precision bf16 \
  -m sallm.main --config-name base/llama_125m
```

### SLURM Cluster

**Remote Server Access**

```bash
ssh hex  # Connect to HPC cluster where sbatch commands run
```

**Job Submission**

```bash
# Fine-tuning (MUST include finetune/ prefix, no .yaml extension)
sbatch scripts/launch_finetune.sh finetune/llama_afrihg_all

# Evaluation (auto-prefixes with "eval/")
sbatch scripts/launch_evaluation.sh run_llama_belebele_eng

# Foundation training (hardcoded config in script)
sbatch scripts/train_final_model.sh

# Batch submission example for multiple fine-tuning jobs
for config in finetune/mamba_ner_tsn finetune/mamba_ner_xho finetune/mamba_ner_zul; do
  sbatch scripts/launch_finetune.sh $config
done
```

## Configuration System

### Hydra Config Composition

Base config (`src/conf/config.yaml`) uses `defaults` to compose from groups:

- `model`: Architecture definition (base/llama_125m.yaml, base/mamba_125m.yaml)
- `data`: Dataset paths and splits
- `tokenizer`: Tokenizer path
- `training`: Trainer arguments (learning rate, batch size, epochs, etc.)
- `evaluation`: Task packs and generation tasks
- `finetune`: Dataset, templates, PEFT settings

### Key Config Patterns

**Foundation Training**

- Set `mode: TRAIN`
- Specify `model.architecture` (llama/mamba) and `model.config` for architecture hyperparams
- Point `data.path` to pretokenized T2X dataset
- Configure `training` block with optimizer settings

**Fine-tuning**

- Set `mode: FINETUNE`
- Use `model.init_checkpoint` to load pretrained model
- Configure `dataset` block with HF dataset name, templates, packing, etc.
- Optionally add `peft` config for LoRA/QLoRA

**Evaluation**

- Set `mode: EVALUATE`
- Configure `eval_model` block (checkpoint path, device, dtype)
- Specify `evaluation.task_packs` for lm-eval tasks
- Add `evaluation.generation_tasks` for custom generation metrics

### Checkpoint Resolution

The `ModelEvalConfig` class automatically resolves checkpoint paths with fallback logic:

- Tries `final_merged_model`, `final_adapter`, `final_model` subdirectories
- Auto-detects PEFT adapters by looking for `adapter_config.json`
- If adapter found, loads base model + adapter (can merge with `merge_lora: true`)

## Tokenizer

Custom BPE tokenizer located in `tokenizer/sallm_bpe_tokenizer/`:

- Trained on multilingual South African language data
- Training scripts in `tokenizer/train.py`
- Pretokenize datasets with `scripts/pretokenize_dataset.py`

## WandB Integration

All runs log to WandB by default:

- Project name specified in `wandb.project`
- Supports run resumption with `wandb.id`
- HPO sweeps detected by "sweep" in `wandb.id`

## Important Patterns

### PEFT (LoRA/QLoRA)

Fine-tuning supports PEFT via the `peft` config block:

```yaml
peft:
  method: qlora
  kwargs:
    r: 64
    lora_alpha: 16
    lora_dropout: 0.05
    target_modules: [q_proj, v_proj]
```

### Mixture Datasets

Use `mix:` prefix in `dataset.hf_name` or specify `dataset.mix_name`:

```yaml
dataset:
  mix_name: multilingual_tasks
  mix_weights:
    task_a: 0.6
    task_b: 0.4
  mix_temperature: 1.0
```

### Template System

Templates define prompt formatting for tasks. Reference templates in datasets:

```yaml
dataset:
  templates:
    - id: template_v1
      weight: 1.0
  template_choice: cycle # cycle | random | all
```

### Distributed Training Context

The codebase handles multi-GPU training via Accelerate. Most modules check `LOCAL_RANK` environment variable to determine main process for logging.

## Code Style (from AGENTS.md)

- No inline comments or docstrings in source files
- Use type hints and expressive signatures
- 88-character line length (Black/Ruff)
- snake_case for functions/variables, UpperCamelCase for classes
- Fail fast rather than defaulting
- Keep imports at module top without try/except wrappers

## Testing

- Tests in `tests/`, mirroring `src/main/` structure
- Run targeted tests: `pytest -k pattern`
- Use deterministic fixtures for config parsing and data pipeline tests
- Avoid network-bound tests without explicit markers

## Common Gotchas

1. **Config Path Prefixes**: Fine-tuning configs MUST include the `finetune/` prefix when using sbatch (e.g., `finetune/mamba_ner_xho` not `mamba_ner_xho`). Evaluation configs are auto-prefixed with `eval/`.
2. **Checkpoint Paths**: Always verify checkpoint directory structure. PEFT adapters need separate base model path.
3. **Hydra Config**: Use `--config-name` (not `--config-path`) for primary config. Override with `key=value` syntax.
4. **Accelerate**: SLURM scripts automatically set `num_processes` based on `SLURM_GPUS_ON_NODE`.
5. **Environment Variables**: Foundation training scripts expect `$SCRATCH` and `$HOME` set for HPC environments.
6. **Pretokenized Data**: Foundation training expects T2X format. Use `scripts/pretokenize_dataset.py` for preprocessing.
