# MzansiText & MzansiLM

**An open corpus and decoder-only language model for South African languages.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Paper](https://img.shields.io/badge/Paper-arXiv_2603.20732-red.svg)](https://arxiv.org/abs/2603.20732)

This repository accompanies **"MzansiText and MzansiLM: An Open Corpus and
Decoder-Only Language Model for South African Languages"**
([arXiv:2603.20732](https://arxiv.org/abs/2603.20732)).

The public workflow is recipe-first: choose a recipe ID, inspect the Hydra
configs it resolves to, then launch fine-tuning or evaluation.

```bash
uv sync
uv run sallm recipes list
uv run sallm recipe show llama_t2x_xho
uv run sallm finetune llama_t2x_xho --dry-run
uv run sallm evaluate llama_t2x_xho --dry-run
```

Recipe IDs are defined in `recipes/registry.yaml`. Each recipe points to known
Hydra configs under `src/conf`; those YAML files remain the source of truth for
hyperparameters.

## Docs

- [Quickstart](docs/quickstart.md): install, inspect recipes, and dry-run the CLI.
- [Recipes](docs/recipes.md): supported recipe IDs and their config targets.
- [Configuration](docs/configuration.md): how recipe targets map to Hydra YAML.
- [SLURM](docs/slurm.md): advanced cluster script compatibility notes.

## Paper Snapshot

For exact reproduction of the LREC 2026 paper results, use the permanent
snapshot:

```bash
git clone https://github.com/Anri-Lombard/sallm.git
cd sallm
git checkout tags/mzansitext-mzansilm-lrec2026-v1
```

The `main` branch is actively maintained and may differ from the paper snapshot.

## Releases

- Paper: [arXiv:2603.20732](https://arxiv.org/abs/2603.20732)
- Model: [uctnlp/mzansilm-125m](https://huggingface.co/uctnlp/mzansilm-125m)
- Raw corpus: [uctnlp/mzansi-text](https://huggingface.co/datasets/uctnlp/mzansi-text)
- Tokenized corpus: [uctnlp/mzansi-text-tokenized](https://huggingface.co/datasets/uctnlp/mzansi-text-tokenized)
- Collection: [MzansiLM](https://huggingface.co/collections/anrilombard/mzansilm-69635ca7b60efedb9dfcb09e)

Rebuild the raw release from the complete filtered source tree with:

```bash
uv run python data/prepare_datasets.py
```

The release guard requires all 42 source files and the exact paper splits:
3,943,584 train, 19,379 validation, and 19,341 test rows. Output columns are
`text` and `lang`.

## Running the Pipeline

Every run goes through one Hydra entrypoint, `python -m sallm.main
--config-name <target>`. The `mode` field in the config (`TRAIN`, `FINETUNE`
or `EVALUATE`) selects the stage. Config paths such as `${oc.env:SCRATCH}`
resolve from the environment, so set `SCRATCH` and `HOME` or override the
paths on the command line.

**1. Corpus and tokenizer.** Clean the raw sources as described in
[data/cleaning/README.md](data/cleaning/README.md), then:

```bash
uv run python data/prepare_datasets.py          # src/conf/datasets/sallm_dataset.yaml
uv run python tokenizer/train.py                # src/conf/tokenizers/bpe.yaml
uv run python tokenizer/process.py              # src/conf/datasets/sallm_processed.yaml
```

The Llama base configs read the processed corpus from disk. The Mamba, xLSTM,
RWKV and RecurrentGemma base configs load the tokenized corpus from the Hub
instead.

**2. Pretraining.** Base model configs live in `src/conf/base/`:

```bash
uv run python -m sallm.main --config-name base/llama_125m
```

**3. Fine-tuning and evaluation.** Use a recipe, or a config target directly:

```bash
uv run sallm finetune llama_t2x_xho --dry-run
uv run python -m sallm.main --config-name finetune/llama_t2x_xho
uv run python -m sallm.main --config-name eval/run_llama_t2x_xho
```

**4. HPO.** Sweep configs live in `src/conf/sweeps/`. On a SLURM cluster,
`ops/slurm/` wraps fine-tuning, evaluation and HPO; see [SLURM](docs/slurm.md).

## Repository Layout

| Path | Contents |
| --- | --- |
| `src/main/sallm` | Python package: `training/` (pretraining), `fine_tune/`, `evaluation/`, `hpo/`, `data/` (dataset adapters, formatters, loaders), `models/`, `configs/` (typed config schema), `cli.py` (recipe CLI) and `main.py` (Hydra entrypoint) |
| `src/conf` | Hydra configs: `base/` (pretraining), `finetune/`, `eval/` (evaluation runs and lm-eval task packs), `rerank/`, `sweeps/` (HPO), `templates/` (prompt templates), `datasets/`, `tokenizers/` |
| `recipes/registry.yaml` | Public recipe IDs mapped to finetune and eval configs |
| `data/` | Corpus cleaning and release preparation |
| `tokenizer/` | Tokenizer training and corpus tokenization; `tokenizer/tokenizer/` holds a committed tokenizer |
| `ops/slurm/` | SLURM launchers for fine-tuning, evaluation and HPO |
| `scripts/` | Small utilities (corpus pre-tokenization, Grype policy check) |
| `tests/` | CPU pytest suite, run in CI |
| `archive/` | Superseded environments, one-off scripts and dated reports, kept for provenance |

## Development

```bash
uv sync --extra dev
uv run pytest -q
uv run pre-commit run --all-files
```

## Citation

Please cite the paper:

```bibtex
@misc{lombard2026mzansitextmzansilmopencorpus,
      title={MzansiText and MzansiLM: An Open Corpus and Decoder-Only Language Model for South African Languages},
      author={Anri Lombard and Simbarashe Mawere and Temi Aina and Ethan Wolff and Sbonelo Gumede and Elan Novick and Francois Meyer and Jan Buys},
      year={2026},
      eprint={2603.20732},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2603.20732},
}
```

## License

This repository is released under the Apache License 2.0. See `LICENSE`.
