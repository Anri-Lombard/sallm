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
- Model: [anrilombard/mzansilm-125m](https://huggingface.co/anrilombard/mzansilm-125m)
- Raw corpus: [anrilombard/mzansi-text](https://huggingface.co/datasets/anrilombard/mzansi-text)
- Tokenized corpus: [anrilombard/mzansi-text-tokenized](https://huggingface.co/datasets/anrilombard/mzansi-text-tokenized)
- Collection: [MzansiLM](https://huggingface.co/collections/anrilombard/mzansilm-69635ca7b60efedb9dfcb09e)

## Repository Layout

- `src/main/sallm`: Python package for training, fine-tuning, and evaluation.
- `src/conf`: Hydra configs for model, data, fine-tuning, evaluation, and HPO.
- `recipes/registry.yaml`: public recipe IDs mapped to known configs.
- `ops/slurm`: advanced SLURM workflows for cluster execution.
- `scripts`: local Python utilities.
- `tokenizer`: tokenizer training and processing utilities.
- `data`: dataset preparation and cleaning utilities.

## Development

```bash
uv sync --extra dev
uv run pytest tests/test_cli.py tests/test_recipes.py
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
