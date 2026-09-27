# MzansiText & MzansiLM

**An open corpus and decoder-only language model for South African languages.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Paper](https://img.shields.io/badge/Paper-arXiv_2603.20732-red.svg)](https://arxiv.org/abs/2603.20732)

This repository accompanies **"MzansiText and MzansiLM: An Open Corpus and
Decoder-Only Language Model for South African Languages"**
([arXiv:2603.20732](https://arxiv.org/abs/2603.20732)).

It also holds the code for the architecture comparison (Transformer, Mamba-2,
xLSTM and Gated DeltaNet) built on the same corpus.

## Setup

```bash
uv sync                      # add --extra pure-gdn on Linux for Gated DeltaNet
uv run hf download uctnlp/mzansilm-125m tokenizer.json tokenizer_config.json \
  special_tokens_map.json --local-dir tokenizer/sallm_bpe_tokenizer
```

The tokenizer is published with MzansiLM rather than committed. Configs that
set `tokenizer.path` expect it at `tokenizer/sallm_bpe_tokenizer` (under
`$HOME/masters/sallm` on the cluster).

## Running

Every run goes through one Hydra entrypoint:
`python -m sallm.main --config-name <target>`, where the target is a path under
`src/conf` without `.yaml`. The config's `mode` (`TRAIN`, `FINETUNE` or
`EVALUATE`) picks the stage. Paths such as `${oc.env:SCRATCH}` resolve from the
environment; override any value on the command line.

| Stage | Command |
| --- | --- |
| Build the corpus release | `uv run python data/prepare_datasets.py` (after [data/cleaning](data/cleaning/README.md)) |
| Train the tokenizer | `uv run python tokenizer/train.py` |
| Tokenize the corpus | `uv run python tokenizer/process.py` |
| Pretrain | `uv run python -m sallm.main --config-name base/llama_125m` |
| Fine-tune | `uv run python -m sallm.main --config-name finetune/llama_t2x_xho` |
| Evaluate | `uv run python -m sallm.main --config-name eval/run eval_model.checkpoint=<model> 'evaluation.task_packs=[sib_xho]' wandb.name=<run>` |

Base configs exist for `llama_125m`, `llama_400m`, `mamba_125m`, `xlstm_125m`
and `gated_deltanet_125m`. The Mamba, xLSTM and Gated DeltaNet ones read the
released tokenized corpus from the Hub.

Recipes in `recipes/registry.yaml` name common fine-tune and evaluate pairs:

```bash
uv run sallm recipes list
uv run sallm finetune llama_t2x_xho --dry-run
```

On SLURM, `ops/slurm/` wraps the same entrypoint. Cluster paths come from
`ops/slurm/lib/env.sh` (`SALLM_HOME_DIR`, `SALLM_SCRATCH_DIR`, `SALLM_REPO_DIR`).

```bash
sbatch ops/slurm/launch_pretrain.sh base/llama_125m
sbatch ops/slurm/launch_finetune.sh finetune/llama_t2x_xho
sbatch ops/slurm/launch_evaluation.sh eval/run_llama_t2x_xho
sbatch ops/slurm/launch_hpo.sh llama_t2x_xho 10      # sweep in src/conf/sweeps
```

## Layout

| Path | Contents |
| --- | --- |
| `src/main/sallm` | Library: `training/` (pretraining), `fine_tune/`, `evaluation/`, `hpo/`, `data/`, `models/`, `configs/` (typed schema), `main.py` (Hydra entrypoint), `cli.py` (recipe CLI) |
| `src/conf` | Hydra configs: `base/`, `finetune/`, `eval/`, `rerank/`, `sweeps/`, `templates/`, `datasets/`, `tokenizers/`; see [docs/configuration.md](docs/configuration.md) |
| `data/`, `tokenizer/` | Corpus preparation and tokenizer training |
| `ops/slurm/` | SLURM launchers |
| `tests/` | CPU test suite, run in CI |

## Reproducing the Papers

- MzansiLM (LREC 2026): tag `mzansitext-mzansilm-lrec2026-v1`.
- Architecture comparison: branch `paper/architecture-comparison-2026-09`, which
  keeps the as-run runners (`scripts/paper/`), their environment lock and a
  README mapping each result to its runner. It will be tagged once the runs
  finish.

`main` is maintained code and may differ from both snapshots.

## Releases

- Paper: [arXiv:2603.20732](https://arxiv.org/abs/2603.20732)
- Model: [uctnlp/mzansilm-125m](https://huggingface.co/uctnlp/mzansilm-125m)
- Raw corpus: [uctnlp/mzansi-text](https://huggingface.co/datasets/uctnlp/mzansi-text)
- Tokenized corpus: [uctnlp/mzansi-text-tokenized](https://huggingface.co/datasets/uctnlp/mzansi-text-tokenized)
- Collection: [MzansiLM](https://huggingface.co/collections/anrilombard/mzansilm-69635ca7b60efedb9dfcb09e)

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
