# MzansiText & MzansiLM

**An open corpus and decoder-only language model for South African languages**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Paper](https://img.shields.io/badge/Paper-arXiv_2603.20732-red.svg)](https://arxiv.org/abs/2603.20732)

This repository accompanies the paper **"MzansiText and MzansiLM: An Open Corpus and Decoder-Only Language Model for South African Languages"** ([arXiv:2603.20732](https://arxiv.org/abs/2603.20732)).

**For exact reproduction of the LREC 2026 paper results, use the permanent snapshot:**

```bash
git clone https://github.com/Anri-Lombard/sallm.git
cd sallm
git checkout tags/mzansitext-mzansilm-lrec2026-v1
```

The `main` branch is actively maintained and will continue to evolve beyond the paper snapshot.

The repository contains code and configuration for:

- MzansiText data preparation
- tokenizer training
- decoder-only Llama pretraining
- downstream fine-tuning
- downstream evaluation

## Releases

Project artifacts are released on Hugging Face:

- Paper: [arXiv:2603.20732](https://arxiv.org/abs/2603.20732)
- Model: [anrilombard/mzansilm-125m](https://huggingface.co/anrilombard/mzansilm-125m)
- Raw corpus: [anrilombard/mzansi-text](https://huggingface.co/datasets/anrilombard/mzansi-text)
- Tokenized corpus: [anrilombard/mzansi-text-tokenized](https://huggingface.co/datasets/anrilombard/mzansi-text-tokenized)
- Collection: [MzansiLM](https://huggingface.co/collections/anrilombard/mzansilm-69635ca7b60efedb9dfcb09e)

## Model Details

- Parameters: `125,008,384`
- Architecture: decoder-only `LlamaForCausalLM`
- Hidden size: `512`
- Intermediate size: `1536`
- Layers: `30`
- Attention heads: `9`
- Key/value heads: `3`
- Context length: `2048`
- RoPE theta: `10000.0`
- RMSNorm epsilon: `1e-5`
- Tied word embeddings: `true`
- Attention implementation used for training: `flash_attention_2`

## Key Results

- T2X (isiXhosa data-to-text): `20.65` BLEU with monolingual task-specific fine-tuning
- MasakhaNEWS (isiXhosa): `78.5%` macro-F1 with multilingual task-specific fine-tuning
- On selected generation and NER tasks, MzansiLM outperforms much larger decoder-only baselines

## Tokenizer

MzansiLM uses a custom BPE tokenizer trained with the tooling in `tokenizer/train.py`.

- Vocabulary size: `65536`
- Special tokens:
  - `[BOS] = 0`
  - `[EOS] = 1`
  - `[PAD] = 2`
  - `[UNK] = 3`
- Normalizer: `NFD`
- Pre-tokenizer: `ByteLevel`
- Post-processing:
  - single sequence: `[BOS] $A [EOS]`
  - pair sequence: `[BOS] $A [EOS] [BOS] $B [EOS]`

## Data

The dataset pipeline prepares multilingual text covering all eleven official South African languages:

- `af`
- `en`
- `nso`
- `sot`
- `ssw`
- `tsn`
- `tso`
- `ven`
- `xho`
- `zul`
- `nbl`

The raw and tokenized dataset releases are available on Hugging Face:

- [anrilombard/mzansi-text](https://huggingface.co/datasets/anrilombard/mzansi-text)
- [anrilombard/mzansi-text-tokenized](https://huggingface.co/datasets/anrilombard/mzansi-text-tokenized)

### Token Distribution

Token counts below are from the paper release after filtering with the MzansiText cleaning pipeline and tokenization with the `65536`-vocabulary BPE tokenizer.

| Language | Train Tokens | % | Val Tokens | Test Tokens |
| --- | ---: | ---: | ---: | ---: |
| Afrikaans | 2,475,913,822 | 64.96 | 1,865,255 | 1,875,605 |
| English | 740,994,679 | 19.44 | 1,813,651 | 1,821,803 |
| isiZulu | 320,224,015 | 8.40 | 2,017,406 | 2,021,343 |
| isiXhosa | 152,212,403 | 3.99 | 2,016,503 | 2,012,000 |
| Sesotho | 97,558,939 | 2.56 | 2,315,298 | 2,316,170 |
| Setswana | 10,082,930 | 0.26 | 1,216,539 | 1,413,473 |
| Sepedi | 6,697,358 | 0.18 | 685,425 | 778,656 |
| Xitsonga | 3,013,408 | 0.08 | 510,463 | 319,496 |
| siSwati | 1,932,989 | 0.05 | 196,247 | 225,810 |
| Tshivenda | 1,852,481 | 0.05 | 191,495 | 243,315 |
| isiNdebele | 818,549 | 0.02 | 106,224 | 143,458 |
| **Total** | **3,811,301,573** | **100** | **12,934,506** | **13,171,129** |

Validation and test sets are capped at approximately `2M` tokens per language.

The exact cleaning pipeline used to produce the filtered corpus is included in
`data/cleaning/`.

## Repository Layout

- `src/main/sallm`: Python package for pretraining, fine-tuning, and evaluation
- `src/conf`: Hydra configs for training, fine-tuning, evaluation, HPO, and tokenization
- `scripts`: cluster launch and orchestration scripts
- `tokenizer`: tokenizer training and processing utilities
- `data`: dataset preparation utilities

## Installation

This repository uses `uv` for environment management.

```bash
uv sync
source .venv/bin/activate
```

For local quality checks, use the same pre-commit entrypoint as CI:

```bash
uv sync --extra dev
uv run pre-commit install
uv run pre-commit run --all-files
```

Some workflows expect external credentials to be available through environment variables or standard local auth files, especially for Hugging Face and Weights & Biases.

Cluster helper scripts under `scripts/` are advanced workflows. Pass SLURM
account/mail settings through `sbatch` flags for your environment, and override
runtime paths with `SALLM_SCRATCH_DIR`, `SALLM_HOME_DIR`, `SALLM_REPO_DIR`, and
`SALLM_SLURM_USER` when the defaults do not match your cluster.
Wrapper scripts that submit nested jobs also respect `SALLM_SLURM_ACCOUNT` and
`SALLM_SLURM_MAIL_USER`.

## Reproducing Runs

All workflows route through Hydra:

```bash
python -m sallm.main --config-name base/llama_125m
python -m sallm.main --config-name finetune/llama_t2x_xho
python -m sallm.main --config-name eval/run_llama_t2x_xho
```

Useful helper scripts include:

- `scripts/train_final_model.sh`
- `scripts/launch_finetune.sh`
- `scripts/launch_evaluation.sh`
- `scripts/launch_hpo.sh`
- `scripts/submit_llama_finetune_eval.sh`

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
