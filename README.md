# MzansiText & MzansiLM

**An open corpus and decoder-only language model for South African languages**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Paper](https://img.shields.io/badge/Paper-LREC_2026-red.svg)](https://github.com/Anri-Lombard/sallm#citation)

This repository accompanies the paper **"MzansiText and MzansiLM: An Open Corpus and Decoder-Only Language Model for South African Languages"**.

The repository contains code and configuration for:

- MzansiText data preparation
- tokenizer training
- decoder-only Llama pretraining
- downstream fine-tuning
- downstream evaluation

## Releases

Project artifacts are released on Hugging Face:

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

Some workflows expect external credentials to be available through environment variables or standard local auth files, especially for Hugging Face and Weights & Biases.

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

Until the public paper record is live, please cite the repository:

```bibtex
@misc{mzansitext-mzansilm-github,
  title = {MzansiText and MzansiLM: An Open Corpus and Decoder-Only Language Model for South African Languages},
  author = {Lombard, Anri and Mawere, Simbarashe and Aina, Temi and Wolff, Ethan and Gumede, Sbonelo and Novick, Elan and Meyer, Francois and Buys, Jan},
  year = {2026},
  howpublished = {\url{https://github.com/Anri-Lombard/sallm}},
  note = {GitHub repository}
}
```

This citation should be updated once the final public paper record is available.

## License

This repository is released under the Apache License 2.0. See `LICENSE`.
