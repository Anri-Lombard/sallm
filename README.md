# MzansiText, MzansiLM & SALLM

An open corpus, pretrained language models, and training and evaluation codebase for South Africa's eleven official written languages.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![LREC 2026](https://img.shields.io/badge/LREC_2026-10.63317%2F4g7i3kqszquh-red.svg)](https://aclanthology.org/2026.lrec-1.109/)
[![Hugging Face Collection](https://img.shields.io/badge/Hugging_Face-MzansiLM_%26_MzansiText-orange.svg)](https://huggingface.co/collections/uctnlp/mzansilm-and-mzansitext-6a6b3de8f1bff1af24030440)

This repository contains the code behind **MzansiText and MzansiLM: An Open Corpus and Decoder-Only Language Model for South African Languages** (LREC 2026), together with the **SALLM** framework we use to pretrain, fine-tune, and evaluate five matched ~125M decoder architectures on the same corpus.

The corpus and models cover all eleven official written South African languages: Afrikaans (`afr`), English (`eng`), isiNdebele (`nbl`), Sepedi (`nso`), Sesotho (`sot`), siSwati (`ssw`), Setswana (`tsn`), Xitsonga (`tso`), Tshivenda (`ven`), isiXhosa (`xho`), and isiZulu (`zul`).

## Published releases

If you only want the data or pretrained weights, you do not need to clone this repository. Everything is hosted on the ACL Anthology and Hugging Face:

| Resource | Link |
| --- | --- |
| LREC 2026 paper | [ACL Anthology](https://aclanthology.org/2026.lrec-1.109/) / [arXiv:2603.20732](https://arxiv.org/abs/2603.20732) |
| Pretrained model (125M) | [uctnlp/mzansilm-125m](https://huggingface.co/uctnlp/mzansilm-125m) |
| Raw corpus (3.8B tokens) | [uctnlp/mzansi-text](https://huggingface.co/datasets/uctnlp/mzansi-text) |
| Document-tokenized corpus | [uctnlp/mzansi-text-tokenized](https://huggingface.co/datasets/uctnlp/mzansi-text-tokenized) |
| Hugging Face collection | [MzansiLM and MzansiText](https://huggingface.co/collections/uctnlp/mzansilm-and-mzansitext-6a6b3de8f1bff1af24030440) |

## Quick try: load MzansiLM

MzansiLM is a base language model (not instruction-tuned), so it generates plain text continuations out of the box and is meant to be fine-tuned for downstream tasks.

```bash
pip install "transformers>=4.52.4,<5" torch accelerate safetensors
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "uctnlp/mzansilm-125m"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

prompt = "Namuhla sifunda ngolimi ngoba abantu"
inputs = tokenizer(prompt, return_tensors="pt")
# The tokenizer appends [EOS] automatically; drop it before continuing a prompt.
if inputs["input_ids"][0, -1].item() == tokenizer.eos_token_id:
    inputs = {k: v[:, :-1] for k, v in inputs.items()}

outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

On Transformers 5.x, call `sallm.models.llama_compatibility.register_llama_compatibility()` before loading `uctnlp/mzansilm-125m`, because its config sets `head_dim=56` with `hidden_size=512` and `num_attention_heads=9`.

## Supported architectures

SALLM supports controlled pretraining, full fine-tuning, and evaluation across five decoder architectures sized to roughly 125M parameters on the same 65,536-token vocabulary:

| Architecture | Description | Layers | Hidden size | Parameters | Pretrain config |
| --- | --- | ---: | ---: | ---: | --- |
| Transformer (MzansiLM) | Llama-style GQA (9 Q / 3 KV heads) | 30 | 512 | 125.0M | `src/conf/pretrain/mzansilm_wsd.json` |
| Mamba-2 | Selective state space model (FLA or Transformers with grouped RMSNorm fix) | 27 | 512 | 126.4M | `src/conf/pretrain/mamba2_wsd.json` |
| xLSTM | mLSTM blocks with Triton chunkwise kernels | 12 | 736 | 126.9M | `src/conf/pretrain/xlstm_wsd.json` |
| Gated DeltaNet (GDN) | Linear recurrence via `flash-linear-attention` | 21 | 512 | 127.4M | `src/conf/pretrain/gdn_wsd.json` |
| GDN + Attention Hybrid | 3:1 layout (17 GDN + 5 softmax attention layers) | 22 | 512 | 126.6M | `src/conf/pretrain/hybrid_wsd.json` |

## Setup

Install dependencies with [uv](https://docs.astral.sh/uv/) (Python 3.12 or 3.13) and download the MzansiLM tokenizer from Hugging Face:

```bash
uv sync
uv sync --extra gpu --extra pure-gdn        # Linux GPU machines: adds liger-kernel and flash-linear-attention

uv run hf download uctnlp/mzansilm-125m \
  tokenizer.json tokenizer_config.json special_tokens_map.json \
  --local-dir tokenizer/sallm_bpe_tokenizer
```

The tokenizer files are hosted on Hugging Face rather than stored in git. Local configs look for them at `tokenizer/sallm_bpe_tokenizer` (or `${HOME}/masters/sallm/tokenizer/sallm_bpe_tokenizer` on SLURM).

Tests and small evaluations run on CPU (macOS or Linux). Pretraining, full fine-tuning, and the fast kernel paths (Liger, Gated DeltaNet, Hybrid, Mamba-2 FLA, and xLSTM Triton) require a Linux machine with a CUDA GPU.

## Running experiments

### Pretraining

`sallm.pretrain` runs distributed pretraining across all five architectures under the same setup: fp32 master weights with bf16 autocast, fused AdamW, optional Liger fused linear cross-entropy (`--stack fast`), WSD or cosine learning-rate schedules, and periodic per-language bits-per-byte evaluation.

It reads a directory containing a packed `uint16` token stream (`train.bin`) alongside a `meta.json` file with `{"n_tokens": ...}`, plus a validation Parquet file with `text` and `lang` columns:

```bash
export MP_TOKENIZER=tokenizer/sallm_bpe_tokenizer
export MP_VAL_PARQUET=/path/to/validation.parquet

torchrun --nproc_per_node=2 -m sallm.pretrain \
  --config src/conf/pretrain/gdn_wsd.json \
  --data /path/to/packed_train_bin \
  --out /path/to/output_dir \
  --stack fast
```

Swap `--config` for any of the five JSON files in `src/conf/pretrain/`.

### Fine-tuning

Fine-tuning updates all model weights from float32 checkpoints under bf16 autocast. Rather than maintaining one YAML per `(model, task, language)` combination, `finetune/run` composes any architecture (`llama`, `mamba2`, `xlstm`, `gated_deltanet`, `hybrid`) with any task config:

- Topic and intent classification: `sib` (SIB-200), `injongointent`, `masakhanews`
- Sequence tagging: `masakhaner`, `nchlt_ner`, `masakhapos`
- Generation: `afrihg` (headline generation), `t2x` (triples-to-isiXhosa verbalisation)
- Multitask mixture: `sa_multitask` (capped FLAN-style mix across all seven task families)

```bash
uv run python -m sallm.main --config-name finetune/run \
  finetune/defaults/arch@_global_=hybrid \
  finetune/defaults/datasets@_global_=t2x \
  dataset.subset=xho \
  model.init_checkpoint=/path/to/hybrid-checkpoint \
  wandb.name=hybrid-t2x-xho
```

See [docs/configuration.md](docs/configuration.md) for all Hydra overrides, early-stopping options, W&B sweeps, and cache environment variables.

### Evaluation

`eval/run` evaluates any Hugging Face model or local checkpoint on one or more `lm-eval` task packs:

```bash
uv run python -m sallm.main --config-name eval/run \
  eval_model.checkpoint=uctnlp/mzansilm-125m \
  'evaluation.task_packs=[sib_xho]' \
  wandb.name=eval-mzansilm-sib-xho
```

For generative benchmarks, use `eval/generate_t2x`, `eval/generate_t2x_base`, `eval/generate_afrihg`, or `eval/generate_afrihg_base` (see [docs/configuration.md](docs/configuration.md)).

### Corpus and tokenizer preparation

If you want to rebuild the corpus or train a new tokenizer from scratch:

1. Clean and filter raw documents with the Datatrove pipeline in [data/cleaning/README.md](data/cleaning/README.md), then build the train/validation/test splits with `uv run python data/prepare_datasets.py`.
2. Train a BPE tokenizer with `uv run python tokenizer/train.py`. It defaults to NFC normalization; set `normalizer: NFD` only if you need to reproduce the original MzansiLM tokenizer.
3. Tokenize the corpus with `uv run python tokenizer/process.py`.

On SLURM clusters, the launch scripts under `ops/slurm/` wrap pretraining, fine-tuning, evaluation, and W&B sweeps using cluster paths configured in `ops/slurm/lib/env.sh` (`SALLM_HOME_DIR`, `SALLM_SCRATCH_DIR`, `SALLM_REPO_DIR`).

## Repository layout

```text
sallm/
├── src/main/sallm/       # Core library (pretrain.py, training/, fine_tune/, evaluation/, models/, data/, configs/, hpo/)
├── src/conf/             # Pretraining JSONs (pretrain/) and Hydra configs (finetune/, eval/, sweeps/, templates/)
├── data/                 # Datatrove cleaning pipeline and dataset split preparation
├── tokenizer/            # BPE tokenizer training and corpus tokenization scripts
├── ops/slurm/            # SLURM job launchers and cluster environment helpers
├── docs/                 # Configuration reference and paper reproduction guide
└── tests/                # CPU test suite run in CI
```

## Reproducing the papers

`main` tracks the current Transformers 5.x, TRL, and `lm-eval` 0.4.13 stack and includes bug fixes added after the paper runs. To reproduce published numbers exactly, use the frozen snapshots:

- MzansiLM and MzansiText (LREC 2026): check out tag `mzansitext-mzansilm-lrec2026-v1`.
- Five-architecture comparison: follow [docs/reproducing.md](docs/reproducing.md) on branch `paper/architecture-comparison-2026-09` (pinned to `transformers 4.57.3` with the as-run scripts and lockfile).

## Development

```bash
uv sync --extra dev
uv run pytest -q
uv run ruff check src tokenizer tests
uv run ty check
```

## Citation

If you use MzansiText, MzansiLM, or SALLM in your work, please cite the LREC 2026 paper:

```bibtex
@inproceedings{lombard-etal-2026-mzansitext,
    title = "{M}zansi{T}ext and {M}zansi{LM}: An Open Corpus and Decoder-Only Language Model for {S}outh {A}frican Languages",
    author = "Lombard, Anri M.  and
      Mawere, Simbarashe  and
      Aina, Temi  and
      Wolff, Ethan  and
      Gumede, Sbonelo  and
      Novick, Elan  and
      Meyer, Francois  and
      Buys, Jan",
    booktitle = "Proceedings of the Fifteenth Language Resources and Evaluation Conference",
    year = "2026",
    publisher = "ELRA Language Resource Association",
    url = "https://aclanthology.org/2026.lrec-1.109/",
    doi = "10.63317/4g7i3kqszquh",
    pages = "1394--1408"
}
```

## License

Released under the Apache License 2.0. See [LICENSE](LICENSE).
