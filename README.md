# MzansiText, MzansiLM & SALLM

**An open corpus, pretrained language models, and training/evaluation framework for all eleven official South African languages.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![LREC 2026](https://img.shields.io/badge/LREC_2026-10.63317%2F4g7i3kqszquh-red.svg)](https://aclanthology.org/2026.lrec-1.109/)
[![Collection](https://img.shields.io/badge/Hugging_Face-MzansiLM_%26_MzansiText-orange.svg)](https://huggingface.co/collections/uctnlp/mzansilm-and-mzansitext-6a6b3de8f1bff1af24030440)

This repository accompanies **"MzansiText and MzansiLM: An Open Corpus and Decoder-Only Language Model for South African Languages"** (LREC 2026, [ACL Anthology](https://aclanthology.org/2026.lrec-1.109/)) and supports controlled pretraining, full fine-tuning, and downstream evaluation across five ~125M decoder architectures on the same corpus:

- **Transformer** (MzansiLM / Llama-style GQA)
- **Mamba-2** (selective state space model, via FLA or Transformers with grouped RMSNorm fix)
- **xLSTM** (mLSTM blocks with Triton chunkwise kernels)
- **Gated DeltaNet (GDN)** (pure linear recurrence via `flash-linear-attention`)
- **GDN + Attention Hybrid** (3:1 GDN-to-softmax-attention layout, 17 GDN + 5 attention layers)

---

## Quick Start

```bash
uv sync                                     # core CPU/GPU environment
uv sync --extra gpu --extra pure-gdn        # Linux: adds liger-kernel and flash-linear-attention
uv run hf download uctnlp/mzansilm-125m \
  tokenizer.json tokenizer_config.json special_tokens_map.json \
  --local-dir tokenizer/sallm_bpe_tokenizer
```

The MzansiLM BPE tokenizer (65,536 tokens) is published on the Hugging Face Hub rather than committed to git. Configs expect it at `tokenizer/sallm_bpe_tokenizer` (or `${HOME}/masters/sallm/tokenizer/sallm_bpe_tokenizer` on SLURM).

---

## Workflows

### 1. Matched-Budget Pretraining (All 5 Architectures)

Matched multi-architecture pretraining runs through `sallm.pretrain` with identical packed `uint16` token streams (`train.bin` plus `meta.json` containing `{"n_tokens": ...}`), WSD or cosine schedules, fp32 master weights with bf16 autocast, fused AdamW, Liger fused linear cross-entropy, and per-language held-out bits-per-byte scoring over a Parquet file with `text` and `lang` columns. Configs live in `src/conf/pretrain/`:

```bash
export MP_TOKENIZER=tokenizer/sallm_bpe_tokenizer
export MP_VAL_PARQUET=/path/to/validation.parquet
torchrun --nproc_per_node=2 -m sallm.pretrain \
  --config src/conf/pretrain/gdn_wsd.json \
  --data /path/to/packed_train_bin \
  --out /path/to/output_dir \
  --stack fast
```

Available matched 125M configs: `mzansilm_wsd.json`, `mamba2_wsd.json`, `xlstm_wsd.json`, `gdn_wsd.json`, and `hybrid_wsd.json`.

### 2. Full Fine-Tuning

Fine-tuning updates all weights from float32 checkpoints under bf16 autocast with fused AdamW and TF32 enabled. Use the composable `finetune/run` config to mix any architecture (`llama`, `mamba2`, `xlstm`, `gated_deltanet`, `hybrid`) with any dataset (`sib`, `injongointent`, `masakhanews`, `masakhaner`, `nchlt_ner`, `masakhapos`, `afrihg`, `t2x`, `sa_multitask`):

```bash
uv run python -m sallm.main --config-name finetune/run \
  finetune/defaults/arch@_global_=hybrid \
  finetune/defaults/datasets@_global_=t2x \
  dataset.subset=xho \
  model.init_checkpoint=/path/to/hybrid-checkpoint \
  wandb.name=hybrid-t2x-xho
```

Named recipe shortcuts are also available via the CLI:

```bash
uv run sallm recipes list
uv run sallm finetune llama_t2x_xho --dry-run
```

### 3. Evaluation (`lm-eval` Task Packs & Generation)

```bash
uv run python -m sallm.main --config-name eval/run \
  eval_model.checkpoint=uctnlp/mzansilm-125m \
  'evaluation.task_packs=[sib_xho]' \
  wandb.name=eval-mzansilm-sib-xho
```

### 4. Corpus & Tokenizer Preparation

| Stage | Command |
| --- | --- |
| Clean and deduplicate corpus | See [data/cleaning/README.md](data/cleaning/README.md), then `uv run python data/prepare_datasets.py` |
| Train a new BPE tokenizer | `uv run python tokenizer/train.py` (defaults to NFC normalization; set `normalizer: NFD` only to reproduce legacy MzansiLM) |
| Tokenize corpus | `uv run python tokenizer/process.py` |

On SLURM clusters, `ops/slurm/` wraps the training, fine-tuning, evaluation, and HPO entrypoints using environment variables from `ops/slurm/lib/env.sh` (`SALLM_HOME_DIR`, `SALLM_SCRATCH_DIR`, `SALLM_REPO_DIR`).

---

## Repository Layout

| Path | Contents |
| --- | --- |
| `src/main/sallm` | Core package: `pretrain.py` (matched DDP pretrainer), `training/` (SFT trainer & callbacks), `fine_tune/`, `evaluation/` (classification, generation, POS, and `lm-eval` runners), `models/` (registry, architecture profiles, Mamba-2/xLSTM/Llama compatibility), `data/` (adapters, loaders, multitask sampler), `configs/` (typed Hydra schema), `hpo/`, `cli.py` |
| `src/conf` | Configs: `pretrain/` (5 matched 125M WSD JSONs), `base/` (legacy Hydra pretraining configs), `finetune/` (composable defaults + recipe targets), `eval/` & `rerank/` (`lm-eval` task packs & generation evals), `templates/`, `sweeps/`, `datasets/`, `tokenizers/` (see [docs/configuration.md](docs/configuration.md)) |
| `data/`, `tokenizer/` | Corpus cleaning, deduplication, and BPE tokenizer training |
| `ops/slurm/` | SLURM launch scripts |
| `tests/` | CPU test suite run in CI |

---

## Reproducing Published Results

- **MzansiLM (LREC 2026):** check out tag `mzansitext-mzansilm-lrec2026-v1`.
- **Five-architecture comparison:** see [docs/reproducing.md](docs/reproducing.md) and the frozen `paper/architecture-comparison-2026-09` branch (`transformers 4.57.3` lockfile and as-run scripts).

`main` tracks the latest Transformers 5.x / TRL / `lm-eval` 0.4.13 stack and includes bug fixes and cleanups beyond the paper snapshots.

---

## Releases

- **Paper (LREC 2026):** [MzansiText and MzansiLM (ACL Anthology)](https://aclanthology.org/2026.lrec-1.109/) | [arXiv:2603.20732](https://arxiv.org/abs/2603.20732)
- **Model:** [uctnlp/mzansilm-125m](https://huggingface.co/uctnlp/mzansilm-125m)
- **Raw corpus:** [uctnlp/mzansi-text](https://huggingface.co/datasets/uctnlp/mzansi-text)
- **Tokenized corpus:** [uctnlp/mzansi-text-tokenized](https://huggingface.co/datasets/uctnlp/mzansi-text-tokenized)
- **Hugging Face collection:** [MzansiLM and MzansiText](https://huggingface.co/collections/uctnlp/mzansilm-and-mzansitext-6a6b3de8f1bff1af24030440)

---

## Development

```bash
uv sync --extra dev
uv run pytest -q
uv run ruff check src tokenizer tests
uv run ty check
```

---

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
