# xLSTM Architecture Literature Notes

Created: 2026-05-24.

Purpose: citation and implementation trail for the SALLM xLSTM follow-up after
the Mamba root-cause/rescue branch.

## Beck et al. (2024) - xLSTM: Extended Long Short-Term Memory

- NeurIPS 2024 paper: https://papers.neurips.cc/paper_files/paper/2024/file/c2ce2f2701c10a2b2f2ea0bfa43cfaa3-Paper-Conference.pdf
- arXiv: https://arxiv.org/abs/2405.04517
- Key architecture points:
  - xLSTM extends LSTM with exponential gating plus matrix-memory variants.
  - The best language-model variant in the paper is the mixed
    `xLSTM[7:1]` family, combining mLSTM and sLSTM blocks.
  - For 125M-sized scaling, the table reports embedding dim `768`,
    `24` blocks, `4` heads / head dim `384`, about `163.8M` parameters, peak
    LR `3e-3` for 15B-token scaling and `1.5e-3` for 300B-token scaling.
  - The authors train with context length `2048`, batch size `256`, AdamW
    betas `(0.9, 0.95)`, epsilon `1e-5`, gradient clipping `1.0`, linear
    warmup `750` steps, cosine decay to 10% of peak LR, and weight decay
    `0.1` excluding embeddings, biases, and LayerNorm weights.
  - For 125M xLSTM, the paper notes sLSTM positions `[3, 20]`, effectively an
    `11:1` mLSTM:sLSTM ratio rather than the larger-model `7:1` ratio.
  - The paper explicitly says no positional encoding is used for xLSTM models.
- SALLM implication:
  - Our local `src/conf/base/xlstm_125m.yaml` is not paper-faithful: it uses
    12 layers, max sequence length `512`, LR `4e-4`, weight decay `0.01`, and
    tied embeddings. It may still be a practical 125M-ish baseline, but should
    be labelled as SALLM-HF xLSTM rather than the paper's 125M recipe.

## NX-AI/xlstm official repository

- Repository: https://github.com/NX-AI/xlstm
- The official repo documents two relevant code paths:
  - the NeurIPS xLSTM architecture through `xLSTMBlockStack` /
    `xLSTMLMModel`;
  - the newer `xLSTM Large` implementation for the 7B model.
- The repo's language-model config example uses:
  - mLSTM with `conv1d_kernel_size=4`, `qkv_proj_blocksize=4`, `num_heads=4`;
  - sLSTM with CUDA backend, `conv1d_kernel_size=4`,
    `bias_init=powerlaw_blockdependent`;
  - feed-forward `proj_factor=1.3`, `act_fn=gelu`;
  - explicit `slstm_at` positions.
- SALLM implication:
  - We should run a small HF-vs-official xLSTM viability/parity check before
    spending a full compute block, because the official code exposes block
    placement and kernel settings that are not obvious from the HF config.

## Hugging Face Transformers xLSTM implementation

- Docs: https://huggingface.co/docs/transformers/model_doc/xlstm
- The HF docs expose `xLSTMConfig`, `xLSTMModel`, and `xLSTMForCausalLM`.
- Local Transformers version `4.57.3` exposes config knobs including
  `hidden_size`, `embedding_dim`, `num_hidden_layers` / `num_blocks`,
  `num_heads`, `chunkwise_kernel`, `sequence_kernel`, `step_kernel`, `mode`,
  `chunk_size`, `ffn_proj_factor`, `gate_soft_cap`, `output_logit_soft_cap`,
  and `tie_word_embeddings`.
- Local quick parameter counts with SALLM vocab `65536` and tied embeddings:
  - hidden `768`, 12 layers: `135.37M` params;
  - hidden `736`, 12 layers: `126.90M`;
  - hidden `704`, 12 layers: `118.68M`;
  - hidden `672`, 16 layers: `130.86M`;
  - hidden `640`, 18 layers: `131.28M`.
- SALLM implication:
  - If the user wants stricter 125M comparability, `hidden_size=736`,
    `num_hidden_layers=12`, `num_heads=4`, tied embeddings is the closest local
    HF xLSTM shape found so far.
  - If paper-faithfulness matters more than exact 125M, a 24-block official
    style recipe is closer, but it exceeds the strict 125M target under our
    vocabulary/settings.

## Beck et al. (2025) - xLSTM 7B

- arXiv: https://arxiv.org/abs/2503.13427
- The 7B paper reports xLSTM as a recurrent LLM with linear compute scaling in
  sequence length and constant memory usage, targeting fast and efficient
  inference.
- It introduces/uses the newer `xLSTM Large` implementation rather than only
  the NeurIPS architecture code.
- SALLM implication:
  - Useful for motivation and future-work framing, but not a direct 125M
    hyperparameter recipe.
