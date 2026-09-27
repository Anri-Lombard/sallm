# Mamba-2 T2X post-hoc HPO activation — 2026-08-30

Preregistered at 13:56 SAST before any Mamba validation result was produced.

## Scientific boundary

- This is a disclosed uniform post-hoc reproduction. It is analytically
  isolated from pure-GDN and the LLaMA GPU-0 lane.
- Use Kombuys GPU 1 only: NVIDIA GeForce RTX 3080 Ti, 12 GB. Run one Mamba
  trial at a time. GPU 0 remains LLaMA-only.
- Never open or use a held-out/test metric, prediction, or artifact for
  selection, correction, retry, checkpoint retention, prompting, ordering, or
  protocol changes.
- A failed activation gate is preserved and stops the lane. It does not
  authorize changing the candidate grid, data, metric, or held-out boundary.

## Frozen model and runtime

- Cached base commit:
  `/scratch/alombard/sallm/hf/hub/models--anrilombard--sallm-mamba-125m/snapshots/0c57d7bdcd47209894bdc5098e62658c4f05fa59`.
  - `config.json`: `fbdb689801e4aa7320f41b71c1d8409a664c405b12c69dff3e7dfb54997623db`
  - model weights: `7eaa6b8a1c24ce1a0638c17efb84cdaa44eab5ec167f76881742e2f89414abde`
  - cached tokenizer JSON: `3be3a5fda9551681d05a215392a292cff148b132fb9a5a7c693f278d37f20d13`
- Frozen experiment tokenizer:
  `/scratch/alombard/masters/sallm/tokenizer/sallm_bpe_tokenizer/tokenizer.json`,
  SHA-256 `446895905ea9b20c746317eefd0c6a3b097bcbbef71e8e44b0bf9772d664782a`.
  Its vocabulary, merges, and special-token IDs match the cached tokenizer;
  raw JSON metadata differs.
- Baseline immutable source snapshot:
  `/scratch/alombard/sallm_snapshots/uniform-adapter-hpo-20260811-6aabf717`.
  - deployment manifest: `da1c456774b789578bc6d25f816e627fa0d6677de928802a89fb5a226557b285`
  - candidate registry: `8fdd6ea5f192a1d2ef2267353162b9a640ccd09ae0d889cb5a999645198bb726`
  - Mamba T2X config: `82c73bd1c013a5722500838d0dfeb4eb96a26e651ba70f7c9c2f5f360d316b52`
  - `scripts/hpo_protocol.py`: `edf5638cae9d783b840088a16079770a907208d518602bad10e210829ed53568`
  - manifest builder: `55095f225dba0710b23831fb306ca3e471cbb1bbaaeec48529dd080fec235024`
  - finetune launcher: `828eb3194dc1c6dfd5ed9cf4174a81e9a243db5223bbc460e147492e11f5f423`
  - finetune entrypoint: `e5a0df4811c87185cfe4091255fa1f4a0311600d84e215667641fbe56be7b056`
  - training factory: `5ae98f4f95b64a0d97bb2aba779b8738ad4ef31f616ed3e160dcac36853dbd51`
  - callbacks: `6fae7808b0097d2e5aee172e781be82b3c27650a872be0e0a3e00adf0539ecd8`
- The baseline snapshot's generic wrapper hashes are
  `f92329a9f510edddf7a5cf521cd52808309511d8a67d942838dc3d7bf44b4e59`
  and `eeb88f17ff5d2cb4a2d4b16851ab0ddbc1c2c667f3cd4d34213d880dc88eaa53`.
  They do not encode the newly frozen Mamba runtime profile. Before activation,
  create one immutable prospective correction snapshot containing only the
  reviewed wrapper and metric-free preflight changes, record every changed
  file hash, and verify its deployment manifest. Never mutate the baseline
  snapshot.
- Runtime: torch `2.9.1+cu128`, transformers `4.57.3`, peft `0.18.1`,
  mamba-ssm `2.3.2.post1`, causal-conv1d `1.6.2.post1`.
  - `causal_conv1d_cuda`:
    `5c416a4b6351483f6942684e1817461b3d7a709b4df29c5216bff5d158fc98b1`
  - `selective_scan_cuda`:
    `e9489a6a560600d87f540636a46bf7f0ac86b69b19b5231eb838c5de06166de6`

## Frozen HPO contract

- Architecture-native LoRA targets are exactly `in_proj` and `out_proj`
  across all 27 mixer layers: 54 target modules. `x_proj` is forbidden because
  it does not exist in this checkpoint.
- Use the unchanged registry candidates in order: a0, a1, a2, then b0 through
  b7 at seed 42. A0 is a real candidate. Rank/alpha/dropout/LR/warmup remain
  exactly registry-controlled; alpha is twice rank.
- Data are the frozen Xhosa T2X train/validation rows: 3,859/460. Train four
  epochs with max length 1,024, BF16, train/eval batch 1, accumulation 8, and
  gradient checkpointing disabled.
- Checkpoint retention and final candidate ranking use only the frozen
  deterministic 64-row validation chrF evaluation with beam 5 and at most 64
  generated tokens.
- After all 11 terminal artifacts, coverage, hashes, and retained-to-final
  adapter roundtrips verify, rank validation-only. Freeze the top two, run only
  those candidates at seeds 13 and 87, then select and hash the Mamba recipe
  from seeds 13/42/87.

## Metric-free activation gate

1. Verify immutable hashes, absent output/no duplicate process, cached
   model/tokenizer, GPU 1 identity/isolation, target names/count, and wrapper
   dry-run output.
2. Before any validation evaluation, run the worst-case rank-32 preflight on
   one frozen training row and one frozen training prompt. It must:
   - instantiate exactly 54 targets and 6,663,168 trainable parameters;
   - complete one BF16 forward/backward with finite loss and gradients;
   - load the native selective-scan and causal-convolution CUDA extensions;
   - complete beam-5, max-64 generation, discard the text, and compute no task
     metric; and
   - report peak allocated memory below 11 GiB with no fallback or kernel
     error.

Only a structurally passing gate may release a0. No value produced by later
validation may change this frozen sequence.
