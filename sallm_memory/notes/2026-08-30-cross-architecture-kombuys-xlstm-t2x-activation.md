# Cross-architecture Kombuys xLSTM T2X activation — 2026-08-30

Status: prospective structural gate only. Recorded before any new xLSTM
validation result. This is a separately labelled post-hoc reproduction and
cannot influence pure-GDN.

## Scientific boundary

- Historical xLSTM held-out results already exist. They were not opened or
  used to choose this task, checkpoint, target modules, grid, order, retry
  rule, or metric.
- No official held-out evaluation is authorized. Selection uses validation
  chrF only, and results must be reported as post-hoc reproduction evidence.
- The failed Mamba activation is unrelated and remains terminal. This xLSTM
  lane is not a Mamba retry.

## Frozen base and runtime

- Base repository: `anrilombard/sallm-xlstm-125m-native-3epoch-20260531`.
- Immutable revision: `ba2ff845335c8cbf750f8f6f3ebc09468008fef9`.
- Required file SHA-256 values:
  - `config.json`: `8f837e6d7efa905d958cf11308187fdc07aec7a218caa5390a5b1fe5089663c9`
  - `generation_config.json`: `8abc09d0606da28291b9ab4be66db068d3747ddea747d28f7560930928a3eba5`
  - `pytorch_model.bin`: `4e3e37db9a43089e1683a56c4940bc05e9e446ae7da82b5dc5d8621beda09674`
  - `tokenizer.json`: `446895905ea9b20c746317eefd0c6a3b097bcbbef71e8e44b0bf9772d664782a`
  - `tokenizer_config.json`: `169bedadc3f18d3d1bad46dd10450d811f7a14dc50bb752f0c30dc899b5f0b3a`
  - `special_tokens_map.json`: `72a8eb0b88e02619b0c3b7da0f6b1fab8ee29283f6f18cc362b9d63749c1d628`
- The config is xLSTM hidden size 736, 12 blocks, four heads, vocabulary
  65,536, chunk size 64, and Transformers 4.57.3.
- Required runtime: Python 3.12, PyTorch 2.9.1+cu128, Transformers 4.57.3,
  PEFT 0.18.1, xLSTM 2.0.5, and mlstm-kernels 2.0.2. Install the two xLSTM
  packages in an isolated overlay; do not mutate the active LLaMA runtime.
- Source snapshot and candidate registry remain the immutable 11 August
  uniform-HPO snapshot and registry. Their deployment-manifest and registry
  SHA-256 values are
  `da1c456774b789578bc6d25f816e627fa0d6677de928802a89fb5a226557b285`
  and `8fdd6ea5f192a1d2ef2267353162b9a640ccd09ae0d889cb5a999645198bb726`.

## Frozen adapter and compute protocol

- Host/GPU: Kombuys GPU 1 only, NVIDIA GeForce RTX 3080 Ti 12 GB. Keep all
  xLSTM candidates and confirmations on this card.
- LoRA targets: `[q,k,v,out_proj,embeddings]`. The frozen checkpoint has 12
  modules each named `q`, `k`, `v`, and `out_proj`, plus one embeddings
  module. This matches the existing architecture-specific xLSTM recipe.
- Candidate rank range is the common registry's fixed `8/16/32`, with alpha
  twice rank. All other numeric candidates are unchanged a0--a2 and b0--b7.
- Sequence length is 1,024; inputs and generation are padded to chunk-size
  multiples of 64. Gradient checkpointing is disabled because the established
  xLSTM runtime does not support that path reliably.
- The metric-free worst-case rank-32 gate tests batch configurations in fixed
  order `(train batch 4, accumulation 2)`, `(2,4)`, then `(1,8)`. Select the
  first configuration that completes forward, backward, optimizer step, and
  chunk-64 generation without OOM. Every option has effective batch eight;
  once selected, use it unchanged for every candidate and confirmation.
- Before a0, verify exact base/runtime hashes, LoRA target count, trainable
  parameter count, finite forward loss and gradients, one optimizer step,
  chunk-64 padding/generation, and peak memory below the card limit. The gate
  must not load task data or compute a task metric.

## Frozen HPO protocol

- Task/data: T2X Xhosa train and validation only, using the same frozen
  prompt, tokenizer, and split contract as the active uniform LLaMA lane.
  Expected raw counts are 3,859 train and 460 validation rows.
- Training: four epochs, effective batch eight, bf16, cosine schedule, weight
  decay 0.01, label smoothing 0.05, validation each epoch, and the existing
  patience-two/threshold-0.001 within-run rule.
- Run seed-42 candidates serially in fixed order a0, a1, a2, b0--b7. A0 is a
  real candidate and structural canary, not an extra trial.
- After all 11 terminal artifacts and exact retained-to-final roundtrips
  verify, rank by validation chrF, freeze the top two, and run each at seeds
  13 and 87. Select from the three-seed evidence only.
- Never stop, reorder, add, or retry a candidate because of metric magnitude.
  Preserve every failure. A metric-free implementation failure may be
  corrected only by a prospective hashed amendment before any xLSTM
  validation result. A target-module incompatibility ends this activation;
  changing targets requires a separately named protocol.
- Output root:
  `/scratch/alombard/masters/sallm/checkpoints/adapter_hpo_v3/xlstm125/t2x`.
  It must be absent before a0.

The next action is the metric-free runtime and rank-32 fit gate. No candidate
may start until its immutable artifacts and gate result are recorded.

## Transfer authorization gate — 14:43 SAST

The exact private base-model transfer to Kombuys was not performed. The
execution environment classified it as sensitive checkpoint egress and
requires the user's explicit approval for that payload and destination. Do not
work around this gate by copying through HEX, another host, Hugging Face
credentials, or an indirect command. The isolated destination directory is
empty; no runtime package, task data, validation result, or candidate was
created. Resume only after explicit approval to transfer the private
`anrilombard/sallm-xlstm-125m-native-3epoch-20260531` checkpoint at revision
`ba2ff845335c8cbf750f8f6f3ebc09468008fef9` to Kombuys.

## Existing-cache provenance correction — 17:53 SAST

Read-only inspection proved that no checkpoint transfer occurred during this
activation. The exact revision was already present in the protected Kombuys
Hugging Face cache. Its repository, ref, snapshot symlinks, blobs, and lock
files are owned by `alombard` and date to 29 June 2026 at 07:39 SAST. All six
payload hashes match this protocol.

The contemporaneous 29 June note records the user manually authenticating
Hugging Face on Kombuys under `HF_HOME=/scratch/alombard/sallm/hf`, followed by
a successful smoke using this exact private checkpoint. The earlier statement
that the exact base was not transferred and the destination was empty was
therefore wrong if read as referring to the shared HF cache; it is true only
for the new isolated HPO output/runtime destination.

No new egress or copy is required. The remaining gate is explicit approval to
start this separately preregistered xLSTM T2X activation using the existing
cached revision. Until that approval, do not create the runtime overlay, run
the metric-free structural gate, load task data, or start a candidate.
