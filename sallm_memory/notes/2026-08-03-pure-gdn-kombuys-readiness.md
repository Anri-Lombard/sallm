# Pure-GDN Kombuys downstream readiness — 2026-08-03

## Scope

- Diagnostic-only readiness work on Kombuys; no benchmark score, checkpoint,
  prompt, or recipe selection is permitted from the two-step implementation
  canary.
- RTX 5090 was occupied by another workload. The isolated work uses only the
  idle RTX 3080 Ti through `CUDA_VISIBLE_DEVICES=1`.

## Accepted inputs and transfer

- Synced the reviewed `src`, `scripts`, `pyproject.toml`, and `uv.lock` trees
  to `/scratch/alombard/sallm` and obtained an empty checksum dry-run.
- Copied only accepted canary job `1165989`'s `final_model` to
  `/scratch/alombard/sallm/checkpoints/sallm-pure-gdn-125m-canary/a10080-1165989/final_model`.
  All six source/destination SHA256 values match; the transferred artifact is
  248 MB and the accepted HEX canary remains untouched.
- Kombuys scratch has about 2.3 TB free.

## Findings

- FLA 0.5.1 imports on Kombuys with PyTorch 2.9.1/CUDA 12.8.
- The copied checkpoint reloads as pure `GatedDeltaNetForCausalLM` with
  exactly `127,425,448` parameters.
- Pure-GDN linear projection suffixes are `q_proj`, `k_proj`, `v_proj`,
  `a_proj`, `b_proj`, `g_proj`, `o_proj`, plus the MLP projections. The
  existing conservative fine-tune default `q_proj`/`v_proj` is therefore
  mechanically valid, but it is not evidence that those targets are the
  architecture-optimal recipe.
- The real MasakhaNews classification readiness lane loaded the pure model,
  attached rank-4 LoRA (`216,576` trainable parameters), formatted 3,309
  training examples and expanded 472 validation examples over five prompts,
  tokenized the resulting 3,309/2,360 train/validation rows, and entered one
  BF16 optimizer step at 256-token context on GPU 1. No held-out test is used.
- Two strict-Hydra launcher errors (`max_steps` and a config-absent
  `gradient_checkpointing` field) and a nullable W&B schema misuse were fixed
  only in the temporary Kombuys smoke launcher before GPU work. These are not
  repository implementation defects.

## Running / next

- Tmux session: `pure-gdn-downstream-readiness`.
- Log:
  `/scratch/alombard/sallm/logs/pure_gdn_readiness/downstream_smokes.log`.
- Output root:
  `/scratch/alombard/sallm/results/diagnostics/pure_gdn_downstream_readiness_20260803`.
- After classification succeeds, the same launcher runs one AfriHG Xhosa
  generation-task train step. Both outputs remain diagnostic-only.
- Kombuys SSH began timing out during the first FLA JIT/optimizer step; tmux
  continues independently. Reconnect and verify completion, adapter save, and
  absence of traceback/OOM before using the final pretrained checkpoint.

## Closeout — 2026-08-04

- MasakhaNews classification completed and saved its adapter; it was not
  repeated.
- The first AfriHG Xhosa attempt stopped during dataset loading when Kombuys
  DNS could not resolve `raw.githubusercontent.com`. After DNS recovered, only
  the generation arm was retried on RTX 3080 Ti GPU 1.
- The retry completed one BF16 step through the TileLang backward path, ran
  generation callbacks, and saved `afrihg_generation/final_adapter`. There was
  no traceback or CUDA OOM in the retry.
- The diagnostic step logged zero loss and zero gradient norm because the
  selected long/truncated batch exposed no trainable target tokens. This means
  the lane verifies mechanical task execution and adapter persistence only; it
  does not validate learning or establish `q_proj`/`v_proj` as an optimized
  pure-GDN target set. No diagnostic metric is reportable or selectable.
