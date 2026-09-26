# Equal-recipe full fine-tuning: Monolingual T2X (isiXhosa) pilot, 2026-09-25

One recipe, identical for MzansiLM (Transformer), Mamba-2, xLSTM and Gated DeltaNet (~125M each).
Written before any training run. Results root on HEX:
`/scratch/lmbanr001/masters/sallm/results/equal_recipe_fullft_t2x_pilot_20260925/` (scripts in `scripts/`,
hashed in `scripts/SCRIPTS.sha256`).

## Models (full fine-tuning of every parameter, no LoRA/PEFT)

| arch | Hydra `model.architecture` | base (tree sha256 verified in-job against the paper's bindings) |
|---|---|---|
| mzansilm | llama | `/scratch/lmbanr001/masters/sallm/checkpoints/sallm-llama-125m/final_model` (b7572280...) |
| mamba2 | mamba2 | copy of `full-matrix-retained-bindings-20260916-v1/bases/mamba2` (source tree 5db6860f...) with `config.json`/`generation_config.json` eos_token_id 2->1, pad_token_id 1->2 (tokenizer: [BOS]=0, [EOS]=1, [PAD]=2) at `bases/mamba2_eosfix` |
| xlstm | xlstm | `full-matrix-retained-bindings-20260916-v1/bases/xlstm` (ff2e7c99..., same tree as the paper's xLSTM base binding) |
| gdn | gated_deltanet | `/scratch/lmbanr001/masters/sallm/checkpoints/sallm-pure-gdn-125m/a10080-3epoch-20260802/final_model` (5ceca7d0...) |

`peft.method=none`; the entry point asserts trainable params == all params and all params fp32.
Tokenizer = each base's own tokenizer (as in the existing T2X recipes; all four have [BOS]=0, [EOS]=1, [PAD]=2).
Special tokens / chat template: unchanged sallm fine-tuning path (`sallm.fine_tune.run`): add `<|system|>`, `<|user|>`,
`<|assistant|>` as additional special tokens, resize embeddings (65536 -> 65539; new rows are trained like every other
parameter), install the canonical chat template, left truncation, `assistant_only_loss=true`, no packing, Mamba-2
lm_head re-tied to the resized embeddings (existing resize-bug fix). Note: the MzansiLM base config says bos 1 / eos 2,
which disagrees with its tokenizer; left unchanged (as for the LoRA units) because generation passes the tokenizer EOS.

## Data and prompt

Same as the recent T2X retrains: `full_matrix_targeted_recovery_20260916_v9/assets/t2x_train_validation_only`
(3859 train / 460 validation; sha256 of train/valid .data/.text = the runner's pinned hashes, re-checked in-job), copied
to `assets/t2x_train_validation_only`. Loaded through the v9 fail-closed train/validation-only loader (no test split
reachable during training). Template `t2x_verbalisation/v1` from `finetune/llama_t2x_xho` (the paper's T2X protocol).

## Optimisation (identical for all four)

- AdamW (`adamw_torch`), betas (0.9, 0.95), eps 1e-8, weight decay 0.01
- cosine schedule, warmup ratio 0.10 (242 steps/epoch, 968 steps total, 97 warmup steps). Author update 2026-09-25 21:20:
  0.10, not 0.05; a first submission with 0.05 was cancelled after ~5 min (no run finished) and archived in `aborted/`
- effective batch 16 = 16 per device x 1 accumulation step, 1 GPU (L40S)
- 4 epochs, as instructed ("T2X stays at 4 epochs"). Open question: the stated rollout rule "10 epochs if the training set
  has <5000 examples, else 4" would give 10 for T2X (3859 rows); the pilot follows the explicit 4. Gradient clipping 1.0, seed 42, data_seed 42 (seed runs: seed = data_seed = 43 / 44)
- max_seq_length 1024 (the MzansiLM, xLSTM and GDN T2X recipes; Mamba's used 2048 - immaterial, T2X examples are far shorter)
- label smoothing 0 (not part of the shared spec; the existing recipes disagreed: 0.05 for three, none for xLSTM)
- no gradient checkpointing; model dropout as in each pretrained config
- weight decay applies to the parameters the sallm trainer decays (HF default: no decay on biases/norms; the sallm
  trainer additionally exempts Mamba-2 `A_log` and `D`, as in every existing Mamba run)
- precision: bf16 autocast with fp32 master weights and fp32 AdamW state. The stock sallm loader would load the model
  *in* bf16 when `bf16=true` (fine for LoRA, pure-bf16 for full fine-tuning), so `train_fft.py` forces fp32 loading.
  If an architecture produces non-finite values in bf16 (anomaly detection with `check_nan=True` is on, as in the
  existing runner), the job stops, the architecture is marked `PRECISION_FP32`, and its whole sweep is rerun in fp32
  (reason recorded).
- xLSTM only: `pad_to_multiple_of=64` (its training kernel requires chunk-size multiples; pad labels are masked, so the
  loss is unchanged). xLSTM trains in the sealed runtime (no mlstm kernels) as the paper's xLSTM runs did; the others
  in the main venv (mamba_ssm fast path asserted for Mamba-2; FLA 0.5.1 for GDN).
- per-epoch checkpoints (model weights only, fp32 `pytorch_model.bin` via the sallm trainer's save_model, ~0.5 GB) written to node-local `/dev/shm`; in-training eval =
  validation loss only (task-metric callbacks disabled; selection uses the paper's generation protocol below).

## Learning-rate sweep and selection

- Grid {1e-5, 3e-5, 1e-4}, identical for all four. Edge rule: if any architecture's best lr on this grid is 1e-4, add 3e-4
  for all four; if any is at 1e-5, add 3e-6 for all four (one value per edge, grid kept identical).
  To avoid a second queue round trip, 3e-4 and 3e-6 are trained up front for every architecture; they enter
  selection only if the rule fires, and unused runs are reported as such.
- Every epoch checkpoint is scored on the FULL validation split (460) with the paper's generation protocol: greedy,
  system prompt dropped, `run_generation_direct.py --split val --system-prompt drop --decoding greedy` from
  `generation-protocol-v3-greedy-20260924` (copy with one change: skip hash checks for a null adapter; the fine-tuned
  model is passed as the base with no adapter, as for the Base-model units); xLSTM uses the v4 batch-1 + use_cache runner
  (`generation-protocol-v4-xlstm-bs1-20260925`, already null-adapter aware).
- Best (lr, epoch) per architecture by validation chrF; ties -> smaller lr, then earlier epoch.
- Only the selected checkpoint per architecture is scored on TEST (same runner, `--split test`), unit id
  `u-fft-mono-<arch>-t2x-xho`; UNIT_DONE.json carries lr, epoch, n_trainable_params, model tree sha256, precision.
- Seeds (author update): at each architecture's selected lr, two more runs with seed 43 and 44 (same recipe); each seed's
  best epoch is chosen on validation the same way and scored on TEST (`u-fft-mono-<arch>-t2x-xho-s43/-s44`). Reported:
  the three test chrF values, mean and sample std; seed 42 is the primary. Per-item test outputs
  (`t2x_xho/examples.jsonl`) are kept for every test unit for bootstrap CIs.
- Batch-size sensitivity (author addition): at each architecture's selected lr, seed 42, effective batch 8 and 32
  (8 x 1 and 32 x 1 per device, lr NOT rescaled, same warmup ratio / epochs, so 483 or 121 steps per epoch), best
  epoch on validation, then TEST (`u-fft-mono-<arch>-t2x-xho-b8/-b32`); reported with batch 16 in batch_sensitivity.csv.
- All validation/test scoring on HEX L40S (one GPU type). Per training run: GPU type, wall-clock training time, optimizer
  steps, examples seen, non-padding tokens processed (and assistant/loss tokens), tokens/s, samples/s, peak
  `torch.cuda.max_memory_allocated`, trainable parameters, per-epoch validation curve; per scoring run: wall time and
  generated tokens/s (sweep.csv / test.csv).

## Storage

Epoch checkpoints never touch HEX /scratch: they live in node-local `/dev/shm` while being scored. Each lr's best epoch
is copied to `keep/` (needed for the cross-architecture selection; at most 20 x 0.5 GB); once the edge rule is applied
the unselected ones are deleted, and the selected checkpoints (seed 42 and seeds 43/44) are moved to Kombuys
(`/scratch/alombard/sallm/results/equal_recipe_fullft_t2x_pilot_20260925/`) and deleted on HEX.

## Execution

One L40S job per (architecture, lr group): core {1e-5, 3e-5, 1e-4} and extension {3e-4, 3e-6}, each trained and scored
serially on its own GPU, so per-run timings are unshared. When all five lrs of all four architectures are done, the last
job applies the edge rule, submits the eight seed jobs and scores the four seed-42 selections on test.

## Pre-flight check (done before the HEX jobs started)

CPU smoke on Kombuys (MzansiLM, 6 steps, same overrides except max_steps/eval/save every 3 steps): 125,009,920 of
125,009,920 parameters trainable, all fp32, embeddings tied, vocab 65539; checkpoints hold fp32 `pytorch_model.bin`
(500 MB) + tokenizer/chat template; the null-adapter protocol runner loaded a checkpoint and generated (5 val rows,
3080 Ti). Kernel-dependent architectures (Mamba-2, GDN, xLSTM train mode) are first exercised in the HEX jobs.
