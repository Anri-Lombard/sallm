# Mamba-2 T2X activation failure — 2026-08-30

Recorded at 14:18 SAST after the frozen metric-free activation gate failed and
before any Mamba training row, validation row, held-out row, task metric, or
candidate was loaded.

## Preserved gate failure

The one-time activation gate ran from immutable snapshot
`/scratch/alombard/sallm_snapshots/uniform-adapter-hpo-postbos-mamba2-news-20260830-9530feb6`
and stopped during tokenizer-contract verification. The isolated diagnostic
files are:

- `preflight.json`: SHA-256
  `886cc538e0adc6b16c6d15efa0ebdaeb35d2f1b5eba0c4cd4feee44804b3d3cd`
- `preflight.log`: SHA-256
  `6f8604f2eb46b3bd1c5261fb7cbb3b968b4df8f39955db7c563bdf46b8e577b4`

The immediate error was `Frozen tokenizer is missing canonical chat special
tokens.` This was a verifier assumption defect: the frozen training path is
supposed to add exactly `<|system|>`, `<|user|>`, and `<|assistant|>`, resize
the embeddings, and train those three rows. Correcting that assumption would
change the expected trainable count from `6,663,168` LoRA-only parameters to
`6,664,704` including the 1,536 token-row parameters.

## Deterministic structural blocker

A CPU-only diagnostic mirrored that real token-registration path without
loading data or computing a metric. The preserved probe files are:

- `cpu_structural_probe.py`: SHA-256
  `c203de4932020185e1d9d421f6363f19e5243bee6503fd23df041d8fc7ab7c39`
- `cpu_structural_probe.log`: SHA-256
  `0497c4fff1bf405540a3e02460958751be3ff67f8510d2f0e027e38d5925a957`
- `cpu_structural_probe.exit`: SHA-256
  `4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865`
  and exit code `1`

PEFT `0.18.1` then deterministically rejects the frozen `out_proj` target for
`model_type=mamba2`:

> Module `out_proj` is incompatible with Mamba-based models. Incompatible
> modules: `out_proj`, `conv1d`.

The frozen protocol requires both `in_proj` and `out_proj`. Removing
`out_proj`, bypassing PEFT, or replacing the adapter implementation would be a
new scientific protocol, not an implementation correction. Therefore this
Mamba activation is terminally blocked: do not retry the gate, launch a Mamba
candidate, or change targets under this preregistration. Kombuys GPU 1 remains
idle. The isolated LLaMA lane on GPU 0 is unaffected.
