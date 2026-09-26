# Pure-GDN Kombuys RTX 5090 equivalence preregistration, 2026-08-27

Preregistered at 20:49 SAST before any pure-GDN RTX 5090 gate or downstream
run. The user authorized concurrent Kombuys use to shorten the complete-table
timeline. No held-out split may be loaded, inspected, or scored by this gate.

## Frozen scope

- Use only Kombuys GPU 0, NVIDIA GeForce RTX 5090 with 32 GB. Exclude the
  12 GB RTX 3080 Ti because the unchanged General recipe has exceeded 12 GB.
- Keep every recipe, seed, prompt, tokenization, decoding, validation split,
  aggregation, checkpoint rule, early-stopping rule, and confirmation rule
  unchanged.
- Existing HEX work may continue while the gate runs, but no Kombuys result may
  enter selection and no downstream Kombuys job may start before the comparator
  passes.
- Preserve a failed gate and exclude Kombuys. Do not rerun, relax a threshold,
  or consult held-out evidence.

## Validation-only gate

Compare the preserved, ratified A100-80GB POS full-prefix artifact from job
`1271038` with one Kombuys RTX 5090 run. The candidate must use the exact
read-only A100 equivalence source snapshot, the canonical pure-GDN model, the
frozen POS b0 seed-42 checkpoint-566 adapter, BF16, no merge, and the same first
row from all 12 POS validation cells.

The comparator must require:

1. The preserved A100 gate passed and its verification artifact has SHA-256
   `9f699ff2fe0377d6d149c25863500b3c252565b0d477fa2546253f25bd1fa377`.
2. Exact source hashes, model file hashes by basename, adapter hashes,
   validation row count, and subset indices.
3. Exact predictions, gold labels, token counts, correctness, cell identities,
   cell counts, cell metrics, and aggregate accuracy.
4. Maximum selected-sequence score difference at most `0.05` and mean absolute
   difference at most `0.01`.
5. Candidate host `kombuys`, GPU `NVIDIA GeForce RTX 5090`, one visible CUDA
   device, and both stable-environment flags set to `1`.

Python, CUDA, PyTorch, package inventory, and runtime are recorded but are not
required to equal HEX. The gate tests whether that different declared runtime
reproduces the frozen validation behavior. Runtime never selects a recipe.

## Decision rule

If every check passes, one RTX 5090 job may run concurrently with eligible HEX
A100-80GB jobs. Kombuys may execute only preregistered validation work and,
after all validation freezes, the already-frozen one-time downstream program.
If any check fails, preserve the evidence and continue on HEX only.

## Terminal result, 20:58 SAST

The one-time gate completed without an execution fault, but failed the frozen
scientific comparator. Candidate artifact SHA-256 is
`929eb1b5419821d39d38f5a13b3782c9492ecdbf6c181681848861f51448c6c2`;
candidate manifest SHA-256 is
`82dfde789c938081c76183b9f24503046dc8318c6b85b026855910cf0245f27d`.
The comparator artifact SHA-256 is
`51fc366d4e1a2f5665a8aa33991f8f383290680a51ba6ed919cb559b46f99492`.
The immutable launcher and comparator hashes are respectively
`155353ffe363ae21eb9ce0a4fa60b01069df986614385541f675fb8452031b1d`
and
`5cb064d879a246347b4245ebf4dd0d438dea821ca5bbf1df3c3f9508159aaed4`.

Source, model, adapter, validation boundary, hardware identity, and stable
environment checks passed. Predictions, cell counts, cell metrics, and
aggregate accuracy differed. Maximum selected-score difference was
`0.053466796875`, above `0.05`; mean absolute difference was
`0.015472412109375`, above `0.01`. Frozen scoring time was `204.61164229223505`
seconds versus `36.0765298968181` seconds on A100-80GB, so the RTX 5090 was
about 5.67 times slower on this workload. The gate is terminal and will not be
rerun or relaxed. Both Kombuys GPUs remain excluded from pure-GDN model work.
