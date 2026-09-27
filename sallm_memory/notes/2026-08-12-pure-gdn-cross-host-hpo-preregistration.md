# Pure-GDN cross-host HPO assignment — 2026-08-12

Frozen at 22:20 SAST before any T2X enhanced-HPO validation score was produced
or consulted.

## Assignment

- NER remains entirely on HEX `gpu:ampere` (A100-40GB). Its candidates and
  confirmation seeds will not move to Kombuys.
- T2X is assigned entirely to Kombuys GPU 1, UUID
  `GPU-fec43e16-3955-238e-e517-e80cf92d0383` (RTX 3080 Ti, 12 GB).
  Stage A, Stage B, and confirmation seeds 13/42/87 will stay on that GPU
  class. GPU 0 / RTX 5090 and foreign processes are out of scope.
- Later families may be assigned prospectively, but no family may compare or
  combine candidates run on different host/GPU classes.

## Equivalence and launch gates

Before the first T2X candidate starts, all of these must pass:

1. Deploy the enhanced immutable source snapshot corresponding to HEX
   `/home/lmbanr001/masters/sallm_snapshots/uniform-adapter-hpo-20260811-6aabf717`.
   All source/config hashes must match, including registry SHA-256
   `8fdd6ea5f192a1d2ef2267353162b9a640ccd09ae0d889cb5a999645198bb726`.
2. Verify the six canonical pure-GDN model artifact hashes and exact T2X train
   and validation input hashes without loading or inspecting the test split.
3. Record the Python/package environment, GPU name/UUID, CUDA visibility, and
   storage state. Exactly GPU 1 may be visible to the run.
4. Run a bounded validation-only implementation/memory canary. It may inspect
   coverage, prompt/token contracts, model identity, finite execution, and
   peak memory, but its task score is neither a selection result nor a reason
   to alter the search. The canary must exercise evaluation batch 4; a launcher
   dry run must confirm the frozen real-trial semantics: train batch 4,
   evaluation batch 4, gradient accumulation 2, BF16, LoRA, and T2X max length
   1024.
5. Recheck that GPU 1 is idle immediately before launch and that GPU 0/foreign
   work is untouched.

Environment versions may differ between HEX and Kombuys because every T2X
candidate remains within one Kombuys environment. Such differences must be
manifested and must not change source, model, data, prompt, metric, optimizer,
batch, seed, early-stopping, or search-space contracts. Any failed gate blocks
launch; it cannot be waived using a validation score. No held-out split may be
loaded, scored, or used for any decision.

## Frozen execution order

Run T2X seed-42 candidates in registry order `a0`, `a1`, `a2`, then
`b0`--`b7`, one at a time on GPU 1. Rank only after all 11 are terminal-valid.
Then rerun the top two at seeds 13 and 87 and apply the already-preregistered
three-seed confirmation/tie-break rule. Infrastructure failures may receive
only an unchanged provenance-preserving retry; result-driven retries or search
changes are forbidden.
