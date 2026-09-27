# Pure-GDN A100 hardware-equivalence amendment, 2026-08-22

Preregistered at 20:35 SAST before any A100-80GB validation canary or adapter
run. The user authorized this scheduling-only expansion to target a complete
downstream table by 31 August without reducing any validation grid. No held-out
split may be loaded, inspected, or scored by this gate.

## Frozen scientific scope

- Keep every existing AfriHG and General seed-42 candidate and every Stage-C
  confirmation. The 31 August date is a target, not a stopping rule.
- Keep all recipes, seeds, prompts, tokenization, decoding, coverage,
  aggregation, checkpoint selection, early stopping, and tie rules unchanged.
- Continue to allow at most three owned GPU jobs. A100-40GB and A100-80GB jobs
  may overlap only after the gate below passes. Do not use or overlap L40S.
- Preserve all completed, running, pending, failed, and cancelled jobs and
  artifacts. Never-started pending jobs may be replaced unchanged to free a
  slot for the paired gate.

## Paired validation-only gate

Run `scripts/benchmark_pos_incremental_cache.py --mode full` once on
`gpu:ampere:1` A100-40GB and once on `gpu:ampere80:1` A100-80GB. Both jobs
must use:

- the same new immutable read-only source snapshot and complete execution
  manifest;
- the canonical pure-GDN checkpoint at
  `/scratch/lmbanr001/masters/sallm/checkpoints/sallm-pure-gdn-125m/a10080-3epoch-20260802/final_model`;
- the frozen POS b0 seed-42 adapter at
  `/scratch/lmbanr001/masters/sallm/checkpoints/adapter_hpo_v3/pure_gdn/pos/stage_b/b0/seed_42/checkpoint-566`;
- BF16, no adapter merge, the existing full-prefix scorer, and the deterministic
  first row from each of the 12 POS language/template validation cells.

`scripts/verify_a100_hardware_equivalence.py` must require:

1. Exact schema, validation-only boundary, source checkpoint, adapter hashes,
   validation row count, and subset indices.
2. Exact predictions, gold labels, token counts, correctness counts, cell
   identities, cell counts, cell metrics, and aggregate token accuracy.
3. Maximum selected-sequence score difference at most `0.05` and mean absolute
   difference at most `0.01`.
4. Reference GRES exactly `gpu:ampere:1` and candidate GRES exactly
   `gpu:ampere80:1`.

Record runtime and the reference-over-candidate ratio, but apply no runtime
acceptance threshold. Hardware speed cannot select a recipe or checkpoint.

## Decision rule

If every check passes, future unchanged pure-GDN validation HPO and Stage-C
jobs may use either A100 GRES concurrently within the three-job cap. The frozen
validation metrics remain the only selection evidence. If any check fails,
exclude A100-80GB and continue unchanged on A100-40GB. Do not relax a threshold,
rerun based on a score, or consult held-out data.

## Pre-gate sealed execution amendment, 22:50 SAST

The user approved this amendment before dependency removal from job `1257520`,
before any A100-80GB HPO submission, and without inspecting any A100-80GB result
contents. This section supersedes only the earlier scheduling rule that barred
A100 variant overlap before the gate passed. Every scientific setting and gate
threshold above remains frozen.

- A100-80GB validation jobs may execute before the paired hardware gate is
  complete, but their logs, metrics, checkpoints, generation artifacts, and
  result contents remain sealed. Slurm state, assigned GRES, elapsed time, exit
  code, file existence, and artifact hashes may be recorded for operations.
- The dependency on A100-80GB gate replacement `1257520` may be removed so it
  can run on the currently free device. Its result remains sealed until the
  A100-40GB reference `1257517` completes and the unchanged comparator runs.
- After `1257520` reaches terminal exit `0:0`, the exact frozen AfriHG b5
  seed-42 candidate may use the freed A100-80GB slot. It must use the same
  immutable HPO snapshot, registry, model, data, evaluator, seed, output root,
  24-hour limit, and eight-CPU envelope as the cancelled, never-started job
  `1257468`. Only the account, QOS, and GRES change to
  `nlpgroup80/nlpgroup80/gpu:ampere80:1`.
- At most three owned GPU jobs remain allowed. L40S remains prohibited. No
  held-out split may be accessed, and sealed A100-80GB results cannot affect
  recipe, checkpoint, retry, rerun, or confirmation selection.
- If the hardware gate passes, sealed A100-80GB validation artifacts may enter
  the ordinary verification and validation-only selection process. If it
  fails, preserve those artifacts as quarantined provenance and rerun the same
  jobs on A100-40GB. Do not change thresholds or decide from the sealed scores.
- The completed overlapping canary `1257518` remains quarantined and cannot
  replace either member of the paired gate.
