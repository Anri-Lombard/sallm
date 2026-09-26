# Pure-GDN POS runtime optimization preregistration — 2026-08-16

Preregistered at 09:49 SAST before any optimized or cross-GRES run. This is
an implementation-only correction prompted by measured runtime, not by a
candidate comparison. No held-out split may be loaded or inspected.

## Frozen scope

- Job `1238876` continues unchanged on `gpu:ampere`; its source, evaluator,
  checkpoint decisions, and artifacts will not be modified.
- Pending jobs `1238877/1238878` have not loaded model or data. They may be
  cancelled only after the gates below pass, then resubmitted unchanged from
  a new immutable source snapshot. Their recipes, seeds, data, prompts,
  label set, score aggregation, early-stopping rule, and output contract stay
  frozen.
- The optimization may replace repeated full-prefix inference with the
  model's incremental recurrent cache. It may not alter tokenization, legal
  labels, selected-label scoring, row coverage, cell aggregation, or metric
  values.
- `srvrocgpu009` `gpu:amperemk` may be added only as another A100-40GB GRES
  after prospective equivalence against `srvrocgpu010` `gpu:ampere`.

## Fixed gates

1. Unit regression: cached and full-prefix decoders must return identical
   predictions and selected-label scores on a deterministic cache-aware fake
   model, while cached inference processes fewer tokens.
2. Real-model implementation gate: on the frozen POS b0 checkpoint-566
   adapter and a deterministic validation-only subset containing all 12
   language/template cells, full-prefix and cached modes must have identical
   row predictions, token counts, cell counts, and aggregate accuracy. The
   maximum absolute selected-sequence log-score difference must be at most
   `0.05`, with mean absolute difference at most `0.01`. Cached elapsed time
   must be no more than one third of full-prefix elapsed time.
3. Hardware gate: cached evaluation of the identical frozen subset on
   `gpu:ampere` and `gpu:amperemk` must have identical predictions, token and
   cell counts, and aggregate accuracy. Maximum absolute selected-sequence
   log-score difference must be at most `0.05`, with mean absolute difference
   at most `0.01`.
4. Provenance gate: the optimized source, launcher, configuration,
   environment, model, adapter, subset indices, and output hashes must be
   captured in immutable manifests before Stage-B resubmission.

If all gates pass, pending-only b1/b2 may be replaced by unchanged jobs using
the cached evaluator and available A100-40GB GRES. The old Slurm job records
remain failure/provenance evidence. If any gate fails, no HPO job will use the
new path. Held-out evaluation, winner freezing, and Sheet E/F/G remain
blocked.

## Queue-fusion amendment — 10:07 SAST

The live `nlpgroup` association has `GrpTRES gpu:amperemk=0`, so the
cross-GRES gate cannot run and no hardware expansion is possible. To avoid a
separate `gpu:ampere` queue turn, the unchanged implementation gate above may
run at the start of b1's allocation, before b1 loads its training data or
updates any weights. It still uses frozen b0 checkpoint 566 and the same 12
validation-only cells. B1 may proceed in the same allocation only if gates
1, 2, and 4 pass. Gate 3 is inapplicable while all scientific runs remain on
the original `gpu:ampere` GRES; it becomes mandatory before any future
`gpu:amperemk` use. This changes scheduling only, not acceptance thresholds,
recipes, data, metrics, or selection.

## First real-model gate result — 10:13 SAST

- At the user's explicit direction, slow POS Stage-B b0 job `1238876` was
  cancelled at `10:09:55 SAST` after `06:49:23`. Slurm records
  `CANCELLED by 733329384`, exit `0:0`. Its original execution manifest,
  trial record, validation artifacts at steps 283 and 566, and complete
  `checkpoint-566` remain preserved. The interrupted b0 is provenance-only
  and is not a terminal-valid candidate.
- Fused cache-gate/b1 job `1238979` immediately acquired the released
  `gpu:ampere` A100-40GB on `srvrocgpu010` at `10:09:56 SAST`; there was no
  projected two-hour wait in practice. It stopped before b1 training after
  `00:01:49`, exit `1:0`, because the preregistered runtime gate failed.
- Scientific equivalence passed exactly on the frozen 12-cell,
  validation-only subset: predictions, row and cell counts, cell metrics,
  aggregate accuracy, and selected-label scores all match; maximum and mean
  score differences are both `0.0`. Runtime improved from
  `48.3900309689343 s` to `34.23066069406923 s`, only
  `1.413645836445053x`, so `cached_time_at_most_one_third=false`. The locked
  3x threshold was not relaxed or bypassed.
- Gate artifacts are preserved under
  `/scratch/lmbanr001/masters/sallm/results/diagnostics/pos_runtime_gate_20260816/job-1238979/`.
  Equivalence and verification SHA-256 values are
  `70feaadcedb149d4b27245121440c3f3db47b27ae48a5ae68de6fd538f2f742c`
  and `6e146232ef271397f7ecb608a3fd9fcae995d6c1748280ee1f943014f474beac`.
  No held-out split was loaded, no b1 weight update occurred, and no HPO job
  may use `incremental_cache_v1` under this preregistration.

## Last-logit-only v2 amendment — 10:17 SAST

After v1 failed on runtime only, read-only inspection of the installed FLA
0.5.1 `GatedDeltaNetForCausalLM.forward` signature established native
`logits_to_keep` support. The scorer consumes only the last-position logits
for every one-token UPOS label, while v1 unnecessarily projected every new
position through the full vocabulary head.

V2 may add only `logits_to_keep=1` to the already-equivalent cached forward.
The reference remains the original full-prefix v1 path. The frozen model,
b0 checkpoint-566 adapter, 12 validation-only rows, row order, tokenization,
labels, predictions, score aggregation, equality checks, score-difference
tolerances, and 3x runtime threshold remain unchanged. A deterministic unit
check must prove the optimized path requests one final logit and preserves
predictions/scores. B1 may train in the same allocation only after the same
real-model and provenance gates pass; otherwise it must fail closed before
loading training data or updating weights. No held-out split may be loaded.

## Last-logit-only v2 result — 10:21 SAST

- Focused local checks passed: `15/15` tests, Ruff, shell syntax, and
  whitespace validation. Immutable snapshot
  `/home/lmbanr001/masters/sallm_snapshots/pure-gdn-hpo-poscache-lastlogit-20260816-61ce254f`
  verifies all `698` source/config files; deployment-manifest SHA-256 is
  `c491eaa943ea9ad3aa4f100cacc2eaa7dbfeed5b767b67086eae2645f41d98f5`.
- Fused gate/b1 job `1238982` started immediately on `srvrocgpu010`
  A100-40GB at `10:19:01 SAST` and failed closed after `00:01:33`, before
  training. All prediction, coverage, cell metric, aggregate accuracy, and
  score checks again matched exactly; maximum and mean score differences
  were `0.0`.
- Full-prefix and v2 cached timings were `35.0182068439899 s` and
  `33.77654465707019 s`, respectively: only
  `1.0367610778285992x`, so the unchanged 3x runtime gate failed. The
  equivalence and verification SHA-256 values are
  `8e0e8c98d23293a178c434a9c34b460451a540d0f39e19ac0587982bc611610d`
  and `69a43d81d4d59cbb94443c54cf2a33f623fd512f56f25ef04dbe52024a943827`.
  No held-out split or training data was loaded and no weight update occurred.
- Native last-logit projection is not the material bottleneck. No further
  speculative POS runtime job is authorized by this protocol, and neither
  cached implementation may be used for HPO. The scheduler is left with
  zero owned jobs pending a scientifically explicit next execution choice.

## Cross-row full-prefix batching v3 amendment — 10:40 SAST

Preregistered before implementing or benchmarking v3. The two rejected cache
paths remain prohibited. Timing evidence from the original full-prefix run
shows that serial exact validation, rather than training, consumes roughly
85--90% of POS wall time. V3 may therefore batch independent validation rows
at each tuple position while preserving a full-prefix, `use_cache=False`
forward for every active row. The fixed row batch size is `8`; it is not a
tunable hyperparameter.

The reference remains the original serial `full_prefix_v1` implementation.
The real-model gate remains frozen to b0 checkpoint-566 and the deterministic
first validation row from each of the 12 language/template cells. V3 must
match reference predictions, token counts, row order, cell counts, cell
metrics, and aggregate accuracy exactly. Maximum absolute selected-sequence
score difference must be at most `0.05`, mean difference at most `0.01`, and
batched elapsed time must be no more than one third of serial elapsed time.
Every label continuation must remain exactly one token; v3 fails closed
rather than silently changing algorithms. A deterministic unit regression
must also match serial predictions and scores while using fewer forward calls.

Only pending b2 may be replaced after local checks and immutable-provenance
verification. Its A100 allocation must execute the real-model gate before
loading b2 training data or updating weights, then proceed in the same
allocation only if all gates pass. If the gate fails, b2 aborts before
training and must later be resubmitted with `full_prefix_v1`. Running b0/b1
jobs are untouched. Recipes, data, prompts, label set, validation frequency,
early stopping, checkpoint retention, and aggregation remain frozen. No
held-out split may be loaded or inspected. All v3 work remains on the original
`nlpgroup/a100/gpu:ampere` A100-40GB envelope; L40S and A100-80GB remain out of
scope.
