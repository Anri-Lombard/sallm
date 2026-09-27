# Pure-GDN family-wise gate review correction

Frozen prospectively on 31 August 2026 after independent review, before the
metric-free CUDA gate ran and before any current adapter official held-out
access. Pending gate job `1279742` was cancelled at elapsed zero. It produced
no model, data, evaluator, prediction, metric, checkpoint, or result payload.

This correction supersedes only ambiguity and implementation gaps in the
family-wise acceleration amendment. Its scientific order, recipes, frozen
checkpoints, one-time official-test rule, and held-out firewall do not change.

## Ordering clarification

The sentence beginning "Submit no new General candidate" means no new
**General-scoped** candidate, recovery, ranking, confirmation, Mono, or
held-out job until all non-General family-wise results verify. It does not
block the non-General HPO, Mono, CUDA-gate, or official-test jobs required by
the fixed execution order. Each non-General family still advances and releases
independently.

## Exact-checkpoint gate correction

The replacement gate must use one new immutable snapshot and result root. It
adds the following fail-closed checks before any official test:

- the wrapper accepts only the exact frozen pure-GDN base checkpoint path that
  the deployment manifest hashes;
- manifest verification rechecks the recorded Python executable, Python
  version, platform, and complete installed-package version mapping in the
  execution environment;
- BF16 forward/backward rejects a non-finite loss, any missing trainable
  parameter gradient, or any non-finite trainable parameter gradient; and
- save/reload equality and deterministic greedy generation remain required.

These are metric-free execution checks. They do not load a task or held-out
split and cannot affect HPO or checkpoint selection. The focused regression
tests first failed against the old implementation, then all `13` relevant
tests passed and Ruff was clean after the correction. Corrected SHA-256 values
are:

- execution-manifest utility:
  `6d02c4e020c7476a34c123130ce674526dd15624a40e17a4ae6ed11e52e87546`;
- exact-checkpoint verifier:
  `0c2958dc7c8ae02e70555544ad89f93d6f4b6262034e9efd9aaa337ca4848a5f`;
- gate wrapper:
  `1d9f31497061081d862b2c4d122d5afb54c33aa5313c52c5dff4c679357d25f5`;
- manifest regression test:
  `54970ffe9f2198838981752838c9b30fc607fd13ff363aa8eb27d54b14eff4bf`;
- runtime regression tests:
  `ba9197b0f142fc75a1c5c6ef8b9cdf0d629a11e915c06852d0e864938aed2d60`.

No replacement gate may be submitted until its new deployment manifest and
snapshot hashes verify. Never reuse job `1279742`, its snapshot, or its result
root.
