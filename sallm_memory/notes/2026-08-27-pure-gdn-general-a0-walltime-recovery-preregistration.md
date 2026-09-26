# Pure-GDN General a0 wall-time recovery — prospective draft, 2026-08-27

## Authorization and one-time submission — 20:38 SAST

The user authorized this frozen recovery after the timeout diagnosis. The
fail-closed preflight confirmed job `1271042` terminal `TIMEOUT`, exact hashes
for all five checkpoint-10912 state files, absent final adapter, absent prior
resume manifest, and no active duplicate. Resume job `1275241` was submitted
once under `nlpgroup80/a100/nlpgroup80` with one `gpu:ampere80`, 24 hours,
eight CPUs, and the unchanged immutable launcher. It is currently
`AssocGrpGRES` pending. This authorization applies only to this exact recovery
and does not relax any validation, held-out, retry, or selection rule.

Drafted at 16:27 SAST before job `1271042` reaches its 24-hour limit and
without held-out access. This document does not authorize or submit a recovery.

## Terminal confirmation — 17:26 SAST

Job `1271042` reached step `12131/13640` cleanly and Slurm ended it at
`16:41:29` with state `TIMEOUT` due solely to the 24-hour limit. No final
adapter or post-step-10912 checkpoint exists. The five checkpoint-10912 hashes
below reverified unchanged after termination. The exact-resume recovery remains
unsubmitted pending scientific authorization.

## Observed mechanism

- Job `1271042` remains healthy, with no model, data, evaluator, or training
  fault. At `23:41:43` elapsed it was near step `11964/13640`; its allocation
  ends at `16:41:18` SAST. The remaining training alone requires roughly three
  hours at the observed step rate, so terminal completion is impossible in the
  original allocation.
- The latest complete checkpoint is `checkpoint-10912`, written at the fourth
  frozen validation boundary. Its trainer state records epoch `4.0`, global
  step `10912`, best macro NLL `0.9838042431257504`, and itself as the best
  checkpoint.
- The paired step-10912 validation artifact has exact 22,167-row six-family
  coverage and sidecar-verified SHA-256
  `439f31199f759ba623f667faf8e6974695cd954cf8c95c4288bced56884c124d`.
- The checkpoint contains adapter, optimizer, scheduler, RNG, and trainer
  state. Their SHA-256 values are respectively
  `4b4cfad3fb1623dd8e3dee86ee2f886115ae2204d962c206d1ed0818e9818151`,
  `b6d542bec628c82406bc2cf7b5ad7e7c8926e6e2a9156c8fc69350c2401d8088`,
  `fbc66d5d0b0f0c06cf7d343921624ddda9250d0e2fd4cd0dbf87f8eea88c85aa`,
  `0e8011af71b30b3d6ccfc2254349da97f7f40efd8a480228ecb2abc78b3229a9`,
  and `ab85f2c2a4053a3d3fd92535ae4e8987044db2765086a54cced615edf95fb5a3`.

## Frozen recovery, if separately authorized after terminal diagnosis

Resume once from the complete `checkpoint-10912` in the same output directory,
using the same immutable `uniform-adapter-hpo-20260811-6aabf717` launcher,
current frozen runtime, candidate a0, seed 42, registry, General coverage and
equal-family macro-NLL protocol, BF16 recipe, 24-hour A100-80GB resource
request, early-stopping state, and all original selection rules. Set only
`SALLM_HPO_RESUME_FROM_CHECKPOINT` to the exact checkpoint path. The existing
launcher already validates checkpoint locality and completeness and records a
new resume-specific execution manifest and trial record.

Before submission, require job `1271042` to be terminal, preserve its exact
Slurm state/log, verify the five checkpoint hashes again, verify final adapter
absence, and verify no active duplicate. After completion, require the normal
terminal artifact/sidecar checks and exact retained-to-final adapter roundtrip.
Do not inspect held-out data, change a recipe, restart from scratch, or reuse
any incomplete post-checkpoint state.
