# Pure-GDN General b4 wall-time continuation implementation — 2026-08-31

Status: prospective and frozen at 21:42 SAST before any b4 continuation
submission or recovery manifest creation.

## Eligibility

Original b4 job `1277379` ended `TIMEOUT`/`0:0` at the fixed 24-hour wall,
without a final adapter. The numerically latest complete scheduled training
state is checkpoint `10912`. No b4 job is active, no final adapter exists, and
no resume-specific manifest exists. This satisfies the metric-independent
eligibility rule frozen in the 30 August later-timeout amendment.

The original root is preserved in a read-only archive:

- path: `/scratch/lmbanr001/masters/sallm/recovery_archives/general-stage-b/b4-seed42-pre-recovery.tar`;
- mode: `0444`;
- size: `195,543,040` bytes;
- SHA-256: `ab2c93160222ba8bfe0a48e086a67a1548cbcbfcc67ce0977e60aa583a4ac399`.

## Frozen bindings

- original execution manifest:
  `5791f6fb4dcb03c3cf41cc3f2d1c6ab91627f1cea71d05454c8bc645a0e7ae9e`;
- checkpoint adapter:
  `edc7cdac66426ac5824903492a65da3987503309b5d4cbdfb06af0406c29be66`;
- optimizer:
  `f5cbd3ac2e9d0ab27464fde538bf01e3a2eb75f1652ea2bb260d668fdba04e5d`;
- scheduler:
  `ce4fcc97ebd7b7c36ccf7a5faa5df7b8a4794b4841d3f329517b029a0eff92a5`;
- RNG state:
  `d31146d7c73d1c62b0541ebcd0c731dcb2a9aa1652731b25333180d24e9d170f`;
- trainer state:
  `13f8c6df16c4ac250b17642868499db350be5a0bf98b0160c8f7c4a52189abac`.

## Candidate-specific wrapper

The b4-only wrapper is
`scripts/run_general_stage_b_b4_walltime_recovery.sh`, SHA-256
`770834281b39f36f82ef095d80caa6eaa9c1bb2c3101e152b678307d1d3b249a`.
It hard-binds b4, seed 42, the existing output root, checkpoint 10912, every
hash above, the immutable corrected launcher snapshot, all 695 source/config
files, and the exact original Python/package runtime. It requires an explicit
`SALLM_RECOVERY_BUNDLE`, preventing Slurm spool-relative lookup. The bundled
runtime verifier remains SHA-256
`cfe2eb873f53d83de2c2c296c6a172f74a8059ee4bd11f296785c444e9525b23`.

The intended immutable HEX bundle is
`/home/lmbanr001/masters/sallm_snapshots/pure-gdn-general-stage-b-b4-recovery-20260831-77083428`.
The public `--preflight-only` seam was developed red-to-green. Three focused
recovery-wrapper tests, shell syntax, shfmt, and diff checks pass.

## Scientific boundary

This is the one continuation authorized prospectively for the original b4
administrative timeout. It changes no candidate, recipe, seed, data,
evaluator, metric, checkpoint schedule, early-stopping rule, output root,
runtime, or hardware request. It must advance directly from checkpoint 10912
to step 10913. Once it produces model, data, training, validation, or evaluator
output, it is terminal and cannot be retried. No held-out or cross-candidate
metric informed this implementation.
