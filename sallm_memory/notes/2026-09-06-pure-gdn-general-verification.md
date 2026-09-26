# General verification after access restoration

Date: 2026-09-06

Normal relay SSH works again. Every HEX access began with purequota. Home is
89.5% used and scratch 51.2%. The owned queue was empty before this pass.
No held-out scores were read, no candidate was ranked, and no Sheet changed.

## Completed b1, b2 and b4

CPU job `1311939` completed 0:0 in six seconds using the existing
`verify_adapter_roundtrip.sbatch`, SHA-256
`a138101a0be2dbb856c07bc2d5d5e59fd31436c2c30de4788287bf5ba0d75184`.
It proved exact retained-to-final equality for b1 checkpoint 8184, b2
checkpoint 13640, and b4 checkpoint 8184. Each has 424 state keys; b1 has
76,410,112 values and b2/b4 each have 69,438,784 values.

All five scheduled validation artifacts per candidate passed their SHA-256
sidecars, the equal-family protocol check, exact 22,167 processed rows,
six-family raw/processed counts, and AfriHG Xhosa/Zulu counts. Original and
resume execution-manifest sidecars passed. Retained trainer states bind the
same checkpoint steps used by the roundtrip verifier. These checks complete
verification of the already completed continuations, not new training runs.

Terminal validation artifact SHA-256 values:

- b1: `c2337fa940ee4ff2df302a8edbc3cf64d41405673c8b8d022db32fbf20d597f4`
- b2: `2225522ab5e57d3c82441fe05a5bbe5ae542289609d752a606871e29062eb11d`
- b4: `475e24c6031783846a91834a1cba98b9d49a3edb9a18f5a9506b7c3953b5393b`

Verifier log:
`/scratch/lmbanr001/masters/sallm/logs/jobs/gdn-general-b124-verify-20260906.out`,
SHA-256 `80bdfef6e00f4098815400a3a540913f65a61378307ebb89cad10d3761f52647`.

## B5 preservation preflight

The previous no-launch preflight 1279656 was cancelled without running.
Fresh no-launch preflight `1311940` completed 0:0 in three seconds on
srvrocgpu011 with the frozen A100-80GB/eight-CPU contract and ten-minute
preflight limit. The unchanged b5 wrapper verified all 695 source/config
files, exact runtime, immutable archive, original manifest, five checkpoint
state hashes, and absent final/recovery output. No scientific continuation
or resume manifest was created.

Preflight log:
`/scratch/lmbanr001/masters/sallm/logs/jobs/gdn-general-b5-preflight-20260906.out`,
SHA-256 `743051d736b52dce6a43e604326ee6b718a592b508d68f468961e8fe20f5ed91`.

This is not sufficient offline-data preflight. Inspection of the immutable
snapshot's `src/main/sallm/data/afrihg.py` found an unconditional
`session.get` before the cached-file existence check, for every split. Thus
existing cached CSVs and HF offline flags alone do not remove the network
dependency that failed b3. Training was not launched with that known gap.

Next work is the prospective offline infrastructure correction and exact
data/state/runtime binding recorded by the 4 September resumption authority,
including a valid offline-data preflight for b5/b6 before continuation.
B3's failed original remains terminal and preserved. B6 still needs its
candidate-specific archived and hashed continuation bundle. Do not change
recipes, use test metrics, restart from scratch, or rank the incomplete set.

The two verification/preflight jobs are terminal; no owned training job is
running at the end of this pass. Base provenance reconciliation is unchanged.
