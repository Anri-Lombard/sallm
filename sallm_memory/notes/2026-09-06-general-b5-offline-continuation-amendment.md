# General b5 cache-only continuation

Date: 2026-09-06. Prospective before the b5 scientific continuation.

The user instructed "go launch, why waste time not having a gpu running?"
after being informed that the original b5 preservation/runtime preflight
passed but its loader still contacted GitHub despite cached input files.
This authorizes the narrow infrastructure repair and launch of the existing
same-trial b5 continuation, without relaxing its scientific checks.

The 31 August b5 archive, original manifest, checkpoint-10912 state hashes,
runtime, candidate/seed, recipe, evaluator, 13,640-step schedule, checkpoint
retention, original output root, and one-continuation limit remain fixed.
The original cancelled no-launch preflight and all old snapshots are preserved.
This amendment does not authorize a b3 replacement or a new trial.

## Isolated correction

New source root:
`/home/lmbanr001/masters/sallm_snapshots/general-b5-offline-20260906`.

The only change among the original 695 source/config files is the existing
cache-first AfriHG loader. No unrelated working-tree source was copied.
Its SHA-256 is
`12e0de1b6b4d851e9ee8426f153632388a0dad2c9073478ec2775bec523181c6`.
The new deployment manifest is
`19ce17d26bd4cc4fc1cd964230cf87d1c3a85632da16154ff7182c3e58c52795`.

The wrapper exports cache-only mode, an isolated train/validation-only AfriHG
cache, HF offline flags, W&B offline mode and UV offline mode. It first runs
the unchanged b5 preservation preflight against the original snapshot, then
verifies the separately named corrected manifest and offline-data preflight.
Only after those pass does it invoke the unchanged corrected-resume launcher
against the new source root and original checkpoint 10912. Execution and HPO
manifests remain resume-specific; the original root manifests are preserved.

Wrapper SHA-256:
`df38cc131c9ae46f96a23a4279b2933fe2d822a4ac1283df52936672abb08d8a`.
Data-preflight SHA-256:
`a6924de15d3274cc4896d4a6e177fbccd8432a48f6b2c94bb157f80864075966`.

## Data and preflight

Exactly four existing cached files are copied and made read-only. Unexpected
files, including cached validation.csv or test.csv, fail the cache inventory.
All four hashes are bound by the new deployment manifest and preflight:

- xho_train.csv: `07e2c01be2a5a187559636f4a7f04be761ce8ef497e9cd0f1a9f6bbd605a9e2b`
- xho_dev.csv: `e8ce73f843f1c997cdefeb5bac9c48e76ba43704e2dec299f3383d651b35cc58`
- zul_train.csv: `75fb3b05a4d2f9e9a99200c28bddf9486f37c3444e645b6968b58423c69482ba`
- zul_dev.csv: `b84feccc95717cbf442a5ad61d34d38de9c803c9bcc4a1925201186a050faaaf`

The focused existing cache-only test passed locally. Compute preflight
1312019 reproduces the original loader's unnecessary request with requests
blocked, then requires exact ordered train/validation record equality between
the original CSV-processing function and corrected loader. It also loads every
General raw train/validation component with requests blocked and requires the
frozen family validation counts. It records ordered record digests without
scoring any metric. No held-out outcome informs this correction or launch.

Before scientific submission require that preflight completes 0:0, freeze
its log hash, recheck absent final/resume outputs and active duplicates, then
submit once on nlpgroup80/a100/nlpgroup80, one A100-80GB, eight CPUs and
24 hours. Verify trainer continuation from 10912 rather than step zero.

## Preflight-only correction and final launch binding

Preflight 1312019 failed before training because its new check instantiated
the dataset configuration without resolving max_seq_length. Preserve that
job, log and first snapshot; it produced no continuation manifest or training.
The corrected preflight explicitly uses the frozen max_seq_length=2048,
rejects extra source files, and the wrapper clears PYTHONOPTIMIZE and pins
runtime/scratch inputs. None of these changes alter the training recipe.

The scientific launch uses the separate read-only root
`/home/lmbanr001/masters/sallm_snapshots/general-b5-offline-20260906-v2`.
Its final bindings supersede the first root for launch only:

- deployment manifest: `236a47dc6993851d299780161b2b517e55d5af61582ffc4b1c9143229c5692d1`
- wrapper: `e464335d8ba7fec101fb39867c4e2d860eff8501d5a3562c8c1de299687b4ade`
- data preflight: `fc8cf5b267d1c15237118cc25220feb06306808dcb8f73d737e4d7f4e948ed41`
- successful preflight log: `956d1ac2ca86d927e1b48992758b48ab19f8361fd4867668919b2ea30b11f0d0`

Preflight 1312049 completed 0:0 on A100-80GB in 20 seconds. It verified the
original checkpoint/archive/runtime, corrected source inventory and four
cache hashes; reproduced the old network dependency; proved exact ordered
AfriHG train/validation equality; and loaded all six raw General components
offline with the required validation counts. The wrapper and check hashes
above separately bind files outside the deployment manifest's source roots.
No scores or held-out outcomes were used. Fresh scheduler readback showed
no owned jobs and the b5 root had neither final adapter nor resume manifest.
