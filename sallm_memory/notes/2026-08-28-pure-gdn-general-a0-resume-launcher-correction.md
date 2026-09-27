# Pure-GDN General a0 resume-launcher failure — 2026-08-28

Status: diagnosed and contained before a validation boundary. The one
authorized corrected replacement is now running as job `1276350`.

## Replacement authorization — 22:29 SAST

The user explicitly authorized one corrected a0 implementation-recovery
submission. The authorized replacement must resume the unchanged complete
checkpoint-10912 once on the ratified A100-80GB family. It may change only the
launcher behavior already tested prospectively: validate and propagate the
resume checkpoint and write resume-specific manifest/trial filenames. Before
submission, create and verify a new immutable source/hash manifest, recheck the
five checkpoint hashes, require the final adapter and active duplicate to be
absent, and preserve jobs `1271042` and `1275241` and their artifacts. This
authorization does not permit a fresh start, another recovery attempt, held-out
access, or any recipe or selection change.

The fail-closed preflight passed and job `1276350` started at 22:33:30 SAST.
Immutable launcher, batch, and deployment-manifest SHA-256 values are
`45fec06bf2ea71d4f69a43a686c88b9b8d895750d85b797ed104ac59271cceab`,
`7f3d06447b72ae06882d0dbe11e6cde452eb09d804eaebb94978eedb05ae0c2f`,
and `8b6a744bc43d402aedccb44e6d62ef5f34144b95c7e871bfd7e2be823d0d9bb6`.
The resume-specific execution manifest records the exact checkpoint path and
matches its sidecar at SHA-256
`c47a2cf9e68018d95524bd4b63d074dbb81a6f85036aaaa821749262e97879bd`.
Trainer-level resume is confirmed: the log records the exact checkpoint path,
then jumps directly to step `10913/13640` and continues past `10919`. There is
no step-0 restart or fault marker.

## Observed failure

- Authorized recovery job `1275241` started on A100-80GB at 11:39:37 SAST.
  Its log printed `resume_from_checkpoint=None` and began at optimizer step 0,
  despite the Slurm wrapper exporting the intended checkpoint-10912 path.
- The job was cancelled at 11:45:26 after 5:49, before any checkpoint or
  validation metric was produced. Preserve the job and log as failure
  provenance.
- The invalid start overwrote the root `execution_manifest.json`, its sidecar,
  and `hpo_trial.json` at 11:39:39. It did not modify the checkpoint-10912
  state. The five frozen hashes remain exactly:
  - adapter: `4b4cfad3fb1623dd8e3dee86ee2f886115ae2204d962c206d1ed0818e9818151`
  - optimizer: `b6d542bec628c82406bc2cf7b5ad7e7c8926e6e2a9156c8fc69350c2401d8088`
  - scheduler: `fbc66d5d0b0f0c06cf7d343921624ddda9250d0e2fd4cd0dbf87f8eea88c85aa`
  - RNG: `0e8011af71b30b3d6ccfc2254349da97f7f40efd8a480228ecb2abc78b3229a9`
  - trainer state: `ab85f2c2a4053a3d3fd92535ae4e8987044db2765086a54cced615edf95fb5a3`

## Proven cause

The immutable `20260811-6aabf717` trial launcher has SHA-256
`eeb88f17ff5d2cb4a2d4b16851ab0ddbc1c2c667f3cd4d34213d880dc88eaa53`
and never reads `SALLM_HPO_RESUME_FROM_CHECKPOINT` or passes a Hydra resume
override. The recovery preregistration's claim that this frozen launcher
already supported the variable was incorrect. The Slurm export was present;
the nested launcher ignored it.

## Prospective correction prepared, not submitted

The current local trial launcher differs from the frozen launcher only by the
resume-path implementation: it validates a numbered checkpoint inside the
trial output directory, requires adapter/optimizer/scheduler/RNG/trainer state,
passes `+training.resume_from_checkpoint`, and writes resume-specific manifest
and trial-record filenames rather than overwriting the originals. Its SHA-256
is `45fec06bf2ea71d4f69a43a686c88b9b8d895750d85b797ed104ac59271cceab`.

Local dry-run checks prove that the valid complete checkpoint is propagated
and that a missing optimizer state fails closed. This overlay must receive a
new immutable deployment/hash manifest and explicit scientific authorization
before any single corrected replacement recovery. Do not restart from scratch,
reuse job `1275241`, or submit another recovery under the existing instruction.
