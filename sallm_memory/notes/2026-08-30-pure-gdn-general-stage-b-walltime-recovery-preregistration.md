# Pure-GDN General Stage-B wall-time recovery preregistration — 2026-08-30

Status: prospective only. No recovery job has been submitted.

## Reason

General Stage-B b1 and b2 jobs `1276432/1276433` reached the fixed 24-hour
limit on 30 August. Both ended before their next frozen validation boundary
and before writing a final adapter. Their complete checkpoint-10912 training
states and validation artifacts were created before the timeout and are
preserved. The timeout does not change either recipe, seed, checkpoint
schedule, early-stopping rule, or selection metric.

## Frozen recovery rule

After unchanged b7 has been submitted in the preregistered candidate order,
use later released A100-80GB slots for one exact same-trial resume of b1 and
then b2. Each recovery must:

- use the existing output root and exact checkpoint-10912;
- validate the adapter, optimizer, scheduler, RNG, and trainer state before
  submission;
- use the already-proven corrected resume propagation from immutable snapshot
  `uniform-adapter-hpo-general-a0-resume-correction-20260828-45fec06b`;
- write resume-specific execution-manifest and trial-record filenames;
- jump directly to step 10913, with any step-0 start treated as a failed run;
- keep the original recipe, seed 42, validation coverage assertions,
  early-stopping rule, and 24-hour A100-80GB resource request unchanged;
- preserve the original timeout jobs and never overwrite their logs or
  checkpoint-10912 states.

The corrected nested launcher SHA-256 is
`45fec06bf2ea71d4f69a43a686c88b9b8d895750d85b797ed104ac59271cceab`.
It was prospectively tested before a0 recovery `1276350` and then proved at
trainer level by that run's direct step-10912 resume. This recovery reuses
that verified mechanism and makes no new implementation change.

## Frozen checkpoint hashes

B1 checkpoint-10912:

- adapter: `29aa21d2237b40efffa70f719a364d92addd03042eb3b1dffe8d88a1d12819f0`
- optimizer: `8557158f6b2f9580089f6844f4a5b2cc0210cd9e30d5a39d1b69ed700f484f6e`
- scheduler: `58d820ba2be4b96b51823238bab995b391147e7c05fc0223f29f94d6a6473407`
- RNG: `0d4e900b49e99f73ab4c969265a4600f448e5236d37bed123a9baa3915e9013c`
- trainer state: `2b5d9b74c8c8ba71ac86bd79dd216edabb6e98316d299944c79fcab5bbabade0`

B2 checkpoint-10912:

- adapter: `cc2a37683387dfc2ebb4969ab8337ce24f42bdf372d568be628545b3f34aedcb`
- optimizer: `5a42ee1ef504c3a4e43990f04ad983142debf06691bde0f7ca1d2a6808bcaac8`
- scheduler: `8ad971f1950fc2d8cbce77d99f29c9806ee058be8ba844f45d74d9a63fed6707`
- RNG: `3180a03567b7f40c2f694e184313fd4ca7212fda830074f2e9bc3350b97907e5`
- trainer state: `fe862cfbfc8430edd220ec888a7b4ed3dab4151f26e26f0aa090d9341314c804`

## Scientific boundary

No held-out artifact, test metric, or cross-candidate ranking informed this
recovery. The recoveries exist only to finish the already-started frozen
trials after an administrative wall-time interruption. Any second recovery
attempt, fresh start, recipe change, or checkpoint change requires a new
prospective amendment.

## B3 extension — 13:30 SAST

B3 job `1276434` also reached the fixed 24-hour limit, at optimizer step
11991 of 13640, without a final adapter. It did not reach terminal validation;
the earlier inference that it had entered terminal validation was disproved by
the exact terminal log. Its complete checkpoint-10912 state is preserved.

After unchanged b7 has been submitted, use later released slots for the exact
same-trial recoveries in frozen b1, b2, then b3 order. B3 uses the same
validated resume mechanism and all original recipe, seed, validation,
checkpoint, early-stopping, hardware, and output-root settings. It must jump
directly to step 10913 and may be submitted only once.

B3 checkpoint-10912 hashes:

- adapter: `e591c605336ea0fbc431199b59344e768b3242a56ad0f5e5ebe8bd3a201adfb5`
- optimizer: `98c23a9b4f6debdfe49bc5fdd64ce0388d209354f470ce2957828fca1149edd3`
- scheduler: `2244542dfc73c540f56c4397f999c55d9557aadb704e404ac6d3861dcd996027`
- RNG: `47a82e7665a2fa0070b6758de26b3cb3469af4e8573bfae25bf22006f339fc26`
- trainer state: `968e989e31e8f9248957eb90b7adaff293aba12d3c3cfea06b19f50c66d11031`

No held-out, cross-candidate, or later-candidate evidence informed this
extension. B3 remains one interrupted trial, not an added candidate.

## 15:15 SAST implementation readiness

The candidate-generic recovery wrapper was implemented and tested before any
recovery slot opened. It accepts only b1, b2, or b3; binds each candidate to
the checkpoint-10912 hashes above; verifies the corrected nested launcher;
rejects an existing final adapter or recovery manifest; and propagates the
exact resume checkpoint into the unchanged registry-backed trial.

- Wrapper SHA-256:
  `b4984293b709c4e7cec6f98e288cf73923f362303f5ad2b2513b801f1ba9cdc8`
- Immutable HEX path:
  `/home/lmbanr001/masters/sallm_snapshots/pure-gdn-general-stage-b-recovery-20260830-b4984293/run_general_stage_b_walltime_recovery.sh`
- Local focused tests: 2 passed; shell syntax, shfmt, and diff checks passed.
- Actual HEX dry runs verified all five checkpoint hashes for b1, b2, and b3
  and printed the exact original recipes, output roots, and checkpoint-10912
  resume paths.

No recovery job has been submitted. B4--b7 still occupy all four owned
submission positions, so the frozen b1, b2, then b3 order remains gated.

## 15:25 SAST independent-review correction

Independent review found that wrapper
`b4984293b709c4e7cec6f98e288cf73923f362303f5ad2b2513b801f1ba9cdc8`
hashed the corrected nested launcher but not the outer launcher and helper
chain it actually executed. It was preserved but rejected before submission;
the earlier implementation-ready statement is superseded.

The corrected wrapper now verifies the known hashes of the deployment
manifest, its sidecar, and the manifest verifier, then uses that verified
program to check all 695 frozen source/config files before checking the five
candidate checkpoint files. It derives both the checked and executed output
root from the same `SCRATCH` value.

- Corrected wrapper SHA-256:
  `1e9d3e6cb5b7b4afe1cd98bf30c58bafc03add7b4becfbb63fad0326a6212ab2`
- Immutable HEX path:
  `/home/lmbanr001/masters/sallm_snapshots/pure-gdn-general-stage-b-recovery-20260830-1e9d3e6c/run_general_stage_b_walltime_recovery.sh`
- Focused tests: 2 passed after a red test required full-manifest
  verification; shell syntax, shfmt, and diff checks passed.
- Actual HEX dry runs for b1, b2, and b3 each verified all 695 frozen files,
  all five checkpoint-state hashes, and the exact output/resume binding.

Submission eligibility remains an external scheduler preflight, not a claim
of this execution wrapper: b7 job 1277552 must remain submitted, all four
owned positions must be recounted live, and recoveries must be submitted in
b1, b2, then b3 order. No recovery has been submitted.

## 15:39 SAST preservation and runtime correction

A second independent review correctly found that an in-place resume could let
Transformers `save_total_limit=1` prune the original checkpoint-10912 evidence,
and that the queued recovery did not fail closed on Python or package drift.
Wrapper `1e9d3e6...` was therefore blocked before submission and is superseded.

Each interrupted output root now has a separate read-only tar archive outside
the training root. The archives include the original execution manifest,
trial record, validation artifacts, retained checkpoints, and complete
checkpoint-10912 state:

- b1: `0c972a36ed854b27e2512350ce947d22581a145a648b9919b2959ef4df736c27`
  (`346388480` bytes);
- b2: `c82dc55f3ed836759015f68a9851318fc1bdd2e1584b74081cf0b45362705e96`
  (`97904640` bytes);
- b3: `a51c35ead0e3e794be9209728c12ed806e7471a18ad7d0334aeb5c5b88f3d918`
  (`245852160` bytes).

This prospective implementation correction supersedes the earlier literal
live-path preservation clause: the read-only archive is the authoritative
pre-recovery evidence. After every hash and runtime preflight passes, only the
unchanged frozen trainer may prune the live output root under its original
`save_total_limit=1`; the archive and original timeout log remain untouched.

The final wrapper requires the candidate-specific archive hash and read-only
mode before it can resume. It also invokes a separately hashed verifier that
requires exact equality with the original manifest for Python, executable,
platform, and all 179 installed package versions.

- Final wrapper SHA-256:
  `39bc3ebc1b50f17d62c28484c7a2a40f337ae3cee9c1dcca18bc844e1a769e8c`
- Runtime verifier SHA-256:
  `cfe2eb873f53d83de2c2c296c6a172f74a8059ee4bd11f296785c444e9525b23`
- Immutable HEX bundle:
  `/home/lmbanr001/masters/sallm_snapshots/pure-gdn-general-stage-b-recovery-20260830-39bc3ebc`

The wrapper has an explicit no-launch preflight mode. Durable read-only HEX
preflight artifacts for b1/b2/b3 have SHA-256 values
`cd7f58ee3588473dbf1df6f425175b3d6e4db00b326978a099c96bcbb90f89cc`,
`fb14f1c32a107d0f442baa9d7aae236b2baa84b08c46c7bbddd5a9038ca2096c`,
and `1e0259581976bcb7884ece6354bdec0edc0981035eb945c7ca8af3b9139e0629`.
Each verified all 695 frozen files, exact runtime equality, the preserved
archive, all five checkpoint hashes, and output/resume binding. Three focused
tests, 128 full CPU tests, shell syntax, shfmt, and Ruff pass. No recovery job
has been submitted; the external b1--b2--b3 eligibility gate remains.

## 15:44 SAST original-manifest binding correction

Final review found one remaining mutable input: the runtime verifier read each
live `execution_manifest.json` without first binding its hash. Wrapper
`39bc3ebc...` was therefore superseded before submission. The candidate cases
now require the original manifest hashes before comparing the runtime:

- b1: `ecb840c3aa6ba93a69daa8083b4550f4e4aafbaf1aa06fbde4faf175b96bb404`;
- b2: `32371bdd0c4f9a7b571a70b369cc6f1dccb9ac2665be4489b2d7be05dcec274e`;
- b3: `1344f1c7d2ea38b1f90ffc0986e49e8296eaa44f8f3668b6029fea1ea4255bdf`.

The only eligible wrapper is now SHA-256
`b3cd62c36dd368e4ee90f32324651f7de04dbcce8bc6914a260e1b48c7a730fa`
in immutable bundle
`/home/lmbanr001/masters/sallm_snapshots/pure-gdn-general-stage-b-recovery-20260830-b3cd62c3`.
Its durable no-launch preflights are:

- `/scratch/lmbanr001/masters/sallm/recovery_archives/general-stage-b/preflight-b3cd62c3-b1.txt`
- `/scratch/lmbanr001/masters/sallm/recovery_archives/general-stage-b/preflight-b3cd62c3-b2.txt`
- `/scratch/lmbanr001/masters/sallm/recovery_archives/general-stage-b/preflight-b3cd62c3-b3.txt`

Their SHA-256 values remain `cd7f58ee...89cc`, `fb14f1c3...096c`, and
`1e025958...0629` because the successful output text is unchanged. A red test
proved a manifest mismatch was previously accepted; it now fails closed.
Three focused and 128 full CPU tests pass. No recovery job has been submitted.

## Uniform terminality clarification — 16:38 SAST

Before any recovery submission, the terminal rule is tightened uniformly for
b1 through b7. A same-trial recovery that produces any model, data, training,
validation, or evaluator output is the only scientific recovery and is
terminal. Only an implementation failure before any such payload exists may
receive a separately hashed prospective execution correction, never a second
scientific recovery and never because of a metric. This supersedes earlier
language that a later amendment could authorize an unrestricted second attempt.
