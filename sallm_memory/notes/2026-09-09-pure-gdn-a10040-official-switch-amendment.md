# Pure-GDN official-test A100-40GB switch amendment — 9 September 2026

Frozen prospectively at 10:12 SAST, before cancelling or replacing any pending
job and without opening any held-out result value.

## Reason

Both owned A100-80GB jobs are still unstarted and Slurm estimates a start on
11 September at 15:37 SAST. All four A100-80GB cards are occupied. In contrast,
`srvrocgpu010` has three physically free `gpu:ampere` A100-40GB cards. L40S is
more heavily occupied and has a long queue, so it has no demonstrated capacity
benefit and remains excluded.

The corrected cross-hardware gate `1271352` already ratified the exact
`gpu:ampere` / `gpu:ampere80` pair under the current pure-GDN runtime: exact
predictions, scores and metrics, zero numerical difference, and all manifest
checks passed. This amendment does not extend that result to `amperemk` or
L40S.

## Frozen switch

- Replace only never-started A100-80GB jobs `1323738` and `1325602`, after
  rechecking zero runtime and absent target outputs. Preserve their Slurm
  records and logs. Replace dependent never-started CPU seal `1323744` with a
  new dependency-bound seal; do not overwrite its record.
- Use only `nlpgroup/a100/nlpgroup`, `gpu:ampere:1`, one node, eight CPUs, and
  the existing wall times. Do not request `amperemk`.
- Run one fresh metric-free A100-40GB exact-checkpoint canary for corrected
  Base, using the unchanged v2 snapshot, gate manifest, checkpoint, runtime,
  checks and thresholds. The original target root is absent because job
  `1323738` never started, so the replacement keeps that already-frozen v2
  result root rather than introducing another scientific bundle.
- After that canary succeeds, submit a new CPU seal against it. The existing
  v2 protocol is unsealed and its output root is absent, so they remain the
  single authoritative bundle. Base stays raw, no adapter, BF16, and exactly
  16 packs covering the corrected 14 logical lanes.
- The General replacement may run only the four units never opened by timed-out
  job `1319959`: AfriMMLU, Belebele Southern Sotho, Belebele Xhosa and AfriHG
  Zulu. It must use the existing sealed General snapshot, configs, manifests,
  adapter, cache and output roots unchanged. POS is terminally missing and may
  not be retried.
- Never repeat a claimed official unit. A scientific payload cannot trigger a
  retry, resource change, recipe change or regrouping. Keep all concurrent GPU
  work on `gpu:ampere` until this wave ends, with at most four owned GPU jobs.

## Acceptance

The Base canary must complete `0:0` and preserve finite BF16
forward/backward, save/reload equality and deterministic generation before any
Base official job starts. Every official unit remains acceptable only after
terminal `0:0`, exact coverage, finite declared metrics and a structural
verification sidecar. Hardware changes cannot select or exclude a result.

## Execution

Immediately before the switch, jobs `1323738`, `1323744` and `1325602` were
still pending with `00:00:00` runtime. The Base gate/result roots, all four
General remainder outputs, and all four General claims were absent. The three
jobs were cancelled and preserved.

Replacement Base canary `1326110` ran on `srvrocgpu010` with
`gres/gpu:ampere=1` and completed `0:0` in 42 seconds. Its result SHA-256 is
`d1f13d836e18b77f2af86db70466b5c19818a52a0e4daad34d2b645b81f1248e`;
all finite-BF16, save/reload and deterministic-generation checks passed.
Dependent CPU seal `1326111` completed `0:0` in 2:49. The sealed Base
`bindings.sha256` digest is
`07aa04f2b4f6514fc2c7ba36344d98848a0509866893550a6a7f4646bef0cb80`.

General remainder `1326112` began on `srvrocgpu010` at 10:13:13 SAST and
passed all source/base/adapter/runtime bindings before starting AfriMMLU on
CUDA. Corrected Base groups `1326143` and `1326144` began on the same node at
10:17:23; both passed source/base bindings and started News and NER on CUDA.
Base group `1326145` is resource-pending for the next `gpu:ampere` card. These
four jobs are the complete owned GPU set: one General remainder and three Base
groups. No `ampere80`, `amperemk` or L40S job remains owned or schedulable.
