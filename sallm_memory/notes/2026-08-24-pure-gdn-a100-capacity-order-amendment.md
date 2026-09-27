# Pure-GDN A100 capacity-order amendment — 2026-08-24

## Frozen before dependency edits — 09:35 SAST

Slurm moved the eligible A100-80GB gate candidate `1258898` estimate from
`2026-08-25 21:47:49` to `2026-09-01 17:31:43 SAST`. All four approved
A100-80GB devices and all four approved A100-40GB devices are currently
occupied by other users. Waiting for the capacity-only gate before releasing
the already-submitted A100-40GB b6 candidate would now create avoidable idle
scientific time once A100-40GB capacity returns.

The prospective execution-only amendment is:

1. Hold `1258898` before changing any dependency.
2. Change existing unchanged b6 job `1258900` from `afterok:1258899` to
   `afterok:1258485`; b5 `1258485` is already scientifically terminal-valid.
3. Change `1258898` to `afterok:1258900`, then release it.
4. Keep comparator `1258899` unchanged at `afterok:1258898`.

This preserves strict single-family hardware ordering: b6 can use only
`gpu:ampere` A100-40GB, then the gate candidate can use only
`gpu:ampere80` A100-80GB, and the comparator runs only after the candidate.
There can be no owned A100-40GB/A100-80GB overlap. Candidate b6, seed, source
snapshot, registry, model, data, evaluator, checkpointing, output root,
resources, and selection rule remain unchanged. B6 was submitted before b5
metrics existed, so this order change cannot select from b5 or held-out
results. Held-out access remains zero.

Before b6 ends, b7 may be dependency-queued from the same immutable protocol
and inserted ahead of `1258898` only after a fresh absent-output and manifest
preflight. No result may be inspected to make that scheduling decision.

The frozen pre-edit note SHA-256 was
`e8dbf98e95a10048d240f6de867a59f5f983e59f24cb4feb415faaf57ef5c83d`.
That immutable pre-edit version contained a typographical `09:40` heading;
the heading above is corrected to the actual pre-edit `09:35` time.

## Execution — 09:37 SAST

- `1258898` was held before dependency edits. Slurm rejected
  `afterok:1258485` for b6 because the predecessor was already complete; no
  dependency changed and the held gate prevented any race.
- With the gate still held, b6 dependency was safely cleared, `1258898` was
  set to `afterok:1258900`, and the gate was released. Comparator `1258899`
  remains `afterok:1258898`.
- Post-edit state is exact: b6 `1258900` is `Resources`-pending and eligible
  on `gpu:ampere`; gate `1258898` is dependency-pending after b6 on
  `gpu:ampere80`; comparator `1258899` is dependency-pending after the gate.
  The b6 output root remains absent. No job started during the edit and no
  A100-family overlap occurred.

## B6 zero-second launcher correction and b7 queue — frozen 10:38 SAST

At `10:12:03 SAST`, b6 job `1258900` received an A100-40GB allocation but
failed `1:0` in zero seconds before model, data, or evaluator access. Its only
output is the 157-byte Slurm log stating that the obsolete mutable-repository
path `scripts/hpo_protocol.py` does not exist; SHA-256 is
`76ca9c0cb2bcd12791eb78d70bcf28212a9a98ec9887797dfefa39db8d8de94f`.
The candidate output and logging roots are both absent. The dependent hardware
gate therefore moved to `DependencyNeverSatisfied` without starting.

This is an execution-only launcher failure, not a candidate result. Before
resubmission, the immutable snapshot sidecar passed, all `694/694` source
hashes matched, the registry hash remained
`8fdd6ea5f192a1d2ef2267353162b9a640ccd09ae0d889cb5a999645198bb726`,
and the b6 and b7 output/logging roots were absent. The frozen correction is:

1. Submit unchanged b6 from the immutable snapshot launcher with the exact
   registry parameters and standard isolated b6 paths.
2. Submit unchanged b7 from the same launcher and registry, dependency-gated
   after successful b6, before any b6 result exists.
3. Hold gate `1258898`, replace its failed b6 dependency with successful b7,
   then release it. Keep comparator `1258899` after the gate.

This preserves the candidate order, keeps all selection validation-only,
prevents A100-family overlap, and removes a manual gap. No held-out or b6/b7
metric was inspected.

The complete frozen pre-submission note SHA-256 was
`e5c08c409c25f43d377469b0f6fcccb4aa20ec21b4a59c7906f4b8fb444ed63b`.

## Corrected execution — 10:40 SAST

- Corrected b6 is job `1262542`, exact A100-40GB `gpu:ampere:1`. It started
  at `10:37:56 SAST` on `srvrocgpu010`, verified all `694/694` immutable
  files, wrote execution-manifest SHA-256 `94a1587e...a3bb6b`, passed the
  fast-GDN gate, and loaded the canonical pure-GDN model cleanly.
- B7 is job `1262543`, the exact same immutable protocol on A100-40GB, and is
  dependency-pending after successful b6.
- Gate `1258898` was held before its invalid dependency was replaced, then
  released dependency-pending after successful b7. Comparator `1258899`
  remains dependency-pending after the gate.
- The three owned GPU jobs are therefore strictly serial:
  `1262542 -> 1262543 -> 1258898`; there is no A100-family overlap and no
  manual inter-candidate gap. Failed job `1258900` and its 157-byte log are
  preserved as execution provenance.

## Same-family b6/b7 concurrency amendment — frozen 12:37 SAST

Live capacity now has only two of four `gpu:ampere` devices allocated:
corrected b6 `1262542` and another user's job. B7 `1262543` remains unstarted,
with both output and logging roots absent. Stage-B candidates are independent
fixed registry trials; their numerical order is not a scientific selection
dependency. Keeping b7 behind b6 would therefore leave approved same-family
capacity idle for operational reasons.

Without inspecting a b6 metric or any held-out result, freeze this execution-
only change:

1. Hold A100-80GB gate `1258898` before changing dependencies.
2. Make the gate depend on successful completion of both b6 `1262542` and b7
   `1262543`.
3. Clear b7's b6 dependency, then release the gate.

This allows only the already-frozen b6 and b7 trials to overlap on identical
A100-40GB hardware. The gate cannot start until both finish successfully, so
owned A100-40GB and A100-80GB still cannot overlap. The owned GPU-job count
remains exactly three (running/eligible b6 and b7 plus the dependency-pending
gate), no L40S is used, and every scientific setting remains unchanged.

The complete frozen pre-edit note SHA-256 was
`5c4490b94ed25a7825d9806c1009f3199d29f41d5793633846dc4100bb25c7af`.

## Same-family concurrency execution — 12:36 SAST

- Gate `1258898` was held first and now depends on successful completion of
  both b6 `1262542` and b7 `1262543`; comparator `1258899` remains after the
  gate. The gate was released only after both dependencies were installed.
- B7's operational b6 dependency was cleared. It started at `12:35:59 SAST`
  on a second `srvrocgpu010` A100-40GB, wrote execution-manifest SHA-256
  `ab0a2d96...c45c79`, and verified all `694/694` immutable files.
- B6 and b7 now run concurrently on identical approved hardware. A100-80GB
  remains dependency-blocked, so no cross-family overlap occurred. The change
  used live capacity and absent-output state only, not a candidate metric.
