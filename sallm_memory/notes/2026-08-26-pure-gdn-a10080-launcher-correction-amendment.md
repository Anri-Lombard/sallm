# Pure-GDN A100-80GB launcher-correction amendment, 2026-08-26

Preregistered at 15:39 SAST before any corrected launcher executes and without
any hardware-gate result. The user explicitly authorized prioritising a move to
the A100-80GB pool. Held-out data remain inaccessible to this gate.

## Observed failure

- A100-40GB reference `1269243` exited `126:0` before model, data, or evaluator
  access because immutable nested launcher `run_pos_runtime_equivalence_gate.sh`
  had mode `0444` and the pinned wrapper attempted to execute it directly.
- No benchmark or comparison result exists. Preserve job `1269243`, its log,
  immutable sources, hashes, and absent result. Preserve cancelled, never-run
  candidate `1269245` and comparator `1269246`.

## Frozen correction

- Supersede only the 25 August no-rerun clause for this pre-science launcher
  failure. Run one new isolated gate chain.
- Change only the pinned wrapper's final invocation from direct execution to
  `exec bash "$repo/scripts/run_pos_runtime_equivalence_gate.sh"`. Do not
  change the nested launcher, runtime, model, adapter, validation subset,
  scorer, BF16 mode, manifests, thresholds, equality checks, resources, or
  fail-closed comparator.
- Write the corrected reference, candidate, manifests, and comparison to a new
  `2026-08-26-launcher-correction` result directory. Verify every sidecar before
  comparison.
- Run the A100-40GB reference first, then the A100-80GB candidate only after the
  reference succeeds, then the unchanged CPU comparator. Do not overlap A100
  families before the comparator passes.
- To prioritise the short gate, cancel and preserve running b7 seed-87
  `1270629` and never-started A100-40GB jobs `1270630/1270631` only after the
  corrected gate chain is ready for submission. This scheduling decision uses
  no validation metric. After a pass, resubmit the same required work from its
  immutable protocol on A100-80GB; after a failure, resubmit it on A100-40GB.
- If every original comparison check passes, activate the already-preregistered
  four-job A100-80GB capacity amendment. If any scientific or execution check
  fails, preserve the attempt and return to A100-40GB. Do not rerun again,
  relax a threshold, or consult held-out evidence.

## Execution — 15:45 SAST

- The frozen amendment and corrected pinned wrapper SHA-256 values are
  `73988566bf542eb78d5e750cde78e4d1de6722240b2c274fa54f939329ee3f4a`
  and `3602211536eecb76dfd4494710c03a640437166f181ddfccaf1a47e1d0ae3c81`.
  Both were deployed read-only in immutable protocol snapshot
  `a10080-launcher-correction-20260826-73988566` and verified remotely.
- B7 seed-87 `1270629` was cancelled and preserved after `02:48:42`; no
  validation result was inspected. Never-started a2 `1270630` and General a0
  `1270631` were cancelled at the same time.
- First queued replacement chain `1271028 -> 1271029 -> 1271030` and its four
  dependent jobs `1271031--1271034` were cancelled before execution when
  preflight found the already-preregistered explicit `SLURM_JOB_GRES`
  instrumentation missing from their submit commands. The isolated result
  directory remained entirely empty.
- Final reference `1271037` explicitly records `gpu:ampere:1`; candidate
  `1271038` explicitly records `gpu:ampere80:1` and depends on reference
  success; unchanged comparator `1271039` depends on candidate success.
  Slurm reserves the reference for `17:41:17 SAST`.
- Four scientifically required A100-80GB jobs are dependency-held after a
  successful comparator: b7 seed-87 `1271040`, a2 seed-87 `1271041`, General
  a0 `1271042`, and General a1 `1271043`. All use the immutable HPO snapshot,
  exact account/QOS/GRES, 24 hours, and eight CPUs. B7 alone uses a fresh
  isolated output suffix so its cancelled partial artifacts are not
  overwritten.
