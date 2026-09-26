# Pure-GDN General recovery bundle-path launch correction — 2026-08-31

Status: prospective implementation correction frozen at 20:45 SAST before any
replacement submission and without opening held-out or cross-candidate metrics.

## Observed failure

Exact b1/b2 recovery jobs `1278130/1278993` each ended `FAILED`/`1:0` at
elapsed `00:00:00`. Their complete logs contain only:

```text
ERROR: missing /var/spool/slurmd.spool/job<job-id>/verify_execution_runtime.py
```

Neither job reached model, data, training, validation, evaluator, manifest
creation, or checkpoint mutation. Preserve both jobs and logs. They are
pre-payload implementation failures under the frozen terminality rule, not
scientific recovery attempts.

## Root cause

The immutable wrapper derives its default recovery bundle with
`dirname "$0"`. Slurm executes a spooled copy of a submitted batch script, so
`$0` resolves inside `/var/spool/slurmd.spool/job<job-id>` rather than the
immutable recovery bundle. Direct no-launch preflights did not reproduce this
because they executed the wrapper at its immutable path.

## Frozen correction

Keep the reviewed wrapper byte-identical at SHA-256
`b3cd62c36dd368e4ee90f32324651f7de04dbcce8bc6914a260e1b48c7a730fa`.
For the corrected b1/b2 launches and the first b3 launch, submit it with the
single explicit Slurm export:

```text
SALLM_RECOVERY_BUNDLE=/home/lmbanr001/masters/sallm_snapshots/pure-gdn-general-stage-b-recovery-20260830-b3cd62c3
```

No recipe, seed, checkpoint, output root, archive, source snapshot, runtime,
evaluator, metric, resource request, or recovery order changes. The wrapper
must still pass its complete fail-closed checks on the compute node and each
trainer must jump from checkpoint 10912 to step 10913. Any model/data/training
payload makes that corrected launch terminal under the existing rule.

## Scientific boundary

This correction is determined solely by the two exact zero-second error logs
and Slurm batch-script spooling semantics. It does not authorize a fresh start,
an added candidate, a checkpoint change, a retry based on a metric, or any
held-out access.
