# Pure-GDN corrected A100 gate queue clarification, 2026-08-25

Preregistered at 21:51 SAST before submitting the corrected A100-80GB gate
candidate and before any corrected gate result.

The four-job association limit permits the dependency-held candidate to be
queued now as the fourth owned GPU job. This is a scheduling clarification,
not permission for early execution or A100-family overlap.

The candidate must depend on terminal jobs `1267877` and `1267878` and on
successful A100-40GB reference `1269243`. Slurm therefore cannot allocate its
`gpu:ampere80:1` until both current A100-40GB confirmations have ended and the
reference has completed successfully. All runtime, source, model, adapter,
validation rows, metrics, thresholds, and fail-closed rules remain unchanged.

No A100-80GB HPO may start until the subsequent unchanged comparator passes.
