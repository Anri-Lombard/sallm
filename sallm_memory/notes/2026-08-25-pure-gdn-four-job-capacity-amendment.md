# Pure-GDN four-job A100-80GB capacity amendment, 2026-08-25

Preregistered at 21:49 SAST before any fourth GPU job or corrected A100-80GB
gate result. The user explicitly authorized trying four concurrent jobs to
accelerate the validation-only program.

Live read-only Slurm accounting shows that the existing `nlpgroup80`
association on partition `a100` already has `MaxJobs=4`, while QOS
`nlpgroup80` permits per-user `cpu=56` and `gres/gpu:ampere80=4`. All four
A100-80GB devices were idle. No temporary permission or limit change is needed
to request four one-GPU jobs under the current configuration.

This changes capacity only:

- The corrected sequential A100 equivalence gate must still pass before any
  A100-80GB HPO or confirmation result is eligible.
- After a pass, at most four unchanged validation HPO or confirmation jobs may
  run concurrently on `gpu:ampere80:1`, using exact `nlpgroup80/a100/nlpgroup80`,
  24 hours, eight CPUs, and `--chdir=$HOME/masters/sallm`.
- Recipes, candidates, seeds, prompts, tokenization, metrics, checkpoint rules,
  confirmations, coverage checks, and held-out boundaries remain unchanged.
- Never add a candidate merely to occupy capacity. Submit only preregistered,
  scientifically required work after absent-output and no-duplicate checks.
- If administrators request capacity back or Slurm lowers the limit, allow the
  queue to reduce naturally; do not alter scientific selection or cancel a
  valid running job unless explicitly directed.

The first eligible four-job wave after a gate pass is b7 seed-87, a2 seed-87,
General Stage-A a0 seed-42, and the next preregistered General Stage-A candidate
whose dependencies and output preflight are satisfied.
