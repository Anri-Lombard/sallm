# Pure-GDN General confirmation wall-time amendment - 2026-08-31

Status: frozen prospectively at 23:15 SAST, before any General Stage-C
confirmation was submitted or evaluated. The user explicitly authorized this
change. No held-out metric, candidate score, or interim validation magnitude
was used.

## Decision

The four not-yet-submitted General Stage-C confirmations, comprising the
frozen top two candidates at seeds 13 and 87, will request a 36-hour Slurm
ceiling. Submission must use the native scheduler override
`sbatch --time=36:00:00` with the unchanged immutable scientific launcher.

The remaining resource contract stays fixed:

- account `nlpgroup80`;
- partition `a100`;
- QOS `nlpgroup80`;
- one `gpu:ampere80` A100-80GB;
- eight CPUs;
- canonical home working directory;
- at most four active owned HEX submissions; and
- no overlap with another pure-GDN GPU family.

The 36 hours is a ceiling, not a target. A confirmation exits as soon as its
unchanged training, validation, retention, and final-adapter work completes.

## Reason

General has a fixed 13,640-step schedule. Current operational evidence is
about 6.8 seconds per optimizer step, or about 25.8 hours of training before
the exact 22,167-row validation callbacks and finalization. A 24-hour ceiling
therefore forces otherwise healthy runs into checkpoint continuations. The
36-hour ceiling covers the observed full-run envelope while remaining below
UCT's 48-hour maximum.

This decision uses only scheduler and runtime evidence. It does not change
the search space, candidate order, seed policy, checkpoint rule, early
stopping, validation metric, data, prompts, model, optimizer, decoding, or
failure semantics.

## Scope boundary

Current General jobs remain unchanged under their original 24-hour requests:

- b1 continuation `1279470`;
- b2 continuation `1279471`;
- b4 continuation `1279652`; and
- b5 no-launch preflight `1279656`.

The amendment does not authorize another b3 attempt, resolve the blocked
all-11 ranking, alter News/SIB/Intent budgets, or touch held-out evaluation.
It applies only after a valid General ranking freezes the two confirmation
candidates.

## Submission and verification gates

Before each confirmation submission:

1. verify the frozen ranking artifact and candidate order;
2. verify the output, log, manifest, and active-job duplicate are absent;
3. hash and freeze the candidate-specific execution bundle;
4. submit once with `--time=36:00:00`; and
5. read back the Slurm job and require a `1-12:00:00` time limit before
   treating the launch as valid.

Every scientific artifact, coverage check, sidecar, and exact
retained-to-final roundtrip remains mandatory.
