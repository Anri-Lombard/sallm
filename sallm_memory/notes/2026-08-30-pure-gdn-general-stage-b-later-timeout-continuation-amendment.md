# Pure-GDN General Stage-B later-timeout continuation amendment — 2026-08-30

Status: prospective, frozen at 16:30 SAST before any terminal outcome for b4,
b5, b6, or b7. No recovery under this amendment has been submitted.

## Reason

General b1, b2, and b3 each reached the fixed 24-hour scheduler limit with a
complete training-state checkpoint but no final adapter. B4 and b5 are still
running, and b6 and b7 are still pending. The same administrative interruption
is therefore possible for the remaining candidates. This rule is recorded
uniformly now rather than amended after another outcome is known.

## Frozen rule

If an original b4, b5, b6, or b7 job ends `TIMEOUT` at the fixed 24-hour limit
before writing a final adapter, it may receive exactly one same-trial
continuation. The continuation must:

- keep the candidate, seed 42, recipe, evaluator, data, output root, checkpoint
  schedule, early-stopping rule, runtime, A100-80GB request, and selection
  metric unchanged;
- resume from the numerically latest complete checkpoint in the already-frozen
  checkpoint schedule, chosen only by step number and complete adapter,
  optimizer, scheduler, RNG, and trainer state, never by metric value;
- fail closed if no complete scheduled checkpoint exists, if a final adapter
  exists, or if the terminal state was anything other than the administrative
  24-hour timeout;
- create and hash a read-only archive outside the live output root, bind the
  original execution manifest, frozen source manifest, exact runtime, and all
  resume-checkpoint state files before submission, using the same preservation
  standard as the reviewed b1--b3 recovery path;
- use a candidate-specific immutable wrapper and no-launch preflight produced
  only after the timeout supplies the exact archive, manifest, and checkpoint
  hashes; and
- preserve the original job, log, artifacts, and archive. A failed continuation
  is terminal once it produces any model, data, training, validation, or
  evaluator output. An execution-only implementation failure before any such
  payload exists may receive a separately hashed prospective correction, but
  never a second scientific continuation and never because of a metric value.

Existing b1, b2, and b3 recoveries remain first in that frozen order. Any
eligible b4--b7 continuations follow in candidate order, skipping candidates
that completed normally. The four-owned-job cap and absent-output/no-duplicate
checks remain unchanged.

The reviewed wrapper
`b3cd62c36dd368e4ee90f32324651f7de04dbcce8bc6914a260e1b48c7a730fa`
remains eligible only for b1, b2, and b3. This amendment does not authorize it
for b4--b7 and does not authorize a fresh start.

## Scientific boundary

No held-out result, cross-candidate comparison, or b4--b7 terminal metric
informed this rule. Interim validation remains usable only for the unchanged
within-run checkpoint-retention rule. This amendment adds no candidate and
changes no selection evidence.

## Independent-review tightening — 16:35 SAST

The original SHA-256 `b41031db...0c58e` is superseded before any use. The rule
above now distinguishes a pre-payload implementation failure from a scientific
continuation and explicitly forbids a second continuation after any scientific
output. No job or preflight used the superseded wording.
