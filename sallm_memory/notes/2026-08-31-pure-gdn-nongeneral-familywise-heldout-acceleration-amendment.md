# Pure-GDN non-General family-wise held-out acceleration amendment

Frozen at 23:27 SAST on 31 August 2026, before any current pure-GDN adapter
official held-out result was opened or scored. The user explicitly prioritized
complete Monolingual and Multilingual results over General because General is
the remaining long pole.

This amendment changes execution order and the held-out release barrier only.
It does not change any candidate, seed, training recipe, validation metric,
checkpoint rule, evaluator, test split, or reporting metric. It prospectively
supersedes the global `8/8` ordering barrier in the 30 August global-freeze
reconciliation, Monolingual recipe-authority clarification, budget-limited
News/SIB/Intent close-out, and results audit. Their recipe, provenance,
coverage, and one-time-test requirements remain binding.

## General priority change

- Let the already running exact General b1, b2, and b4 continuations finish and
  preserve their artifacts.
- Cancel the pending b5 no-launch preflight before it starts. It is an
  infrastructure preflight, not a scientific trial, and has produced no model,
  data, validation, evaluator, checkpoint, or result artifact.
- Submit no new General candidate, recovery, ranking, confirmation, Mono, or
  held-out job until every non-General family-wise result described below has
  verified. Preserve General as an incomplete separately disclosed workstream.

## Family-wise release rule

Each non-General family may advance independently. A family becomes eligible
for its one-time official held-out evaluation only after:

1. its corrected post-BOS Multilingual winner and retained checkpoint are
   frozen under that family's existing validation-only protocol;
2. every applicable Monolingual adapter for that family is trained and its
   checkpoint is frozen using validation only, with the existing fixed Mono
   recipe and the frozen family learning rate;
3. manifests, hashes, declared coverage, sidecars, and exact
   retained-checkpoint-to-final-adapter equality verify; and
4. the exact frozen pure-GDN checkpoint passes the metric-free A100-80GB CUDA
   runtime gate: BF16 forward/backward, save/reload equality, and deterministic
   greedy generation under the corrected runtime.

Then, and only then, run that family's applicable official held-out arms once.
Verify their coverage, hashes, artifacts, notes, and exact Sheet readback before
filling only that family's eligible E/F/G cells. General cells remain blank.
T2X reuses its frozen Xhosa winner as its single Mono arm and has no duplicate
Multilingual arm.

## Fixed execution order

The non-General queue is now the priority queue:

1. run the metric-free exact-checkpoint CUDA gate;
2. release T2X for its one official held-out arm after the gate passes;
3. train and validation-freeze the eight missing Mono adapters for already
   frozen NER (3), POS (3), and AfriHG (2), then release each complete family;
4. run the already frozen reduced News, SIB, and Intent protocols, five jobs per
   family: a0/a1/a2 at seed 42 and seed-13 confirmations for the two finalists;
5. after each of those family winners freezes, train its News (2), SIB (6), or
   Intent (4) Mono adapters and release that family independently.

Use up to four A100-80GB jobs concurrently under
`nlpgroup80/a100/nlpgroup80`, `gpu:ampere80:1`, 24 hours, eight CPUs, and
`--chdir=$HOME/masters/sallm`. Keep candidate-major order across News, SIB,
and Intent. Fill a released slot only with the next already required job after
absent-output and no-duplicate checks. Never run pure-GDN on A100-40GB, L40S,
or Kombuys.

## Held-out firewall

Official test results are report-only. They may not select, correct, retry,
prompt, stop, reorder, or otherwise influence any unfinished HPO candidate,
seed, checkpoint, Mono adapter, family, evaluator, or General decision. A
surprising or weak official result is reported and investigated without rerun
or recipe change. No family receives a second official test pass.

The resulting paper table must disclose that non-General families were tested
and released family by family before General completed, under this prospective
metric-independent amendment. Validation/HPO values remain recipe-selection
evidence and must never be compared as official test scores.
