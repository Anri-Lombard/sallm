# Pure-GDN News/SIB/Intent budget-limited close-out preregistration — 2026-08-30

Preregistered at 21:29 SAST after the user approved a reduced validation-only
budget and before any corrected post-BOS News, SIB, or Intent adapter run or
metric existed. A quota-first HEX audit found no new corrected output root and
no submitted job for these families. No held-out split or Sheet E/F/G value was
loaded, inspected, or scored.

## Scope and disclosure

- Apply this reduction uniformly to News, SIB, and Intent only. General and all
  already frozen families keep their existing protocols and evidence.
- This supersedes the 11-candidate plus four-confirmation budget for these three
  families only. It produces a transparent budget-limited coarse search, not
  the enhanced 11-candidate search.
- Preserve all historical pre-BOS artifacts as provenance. They remain
  scientifically ineligible and cannot select a recipe, checkpoint, seed, or
  retry.

## Frozen candidate and confirmation budget

For each of News, SIB, and Intent:

1. Run exactly the corrected post-BOS Stage-A candidates `a0`, `a1`, and `a2`
   once at selection seed `42`, using their immutable recipes from
   `src/conf/hpo/pure_gdn_enhanced_v1.json`.
2. Require exact declared validation coverage: News `3,095`, SIB `2,970`, and
   Intent `4,155` prompt-expanded rows. Require finite validation output,
   sidecar verification, registry/config reconciliation, and exact
   retained-to-final adapter equality.
3. Rank the three terminal-valid seed-42 candidates by corrected validation-only
   `all_macro_f1`, greater is better. Take the leading two candidates.
4. Run exactly one additional validation-only replication for each finalist at
   seed `13`. Seed `87` and Stage-B candidates `b0` through `b7` are omitted
   uniformly and may not be launched later because a result is disappointing,
   close, or ambiguous.
5. Select the family winner by the arithmetic mean of each finalist's seed-42
   and seed-13 `all_macro_f1`. Exact mean ties prefer lower LoRA rank, then
   lower learning rate, then the lexicographically earlier candidate ID.
6. Freeze the winner's seed-42 retained checkpoint for later one-time official
   held-out evaluation. Each seed selects its checkpoint only with the existing
   within-run validation metric, patience, threshold, and checkpoint tie rule.

This is five GPU runs per family and fifteen across the three families, instead
of forty-five under the enhanced protocol. The paper, result manifest, and
table notes must disclose the reduced budget and report all candidates,
failures, seed values, arithmetic means, and runtime/GPU hours.

## Execution and isolation

- Continue pure-GDN on A100-80GB only under
  `nlpgroup80/a100/nlpgroup80`, `gpu:ampere80:1`, 24 hours, eight CPUs, and
  `--chdir=$HOME/masters/sallm`, with at most four concurrently running owned
  jobs. Do not overlap A100-40GB, L40S, or Kombuys pure-GDN work.
- Current General Stage-B candidates, exact same-trial recoveries, and four
  General robustness confirmations retain priority. Once eligible, the three
  family `a0` runs may act as runtime and structural canaries, but their metric
  magnitudes cannot alter submission order, budget, retry, or family scope.
- Use absent-output/no-duplicate checks and a new immutable execution manifest
  before every submission. Infrastructure-only failures may receive only an
  unchanged recovery that preserves provenance and has not consulted a metric.
- Monolingual training, official held-out adapter tests, Sheet E/F/G, and
  publication remain blocked until all eight family winners are frozen and the
  consolidated freeze manifest verifies.
