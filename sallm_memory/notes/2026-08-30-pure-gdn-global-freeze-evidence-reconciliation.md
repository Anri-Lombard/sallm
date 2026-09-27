# Pure-GDN global-freeze evidence reconciliation — 2026-08-30

Recorded at 13:50 SAST after a read-only local and HEX audit. No held-out or
Sheet E/F/G value was opened, and no running job was changed.

## Correction

The repeated `7/8` global-freeze shorthand is not supported by the surviving
post-BOS evidence and is superseded by this audit. The 9 August prompt-contract
correction requires all eight Multilingual HPO families to use the corrected
BOS/chat contract. The 11 August enhanced protocol then requires 11 seed-42
candidates and four top-two confirmations per family, except where a later
prospective amendment explicitly reduces a family uniformly.

The defensible family state is therefore `4/8` frozen:

| Family | State | Basis |
| --- | --- | --- |
| T2X | frozen | full 11-candidate and three-seed confirmation ranking |
| NER | frozen | full 11-candidate and three-seed confirmation ranking |
| POS | frozen, budget-limited | prospective three-candidate/two-seed close-out |
| AfriHG | frozen after reconciliation | full 11-candidate program; original three-seed-mean rule still selects b7 |
| News | not started post-BOS | `0/11` candidates and `0/4` confirmations |
| SIB | not started post-BOS | `0/11` candidates and `0/4` confirmations |
| Intent | not started post-BOS | `0/11` candidates and `0/4` confirmations |
| General | incomplete | Stage A complete; Stage B and confirmations incomplete |

Every earlier `7/8` statement remains provenance for the operational history,
but may not authorize Monolingual training, official held-out evaluation, Sheet
publication, or a global winner manifest.

## Reconciled evidence

- POS remains a transparent reduced-budget result under the prospectively
  frozen 16 August close-out. The new evidence artifact merely packages the
  already frozen two-seed values and selection:
  `2026-08-30-pure-gdn-pos-budget-closeout-ranking.json`, SHA-256
  `e4a9032e98c5cfcc80ed1d4b5a8d51862707495cc27b7353d93fa5da26ff9e95`.
- The 27 August AfriHG artifact incorrectly made seed 42 decisive, contrary to
  the 11 August three-seed-mean rule. Applying that original rule to the same
  already verified values gives b7 `25.24535303429672` versus a2
  `25.030206863181398`; b7 remains the winner. The reconciled ranking is
  `2026-08-30-pure-gdn-afrihg-confirmation-ranking-reconciliation.json`,
  SHA-256
  `dcf5a9202b7dbabef28beb69c56521a7d4ba52d770e73dcdba70faf69c5d6f1b`.
  No candidate, seed, checkpoint, metric, or adapter changed.

## Missing-family audit

HEX has no post-BOS News, SIB, or Intent adapter-HPO output, manifest, job, or
ranking artifact. Their corrected roots are absent. Post-9-August accounting
contains corrected base-evaluation jobs only; the historical adapter jobs are
pre-BOS provenance and cannot count toward freeze.

Required validation coverage remains News `3,095`, SIB `2,970`, and Intent
`4,155` prompt-expanded rows. Each family must run a0--a2 and b0--b7 at seed
42, rank all 11 using its corrected validation-only `all_macro_f1`, and run
the frozen top two at seeds 13 and 87. The fixed enhanced-registry SHA-256 is
`8fdd6ea5f192a1d2ef2267353162b9a640ccd09ae0d889cb5a999645198bb726`.

## Prospective execution order

Current General b4--b7 submissions and the already preregistered exact b1,
b2, then b3 same-trial recoveries keep priority. After those slots release:

1. submit unchanged a0 once for News, SIB, and Intent as real candidates and
   structural canaries;
2. require immutable source/registry/prompt hashes, absent-output and
   no-duplicate checks, exact declared coverage, finite validation output, and
   retained-to-final adapter equality; metric magnitude cannot release, stop,
   retry, or reorder work;
3. fill eligible slots in fixed candidate-major order across News, SIB, then
   Intent for a1, a2, and b0--b7;
4. after all 11 candidates in a family verify, create its hashed seed-42
   ranking and submit exactly four fixed confirmations; and
5. create one consolidated eight-family freeze manifest before any Monolingual
   adapter is trained.

The prior 5--6 September complete-table target is no longer evidence-based.
A replacement ETA must use observed post-BOS classification runtime after the
three a0 canaries, not the invalid pre-BOS result values.
