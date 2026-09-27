# Pure-GDN POS family-wise official-test freeze

Frozen prospectively on 2 September 2026 after all three POS Monolingual
adapters became terminal-valid and before any current-adapter POS official
held-out row was opened.

The frozen Multilingual winner is Stage-A a2 seed 42 at LR `1.5e-4`; its
budget-limited validation-only two-seed ranking is already hashed. The frozen
Mono adapters are Tswana checkpoint 570, Xhosa checkpoint 752, and Zulu
checkpoint 570. Each final adapter exactly matches its retained checkpoint
across `424` keys and `71,762,560` values. Every validation sidecar covers the
declared `750` rows and its SHA-256 verifies.

The one-time adapter bundle contains six fixed arms: Multilingual and
Monolingual adapters for Tswana, Xhosa, and Zulu. It reuses the existing
official MasakhaPOS task packs, four prompts per language, with exact
prompt-expanded coverage `2,408/2,404/2,404`. The already finalized Base
official artifact is not rerun.

All twelve official task YAMLs bind both train and test sources to MasakhaPOS
revision `376f4161f0425584d4bd7664122b56fa026926d3`; no `raw/main` source remains.

The existing metric-free task-pack verifier previously encoded NER's five
prompts, `_test` task suffix, and F1 metric. Independent pre-deployment review
caught that POS instead uses four prompts, no task suffix, and
`token_accuracy,flexible-extract`. A public-seam regression test now reproduces
that exact POS artifact shape while preserving NER defaults. The verifier and
wrapper explicitly freeze the POS task names and required metric; metric values
remain excluded from structural-verification output.

The wrapper, output root, manifests, resolved configs, and coverage must be
frozen in a fresh immutable snapshot. A metric-free preflight must pass with
the official result root absent before the single GPU execution may be
submitted. Once any official payload is entered, the bundle is terminal and
may never be corrected or retried.

Independent review found the task-ID/metric and mutable-source blockers before
deployment; both were corrected and re-reviewed with no remaining substantial
blocker. Nine focused tests, Ruff, and wrapper syntax checks pass. The immutable
snapshot is
`pure-gdn-pos-official-test-20260902-v1-63f90569`; the manifest and metric-free
preflight jobs `1286173/1286174` completed `0:0`, verified `718` source/config
files per manifest, verified all artifact roots, reproduced six resolved
configs, and left the official result root absent. Source, base, Multi, Tswana,
Xhosa, and Zulu manifest SHA-256 values are respectively
`a340f4c8652671db3ae2e0f8bf9fc19e141248bb4b1d6e8829131b976b47ba73`,
`646cc42a150934a76a7028e96106e2823ebc3af56f92eb7fea33c7404375d756`,
`929809bd9cf3a688ea6111abd744916a9cb110479729ae39d15df35791b36356`,
`df7179c62f97ae3b29302fb3b77732aecf4d0d030f6f7e71cabd2358829fb7bc`,
`ae38e2ca6863e56b2bcf4c980260c8b2870f7414fcdde544fde8e3ef84431812`,
and `51910ebb63bd2a2eb06f276ac221d41713cd03abb57bea1eda9857950c5ae031`.
