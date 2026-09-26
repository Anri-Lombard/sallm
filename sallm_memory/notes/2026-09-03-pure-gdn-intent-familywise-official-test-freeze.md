# Pure-GDN Intent family-wise official-test freeze

Frozen prospectively on 3 September 2026 after all four Intent Monolingual
adapters became terminal-valid and before any current-adapter Intent official
held-out evaluation started.

The frozen Multilingual winner is Stage-A a1 seed 42 at LR `8e-5`. The frozen
Monolingual adapters are English, Southern Sotho, Xhosa, and Zulu at the same
family LR. Exact retained-to-final verification passed for all five adapters
across `424` keys and `71,762,560` values.

The one-time bundle contains eight fixed arms: separate Multilingual and
Monolingual evaluation for each language. Accepted Base artifacts are not
rerun. Each arm uses the existing five-prompt chat-template InjongoIntent test
pack and requires `f1,none`. English must cover exactly `622` source rows and
`3,110` prompt-expanded rows; each other language must cover `640` and `3,200`
respectively.

Metric-free cache job `1290434` materialized exact
`masakhane/InjongoIntent` English/Southern-Sotho/Xhosa/Zulu test configurations
and completed `0:0`. Dataset-cache manifest SHA-256 is
`1466acb4283645f102afad5f0ad33b38c3b5e5eae1189920fcf54f762db79f9b`;
full cache-tree manifest SHA-256 is
`71b2241b32efc89f3058d0169cdc807c62bd1cf1763f58b46c2314f3d0c2405e`.
No model inference or metric was computed.

Immutable snapshot is
`pure-gdn-intent-official-test-20260903-v1-50cf664f-final`. Wrapper SHA-256 is
`50cf664f756ae443a2f70b0365c36c1929a62558a04fd311cd2f95d4d6fb1762`;
runtime-manifest helper SHA-256 is
`f1f303a7cd6f2578b296de131a71eb863eee71692b8be679183827bc0bf2b1a0`;
structural verifier SHA-256 is
`191f06313dbdb86fecccaa1806c0610fba295dcbe6a12f5b900ef8b669b2ecfd`.
Bash syntax, focused tests, and Ruff pass.

Fresh source, base, cache, and adapter manifests plus resolved configs must
verify in a metric-free CPU preflight while the result root is absent. Once
the GPU bundle opens any Intent held-out payload, it is terminal and may never
be corrected or retried. Scores may be opened only after all eight arms and
their structural sidecars verify.
