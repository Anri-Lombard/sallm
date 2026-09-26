# Pure-GDN News family-wise official-test freeze

Frozen prospectively on 3 September 2026 after both News Monolingual adapters
became terminal-valid and before any current-adapter News official held-out
evaluation started.

The frozen Multilingual winner is Stage-A a0 seed 42 at LR `3e-5`. The frozen
Monolingual adapters are English and Xhosa at the same family LR. Exact
retained-to-final verification passed for all three adapters across `424` keys
and `71,762,560` values.

The one-time bundle contains four fixed arms: separate Multilingual and
Monolingual evaluation for English and Xhosa. Accepted Base artifacts are not
rerun. Each arm uses the existing five-prompt chat-template News test pack and
requires `f1,none`. English must cover exactly `948` source rows and `4,740`
prompt-expanded rows; Xhosa must cover `297` and `1,485` respectively.

Metric-free cache job `1290433` materialized exact `masakhane/masakhanews`
English/Xhosa test configurations and completed `0:0`. Dataset-cache manifest
SHA-256 is
`753619b4b3ad7871f021fc9ffa4590f8322ffbfe9a8d468296a58643ae25e9f3`;
full cache-tree manifest SHA-256 is
`b4be48ddb3c61ac31ef5e0803f582cc8e069204c1a2b8aa20d814b61ec8d521f`.
No model inference or metric was computed.

Immutable snapshot is
`pure-gdn-news-official-test-20260903-v1-50cf664f-final`. Wrapper SHA-256 is
`50cf664f756ae443a2f70b0365c36c1929a62558a04fd311cd2f95d4d6fb1762`;
runtime-manifest helper SHA-256 is
`f1f303a7cd6f2578b296de131a71eb863eee71692b8be679183827bc0bf2b1a0`;
structural verifier SHA-256 is
`191f06313dbdb86fecccaa1806c0610fba295dcbe6a12f5b900ef8b669b2ecfd`.
Bash syntax, focused tests, and Ruff pass.

Fresh source, base, cache, and adapter manifests plus resolved configs must
verify in a metric-free CPU preflight while the result root is absent. Once
the GPU bundle opens any News held-out payload, it is terminal and may never
be corrected or retried. Scores may be opened only after all four arms and
their structural sidecars verify.
