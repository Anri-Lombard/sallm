# Pure-GDN SIB family-wise official-test freeze

Frozen prospectively on 3 September 2026 after all six SIB Monolingual
adapters became terminal-valid and before any current-adapter SIB official
held-out evaluation started.

The frozen Multilingual winner is Stage-A a0 seed 42. The frozen Monolingual
adapters are Afrikaans, English, Northern Sotho, Southern Sotho, Xhosa, and
Zulu at the family LR `3e-5`. Exact retained-to-final checks passed for every
adapter across `424` keys and `71,762,560` values.

The one-time adapter bundle contains twelve fixed arms. It evaluates the
Multilingual adapter and the corresponding Monolingual adapter separately for
each of the six languages. The accepted Base artifacts are not rerun. Each arm
uses the existing five-prompt SIB test pack, requires the existing `f1,none`
metric, and must cover exactly `1,020` prompt-expanded rows.

Before GPU evaluation, a CPU-only cache job may materialize the exact six
`Davlan/sib200` test configurations into a dedicated scratch cache. This job
must assert 204 source rows per language and record the resulting cache hashes.
It may not run inference or compute a model metric. The official wrapper then
runs with Hugging Face and datasets offline against that frozen cache.

The wrapper must verify immutable manifests for its source snapshot, runtime,
base checkpoint, cache, Multilingual adapter, and all six Monolingual
adapters. A metric-free preflight must resolve and hash all twelve configs
while the official result root is absent. Once GPU evaluation opens any SIB
held-out payload, the bundle is terminal and may never be corrected or
retried.

Wrapper SHA-256 is
`fe9c7362b33cca411be895f40cfc9c4fae5bc3c0c6a71706615b42f0fe2a2cd3`.
Focused tests pass `5/5`; Bash syntax and Ruff checks pass.
