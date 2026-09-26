# Pure-GDN POS recovery cache-key correction

Date: 2026-09-03

This amendment is prospective and precedes any replacement POS inference.

POS recovery job 1291769 failed before loading the dataset and before writing
predictions, metrics, or result artifacts. The frozen POS lm-eval task files
request the Masakhane POS `main` URLs, while the recovery cache materializer
had staged the same expected files only under the already-frozen commit URL
`376f4161f0425584d4bd7664122b56fa026926d3`. Offline mode therefore could
not resolve the URL cache key even though the intended data were present.

The correction materializes both URL forms and requires exact equality of
every raw test line between `main` and the frozen revision for Tswana, Xhosa,
and Zulu. A mismatch aborts before model loading. The resulting supplemental
cache is independently hashed and sealed, and the replacement POS job uses a
fresh private writable runtime copy. Job 1291769, its log, and runtime cache
remain preserved.

No model, adapter, task, prompt, split, example, decoding setting, metric,
expected row count, or reporting rule changes. No held-out outcome may inform
the correction. Once the replacement writes any POS predictions or metrics,
POS is final and cannot be retried or corrected from its held-out outcome.
