# General wall-time terminality and untouched remainder — 9 September 2026

General group g2 job `1319959` reached its immutable 24-hour scheduler limit
and ended `TIMEOUT` after `1-00:00:08`. It had claimed only
`masakhapos_all`. That unit has zero result files and no structural
verification after partial official inference, so General POS is terminally
missing and must never be retried.

The four later units never started: `afrimmlu_sa`, `belebele_sot`,
`belebele_xho`, and `afrihg_zul` have no claims, output roots, result files, or
structural sidecars. They therefore retain their single official access. A
single 48-hour continuation may run those four units once, in that frozen
order, through the unchanged immutable General snapshot, configs, manifests,
cache, model, adapter, verifier, and exclusive-claim wrapper. This changes
only the allocation wall time and omits the terminal POS unit; it is frozen
before submission and cannot depend on any score.

The Base checkpoint canary remains older in the A100-80GB queue and has
priority. The continuation may wait behind it; no active job is disturbed.

Continuation job `1325602` was submitted once with the frozen four-unit order,
one A100-80GB, eight CPUs, and a 48-hour limit. Scheduler readback shows it
capacity-pending behind the older Base canary. All four claims remain absent.
