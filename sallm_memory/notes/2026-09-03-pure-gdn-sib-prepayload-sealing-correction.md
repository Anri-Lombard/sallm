# Pure-GDN SIB pre-payload sealing correction

Frozen on 3 September 2026 before any current-adapter SIB held-out inference.
The SIB result root is absent. Cache job `1289189`, manifest job `1289204`,
and metric-free preflight `1289205` completed `0:0`, but the first immutable
snapshot omitted `scripts/verify_official_task_pack_eval.py`. The wrapper only
calls that verifier after an arm finishes, so the preflight could not detect
the omission. The snapshot and its protocol directory are preserved and will
not be used for GPU evaluation.

One prospective pre-payload correction is allowed. The replacement snapshot
must include the unchanged task-pack verifier. Its wrapper must fail before
payload unless that verifier exists, verify a SHA-256 inventory covering the
full dedicated SIB cache tree, and reject writable resolved configs during GPU
execution. The cache data files and all twelve resolved configs must be sealed
before launch. Empty lock files may remain writable for the offline datasets
runtime, but their pre-launch content remains covered by the cache inventory.

The replacement keeps the frozen scientific design unchanged: the same base,
SIB a0 Multilingual winner, six frozen Monolingual adapters, six languages,
five prompts, `f1,none`, and exactly `1,020` expanded rows per arm. It uses a
new immutable snapshot and `sib_v2` protocol root. The official result root
does not change. A fresh metric-free manifest and preflight chain must pass
before the one-time A100-80GB bundle may start.

Corrected wrapper SHA-256 is
`7a0000f735c494ecaadf493bb699791709bd4046763e13b7df17fd85542c15b0`.
Verifier SHA-256 is
`191f06313dbdb86fecccaa1806c0610fba295dcbe6a12f5b900ef8b669b2ecfd`.
Bash syntax, two focused tests, and Ruff pass.
