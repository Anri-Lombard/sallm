# Pure-GDN SIB kernel-runtime correction

Frozen on 3 September 2026 after official SIB job `1289209` failed in two
seconds and before any held-out payload opened. The job verified the sealed
cache tree, then stopped during execution-manifest verification. The result
root remained absent: no model, dataset, prediction, summary, structural
verification, or metric artifact exists.

The exact failure was a Linux kernel build-string difference between the CPU
manifest node and A100-80GB execution node:

- recorded: `5.14.0-687.30.1.el9_8`;
- execution: `5.14.0-687.29.1.el9_8`.

The `v2` snapshot carried an older `create_execution_manifest.py` that compared
the raw platform strings. The repository already contains the public-seam
normalizer frozen for earlier official execution. Reuse that unchanged helper
in a new immutable `v3` snapshot and change only the wrapper's expected
protocol directory from `sib_v2` to fresh `sib_v3`. The helper ignores Linux
build and patch drift while
still requiring the same kernel major/minor family, RHEL family, architecture,
libc, Python, executable, and complete package inventory. Its SHA-256 is
`f1f303a7cd6f2578b296de131a71eb863eee71692b8be679183827bc0bf2b1a0`.
The `sib_v3`-bound wrapper SHA-256 is
`9772f2373cd046a3ee54576d4245a5ae3630085e6f43ca45d9d8d56900359dfc`.
Focused manifest and SIB-wrapper tests pass `4/4`; Ruff passes.

This is the final prospective pre-payload SIB execution correction. It changes
no model, adapter, data row, prompt, metric, checkpoint, candidate, or arm. The
existing sealed cache remains unchanged. Create fresh `sib_v3` source and
artifact manifests from the new immutable snapshot, run the metric-free CPU
preflight, prove the result root is absent, and only then submit one replacement
bundle on A100-80GB. Preserve `1289209`, its log, the `v2` snapshot, and the
`sib_v2` protocol directory permanently.
