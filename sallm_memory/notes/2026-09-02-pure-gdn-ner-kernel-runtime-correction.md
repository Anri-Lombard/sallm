# Pure-GDN NER kernel-runtime verification correction

Frozen prospectively on 2 September 2026 after official NER job `1285944`
failed in one second and before any current-adapter NER held-out row loaded.

The job stopped during immutable execution-manifest verification. The frozen
preflight recorded platform string
`Linux-5.14.0-687.30.1.el9_8.x86_64-x86_64-with-glibc2.34`; the A100-80GB
compute node reported
`Linux-5.14.0-687.29.1.el9_8.x86_64-x86_64-with-glibc2.34`. The official
result root remains absent. No model, adapter, evaluator, dataset, prediction,
metric, summary, or Sheet cell was opened or written.

The prospective scientific correction changes only runtime identity comparison. Linux
kernel release text is normalized away while operating system, machine
architecture, libc identity, exact Python version, executable, and complete
package versions remain fail-closed. The six fixed NER arms, task configs,
adapters, order, expected coverage, structural verifier, output root, and
post-payload terminality are unchanged. A new protocol root isolates the
replacement's metric-free resolved configs; artifact contents remain unchanged
and all source/artifact manifests must be regenerated for the corrected
snapshot.

A public-seam regression test first reproduced the failure for the two kernel
patch strings. The minimal implementation then made that test pass alongside
the existing package-drift failure test. Before one isolated replacement may
run, create a fresh immutable snapshot, regenerate all six source/artifact
manifests on a compute node, reproduce the metric-free resolved-config hashes,
and confirm the official output root and any duplicate job remain absent.

Job `1285944`, its log, frozen manifests, and absent result are preserved.
Never reuse or rerun it. Any replacement that reaches official payload is the
single terminal execution; no post-payload correction or retry is allowed.

The corrected implementation SHA-256 values are
`b210377293683ce24c7e4a905821d8d41b583f14224aa404edbb7cb619f6fbfc`
for `create_execution_manifest.py` and
`6508743c7fce18f3255683266b5c70d99fe3be9dc6ddce4ff49468892ee513ae`
for the isolated wrapper. Fresh immutable snapshot
`pure-gdn-ner-official-test-20260902-v4-6508743c`, manifest job `1285991`,
and metric-free preflight `1285995` completed `0:0`. The preflight verified
all `717` source/config files per manifest, reproduced all six frozen resolved
config hashes exactly, and left the official result root absent.

The single replacement is job `1285997`. Slurm readback proves
`nlpgroup80/a100/nlpgroup80`, `gpu:ampere80:1`, eight CPUs, a 24-hour limit,
and the required checkout. Startup reverified the corrected immutable source,
base, and adapter manifests. This job is now terminal under the family-wise
official-test protocol and must never be rerun.

Job `1285997` then failed `1:0` after `00:01:38`. All manifests passed and the
model plus frozen Multi adapter loaded, but the first official task stopped in
`datasets.load_dataset` because the offline runtime could not resolve
`anrilombard/masakhaner-x-parquet`. The result root contains only empty
`multi_tsn/masakhaner_tsn` directories: zero files, predictions, summaries, or
metrics. No held-out row loaded. This remains post-payload under the frozen
rule, so NER official results are terminally missing and no further retry is
authorized.

Independent review also found that the deployed normalization kept OS,
machine, and libc but dropped the Linux kernel and distribution families more
broadly than intended. That did not mask this run: both observed nodes were
Linux `5.14`/`el9_8`. For future official-family snapshots only, the shared
helper is narrowed to retain kernel major/minor and RHEL family while ignoring
only build drift. Regression checks now accept the observed `.29.1`/`.30.1`
pair but reject Linux `6.12` and `el8_9`; all five focused tests and Ruff pass.
The future-path helper SHA-256 is
`f1f303a7cd6f2578b296de131a71eb863eee71692b8be679183827bc0bf2b1a0`.
