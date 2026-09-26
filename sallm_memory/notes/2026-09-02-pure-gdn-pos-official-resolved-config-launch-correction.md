# Pure-GDN POS official resolved-config launch correction

Frozen prospectively on 2 September 2026 after job `1286189` failed in three
seconds and before any official payload began.

All source, Base, and Multi manifests verified. The wrapper then attempted to
overwrite the already frozen read-only `multi_tsn.resolved_config.yaml` while
reproducing metric-free config resolution. Shell redirection failed with
permission denied. The official result root remains absent; no model, dataset,
evaluator, held-out row, prediction, metric, summary, or Sheet cell was opened
or written. Preserve job `1286189`, its log, snapshot, and manifests; never
reuse or rerun it.

The prospective correction changes only resolved-config handling. Metric-free
preflight still creates and hashes all six resolved configs. Scientific
execution now verifies their existing SHA-256 sidecars before payload instead
of trying to overwrite them. Exact Base/Multi/Mono roots, task packs, pinned
dataset revision, task IDs, metric requirement, coverage, offline runtime,
output root, arm order, and structural verification are unchanged.

A regression check first failed against the old wrapper and now proves that
config resolution exists only before the payload split and frozen configs are
verified rather than regenerated for execution. Ten focused tests, Ruff, and
shell syntax checks pass. A fresh immutable snapshot, source/artifact manifests,
and metric-free preflight are required before at most one isolated replacement
may be submitted. Any replacement that enters payload is terminal and may
never be retried or corrected.

Unused preparation snapshot
`pure-gdn-pos-official-test-20260902-v2-fce1ca68` and manifest job `1286720`
are excluded before preflight because the wrapper still bound the original
`pos_v1` protocol root. No result root or payload was touched. The corrected
wrapper now binds the isolated protocol root
`pos_resolved_config_correction_v3`; its SHA-256 is
`41626421d6b23f6042863e3aca4e8952d9ec374286b4023565c64227a9537d06`.

Fresh immutable snapshot
`pure-gdn-pos-official-test-20260902-v3-41626421`, manifest job `1286723`,
and metric-free preflight `1286724` completed `0:0`. The preflight verified
all six exact source/artifact/runtime manifests, regenerated the six isolated
resolved configs, and left the official result root absent.

Independent final review found no substantial blocker and confirmed that one
isolated replacement is scientifically permissible because `1286189` never
entered payload. Replacement job `1286725` was submitted once after exact
absence, no-duplicate, active-cap, manifest, and preflight checks. It is
`AssocGrpGRES`-pending as the fourth owned A100-80GB submission and becomes
terminal as soon as payload starts.
