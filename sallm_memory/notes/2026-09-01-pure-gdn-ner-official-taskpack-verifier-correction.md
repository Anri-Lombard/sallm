# Pure-GDN NER official task-pack verifier correction

Frozen prospectively at `2026-09-01T16:53:37Z`, before any current-adapter
NER held-out access.

Independent review found that the initial metric-free preflight used the T2X
generation-artifact verifier for NER lm-eval task packs and declared `5,000`
rows for every language. That verifier could not validate NER's
`<task_pack>/results.json` artifact, and Tswana's exact five-prompt coverage is
`4,980` (`996` rows per prompt), not `5,000`. Xhosa and Zulu remain `5,000`.
The initial snapshot, manifest job `1283916`, preflight `1283917`, and freeze
note are preserved as superseded preparation. They loaded no held-out row and
produced no official output.

The prospective correction adds a task-pack-specific structural verifier. It
requires the exact five `_test` tasks, finite per-prompt
`f1,flexible-extract`, matching summary/result artifacts, exact `n-samples`
and sample-list coverage, and no extra or missing task. It records hashes and
coverage but no metric values. Focused tests pass `5/5`, Ruff and shell syntax
checks pass, and the verifier accepted the already finalized Base official
artifacts at exact `4,980/5,000/5,000` Tswana/Xhosa/Zulu coverage without
printing metric values.

The corrected immutable snapshot is
`/home/lmbanr001/masters/sallm_snapshots/pure-gdn-ner-official-test-20260901-v3-731102b2`.
The corrected wrapper SHA-256 is
`731102b23e11b1d45d8473fff3f6dc263219e7dbc36351fdc838be1866b1af42`;
the task-pack verifier SHA-256 is
`23523234228703e48339fde015100b7e076cc33d00ee0c3837eeaaf23c9a27dd`.
Manifest-freeze job `1283922` completed `0:0` with hashes:

- source: `749570194cb747dcbf5bf166c782c6e1b45e685a83f28a7d73da0f8015c9fb6b`
- base: `be661cc169215406596ed2f60699b850485d80bb16b9b7edb6296070cc6550cb`
- Multi NER: `da14e5b8cbefc8febd38f8142cee0a7aba10a22c452cc0a355c28f374d750780`
- Mono Tswana: `78b34cc40735041b9ead7d4ffc9440502805e6c7decbf848942c25b4cac035d9`
- Mono Xhosa: `7261dcbd007825dd3d77232c890c6cb82e5ab0e983265d35cb36eee327bb6bac`
- Mono Zulu: `fcff88b3a1157c01cc512dccaa5155de47e4acdbb2b1b531ef6b395372531165`

Metric-free preflight `1283923` completed `0:0`, verified all `717` immutable
source/config files and all artifact manifests, and reproduced the six prior
resolved-config hashes exactly. The fixed execution order remains Multi
Tswana/Xhosa/Zulu then Mono Tswana/Xhosa/Zulu. The current result root remains
absent and fixed at
`/scratch/lmbanr001/masters/sallm/results/official_test/familywise_20260901/ner`.
Any post-payload failure is terminal for the affected arm and is not retried.
Sheet E/F/G remain blank until the official artifacts and exact readback verify.
