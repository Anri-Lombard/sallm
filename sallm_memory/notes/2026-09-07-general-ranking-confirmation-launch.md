# General ranking and prospective confirmation bundle

B3 1312957 completed 0:0 in 5:25:25. All five scheduled validation
sidecars and exact 22,167-row six-family coverage checks passed, including
original, failed and successful resume-manifest sidecars. CPU verifier
1314418 completed 0:0 and proved retained step8184 exactly equals final:
424 keys, 71,762,560 values. Verifier log SHA-256:
`7b337f4b3d2f1228acfb5c496018c9b35250dbb35e5bb7cc0c502d957d6f34b1`.
Preserve original and failed recoveries and the disclosed exception.

All three Stage-A and all eight Stage-B candidates are now terminal-verified.
Fresh checks confirmed all eleven final adapters exist, original manifest
sidecars and all existing validation artifact sidecars pass. Frozen ranking
script `edf5638cae9d783b840088a16079770a907208d518602bad10e210829ed53568`
ran as CPU job 1314420, completed 0:0. Registry SHA-256 remains
`8fdd6ea5f192a1d2ef2267353162b9a640ccd09ae0d889cb5a999645198bb726`.
The routine validates registry/config/manifest bindings, selects retained
trainer best_metric, and ranks ascending validation-only macro NLL with the
unchanged tie-breaks. Ranking order: b7, a2, b0, a1, b3, b1, b4, b5, a0, b2, b6.

The ranking artifact is read-only at
`/scratch/lmbanr001/masters/sallm/results/general-seed42-ranking-20260907.json`,
SHA-256 `1773058c448a619c3046fec56ae80347e152d95d65585e3af05df9870b54d87a`.
Frozen finalists are b7 and a2. Their seed42 validation values are
0.9166715052168936 and 0.9284030074439155, respectively, not official test scores.

## Prospective execution binding

Before any confirmation, freeze four candidate-specific wrappers under
`/home/lmbanr001/masters/sallm_snapshots/general-confirm-20260907`.
Use unchanged offline-v2 scientific source and launcher from the 6 September
amendments. Repeat exact source/runtime/offline-record checks at startup.
Only cache transport differs from the original snapshot, not records,
recipe, evaluation or retention. Seeds13/87 are the preregistered new trials,
not retries. No held-out outcome informed ranking or execution.

Wrapper SHA-256 values:

- b7-s13: `815ef37ffdd3a0961e068c9247bd4b0f1fe6d2966a55d7cedcf4bb72a304a778`
- a2-s13: `a6929c236ecfa1b500435c9597c49ec30fd5a2dbc10dc92d956443aab46c0d0f`
- b7-s87: `994884f2745d55fe852942cfac5bf3fb2d2e805571f1c5b065910859e05fb4aa`
- a2-s87: `9a464b05843235f2b8869103fd041953c11fcc4fde8a04b592f6e2a337fba4d6`

Submit in that order after absent output/log/manifest and duplicate checks,
with explicit native --time=36:00:00, one A100-80GB each, eight CPUs,
nlpgroup80/a100/nlpgroup80 and canonical chdir. All four 80GB cards are idle
and no owned GPU job remains; no net queue benefit justifies moving to40GB.
Maximum four owned jobs, one GPU family. Verify scheduler36h readback and
startup before treating launches as valid. A failure is preserved, not retried.
Global winner remains unfrozen until all confirmations verify. No Sheet edit.
