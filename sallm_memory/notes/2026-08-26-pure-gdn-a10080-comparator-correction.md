# Pure-GDN A100 comparator launcher correction, 2026-08-26

Preregistered at 16:14 SAST after gate jobs `1271038/1271037` completed and
before any result payload or metric was inspected.

- Comparator `1271039` verified all four SHA-256 sidecars, then exited `2:0`
  in `argparse` before loading either benchmark or manifest. Its stale verifier
  rejected the frozen `--reference-manifest/--candidate-manifest` arguments and
  wrote no comparison artifact.
- Preserve `1271039` and its log. Submit one execution-only CPU comparator with
  the same four sidecar-verified inputs and the already-frozen manifest-aware
  verifier at SHA-256
  `64c8e42aeba1e32fed011e901cd2ebecf677190e75d628f984b046c979f81a89`.
- Write to a new `a100-launcher-correction-verification-r2.json` output. Do not
  change any input, check, threshold, equality requirement, or fail-closed
  decision rule. Do not rerun either GPU benchmark.
- Dependency-never downstream jobs `1271040--1271043` may be rebound to this
  corrected comparator before it runs. They remain blocked unless it exits
  successfully. Held-out data remain untouched.

## Execution

- Corrected CPU comparator `1271352` completed `0:0`. All four input sidecars
  reverified and the output sidecar verifies. Comparison artifact SHA-256 is
  `9f699ff2fe0377d6d149c25863500b3c252565b0d477fa2546253f25bd1fa377`.
- The gate passed every frozen check: exact predictions and aggregate metrics,
  zero maximum/mean selected-score difference, exact hardware pair and runtime
  environment, and no failed check. Recorded reference/candidate runtime ratio
  is `0.9648058493568341`; runtime is not an acceptance criterion.
- A100-80GB is therefore ratified for unchanged downstream validation work.
