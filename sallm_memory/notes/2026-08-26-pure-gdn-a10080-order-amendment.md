# Pure-GDN corrected gate order amendment, 2026-08-26

Preregistered at 15:57 SAST before either corrected gate job starts and without
inspecting any gate result. The user explicitly authorised using the currently
idle A100-80GB capacity rather than waiting for a congested A100-40GB device.

- Reverse only the execution order of unstarted jobs `1271037/1271038`:
  A100-80GB candidate `1271038` runs first; A100-40GB reference `1271037` may
  start only after the candidate completes successfully.
- Candidate output remains sealed. Do not inspect its benchmark, metric,
  prediction, score, manifest contents, or result before the reference and
  unchanged comparator complete.
- Comparator `1271039` may run only after reference success and must apply all
  unchanged equality checks and thresholds to the same isolated pair.
- Keep the same immutable wrapper/source, explicit requested/exported GRES,
  runtime, model, adapter, validation rows, BF16 mode, resources, sidecar
  verification, and fail-closed rule. No A100-family overlap is allowed before
  comparator success.
- Four downstream A100-80GB jobs `1271040--1271043` remain dependency-held on
  comparator success. No held-out split may be accessed.

## Execution

The scheduler allocated the reference while dependency edits were being
applied, so the actual safe order remained reference `1271037` then candidate
`1271038`. They did not overlap: `16:08:13--16:09:13` and
`16:09:13--16:10:15`, respectively. Both completed `0:0`; no result was
inspected before the comparator path was resolved.
