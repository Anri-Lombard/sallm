# General confirmations submitted

At 00:26 SAST on 7 September, the four frozen confirmations were submitted
once after absent output/log and no-owned-job checks. All started immediately
on srvrocgpu011, one A100-80GB each, eight CPUs, nlpgroup80/a100/nlpgroup80.
Scheduler readback proves all four have TimeLimit=1-12:00:00 (36 hours).

- b7 seed13: 1314428
- a2 seed13: 1314429
- b7 seed87: 1314430
- a2 seed87: 1314431

Prospective bundle note SHA-256:
`867a195e35ca8572e6eaf30deb8961737e022cb9fd2041642c42020280596696`.
Ranking and wrapper bindings are in `2026-09-07-general-ranking-confirmation-launch.md`.
Source, original runtime and offline-record checks passed for all four.
At 00:28 SAST all four entered training and advanced to step1 or2/13640.
All four execution-manifest sidecars passed exact readback.
Preserve each submission; do not duplicate or retry. General remains unfrozen.
No official tests or Sheet changes were made.

At 06:27 SAST, all four confirmations remained RUNNING at steps 3048-3074
of 13640, approximately 6.8 seconds per step. Each first-boundary step2728
artifact passed its SHA-256 sidecar and exact 22,167-row six-family coverage,
raw-row counts, template multiplicities and AfriHG language counts under
equal_family_assistant_token_nll_v1. These checks establish validation
coverage, not terminal acceptance or cross-candidate ranking. All four owned
80GB slots are doing useful work; no queue or demonstrated migration benefit
justifies interruption. Quota is home90.1%, scratch155/300GB (51.7%).
No job, held-out evaluation or Sheet value was changed.

At 11:40 SAST, all four step5456 sidecars passed with the same exact
22,167-row coverage, raw counts, template multiplicities and AfriHG language
counts. All four resumed training and reached steps5653-5695/13640.
No interim cross-candidate selection was performed. Quota: home90.1%,
scratch51.8%. Continue unchanged toward the next frozen boundary.

At 16:44 SAST, all four step8184 sidecars and exact coverage checks passed
(22,167 processed rows, frozen raw/template counts and AfriHG languages).
All jobs remain RUNNING; three logs show resumed training beyond8184,
while a2 seed13 still shows validation data processing. No terminal
acceptance or cross-candidate selection is implied. Scratch51.9%, home90.1%.

At 22:21 SAST, sacct confirmed all four jobs COMPLETED 0:0 in 21:38:13,
21:45:15, 21:38:36 and21:40:35 respectively. Each ended at step10912 and
saved a final adapter; only retained checkpoint5456 remains in each root.
All step10912 sidecars and exact coverage checks passed. CPU job1317663
was submitted once, after absent-log/no-duplicate checks, to run unchanged
verify_adapter_roundtrip.sbatch for all four retained5456/final pairs.
Verifier source hash a138101a0be2dbb856c07bc2d5d5e59fd31436c2c30de4788287bf5ba0d75184.
Log: /scratch/lmbanr001/masters/sallm/logs/jobs/gdn-general-confirm-verify-20260907.out.
Do not duplicate verification while pending. General remains unfrozen pending
verification and the frozen three-seed ranking/winner rule. No tests or Sheet edits.
