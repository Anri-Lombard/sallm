# General official execution preparation

The user instructed completion now, including safe autonomous routine repairs.
General remains B7 seed42 checkpoint5456. Static contract audit1319707 passed
all187 prompt tasks and3 generation tasks; contract SHA256
d4c63e1c3d8cff8a5440e833bb861939bae12642b0433a61242432dd15479699.

Data-only staging1319844 copied verified News/NER/POS, SIB and Intent caches
to a new General cache, materialized only missing datasets, and reloaded
offline. Its initial exporter rejected Belebele datetime fields; no scientific
metric was computed. V2 encodes those fields with explicit type and ISO8601.
Offline job1319869 completed with complete=true for all38 dataset specifications.
Prior incomplete reports are preserved; no row, prompt or split was changed.

New isolated snapshot: pure-gdn-general-official-20260908-v1, copied from the
corrected pure-gdn-heldout-recovery-20260903-v2-46d8cf11 snapshot. CPU1319896
checks the actual TaskManager evaluation rows and native generation loader
rows before writing19 execution-unit configs:16 packs and3 native generation
tasks, covering the unchanged16 logical lanes. Splitting SA-general into its
three packs and AfriHG into two language tasks changes execution grouping only.
The test examples, prompts, scoring, seed, beam settings and adapter remain
unchanged. POS uses its established safe batch1, max_batch1; no task is omitted.

Each task include path is bound to the isolated snapshot, not the mutable
canonical checkout. The runner verifies source/base/adapter/runtime and data
hash bindings, copies the frozen cache into a fresh per-arm runtime cache, and
claims the arm exclusively before opening it. Only after all metric-free
checks and final bindings pass may four A10080 jobs start. Never repeat an
arm after scientific payload. Preserve all output and require exact per-task
coverage and finite declared metrics before releasing scores.

Local pooled-result verifier self-check passed, including rejection of a
missing sample. Bash syntax checks passed. Generation reuses the existing
corrected structural verifier. No official General arm has yet run.

Actual-loader preflight1319896 completed0:0 in4:19. It verified all187 task
row counts plus native T2X378, AfriHGXhosa1305 andAfriHGZulu1776. Nineteen
unit configs and preflight/units JSON are written. Sealing CPU1319932 is
running: it resolves every generated config, checks the frozen General
adapter hash and previously passed exact-base CUDA gate1279771, creates
source/base/adapter/runtime manifests and file bindings, and only then writes
READY. Official tests remain unstarted until this succeeds.

Prospective execution grouping is four A10080 jobs, each sequentially running
every fourth unit in the frozen order. This keeps exactly four owned GPU jobs
without a large pending array and fills each allocation without half-hour
dispatch gaps. All units remain independent exclusive official accesses.
Each job stops on failure; remaining never-started units retain their status.
No failure or output is overwritten, and scores cannot change the grouping.

Base provenance audit found only the August BOS/raw outputs in the canonical
evaluation root, no newer Base result root, and no Base-named job since
30August. The1September familywise note's unsupported16/16 statement does
not supersede the30August explicit14-lane correction amendment. Those Base
corrections remain required for the complete paper-ready table; T2X/AfriHG
Base remain excluded from that rerun.

Seal1319932 completed0:0 in2:34. READY/bindings SHA256:
8e717f9b75070177234c2c3604082c0ad63ebfa94a0bce49f241339f8c8a742e.
Source manifest:5eba6ffb1eb8815cb5b8186a0b07900d76e5d11fa3dc01adbc4f63a71b62f04f.
Base manifest:fa4e9c3c5903c140921d3520fdbd965c0e0431dab1d43887b8cd69a4ee59a5d5.
Adapter manifest:724b2b693832eb8f31ec082fd295caf2e4ec7cb324a9455474de064cf9357dee.

Four official jobs submitted once after empty-owned-queue and absent-result
checks, all RUNNING on srvrocgpu011 with one A10080,8CPUs,24hours each:

- g0 1319956: News, Intent, BelebeleAfr, BelebeleTsn, T2X.
- g1 1319958: NER, AfriMGSM, BelebeleEng, BelebeleTso, AfriHGXho.
- g2 1319959: POS, AfriMMLU, BelebeleSot, BelebeleXho, AfriHGZul.
- g3 1319960: SIB, AfriXNLI, BelebeleSsw, BelebeleZul.

All four launch-time manifests and file bindings passed. Logs:
/scratch/lmbanr001/masters/sallm/logs/jobs/gdn-general-official-g{0,1,2,3}-20260908.out.
Outputs: /scratch/lmbanr001/masters/sallm/results/official_test/general_20260908_v1/<unit>.
Protocol: /scratch/lmbanr001/masters/sallm/manifests/general_official_20260908_v1.
Each output must gain structural_verification.json and finish successfully
before acceptance. No scores or workbook cells have been opened/changed.

09:04 SAST monitoring: g3 1319960 completed 0:0 in 19:51. Structural
verification files exist for News, SIB, AfriXNLI, BelebeleSsw and BelebeleZul.
Other three groups remain RUNNING. No scores inspected.
POS reached 107/7216 at about 19.27 seconds/request, projecting 38 hours
remaining against a 24-hour allocation; later request lengths may vary.
Attempted in-place runtime-only extension of job1319959 to48hours;
scheduler denied permission. Verified limit remains24hours, deadline
9September08:27:19 SAST. No cancellation, restart or scientific changes.
Administrator extension needed if pace persists; user action requested.

09:33 SAST heartbeat: same three jobs running and g3 completed0:0.
Read all five structural JSONs directly: verified=true and metric values
excluded. Counts: News10tasks/6225rows; SIB30/6120; XNLI20/12000;
BelebeleSsw5/4500; BelebeleZul5/4500. Intent293770/508400 likelihood
requests, NER8081/14980 generation requests, POS207/7216. POS ETA remains
about37hours, already reported to user; no duplicate extension attempt.
Re-read30August Base amendment and existing General preparation/runner:
reuse the frozen corrected evaluator and coverage, but create separate
Base bindings/configs with no adapter and explicit raw mode for16packs
covering14logical lanes. Exclude native T2X/AfriHG. No Base job submitted.

10:33 SAST: nine of19 units structurally verified. Newly inspected JSONs
have verified=true, metric_values_included=false: NER15tasks/14980rows,
Intent20/12710, BelebeleAfr5/4500, BelebeleTsn5/4500. g0 is running T2X
(378 prepared examples, auto batch64); g1 AfriMGSM2121/5000 requests;
g2 POS409/7216, ETA about36hours remaining. g3 remains completed0:0.
No scientific result values released, retries or Sheet edits. Quota
home90.1%, scratch159GB53.1%. Base preparation remains outstanding.

11:07 SAST: g1 failed1:0 only in AfriMGSM coverage verification. Corrected
filter-aware standalone verifier CPU1321090 passed20tasks/5000documents;
General now10/19 verified. Preserved inference, metrics and frozen snapshot.
Never-started g1 remainder submitted as1321101, A10080/8CPU/48h:
BelebeleEng, BelebeleTso, AfriHGXho. See filter-verifier-correction note.

11:34 SAST: g0 completed0:0 in3:01:29; g1 remainder1321101 completed0:0
in21:11. Four new verification artifacts: T2X378, BelebeleEng4500,
BelebeleTso4500, AfriHGXho1305. General14/19; only g2 remains running,
POS615/7216 then MMLU/BelebeleSot/Xho/AfriHGZulu. POS ETA about34hours
remaining, original24h allocation unchanged. Base preparation outstanding.
Incidental native-generation metrics were visible in terminal log tails;
no result-based changes or public release. Prefer structural output only.

09:00 SAST on 9 September: g2 `1319959` timed out at its fixed 24-hour
boundary after claiming only POS. POS has zero result files and no structural
sidecar after partial inference, so it is terminally missing and cannot be
retried. AfriMMLU, BelebeleSot, BelebeleXho and AfriHGZulu have no claim or
output and retain their one official access. Their prospective 48-hour
execution-only continuation is frozen in
`2026-09-09-general-walltime-terminal-and-remainder.md` before submission.
