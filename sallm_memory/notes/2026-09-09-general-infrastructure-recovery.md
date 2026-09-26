# General infrastructure recovery — 9 September 2026

The user’s standing instruction is to finish the official result set, and at
13:52 SAST he asked why the two missing General units were slow and whether we
could fix them. This amendment prospectively supersedes the earlier terminal
rule for exactly General POS and AfriHG Zulu. Neither failed attempt produced a
result or metric artifact, and no metric value informed this correction.

## Confirmed causes

- General POS job `1319959` ran `6,604/7,216` free-generation requests in 24
  hours and timed out at 92%. The frozen full-matrix configuration specified
  batch size 8, but `.audit/prepare_general_official.py` overrode POS to batch
  size 1. The sealed execution config confirms `batch_size: 1` and
  `max_batch_size: 1`. This avoidable serial override is the runtime cause.
- The same POS task family previously completed 14,424 Mono/Multi requests in
  `04:32:43` using its established batched task packs. General’s poorer POS
  output termination can still make each request longer, but it does not
  justify forcing independent rows to run serially.
- AfriHG Zulu job `1326112` failed because the frozen snapshot always made a
  GitHub request before registering already-present cache files. The sealed
  offline wrapper therefore could not use the existing Zulu CSVs. The three
  cache files are present and now bound by SHA-256.

## Frozen recovery

Use snapshot
`/home/lmbanr001/masters/sallm_snapshots/pure-gdn-general-recovery-20260909-v3`
and protocol
`/scratch/lmbanr001/masters/sallm/manifests/general_recovery_20260909_v3`.
POS restores the original full-matrix `batch_size: 8`, keeps
`max_batch_size: 16`, and changes no model, adapter, official rows, prompts,
decoding, filter, metric or aggregation. AfriHG Zulu changes only offline cache
discovery and keeps the same three CSV contents.

Before a held-out recovery, batch 1 and batch 8 must produce identical raw
responses, filtered responses, coverage and metrics on 12 deterministic
synthetic POS rows through the same lm-eval path. This gate contains no
official test row. Source, cache, configs, model and adapter are sealed only
after that gate passes. Each recovery uses a new output root and is labelled as
an infrastructure recovery; the failed roots and jobs remain preserved.

Preparation attempts `1327418` and `1327432` failed before model loading or
metrics because the old validation URLs were unsuitable as a stable gate
dependency. Their logs, v1/v2 snapshots and cache roots are preserved read-only.
Synthetic-gate preparation `1327446` completed `0:0`; its preflight reports
`model_loaded=false`, `metrics_computed=false`, 12 synthetic rows and 1,776
offline AfriHG Zulu test rows. Batch-equivalence job `1327451` is queued on the
ratified `gpu:ampere` A100-40GB family behind the currently occupied cards.

At 17:32:52 SAST the remaining corrected-Base job `1326144` completed `0:0`;
all five assigned sidecars verified without exposing metrics, closing Base at
15/16 with only its unrecovered POS unit missing. Another user's job filled the
released `gpu:ampere` card, so gate `1327451` remains priority-pending and has
not started or created an output root.

## Runtime-manifest correction — 10 September

Gate `1327451` eventually received `srvrocgpu010` but failed `1:0` after one
second, before creating an output or runtime directory and before loading a
model or data. Its CPU-created source manifest included Ada-node management
packages and a different `idna` version, so strict full-package equality could
not hold on the A100 compute node. The failed log is preserved at SHA-256
`d7de90b70569051dbb4b1f46f67acc70909d5247b13284b57646590ec90ad508`.

The versioned runtime-v2 wrapper keeps the frozen v3 snapshot, configs, cache,
synthetic rows and absent official result root unchanged. It verifies source
and artifact hashes independently, creates a complete runtime manifest on the
target A100 before model loading, verifies that manifest in place, and requires
future official jobs to match it exactly. Wrapper SHA-256 is
`2b645e291b73d8b9cae31972ec71ccf5e0a7142210a42a182b3903724e004db2`;
the gated seal helper is
`4810b7595b78143ec980991e47763991b52a695493a9e1b3d232a84551a4e28d`.
Replacement validation-only gate `1331858` was submitted once on
`gpu:ampere` and is resource-pending. No official test payload has run.

## Canonical batch-1 fallback — 10 September

Gate `1331858` ran on `srvrocgpu010` and failed closed after `00:06:01`.
Batch 8 reduced the measured synthetic evaluation from 279 to 70 seconds, but
changed raw generated responses on four of twelve rows and changed filtered
responses on two rows. Aggregate synthetic metrics and sample counts happened
to match; that is insufficient for inference equivalence. Verification SHA-256
is `72700530182d10f25260e5c6037de6633b04bf4aa0501fd36e1c0aa4445ce940`.

Batch 8 is rejected. The only eligible POS recovery now uses the original
sealed `batch_size=1` and `max_batch_size=1` semantics with a 48-hour walltime.
This is a walltime-only recovery of the result-missing unit. It changes no
model, adapter, official row, prompt, decoding, filter, metric or aggregation.
AfriHG Zulu remains the separate cache-discovery-only recovery.

The canonical wrapper is frozen at SHA-256
`937f4feab7d93bcc730de8bf63f6503cb13182b777763061dec82b1460b11c51`;
its seal helper is
`835932eb834d53f6bd33b36a173d1f14a8b58c20b905ceaafaf4e56a56190d43`.
They bind the rejected gate artifacts and the exact A100 runtime manifest so
the failed optimization cannot silently re-enter execution.

CPU seal `1331982` completed `0:0` in five seconds. Binding SHA-256 is
`8e10d466023caf518490974912b0981da47d19f02292259d09549527fef2f230`.
After exact absence and binding checks, canonical POS job `1331984` was
submitted once with 48 hours and cache-only AfriHG Zulu job `1331985` once
with four hours, both on `gpu:ampere`. At 16:37 SAST both were pending; Slurm
projected POS to start at 01:00 on 11 September and supplied no start estimate
for AfriHG. Based on the failed canonical run's 6,604 rows in 24 hours, POS is
expected to need roughly 26–32 hours after it starts. Scheduler estimates may
move; no metric value has been opened.

## Hydra override-path correction — 10 September

Canonical POS job `1331984` started at 18:19:24 SAST and failed `1:0` after
19 seconds. Its command used `evaluation.overrides...`, but Hydra composes the
selected config under the `eval` group, so it rejected the override before
entering the application. The official output root is absent; no model, test
row, response or metric was loaded or written. The failed claim, runtime and
log remain preserved.

The isolated v4 wrapper changes only the two command-line paths to
`++eval.evaluation.overrides.masakhapos_all.batch_size=1` and
`++eval.evaluation.overrides.masakhapos_all.max_batch_size=1`. CPU seal
`1332060` completed `0:0`, composing and resolving the command on a compute
node and proving exact equality to the prior resolved configuration after
only those two canonical batch values are changed. Wrapper SHA-256 is
`242b790daa0d9097bad2b77ace2b0a134d597e263dc33631cec64cfa1e469549`;
seal-helper SHA-256 is
`8e1b7a65b45ca485a474a5b2a149da2dc860bf6ae0eee0bd6955b9ac1ffa5a86`;
binding SHA-256 is
`73db8b0c2b5d03efefa09b2cd5109e85d222575569beb1ee755fb4c7f6c0a175`.

After fresh output, runtime, claim and duplicate checks, replacement POS job
`1332061` was submitted once with the same 48-hour A100-40GB resources. It
started at 18:52:03 SAST and passed the former Hydra failure point into model
loading. AfriHG Zulu `1331985` has run since 18:19:31 on the same GPU family.
Neither job may be repeated after scientific payload; metric values remain
unopened until both are terminal and structurally verified.

At 18:53:53 SAST AfriHG Zulu `1331985` completed `0:0` in `00:34:22`.
Structural sidecar SHA-256
`ce92187905247980bbc24d93155805c65080cc7ce9d2665e71a5dabdfb9b4640`
reports `verified=true`, `rows=1776`, language `zul`, the required metric
present, and `metric_values_included=false`. General is now 18/19
structurally verified. POS `1332061` reached the `7,216`-request generation
loop and remains the sole unfinished unit; no metric value was opened.

## Terminal completion and final promotion — 11 September

POS replacement `1332061` completed `0:0` at 19:38:24 SAST after
`1-00:46:21`. Its structural sidecar verifies all 12 prompt tasks and 7,216
rows without including metric values; SHA-256 is
`6169b2388a801c9a95f4f5dac4dfebcb3d7d1df72c57ebf78594f35ba2d50787`.
General is therefore structurally complete at 19/19. Corrected Base remains
15/16 because its POS pack has no result.

After both recovery jobs were terminal-valid, the sealed summaries were read
and promoted once under the existing descriptive best-prompt convention.
General headline results are:

- News Eng/Xho F1: `0.0753/0.2788`.
- NER Xho/Zul/Tsn F1: `0.5258/0.5657/0.6051`.
- POS Xho/Zul/Tsn token accuracy: `0.9434/0.9301/0.9817`.
- SIB Zul/Xho/Sot/Nso/Eng/Afr F1:
  `0.2513/0.2000/0.2197/0.2006/0.3216/0.4487`.
- Intent Eng/Xho/Zul/Sot F1: `0.0024/0.0016/0.0029/0.0037`.
- Belebele Xho/Zul/Tsn/Ssw/Sot/Eng/Afr/Tso `acc_norm`: `0.2289` for
  every language.
- AfriXNLI Xho/Zul/Sot/Eng accuracy:
  `0.3617/0.3517/0.3333/0.3417`.
- AfriMMLU Xho/Zul/Sot/Eng accuracy:
  `0.2180/0.2020/0.2180/0.1840`.
- AfriMGSM Xho/Zul/Sot/Eng flexible-extract exact match:
  `0.0240/0.0080/0.0080/0.0080`.
- T2X Xho chrF `48.7769`; AfriHG Xho/Zul chrF
  `21.3814/20.9593`.

The exact prompt-level values, summary hashes, structural hashes, job IDs and
Sheet rows are sealed locally in
`.audit/pure-gdn-final-official-results-20260911.json`.

Google Sheet `GDN Results` now has verified dates in 41 reportable rows,
35 corrected-Base values, three intentionally blank Base POS cells, and all
41 General values. The three superseded Base POS zeros were cleared and
replaced with missing-result provenance notes. All 79 result/missing cells
have exact provenance notes, and wrapping plus the blank spacer row are
unchanged.

A downstream contract check caught that Belebele must be labelled
`acc_norm`, not generic Accuracy. The sealed `acc_norm,none` values equal
the raw accuracies in these GDN outputs, so the numeric headlines remain
`0.2289`, but correcting the label makes the comparison parser include
them. Exact readback now resolves 40 General and 34 reportable corrected-Base
references plus three blank Base POS references in `Comparison Data`, with
zero errors. `Variant Comparison` shows the promoted General values, and
both chart-source sheets recalculate with zero formula errors. Belebele Tso
has no pre-existing comparison/chart row; its canonical GDN result is still
present in `GDN Results!G27`.
