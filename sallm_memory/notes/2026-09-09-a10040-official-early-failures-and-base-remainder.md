# A100-40GB official early failures and Base remainder — 9 September 2026

Frozen at 11:08 SAST after inspecting only scheduler state, structural
sidecars, error traces, claims and file existence. No held-out metric value was
opened or used.

## Terminal units

General replacement `1326112` completed AfriMMLU, Belebele Southern Sotho and
Belebele Xhosa with structural verification, then failed `1:0` at 10:27:22
while opening AfriHG Zulu. The loader raised `RuntimeError: No AFriHG CSVs
found on GitHub for languages=['zul']`. The unit was claimed but wrote no
result file or structural sidecar. Under the frozen no-repeat rule, General
AfriHG Zulu is terminally missing. Together with the earlier terminal POS
unit, General closes at 17 structurally verified units out of 19.

Corrected Base group `1326145` started at 10:44:32 and failed `1:0` after 38
seconds while opening POS. The offline loader could not find the fixed
MasakhaPOS Tswana training URL. POS was claimed but wrote no result file or
structural sidecar, so corrected Base POS is terminally missing and must not
be retried.

## Untouched Base remainder

The four later group-2 units — AfriMGSM, Belebele Afrikaans, Belebele siSwati
and Belebele Xhosa — have no claim, output root, result file or structural
sidecar. They retain their single official access. Submit exactly one
A100-40GB `gpu:ampere` remainder in that frozen order through the unchanged
sealed Base snapshot, configs, manifests, cache, model and verifier. Preserve
the existing 48-hour envelope and eight CPUs. A failure stops the chain; no
claimed unit may be repeated.

Base group `1326143` completed `0:0` with all six assigned units structurally
verified. Group `1326144` remains healthy on NER. The remainder may run beside
it on the same GPU family within the four-job cap.

## Remainder submission

After verifying the failed POS root had zero files and all four later units
were unclaimed and output-absent, remainder `1326407` was submitted once with
the frozen order above. It is resource-pending on `gpu:ampere`; the only
running owned GPU job is Base group `1326144`. At 10:59 SAST NER was
`1601/14980` with a live estimate of about 5 hours 33 minutes remaining.

At 11:35 SAST, remainder `1326407` was running on `srvrocgpu010`; AfriMGSM
was `1137/5000` with about 1 hour 42 minutes remaining. Group `1326144`
remained healthy; NER was `3025/14980` with about 4 hours 58 minutes remaining.
No new error or structural sidecar appeared. Both jobs use `gpu:ampere`.

Remainder `1326407` completed `0:0` at 13:17:12 SAST in 2:13:10. All four
assigned sidecars report `verified=true` and `metric_values_included=false`:
AfriMGSM 20 tasks/5,000 rows, plus Belebele Afrikaans, siSwati and Xhosa at
5 tasks/4,500 rows each. Corrected Base is now 10/16 structurally verified;
POS is terminally missing and the five group-1 units remain. At 13:40 SAST,
NER was `7993/14980` with about 2 hours 54 minutes remaining. No metric value
was opened.

Group `1326144` completed `0:0` at 17:32:52 SAST in `07:15:29`. All five
assigned structural sidecars pass `verified=true` and
`metric_values_included=false`. Together with the other groups, corrected
Base closes at 15/16 verified units; only the earlier result-missing POS unit
is absent. No metric value was opened during terminal verification.
