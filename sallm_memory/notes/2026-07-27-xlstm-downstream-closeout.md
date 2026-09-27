# xLSTM downstream closeout inventory — 2026-07-27

## Scope and policy

- Canonical workbook: `1Ph_zVcSuLZy0dUBnDPF4tkLqAfEVCwK9JVybB2_8x6U`,
  tab `XLSTM Results` (`sheetId=1004873333`), live range `A1:J44`.
- Canonical checkout/evidence root:
  `/Users/anrilombard/Desktop/Masters/sallm`.
- Report the highest prompt as the headline. Retain prompt mean, range, and
  winning prompt in provenance.
- Select recipes/checkpoints on validation only. Evaluate held-out test once
  after freezing the selection. Diagnostic task-head results remain a separate
  protocol and do not replace the original decoder-only matrix.

## Live compute state at 10:49 SAST

- HEX quota: home `32.1%`; scratch `88.0%`.
- Active work is A100-only GDN arrays `1118020` and `1118021`.
- Held duplicate GDN jobs `1117467` and `1117468` were not touched.
- No xLSTM job was running before this audit.
- Three xLSTM base closeout jobs were submitted on A100-80 with
  `afterany:1118020:1118021`. They remain dependency-held and therefore cannot
  compete with the active GDN arrays:
  - `1118353`: corrected held-out MasakhaNER base evaluation.
  - `1118354`: corrected closed-label MasakhaPOS base evaluation.
  - `1118355`: corrected held-out InjonGoIntent base evaluation.

## Exact gap matrix

| Sheet cells / row | Status | Evidence or required action |
|---|---|---|
| Base MasakhaNER: Xho, Zul, Tsn (`D4:D6`) | Queued | Old 2/3-shot artifacts used validation/unverified splits. Job `1118353` uses the frozen 3-epoch base and the corrected test-only task YAMLs. |
| Base MasakhaPOS: Xho, Zul, Tsn (`D7:D9`) | Queued | Historical list-target scorer is invalid. Job `1118354` uses four prompts, the fixed 17-label set, mean token-logprob scoring, and held-out test. |
| Base InjonGoIntent: Eng, Xho, Zul, Sot (`D16:D19`) | Queued | Historical split provenance is unverified/partly non-test. Job `1118355` reruns the current held-out task pack at zero-shot; zero-shot avoids the known corrupted Zulu validation demonstrations. |
| News mono: Eng, Xho (`E2:E3`) | Invalid/quarantined | Historical xLSTM LoRA adapters predate selective training of the added chat-role token rows and the classification left-truncation repair. |
| News multi: Eng, Xho (`F2:F3`) | Invalid/quarantined | Adapter `ft_xlstm_125m_news_all/7mq0snuc` is the affected historical adapter. Corrected validation audit job `1091598` collapsed to Eng F1 `0.044444` and Xho F1 `0.172658`; the promoted test values are not defensible. |
| News general: Eng, Xho (`G2:G3`) | Invalid/quarantined | The general adapter was trained through the same historical chat-token/truncation path. A corrected general-adapter retrain would affect the full general matrix and was not launched as a “small missing” job. |
| AfriHG mono: Xho, Zul (`E42:E43`) | Invalid/quarantined | Right truncation loses the complete assistant headline in `906/24,649` train rows (`3.68%`). The defect is material even though it does not fully explain collapse. |
| AfriHG multi: Xho, Zul (`F42:F43`) | Invalid/quarantined | Same target-truncation exposure; validation-selected retraining is required before one frozen test evaluation. |
| AfriHG general: Xho, Zul (`G42:G43`) | Invalid/quarantined | Same exposure in the historical general mix. Do not promote without a corrected general-adapter retrain. |
| Orphan base value (`D40`) | Invalid/quarantined provenance | Cell contains `2-shot: 4.73 chrF; 3-shot: 4.50 chrF` but has no task or language key in `A40:B40`. Do not report it until its source row is reconstructed. |
| Belebele Tso row | Sheet-only missing, compute complete | Completed jobs: base 2-shot `898491`, base 3-shot `899427`, general `896579`. Base 2-shot headline `0.243333` (prompt 3), mean `0.234889`, range `0.227778-0.243333`; base 3-shot headline `0.260000` (prompt 2), mean `0.243333`, range `0.236667-0.260000`; general headline `0.271111` (prompt 5), mean `0.262222`, range `0.251111-0.271111`. The workbook omitted the row; no rerun is needed. |
| AfriHG Eng (`A44:G44`) | Source-blocked/unavailable | The row is blank and the prior English source path is known to fail. This is not a defensible zero and was not resubmitted. |

All other populated xLSTM sheet result cells remain complete under their
recorded protocol, subject to their existing comparability notes. Blank
mono/multi cells for general-only multiple-choice tasks and Belebele are
structural `N/A`, not missing experiments.

## Queued artifacts and dependencies

| Job | Output | Dependency | Expected runtime after release |
|---|---|---|---|
| `1118353` | `/scratch/lmbanr001/masters/sallm/results/eval/xlstm_downstream_closeout_20260727/base/masakhaner_all_test/evaluation_summary.json` | `afterany:1118020:1118021` | roughly 1-3 h; 12 h limit |
| `1118354` | `/scratch/lmbanr001/masters/sallm/results/eval/xlstm_downstream_closeout_20260727/base/masakhapos_constrained_test.json` | `afterany:1118020:1118021` | roughly 3-5 h; 12 h limit |
| `1118355` | `/scratch/lmbanr001/masters/sallm/results/eval/xlstm_downstream_closeout_20260727/base/injongointent_all_test/evaluation_summary.json` | `afterany:1118020:1118021` | roughly 1-3 h; 6 h limit |

The active GDN NER long poles projected approximately 2.5-3 hours remaining
at submission. Remaining GDN array indices are short multiple-choice packs.
Working ETA for the three xLSTM base artifacts is approximately
`17:00-20:00 SAST` on 2026-07-27, scheduler permitting. No exact scheduler
start time exists while the dependencies are unfulfilled.

## Existing exact evidence

- Corrected xLSTM HPO held-out wave:
  `/scratch/lmbanr001/masters/sallm/results/eval/xlstm_hpo_test_20260624`
  (jobs `952247-952252`; compact artifacts were later removed from scratch,
  but manifests, logs, and the recorded summaries remain).
- Corrected historical News validation audit:
  `/scratch/lmbanr001/masters/sallm/results/news_rootcause/xlstm_historical_fixed_val_retry4/evaluation_summary.json`,
  job `1091598`, adapter
  `/scratch/lmbanr001/masters/sallm/checkpoints/ft_xlstm_125m_news_all/7mq0snuc/final_adapter`.
- Omitted Belebele Tso logs:
  `/home/lmbanr001/masters/sallm/slurm-898491.out`,
  `/home/lmbanr001/masters/sallm/slurm-899427.out`, and
  `/home/lmbanr001/masters/sallm/slurm-896579.out`.
- Separate held-out task-head branch, not a sheet replacement:
  xLSTM trainable-head NER F1 `0.4155` (job `955146`) and SIB macro F1
  `0.7022` (job `955145`).

## Remaining dependency-safe work

1. Let `1118353-1118355` finish, audit completeness/splits/prompts, then report
   each language’s highest prompt with mean/range/winning prompt.
2. Do not rerun News test yet. Retrain corrected mono/multi News adapters,
   select on validation, freeze, then evaluate test once.
3. Do not rerun AfriHG yet. First implement or reuse a target-preserving
   truncation protocol and validate it against the current right-truncation
   control; then retrain/select on validation and touch test once.
4. Treat a corrected general-adapter retrain as a separate full-matrix
   dependency, not as part of the three small base jobs.
