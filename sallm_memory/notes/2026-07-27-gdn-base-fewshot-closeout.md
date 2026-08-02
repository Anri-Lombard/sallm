# GDN base few-shot closeout — 2026-07-27

## Decision

Arrays `1118020` (2-shot) and `1118021` (3-shot) have completed every
non-NER index. The accepted completed results below are held-out test metrics
from the exact GDN base checkpoint
`anrilombard/sallm-gated-deltanet-125m-shallowwide-4x40-20260707`.
Canonical sheet headlines use the highest prompt and are explicitly labelled
`best prompt`; means, ranges, winning prompts, and every prompt score remain
below as provenance.

Zulu few-shot INJOngoIntent is quarantined. Its validation demonstration pool
contains visibly shifted labels, including
`Ungangiphakamisela ukudla okuvela eGhana? -> car_rental`. Its numerical scores
are retained only for audit and were not promoted.

## Artifact inventory and protocol audit

- 2-shot root:
  `/scratch/lmbanr001/masters/sallm/results/eval/gdn_125m_base_2shot_{sib_all,injongointent_all,sa_general_all,belebele_afr,belebele_eng,belebele_sot,belebele_ssw,belebele_tsn,belebele_tso,belebele_xho,belebele_zul}_r1/`
- 3-shot root:
  `/scratch/lmbanr001/masters/sallm/results/eval/gdn_125m_base_3shot_{masakhanews_all,sib_all,injongointent_all,sa_general_all,belebele_afr,belebele_eng,belebele_sot,belebele_ssw,belebele_tsn,belebele_tso,belebele_xho,belebele_zul}_r1/`
- Each root has `evaluation_summary.json`; each pack has its corresponding
  `<task_pack>/results.json`.
- The summary wrapper's top-level `fewshot: 0` is stale pack metadata.
  Every detailed task config records the executed `num_fewshot` as exactly
  `2` or `3`.
- Every accepted task/language has all five prompts. Row coverage per prompt:
  News Eng `948`, News Xho `297`; SIB each language `204`; Intent Eng `622`
  and Sot/Xho/Zul `640`; Belebele each language `900`; AfriMMLU `500`;
  AfriXNLI `600`; AfriMGSM `250`.
- Evaluation split is `test` throughout. Few-shot demonstrations use
  `validation` for News, SIB, AfriMMLU, and AfriXNLI. Intent uses the task's
  configured non-test demonstration pool, but Zulu is corrupt and quarantined.
  Belebele and AfriMGSM expose no separate few-shot split in the task config;
  lm-eval therefore draws demonstrations from the benchmark's available
  evaluation pool. These are benchmark-style few-shot results, not clean
  train-demo estimates.
- Mapping checks passed: News retains its seven topic choices; SIB its seven
  topic choices; Intent its fixed 40-intent order; Belebele/AfriMMLU their
  four-choice mappings; AfriXNLI its three-class mapping; AfriMGSM its numeric
  target and flexible numeric extractor. Prompt IDs are complete and mapped to
  prompts 1-5 without duplication or omission.
- Multiple-choice packs have a score for every row and no missing outputs.
  AfriMGSM has no missing rows, but raw generation empties are material outside
  prompt 1. Empty flexible-extract responses by language order
  `Sot/Zul/Xho/Eng` are:
  - 2-shot P1 `0/0/0/0`, P2 `199/58/157/176`, P3 `250/231/241/209`,
    P4 `216/136/168/191`, P5 `136/96/209/108`.
  - 3-shot P1 `0/0/0/0`, P2 `184/40/135/139`, P3 `250/221/243/188`,
    P4 `194/94/145/140`, P5 `114/79/200/86`.
  The promoted AfriMGSM best prompt is P1 for every language and shot count,
  and P1 has zero empty responses.
- Representative checks are qualitatively consistent with the metrics.
  A 2-shot Sotho AfriMGSM P1 example correctly returns `40`; a 3-shot Sotho
  P4 example correctly returns `10`. Failures include copying a demonstration
  answer instead of solving the query and invalid/empty generations on weak
  AfriMGSM prompts. Multiple-choice samples contain valid choice likelihoods;
  inspected Belebele and SIB failures are ordinary wrong argmax choices rather
  than label-map or missing-output failures.

## Complete prompt evidence

Metric is weighted F1 for News, SIB, and Intent; accuracy for Belebele,
AfriMMLU, and AfriXNLI; flexible exact match for AfriMGSM. Quarantined Zulu
Intent rows are shown for audit only.

| Result | Task/language | Prompt scores P1-P5 | Mean | Range | Best prompt |
|---|---|---|---:|---|---|
| 2-shot | belebele_afr | 0.218889 / 0.222222 / 0.236667 / 0.220000 / 0.217778 | 0.223111 | 0.217778-0.236667 | P3 (0.236667) |
| 2-shot | belebele_eng | 0.242222 / 0.237778 / 0.260000 / 0.228889 / 0.203333 | 0.234444 | 0.203333-0.260000 | P3 (0.260000) |
| 2-shot | belebele_sot | 0.262222 / 0.262222 / 0.254444 / 0.208889 / 0.234444 | 0.244444 | 0.208889-0.262222 | P1/P2 (0.262222) |
| 2-shot | belebele_ssw | 0.282222 / 0.276667 / 0.280000 / 0.284444 / 0.285556 | 0.281778 | 0.276667-0.285556 | P5 (0.285556) |
| 2-shot | belebele_tsn | 0.271111 / 0.246667 / 0.246667 / 0.240000 / 0.247778 | 0.250444 | 0.240000-0.271111 | P1 (0.271111) |
| 2-shot | belebele_tso | 0.228889 / 0.228889 / 0.228889 / 0.228889 / 0.228889 | 0.228889 | 0.228889-0.228889 | P1/P2/P3/P4/P5 (0.228889) |
| 2-shot | belebele_xho | 0.274444 / 0.271111 / 0.275556 / 0.272222 / 0.268889 | 0.272444 | 0.268889-0.275556 | P3 (0.275556) |
| 2-shot | belebele_zul | 0.274444 / 0.272222 / 0.273333 / 0.266667 / 0.267778 | 0.270889 | 0.266667-0.274444 | P1 (0.274444) |
| 2-shot | intent_eng | 0.006358 / 0.010375 / 0.003230 / 0.003567 / 0.004209 | 0.005548 | 0.003230-0.010375 | P2 (0.010375) |
| 2-shot | intent_sot | 0.007167 / 0.007120 / 0.005605 / 0.005965 / 0.004696 | 0.006110 | 0.004696-0.007167 | P1 (0.007167) |
| 2-shot | intent_xho | 0.004104 / 0.003770 / 0.002937 / 0.003118 / 0.004801 | 0.003746 | 0.002937-0.004801 | P5 (0.004801) |
| 2-shot | intent_zul — quarantined | 0.006620 / 0.009705 / 0.005144 / 0.003902 / 0.002586 | 0.005591 | 0.002586-0.009705 | P2 (0.009705) |
| 2-shot | afrimgsm_eng | 0.008000 / 0 / 0 / 0 / 0 | 0.001600 | 0-0.008000 | P1 (0.008000) |
| 2-shot | afrimgsm_sot | 0.008000 / 0 / 0 / 0 / 0 | 0.001600 | 0-0.008000 | P1 (0.008000) |
| 2-shot | afrimgsm_xho | 0.012000 / 0 / 0 / 0 / 0 | 0.002400 | 0-0.012000 | P1 (0.012000) |
| 2-shot | afrimgsm_zul | 0.012000 / 0 / 0 / 0 / 0 | 0.002400 | 0-0.012000 | P1 (0.012000) |
| 2-shot | afrimmlu_eng | 0.228000 / 0.228000 / 0.226000 / 0.238000 / 0.244000 | 0.232800 | 0.226000-0.244000 | P5 (0.244000) |
| 2-shot | afrimmlu_sot | 0.240000 / 0.230000 / 0.238000 / 0.260000 / 0.248000 | 0.243200 | 0.230000-0.260000 | P4 (0.260000) |
| 2-shot | afrimmlu_xho | 0.224000 / 0.246000 / 0.252000 / 0.252000 / 0.250000 | 0.244800 | 0.224000-0.252000 | P3/P4 (0.252000) |
| 2-shot | afrimmlu_zul | 0.254000 / 0.272000 / 0.268000 / 0.262000 / 0.262000 | 0.263600 | 0.254000-0.272000 | P2 (0.272000) |
| 2-shot | afrixnli_eng | 0.323333 / 0.318333 / 0.315000 / 0.318333 / 0.323333 | 0.319667 | 0.315000-0.323333 | P1/P5 (0.323333) |
| 2-shot | afrixnli_sot | 0.323333 / 0.330000 / 0.326667 / 0.316667 / 0.325000 | 0.324333 | 0.316667-0.330000 | P2 (0.330000) |
| 2-shot | afrixnli_xho | 0.323333 / 0.323333 / 0.331667 / 0.331667 / 0.323333 | 0.326667 | 0.323333-0.331667 | P3/P4 (0.331667) |
| 2-shot | afrixnli_zul | 0.323333 / 0.316667 / 0.326667 / 0.321667 / 0.321667 | 0.322000 | 0.316667-0.326667 | P3 (0.326667) |
| 2-shot | sib_afr | 0.175019 / 0.164017 / 0.165874 / 0.171633 / 0.167731 | 0.168855 | 0.164017-0.175019 | P1 (0.175019) |
| 2-shot | sib_eng | 0.157977 / 0.164701 / 0.185139 / 0.164938 / 0.189282 | 0.172407 | 0.157977-0.189282 | P5 (0.189282) |
| 2-shot | sib_nso | 0.167888 / 0.164076 / 0.173688 / 0.152255 / 0.159589 | 0.163499 | 0.152255-0.173688 | P3 (0.173688) |
| 2-shot | sib_sot | 0.158895 / 0.158396 / 0.176102 / 0.168189 / 0.156130 | 0.163543 | 0.156130-0.176102 | P3 (0.176102) |
| 2-shot | sib_xho | 0.161209 / 0.156642 / 0.157029 / 0.154100 / 0.178525 | 0.161501 | 0.154100-0.178525 | P5 (0.178525) |
| 2-shot | sib_zul | 0.165610 / 0.144518 / 0.153251 / 0.169324 / 0.169495 | 0.160439 | 0.144518-0.169495 | P5 (0.169495) |
| 3-shot | belebele_afr | 0.234444 / 0.255556 / 0.228889 / 0.236667 / 0.225556 | 0.236222 | 0.225556-0.255556 | P2 (0.255556) |
| 3-shot | belebele_eng | 0.248889 / 0.251111 / 0.250000 / 0.224444 / 0.217778 | 0.238444 | 0.217778-0.251111 | P2 (0.251111) |
| 3-shot | belebele_sot | 0.263333 / 0.267778 / 0.241111 / 0.221111 / 0.235556 | 0.245778 | 0.221111-0.267778 | P2 (0.267778) |
| 3-shot | belebele_ssw | 0.280000 / 0.274444 / 0.281111 / 0.271111 / 0.278889 | 0.277111 | 0.271111-0.281111 | P3 (0.281111) |
| 3-shot | belebele_tsn | 0.276667 / 0.266667 / 0.266667 / 0.255556 / 0.265556 | 0.266222 | 0.255556-0.276667 | P1 (0.276667) |
| 3-shot | belebele_tso | 0.226667 / 0.228889 / 0.228889 / 0.228889 / 0.227778 | 0.228222 | 0.226667-0.228889 | P2/P3/P4 (0.228889) |
| 3-shot | belebele_xho | 0.270000 / 0.281111 / 0.287778 / 0.271111 / 0.261111 | 0.274222 | 0.261111-0.287778 | P3 (0.287778) |
| 3-shot | belebele_zul | 0.276667 / 0.284444 / 0.277778 / 0.278889 / 0.255556 | 0.274667 | 0.255556-0.284444 | P2 (0.284444) |
| 3-shot | intent_eng | 0.007526 / 0.004192 / 0.006155 / 0.006076 / 0.002218 | 0.005233 | 0.002218-0.007526 | P1 (0.007526) |
| 3-shot | intent_sot | 0.007514 / 0.005924 / 0.008254 / 0.008429 / 0.005358 | 0.007096 | 0.005358-0.008429 | P4 (0.008429) |
| 3-shot | intent_xho | 0.013603 / 0.005623 / 0.008178 / 0.008825 / 0.007132 | 0.008672 | 0.005623-0.013603 | P1 (0.013603) |
| 3-shot | intent_zul — quarantined | 0.012240 / 0.007848 / 0.006186 / 0.007856 / 0.009994 | 0.008825 | 0.006186-0.012240 | P1 (0.012240) |
| 3-shot | news_eng | 0.159248 / 0.144162 / 0.155117 / 0.145837 / 0.158935 | 0.152660 | 0.144162-0.159248 | P1 (0.159248) |
| 3-shot | news_xho | 0.173962 / 0.194341 / 0.200861 / 0.189557 / 0.208333 | 0.193411 | 0.173962-0.208333 | P5 (0.208333) |
| 3-shot | afrimgsm_eng | 0.012000 / 0 / 0 / 0.004000 / 0 | 0.003200 | 0-0.012000 | P1 (0.012000) |
| 3-shot | afrimgsm_sot | 0.016000 / 0 / 0 / 0.004000 / 0 | 0.004000 | 0-0.016000 | P1 (0.016000) |
| 3-shot | afrimgsm_xho | 0.012000 / 0 / 0 / 0.004000 / 0 | 0.003200 | 0-0.012000 | P1 (0.012000) |
| 3-shot | afrimgsm_zul | 0.012000 / 0 / 0 / 0.004000 / 0 | 0.003200 | 0-0.012000 | P1 (0.012000) |
| 3-shot | afrimmlu_eng | 0.228000 / 0.232000 / 0.236000 / 0.200000 / 0.228000 | 0.224800 | 0.200000-0.236000 | P3 (0.236000) |
| 3-shot | afrimmlu_sot | 0.232000 / 0.234000 / 0.224000 / 0.236000 / 0.232000 | 0.231600 | 0.224000-0.236000 | P4 (0.236000) |
| 3-shot | afrimmlu_xho | 0.234000 / 0.226000 / 0.224000 / 0.226000 / 0.232000 | 0.228400 | 0.224000-0.234000 | P1 (0.234000) |
| 3-shot | afrimmlu_zul | 0.230000 / 0.220000 / 0.228000 / 0.218000 / 0.230000 | 0.225200 | 0.218000-0.230000 | P1/P5 (0.230000) |
| 3-shot | afrixnli_eng | 0.348333 / 0.340000 / 0.345000 / 0.351667 / 0.356667 | 0.348333 | 0.340000-0.356667 | P5 (0.356667) |
| 3-shot | afrixnli_sot | 0.348333 / 0.340000 / 0.331667 / 0.315000 / 0.328333 | 0.332667 | 0.315000-0.348333 | P1 (0.348333) |
| 3-shot | afrixnli_xho | 0.348333 / 0.333333 / 0.340000 / 0.345000 / 0.341667 | 0.341667 | 0.333333-0.348333 | P1 (0.348333) |
| 3-shot | afrixnli_zul | 0.348333 / 0.325000 / 0.315000 / 0.316667 / 0.338333 | 0.328667 | 0.315000-0.348333 | P1 (0.348333) |
| 3-shot | sib_afr | 0.189734 / 0.202373 / 0.187715 / 0.199410 / 0.195287 | 0.194904 | 0.187715-0.202373 | P2 (0.202373) |
| 3-shot | sib_eng | 0.209384 / 0.200191 / 0.164509 / 0.184568 / 0.184767 | 0.188684 | 0.164509-0.209384 | P1 (0.209384) |
| 3-shot | sib_nso | 0.161001 / 0.141294 / 0.173609 / 0.157301 / 0.146935 | 0.156028 | 0.141294-0.173609 | P3 (0.173609) |
| 3-shot | sib_sot | 0.158107 / 0.167585 / 0.151011 / 0.150712 / 0.127400 | 0.150963 | 0.127400-0.167585 | P2 (0.167585) |
| 3-shot | sib_xho | 0.183388 / 0.170738 / 0.170406 / 0.171166 / 0.160462 | 0.171232 | 0.160462-0.183388 | P1 (0.183388) |
| 3-shot | sib_zul | 0.194372 / 0.166321 / 0.175257 / 0.169252 / 0.150121 | 0.171065 | 0.150121-0.194372 | P1 (0.194372) |

## Canonical sheet write

Spreadsheet `1Ph_zVcSuLZy0dUBnDPF4tkLqAfEVCwK9JVybB2_8x6U`, tab
`GatedDeltaNet Results` (`sheetId=1825869548`):

- `C2:D3`: before `26 Jul` with 3-shot News `Not run`; after `27 Jul` with
  Eng `0.1592` P1 and Xho `0.2083` P5, both labelled `best prompt`.
- `C10:D26`: before `16 Jul` and `Queued` throughout; after `27 Jul` with
  accepted SIB, non-Zulu Intent, and seven existing Belebele rows. Zulu Intent
  is explicitly `Quarantined`; no score was promoted.
- `C28:D39`: before `16 Jul` and `Queued` throughout; after `27 Jul` with
  accepted AfriXNLI, AfriMMLU, and AfriMGSM best-prompt headlines.
- The live ranges were reread after the batch and match the intended values.
  Only `userEnteredValue` changed, so formulas, validation, and formatting were
  preserved. The target cells have wrapped text and no formulas or validation.
- Belebele Tso is complete (`2-shot 0.228889`, all prompts tied;
  `3-shot 0.228889`, P2/P3/P4 tied) but the GDN sheet has no Tso row. No row
  was inserted.

## Live state at closeout pass

- HEX quota-first check: home `32.1%`, scratch `89.1%`.
- NER indices `1118020_1` and `1118021_1` remain the only running array cells.
  At the final snapshot they were `9577/10760` (`89%`, ETA about `40m`) and
  `9481/10760` (`88%`, ETA about `44m`).
- xLSTM jobs `1118353` NER, `1118354` POS, and `1118355` Intent remain
  dependency-held behind both arrays.
- Held jobs `1117467/1117468` remain untouched.
- Kombuys base POS is still running in prompt 4 Zulu (`100/601` at the final
  snapshot); the final artifact
  `/scratch/alombard/masters/sallm/results/final/gdn_pos_constrained_base_test_20260727.json`
  had not landed at the last check and was not promoted.
- Mamba/LLaMA Intent recovery and the GDN NER batch-size canary remain blocked
  on fresh explicit external-compute approval. No GPU work was created.
