# GDN POS best-prompt reporting correction - 2026-07-27

## Decision

The canonical Google Sheet headline for each complete prompt-based result is
the maximum held-out prompt score, labelled `best prompt`. Prompt means remain
descriptive provenance, not the headline. No prompt was selected during HPO:
all five artifacts below are completed held-out `test` evaluations under
`closed_label_token_logprob`, and this reporting correction does not change
model or checkpoint selection.

Base GDN POS remains pending. Historical list-target POS results remain
quarantined and were not changed.

## Shared provenance

- Dataset/split: `masakhane/masakhapos`, `test`
- Base checkpoint:
  `anrilombard/sallm-gated-deltanet-125m-shallowwide-4x40-20260707`
- Metric/protocol: token accuracy,
  `closed_label_token_logprob`, tuple contract, mean label-logprob scoring
- Prompt templates:
  1. `masakhane_pos_tagging/lm_eval_p1`
  2. `masakhane_pos_tagging/lm_eval_p2`
  3. `masakhane_pos_tagging/lm_eval_p3`
  4. `masakhane_pos_tagging/lm_eval_p4`
- Complete count per prompt:
  Xhosa `601` sentences / `9,749` tokens; Zulu `601` / `9,226`;
  Tswana `602` / `15,484`
- Artifacts on Kombuys:
  - `/scratch/alombard/masters/sallm/results/final/gdn_pos_constrained_multi_test_20260726.json`
  - `/scratch/alombard/masters/sallm/results/final/gdn_pos_constrained_mono_xho_test_20260726.json`
  - `/scratch/alombard/masters/sallm/results/final/gdn_pos_constrained_mono_zul_test_20260726.json`
  - `/scratch/alombard/masters/sallm/results/final/gdn_pos_constrained_mono_tsn_test_20260726.json`
  - `/scratch/alombard/masters/sallm/results/final/gdn_pos_constrained_general_test_20260727.json`
- Adapters:
  - multilingual:
    `anrilombard/sallm-gated_deltanet-masakhane-masakhapos-tsn-xho-zul`
  - Xhosa/Zulu/Tswana monolingual:
    `anrilombard/sallm-gated_deltanet-masakhane-masakhapos-{xho,zul,tsn}`
  - general: `anrilombard/sallm-gated_deltanet-sa_general-all`

## Prompt evidence

Scores are ordered prompt 1, 2, 3, 4.

| Language | Adapter | Prompt scores | Mean | Range | Winning prompt/template |
| --- | --- | --- | ---: | --- | --- |
| Xho | monolingual | `0.236127`, `0.235204`, `0.222690`, `0.226998` | `0.230254` | `0.222690-0.236127` | prompt 1, `masakhane_pos_tagging/lm_eval_p1` |
| Xho | multilingual | `0.783465`, `0.785516`, `0.782850`, `0.784080` | `0.783978` | `0.782850-0.785516` | prompt 2, `masakhane_pos_tagging/lm_eval_p2` |
| Xho | general | `0.810237`, `0.809519`, `0.807980`, `0.809211` | `0.809237` | `0.807980-0.810237` | prompt 1, `masakhane_pos_tagging/lm_eval_p1` |
| Zul | monolingual | `0.727292`, `0.725667`, `0.722632`, `0.730327` | `0.726480` | `0.722632-0.730327` | prompt 4, `masakhane_pos_tagging/lm_eval_p4` |
| Zul | multilingual | `0.805008`, `0.802840`, `0.802298`, `0.800672` | `0.802704` | `0.800672-0.805008` | prompt 1, `masakhane_pos_tagging/lm_eval_p1` |
| Zul | general | `0.822567`, `0.819857`, `0.818773`, `0.820399` | `0.820399` | `0.818773-0.822567` | prompt 1, `masakhane_pos_tagging/lm_eval_p1` |
| Tsn | monolingual | `0.810320`, `0.811160`, `0.807220`, `0.810385` | `0.809771` | `0.807220-0.811160` | prompt 2, `masakhane_pos_tagging/lm_eval_p2` |
| Tsn | multilingual | `0.801279`, `0.803345`, `0.800116`, `0.800181` | `0.801230` | `0.800116-0.803345` | prompt 2, `masakhane_pos_tagging/lm_eval_p2` |
| Tsn | general | `0.818070`, `0.818070`, `0.813937`, `0.817424` | `0.816875` | `0.813937-0.818070` | prompts 1 and 2 tied, `lm_eval_p1` / `lm_eval_p2` |

## Canonical sheet correction

Spreadsheet:
`1Ph_zVcSuLZy0dUBnDPF4tkLqAfEVCwK9JVybB2_8x6U`,
`GatedDeltaNet Results!E7:G9`.

| Cell | Before | After |
| --- | --- | --- |
| E7 | `0.2303` four-prompt mean | `0.2361` best prompt, prompt 1 |
| F7 | `0.7840` four-prompt mean | `0.7855` best prompt, prompt 2 |
| G7 | `0.8092` four-prompt mean | `0.8102` best prompt, prompt 1 |
| E8 | `0.7265` four-prompt mean | `0.7303` best prompt, prompt 4 |
| F8 | `0.8027` four-prompt mean | `0.8050` best prompt, prompt 1 |
| G8 | `0.8204` four-prompt mean | `0.8226` best prompt, prompt 1 |
| E9 | `0.8098` four-prompt mean | `0.8112` best prompt, prompt 2 |
| F9 | `0.8012` four-prompt mean | `0.8033` best prompt, prompt 2 |
| G9 | `0.8169` four-prompt mean | `0.8181` best prompt, prompts 1/2 tied |

## Pending broader audit

The broader architecture-wide reporting scan is intentionally left for the
orchestrator. Do not relabel or replace unsupported historical cells without
their underlying prompt artifacts and split provenance.
