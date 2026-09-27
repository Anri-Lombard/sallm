# Mamba Root-Cause Rescue Evidence Matrix

Updated: 2026-05-24 20:30 SAST.

Scope: decoder-only Mamba rescue work only. This matrix summarizes what is
currently defensible, where LLaMA controls matter, and the base-follow-up
decision after the completed D1/D2 gates.

## Current Position

- POS/NER collapse is not solved by output-shape controls alone.
- T2X remains a Mamba-specific source-preservation weakness under current
  decoder-only training/evaluation.
- AfriHG has a defensible Mamba decoding rescue: keep C1 beam5/lp1.2.
- The stricter AfriHG focus prompt helps validation but did not survive cleanly
  on official test; do not adopt it as final.
- Fresh base gates so far are negative. D2 first arm, D2 wide lower-LR shape
  arm, and D5 Waleffe-style Mamba-2 hybrid all closed negative; lower
  LR/longer warmup, the tested wide pure-Mamba shape, and a shallow attention
  hybrid are not enough at the tested 10k-step budget.
- D4 full-data atomic tag-sequence LoRA closed negative: more task data and
  the best prior POS/NER canary recipes do not rescue those downstream failures
  under the current pure-Mamba base.
- HF-vs-official Mamba parity is unresolved; final claims should be scoped to
  the current HF-Mamba implementation/config path unless official parity is
  later repaired.
- Architecture pivot evidence has landed: strict-125M HF xLSTM (`hidden=736`,
  12 layers, `126,901,952` params) completed the 10k streaming screen and is
  ambiguous/promising. It beats current Mamba base on T2X clean loss but not
  on AfriHG Xho/Zul, and LLaMA base remains better on all three clean-loss
  audits.

## Evidence Matrix

| Area | Completed gates | Mamba result | LLaMA/control result | Carry-forward | Decision |
| --- | --- | --- | --- | --- | --- |
| POS | A1/A2/D4 | Label scoring/formulation helped diagnose shape but did not rescue task quality. D4 full-data atomic LoRA remains weak: best validation token accuracy `0.1960`, exact length `0.0400`, and the better nonempty arm overgenerates (`1.1486`). | LLaMA control needed for final fair downstream comparison. | Negative. | POS is not rescued under current pure-Mamba decoder-only task adaptation. |
| NER | A1/A2/A3/D4 | NER still shows all-O / label-prior collapse tendencies. D4 full-data atomic LoRA gives validation token accuracy `0.4252`, non-`O` recall `0.1772`, all-`O` rate `0.9375`, and BIO F1 `0.1769`, i.e. no material improvement over the prior narrow canary signal. | LLaMA can exploit some formulations better, strengthening Mamba-specific root-cause suspicion. | Negative. | Need different base/architecture, not more of this decoder-only atomic LoRA route. |
| T2X | B1/B2/B3 | B1 shows disproportionate loss on source entity/value tokens; B2 placeholders fail for Mamba; B3 source-checklist/repetition decoding does not rescue source preservation. | B2 is much more usable for LLaMA; B3 LLaMA baseline remains far stronger on chrF/entity/value coverage. | Mamba-only weakness; B2 is LLaMA-only diagnostic gain; B3 negative. | Root cause points to source-token learning/copying, likely base/task learning rather than simple decoding. |
| AfriHG decoding | C1/C1b/C2 | C1 beam5/lp1.2 improves Mamba official test Xho/Zul over prior Mamba baselines. C2 shows remaining generic/wrong-focus errors. | C1b says beam5/lp1.2 is not a shared LLaMA improvement; it overgenerates/does not cleanly help LLaMA. | Mamba-only. | Keep beam5/lp1.2 as current Mamba AfriHG recipe. |
| AfriHG prompt focus | C3/C4 | C3 validation improves Mamba, especially Zul. C4 official test is mixed: Xho worse, Zul tiny gain. | C3 had focus-prompt LLaMA controls but no matched non-focus validation comparator sufficient to call shared. | Validation-only / language-specific idea, not final. | Do not adopt focus_v1 for final AfriHG. |
| Implementation hygiene | E1/E2 | E1 HF-vs-official logits parity fails under simple mapping. E2 shows Mamba generation is not perfectly invariant to cache/batch settings. | Not a direct LLaMA rescue; this is Mamba implementation/eval hygiene. | Caveat/hygiene. | Scope claims to HF-Mamba path; pin/document generation settings. |
| Base model | D1, prior fresh probes, D2 first arm, D2 wide arm, D5 hybrid | Fresh 3k, 20k current-shape, D1 wide shape, D2 current-shape lower-LR/warmup, D2 wide lower-LR/warmup, and D5 Waleffe-style 8.3%-attention Mamba-2 hybrid all remain poor relative to current Mamba base and LLaMA base on clean generation loss. D2 current-shape checkpoint-10000 PPLs are T2X Xho `11925.1`, AfriHG Xho `26758.1`, AfriHG Zul `28854.2`; D2 wide best clean-loss checkpoint is checkpoint-5000 with PPLs `9983.6`, `21768.7`, `20378.4`; D5 hybrid checkpoint-10000 is T2X Xho `12364.7`, AfriHG Xho `14020.5`, AfriHG Zul `14634.8`. Current Mamba base remains far lower at `347.8`, `438.5`, `605.9`. | LLaMA base remains much lower perplexity/clean loss: `64.0`, `205.6`, `267.5` on the same audits. | Negative for this exact implementation and budget. LR/warmup-only current-shape rescue fails, the tested wide-shape rescue fails, and the tested shallow-attention hybrid does not rescue the base. | D3 cannot run unless a later base arm finds a credible base candidate; D5 is an architecture-follow-up data point, not a usable base. Audit hybrid implementation before making a broad hybrid-negative claim. |
| xLSTM architecture pivot | xLSTM h736 10k screen | Strict-125M HF xLSTM checkpoint-10000 reaches trainer eval loss `3.310756`. Clean-loss PPLs are T2X Xho `283.95`, AfriHG Xho `627.67`, AfriHG Zul `852.06`. This beats current Mamba base on T2X (`347.77`) but trails current Mamba on AfriHG Xho/Zul (`438.50`, `605.85`). | LLaMA base remains stronger on all three clean-loss audits: T2X Xho `64.20`, AfriHG Xho `205.48`, AfriHG Zul `267.32`. | Ambiguous/promising architecture-follow-up. | xLSTM deserves downstream adaptation or a longer base screen; not yet a base-wide replacement. |

## Current Root-Cause Read

1. Classification/generation gap is not one single bug.
2. Mamba can be made better on some generation tasks through decoder settings
   (AfriHG C1), so the system is not globally broken.
3. Mamba still struggles where the task demands stable source-conditioned token
   preservation or label-prior control: T2X source values/entities and NER non-O
   tags are the clearest cases.
4. Fresh-base attempts so far made downstream likelihood worse, not better,
   even when held-out pretraining loss improved. D5 hybrid is the cleanest
   example: trainer eval reached `5.6123`, but downstream clean-loss PPL stayed
   in the `12k`-`15k` range.
5. D2 now says the tested pure-Mamba base rescue path is negative: lower
   LR/warmup did not rescue the current shape, and the wide shape did not become
   competitive on clean generation loss.
6. D5 now says adding sparse attention in this Waleffe-style shallow hybrid is
   also negative at this budget: it beats pure-Mamba fresh HPO arms on
   pretraining eval loss, but not on the decisive clean generation-loss gate.
   Caveat: the tested preset has 2 attention layers out of 24 (8.3%
   attention), not 8 attention layers, and the current custom attention block
   has no explicit positional encoding/RoPE.
7. D4 now says full-data POS/NER task adaptation is also negative: the model can
   learn training tag strings, but validation remains weak or collapses to
   dominant label priors.

## Defensible Next Step

D2 wide arm, D4, and D5 are closed negative:

1. Do not run D3 from D1, D2 current-shape, or D2 wide.
2. Stop pure-Mamba base HPO here unless an exhaustive ablation table is worth
   one more low-probability run on the remaining conservative wide arm
   (`wide_lr1e4_wu1000_10k`). The tested D5 hybrid should not replace the
   current Mamba base.
3. Treat POS/NER rescue under the current pure-Mamba base as exhausted for
   decoder-only formulation/LoRA/data-scaling routes.
4. Report the base-rescue branch as negative and pivot final
   claims toward observed Mamba task weaknesses, the AfriHG decoding rescue,
   implementation-hygiene caveats, and any explicitly labelled architecture
   follow-up beyond the failed shallow Mamba-attention hybrid. Do not present
   D5 as a final negative result for hybrids generally until the positional
   attention and block-structure implementation questions are resolved.
5. Prefer xLSTM follow-up over more cheap Mamba rescue arms. The completed
   xLSTM strict-125M screen is competitive enough to justify downstream
   adaptation or a longer base screen, especially for T2X, but not strong
   enough to declare a base-wide win over the current Mamba base.

Do not run D3 from the completed D1/D2 base candidates.
