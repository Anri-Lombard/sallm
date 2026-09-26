# Pure-GDN HPO implementation-correction preregistration — 2026-08-09

Status: **frozen before any corrected validation metric is produced**. This is
an implementation-correction addendum to the 2026-08-07 adapter protocol. It
does not authorize held-out adapter evaluation. The existing invalid generation
and General artifacts may be used only as failure provenance; held-out metrics
may not select a recipe, checkpoint, prompt, retry, correction, or
hyperparameter.

## Decision

Corrected pure-GDN HPO is **not cleared to run** until the implementation below
passes regression tests and a validation-only Kombuys canary on RTX 3080 Ti.
RTX 5090 remains untouched. After that gate, deploy one immutable code snapshot
or a complete execution manifest to HEX and run only the preregistered recovery
sequence.

This remains a bounded, one-dimensional learning-rate screen, not exhaustive
multidimensional HPO. Final reporting must use that description.

## Invariants retained from the original protocol

- Canonical base:
  `/scratch/lmbanr001/masters/sallm/checkpoints/sallm-pure-gdn-125m/a10080-3epoch-20260802/final_model`.
- Identity: pure FLA `GatedDeltaNetForCausalLM`, `attn=None`, exactly
  `127,425,448` parameters.
- Architecture-complete LoRA targets: `q_proj`, `k_proj`, `v_proj`, `a_proj`,
  `b_proj`, `g_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`; rank 16,
  alpha 32, dropout 0.05.
- BF16; seed and data seed 42; three learning rates `3e-5`, `8e-5`, and
  `1.5e-4`; AdamW betas 0.9/0.95; weight decay 0.01; cosine schedule; warmup
  ratio 0.03; maximum gradient norm 1.0; assistant-only loss; no packing.
- Early stopping remains patience 2 with absolute threshold `0.001`. Retaining
  the already preregistered threshold avoids a post-hoc rule change. A strict
  zero threshold would require rerunning the affected Intent grid and is not
  part of this correction.
- Exact ties select the lower learning rate; checkpoint ties select the earlier
  checkpoint. Seed 42 remains the selection seed. Any extra seeds are
  robustness analysis only and may not select a lucky recipe or checkpoint.
- A100-40GB `gpu:ampere` only on HEX, at most three owned jobs, with no
  A100-80GB or L40S overlap. No Hub write.

## Corrected selection contracts

### News, SIB, and Intent

- Add and select on `all_macro_f1`: compute label-macro F1 within each language
  over the complete P1--P5 prompt-expanded validation set, then take the
  arithmetic mean across languages.
- Do not use `all_f1`, which averages support-weighted per-language F1.
- Existing validation-only jobs may be reconciled without retraining because
  the stored confusion histories are sufficient. Ratification requires a
  local reconciliation artifact containing all inputs, per-run/per-checkpoint
  values, the aggregation rule, winner, and SHA-256.
- The presently reconciled winners remain News job `1183134`, LR `3e-5`;
  SIB job `1189730`, LR `3e-5`, checkpoint `526`; and Intent job `1192267`,
  LR `8e-5`, checkpoint `2643`. These are not globally frozen until the hashed
  artifact exists.

### MasakhaNER

- Rendered prompt retokenization must use `add_special_tokens=False` and must
  exactly equal direct `apply_chat_template(..., tokenize=True,
  add_generation_prompt=True)` token IDs.
- Parse entities only at explicit `$$`, `$`, or line boundaries. Parse an
  exact label field before the colon and map only complete label aliases to
  `PER`, `LOC`, `ORG`, or `DATE`. Entity text is opaque until the established
  final span-text normalization. Never split entity values on comma-period
  punctuation and never perform substring label replacement inside entity
  text.
- Regression fixtures must include `David A. Gross`, `Kazan, Russia`,
  `The Bomb Shelter Film Company`, and `Stimela`.
- Selection metric is the arithmetic mean of per-language span micro-F1 over
  the complete P1--P5 prompt-expanded validation set. The entire three-LR grid
  must rerun because the invalid metric controlled early stopping.

### MasakhaPOS

- Free generation is not a valid selection contract. Use the established
  constrained tuple evaluator: continuation-logprob scoring over the fixed
  UPOS set, mean label-token score, and exactly one legal label per input
  token.
- Primary metric is token accuracy. Evaluate the complete validation split for
  canonical prompts P1--P4, and select on the arithmetic mean of the 12
  language-by-prompt token accuracies for Tsn/Xho/Zul.
- A correct prefix plus extra labels must never receive full credit. The entire
  three-LR grid must rerun.

### T2X and AfriHG

- Apply the same corrected prompt tokenization equality contract as NER.
- T2X selects on full-validation chrF under `t2x_verbalisation/v1`, five beams,
  length penalty 1.0, and the saved model cache setting.
- AfriHG selects on the arithmetic mean of full-validation Xhosa and Zulu chrF
  under `afrihg_headline/v1`, five beams, length penalty 0.7, and the saved
  model cache setting.
- Both three-LR grids must rerun because invalid generation metrics controlled
  early stopping.

### General

- Every processed validation row must have an explicit `task_name` in
  `{sib, news, ner, pos, afrihg, t2x}`. Language may not be used as the sole
  coverage key. Assert exact raw and prompt-expanded coverage before scoring.
- The canonical selection view expands SIB and News across P1--P5, NER across
  P1--P5, POS across P1--P4, and uses the single established prompt for AfriHG
  and T2X. Expected processed rows are:

  | Family | Processed validation rows |
  | --- | ---: |
  | SIB | 2,970 |
  | News | 3,095 |
  | NER | 10,760 |
  | POS | 1,800 |
  | AfriHG | 3,082 |
  | T2X | 460 |
  | **Total** | **22,167** |

- For family `f`, compute assistant-token NLL as
  `sum(valid assistant-token NLL) / count(valid assistant tokens)` over all its
  validation examples and prompts. Select on the arithmetic mean of the six
  family NLL values. Thus weighting is token-level within family and equal
  across families, matching the intent of the token-balanced training mix.
- Log and persist raw row counts, valid assistant-token counts, summed NLL,
  per-family NLL, and the six-family macro NLL at every checkpoint. Any missing
  family, null task name, masked-only example, non-finite token loss, or count
  mismatch fails the evaluation rather than silently dropping data.
- Training mixture weights remain unchanged. The complete General three-LR
  grid must rerun. Jobs `1204261`, `1204262`, and `1207524` are provenance
  only because their loss omits all 3,082 AfriHG rows and is sample-weighted.

## Required validation-only canary

Before any corrected HEX HPO submission, run a bounded canary on Kombuys RTX
3080 Ti with RTX 5090 untouched. It must prove:

1. direct and rendered-template token IDs are exactly equal for representative
   NER, POS, T2X, and AfriHG prompts;
2. no terminal EOS is appended after the assistant generation marker;
3. the NER parser fixtures and a delimiter-faithful full-validation reference
   audit pass;
4. POS returns exactly one legal label per input token under P1--P4 and the
   scorer rejects prefix-plus-extra output;
5. General coverage is exactly 22,167 processed rows with all six task names,
   and a toy/manual calculation matches the equal-family token-NLL result;
6. canonical model identity, no adapter/base mix-up, BF16 loading, and output
   provenance are recorded.

Non-empty or higher-scoring canary output is not itself a success criterion;
the canary validates implementation and contracts, not recipe quality.

## Frozen recovery order

1. Implement the correction and regression tests without consulting held-out
   metrics.
2. Pass the validation-only Kombuys canary above on RTX 3080 Ti.
3. Deploy an immutable commit or complete imported-source/launcher/config/
   environment hash manifest to HEX.
4. Rerun quarantined base generation lanes 14 (T2X) and 15 (AfriHG) once under
   the otherwise frozen zero-shot base protocol; retain invalid originals as
   provenance and update only verified replacement Sheet cells.
5. Produce and hash the News/SIB/Intent validation reconciliation.
6. Rerun the NER, POS, T2X, AfriHG, and General validation grids.
7. Freeze all eight winners in a signed or hashed manifest.
8. Train and validation-select each applicable Monolingual adapter at its
   already frozen family LR.
9. Touch each applicable Mono/Multi/General held-out test once, then update
   `Pure GatedDeltaNet Results` incrementally from verified artifacts.

No held-out adapter test, Sheet adapter-column write, or Hugging Face
publication is authorized before these gates pass.
