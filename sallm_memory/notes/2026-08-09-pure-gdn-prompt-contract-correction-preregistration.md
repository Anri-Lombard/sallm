# Pure-GDN prompt-contract correction preregistration — 2026-08-09

## Status and contamination disclosure

This protocol is frozen before any corrected validation or held-out metric is
computed. The pathological marker-only output from the first T2X correction
artifact has already been observed and triggered this audit. That held-out
score and output quality must not be used to choose between prompt variants.
The correction below is instead determined mechanically from the canonical
pretraining data and tokenizer contracts.

No held-out adapter evaluation is authorized by this document.

## Source facts that determine the correction

1. Canonical pretraining used `anrilombard/mzansi-text-tokenized` through
   `src/conf/base/gated_deltanet_125m_pure.yaml`.
2. Direct read-only inspection of that dataset's validation rows shows that
   every inspected document begins with token `0`, `[BOS]`, and ends with token
   `1`, `[EOS]`.
3. The tokenizer post-processor is `[BOS] $A [EOS]`; the pretokenization path
   used normal tokenizer calls and therefore materialized those boundaries.
4. The canonical base tokenizer has only `[BOS]`, `[EOS]`, `[PAD]`, and
   `[UNK]` as special tokens. `<|system|>`, `<|user|>`, and `<|assistant|>` are
   absent from both the added-token table and the vocabulary as atomic tokens.
   They tokenize as ordinary punctuation and word pieces.
5. The base model was pretrained on ordinary text documents, not SALLM chat
   conversations. Fine-tuning later adds the three chat markers, resizes the
   embeddings, and explicitly trains the new token rows through PEFT.
6. The current fallback chat template does not insert `[BOS]`. The base
   lm-eval runner also forces `add_bos_token=false`. Thus both base chat-wrapped
   prompts and adapter training/evaluation prompts violate the observed
   document-start contract.

## Frozen correction

### Base-model evaluation

The canonical raw base model must be evaluated as a plain-text continuation
model, not as a chat model.

- lm-eval task packs use their existing `doc_to_text` prompt verbatim with
  `apply_chat_template=false`.
- lm-eval prepends exactly one `[BOS]` through `add_bos_token=true` and does not
  append `[EOS]` to the incomplete prompt.
- generation tasks use `prompt_format=raw`: prepend exactly one `[BOS]`, then
  concatenate the configured system text (when present) and user/demo message
  contents in their existing order with two newline separators. Do not insert
  role markers or a terminal `[EOS]`.
- The model, checkpoint, BF16 precision, zero-shot setting, datasets, splits,
  task prompt wording, decoding configuration, metrics, and hardware envelope
  otherwise remain unchanged.

### Fine-tuning and adapter evaluation

The canonical SALLM chat template must begin with exactly one `bos_token`.
Existing message-level `[EOS]` separators and the terminal assistant generation
marker remain, but the incomplete generation prompt must not end in `[EOS]`.
Because fine-tuning adds and trains the chat-marker embeddings, chat formatting
remains appropriate only after that resizing/training step.

All train, validation, constrained-scoring, callback, harness, and adapter
evaluation paths must share the same exact token IDs.

## Required regression gates

1. Dataset-contract fixture: representative pretraining rows start in `[BOS]`
   and end in `[EOS]`.
2. Chat fixture: direct `apply_chat_template(..., tokenize=True)` and rendered
   retokenization with `add_special_tokens=false` are identical, start in
   `[BOS]`, and do not end in `[EOS]` at the generation marker.
3. Raw-base fixture: prompt IDs start in exactly one `[BOS]`, contain no chat
   role-marker token sequence, and do not end in `[EOS]`.
4. lm-eval fixture: chat packs resolve to `add_bos_token=false` because BOS is
   explicit in the template; raw base packs resolve to
   `add_bos_token=true`.
5. Existing parser, constrained POS, macro-F1, AfriHG-label, General coverage,
   equal-family NLL, canonical-model, no-adapter, and provenance tests remain
   passing.

## Scientific recovery scope

- All 16 base lanes are quarantined and rerun once after immutable deployment.
  News, NER, POS, Intent, all eight Belebele lanes, T2X, and AfriHG used chat
  wrapping or chat generation. SIB-200 and SA-general used raw lm-eval packs,
  but the historical runner unconditionally forced `add_bos_token=false`, so
  they also omitted the required document-start token.
- Every Multilingual HPO family is rerun validation-only after the BOS-correct
  chat contract. Previously reconciled News/SIB/Intent winners are provenance,
  not ratified winners, because their training inputs lacked BOS.
- General is rerun only after all previously preregistered label, coverage, and
  aggregation corrections remain in place.
- Monolingual training and all held-out adapter evaluation stay blocked until
  all corrected validation winners are frozen.

No prompt variant may be selected from T2X, AfriHG, or any other held-out test
metric. Synthetic/token-identity tests and validation-only canaries are the
only admissible implementation evidence.
