# Evaluation token-boundary and runtime compatibility audit — 2026-08-30

## Confirmed findings

- The August BOS/raw correction is not safe to port unchanged. With the
  frozen SALLM tokenizer and lm-eval 0.4.9.2, `add_bos_token=true` becomes
  `add_special_tokens=True`, and the tokenizer postprocessor adds both BOS
  and EOS. lm-eval then slices an incomplete continuation ending in EOS.
- The same correction renders an explicit BOS for classification and then
  re-tokenizes with default special-token handling, producing two BOS tokens
  and a terminal EOS in the actual evaluator path.
- Draft PR #126's Transformers 5.15.1 lock rejects the checked-in
  LLaMA-125M base configuration (`hidden_size=512`,
  `num_attention_heads=9`) before model construction. The existing synthetic
  compatibility test does not exercise that real configuration. All other
  checked-in base configurations construct under the proposed lock.

## Required corrections

- Issue #132 is implemented in reviewed draft PR #134. It uses one canonical
  fallback without replacing model-specific templates, prevents repeated
  special-token insertion, gives raw lm-eval a BOS-only tokenizer, and fails
  closed if the corrected tokenizer cannot load or save. It must rebase after
  PR #126 before its inherited Grype failure can clear.
- Issue #133 is resolved in green, reviewed draft PR #126 without changing
  architecture values. A strict local compatibility class accepts only the
  frozen `512/9/3/56` tuple, preserves checkpoint tensor shapes and
  save/reload, and delegates all other LLaMA validation upstream. The draft
  remains unmerged pending explicit confirmation to mark it ready.

## Result impact

- Quarantine and prospectively rerun the 14 corrected raw lm-eval base lanes:
  News, NER, POS, SIB, Intent, SA-general, and eight Belebele lanes.
- Corrected base T2X/AfriHG raw generation is unaffected.
- Frozen NER, POS, T2X, and AfriHG adapter winners are unaffected.
- General validation HPO and post-hoc LLaMA T2X HPO are unaffected.
- News/SIB/Intent post-BOS HPO has not started and must use the corrected
  evaluator path.
- Historical base held-out outputs already exist but remain excluded from every
  selection. No adapter held-out output was inspected; adapter held-out access
  in the current program remains zero. Sheet E/F/G remain blank.
