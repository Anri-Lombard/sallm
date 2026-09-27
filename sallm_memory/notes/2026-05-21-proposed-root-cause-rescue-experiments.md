# Proposed Root-Cause Rescue Experiments

Created: 2026-05-21 20:26 SAST.

Status: proposal only. No jobs submitted and no implementation started.

Goal: design experiments that can tell us whether Mamba's weak tasks are
fixable through decoder-only formulation/decoding/metric hygiene, or whether
the remaining gap is mainly base-model quality. The experiments should produce
reviewable evidence for the supervisors before we spend on larger base
pretraining.

## Design Principles

- Use validation or small diagnostic subsets first; promote to official test
  only after validation-selected settings improve the right metric and the
  qualitative shape.
- When changing the task formulation, run matched Mamba and LLaMA controls.
  Otherwise we cannot tell whether we improved Mamba specifically or made the
  task easier for any decoder-only model.
- Keep all rescue attempts decoder-only: no encoder, CRF, classifier head, or
  non-comparable architecture.
- Record both metric changes and output-shape diagnostics:
  parseable-rate, length ratio, repetition, entity/value preservation, non-O
  recall, all-O rate, and sample outputs.
- Separate "can train/run" implementation repairs from scientific results.
- Promote only final optimized rows to the Google Sheet; keep diagnostics in
  the vault.

## Experiment Set A: POS/NER Output-Shape Rescue

Root-cause hypothesis: POS/NER are currently failing because free generation
does not obey a strict tag-output shape. A better base model may help
representations, but it will not by itself guarantee exact tag sequences.

### A1. Per-token decoder-only label scoring

Question: If we remove free-form generation and ask the decoder-only model to
choose among allowed labels token by token, does POS/NER recover?

Design:

- For each input token, score the allowed label strings with teacher-forced
  loglikelihood under the fine-tuned model.
- POS labels: the UPOS set, e.g. `ADJ`, `ADP`, `NOUN`, `PROPN`, `VERB`, etc.
- NER labels: either BIO labels or atomic labels, e.g. `O`, `B-PER`, `I-PER`,
  `B-LOC`, etc.
- Choose the highest-scoring label per token.
- Run on Mamba current best POS/NER checkpoints.
- Run matched LLaMA checkpoints if available under the same scoring method.

Metrics:

- Parseable rate: should be `100%` by construction.
- POS token accuracy.
- NER BIO/entity F1.
- NER non-O recall.
- All-O rate.
- Length/exact-label-count is also `100%` by construction.

Decision:

- If Mamba improves sharply, the main bottleneck was output control, not the
  model's ability to represent the task.
- If Mamba remains poor while LLaMA is strong, the issue is deeper than output
  formatting: likely base quality, representations, or fine-tuning recipe.
- If both are weak, the decoder-only prompt/formulation or label scoring
  method is flawed.

Priority: highest. This is the cleanest POS/NER root-cause test.

### A2. Constrained tag-sequence decoding

Question: Can we keep generation but constrain the decoder to only emit valid
tag labels and separators?

Design:

- Generate a label sequence with an allowed-token trie or constrained decoding
  over legal tag labels and separators.
- Dynamically cap output length from the number of input tokens.
- Enforce exactly one label per input token where possible.
- Compare greedy constrained decoding against per-token label scoring.

Metrics:

- Parseable rate.
- Exact length rate.
- POS token accuracy.
- NER F1 and non-O recall.
- Runtime/memory overhead.

Decision:

- If constrained decoding works nearly as well as label scoring, we can keep a
  generation-style decoder-only formulation.
- If label scoring works but constrained decoding does not, the final fair
  diagnostic should use label scoring.

Priority: high, after A1.

### A3. NER label-prior calibration

Question: Is the NER all-O collapse caused by label prior bias rather than
inability to detect entities?

Design:

- Use per-token label scoring.
- Compute label scores both with the sentence context and with a null/minimal
  context.
- Subtract or normalize by the label prior score.
- Compare raw scoring vs calibrated scoring.

Metrics:

- Non-O recall.
- All-O rate.
- Entity F1.
- False-positive rate.

Decision:

- If calibration improves non-O recall without exploding false positives,
  Mamba has entity signal but biased label priors.
- If calibration does not help, entity recognition itself is weak.

Priority: medium.

## Experiment Set B: T2X/Data-to-Text Copy and Coverage Rescue

Root-cause hypothesis: T2X is not primarily malformed-output failure; it is
source coverage/copy failure. Mamba preserves entities and values much worse
than LLaMA, and repeats more.

### B1. Token-class conditional-loss audit

Question: Does Mamba specifically assign worse likelihood to entity/value
tokens than to relation/function words?

Design:

- Use existing T2X prompts/references.
- Align reference tokens into crude classes:
  source-entity words, source-value words, relation/verbalization words, and
  other tokens.
- Compute teacher-forced NLL per class for Mamba and LLaMA.
- Run on current Mamba T2X checkpoint-selected/continued checkpoint and LLaMA
  reference.

Metrics:

- NLL/PPL by token class.
- Entity/value NLL gap: Mamba minus LLaMA.
- Relationship between token-class NLL and source-preservation failures.

Decision:

- If entity/value NLL is disproportionately bad, target copy/preservation and
  base quality.
- If all token classes are uniformly bad, base likelihood is the dominant
  explanation.

Priority: high.

### B2. Delexicalized T2X formulation

Question: Is Mamba bad at verbalizing relations, or bad at copying names and
values?

Design:

- Replace entity and value with placeholders in the prompt:
  `ENTITY_A`, `VALUE_A`.
- Train/evaluate a diagnostic formulation where the model verbalizes relation
  structure using placeholders.
- Deterministically reinsert the original entity/value after generation.
- Run matched Mamba and LLaMA controls.

Metrics:

- Placeholder preservation rate.
- Relation verbalization quality.
- chrF/BLEU/ROUGE after reinsertion.
- Entity/value preservation after reinsertion.

Decision:

- If Mamba improves sharply, the core issue is copying/proper-name/value
  handling rather than relation semantics.
- If it remains weak, relation planning or base quality is the issue.

Priority: high, but requires a small formulation implementation after review.

### B3. Source-preservation decoding diagnostics

Question: Can decoding/prompting rescue entity/value preservation without
retraining?

Design:

- On validation, compare current best decode against:
  - lower temperature/greedy;
  - beam with conservative length penalty;
  - prompt with explicit checklist: include entity and value;
  - no-repeat/repetition penalty variants.
- Do not promote unless both chrF and preservation metrics improve.

Metrics:

- Entity word preservation.
- Value word preservation.
- Repetition rate.
- chrF/BLEU/ROUGE.

Decision:

- If decoding helps preservation but hurts chrF, report as diagnostic only.
- If preservation and chrF both improve, promote selected setting to official
  test rerun.

Priority: medium. Previous T2X length-control diagnostics were not promising,
so B1/B2 are more informative.

## Experiment Set C: AfriHG Headline Length and Focus Rescue

Root-cause hypothesis: AfriHG is mostly under-generation and weak hook/focus,
not malformed output. Validation already suggested beam/length-penalty can help.

### C1. Official test rerun for validation-selected beam/length settings

Question: Do the AfriHG validation beam/length gains survive on official test?

Design:

- Use current best Mamba AfriHG checkpoints:
  - Xho checkpoint-656.
  - Zul checkpoint-892.
- Use validation-selected decode setting, currently beam5/length-penalty style
  settings that improved validation chrF and length ratio.
- Run official test only for selected settings.
- Consider matched LLaMA decode variant if formulation/decoding comparison
  needs fairness.

Metrics:

- chrF/BLEU/ROUGE-L.
- Length ratio.
- Repetition.
- Named-entity/hook preservation by heuristic.

Decision:

- If test improves, update Mamba AfriHG final candidate rows after review.
- If not, keep checkpoint-selected greedy/current as final and treat beam gains
  as validation-only.

Priority: high. This is the most immediately actionable AfriHG rescue.

### C2. Headline hook/focus diagnostics

Question: Are weak headlines missing the main entity/event, or only too short?

Design:

- For each article/reference/prediction, compute crude hook preservation:
  - named-looking tokens from reference and article;
  - number/date/event tokens;
  - overlap with Mamba and LLaMA predictions.
- Build a small manual error table for 20 Xho and 20 Zul cases.

Metrics:

- Reference-entity overlap.
- Article-top-token overlap.
- Date/number preservation.
- Error labels: generic, wrong focus, entity missing, date/event missing,
  repetition, hallucinated entity.

Decision:

- If missing-entity/hook dominates, test prompt/formulation that asks for the
  main actor/event.
- If length dominates with preserved hook, decoding length controls are enough.

Priority: high, local/offline.

### C3. Prompted headline focus variant

Question: Can a stricter decoder-only prompt reduce generic headlines?

Design:

- Validation-only matched Mamba/LLaMA diagnostic.
- Prompt examples:
  - "Write a 4-8 word headline. Include the main person, team, or event when
    present."
  - "Do not write a generic country headline."
- Keep output headline-only.

Metrics:

- chrF/BLEU/ROUGE-L.
- Length ratio.
- Hook/entity preservation.
- Generic headline rate.

Decision:

- If Mamba improves and LLaMA does not move much, prompt formulation is a
  Mamba-specific rescue.
- If both improve, use matched formulation for final comparison or mark as
  formulation diagnostic.

Priority: medium.

## Experiment Set D: Base-Model Quality and Recipe Path

Root-cause hypothesis: the Mamba base is weaker than LLaMA before fine-tuning,
and generation gaps will persist unless the base improves. The active wide
pure-Mamba2 gate is the next evidence point.

### D1. Finish active wide pure-Mamba2 gate

Question: Does the wide public-Mamba-like shape learn better than our current
shape?

Current active jobs:

- `854987`: canary completed successfully.
- `854988`: full 20k run active.
- `854989`: clean generation-loss audit pending.

Decision metrics:

- Held-out pretraining eval loss by checkpoint.
- Clean generation-loss audit on T2X/AfriHG.
- Compare against:
  - current Mamba base;
  - LLaMA base;
  - failed/current-shape 20k fresh Mamba run.

Decision:

- If wide shape is materially better, promote it as next base candidate and
  run small downstream T2X/AfriHG fine-tune probes.
- If wide shape is not better, do not spend on full retrain yet; design a
  small HPO matrix.

Priority: already running.

### D2. Pure-Mamba base HPO matrix

Question: If the wide shape is not enough, what is the smallest HPO matrix that
can identify a better pure-Mamba recipe?

Proposed axes:

- Shape:
  - current `expand=4/state64`;
  - wide `expand=2/state128`;
  - possibly one intermediate shape if parameter count remains matched.
- LR:
  - `2e-4`, `4e-4`, maybe `6e-4` if stable.
- Warmup:
  - `1000` and maybe `2000`.
- Training horizon:
  - short screen at `5k` or `10k`;
  - continue only promising arms.

Metrics:

- Eval loss trajectory.
- Clean generation-loss audit at screen checkpoints.
- GPU hours and scratch footprint.

Decision:

- Pick one base candidate for longer pretraining only if it beats current
  Mamba base trajectory or clean generation-loss direction.

Priority: after D1.

## Experiment Set E: Implementation Parity

Root-cause hypothesis: some failures may be HF/Mamba implementation quirks,
not architecture quality.

### E1. HF-vs-official Mamba logits parity

Question: Does the HF Mamba checkpoint produce equivalent logits to the
official Mamba implementation for the same prompt/checkpoint?

Design:

- Use a representative current Mamba checkpoint if loadable in both stacks.
- Same tokenizer, same dtype, same prompt, no generation cache.
- Compare next-token logits or top-k token rankings for a handful of prompts.

Metrics:

- Max/mean logit delta.
- Top-k overlap.
- Same greedy next token rate.

Decision:

- If parity is close, HF implementation is not the root metric issue.
- If parity is poor, implementation path needs fixing before final claims.

Priority: medium-high if feasible.

### E2. Generation parity smoke test

Question: Are generation outputs stable across HF settings/cache/fallbacks?

Design:

- Compare short greedy generations:
  - cache on/off where possible;
  - fast path vs fallback where possible;
  - batch size 1 vs small batch.

Metrics:

- Exact output match or controlled differences.
- Runtime/memory notes.
- Failure modes.

Decision:

- If outputs differ, final evaluation must pin the safe setting and document
  why.

Priority: medium.

## Recommended Review Order

1. Approve A1: per-token decoder-only label scoring for POS/NER.
2. Approve C1: AfriHG official rerun for validation-selected beam/length
   settings.
3. Approve B1: T2X token-class conditional-loss audit.
4. Approve B2: delexicalized T2X formulation diagnostic.
5. Let D1 finish before deciding on any larger base run.
6. Run E1/E2 if implementation parity remains a concern or if D1 behaves oddly.

## What Each Outcome Would Mean

| Outcome | Interpretation | Next action |
|---|---|---|
| POS/NER label scoring works | Main root cause is free-generation output shape | Use label scoring/constrained decoding as matched decoder-only diagnostic |
| POS/NER label scoring fails for Mamba but works for LLaMA | Mamba representation/base/fine-tune issue remains | Try small task HPO or defer until better base |
| T2X delexicalized improves Mamba | Copying/proper-name/value burden is root cause | Consider placeholder/reinsertion formulation or copy-focused training |
| T2X delexicalized does not improve | Relation planning/base quality is root cause | Wait for base result or task-specific recipe HPO |
| AfriHG beam/length improves test | Decoding length/focus was recoverable | Promote selected decode setting |
| AfriHG beam/length fails test | Validation gain did not generalize | Keep checkpoint-selected result; focus on base/focus formulation |
| Wide base improves clean losses | Current base recipe/shape is root cause | Fine-tune wide base on T2X/AfriHG/POS/NER probes |
| Wide base fails scientifically | Pure-Mamba base HPO or architecture-family comparison needed | Design small HPO before full pretrain |
