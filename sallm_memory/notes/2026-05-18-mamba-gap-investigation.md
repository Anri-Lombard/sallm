# Mamba Gap Investigation Plan

## Position

We should not yet say "Mamba is bad at T2X, AfriHG, POS, and NER" as a clean
architecture conclusion. The evidence is stronger and more nuanced:

- Mamba can work: MasakhaNews is strong after eval-template correction, and T2X
  improves substantially when the recovered recipe is used.
- Mamba is behind LLaMA on generation tasks even after recovery.
- POS/NER failures are dominated by output control and exact-shape generation,
  not only by inability to learn under teacher forcing.

The next work should explicitly rule out base-model quality, checkpoint
selection, implementation, decoding, and task-formulation failures before moving
to pretraining hyperparameters or architecture changes.

## Why T2X and AfriHG may lag LLaMA

### 1. Base model quality may be lower

If the Mamba base loss is worse than LLaMA's, downstream full fine-tuning starts
from a weaker language model. T2X and AfriHG are generation tasks that require
both language modelling and source conditioning, so worse base quality can
remain visible after fine-tuning.

Tests:

- Compare Mamba vs LLaMA base validation loss/perplexity by language.
- Compare tokenizer fertility by language: mean subword pieces per word or
  sentence for Xho/Zul/Eng.
- Compare base zero-shot/few-shot raw outputs for T2X and AfriHG prompts.
- Compare fine-tune train/eval loss curves for matched recipes.

### 2. Attention may help source-conditioned copying and alignment

T2X and AfriHG are not pure continuation tasks. They require the decoder to
condition tightly on source tokens, preserve named entities, copy or transform
content, and stop at the right length. A Transformer can attend directly over
the source context at every decoding step. A pure Mamba compresses prior tokens
into recurrent state, so it may lose precise source-token addressability unless
the state dimension/depth/training strongly supports it.

Tests:

- Create copy, reorder, and extract-title diagnostics for Xho/Zul/Eng.
- Score exact source-token copy rate, named-entity preservation, length ratio,
  repetition, and hallucination.
- Compare Mamba base, Mamba fine-tuned, LLaMA base, and LLaMA fine-tuned.

### 3. Checkpoint selection may be wrong for Mamba generation

Because trainer-side generation metrics had a Mamba cache failure
(`Mamba2Cache.float`), some runs used `eval_loss` rather than validation chrF
for checkpoint selection. For generation tasks, the lowest eval loss checkpoint
need not be the best chrF checkpoint.

Tests:

- Save multiple checkpoints for the best T2X/AfriHG recipe.
- Run lightweight validation lm-eval on each checkpoint.
- Select by validation chrF and then test once.
- If possible, fix the generation-metric path so Mamba can select by chrF
  during training.

### 4. Current HPO may still be too narrow

T2X showed recipe sensitivity. AfriHG improved under the recovered recipe but
remained weak. Lower LR/no smoothing failed, but that only rules out one nearby
recipe, not the full recipe space.

Minimal next HPO:

- Around recovered recipe: LR near `3e-4`, `4.4e-4`, `6e-4`.
- Label smoothing: `0.0`, `0.02`, `0.05`.
- Epochs/checkpoints: save each epoch and select by validation chrF.
- Sequence length and prompt budget: test shorter source windows and length
  penalties.
- Decode: beam length penalty, repetition penalty, no-repeat-ngram.

### 5. Implementation/generation path may be hurting Mamba

We have seen generation-cache friction and OOM fallback. This does not prove
wrong results, but it is enough to require parity checks.

Tests:

- Run HF vs official `mamba_ssm` logits parity on a fixed checkpoint and prompt.
- Run short generation parity on the same prompt.
- Confirm tokenizer, special tokens, chat template, dtype, and cache behavior.
- Confirm eval batch fallback does not change metrics except speed.

## POS and NER: how to improve output shape while staying decoder-only

The problem is not that decoder-only models cannot do POS/NER. The issue is
that the current free-generation formulation asks the model to emit an exact
length- and label-constrained sequence. Mamba often learns under teacher forcing
but fails at autonomous generation.

### Candidate fixes

1. **Constrained decoding / structured generation**
   - Restrict generated tokens to the allowed tag vocabulary plus separators and
     EOS after the answer prefix.
   - Use Transformers `prefix_allowed_tokens_fn` or a custom `LogitsProcessor`.
   - Equivalent open-source terms: constrained decoding, guided decoding,
     structured generation, grammar-constrained decoding.
   - This remains decoder-only if it only constrains decoding, but it must be
     reported as a constrained-decoding variant rather than the default
     free-generation score.

2. **Dynamic length control**
   - Compute expected number of tags from the input token list.
   - Set dynamic `max_new_tokens` and enforce an EOS/stop sequence after exactly
     N tags.
   - Penalize or truncate only in a predeclared metric variant.

3. **Per-token decoder-only loglikelihood tagging**
   - For each source token, prompt the decoder-only LM to choose/generate one
     tag, or score candidate tags by loglikelihood.
   - This avoids a classifier head and remains decoder-only.
   - It is closer to multiple-choice lm-eval than free-form generation.
   - Needs matched Mamba and LLaMA evaluation.

4. **Better tag language**
   - Add atomic label tokens or delimiter-stable labels.
   - Train targets like `TAG=NN|TAG=VB|...` or one tag per line with explicit
     position indices.
   - Use short target-only loss and strong EOS examples.

5. **Curriculum**
   - Start with short sentences and exact one-token labels.
   - Add longer examples after exact-length behavior stabilizes.
   - For NER, oversample non-O entities or use span-first then BIO conversion.

6. **Raw-output diagnostics**
   - Track exact length match, parseable rate, nonempty rate, repetition,
     first-k token accuracy, strict token accuracy, and entity-level F1.
   - Keep official lm-eval free-generation metrics separate from diagnostics.

## Minimum closure criteria before architecture/pretraining changes

- T2X/AfriHG:
  - validation-chrF checkpoint selection tried or generation-metric path fixed;
  - source-copy/length/repetition diagnostics done;
  - base model quality compared to LLaMA.

- POS/NER:
  - free-generation result recorded;
  - constrained-decoding or dynamic-length decoder-only variant tried;
  - per-token decoder-only loglikelihood variant tried;
  - matched LLaMA control run under the same formulation.

- Implementation:
  - HF-vs-official Mamba generation/logit parity checked for a representative
    checkpoint.

Only after these are done should we move to Mamba-2, hybrid attention-SSM,
xLSTM/ExcelSTM, or new pretraining hyperparameters as the primary explanation.
