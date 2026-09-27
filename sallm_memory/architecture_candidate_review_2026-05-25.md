# Architecture Candidate Review, 2026-05-25

Purpose: record the literature-backed architecture-prioritization pass for the
SALLM master's comparison.

## Thesis Framing

The master's contribution should be framed as a controlled architecture
comparison for South African languages, not merely another model release. The
core experiment is valuable because the same South African corpus, tokenizer,
parameter scale, training budget, adaptation regimes, and task suite can be
used to compare model families.

Defensible gap statement:

- There does not appear to be a modern controlled South-African-language
  comparison of decoder-only Transformer, SSM/Mamba, xLSTM/RNN-style, and
  related efficient LM architectures trained and evaluated under the same
  setup.

Important caveat:

- Do not claim that no South African architecture comparison exists at all.
  Crous and Buys (2021) compared n-gram, feedforward neural, recurrent, and
  Transformer language models for low-resource South African languages on
  small-scale datasets.

## Architecture Priority

### Tier 1: Must have

1. LLaMA-style decoder-only Transformer.
   - Keep as the reference baseline because Transformers remain the dominant
     LLM architecture and MzansiLM already establishes the South African
     decoder-only baseline at 125M.
2. Mamba/Mamba-2.
   - Keep as the efficient SSM baseline, but make final claims carefully:
     current failures should be scoped to the exact HF Mamba path/config unless
     official implementation parity is resolved.
3. xLSTM.
   - Continue the current full-base run and downstream evaluation if the base
     is not clearly negative. xLSTM is a strong recurrent alternative with
     direct language-modeling scaling claims and practical inference-efficiency
     motivation.

### Tier 2: Best additional architecture if compute allows

4. HGRN2 / BabyHGRN.
   - This is the strongest missing candidate. BabyHGRN is directly about
     sample-efficient low-resource language modeling and compares HGRN2 against
     Transformer, LSTM, xLSTM, and Mamba.
   - Recommendation: after xLSTM audits land, do a 125M-ish HGRN2 feasibility
     probe before choosing any lower-priority architecture.

### Tier 3: Useful but lower priority

5. RWKV.
   - Mature recurrent/linear-attention family with efficient inference and
     public scale evidence. Good comparison if implementation is easy.
6. Griffin/RecurrentGemma.
   - Relevant recurrence + local attention family, but current public recipes
     are larger and less convenient for strict 125M from-scratch comparison.
7. RetNet / GLA-style retention.
   - Strong related-work context for efficient attention/recurrent hybrids, but
     only worth running if a maintained implementation slots into SALLM cheaply.
8. Byte-level or tokenizer-free models.
   - Interesting for South African morphology and tokenization burden, but this
     is likely a separate representation/tokenizer study rather than the next
     architecture-comparison slot.

## Why These Architectures Are Hypothesis-Driven

The comparison should not be framed as "try fashionable architectures." Each
architecture should answer a concrete low-resource question.

### Transformer / LLaMA-style decoder

Hypothesis:

- Dense self-attention remains the strongest default for source-conditioned
  generation, copying, in-context conditioning, and flexible downstream
  adaptation even at small scale.

Why it might matter for South African languages:

- Low-resource task examples are scarce, so the model may need to use the
  prompt/context directly rather than rely on memorized parametric knowledge.
- South African generation tasks such as T2X and AfriHG often require copying
  or preserving names, entities, numbers, and source details.

What would support it:

- Lower clean conditional generation loss;
- better entity/value preservation;
- stronger few-shot or prompt-conditioned behavior;
- better downstream adaptation at the same token budget.

What would weaken it:

- Similar downstream quality from recurrent/SSM models with lower memory,
  faster inference, or better data-efficiency curves.

### Mamba / Mamba-2

Hypothesis:

- Selective SSMs may learn useful sequence dynamics with linear scaling and
  efficient inference, potentially giving better compute or memory efficiency
  for small African-language LMs.

Why it might matter for South African languages:

- If data is limited, an architecture with strong inductive bias for sequential
  structure and efficient long-context processing could use scarce examples
  more efficiently.
- Lower inference memory may matter for practical deployment in local or
  low-resource settings.

Risk / counter-hypothesis:

- Fixed-state or compressed-state architectures may struggle with exact
  source copying, entity preservation, and content-based retrieval from the
  prompt, which are important in the current SALLM tasks.

What would support it:

- Competitive base perplexity or downstream scores at lower wall-clock,
  memory, or token budget;
- good performance on classification or sequence tasks where exact copying is
  less central;
- stable generation under documented cache/batch settings.

What would weaken it:

- Persistent conditional-loss and source-preservation gaps under matched
  training budget, tokenizer, prompt, and checkpoint-selection controls.

### xLSTM

Hypothesis:

- Modern recurrent memory with exponential gating and matrix memory may recover
  some advantages of RNNs for sample-efficient learning while remaining more
  scalable than classical LSTMs.

Why it might matter for South African languages:

- Earlier South African LM work found well-regularized RNNs strong on small
  datasets; xLSTM is a modern test of whether that RNN advantage survives in a
  decoder-only LM setup.
- Gating and recurrent memory may help model morphology, agreement, and local
  sequential regularities from fewer examples.
- xLSTM's inference-efficiency claims matter if small local language models
  should be usable outside large-server settings.

What would support it:

- Better base perplexity or downstream adaptation than Mamba at the same
  parameter/token budget;
- stronger learning curve at early checkpoints;
- competitive results with lower inference memory or faster decoding.

What would weaken it:

- Good training loss but poor task-conditioned loss/generation, or large gaps
  to the Transformer after epoch/token-budget parity is enforced.

### HGRN2 / BabyHGRN

Hypothesis:

- HGRN2's gated linear recurrence with state expansion may be especially
  sample-efficient under constrained data, making it a strong candidate for
  low-resource language modeling.

Why it might matter for South African languages:

- The BabyHGRN result is directly about low-resource/sample-efficient language
  modeling and compares against Transformer, LSTM, xLSTM, and Mamba.
- State expansion is explicitly motivated as improving recurrent memory
  capacity without adding many parameters, which is attractive under a strict
  125M budget.

What would support it:

- Better early-checkpoint validation loss than Transformer/Mamba/xLSTM;
- strong downstream adaptation after the same or smaller token exposure;
- favorable parameter-efficiency or token-efficiency curves.

What would weaken it:

- Implementation instability, poor scaling under the SALLM tokenizer/vocab
  budget, or failure to beat xLSTM/Mamba under matched training.

### RWKV

Hypothesis:

- RWKV may offer a practical recurrent/linear-attention point between
  Transformer quality and RNN-style inference efficiency.

Why it might matter for South African languages:

- It is a mature efficient-LM family and could test whether recurrent
  token-mixing, not just Mamba-style SSMs, is viable under the same corpus.

Why it is lower priority:

- The low-resource evidence is less directly compelling than BabyHGRN/HGRN2
  for this thesis, so it should be run only if integration cost is low.

### Griffin / RecurrentGemma

Hypothesis:

- Mixing local attention with recurrence may give a better compromise than
  pure recurrence or pure SSM: local self-attention can handle nearby copying
  and syntax, while recurrence lowers memory cost.

Why it might matter for South African languages:

- Many morphology and agreement cues are local, but tasks still need some
  longer-range conditioning. Local attention plus recurrence is a plausible
  middle ground.

Why it is lower priority:

- Public recipes are larger and less convenient for strict 125M from-scratch
  comparison.

### RetNet / GLA-style retention

Hypothesis:

- Retention/gated linear attention may preserve some attention-like behavior
  while enabling recurrent or chunkwise inference.

Why it might matter for South African languages:

- It is relevant if the main issue is not "attention versus recurrence" but
  "how much content-addressable memory is needed per token."

Why it is lower priority:

- Unless a maintained implementation is easy to integrate, it is better as
  related work or a future extension.

### Byte-level / tokenizer-free models

Hypothesis:

- Subword tokenization may impose uneven burden on morphologically rich or
  underrepresented South African languages; byte/character-level models could
  reduce vocabulary mismatch and improve robustness.

Why it might matter for South African languages:

- Agglutinative languages such as isiZulu and isiXhosa can produce long,
  morphologically complex word forms. If the tokenizer fragments these badly,
  model comparisons may partly measure tokenization burden rather than only
  architecture quality.

Why it is separate:

- Changing the tokenizer changes the experimental axis. It is a valuable
  follow-up, but it should not be mixed casually into the architecture
  comparison unless explicitly framed as architecture-plus-representation.

## Measurement Plan For The Hypotheses

For every architecture, report:

- base validation loss/perplexity under the same tokenizer and held-out corpus;
- clean conditional generation loss on T2X and AfriHG;
- downstream monolingual, multilingual, and general fine-tuning results;
- source-copy/entity/value preservation for generation tasks;
- token-efficiency curves at matched checkpoints;
- wall-clock, memory, and inference-throughput measurements;
- implementation caveats, especially when using HF versus official kernels.

This keeps the thesis question scientific: not "which architecture is trendy?",
but "which inductive bias gives the best quality/efficiency tradeoff for South
African low-resource language modeling and adaptation?"

## Source Trail

- MzansiText and MzansiLM: arXiv:2603.20732.
- Low-Resource Language Modelling of South African Languages: arXiv:2104.00772.
- The State of Large Language Models for African Languages: Hussen et al.,
  PMLR DLI Research Track 2026.
- AfriqueLLM: arXiv:2601.06395.
- Attention Is All You Need: arXiv:1706.03762.
- LLaMA: arXiv:2302.13971.
- Mamba: arXiv:2312.00752.
- Mamba-2 / SSD: arXiv:2405.21060.
- An Empirical Study of Mamba-based Language Models: arXiv:2406.07887.
- xLSTM: arXiv:2405.04517.
- xLSTM 7B: arXiv:2503.13427.
- BabyHGRN: arXiv:2412.15978.
- HGRN2: Gated Linear RNNs with State Expansion, COLM 2024/OpenReview.
- RWKV: Reinventing RNNs for the Transformer Era, Findings EMNLP 2023.
- RecurrentGemma / Griffin: arXiv:2404.07839.
- RetNet: arXiv:2307.08621.
