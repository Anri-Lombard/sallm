# Gated DeltaNet Research For SALLM, 2026-06-21

## Recommendation

Do not spend a full SALLM pretraining run on Gated DeltaNet next. The defensible move is a
small feasibility and learning-curve screen only, and HGRN2 still has the stronger low-resource
case because BabyHGRN directly tests sample-efficient language modeling.

If a GDN screen is approved, test a FLA-based Gated DeltaNet hybrid before a pure GDN full run.
Pure fixed-state recurrence is exactly where SALLM has already seen Mamba-style copying and
source-preservation weakness. The current external trend is also hybrid: Qwen3-Next uses 3 GDN
layers followed by 1 gated-attention layer, while Kimi Linear uses a 3:1 KDA-to-global-MLA
layout.

## What Gated DeltaNet Is

Gated DeltaNet is a linear-attention / recurrent-memory architecture. It keeps a fixed-size
matrix state instead of a growing softmax-attention KV cache. DeltaNet updates this state with a
delta rule: before writing a new value at a key, it subtracts the current value read from that
key, giving targeted overwrite rather than plain accumulation. Gated DeltaNet adds Mamba2-style
decay/gating, so the state can both forget broadly and edit a selected association.

In rough terms:

- Transformer attention keeps token-level access to past keys and values. Strong for copying and
  source-conditioned generation, but quadratic or cache-heavy.
- DeltaNet uses a matrix fast-weight memory and delta updates. Better associative recall than
  plain linear attention, but without learned global decay.
- Gated DeltaNet combines DeltaNet's targeted edit with Mamba2-style adaptive forgetting.
- Gated DeltaNet-2 decouples erase and write into separate channel-wise gates, extending both
  GDN and Kimi Delta Attention.
- RetNet uses retention with parallel, recurrent, and chunkwise modes, but not the delta residual
  overwrite.
- Mamba/Mamba-2 are selective SSMs / SSD-style recurrences. They decay and write compressed
  state, but they do not explicitly subtract the current read before rewriting a key-value
  association.
- RWKV is an RNN-like attention-free family with dynamic recurrence and matrix-valued states.
- xLSTM uses exponential gates plus scalar or matrix memory; it is recurrent, but not a
  key-value delta-rule memory.
- HGRN/HGRN2 are gated linear RNNs; HGRN2 expands state with an outer-product mechanism and has
  the clearest published low-resource/sample-efficiency evidence.

Primary sources:

- Gated DeltaNet paper: https://arxiv.org/abs/2412.06464
- ICLR/OpenReview page: https://openreview.net/forum?id=r8H7xhYPwz
- Official GDN repo: https://github.com/NVlabs/GatedDeltaNet
- Gated DeltaNet-2 paper: https://arxiv.org/abs/2605.22791
- Gated DeltaNet-2 repo: https://github.com/NVlabs/GatedDeltaNet-2
- Flash Linear Attention repo: https://github.com/fla-org/flash-linear-attention
- DeltaNet paper: https://arxiv.org/abs/2406.06484
- Mamba: https://arxiv.org/abs/2312.00752
- Mamba-2: https://arxiv.org/abs/2405.21060
- RetNet: https://arxiv.org/abs/2307.08621
- RWKV Eagle/Finch: https://arxiv.org/abs/2404.05892
- xLSTM: https://arxiv.org/abs/2405.04517
- HGRN2: https://arxiv.org/abs/2404.07904
- BabyHGRN: https://arxiv.org/abs/2412.15978
- Kimi Linear / KDA: https://arxiv.org/abs/2510.26692
- Qwen3-Next model card: https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct
- Copying caveat: https://arxiv.org/abs/2402.01032
- Mamba empirical study: https://arxiv.org/abs/2406.07887

## Evidence Table

| Evidence | What it says | SALLM relevance |
| --- | --- | --- |
| GDN paper, ICLR 2025 | GDN combines Mamba2 gating with DeltaNet delta updates and reports better language modeling, reasoning, retrieval, length extrapolation, and long-context results than Mamba2 and DeltaNet. | Strong reason to screen. Not direct low-resource evidence. |
| GDN official repo | GDN is available as code; FLA is recommended for faster kernels and varlen training. The standalone repo gives no pretrained weights and says evaluation is inconvenient until converting to HF/FLA. | Integration is doable but not drop-in for SALLM. |
| GDN-2, 2026 | At 1.3B/100B FineWeb-Edu, GDN-2 beats Mamba-2, GDN, KDA, and Mamba-3 variants on average LM/reasoning/retrieval; biggest gains are RULER multi-key retrieval. | Best technical evidence for the family, but new code and no low-resource SA evidence. |
| Qwen3-Next | Production-scale hybrid: 48 layers repeat 3 GDN blocks then 1 gated-attention block; 262K native context, extendable to about 1M. | Industry validation favors hybrid GDN, not pure GDN. Too large/MoE-heavy to transfer directly. |
| Kimi Linear | KDA extends GDN with finer-grained gating and reports full-attention-beating hybrid results with large training runs. | Supports the direction but points to KDA/GDN-2, not necessarily vanilla GDN. |
| DeltaNet | Delta-rule linear transformers improve associative recall and LM metrics over Mamba/GLA in 1.3B/100B settings. | Mechanism targets SALLM's Mamba copy/source gap. |
| Copying/SSM literature | Transformers have an advantage on exact copying because fixed-state models compress context. Mamba hybrids recover more than pure SSMs on Phonebook/long-context retrieval. | SALLM T2X/AfriHG need source preservation, so pure fixed-state GDN is risky. |
| BabyHGRN | HGRN2 outperforms transformer baselines in 10M/100M word BabyLM tracks across BLiMP/EWoK/GLUE/BEAR. | Stronger direct sample-efficiency evidence than GDN for low-resource thesis framing. |
| Local SALLM state | Existing repo has Mamba, Mamba2-hybrid, xLSTM, RWKV configs; no FLA/GDN dependency/config. Mamba source-preservation gaps are already documented. | A GDN run needs an integration spike and should be judged against Mamba/xLSTM/LLaMA gates. |

## Data Efficiency And Low-Resource Fit

Evidence for data efficiency is suggestive, not proven. GDN and GDN-2 show better loss and recall
than nearby efficient architectures at 400M/1.3B and 100B-ish token regimes, but that is not the
same as low-resource South African-language pretraining. The closest direct low-resource evidence
among the candidates remains BabyHGRN, not GDN.

For SALLM, the positive hypothesis is narrow: GDN's targeted overwrite may reduce interference in
compressed memory, improving source entity/value preservation compared with Mamba. That matters
for T2X and AfriHG. The counter-hypothesis is just as important: if exact copying from prompt
context is the real bottleneck, GDN may still lose to LLaMA unless attention layers are retained.

No paper I found makes a direct claim about GDN being unusually good for South African languages,
morphologically rich low-resource languages, or MzansiText-like mixtures. Treat any morphology
benefit as a hypothesis, not a literature-backed claim.

## Implementation Risks

- Dependency gap: SALLM currently depends on `transformers`, `mamba-ssm`, and `xlstm`, but not
  `fla-core` / Flash Linear Attention.
- Code path: the official GDN repo recommends FLA for performance and variable-length training.
  That means the sensible implementation path is FLA-first, not porting the standalone NVLabs
  code.
- License: the NVLabs GDN and GDN-2 repos use NVIDIA Source Code License-NC. FLA is MIT, but the
  exact model code path and any copied config need license care.
- Kernels: high performance depends on Triton/FLA/causal-conv-style kernels. CPU or unsupported
  GPU paths may be slow or incomplete.
- Evaluation: official GDN notes warn that generation/evaluation can be inconvenient and prompt
  sensitive for recall tasks. This is a real risk because SALLM downstream evaluation already has
  fragile generation/scoring edges.
- Stability: GDN-2 uses fp32 decay computation, fp32 recurrent state across chunks/decoding, L2
  query/key normalization, and precision-sensitive WY solves. These choices need parity checks in
  BF16 training.
- Checkpoints: no pretrained GDN weights are released in the official GDN repo. SALLM would train
  from scratch and must build its own HF/export/eval path.
- Context length: SALLM's current recurrent baselines mostly train at 2048 context. GDN's strongest
  published win is long-context retrieval, so a 2048-only run may under-test the point.
- Tokenizer: GDN does not solve tokenization. The same 65K ByteLevel BPE should be used for fair
  comparison, but morphology/entity-copy metrics must be checked explicitly.

## Hypotheses To Test

1. Base learning curve: a 125M-ish GDN hybrid should beat current Mamba base validation loss by
   10K steps and should not trail xLSTM by more than a small margin.
2. Conditional generation loss: on clean T2X Xho, AfriHG Xho, and AfriHG Zul teacher-forced gates,
   GDN should be closer to xLSTM/LLaMA than to the current Mamba base.
3. Source preservation: after a tiny downstream adaptation, GDN should improve source entity/value
   coverage over Mamba by at least 20% relative without lowering chrF.
4. Hybrid necessity: GDN plus periodic attention should beat pure GDN on source-copy and
   prompt-conditioned tasks. If not, the architecture story is weak.
5. Morphology/tagging: GDN should improve MasakhaPOS constrained token accuracy or exact-length
   sequence behavior over Mamba. If it only improves long recall but not POS/NER/T2X/AfriHG, it is
   not the right next thesis run.
6. Throughput: a GDN screen must reach acceptable tokens/sec and memory versus Mamba2-hybrid and
   xLSTM. If kernels are fussy, stop early.

## Smallest Defensible Experiment Plan

No jobs should be submitted from this research pass.

1. Integration spike, no full training:
   - Add a branch-only FLA dependency and instantiate one 125M-ish `GatedDeltaNetForCausalLM` or
     equivalent FLA model with the SALLM vocab.
   - Verify one forward/backward pass, parameter count, BF16 behavior, generation, HF/FLA eval
     route, and save/load.
   - Stop if FLA cannot support the exact needed training/eval path cleanly.

2. Cheap base screen:
   - Train one GDN hybrid for the same budget used by previous 10K screens.
   - Use the same tokenizer, splits, optimizer family, and base clean-loss probes.
   - Include a pure-GDN variant only if it is no extra integration work.

3. Gate criteria:
   - Continue only if base validation and clean conditional loss beat Mamba clearly and are at least
     competitive with xLSTM.
   - Require source-preservation and generation-shape checks before any official downstream claim.

4. Full-run decision:
   - Full GDN pretraining is a go only if the screen passes base loss, clean generation loss,
     source preservation, and throughput.
   - Otherwise, spend the next architecture slot on HGRN2/BabyHGRN-style work.

## Final Call

GDN is worth a small screen, not a full next run. The most defensible SALLM ordering is:

1. Finish/interpret current xLSTM gates.
2. Prioritize HGRN2 if the goal is low-resource/sample-efficiency evidence.
3. Run GDN only as a bounded hybrid feasibility screen aimed at the known Mamba source-preservation
   gap.
4. Prefer FLA GDN for maturity; keep GDN-2/KDA as the next refinement if vanilla GDN integration is
   easy and the first screen is positive.
