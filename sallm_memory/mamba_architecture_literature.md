# Mamba Architecture Literature And Citation Trail

Purpose: quick citation/reference list for the SALLM architecture-comparison
write-up. Keep this as the place to collect papers used to justify the
pure-Mamba rescue attempts, the observed failure modes, and the later
Mamba-attention hybrid plan.

## Core Mamba Papers

### Mamba: Linear-Time Sequence Modeling with Selective State Spaces

- Authors: Albert Gu, Tri Dao.
- Link: https://arxiv.org/abs/2312.00752
- Why it matters here:
  - Defines the selective SSM/Mamba architecture family.
  - Claims competitive language-modeling performance and efficient inference.
  - Useful as the primary citation for the pure-Mamba baseline.
- Thesis use:
  - Background section on state-space alternatives to Transformers.
  - Explain why Mamba is a credible low-resource architecture candidate.

### Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality

- Authors: Tri Dao, Albert Gu.
- Link: https://arxiv.org/abs/2405.21060
- Why it matters here:
  - Defines Mamba-2 / SSD and the more attention-like grouped-head SSM
    formulation.
  - Supports our use of Mamba-2 rather than the original Mamba block.
- Thesis use:
  - Technical background for the `mamba2` implementation and configuration
    choices.

### Mamba-3: Improved Sequence Modeling using State Space Principles

- Authors: Aakash Lahoti, Kevin Y. Li, Berlin Chen, Caitlin Wang, Aviv Bick,
  J. Zico Kolter, Tri Dao, Albert Gu.
- Link: https://arxiv.org/abs/2603.15569
- Release note: https://pli.princeton.edu/blog/2026/mamba-3-improved-sequence-modeling-using-state-space-principles
- Why it matters here:
  - Newer pure-SSM direction with richer state dynamics and kernel work.
  - Not yet a cheap drop-in for the current HF Mamba-2 SALLM stack.
- Thesis use:
  - Future-work citation: stronger pure-SSM variants may deserve a later
    feasibility study.

## Hybrid Mamba / Attention Papers

### An Empirical Study of Mamba-based Language Models

- Authors: Roger Waleffe et al.
- Link: https://arxiv.org/abs/2406.07887
- NVIDIA page: https://research.nvidia.com/publication/2024-06_empirical-study-mamba-based-language-models
- Why it matters here:
  - Most directly relevant paper for the current hybrid plan.
  - Reports that pure Mamba/Mamba-2 can lag Transformers on copying,
    in-context learning, Phonebook lookup, and long-context reasoning.
  - Reports 130M hybrid ablations where validation loss is minimized at about
    `8%` attention layers.
  - Their 8B Mamba-2-Hybrid uses `43%` Mamba-2, `7%` attention, and `50%` MLP
    layers, with attention/MLP layers evenly distributed.
- Thesis use:
  - Justifies moving from exhausted pure-Mamba rescue gates to a
    Mamba-2/attention/MLP hybrid.
  - Supports our 125M hybrid layer pattern with `2/24 = 8.3%` attention.
  - Helps frame hybrid as a separate SSM-family architecture, not a pure-Mamba
    rescue.

### Jamba: A Hybrid Transformer-Mamba Language Model

- Link: https://arxiv.org/abs/2403.19887
- HF docs: https://huggingface.co/docs/transformers/v4.49.0/en/model_doc/jamba
- Why it matters here:
  - Independent evidence that practical SSM-family LMs often combine Mamba
    blocks with attention and MLP/MoE structure.
- Thesis use:
  - Related-work support for the hybrid direction.

### Falcon-H1 Technical Report

- Link: https://arxiv.org/abs/2507.22448
- NVIDIA implementation note: https://developer.nvidia.com/blog/implementing-falcon-h1-hybrid-architecture-in-nvidia-megatron-core/
- Why it matters here:
  - Another modern hybrid architecture reference showing industry interest in
    SSM-attention mixtures.
- Thesis use:
  - Related-work support for hybrid SSM/attention models as practical
    successors to pure SSMs.

## Recall, Copying, And Failure-Mode Papers

### Repeat After Me: Transformers are Better than State Space Models at Copying

- Authors: Samy Jelassi, David Brandfonbrener, Sham M. Kakade, Eran Malach.
- Link: https://arxiv.org/abs/2402.01032
- HF paper page: https://huggingface.co/papers/2402.01032
- Why it matters here:
  - Directly supports the hypothesis that fixed-state SSMs can struggle on
    copying/retrieval-from-context tasks.
  - Matches our observed Mamba weaknesses on source preservation, placeholder
    recovery, POS/NER sequence shape, and clean generation-loss gaps.
- Thesis use:
  - Literature support for why the Mamba failures are plausible architecture
    behavior, not just accidental bad prompting.

### Mimetic Initialization Helps State Space Models Learn to Recall

- Authors: Asher Trockman, Hrayr Harutyunyan, J. Zico Kolter, Sanjiv Kumar,
  Srinadh Bhojanapalli.
- Link: https://arxiv.org/abs/2410.11135
- Why it matters here:
  - Shows some Mamba recall/copying weakness may be partly trainability or
    initialization, not only hard capacity limits.
  - Suggests a possible future pure-Mamba intervention, but it would require a
    fresh initialization/pretraining implementation rather than post-hoc LoRA
    rescue of the current base.
- Thesis use:
  - Important caveat: we can defend stopping the current rescue path while
    acknowledging a research-backed future pure-Mamba rescue direction.

### Mimetic Initialization of Self-Attention Layers

- Authors: Asher Trockman, J. Zico Kolter.
- Link: https://arxiv.org/abs/2305.09828
- Why it matters here:
  - Earlier mimetic-initialization paper; useful background for the SSM mimetic
    initialization idea.
- Thesis use:
  - Optional background if discussing initialization as a general trainability
    lever.

### Is Mamba Capable of In-Context Learning?

- Link: https://arxiv.org/abs/2402.03170
- Why it matters here:
  - Nuances the failure story: Mamba can show ICL-like behavior in some
    settings, so the claim should not be "Mamba cannot do ICL/copying at all."
- Thesis use:
  - Helps make the discussion balanced: our result is about this model, data,
    budget, and downstream setup, not a universal impossibility proof.

## Implementation And Configuration References

### state-spaces/mamba repository

- Link: https://github.com/state-spaces/mamba
- Why it matters here:
  - Official implementation and model list.
  - Documents public 130M-ish model shapes and Mamba-2 defaults.
  - Notes that SSMs can be sensitive to precision/recurrent dynamics and
    initialization details.
- Thesis use:
  - Implementation-reference citation for public Mamba configs and caveats.

### Hugging Face Mamba-2 docs and public config

- Docs: https://huggingface.co/docs/transformers/en/model_doc/mamba2
- Public 130M config: https://huggingface.co/state-spaces/mamba2-130m/blob/main/config.json
- Why it matters here:
  - Used when comparing our original pure-Mamba 125M shape against a more
    public Mamba-2-like shape.
- Thesis use:
  - Methods appendix / reproducibility note, not necessarily a main paper
    citation.

## How To Cite In The Thesis Argument

- Pure-Mamba baseline is motivated by Gu and Dao's Mamba work and Dao and Gu's
  Mamba-2/SSD work.
- The negative pure-Mamba rescue story is supported by our experiments plus
  literature on copying/recall limits and Waleffe et al.'s controlled findings.
- The hybrid plan is not an ad-hoc retreat to Transformers: Waleffe et al. give
  a directly relevant 130M ablation pointing to about `8%` attention, which
  matches the current `2/24 = 8.3%` hybrid config.
- Mimetic initialization should be cited as a future-work caveat: it gives a
  plausible pure-Mamba rescue direction, but not one that is already available
  as a cheap repair to the trained SALLM Mamba base.
