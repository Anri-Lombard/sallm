# Uniform SALLM HPO literature audit — 2026-08-11

This is an explanatory literature audit, not an amendment to the frozen
pure-GDN protocol and not evidence from held-out adapter tests.

## Theoretical interpretation

The implemented procedure treats each model/family fine-tuning run as an
expensive, noisy black-box validation objective. Stage A supplies three fixed
learning-rate controls. Stage B maps a scrambled four-dimensional Sobol point
set into log learning rate, categorical LoRA rank, dropout, and warmup. Stage C
estimates seed variation for selected candidates. The held-out test remains
outside this optimization loop. Equal candidate and confirmation budgets are
intended to make architecture comparisons auditable.

## Directly supported

- Bergstra and Bengio (2012) show random search is more efficient than grids
  when only some hyperparameters materially matter. This supports sparse
  space-filling search rather than a Cartesian grid.
- Sobol (1967) and Owen (1997) establish low-discrepancy and scrambled-net
  properties. These support reproducible, more even coverage of a bounded
  space, but their guarantees concern numerical integration, not discovery of
  a black-box optimizer.
- Dodge et al. (2020), Mosbach et al. (2020), and Reimers and Gurevych (2017)
  document large seed variation and instability in NLP fine-tuning. This
  supports multi-seed confirmation and reporting distributions rather than a
  single best run.
- Cawley and Talbot (2010) explain model-selection overfitting and selection
  bias. This strongly supports validation-only HPO and a test set isolated
  until the recipe is frozen.
- Lucic et al. (2018), Dodge et al. (2019), and Bouthillier et al. (2021)
  support controlled tuning budgets, disclosure of search effort, and
  accounting for hyperparameter/seed variance in comparative claims.
- Hu et al. (2021) support low-rank adaptation as an efficient adaptation
  mechanism. They do not establish the present rank set or alpha rule.

## Competing or absent support

- Snoek et al. (2012) support Bayesian optimization for expensive black-box
  objectives. Hyperband (Li et al., 2017), BOHB (Falkner et al., 2018), and
  ASHA (Li et al., 2020) support adaptive/multi-fidelity allocation. Therefore
  the literature does not establish fixed Sobol as the most compute-efficient
  optimizer. Avoiding pruning is a conservative project-specific decision
  based on observed late/crossing NER validation curves.
- Sobol theory does not prove eight points are enough, that the first eight
  find the optimum, or that Sobol beats Bayesian/TPE/BOHB HPO. The exact
  `3 + 8` budget is a resource/preregistration choice.
- LR bounds `[2e-5, 2e-4]`, ranks `{8,16,32}`, `alpha = 2r`, dropout/warmup
  bounds, and seeds `13/42/87` are reasonable heuristics, not universal values
  derived from cited theory.
- Three seeds provide a descriptive mean and sample SD, not a well-powered
  significance test. Selecting candidates first at seed 42 also leaves
  winner's-curse risk. A fixed Stage-A control versus best enhanced candidate
  on matched seeds would evaluate HPO utility more directly.
- Fair-comparison research supports equal tuning opportunity and transparent
  budgets, not necessarily identical numeric spaces. xLSTM, Mamba, LLaMA, and
  GDN may require architecture-appropriate LoRA ranks/targets. Any differing
  space must be frozen prospectively and its effective search volume reported.
- Retrospective HPO of models whose test results were already observed cannot
  become fully prospective. It can be a uniform post-hoc reproduction, but
  architecture claims must disclose that distinction and prevent known test
  results from influencing the frozen protocol.

## Primary sources

- Bergstra and Bengio, *Random Search for Hyper-Parameter Optimization*, JMLR
  2012: https://jmlr.org/papers/v13/bergstra12a.html
- Sobol, *On the distribution of points in a cube and the approximate
  evaluation of integrals*, 1967: https://doi.org/10.1016/0041-5553(67)90144-9
- Owen, *Monte Carlo Variance of Scrambled Net Quadrature*, 1997:
  https://doi.org/10.1137/S0036142994277468
- Snoek et al., *Practical Bayesian Optimization of Machine Learning
  Algorithms*, 2012: https://arxiv.org/abs/1206.2944
- Li et al., *Hyperband*, 2017: https://arxiv.org/abs/1603.06560
- Falkner et al., *BOHB*, 2018: https://arxiv.org/abs/1807.01774
- Li et al., *A System for Massively Parallel Hyperparameter Tuning*, 2020:
  https://arxiv.org/abs/1810.05934
- Dodge et al., *Fine-Tuning Pretrained Language Models*, 2020:
  https://arxiv.org/abs/2002.06305
- Mosbach et al., *On the Stability of Fine-tuning BERT*, 2020:
  https://arxiv.org/abs/2006.04884
- Reimers and Gurevych, *Reporting Score Distributions Makes a Difference*,
  2017: https://arxiv.org/abs/1707.09861
- Cawley and Talbot, *On Over-fitting in Model Selection and Subsequent
  Selection Bias*, JMLR 2010: https://jmlr.org/papers/v11/cawley10a.html
- Lucic et al., *Are GANs Created Equal?*, 2018:
  https://arxiv.org/abs/1711.10337
- Dodge et al., *Show Your Work*, 2019:
  https://arxiv.org/abs/1909.03004
- Bouthillier et al., *Accounting for Variance in Machine Learning
  Benchmarks*, 2021: https://arxiv.org/abs/2103.03098
- Hu et al., *LoRA*, 2021: https://arxiv.org/abs/2106.09685
