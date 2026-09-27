# Pure-GDN HPO acceleration literature review — 2026-08-16

## Frozen questions

1. Which established HPO methods reduce GPU cost without using held-out test
   data?
2. How do they handle late learning-curve rank reversals?
3. Which prospective change fits the untouched AfriHG and General grids while
   preserving the already-running POS protocol?

## Evidence

- Random or low-discrepancy sampling is a sound search-space strategy. Bergstra
  and Bengio (JMLR 2012) found random search more efficient than grids when
  only a subset of hyperparameters materially affects performance. The frozen
  SALLM Sobol design is therefore not the principal inefficiency.
- Uniformly training every sampled configuration to completion is the costly
  baseline. Hyperband (Li et al., JMLR 2018) allocates iterations, data, or
  other resources adaptively and reported more than an order-of-magnitude
  speedup on some studied problems. ASHA (Li et al., MLSys 2020) makes
  successive halving asynchronous and scalable across workers. BOHB (Falkner
  et al., ICML 2018) adds model-based proposal selection to Hyperband.
- Early pruning is not free evidence. Hyperband notes that low-fidelity
  performance can be uninformative when convergence is slow or
  resource-dependent. PASHA (Bohdal et al., ICLR 2023) increases the maximum
  resource only while rankings remain unstable and reports roughly 2--5x
  speedups in several experiments, but explicitly retains the same late-winner
  pruning risk as ASHA-style methods.
- Freeze-thaw methods allocate more budget to promising existing checkpoints
  rather than restarting them. This matches the exact checkpoint-resume
  support already verified for SALLM, but sophisticated surrogate-based
  freeze-thaw BO would add implementation and validation risk before the
  deadline.
- Current PyTorch guidance uses Ray Tune ASHA to stop weak trials and preserve
  checkpoints; Hugging Face Trainer supports Optuna and Ray backends. Bringing
  Ray into the current Slurm/provenance stack is unnecessary for a fixed
  11-candidate search: the same resource schedule can be implemented with the
  existing immutable launcher and exact checkpoint resume.

Primary sources:

- Bergstra and Bengio, Random Search for Hyper-Parameter Optimization:
  https://www.jmlr.org/papers/v13/bergstra12a.html
- Li et al., Hyperband:
  https://www.jmlr.org/papers/v18/16-558.html
- Li et al., A System for Massively Parallel Hyperparameter Tuning:
  https://proceedings.mlsys.org/paper_files/paper/2020/hash/a06f20b349c6cf09a6b171c71b88bbfc-Abstract.html
- Falkner et al., BOHB:
  https://proceedings.mlr.press/v80/falkner18a.html
- Bohdal et al., PASHA:
  https://arxiv.org/abs/2207.06940
- Rakotoarison et al., In-Context Freeze-Thaw Bayesian Optimization:
  https://proceedings.mlr.press/v235/rakotoarison24a.html
- Hugging Face hyperparameter search:
  https://huggingface.co/docs/transformers/hpo_train
- PyTorch Ray Tune ASHA tutorial:
  https://docs.pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html

## Synthesis and recommendation

The SALLM design is scientifically conservative rather than conceptually
wrong. Its Sobol search, full validation objective, top-two confirmation, and
held-out firewall are defensible. The expensive choice is uniform full-budget
allocation plus an exact task metric every epoch. POS measurements make the
problem extreme: exact validation dominates training time.

Do not change POS after observing partial family metrics. Finish its frozen
protocol and preserve the three current job records. For untouched AfriHG and
General, a prospective conservative successive-halving amendment is the
smallest defensible acceleration:

- keep the same 11 candidates, seed 42, full validation sets, exact primary
  metrics, tie breaks, and Stage-C top-two seeds 13/87;
- train all 11 against the original five-epoch scheduler but pause after epoch
  2; promote the top 6;
- resume those 6 from exact checkpoints and pause after epoch 4; promote the
  top 3;
- resume those 3 through epoch 5 and rank them using their best exact rung
  metric; then perform the unchanged four Stage-C confirmation runs;
- use no held-out data, task proxy, truncated validation set, cross-family
  winner, or adaptive change after seeing AfriHG/General metrics.

This schedule is `7.4` full-training equivalents rather than `11` per family,
about a 33% seed-42 training reduction. It requires `20` full exact callbacks
per family instead of at least `33` under patience-2 epoch-wise evaluation, a
minimum 39% callback reduction and potentially more when old trials ran four
or five epochs. With only three GPUs and task-specific callback costs, no
wall-time speedup should be claimed before a validation-only runtime gate.

No experiment, job, checkpoint, Sheet value, or held-out state was changed by
this review.
