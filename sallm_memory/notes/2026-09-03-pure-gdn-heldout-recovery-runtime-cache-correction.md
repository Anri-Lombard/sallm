# Pure-GDN held-out recovery runtime-cache correction

Date: 2026-09-03

This amendment is prospective and precedes any replacement inference.

Recovery jobs 1291753 (T2X), 1291754 (NER), 1291755 (POS), and 1291756
(AfriHG) failed before writing predictions, metrics, or result artifacts. The
canonical recovery cache was correctly sealed read-only, but the evaluation
libraries also tried to use it for runtime locks, generated dataset caches,
and metric state. NER additionally needs offline aliases for the exact config
identifiers computed by the frozen lm-eval task calls. The failed jobs, logs,
protocol files, and empty result roots remain preserved and must not be
represented as successful evaluations.

One replacement recovery implementation, `pure-gdn-heldout-recovery-20260903-v2`,
is authorized by the user's instruction to fix the missing held-out results.
It makes only these execution corrections:

1. Materialize the three deterministic NER offline config aliases from the
   already row- and fingerprint-verified cached datasets.
2. Seal and hash a new canonical v2 cache without model inference or metrics.
3. Before each job, verify the canonical cache tree, copy it to a family- and
   job-specific runtime directory, verify the copy, and make only that private
   copy writable.
4. Point Hugging Face dataset, hub, module, and evaluate caches at the private
   runtime copy.

The frozen base model, adapters, task definitions, prompts, decoding, metric
definitions, row expectations, family order, and reporting rules are
unchanged. No held-out value may influence this correction or any later
selection. The v2 result and protocol roots are isolated from the original
attempt and v1 recovery. Once a v2 family writes predictions or metrics, that
family is final and cannot be retried or corrected from its held-out outcome.
