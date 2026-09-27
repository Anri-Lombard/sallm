# Pure-GDN held-out recovery POS cache count correction - 2026-09-03

This amendment is frozen before another cache job or any recovery model
inference. It supplements the recovery preregistration and the earlier cache
permission correction.

Corrected CPU cache job 1291324 loaded and validated all three NER test splits,
then failed while checking the first POS language. The Hugging Face `text`
builder exposes one row per raw CoNLL line, whereas the frozen MasakhaPOS task's
`process_pos_docs` function groups those lines into sentences. The cache builder
incorrectly compared raw line count 16,086 with the frozen sentence count 602.
No model was loaded, no prediction was generated, and no metric was computed.

The prospective correction counts sentences with the same blank-line,
`-DOCSTART-`, and minimum-two-fields rules as the already-frozen task loader.
It records both raw and sentence row counts. No dataset, task, adapter, prompt,
decoding setting, metric, candidate, or ordering rule changes. Job 1291324 and
its partial cache are preserved; the next cache materialization uses a fresh
canonical root after quarantining that partial tree.
