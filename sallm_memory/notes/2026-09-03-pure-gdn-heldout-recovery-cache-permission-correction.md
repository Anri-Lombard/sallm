# Pure-GDN held-out recovery cache permission correction - 2026-09-03

This amendment is frozen before any replacement cache job or recovery model
inference. It supplements, and does not replace, the recovery preregistration
`2026-09-03-pure-gdn-missing-heldout-recovery-preregistration.md`.

CPU cache job 1291322 failed after eight seconds while copying the already
sealed News cache. `shutil.copytree` preserved the source cache's read-only
directory modes, so the builder could not create `hf/modules`. It produced no
model predictions, no evaluation metric, and did not load a new held-out
dataset.

The prospective correction changes only the cache materializer: immediately
after copying the sealed News cache, copied directories are made owner-writable
so the remaining frozen dataset and metric assets can be added. The completed
recovery cache will still be hashed and sealed before any GPU evaluation. No
task, row, adapter, prompt, decoding setting, metric definition, ordering rule,
or result-handling rule changes. Job 1291322 and its partial cache are preserved
as failed provenance; the replacement uses a new isolated cache root and is
submitted once.
