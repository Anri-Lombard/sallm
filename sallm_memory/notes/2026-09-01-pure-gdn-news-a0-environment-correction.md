# Pure-GDN News a0 environment correction

Frozen before the replacement job or any News model/data/validation payload.

News a0 seed-42 job `1280436` verified its immutable 715-file manifest and GDN
runtime, then failed during Hydra interpolation because
`PURE_GDN_MODEL` was not exported. It did not load a model or dataset and
produced no checkpoint, prediction, validation artifact, or metric. Preserve
the job, log, and original manifest-only output root; never reuse or overwrite
that root.

One prospective pre-payload implementation correction is authorized. It adds
only `PURE_GDN_MODEL` set to the same already frozen local base path passed by
the command-line override and isolates output/logging under
`seed_42_env_correction_20260901`. Candidate a0, seed 42, registry, learning
rate, LoRA recipe, target modules, data, prompt, validation-only macro F1,
epochs, checkpoint rule, immutable source, A100-80GB envelope, and candidate
order remain unchanged. The failure and replacement cannot alter later HPO
from any metric because no metric exists.
