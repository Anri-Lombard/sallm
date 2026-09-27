# Pure-GDN POS source-availability correction preregistration — 2026-08-14

Frozen before any corrected POS run or metric.

## Trigger and scope

POS a0 job `1231553` failed `1:0` after 53 seconds, after immutable-source
verification and canonical-model load but before dataset load, training, or
metric computation. The shared MasakhaPOS loader requested mutable GitHub
`main` raw URLs. Direct checks reproduced intermittent `503` responses; the
loader then tried nonexistent validation aliases and surfaced the final `404`,
masking the transient first error. No held-out split or metric informed this
correction.

This is an infrastructure/source-identity correction only. It does not change
the model, training data content, validation examples, HPO candidates,
candidate order, hyperparameters, prompts, scoring rule, checkpoint rule,
seed, early-stopping rule, or family ranking rule.

## Frozen correction

1. Pin MasakhaPOS to upstream Git commit
   `376f4161f0425584d4bd7664122b56fa026926d3`, whose commit date is
   2025-06-25 and therefore predates this evaluation program.
2. Fetch its file content through GitHub's commit-addressed Contents API with
   raw-content media type, avoiding the failing raw-content CDN route.
3. Retry only transport errors and HTTP `408/429/500/502/503/504`, at most
   three attempts with standard-library backoff. Do not retry or reinterpret
   semantic `404` errors. The same transport helper applies to the existing
   direct InjongoIntent source path without changing its source identity.
4. Require exact prospective MasakhaPOS coverage before deployment:
   Tsn/Xho/Zul train rows `754/752/753`, validation rows `150/150/150`, train
   token counts `22093/13197/12419`, and validation token counts
   `4112/2367/2383`.

Canonical upstream train/dev SHA-256 values are:

- Tsn: `ed9bdd732717d13c9664e22ff388ac147c94a1cefbe89dbda8b97384b84b6785` /
  `e5c3bcee987291c9a795df11092d47bd73fb1ad1d51e5783e2ff4e67321e5b70`.
- Xho: `d552fd8648cc59c9c9e5be02a8c74db4b83806daf2d365b88463bacaee5b04cc` /
  `e7f0c01aafc4af4429c3f96c4b0d5e3196719b83a1504e9f230d50093d62e090`.
- Zul: `c485dfc97250b2fd9330a9734f0aa572d40a8ef14408a9450d03e46892479dff` /
  `7c4ab43bf978338365a5f25e4bb82645377bb30cd0dbc70750500c918e136551`.

## Prospective gates and rerun rule

- Local focused and full tests, Ruff, shell syntax where applicable, and the
  exact count/token audit above must pass.
- Deploy only by copying the previously frozen immutable HPO snapshot and
  replacing the corrected loader, then create and verify a complete new
  deployment manifest and make the snapshot read-only.
- Keep a1 job `1232086` held until the new immutable deployment is verified.
- Rerun failed a0 once from the new snapshot under its otherwise unchanged
  frozen Stage-A protocol. Because `1231553` produced no metric, this is not a
  performance-selected retry. Release/resubmit a1 only after a0 passes its
  source/count/startup gates.
- Preserve `1231553`, all logs, and the old snapshot. Any further source
  failure is provenance and must not trigger dataset, recipe, prompt, or
  hyperparameter selection.
- Held-out access remains forbidden. Sheet E/F/G remain blank; no winner is
  frozen and no Hugging Face publication is authorized by this correction.

Prospective implementation hashes before deployment:

- `src/main/sallm/data/loaders/huggingface.py`:
  `d9501087564982e5e4b07c4b3aada876d5625fa24560024a96002a4ee9ee4006`.
- `tests/data/loaders/test_huggingface.py`:
  `f3b5679a1e38b0a6d8c0f685936a9921ddc27588924f7e0f434a09a7e0fe12c0`.

Local validation passed: `117 passed`, focused Ruff clean, exact prospective
row/token coverage matched, and `git diff --check` clean.
