# Pure-GDN News preflight config correction

Frozen on 3 September 2026 after metric-free News preflight `1290439` failed
and before any current-adapter News held-out payload opened. The result root is
absent. Source, base, cache, and Multi-adapter manifests verified, then Hydra
rejected `eval.eval_model.merge_lora=false` because the existing News config
does not declare that optional key. No model inference, dataset loading,
prediction, summary, or metric occurred.

The prospective correction changes that one override to Hydra's add-or-override
form, `++eval.eval_model.merge_lora=false`. It changes no model, adapter, data,
row, prompt, metric, checkpoint, or arm. The corrected wrapper also binds News
to a fresh `news_v2` protocol root; the `news_v1` manifests and failed preflight
log are preserved.

Corrected wrapper SHA-256 is
`86b29aadf6e534623dce012084dd1cb86ed5109ba2440c77672a5cfa497a2ddd`.
Bash syntax and four focused checks pass. Create a fresh immutable News
snapshot, source/artifact manifests, and metric-free preflight before any GPU
submission. The sealed cache and all frozen scientific inputs remain
unchanged.
