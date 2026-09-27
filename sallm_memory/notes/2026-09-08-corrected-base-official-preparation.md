# Corrected Base official preparation — 8 September 2026

The 30 August amendment remains authoritative: rerun 14 logical Base lanes,
represented by 16 lm-eval task packs, and exclude the unaffected T2X and
AfriHG generation lanes. Every Base pack is bound to raw prompting with
`apply_chat_template=false`, the exact frozen pure-GDN base checkpoint,
`peft_adapter=null`, and `merge_lora=false`.

Preparation job `1323700` failed `1:0` before loading any data or model because
the offline Hugging Face cache path was omitted. Its v1 snapshot, empty
protocol directory, log, and absent result root are preserved. This is a
metric-free environment failure, not a held-out attempt.

The minimal v2 correction supplies the already-verified offline cache path and
uses a fresh snapshot and protocol. CPU job `1323715` completed `0:0` and
verified all 16 packs, 187 tasks, and 110,251 prompt rows with
`model_loaded=false` and `metrics_computed=false`. It generated 16 isolated
configs; static readback confirms raw prompting, no adapter, no generation
tasks, and the untouched correction-specific result root. Preflight SHA-256 is
`1e0d91f1d9f26af4a1757bf2dbefef6cb7e0f540e7dda8c6badb2f5c7c2fe11c`.

V2 snapshot:
`/home/lmbanr001/masters/sallm_snapshots/pure-gdn-base-corrected-20260908-v2`.
Protocol:
`/scratch/lmbanr001/masters/sallm/manifests/base_corrected_20260908_v2`.
Fresh exact-checkpoint A100-80GB canary `1323738` is pending on association
capacity. It must pass save/reload equality, finite BF16 forward/backward, and
deterministic generation without task data. CPU seal `1323744` is already
submitted with `afterok:1323738`; it will resolve all configs, bind source,
runtime, checkpoint and cache hashes, and write `READY` only on success.

No Base official payload, prediction, metric, result, or Sheet value has been
opened or changed. After the canary and seal pass, submit the three frozen
48-hour A100-80GB groups once, keeping General plus Base within four owned GPU
jobs. Preserve every first valid lane as terminal.
