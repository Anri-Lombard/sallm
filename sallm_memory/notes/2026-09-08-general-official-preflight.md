# General official evaluation preparation

The 7 August preregistration requires the full 16-lane matrix: pooled News,
NER, POS, SIB and Intent; SA-general (AfriMGSM, AfriMMLU, AfriXNLI); eight
Belebele languages (afr, eng, sot, ssw, tsn, tso, xho, zul); T2X Xhosa; and
AfriHG Xhosa/Zulu. AfriHG English is inapplicable. This scope comes from the
protocol, not workbook scores. The fixed model is the pure-GDN final_model
and General B7 seed42 checkpoint5456 frozen in the separate ranking note.

Configuration-only CPU preflight will resolve the existing full-matrix config
under immutable corrected snapshot
pure-gdn-heldout-recovery-20260903-v2-46d8cf11. Config SHA-256:
9067f82c66111a7055fae5f575dceb7f0915adcf6cf6fcb32b4b48859367eae5.
Override only the frozen pure-GDN base and adapter, disable adapter merging,
and use a new General result root. Hydra --cfg job --resolve performs no
model inference, dataset loading or metric calculation. The resolved config
is inspection evidence, not permission to run the combined matrix or proof
that all data/runtime/coverage gates pass. Preserve any failed preflight.

Existing family wrappers bind specific Mono/Multi paths and cannot simply be
pointed at General. Reuse their evaluator and structural verifiers through a
General-specific frozen execution binding. Verify exact prompt/decoding,
cache and row coverage before submitting any official arm.

No owned GPU jobs exist. A10080 node011 is idle; both40GB nodes are mixed.
No demonstrated queue benefit justifies a hardware transition for later GPU
evaluation. Current work is CPU-only. The Base 14-lane correction amendment
remains binding; the older16/16 shorthand does not supersede it.

CPU job1318551 completed0:0. Resolved stdout artifact:
/scratch/lmbanr001/masters/sallm/manifests/general-official-config-20260908.yaml
SHA-256: 40ed7aab2adda0ad46f93d979dde08d936d2cfbc7c8e8a53a134c1dbdf936c73.
It includes a WANDB-offline banner before YAML, so treat it as a transcript,
not a directly loadable configuration file. Inspection confirmed exact frozen
base/General final adapter, BF16, merge_lora=false, all16 task packs and the
native generation section. The sixteen logical lanes include three task packs
inside SA-general and two generation tasks inside AfriHG. The prospective
official result root remains absent. No test dataset or model was loaded.

Remaining implementation: bind each official lane's exact tasks, row counts,
prompt/scoring/decoding contract and runtime/cache hashes. Existing
verify_official_task_pack_eval.py only accepts a one-pack summary containing
one language's numbered prompts; it cannot validate pooled/multi-pack outputs
unchanged. Preserve its safety checks when adapting the General verification.
Task construction is in src/main/sallm/evaluation/lm_eval_runner.py using
TaskManager and _prepare_include_paths; inspect that path before a data-only
cache preflight. Source cache_pure_gdn_official_recovery_assets.py covers only
News/NER/POS and does not establish SIB/Intent/Belebele/SA/T2X/AfriHG coverage.
No generic bulk launcher is eligible as-is. No automatic resubmission of1318551.

Task-definition audit1318956 was submitted once on CPU (maths/ada/normal,
4CPUs,8GB,10minutes) using .audit/general_task_contract.py. Sealed remote
script: /home/lmbanr001/masters/sallm_snapshots/general-confirm-20260907/general_task_contract.py
SHA-256: 3a2d54e5a0b57cb201cdf1c0d69158d25c538a76374cb11e1effb460c79ec8ed.
It uses the installed TaskManager and simple YAML resolution (function
references remain strings), checks exact task identity/test splits,187 prompt
tasks across16packs and3 native-generation tasks, and records the task YAML
paths/hashes plus inherited dataset/prompt/metric definitions. It loads no
dataset or model and cannot evaluate a metric. JSON creation is exclusive;
preserve any partial artifact on failure.

Log: /scratch/lmbanr001/masters/sallm/logs/jobs/gdn-general-task-contract-20260908.out
Artifact: /scratch/lmbanr001/masters/sallm/manifests/general-task-contract-20260908.json
Last seen RUNNING. Await terminal status before reading or accepting the JSON;
do not duplicate this job. Use the completed contract to bind data-only cache
coverage and exact per-pack structural verification next. General's original
task-pack settings include raw SIB and SA-general, chat News/NER/POS/Intent/
Belebele. Preserve those settings instead of forcing every pack to chat.

At02:57UTC,1318956 is FAILED1:0. All collect() assertions returned before
json.dump failed: lm-eval mode=simple preserves !function as yaml.ScalarNode,
not a string. Serialization stops at AfriMMLU doc_to_choice. This is an audit
export defect, not model/data/scoring failure. No dataset/model/metric was
loaded; the official result root remains absent. The partial JSON is invalid
and must not be used as the task contract.
Failed log SHA256: 8ff09f17d57d615b84663f9907da5ce74595e67f8972868e0b50ba24526ac8fc.
Partial JSON SHA256: d4b87600ebc4e4b24dce1e60214275888267785bacb7155d3a7e942354e3627f.
Preserve the script, job, log and partial JSON. No repair or rerun was made.
The active no-failed-gate-rerun instruction requires user approval before a
new execution-only audit exports ScalarNode tag/value explicitly. Proposed
scope preserves all task definitions and scientific checks, uses a new output,
and adds a serialization regression check. General official tests stay blocked.
