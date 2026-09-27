# General b5 continuation launched

Job 1312084 was submitted once on 6 September after the user instructed launch.
It is running on srvrocgpu011, nlpgroup80/a100/nlpgroup80, one A100-80GB,
eight CPUs and 24 hours. The root, checkpoint and recipe remain the same trial.

The prospective amendment is
`2026-09-06-general-b5-offline-continuation-amendment.md`, SHA-256
`4a435cce1cb717612750a59b98e75aae9a4165f5e3cf6cd8e6e50c9ee515c1f9`.
No-launch preflight 1312049 completed 0:0. Failed check-only job 1312019 is
preserved; it did not train or write a continuation manifest.

The launch reran the preservation/source/runtime/data checks successfully.
All six launch-time family JSON records exactly matched the frozen preflight
log: all twelve ordered train/validation digests and all counts matched.
This comparison was done explicitly because the checker logs the five
non-AfriHG families' digests rather than asserting historical expected hashes.
AfriHG cache files and ordered records are directly checked by the script.

Resume execution manifest SHA-256:
`a790b10cb2602c41477deff6dfb92f0ee8d3fba50808dc322b10cd0cfa2f9e18`.
Log: `/scratch/lmbanr001/masters/sallm/logs/jobs/gdn-general-b5-offline-1312084.out`.
Training arguments show checkpoint-10912 continuation and push_to_hub=False.
Startup processed all 43,637 training and 22,167 validation rows. At 16:35 UTC,
the trainer advanced directly to step 10913/13640 after checkpoint loading,
proving continuation rather than a step-zero restart. Job remains RUNNING.

Independent GPT-5.6 Sol review found no substantial remaining issue after
launch-time digest equality was verified. No held-out scores informed the
repair; no test evaluation, Sheet changes or other-user job changes occurred.
Home quota is 90.1%, scratch 51.3%. General Stage-B remains 5/8 verified.
