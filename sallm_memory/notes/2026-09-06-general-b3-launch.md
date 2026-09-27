# General b3 infrastructure recovery launched

Job 1312957 was submitted once after no-launch preflight 1312488 completed
0:0 in 24 seconds. Its full output verified original runtime, both 695-file
source inventories, exact AfriHG CSV-processing equality and all twelve
ordered raw train/validation digests. Preflight log SHA-256:
`35f5846841bd1a2f4c726171e83529248de38406c355b5d6878662854adb4e84`.

The sealed amendment and wrapper still match hashes
`2f12c3cf67b37eb5d1bf8a87c0af4170ad696b33dcf3807da452c87b4c112c5b`
and `d2359186a9ead265940efa63fc7da10484ad60f6d6578403ee93c7e94da3eddd`.
See `2026-09-06-general-b3-infrastructure-recovery-amendment.md` for the
explicit post-payload exception, original/failed archives and all exact state
bindings. Preserve failed 1279472 as terminal; this is not a pre-payload retry.

Fresh pre-submission checks found no other b3 job, no final adapter and only
the preserved 1279472 resume manifest, with two owned active jobs. Submitted
the sealed wrapper for A100-80GB, one GPU, eight CPUs and 24 hours. B5/b6
were left running. Switching families now would lose unsaved training and
delay this ready job despite a free 80GB slot, so no switch was made.

No duplicate continuation, held-out evaluation or Sheet change was made.
Startup verified at 18:50 SAST: the trainer jumped directly from checkpoint
10912 to step 10913 and advanced through 10925. Resume execution manifest
SHA-256: `aa90cccc1f4811f15d7f5958f77d79de8a2a35aa5f4e39bf68698797b66330b7`.
All three remaining Stage-B jobs are running on A100-80GB. Monitor terminal
artifacts and adapter roundtrips before ranking. No additional replacement
is authorized. Quota: home 90.1%; scratch 51.5% (154/300GB).
