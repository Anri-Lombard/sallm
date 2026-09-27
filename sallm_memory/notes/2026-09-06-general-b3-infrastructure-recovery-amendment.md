# General b3 disclosed infrastructure recovery

Prospective on 6 September, before replacement submission or new validation.
Authority: the user-authorized General resumption recorded on 4 September
and the current continuation monitor's explicit b3 recovery boundary.

## Protocol exception and preservation

Original b3 job 1276434 timed out. Its continuation 1279472 failed 1:0
after 2:07 on an SSL request for AfriHG zul/validation.csv. It accessed
model/data payload and is terminal under the 30 August rule. Preserve it as
that failure: this is an explicitly disclosed narrow exception for one
infrastructure-corrected continuation, not a reclassification as pre-payload
and not an unrestricted retry allowance. The 4 September resumption permits
this exception only after state/data/runtime binding and offline proof.
Record this deviation in General provenance and paper methods.

The failed log contains no Fine-tuning start marker. More importantly, all
five checkpoint-10912 hashes still exactly match the 30 August registration.
No new checkpoint or final adapter exists. Both original and failed recovery
manifests/trial files remain intact. No score motivated this exception.

Read-only original archive SHA-256:
`a51c35ead0e3e794be9209728c12ed806e7471a18ad7d0334aeb5c5b88f3d918`.
The complete current root, including failed resume provenance and both
1279472 logs, is separately archived at
`/scratch/lmbanr001/masters/sallm/recovery_archives/general-stage-b/b3-seed42-post-1279472-infrastructure.tar`,
245,944,320 bytes, mode 0444, SHA-256
`29f8d65332f61867ccf07e00cea94570ad7d34ce46e9cba1d6e8b57e4186beb4`.
Original timeout and both failed-job logs are preserved outside this archive.
Failed batch/main log hashes are respectively
`85027043697c95729fa4c01c5cd8275b2e8aff05793f02b7dbd479842a8579bd`
and `16b4987c1770831208f1724c1f1ef455570f2b087ab1336a863e11f783ba272b`.

Original manifest/trial SHA-256:
`1344f1c7d2ea38b1f90ffc0986e49e8296eaa44f8f3668b6029fea1ea4255bdf`,
`22edeb51a958778a12c8bedcedfd686337f4e1a8b718b148ea7358d10c1958c7`.
Failed resume manifest/trial SHA-256:
`30bfda0598bd462208411ba332ed0e2c83ea1065ab3e107c3f30d4b4098c028c`,
`dbe23c957d2c0c7f5955c80bcc3589b086a70bd76577c88f884eec2c46e23bf7`.

Checkpoint-10912 adapter/optimizer/scheduler/RNG/trainer hashes:

- `e591c605336ea0fbc431199b59344e768b3242a56ad0f5e5ebe8bd3a201adfb5`
- `98c23a9b4f6debdfe49bc5fdd64ce0388d209354f470ce2957828fca1149edd3`
- `2244542dfc73c540f56c4397f999c55d9557aadb704e404ac6d3861dcd996027`
- `47a82e7665a2fa0070b6758de26b3cb3469af4e8573bfae25bf22006f339fc26`
- `968e989e31e8f9248957eb90b7adaff293aba12d3c3cfea06b19f50c66d11031`

## Narrow correction and launch gate

Reuse the unchanged `general-b5-offline-20260906-v2` source snapshot, whose
only scientific-source difference from the original 695-file snapshot is
the existing cache-only AfriHG loader. Its CSV processing produces exactly
the same ordered train/validation records, proved against the old loader's
CSV-processing function. No model, recipe, seed, data content, evaluator,
checkpoint, retention rule or training schedule changes.

The deployment manifest hash is
`236a47dc6993851d299780161b2b517e55d5af61582ffc4b1c9143229c5692d1`;
checker hash `fc8cf5b267d1c15237118cc25220feb06306808dcb8f73d737e4d7f4e948ed41`.
The wrapper requires all twelve ordered train/validation digests and counts
to match frozen offline preflight 1312049 log hash
`956d1ac2ca86d927e1b48992758b48ab19f8361fd4867668919b2ea30b11f0d0`.
It verifies exact original runtime, source inventories, both archives,
original/failed manifests, and all checkpoint hashes. It permits only the
known failed resume 1279472; any other resume or final adapter blocks launch.

Candidate-specific wrapper SHA-256:
`d2359186a9ead265940efa63fc7da10484ad60f6d6578403ee93c7e94da3eddd`.
Immutable bundle:
`/home/lmbanr001/masters/sallm_snapshots/general-b3-offline-20260906-d2359186`.
Require no-launch A100-80GB preflight 0:0 and fresh owned-cap/no-duplicate
checks before one replacement. Use nlpgroup80/a100/nlpgroup80,
gpu:ampere80:1, eight CPUs, 24 hours, canonical chdir. The unchanged launcher
writes new job-specific resume manifests, preserving failed provenance.
Verify direct checkpoint-10912 to step-10913 advancement. Any further failure
is preserved; this amendment authorizes no additional replacement.

No held-out output, cross-candidate comparison or interim score influenced
the exception, checkpoint or correction. Rank only after every required
seed-42 candidate is terminal-verified. Existing Mono/Multi results remain
separate and unchanged.
