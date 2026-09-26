# General b6 same-trial offline continuation

Prospective on 6 September, before any b6 continuation or new validation.
Authority is the 4 September General resumption and current continuation
monitor: finish required b5/b6 after preservation and offline checks.

B6 original job 1277424 ended TIMEOUT/0:0 after 24:00:17. The only retained
checkpoint directory is checkpoint-10912; all five state hashes exactly
match those recorded on 31 August. No final adapter, resume manifest or owned
b6 job exists. B5 continuation 1312084 is running and is left unchanged.

The entire original b6 seed-42 root was archived without changing it:
`/scratch/lmbanr001/masters/sallm/recovery_archives/general-stage-b/b6-seed42-pre-recovery.tar`,
123,013,120 bytes, mode 0444, SHA-256
`14ab572d07184607ec3d4281a30424c241019259e747e30fa96fbb52351b38d5`.
Original execution manifest and hpo_trial hashes are
`947255bb85a3826d9b60c2efaa62977dd83bb1d1b9500ca7ad400400779a7d3d`
and `13e19b76f9616d6fea618b27ec06969ae7a3b1850f127dc224ed2764af1e9f5c`.

Checkpoint-10912 bindings, in adapter/optimizer/scheduler/RNG/trainer order:

- `0cb7799764dcf92c10af4023c4dbb99312aae39d0ef8cb65580509bbe51f9697`
- `1fd8255bc8d0035a8b787164f8f21474a9c1a83fb19f149fbd76e6f2a8f7303e`
- `dbdb451011aacdbeedc3b19ff1bddd1d5dc1d10fb96ec8b65ec0be5b0331227c`
- `3180a03567b7f40c2f694e184313fd4ca7212fda830074f2e9bc3350b97907e5`
- `7424d09dfa63eaa2d8d9dd2bbd87bda84e0f68f66d4b36308820e2cb472a86bf`

Reuse, without editing, the proven offline source and train/validation cache
at `general-b5-offline-20260906-v2`. The only source change relative to the
695-file original corrected-resume snapshot is the existing cache-only
AfriHG loader. The b5 amendment documents exact ordered AfriHG record equality
with the old CSV-processing function. This b6 wrapper also requires every
raw train/validation count and ordered digest to match the frozen successful
preflight 1312049. No metric is computed for this transport check.

Source deployment manifest SHA-256:
`236a47dc6993851d299780161b2b517e55d5af61582ffc4b1c9143229c5692d1`.
Data checker SHA-256:
`fc8cf5b267d1c15237118cc25220feb06306808dcb8f73d737e4d7f4e948ed41`.
Reference preflight log SHA-256:
`956d1ac2ca86d927e1b48992758b48ab19f8361fd4867668919b2ea30b11f0d0`.
The new candidate-specific wrapper SHA-256 is
`c91418b7d4b17fda29dc451e5c790ab26ce021c600efc829f9f7a0e2d7b3d47c`.
Deploy only to the new isolated bundle
`/home/lmbanr001/masters/sallm_snapshots/general-b6-offline-20260906-c91418b7`.

Require an A100-80GB no-launch preflight to complete 0:0, then fresh
absent-output/no-duplicate and owned-cap checks before a single continuation.
Preserve the original runtime, model, b6 recipe, seed 42, exact checkpoint,
13,640-step schedule, validation evaluator and retention rules. Keep
resume-specific provenance and verify direct advancement to step 10913.
Use nlpgroup80/a100/nlpgroup80, gpu:ampere80:1, eight CPUs, 24 hours and
canonical chdir. No retry, from-scratch restart, b3 replacement, held-out
evaluation, recipe change, or incomplete cross-candidate ranking is authorized
by this amendment. No completed held-out outcome informed this continuation.
