# Pure-GDN AfriHG family-wise official-test freeze — 2026-09-02

This prospective freeze was written before launching or inspecting any new
AfriHG official held-out evaluation. It authorizes exactly four one-time
adapter arms because the accepted Base AfriHG artifact already exists:
Multi Xhosa, Multi Zulu, Mono Xhosa, and Mono Zulu. Held-out output may only
be reported; it must never change a recipe, checkpoint, retry, or schedule.

## Frozen artifacts

- base: `/scratch/lmbanr001/masters/sallm/checkpoints/sallm-pure-gdn-125m/a10080-3epoch-20260802/final_model`
- Multi: `/scratch/lmbanr001/masters/sallm/checkpoints/adapter_hpo_v3/pure_gdn/afrihg/stage_b/b7/seed_42/final_adapter`
- Mono Xhosa: `/scratch/lmbanr001/masters/sallm/checkpoints/pure_gdn_mono_familywise_v1/afrihg/xho/seed_42/final_adapter`
- Mono Zulu: `/scratch/lmbanr001/masters/sallm/checkpoints/pure_gdn_mono_familywise_v1/afrihg/zul/seed_42/final_adapter`
- result root: `/scratch/lmbanr001/masters/sallm/results/official_test/familywise_20260901/afrihg`
- protocol root: `/scratch/lmbanr001/masters/sallm/manifests/official_test/familywise_20260901/afrihg_cache_pinned_v1`

Both Mono adapters passed validation-only selection and exact
retained-to-final roundtrip verification before this freeze.

## Frozen source and data

Immutable snapshot:
`/home/lmbanr001/masters/sallm_snapshots/pure-gdn-afrihg-official-test-20260902-fb19cdc0`.
The AfriHG loader SHA-256 is
`12e0de1b6b4d851e9ee8426f153632388a0dad2c9073478ec2775bec523181c6` and
the official wrapper SHA-256 is
`fb19cdc03f4adaa5c475b62f7bd770764431dedd162a3a1f173a92d86871e63a`.
The only loader correction is deterministic local-cache pinning: execution
sets `SALLM_AFRIHG_CACHE_ONLY=1` and cannot request remote data.

Dataset root: `/home/lmbanr001/masters/sallm/data/afrihg_cache`.

- `xho_train.csv`: `07e2c01be2a5a187559636f4a7f04be761ce8ef497e9cd0f1a9f6bbd605a9e2b`
- `xho_dev.csv`: `e8ce73f843f1c997cdefeb5bac9c48e76ba43704e2dec299f3383d651b35cc58`
- `xho_test.csv`: `67adfa18ead0d9a39b8fd3f4701da0c121dba7438fd5ea04fa68afc73221b3f6`
- `zul_train.csv`: `75fb3b05a4d2f9e9a99200c28bddf9486f37c3444e645b6968b58423c69482ba`
- `zul_dev.csv`: `b84feccc95717cbf442a5ad61d34d38de9c803c9bcc4a1925201186a050faaaf`
- `zul_test.csv`: `9b6feef24111e84ed385ea563728e220ebb99387a9d9af1b77f4fb80d7c0d90b`

The expected structural-verification counts are the already-established
official split sizes: 1,305 Xhosa rows and 1,776 Zulu rows per adapter arm.

## Terminality

Fresh source, base, dataset, and adapter manifests and all four resolved
configs must verify before submission. Once the job opens an official test
payload, its outcome is terminal: preserve success or failure and never retry,
correct, or reopen it. Sheet E/F/G remain blank until all intended official
artifacts are structurally verified and read back exactly.
