# Pure-GDN T2X family-wise official-test execution freeze

Frozen prospectively on 1 September 2026 after the metric-free CUDA gate and
before any current-adapter official held-out access.

Gate job `1279771` completed `0:0`. Its read-only result is
`/scratch/lmbanr001/masters/sallm/results/pure_gdn_runtime_gate/20260831-familywise-release-v4/runtime_gate.json`,
SHA-256 `d1f13d836e18b77f2af86db70466b5c19818a52a0e4daad34d2b645b81f1248e`.
It proves exact-checkpoint save/reload integrity, finite BF16 forward/backward
with complete finite trainable gradients, and deterministic greedy generation.
It contains no task or held-out data.

The one authorized T2X arm is the already frozen seed-42 b7 Xhosa adapter at
checkpoint 1932. Its Kombuys source is
`/scratch/alombard/masters/sallm/checkpoints/adapter_hpo_v3/pure_gdn/t2x/stage_b/b7/seed_42/final_adapter`.
The immutable HEX destination is
`/scratch/lmbanr001/masters/sallm/checkpoints/frozen_family_winners/t2x_b7_seed42_20260813/final_adapter`.
The adapter and config SHA-256 values must be exactly
`f95725b9e5ba175b46a38ad89c61fcbbd459cac2b63339b9d64afd2fb3ff2bc0`
and `e21938cac6ebfa81faa7855b72f9d0ad7ce9dd892f08412ccd9ea2bdb77235ea`.
No training or evaluation may run on Kombuys; the existing frozen files are
transferred byte-for-byte and evaluated only on the ratified A100-80GB lane.

The execution snapshot is
`/home/lmbanr001/masters/sallm_snapshots/pure-gdn-t2x-official-test-20260901-v1-daf6d8cf`.
The official wrapper SHA-256 is
`4bc7a937adea2d74470d695b45db02699e6c2e796e344eb1a1d05bede0a3d31b`;
the structural verifier SHA-256 is
`daf6d8cf004441cdf7d56a232d83b6e193b3ccbbacf93c88202541c0166985d9`.
The base checkpoint remains the exact gate-bound
`/scratch/lmbanr001/masters/sallm/checkpoints/sallm-pure-gdn-125m/a10080-3epoch-20260802/final_model`.

The unchanged frozen evaluation contract is `eval/run_mamba_t2x_xho`: Xhosa
official test split, 378 rows, zero-shot, `t2x_verbalisation/v1`, BF16,
unmerged local PEFT adapter, beam width 5, maximum 512 new tokens, and corrected
post-BOS generation. The output root is fixed to
`/scratch/lmbanr001/masters/sallm/results/official_test/familywise_20260901/t2x_xho`.
The wrapper must verify immutable source, runtime, base, and adapter manifests
before loading task data. The post-run verifier checks exact task/language/row
coverage, finite metrics, agreement among all metric artifacts, and hashes;
it does not expose a metric value to any control decision.

This is the one official T2X access. Its metric is report-only. It cannot
change, retry, prompt, decode, select, schedule, or correct any unfinished
family. A weak or strange result remains final. Sheet cells stay blank until
the terminal artifact, sidecars, and exact readback verify.
