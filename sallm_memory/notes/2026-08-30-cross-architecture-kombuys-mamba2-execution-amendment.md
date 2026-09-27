# Mamba-2 T2X execution amendment — 2026-08-30

Recorded at 14:08 SAST before the metric-free preflight and before any Mamba
validation result.

The prospective Mamba/News correction is deployed only in the immutable
Kombuys snapshot
`/scratch/alombard/sallm_snapshots/uniform-adapter-hpo-postbos-mamba2-news-20260830-9530feb6`.
The 11 August baseline snapshot remains unchanged.

Changed runtime files:

- `scripts/run_validation_hpo_trial.sh`:
  `ae2915da47596fc2b784583dbfaae45c90b785e058c62b4910e129d4c3677eb0`
- `scripts/run_pure_gdn_validation_trial.sh`:
  `9a44bc2950b36c615415e2b2cca9a8f2436997bc9cffff6967f9f98db8d2bf09`
- `scripts/verify_mamba_hpo_runtime.py`:
  `9530feb69a2bb413b59cbe2a34b3a238d459a9ab798fbeb9814d6feb25d77f72`

The deployment manifest covers 695 source/config files and has SHA-256
`93b4dd22a23e4c53f5167e2b70323537b63ba1d9da7a0b07083e37856962ce03`.
It verified before the snapshot was made read-only.

The preflight must run on Kombuys GPU 1 with `CUDA_VISIBLE_DEVICES=1`. Its
checkpoint, tokenizer, config, target/count, training-file hashes, rank-32
contract, native extension hashes, memory limit, metric-free behavior, and
training-row index are frozen by the activation note and training-input
amendment. Write its JSON output to a new isolated diagnostics directory and
hash it. Only a successful structural preflight may release Mamba a0.
