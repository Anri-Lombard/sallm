# Kombuys LLaMA T2X a2 W&B launch correction — 2026-08-30

Preregistered at 11:00 SAST after the failed launcher was preserved and
before any corrected execution. No model, dataset, checkpoint, validation
artifact, or metric was loaded or produced by the failed attempt.

## Preserved failed attempt

- Candidate remains unchanged: LLaMA-125M T2X Stage-A `a2`, seed 42, on
  Kombuys GPU 0 RTX 5090.
- The initial launcher omitted the frozen lane's `WANDB_MODE=offline`
  environment flag and exited during W&B initialization with `UsageError: No
  API key configured`.
- The only files under the failed output root are the execution manifest and
  HPO trial metadata. Preserve the root and never reuse or overwrite it:
  `/scratch/alombard/masters/sallm/checkpoints/adapter_hpo_v3/llama125/t2x/stage_a/a2/seed_42`.
- Failed log SHA-256:
  `886b3f84c786ebf8780fc3ffeb88143a42c956e79c416f343bfdec475db2d990`.
- Failed execution-manifest SHA-256:
  `2dc6696217878f31053c2b1d16d1ec8546dd3f4738c269fce1c892ef0a32f194`.
- Failed HPO-trial SHA-256:
  `6cb5f5211d7ff2aee0414f1acf28badbf9fd8f125213b20c94b433a8befd6dbb`.

## One-time prospective correction

- Keep every scientific field, candidate parameter, seed, model, tokenizer,
  data split, prompt, metric, checkpoint rule, target module, runtime, and GPU
  assignment unchanged.
- Add only `WANDB_MODE=offline`, matching the already verified a0/a1 runtime.
- Use isolated output root
  `/scratch/alombard/masters/sallm/checkpoints/adapter_hpo_v3/llama125/t2x/stage_a/a2/seed_42_wandb_offline_correction`
  and isolated log
  `/scratch/alombard/masters/sallm/logs/jobs/hpo-llama125-t2x-stage_a-a2-seed42-wandb-offline-correction-kombuys-rtx5090.log`.
- Submit this corrected attempt once. Never use the failed metadata in
  ranking, and never repeat or relax the correction.
