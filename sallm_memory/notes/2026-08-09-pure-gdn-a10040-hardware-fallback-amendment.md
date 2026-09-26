# Pure-GDN base-correction A100-40GB hardware fallback amendment — 2026-08-09

Status: **frozen prospectively at 15:17 SAST, before cancellation or submission
of the replacement jobs and before any corrected result exists**.

## Reason

At `2026-08-09 14:39:35 SAST`, Slurm lost contact with the frozen
`gpu:ampere` node `srvrocgpu010`. Corrected base jobs `1210825/1210826` ended
`NODE_FAIL` simultaneously with other colocated work and produced no
`evaluation_summary.json`. Unchanged retries `1210850/1210851` never started
and remain pending solely because `srvrocgpu010` is
`DOWN+NOT_RESPONDING`. No held-out score or prediction from these attempts was
available or consulted.

At 15:17 SAST, `srvrocgpu009` was idle. Slurm describes it as four
A100-40GB devices under GRES `gpu:amperemk`. The user explicitly authorized
progressing on another GPU. This amendment permits only that same-capacity
A100-40GB fallback; A100-80GB and L40S remain disallowed.

## Frozen fallback contract

- Cancel never-started pending jobs `1210850/1210851` before submitting any
  replacement, preventing duplicate execution if `srvrocgpu010` recovers.
- Preserve all `r2` and `r3` jobs, logs, manifests, and output directories.
- Submit exactly one replacement for lane 14 T2X and one for lane 15 AfriHG
  with new `r4` result/run prefixes.
- Use account/partition/qos `nlpgroup/a100/nlpgroup`, one
  `gpu:amperemk`, 24 hours, one node, and eight CPUs per task.
- At runtime, persist `nvidia-smi` identity and hard-fail unless the allocated
  device name contains `A100` and total memory lies between 39,000 and 42,000
  MiB. This is a hardware-identity gate, not a metric gate.
- Retain the exact immutable source-set digest
  `5b8992861abaa6fe90904feafffc45552fef59ccaefecbaf99bd6672dc3228e8`,
  canonical model, six model hashes, BF16, zero-shot, no adapter, no merge,
  few-shot count zero, seed/data/prompt/scoring contracts, and all evaluator
  implementation corrections unchanged.
- Generate and verify a complete execution manifest before evaluation. Record
  this amendment's SHA-256 in the execution-manifest command label and copy
  the byte-identical amendment outside the immutable source tree into the
  correction artifact area.
- Do not restart invalid General job `1207524` and do not submit HPO while the
  base correction gate remains open.
- A poor corrected score is a valid scientific outcome and cannot trigger a
  rerun or protocol change. Only an independently documented infrastructure
  or hard-contract failure before a summary may justify further action.

## Post-run gate

For each replacement, require Slurm `COMPLETED 0:0`, verified source/model/run
manifests, correct A100-40GB hardware record, canonical model/protocol fields,
an `evaluation_summary.json` hash, and full prediction audits for emptiness,
uniqueness, shared prefixes, repetition, and target-language correctness.
Only then may Sheet rows 41--43 be replaced and the base suite move from
`14/16` to `16/16` scientifically trusted lanes.
