# SALLM advisor meeting update - 2026-07-09

Coverage window: Thursday 2026-07-02 through Wednesday 2026-07-08.

## Short version

This week was mainly about making the GatedDeltaNet run real. There is still no completed SALLM GDN downstream result, so do not compare it to LLaMA/Mamba/xLSTM downstream scores yet. But the full-scale shallow/wide GDN pretraining run is now a real training run rather than a queue/config hypothesis: job `997801` is running on 4x L40S, has passed repeated checkpoint saves, reached 90% by the latest 2026-07-08 check, and the loss/token-accuracy curve is healthy.

The research story for advisors:

- GDN is now operationally viable at SALLM scale.
- Earlier GDN failures were launch/config/save/runtime issues, not evidence against the architecture.
- The active GDN run still needs to finish, then be audited and evaluated downstream before making architecture claims.
- Existing xLSTM task-head conclusions from last week still stand; no new xLSTM/Mamba/LLaMA downstream leaderboard result landed this week.

## GDN path this week

At the start of the week, the active GDN job was still queue-bound. A100 job `973395` had a projected start around `2026-07-09T03:42:28`, so it was cancelled and replaced with an L40S job to get a sooner full-node slot.

Replacement submitted on 2026-07-02:

- Cancelled A100 job: `973395` (`gdn33w-b6-wbfix`), no runtime, no checkpoints.
- New L40S job: `990361` (`gdn33w-b6-l40s`)
- Launcher: `~/masters/sallm/scripts/train_final_gated_deltanet_l40s_4gpu_shallowwide_full_b6ga8_wandbtimeout_20260702.sh`
- Partition/account/GRES: `l40s` / `l40sfree` / `gpu:l40s:4`
- Config preserved: `base/gated_deltanet_125m_shallowwide_full_4x40_pack_nogc_b6ga8_20260628.yaml`
- W&B robustness preserved: `WANDB_INIT_TIMEOUT=300`, `WANDB__SERVICE_WAIT=300`

The L40S switch got the run out of the long A100 `AssocGrpGRES` wait, but it exposed real full-scale checkpoint-save issues.

## Failed full-scale attempts and fixes

### Job `990361`: training worked, checkpoint save failed

Job `990361` ran from `2026-07-04T05:25:08` to `2026-07-04T08:14:29`, then failed:

- State: `FAILED`, exit `1:0`
- Elapsed: `02:49:21`
- It reached step `2500` of `38590`
- Speed near failure: about `4.00s/it`
- Loss near failure: about `4.0919`
- Mean token accuracy: about `0.3336`
- Tokens seen: `979135917`
- Incomplete artifact: `checkpoint-2500/config.json`, total checkpoint dir only `2.0K`

First actionable error:

- `TypeError: Object of type ListConfig is not JSON serializable`
- Trigger: `transformers` saving `model.config` during `_save_checkpoint`

Fix:

- `src/main/sallm/config.py`: recursively resolves nested OmegaConf containers in plain dict inputs.
- `src/main/sallm/models/factory.py`: resolves `model.config` through `to_resolved_dict(..., name="model config")` before constructing the Transformers config.
- Tests:
  - `PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/test_config.py tests/models/test_factory.py`
  - Result: `2 passed`
- Kombuys smoke: build model -> `save_pretrained` -> `AutoModelForCausalLM.from_pretrained` passed; `layer_types_type` was `list`.

Interpretation: this was a serialization bug. The model had already trained for nearly 3 hours at usable speed, so this was not a GDN architecture failure.

### Job `995696`: launcher override failed before training

Replacement job:

- Job: `995696` (`gdn33w-b6-l40s-r2`)
- Started: `2026-07-06T01:36:29`
- Failed after `00:00:39`
- Exit: `1:0`

First actionable error:

- Hydra rejected CLI override `wandb.name=...`
- Message: `Key 'wandb' is not in struct`

Fix:

- Baked W&B name and fresh output/log dirs directly into copied config:
  - `src/conf/base/gated_deltanet_125m_shallowwide_full_4x40_pack_nogc_b6ga8_listconfigfix_20260706.yaml`
- Removed problematic CLI overrides from launcher.
- Submitted job `997603` (`gdn33w-b6-l40s-r3`).

Interpretation: launcher/config syntax failure, not model failure.

### Job `997603`: dtype compatibility failure before training

Job `997603` failed quickly:

- State: `FAILED`
- Exit: `1:0`
- Elapsed: `00:00:36`
- Start/end: `2026-07-07T00:02:13` to `2026-07-07T00:02:49`

First actionable error:

- Qwen3Next model construction reached `config.dtype is None`
- Transformers tried `torch.get_current_dtype()`
- HEX Torch build does not have `torch.get_current_dtype`
- Error: `AttributeError: module 'torch' has no attribute 'get_current_dtype'`

Fix:

- `src/main/sallm/models/factory.py`: set `model_config_obj.dtype` from the already-resolved training dtype before constructing the model.
- Tests:
  - `PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/test_config.py tests/models/test_factory.py`
  - Result: `2 passed`
- Submitted replacement job `997801` (`gdn33w-b6-l40s-r4`) with output/log suffix `listconfigfix-dtypefix-20260707`.

Interpretation: runtime compatibility/config propagation issue, not evidence against GDN throughput or learning.

## Active GDN run: job `997801`

Job `997801` is the current active full-scale run:

- Name: `gdn33w-b6-l40s-r4`
- Partition/account/GRES: `l40s` / `l40sfree` / `gpu:l40s:4`
- Node: `srvrocgpu016`
- Start: `2026-07-07T03:49:02`
- Wall end: `2026-07-09T03:49:02`
- Planned steps: `38590`
- Output suffix: `listconfigfix-dtypefix-20260707`

It passed the previous blockers:

- Model construction dtype issue: passed.
- Checkpoint serialization/ListConfig issue: passed.
- Repeated checkpoint saves: passed.
- Training speed: about `4.00s/it`, which keeps the run inside the 48h walltime if stable.

Selected training milestones:

| Time | Progress | Loss | Mean token acc | Epoch | Tokens | Checkpoints |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-07-07 06:50 | around step `2850` | `3.9237` | `0.3491` | `0.37` | `1.116B` | `checkpoint-2500` |
| 2026-07-07 09:48 | around step `5000` | `3.5288` | `0.3846` | `0.69` | `2.099B` | `checkpoint-2500`, `checkpoint-5000` |
| 2026-07-07 12:18 | around step `7500` | `3.4094` | `0.3970` | `0.99` | `2.981B` | through `checkpoint-7500` |
| 2026-07-07 21:00 | around step `15000` | `3.2504` | `0.4149` | `2.00` | `6.035B` | through `checkpoint-15000` |
| 2026-07-08 02:32 | step `20356/38590` (`53%`) | `3.1892` | `0.4231` | `2.64` | `7.970B` | through `checkpoint-20000` |
| 2026-07-08 06:35 | step `24024/38590` (`62%`) | `3.1456` | `0.4297` | `3.11` | `9.407B` | through `checkpoint-22500` |
| 2026-07-08 08:02 | step `25292/38590` (`66%`) | `3.1477` | `0.4288` | `3.28` | `9.904B` | through `checkpoint-25000` |
| 2026-07-08 20:46 | step `34902/38590` (`90%`) | `3.1151` | `0.4340` | `4.52` | `13.668B` | `checkpoint-30000`, `checkpoint-32500` |

Interpretation:

- This is the strongest GDN evidence so far.
- It is not a final result yet because the run is not complete and has not been audited/evaluated downstream.
- But it is now fair to say the shallow/wide GDN configuration trains at SALLM scale and checkpointing works.

## Early comparison read

GDN is still not comparable to LLaMA/Mamba/xLSTM downstream scores. It is only in base pretraining.

Useful early read:

- Around 7 hours into r4, GDN was near loss `3.48`-`3.50`, token accuracy `0.386`-`0.390`, epoch `0.81`-`0.82`, with two checkpoints.
- Existing notes say the xLSTM native pretrain curve was around loss `4.4`-`4.5` near 7.3% progress and `3.8`-`3.9` around 27%-31%.
- This is promising, but not perfectly apples-to-apples because token packing, configs, and step counts differ.
- GDN is slower per step than xLSTM, about `4.0s/it` versus xLSTM around `2.1`-`2.4s/it`, but has fewer planned steps.

Advisor phrasing:

"The GDN pretraining curve looks healthy and may be stronger than xLSTM's early LM curve, but we should not claim architectural superiority until the run finishes and the same downstream audits/evals are done."

## Kombuys sidecar GDN checks

Kombuys was used for small BabyLM-style GDN runtime/optimization checks while full HEX jobs queued.

Completed sidecars:

| Run | Config | Result |
| --- | --- | --- |
| `gdn-h640l33-s256-shallowwide-probe-20260702` | hidden 640, 33 layers, seq256, batch 2, 1000 steps | trained/saved/reloaded; first loss `10.9156`, last loss `2.9262`, reload loss `6.0924` |
| `gdn-h640l33-s512-shallowwide-probe-20260702` | hidden 640, 33 layers, seq512, batch 1, 1000 steps | trained/saved/reloaded; first loss `10.9729`, last loss `4.1074`, reload loss `6.3923` |
| `gdn-h640l33-s256-synthentity5k-empty50-20260702` | seq256, 5000 synthetic entity examples, empty rate 0.5 | trained/saved/reloaded; first loss `11.0204`, last loss `2.7947`, reload loss `1.9560` |
| `gdn-h640l33-s512-synthentity5k-empty50-20260703` | seq512, 5000 synthetic entity examples, empty rate 0.5 | trained/saved/reloaded; first loss `10.9623`, last loss `5.3850`, reload loss `3.5880` |

Interpretation:

- The small sidecars weaken the hypothesis that the shallow/wide GDN architecture has basic runtime/save/reload problems.
- They do not answer full-scale SALLM quality or downstream behavior.
- They are useful as runtime sanity checks, not as dissertation-level SALLM results.

## Operational issues

HEX monitoring was repeatedly blocked by SSH/VPN/DNS timeouts. This created long blind windows, especially on 2026-07-04, 2026-07-05, and parts of 2026-07-08.

Important distinction for advisors:

- Connectivity misses are not job failures.
- The confirmed job failures were specific and actionable:
  - `ListConfig` serialization at checkpoint save.
  - Hydra override path for nested config.
  - Torch dtype compatibility during model construction.
- Active r4 training has not shown a model/runtime failure since those fixes.

Scratch remained high but stable:

- Typical reported quota: `/home` 3GB/10GB, `/scratch` about 85GB-87GB/100GB.
- No checkpoints/results were deleted this week without approval.
- Incomplete old `checkpoint-2500` from failed `990361` remains and should not be used as a resume point.

## What to ask advisors

1. Once `997801` completes, should the immediate next step be clean base LM audit first, or straight downstream fine-tuning/evaluation?
2. Which downstream evals should be first for GDN: task-head diagnostics, generative HPO-selected task suite, or a smaller base-loss audit against xLSTM/Mamba/LLaMA?
3. Should we compare GDN primarily as a base architecture or as a downstream-adapted model family?
4. How much weight should we place on the promising early LM curve given token packing/config differences from xLSTM?

## Immediate next steps

1. Monitor `997801` until completion or failure.
2. If complete, pull final trainer state, final checkpoint metadata, W&B summary, and loss/accuracy curve.
3. Run a clean base LM audit before any downstream claim.
4. Queue the smallest downstream confirmation: likely task-head probes first, then selected generative tasks.
5. Update `sallm_progress.md` only after the GDN run completes and the result is defensible.
