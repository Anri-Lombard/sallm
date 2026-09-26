# Pure-GDN pretraining monitor — 2026-08-05

## Confirmed state at 00:21 SAST

- Canonical recovery job `1179847` is `RUNNING` on two A100-80GB GPUs on
  `srvrocgpu011`, at step `41,766/67,498` after `11:18:01` elapsed.
- Checkpoint rotation retains `checkpoint-40000` and `checkpoint-41000`.
  Step-41,000 validation loss improved to `3.4477896690`, continuing the
  validation-only improvement trajectory.
- No traceback, NCCL timeout, or CUDA OOM follows the stale predecessor error
  in the append-only log. Boundary canary `1179842` remains `COMPLETED 0:0`;
  accepted implementation canary `1165989` remains preserved.
- Scratch is `90/300 GB` (`30.2%`). The only owned Slurm GPU work is job
  `1179847` on `gres/gpu:ampere80:2`; no owned A100-40GB or L40S work is active
  or schedulable.
- ETA remains approximately `2026-08-05 21:00 SAST`, safely inside the current
  wall-time allocation. Kombuys readiness diagnostics are complete and its
  GPUs were idle at the latest read-only check. There is no current blocker.

## Next gate

- Continue validation/checkpoint monitoring through step `67,498` and
  `final_model`. Before downstream validation-only work, require AutoModel
  reload, exact state-dict roundtrip, deterministic generation, and exact
  final artifact SHA256 hashes.

## Follow-up at 01:21 SAST

- Job `1179847` reached step `43,000` and entered its validation/checkpoint
  boundary. At inspection time, checkpoint rotation retained
  `checkpoint-41000` and `checkpoint-42000`; step 43,000 was not yet counted as
  a completed gate.
- Step-42,000 validation loss improved to `3.4442317486`. Slurm remained
  `RUNNING` on two `ampere80` GPUs, and no new traceback, NCCL timeout, or CUDA
  OOM followed the stale predecessor error.
- Scratch remained `90/300 GB` (`30.2%`), no owned A100-40GB or L40S work was
  active or schedulable, ETA remained approximately `2026-08-05 21:00 SAST`,
  and there was no blocker.

## Follow-up at 02:23 SAST

- Job `1179847` reached step `44,195/67,498` and retained checkpoints
  43,000/44,000. Validation loss improved to `3.4407024384` at step 43,000 and
  `3.4379413128` at step 44,000.
- Slurm remained `RUNNING` on two `ampere80` GPUs. No new traceback, NCCL
  timeout, or CUDA OOM followed the stale predecessor error.
- Scratch remained `90/300 GB` (`30.2%`), no owned A100-40GB or L40S work was
  active or schedulable, and ETA remained approximately
  `2026-08-05 21:00 SAST` within wall time. There was no blocker.

## Follow-up at 03:24 SAST

- Job `1179847` reached step `45,450/67,498` and retained checkpoints
  44,000/45,000. Step-45,000 validation loss improved to `3.4351961613`.
- Slurm remained `RUNNING` on two `ampere80` GPUs. No new traceback, NCCL
  timeout, or CUDA OOM followed the stale predecessor error.
- Scratch remained `90/300 GB` (`30.2%`), no owned A100-40GB or L40S work was
  active or schedulable, and ETA remained approximately
  `2026-08-05 21:00 SAST` within wall time. There was no blocker.

## Follow-up at 04:26 SAST

- Job `1179847` reached step `46,706/67,498` and retained checkpoints
  45,000/46,000. Step-46,000 validation loss improved to `3.4325997829`.
- Slurm remained `RUNNING` on two `ampere80` GPUs. No new traceback, NCCL
  timeout, or CUDA OOM followed the stale predecessor error.
- Scratch remained `90/300 GB` (`30.2%`), no owned A100-40GB or L40S work was
  active or schedulable, and ETA remained approximately
  `2026-08-05 21:00 SAST` within wall time. There was no blocker.

## Follow-up at 05:26 SAST

- Job `1179847` reached step `47,937/67,498` and retained checkpoints
  46,000/47,000. Step-47,000 validation loss improved to `3.4308600426`.
- Slurm remained `RUNNING` on two `ampere80` GPUs. No new traceback, NCCL
  timeout, or CUDA OOM followed the stale predecessor error.
- Scratch remained `90/300 GB` (`30.2%`), no owned A100-40GB or L40S work was
  active or schedulable, and ETA remained approximately
  `2026-08-05 21:00 SAST` within wall time. There was no blocker.

## Follow-up at 06:34 SAST

- Job `1179847` reached step `49,291/67,498` and retained checkpoints
  48,000/49,000. Validation loss improved to `3.4292924404` at step 48,000 and
  `3.4275496006` at step 49,000.
- Slurm remained `RUNNING` on two `ampere80` GPUs. No new traceback, NCCL
  timeout, or CUDA OOM followed the stale predecessor error.
- Scratch remained `90/300 GB` (`30.2%`), no owned A100-40GB or L40S work was
  active or schedulable, and ETA remained approximately
  `2026-08-05 21:00 SAST` within wall time. There was no blocker.

## Follow-up at 07:38 SAST

- Job `1179847` reached step `50,606/67,498` and retained checkpoints
  49,000/50,000. Step-50,000 validation loss improved to `3.4262776375`.
- Slurm remained `RUNNING` on two `ampere80` GPUs. No new traceback, NCCL
  timeout, or CUDA OOM followed the stale predecessor error.
- Scratch remained `90/300 GB` (`30.2%`), no owned A100-40GB or L40S work was
  active or schedulable, and ETA remained approximately
  `2026-08-05 21:00 SAST` within wall time. There was no blocker.

## Follow-up at 09:38 SAST

- Job `1179847` reached step `53,010/67,498` and retained checkpoints
  52,000/53,000. Validation loss improved across all intervening gates:
  `3.4249389172` at step 51,000, `3.4241423607` at step 52,000, and
  `3.4233756065` at step 53,000.
- Slurm remained `RUNNING` on two `ampere80` GPUs. The elevated instantaneous
  progress-bar time immediately after step 53,000 reflected the just-completed
  validation/checkpoint boundary; no new traceback, NCCL timeout, or CUDA OOM
  followed the stale predecessor error.
- Scratch remained `90/300 GB` (`30.2%`), no owned A100-40GB or L40S work was
  active or schedulable, and ETA remained approximately
  `2026-08-05 21:00 SAST` within wall time. There was no blocker.

## Follow-up at 10:38 SAST

- Job `1179847` reached step `54,232/67,498` and retained checkpoints
  53,000/54,000. Step-54,000 validation loss improved to `3.4228487015`.
- Slurm remained `RUNNING` on two `ampere80` GPUs. No new traceback, NCCL
  timeout, or CUDA OOM followed the stale predecessor error.
- Scratch remained `90/300 GB` (`30.2%`), no owned A100-40GB or L40S work was
  active or schedulable, and ETA remained approximately
  `2026-08-05 21:00 SAST` within wall time. There was no blocker.

## Follow-up at 11:37 SAST

- Job `1179847` reached step `55,456/67,498` and retained checkpoints
  54,000/55,000. Step-55,000 validation loss improved to `3.4223008156`.
- Slurm remained `RUNNING` on two `ampere80` GPUs. No new traceback, NCCL
  timeout, or CUDA OOM followed the stale predecessor error.
- Scratch remained `90/300 GB` (`30.2%`), no owned A100-40GB or L40S work was
  active or schedulable, and ETA remained approximately
  `2026-08-05 21:00 SAST` within wall time. There was no blocker.

## Follow-up at 12:41 SAST

- Job `1179847` reached step `56,754/67,498` and retained checkpoints
  55,000/56,000. Step-56,000 validation loss improved to `3.4219977856`.
- Slurm remained `RUNNING` on two `ampere80` GPUs. No new traceback, NCCL
  timeout, or CUDA OOM followed the stale predecessor error.
- Scratch remained `90/300 GB` (`30.2%`), no owned A100-40GB or L40S work was
  active or schedulable, and ETA remained approximately
  `2026-08-05 21:00 SAST` within wall time. There was no blocker.

## Follow-up at 13:39 SAST

- Job `1179847` reached step `57,944/67,498` and retained checkpoints
  56,000/57,000. Step-57,000 validation loss improved to `3.4216995239`.
- Slurm remained `RUNNING` on two `ampere80` GPUs. No new traceback, NCCL
  timeout, or CUDA OOM followed the stale predecessor error.
- Scratch remained `90/300 GB` (`30.2%`), no owned A100-40GB or L40S work was
  active or schedulable, and ETA remained approximately
  `2026-08-05 21:00 SAST` within wall time. There was no blocker.

## Follow-up at 14:40 SAST

- Job `1179847` reached step `59,151/67,498` and retained checkpoints
  58,000/59,000. Validation loss improved to `3.4215867519` at step 58,000 and
  `3.4214093685` at step 59,000.
- Slurm remained `RUNNING` on two `ampere80` GPUs. No new traceback, NCCL
  timeout, or CUDA OOM followed the stale predecessor error.
- Scratch remained `90/300 GB` (`30.2%`), no owned A100-40GB or L40S work was
  active or schedulable, and training ETA was approximately
  `2026-08-05 21:20 SAST`, with final verification expected shortly after.
  There was no blocker.
- Reporting terminology is now explicit: describe the contract as a
  `6,635,323,392` token-slot budget matched to the corrected xLSTM
  three-epoch schedule, not as three literal passes over the streamed dataset.

## Canonical completion and verification

- Canonical job `1179847` completed `0:0` at `2026-08-05 21:32:40 SAST`
  after `1-08:28:29`, reaching exactly `67,498/67,498` optimizer steps. The
  executed contract is `6,635,323,392` token slots, xLSTM-matched
  three-epoch-equivalent, not three literal streamed-dataset passes.
- Final validation loss was `3.4211919308`; aggregate reported training loss
  was `1.9098903366`. Checkpoint rotation retained `checkpoint-67000` and
  `checkpoint-67498`, and canonical `final_model` was saved at
  `/scratch/lmbanr001/masters/sallm/checkpoints/sallm-pure-gdn-125m/a10080-3epoch-20260802/final_model`.
- The mandatory post-checkpoint verifier completed inside the `set -e` Slurm
  job and reported exact model identity and parameter count
  (`127,425,448`), AutoModel reload, `save_load_integrity=true`, and
  `deterministic_greedy_generation=true`. The job could not have exited `0:0`
  if this verifier had failed.
- Exact final-model SHA256 manifest:
  - `config.json`: `114fd68f5f522b3627aa10fc34d029e9e200adb5ebb228e8f826c64b93eddf5b`
  - `generation_config.json`: `8abc09d0606da28291b9ab4be66db068d3747ddea747d28f7560930928a3eba5`
  - `pytorch_model.bin`: `37bcc3d080dafcc5d8812821b969a46ef3208b84341bc32ae2c54dd94a9d8108`
  - `special_tokens_map.json`: `72a8eb0b88e02619b0c3b7da0f6b1fab8ee29283f6f18cc362b9d63749c1d628`
  - `tokenizer_config.json`: `169bedadc3f18d3d1bad46dd10450d811f7a14dc50bb752f0c30dc899b5f0b3a`
  - `tokenizer.json`: `3be3a5fda9551681d05a215392a292cff148b132fb9a5a7c693f278d37f20d13`
- Final-model size is `248M`; scratch remained `90/300 GB` (`30.2%`). No
  owned A100-40GB or L40S job was active or schedulable, and the owned Slurm
  queue was empty after completion. All final-model gates passed; downstream
  work may now proceed on validation only, with held-out test still untouched.
