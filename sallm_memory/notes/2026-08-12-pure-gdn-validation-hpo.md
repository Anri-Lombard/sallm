# Pure-GDN corrected validation-only HPO — 2026-08-12

## T2X a0 and NER b2 enter their final callbacks — 23:59 SAST

- Kombuys T2X Stage-A `a0` improved again at epoch 3 step `1449`: Xhosa
  chrF is `41.19802690700617`, a `0.69554673759007` gain over checkpoint
  966, resetting patience and retaining checkpoint 1449. Its exact 64-row
  validation artifact has no literal or whitespace-only empty predictions,
  `63` unique raw predictions, and SHA-256
  `c3941759e90773305be04d08fae6f260403a56fa11803f1308d5e0c3dad74ec8`;
  trainer-state SHA-256 is
  `06d9e7388d5fe555883c27cc3d343f6c6ef440c6a6389199d98748e2af3572dd`.
  This remains within-candidate evidence only; no cross-candidate ranking was
  performed.
- `a0` finished all `1932/1932` training steps and entered its final
  validation/generation callback after health-only loss
  `2.4602493949558424`. At 00:01 SAST the step-1932 selection artifact was
  still absent and GPU 1 remained active near `5,972 MiB`, `95%`, `63 C`.
  Foreign GPU 0 remained untouched. The next preregistered T2X candidate
  `a1` will not launch until `a0` is terminal-valid and GPU 1 is idle;
  terminal ETA is tentatively several minutes.
- HEX NER Stage-B `b2` job `1220056` produced a valid epoch-14 step-7574
  mean F1 `0.4658432729648960833333333333` from Tsn/Xho/Zul
  `0.4641768292682428/0.44702708311710265/0.4863259065093428`. This is
  `0.0003625853158310166666666667` below the epoch-13 best, so frozen patience
  correctly advanced to `1/2` and checkpoint 7033 remains retained. The
  artifact has exact `192` rows and `64/language`, no literal empties or parse
  failures, `57` whitespace-only predictions, and Tsn/Xho/Zul unique counts
  `40/54/37`. Debug/state SHA-256 are
  `02756eab96743bd4076d9254afa81f5f1c72b0134c56e28af59f9cb8526b8d15`
  and `dd84afcff1ca26209aedc390970d0ef93a3aa64f223bbcd4eab094f1c95e5331`.
  It completed all `8115/8115` training steps and entered its final callback.
- `b3` `1222348` remains in epoch-7 corrected generation after health-only
  loss `0.3800166700852405`; `b4` `1224858` entered epoch-3 corrected
  generation after health-only loss `0.4479589072301928`. Their step
  3787/1623 artifacts are absent, so no decision was made from loss or partial
  output. All three jobs remain healthy on A100-40GB `srvrocgpu010`; `b5`
  remains unsubmitted at the cap. HEX quota is home `52.0%`, scratch `37.8%`,
  with no owned A100-80GB/L40S work. Trusted base `16/16`; NER seed-42
  terminal-valid `5/11`; T2X terminal-valid `0/11`; winners `0/8`; held-out
  adapter evaluations `0`; Sheet E/F/G blank; Hugging Face blocked.

## T2X improves through epoch 2; NER b4 improves at epoch 2 — 23:29 SAST

- Kombuys T2X Stage-A `a0` produced valid validation-only artifacts at steps
  `483/966`. Xhosa chrF improved from `33.96025389749359` to
  `40.5024801694161`, a gain of `6.54222627192251`; checkpoint 966 is the
  retained within-run best. Both debug artifacts contain exactly `64` Xhosa
  validation rows, no literal or whitespace-only empty predictions, and
  `64/63` unique raw predictions. Debug SHA-256 are
  `085700d9eecce83d110490a26d35572cc1dface27b762abf49ecde375740ac92`
  and `1041cb5f6c8172586b0259529a1983edaa35b0e6a934e34426dd8cef40861025`;
  checkpoint-966 trainer-state SHA-256 is
  `7fbf1ea1e370c0e72643053ba6dce8276c1a008b26ba65b22418cebf3c7dc8d6`.
  This is only within-candidate validation evidence: no candidate ranking is
  permitted until all 11 seed-42 T2X trials are terminal-valid.
- The T2X tmux retry remains healthy in epoch 3 near `1214/1932`. Kombuys GPU
  1 RTX 3080 Ti remains the only visible run GPU, using about `7,418 MiB` at
  light instantaneous utilization after its callback; foreign GPU 0 remains
  untouched. Root has about `25 GB` free and scratch about `2.1 TB` free.
  Subject to callback duration and no early stop, `a0` is tentatively due in
  roughly 45--60 minutes.
- HEX NER Stage-B `b4` job `1224858` improved at epoch 2 step `1082` to exact
  mean F1 `0.2959148954716148833333333333`, from Tsn/Xho/Zul
  `0.2919144496608809/0.2855311355310859/0.31029910122287785`. Its
  `0.1244542917094380133333333333` gain over epoch 1 resets frozen patience
  and retains checkpoint 1082. The artifact has exact `192` rows and
  `64/language`, no literal empty predictions or parse failures, `51`
  whitespace-only predictions, and Tsn/Xho/Zul unique counts `47/58/37`.
  Debug/state SHA-256 are
  `839a6ba783f956ea72a8a7f728e2137aa35265fbb15b14bdb24d4c6da7d2c347`
  and `ccaa0731096ccb3c4856c3464c937dbafb24c4932b5f4988d188244c8e0152d8`.
- `b2` `1220056` remains in epoch-14 corrected generation with step 7574
  absent; `b3` `1222348` is in epoch-7 corrected generation after health-only
  loss `0.3800166700852405`, with step 3787 absent. No decision was made from
  loss or partial output. Jobs `1220056/1222348/1224858` are healthy on
  A100-40GB `srvrocgpu010`; `b5` remains unsubmitted at the three-job cap.
  HEX quota is home `52.0%`, scratch `37.8%`; no owned A100-80GB or L40S work
  exists. NER seed-42 terminal-valid remains `5/11`; T2X terminal-valid
  remains `0/11`; frozen winners `0/8`; held-out adapter evaluations `0`;
  Sheet E/F/G remain blank and Hugging Face remains blocked.

## b3 improves at epoch 6; T2X a0 reaches its first callback — 22:59 SAST

- Stage-B `b3` job `1222348` improved at epoch 6 step `3246` to exact mean
  NER F1 `0.5652522468935781`, from Tsn/Xho/Zul
  `0.5474323298759937/0.5605612998522397/0.5877631109525009`. The
  `0.0091709042897022` gain over epoch 5 exceeds the frozen `0.001`
  threshold, so patience reset and checkpoint 3246 was retained. Coverage is
  exact `192` rows and `64/language`, with no literal empty raw predictions
  or parse failures; `56` are whitespace-only. Tsn/Xho/Zul unique raw-output
  counts are `41/55/37`. Debug/state SHA-256 are
  `25efeaefdc02f25bb610a0a68b2832d4447ba708951ade8aaad791ab10a0e935`
  and `b4314c653fe6db49fc76a155f45b36bb33cd810421a92329417db696001769d6`.
- `b2` job `1220056` completed its epoch-14 health-only validation loss
  `0.4200252717312384` at step 7574 and entered corrected generation; `b4`
  job `1224858` remains in its epoch-2 corrected generation. Neither step
  7574 nor step 1082 has a complete selection artifact, so neither loss nor
  partial output was used for a decision. All three HEX jobs
  `1220056/1222348/1224858` remain healthy on A100-40GB `srvrocgpu010`; `b5`
  remains unsubmitted at the three-job cap. Quota is home `52.0%`, scratch
  `37.8%`, with no owned A100-80GB or L40S work.
- Kombuys T2X `a0` completed epoch-1 training and health-only validation loss
  `2.658723781419837` at step `483`, then entered the full corrected
  generation callback. No complete chrF artifact exists yet, so no T2X
  metric has been selected or compared. The W&B-offline retry remains alive
  in tmux `sallm-t2x-hpo-a0-r1`; GPU 1 RTX 3080 Ti is healthy at about
  `8,250 MiB`, `96%`, and `62 C`. Foreign GPU 0 remains untouched. Root has
  about `26 GB` free and scratch about `2.1 TB` free. A first complete
  artifact is tentatively expected within tens of minutes; full-run ETA is
  approximately two hours only after steady-state callback timing confirms.
- Scientifically trusted base remains `16/16`; NER seed-42 terminal-valid is
  `5/11`; T2X seed-42 terminal-valid is `0/11`; frozen Multilingual winners
  are `0/8`; held-out adapter evaluations are `0`; Mono is not started. No
  Sheet or Hugging Face write occurred; Sheet E/F/G remain blank.

## Cross-host T2X HPO opens on Kombuys; NER b2 improves — 22:28 SAST

- The prospective whole-family host assignment was frozen before any T2X HPO
  score at SHA-256
  `ca7078aa4bfa36226946e6c00272a793ecd7166f2db8150a8e5b862c94162d51`.
  NER remains entirely on HEX A100-40GB. T2X Stage A, Stage B, and seed
  confirmations 13/42/87 are assigned entirely to Kombuys GPU 1, UUID
  `GPU-fec43e16-3955-238e-e517-e80cf92d0383` (RTX 3080 Ti). Candidate results
  may not cross GPU classes within either family.
- The HEX enhanced snapshot was copied by its manifest file list to read-only
  Kombuys path
  `/scratch/alombard/sallm_snapshots/uniform-adapter-hpo-20260811-6aabf717`.
  All `694/694` source/config files match. The original deployment manifest is
  SHA-256 `da1c4567...57b285`; the registry remains
  `8fdd6ea5...bb726`. All six canonical pure-GDN model hashes match the frozen
  manifest. T2X train/validation data hashes match HEX exactly:
  `train.data=b16dd121...f964`, `train.text=c245b599...aa29`,
  `valid.data=2a61af57...364b`, and `valid.text=ccd63697...f22b`.
- The complete Kombuys preflight execution/environment manifest verified all
  694 files and six model artifacts at SHA-256
  `e00232575162363725bc56b82fdfe6ca8bbd5d08640d32299ce607ef36a80af5`.
  Python is `3.12.3`, PyTorch `2.9.1+cu128`, BF16 is supported, and exactly
  GPU 1 was CUDA-visible. The bounded direct-validation-only batch-4 canary
  passed without opening test files. It verified canonical model identity
  (`GatedDeltaNetForCausalLM`, `attn=None`, `127,425,448` base parameters),
  fresh LoRA shape, BOS/no-terminal-EOS token equality, and batch size 4; its
  artifact SHA-256 is
  `1d5cb4bdf70104abf3fea9f5a8d9f30d3c48b6c7618811ef2944c8df8ee00c36`.
  Peak allocated/reserved memory was `372,868,096/415,236,096` bytes. The
  canary score was deliberately discarded and not recorded. A first canary
  invocation failed before model/data access because the immutable root was
  absent from `PYTHONPATH`; the corrected invocation changed only that path.
- Initial T2X `a0` tmux launch failed before model/data/metric access because
  Kombuys has no W&B API key. Preserve its output and log; execution-manifest
  and trial hashes are `3fb1c271...8086f` and `6517f1de...54a65`. The single
  unchanged logging-only retry runs with W&B offline in tmux
  `sallm-t2x-hpo-a0-r1`, with new output suffix `seed_42-wandb-retry1`.
  Retry execution-manifest/trial hashes are `58e90c6d...8dbe9` and
  `11e8aef8...a76`. It verified `694/694` files, the fast GatedDeltaNet path,
  exact frozen `a0` configuration, `3,859/460` train/validation rows, train
  and evaluation batch 4, gradient accumulation 2, BF16, and began the
  `1,932`-step four-epoch run. At 22:27 GPU 1 used about `1,466 MiB` at 60%
  utilization; ETA is unknown until steady-state step timing appears. GPU 0
  remained untouched and heavily occupied by the foreign wav2vec2 job.
- HEX Stage-B `b2` job `1220056` improved at epoch 13 step `7033` to mean NER
  F1 `0.4662058582807271` from Tsn/Xho/Zul
  `0.4643731478/0.4475677935/0.4866766335`, resetting patience and retaining
  checkpoint 7033. Coverage is exact `192` rows and `64/language`, with no
  literal empty raw predictions or parse failures; `57` are whitespace-only.
  Debug/state hashes are `df0b8498...0443`/`c3476954...fbae`. Jobs
  `1220056/1222348/1224858` remain healthy on A100-40GB `srvrocgpu010`, so
  `b5` remains unsubmitted at the three-job cap. No A100-80GB or L40S work is
  owned. HEX quota is home `52.0%`, scratch `37.8%`.
- Scientifically trusted base remains `16/16`; NER seed-42 terminal-valid is
  `5/11`; T2X seed-42 terminal-valid is `0/11`; frozen Multilingual winners
  are `0/8`; held-out adapter evaluations are `0`; Mono is not started. No
  Sheet or Hugging Face write occurred; Sheet E/F/G remain blank.

## b3 improves epoch 5; b4 initial artifact valid — 21:59 SAST

- Stage-B `b3` job `1222348` produced a valid epoch-5 step-`2705`
  artifact. Tsn/Xho/Zul span micro-F1 is
  `0.5512972474647203/0.5505144745961009/0.5664323057508064`; exact
  arithmetic mean `0.5560813426038758666666666667` matches retained
  trainer-state F1 `0.5560813426038759` to floating precision. Its
  `0.0511304679977406666666666667` gain over epoch 4 exceeds frozen
  threshold `0.001`, so patience reset to zero and epoch 6 resumed near step
  `3136/8115`.
- Fixed Stage-B `b4` retry `1224858` produced its valid initial epoch-1
  step-`541` artifact. Tsn/Xho/Zul F1 is
  `0.15425065731809362/0.16295257108935837/0.19717858287907855`; exact
  arithmetic mean `0.1714606037621768466666666667` matches trainer-state F1
  `0.17146060376217687` to floating precision. This is the preregistered
  candidate's initial within-run point; no low-fidelity pruning or
  cross-candidate selection is permitted. Patience is zero and epoch 2
  resumed near step `772/8115`.
- Both artifacts have exactly `192` rows, `64` per language, finite metrics,
  no literal empty raw strings, no parse failures, and the frozen prompt
  contract on every row. b3 Tsn/Xho/Zul whitespace-only counts are
  `22/8/27`, unique raw-output counts `40/55/37`; b4 counts are `5/9/14` and
  `56/56/50`. b3 debug/state SHA-256 are
  `344e0f858dc85176da6d47f28d5956015b8a0d6a76b6368c45a684aaf3c833ad`
  and `11446d4a2af3ebbdecbccc99a3cc210f2faaa46b9845f5d542e0eb97d1d71364`;
  b4 values are
  `6ef46c8c0feeeb0e0d4fadb28e7769a720d9e27590685442646b6944b3b589c9`
  and `b5c7c3ecded9ff106118a0714fae2c5ed4119a58d663ee5f1ebb593649b9269a`.
- `b2` job `1220056` completed exact epoch-13 loss `0.4188957015821039`
  and remains in corrected generation; two segments began at
  `21:39:22/21:48:28`, with step-`7033` tentatively due `22:10--22:25`.
  b3's next artifact is tentatively due `22:50--23:10`, and b4's next around
  `22:55--23:15`, all output-dependent.
- Jobs `1220056/1222348/1224858` remain healthy on `srvrocgpu010`, one
  A100-40GB `gpu:ampere` each; targeted scans remain empty and no owned
  A100-80GB/L40S work exists. Quota is home `52.0%`, scratch `37.8%`;
  Kombuys was not accessed and remains untouched, Sheet E/F/G remain blank,
  trusted base is `16/16`, NER seed-42 terminal-valid is `5/11`, frozen
  winners are `0/8`, held-out evaluations are `0`, and Mono is not started.

## b2 epoch 12 is first patience miss; b3/b4 callbacks active — 21:29 SAST

- Stage-B `b2` job `1220056` produced a valid epoch-12 step-`6492`
  artifact. Tsn/Xho/Zul span micro-F1 is
  `0.4552082046212279/0.43984191305212866/0.4711779448621056`; exact
  arithmetic mean `0.4554093541784873866666666667` is
  `0.0031122173985638133333333333` below retained epoch-11 best
  `0.4585215715770512`. Frozen patience correctly advanced to `1/2`, kept
  checkpoint `5951`, and epoch 13 resumed near step `6980/8115`.
- The b2 artifact has exactly `192` rows, `64` per language, finite metrics,
  no literal empty raw strings, no parse failures, and the frozen prompt
  contract on every row. Tsn/Xho/Zul whitespace-only counts are `22/8/27`,
  and unique raw-output counts are `41/55/37`. Debug/state SHA-256 are
  `4c6df0b80e2add59a2d04349b51dc4ece92f484e43d477c25241039a9b4499be`
  and `c669358c42becf02c7997707b02d264af2aa69752da0671523e2f6d978b6bd84`.
- `b3` job `1222348` completed exact epoch-5 loss `0.351184297582917` and
  remains in corrected generation; all three segments began by `21:22:59`,
  with step-`2705` absent and tentatively due `21:35--21:45`. Fixed b4 retry
  `1224858` completed exact epoch-1 loss `1.2492069286927858` and remains in
  corrected generation; its first two segments began at `21:07:33/21:16:59`,
  with step-`541` tentatively due `21:45--22:00`. No scientific decision was
  made from either loss. b2's next artifact is tentatively due
  `22:10--22:30`.
- Jobs `1220056/1222348/1224858` remain healthy on `srvrocgpu010`, one
  A100-40GB `gpu:ampere` each; targeted scans remain empty and no owned
  A100-80GB/L40S work exists. Quota is home `52.0%`, scratch `37.8%`;
  Kombuys was not accessed and remains untouched, Sheet E/F/G remain blank,
  trusted base is `16/16`, NER seed-42 terminal-valid is `5/11`, frozen
  winners are `0/8`, held-out evaluations are `0`, and Mono is not started.

## b2/b3 callbacks active; b4 approaches first boundary — 20:59 SAST

- Stage-B `b2` job `1220056` remains in corrected epoch-12 generation after
  exact loss `0.42394250352143353`. All three language segments began by
  `20:49:03`; step-`6492` was still absent at `20:59:10`, so no checkpoint
  or patience decision was made from loss. The output-dependent artifact is
  tentatively due around `21:00--21:10`.
- Stage-B `b3` job `1222348` is in its epoch-5 validation callback; no
  complete loss or step-`2705` F1 artifact exists yet. Its artifact remains
  tentatively due around `21:20--21:40`. Fixed b4 retry `1224858` remains
  healthy near its first epoch boundary at step `528/8115`; its step-`541`
  artifact is tentatively due around `21:35--21:55`.
- Targeted traceback, CUDA/OOM, NCCL, non-finite, manifest, and coverage scans
  are empty. Owned state remains exactly three running A100-40GB jobs
  `1220056/1222348/1224858` on `srvrocgpu010`, with no A100-80GB/L40S work.
  Quota is home `52.0%`, scratch `37.8%`; Kombuys was not accessed and
  remains untouched, Sheet E/F/G remain blank, trusted base is `16/16`, NER
  seed-42 terminal-valid is `5/11`, frozen winners are `0/8`, held-out
  evaluations are `0`, and Mono is not started.

## b1 terminal-valid; b3 improves epoch 4; b4 starts after preserved launcher failure — 20:31 SAST

- Stage-B `b1` job `1219931` completed cleanly `0:0` at `20:06:34 SAST`.
  Its valid epoch-13 step-`7033` Tsn/Xho/Zul span micro-F1 is
  `0.5432415118513274/0.5484956869345177/0.5653786987602433`; exact mean
  `0.5523719658486961333333333333` is `0.0052383270370617667` below the
  epoch-12 numerical best `0.5576102928857579`. This is the second
  frozen-threshold miss after epoch 11 reset, so patience correctly stopped
  the run and restored checkpoint `6492`. NER seed-42 terminal-valid progress
  is now `5/11` (`3/3` Stage-A plus `2/8` Stage-B).
- The terminal b1 artifact has exactly `192` rows, `64` per language, finite
  metrics, no literal empty raw strings, no parse failures, and the frozen
  prompt contract on every row. Tsn/Xho/Zul whitespace-only counts are
  `21/8/27`, and unique raw-output counts are `40/56/37`. Final debug,
  retained-state, execution-manifest, final-adapter weight, and adapter-config
  SHA-256 are `bac19c95c90c092182d23dd0873b9ede024ce2a57ffeb075ce1923280af2d8f1`,
  `f56a1348f063c42cc422f9ec006829f27b9e268eebc1e568da00bc4d91085538`,
  `f67827a869206788a812485fc2e0707a76d112eb8dc936cf1818b1bbf4b8dd22`,
  `94f6028b9cb8cffb65cc9914d541582cb2d744cc3300a27f136280c72602c57c`,
  and `5ecb209c9280c9a53375185a81e4e77d1b0bd4361afa55460d0947502a1390e3`.
- Stage-B `b3` job `1222348` produced a valid epoch-4 step-`2164`
  artifact. Tsn/Xho/Zul F1 is
  `0.4985616010005756/0.5021662468513355/0.5141247759664945`; exact mean
  `0.5049508746061352` matches retained state and improves by
  `0.1014977493973674`, resetting patience. Coverage is exact `192` rows and
  `64` per language, with finite metrics, no literal empty raw strings or
  parse failures, and the frozen prompt contract intact. Tsn/Xho/Zul
  whitespace-only counts are `20/7/26`, unique raw counts `40/56/38`, and
  debug/state SHA-256 are
  `4bef8fa33561c4864741f6f134eaea5ce1590bdc21447cb94969afe662c298ab`
  and `13c3ed834cfbfdcd0c5385703c5227589ef9a18a31ea5e65f839d41b7d8a0fd6`.
- The released slot was assigned to fixed Stage-B `b4`. Initial job
  `1224853` failed in zero seconds before model, data, manifest, or metric
  access because `SALLM_REPO_DIR` was not exported and the wrapper attempted
  the intentionally incomplete mutable tree. That log remains preserved as
  launcher-failure provenance; no b4 result artifact existed. The unchanged
  candidate retry, with immutable source and runtime paths explicitly
  exported, is job `1224858`.
- Job `1224858` started at `20:30:28` on `srvrocgpu010` and verified
  `694/694` immutable source/config files, the A100-40GB fast GatedDeltaNet
  path, and validation-only trial metadata. Frozen seed-42 b4 configuration
  is LR `0.00008981661817441004`, rank/alpha `8/16`, dropout
  `0.0971165418624878`, warmup `0.04982422709465027`. Execution-manifest and
  trial SHA-256 are
  `f4675ec40d169a1e6298dd6ed4fe9db0ab41d79672e191f7c1b0f3d00b196b1a`
  and `62c6e81b29fea2198ebe5f58e317cde2f1fbde82791357617731460c1af25db5`.
- Owned state is three running A100-40GB jobs: `b2/b3/b4`
  `1220056/1222348/1224858`; no A100-80GB/L40S work exists. b2 is in
  epoch-12 corrected generation after exact loss `0.42394250352143353`, with
  step `6492` tentatively due `20:55--21:10`; b3's epoch-5 artifact is
  tentatively due `21:20--21:35`, and b4's first artifact around
  `21:35--21:55`, all output-dependent. Quota is home `52.0%`, scratch
  `37.8%`; Kombuys was not accessed and remains untouched, Sheet E/F/G remain
  blank, trusted base is `16/16`, frozen winners are `0/8`, held-out
  evaluations are `0`, and Mono is not started.

## b2 improves at epoch 11; b1/b3 callbacks active — 19:59 SAST

- Stage-B `b2` job `1220056` produced a valid epoch-11 step-`5951`
  artifact. Tsn/Xho/Zul span micro-F1 is
  `0.461439588688896/0.43819760231495625/0.47592752372730124`; exact
  arithmetic mean `0.4585215715770511633333333333` matches retained
  trainer-state F1 `0.4585215715770512` to floating precision. Its
  `0.0026630207781532933333333333` gain over epoch 10 exceeds frozen
  threshold `0.001`, so patience reset to zero and epoch 12 resumed near
  step `6144/8115`.
- The b2 artifact has exactly `192` rows, `64` per language, finite metrics,
  no literal empty raw strings, no parse failures, and the frozen prompt
  contract on every row. Tsn/Xho/Zul whitespace-only counts are `23/8/28`,
  and unique raw-output counts are `39/54/36`. Debug/state SHA-256 are
  `0ac527c8b68ec57a8bd33baa39333f143e933999845cc08576ed1f98140fa139`
  and `c4f189516a88e0bf4baaa2c832573670453125e77398e2c9907fca38c26f26e1`.
- `b1` job `1219931` is in epoch-13 corrected generation after exact loss
  `0.4200027976337419`; all three language segments had begun by
  `19:52:15`, with step-`7033` absent and tentatively due around
  `20:00--20:10`. `b3` job `1222348` completed exact epoch-4 loss
  `0.36654010758524047` and entered corrected generation; two segments began
  at `19:45:11/19:54:16`, with step-`2164` tentatively due
  `20:15--20:30`. No decision was made from either loss. b2's next artifact
  is tentatively due around `20:55--21:15`.
- Jobs `1219931/1220056/1222348` remain running on `srvrocgpu010`, one
  A100-40GB `gpu:ampere` each; targeted scans remain empty and no owned
  A100-80GB/L40S work exists. Quota is home `52.0%`, scratch `37.7%`;
  Kombuys was not accessed and remains untouched, Sheet E/F/G remain blank,
  trusted base is `16/16`, NER seed-42 terminal-valid is `4/11`, frozen
  winners are `0/8`, held-out evaluations are `0`, and Mono is not started.

## b3 improves at epoch 3; b1/b2 callbacks active — 19:29 SAST

- Stage-B `b3` job `1222348` produced a valid epoch-3 step-`1623`
  artifact. Tsn/Xho/Zul span micro-F1 is
  `0.37280047718456405/0.4128303868369891/0.42472851160475034`; exact
  arithmetic mean `0.40345312520876783` matches retained trainer-state F1
  `0.4034531252087678` to floating precision. Its
  `0.08573398241760233` gain over epoch 2 exceeds frozen threshold `0.001`,
  so patience reset to zero and epoch 4 resumed near step `1993/8115`.
- The b3 artifact has exactly `192` rows, `64` per language, finite metrics,
  no literal empty raw strings, no parse failures, and the frozen prompt
  contract on every row. Tsn/Xho/Zul whitespace-only counts are `18/7/29`,
  and unique raw-output counts are `44/56/35`. Debug/state SHA-256 are
  `80d0d48d03e020f677ad4b46774a711da97d07a89b85a47604aab451fb5b29be`
  and `526bb94fb2a4cfbf405eec55515177ec0f8c57070e3deffb31948b8fbf0882d4`.
- `b2` job `1220056` completed exact epoch-11 loss
  `0.4216838056713232` and entered corrected generation; its first two
  segments began at `19:10:18/19:19:34`, with step-`5951` absent and an
  output-dependent artifact window around `19:45--20:00`. `b1` job
  `1219931` completed exact epoch-13 loss `0.4200027976337419` and began its
  corrected callback at `19:28:15`; step-`7033` is absent and tentatively due
  around `20:00--20:20`. No decision was made from either loss. b3's next
  artifact is tentatively due around `20:20--20:40`.
- Jobs `1219931/1220056/1222348` remain healthy on `srvrocgpu010`, one
  A100-40GB `gpu:ampere` each; targeted scans remain empty and no owned
  A100-80GB/L40S work exists. Quota is home `52.0%`, scratch `37.7%`;
  Kombuys was not accessed and remains untouched, Sheet E/F/G remain blank,
  trusted base is `16/16`, NER seed-42 terminal-valid is `4/11`, frozen
  winners are `0/8`, held-out evaluations are `0`, and Mono is not started.

## b1 epoch-12 marginal numerical best; all three jobs healthy — 19:00 SAST

- Stage-B `b1` job `1219931` produced a valid epoch-12 step-`6492`
  artifact. Tsn/Xho/Zul span micro-F1 is
  `0.54874213836473/0.553729185629334/0.5703595546632098`; exact mean
  `0.5576102928857579` matches the newly retained `checkpoint-6492`. The
  numerical gain over epoch 11 is only `0.0006878515836052`, below the frozen
  early-stopping threshold `0.001`, so the frozen patience counter correctly
  advanced to `1/2` while the trainer retained the numerical-best checkpoint
  and resumed epoch 13 near step `6666/8115`.
- The artifact has exactly `192` rows, `64` per language, finite metrics, no
  literal empty raw strings, no parse failures, and the frozen BOS/chat/EOS
  prompt contract on every row. Tsn/Xho/Zul whitespace-only counts are
  `21/8/26`, and unique raw-output counts are `40/56/38`. Debug/state SHA-256
  are `1f133cd12447eb1957df5677bb6e8c52f56ac53dd51fd038b7e376381e7baf3b`
  and `f56a1348f063c42cc422f9ec006829f27b9e268eebc1e568da00bc4d91085538`.
- `b2` job `1220056` remains healthy in epoch 11 near step `5930/8115` and
  should reach its step-`5951` validation boundary shortly; its
  output-dependent artifact window is roughly `19:45--20:05`. `b3` job
  `1222348` remains in epoch-3 corrected generation after exact loss
  `0.4227915597227869`; all three language segments had started by
  `18:54:29`, with step-`1623` still absent at the check and tentatively due
  around `19:05--19:15`. No decision was made from either loss.
- Jobs `1219931/1220056/1222348` are running on `srvrocgpu010`, one
  A100-40GB `gpu:ampere` each, with targeted fault scans empty and no owned
  A100-80GB/L40S work. Quota is home `52.0%`, scratch `37.7%`; Kombuys was
  not accessed and remains untouched, Sheet E/F/G remain blank, trusted base
  is `16/16`, NER seed-42 terminal-valid is `4/11`, frozen winners are `0/8`,
  held-out evaluations are `0`, and Mono is not started.

## b3 improves at epoch 2; b2 improves at epoch 10 — 18:38 SAST

- Stage-B `b3` job `1222348` produced a valid epoch-2 step-`1082`
  artifact. Tsn/Xho/Zul span micro-F1 is
  `0.3086306800853447/0.31980298166141347/0.3247237666267382`; exact mean
  `0.3177191427911655` matches newly retained `checkpoint-1082`. Its
  `0.14118867509099828` gain over epoch 1 exceeds frozen `0.001`, so patience
  reset. It completed exact epoch-3 loss `0.4227915597227869` and entered its
  next corrected generation callback at `18:30:08`; step-`1623` is absent.
- Stage-B `b2` job `1220056` produced a valid epoch-10 step-`5410`
  artifact. Tsn/Xho/Zul span micro-F1 is
  `0.454290671293157/0.434503108862844/0.47878187224069246`; exact mean
  `0.45585855079889787` matches newly retained `checkpoint-5410`. Its
  `0.020229851946633903` gain over epoch-8 retained best exceeds `0.001`, so
  patience reset and epoch 11 resumed near step `5424/8115`.
- Both artifacts have exactly `192` rows, `64` per language, finite metrics,
  and no literal empty raw strings. b3 debug/state SHA-256 are
  `4974e2998d96c4a7a28630034638a92cdc582a57f2a92a372e03c6268831aac3` and
  `bf9b997fd9288d9263e3f79a90351669756b7cc6f15dc0296b998181bf6ebb8e`;
  b2 values are
  `d0e4bf493f7c63ca87d6ebdb65aa59f15d8df6951391b47994ca68b25d53c793` and
  `8da358de3ffaeb981f9fe75184226043d7166dd058ef8669f698402f1b396d18`.
- `b1` job `1219931` completed exact epoch-12 loss
  `0.41578371409589915` and remains in corrected generation, with two
  segments begun at `18:15:06/18:23:55`; step-`6492` is absent. No decision
  was made from b1 or b3 loss. Output-dependent ETA is b1 `18:45--19:00`, b3
  `19:10--19:30`; b2 next epoch artifact is tentatively `19:45--20:05`.
- Jobs `1219931/1220056/1222348` remain healthy on `srvrocgpu010`, one
  A100-40GB each; targeted fault scans remain empty and no A100-80GB/L40S
  work exists. Quota is home `52.0%`, scratch `37.7%`; Kombuys untouched,
  Sheet E/F/G blank, base `16/16`, NER seed-42 terminal-valid `4/11`, frozen
  winners `0/8`, held-out `0`, Mono not started.

## Stage-B b1 improves at epoch 11; b2 first patience miss — 17:49 SAST

- Stage-B `b1` job `1219931` produced a valid epoch-11 step-`5951`
  artifact. Tsn/Xho/Zul span micro-F1 is
  `0.540983606557327/0.5556978233034073/0.5740858940457237`; exact mean
  `0.5569224413021527` matches newly retained `checkpoint-5951`. Its
  `0.010461328483170385` gain over epoch 10 exceeds frozen `0.001`, so
  patience reset and epoch 12 resumed near step `6120/8115`.
- Stage-B `b2` job `1220056` produced valid epoch-9 step-`4869` mean F1
  `0.4320187263597735` from Tsn/Xho/Zul
  `0.43517249968857896/0.4059122216077123/0.4549714577830293`. This is
  `0.003609972492490454` below epoch-8 best `0.43562869885226396`, so
  `checkpoint-4328` remains retained. This is the first non-improving epoch
  since its patience reset; it correctly resumed epoch 10 near step
  `5400/8115`.
- Both artifacts have exactly `192` rows, `64` per language, finite metrics,
  and no literal empty raw strings. b1 debug/state SHA-256 are
  `3cbc0b36bd0c6d83e2203cb3dcaf715c90617739ed96c8644fe32bd41db57778` and
  `bcc862afbe3802d72a08592630b409c9124ffa6d76c49b4f578c303496c6e009`;
  b2 values are
  `61e1afce8e501de70f35e52d3d436ee12622140de8568aaa31318edb3af24c63` and
  `cf48650c5d5b4dd2d5d686365c7664f68ede4e904d1ee354c30e9e4123571461`.
- `b3` job `1222348` remains in epoch-2 corrected generation after exact
  loss `0.4814153054389812`; its first two segments began
  `17:24:31/17:39:29`, but step-`1082` is absent. No decision was made from
  loss; output-dependent ETA is `18:05--18:20`.
- Jobs `1219931/1220056/1222348` remain healthy on `srvrocgpu010`, one
  A100-40GB each; targeted fault scans remain empty and no A100-80GB/L40S
  work exists. Quota is home `52.0%`, scratch `37.7%`; Kombuys untouched,
  Sheet E/F/G blank, base `16/16`, NER seed-42 terminal-valid `4/11`, frozen
  winners `0/8`, held-out `0`, Mono not started.

## Three corrected callbacks active — 17:19 SAST

- Stage-B `b1` job `1219931` completed exact epoch-11 validation loss
  `0.41051052618204` and is in corrected generation, with its first two
  segments starting `17:00:46/17:09:40`. `b2` job `1220056` remains in
  epoch-9 corrected generation after exact loss `0.427205513312471`, with
  all three segments started by `17:05:16`. `b3` job `1222348` completed
  exact epoch-2 loss `0.4814153054389812` and entered corrected generation at
  `17:15:06`. Step-`5951/4869/1082` artifacts are absent, so no checkpoint or
  patience decision was made from loss.
- Output-dependent ETA is b2 around `17:25--17:35`, b1 around
  `17:35--17:50`, and b3 around `17:55--18:15`. Jobs
  `1219931/1220056/1222348` remain healthy on `srvrocgpu010`, one A100-40GB
  each; targeted fault scans remain empty and no A100-80GB/L40S work exists.
- Quota is home `52.0%`, scratch `37.7%`; Kombuys untouched, Sheet E/F/G
  blank, base `16/16`, NER seed-42 terminal-valid `4/11`, frozen winners
  `0/8`, held-out `0`, Mono not started.

## b1 marginal numerical best; b3 first artifact valid — 16:49 SAST

- Stage-B `b1` job `1219931` produced a valid epoch-10 step-`5410`
  artifact. Tsn/Xho/Zul span micro-F1 is
  `0.5296850800206006/0.5441524932862436/0.5655457651501029`; exact mean
  `0.5464611128189824` matches newly retained `checkpoint-5410`. The
  numerical gain over epoch 9 is only `0.00009749732803343569`, below frozen
  early-stopping threshold `0.001`, so continuing under the preregistered
  patience rule rather than intervening is correct. The trainer retains the
  numerically best checkpoint. It resumed epoch 11 near step `5851/8115`.
- Stage-B `b3` job `1222348` produced its first valid artifact at epoch 1
  step `541`. Tsn/Xho/Zul span micro-F1 is
  `0.1557108538981827/0.17440195194393787/0.199478597258381`, exact mean
  `0.1765304677001672`, matching retained `checkpoint-541`. This is only the
  initial within-candidate point; no low-fidelity pruning or cross-candidate
  rank is permitted. It resumed epoch 2 near step `699/8115`.
- Both artifacts have exactly `192` rows, `64` per language, finite metrics,
  and no literal empty raw strings. b1 debug/state SHA-256 are
  `1bfb9ead810a18d632cc3ae1c202754f3b94f3cd64eed3ddafb4c37a99b5e8ca` and
  `79e3fb55ac7eed3aa80e4f4849ed99e73a38686c0c51183f9de19daf73af6f37`;
  b3 values are
  `12ea8e6f252aae3908d5bf589ebac3bcb1553a00aa83ab32d26257c987f8a12b` and
  `741f49790edd98c83b211ac12246677c8552c104bfd81847334c619d20d3a234`.
- `b2` job `1220056` completed exact epoch-9 validation loss
  `0.427205513312471` and entered corrected generation at `16:41:02`; its
  step-`4869` artifact is absent, so no checkpoint decision was made from
  loss. It is tentatively due around `17:10--17:25`, output-dependent.
- Jobs `1219931/1220056/1222348` remain healthy on `srvrocgpu010`, one
  A100-40GB each; fault scans remain empty and no A100-80GB/L40S work exists.
  Quota is home `52.0%`, scratch `37.7%`; Kombuys untouched, Sheet E/F/G
  blank, base `16/16`, NER seed-42 terminal-valid `4/11`, frozen winners
  `0/8`, held-out `0`, Mono not started.

## Stage-B b2 improves at epoch 8; b1/b3 generation active — 16:19 SAST

- Stage-B `b2` job `1220056` produced a scientifically valid epoch-8
  step-`4328` artifact. Tsn/Xho/Zul span micro-F1 is
  `0.45717948717943724/0.39729403744292036/0.45241257193443424`; exact mean
  `0.43562869885226396` matches newly retained `checkpoint-4328` exactly.
  Its `0.03476144609081261` gain over epoch 7 exceeds frozen `0.001`, so
  patience reset and epoch 9 resumed near step `4604/8115`.
- Independent checks found exactly `192` rows, `64` per language, finite
  metrics, and no literal empty raw strings. Tsn/Xho/Zul whitespace-only
  counts are `22/8/28`, and unique raw-output counts are `40/54/36`.
  Debug/state SHA-256 are
  `3969d89b59d43c74440a4c7e9ec14175ebc71820063e1fbde90d28d8bdb98b98` and
  `11e19866e9ace74821272fa909543df0dc63ff911418fe5b4901d105b3920262`.
- `b1` job `1219931` remains in epoch-10 corrected generation after exact
  loss `0.40588744209601535`; its third segment began at `16:10:17`, but
  step-`5410` is absent. `b3` job `1222348` completed exact epoch-1 loss
  `1.21032742071329` and is in corrected generation; its first two segments
  began at `16:00:36/16:10:00`, but step-`541` is absent. No decision was
  made from either loss. Both artifacts are tentatively expected around
  `16:25--16:40`, output-dependent.
- Jobs `1219931/1220056/1222348` remain healthy on `srvrocgpu010`, one
  A100-40GB each; targeted fault scans remain empty and no A100-80GB/L40S
  work exists. Quota is home `52.0%`, scratch `37.7%`; Kombuys untouched,
  Sheet E/F/G blank, base `16/16`, NER seed-42 terminal-valid `4/11`, frozen
  winners `0/8`, held-out `0`, Mono not started.

## b1/b2 next callbacks active; b3 healthy — 15:48 SAST

- Stage-B `b1` job `1219931` completed exact epoch-10 validation loss
  `0.40588744209601535` and entered corrected generation at `15:45:57 SAST`.
  Stage-B `b2` job `1220056` completed exact epoch-8 validation loss
  `0.4334614580005518` and remains in corrected generation, with its first
  two language segments starting at `15:27:13/15:36:20`. Their step-`5410`
  and step-`4328` F1 artifacts are absent, so no checkpoint or patience
  decision was made from loss. Tentative output-dependent artifact windows
  are b2 `16:00--16:15` and b1 `16:20--16:35`.
- New Stage-B `b3` job `1222348` remains healthy in epoch-1 training near
  step `442/8115`. Its immutable `694/694` source verification, pure
  `GatedDeltaNetForCausalLM` path, BF16 setting, fast kernel, and
  `4,323/10,760` train/validation coverage remain intact; its first F1
  artifact is tentatively due `16:45--17:05`, output-dependent.
- Targeted scans across `1219931/1220056/1222348` remain empty. All three run
  on `srvrocgpu010`, one A100-40GB each; no A100-80GB/L40S work. Quota is
  home `52.0%`, scratch `37.7%`; Kombuys untouched, Sheet E/F/G blank,
  trusted base `16/16`, NER seed-42 terminal-valid `4/11`, frozen winners
  `0/8`, held-out `0`, Mono not started.

## Stage-B b0 terminal; b1/b2 improve; b3 running — 15:26 SAST

- Stage-B `b0` job `1218997` completed cleanly `0:0` at `15:08:27 SAST`
  after the frozen patience rule stopped epoch 14. Its epoch-14 step-`7574`
  Tsn/Xho/Zul F1 is
  `0.6392303580972242/0.6494658119657621/0.6551630187993326`, mean
  `0.6479530629541063`. This is `0.0009183080430341661` below retained
  epoch-12 best `0.6488713709971404`; together with epoch 13 it is the second
  consecutive epoch without a threshold-`0.001` improvement, so stopping and
  restoring `checkpoint-6492` is correct. This makes Stage-B terminal-valid
  `1/8`; no cross-candidate rank is frozen.
- `b0` epoch-14 coverage is exactly `192`, `64` per language, with finite
  metrics and no literal empty raw strings. Debug/retained-state SHA-256 are
  `28a48acc83d695f9e42331ed18457fe3a47591f88bb095ea6e281f12fcb74f7b` and
  `f8319343dad3391f9c46bfce128c98f4b4bc49b157ac17b9f9639acd75e21fed`.
  Execution-manifest, final-adapter weights, and adapter-config SHA-256 are
  `f9f0c20312dfdb5c3cd61195ce7266039d3e99905e7a9e53940a9ff3765c14f2`,
  `2e0684c01c1fa57d937e0c7d3325517fa5ee7dbf240e7f8d694eaebdb4cd412c`,
  and `5c95d408943a37dc6d8384289d183e6ddd28dbeefc31c6887cfdc95d3119dec8`.
- Stage-B `b1` job `1219931` improved at epoch 9 step `4869`: Tsn/Xho/Zul
  `0.5320363164720642/0.5433654558932045/0.5636890741075783`, exact mean
  `0.5463636154909489`, matching retained trainer state. Its
  `0.00698991919554659` gain exceeds `0.001`, resetting patience; it resumed
  epoch 10 near step `5172/8115`. Debug/state SHA-256 are
  `c9a5caf0867761c676eaafcd0e5c489db49a0ef3595436a46954eb2e40fb29c7` and
  `f83e63e97aa30f7aff80d9b326fcf8efa2eedecc68190c5486f42e0b2967738f`.
- Stage-B `b2` job `1220056` improved at epoch 7 step `3787`: Tsn/Xho/Zul
  `0.40815318097586134/0.3588841417441068/0.4355644355643858`, exact mean
  `0.40086725276145135`, matching retained trainer state. Its
  `0.014615084675108858` gain exceeds `0.001`, resetting patience; it resumed
  epoch 8 near step `4328/8115`. Debug/state SHA-256 are
  `cc50ddc6546ca963a5ea4890a3d5f8bfd241008ff5a647cb45e78a6de9f58e67` and
  `959b4db7c822f02fe2d9dbeb5ec54da6683e7b772b4e749e693466a17b9affcc`.
  Both b1/b2 artifacts have exact `192` rows, `64` per language, finite
  metrics, and no literal empty raw strings.
- After verifying only two owned jobs, no A100-80GB/L40S work, immutable
  launcher/registry hashes, and no b3 artifact or duplicate, fixed Stage-B
  `b3` was submitted as job `1222348`. Slurm verified
  `nlpgroup/a100/nlpgroup`, one `gpu:ampere`, 24 hours, eight CPUs, one node,
  and `/home/lmbanr001/masters/sallm`. It started on `srvrocgpu010`, verified
  `694/694` immutable files, the pure `GatedDeltaNetForCausalLM` path, BF16,
  fast kernel, and `4,323` train/`10,760` validation rows; it reached step
  `10/8115` without a fault. Its frozen seed-42 configuration is LR
  `0.00007207193245743205`, rank/alpha `16/32`, dropout
  `0.06486568450927735`, warmup `0.04297354072332382`. Manifest/trial SHA-256
  are `43ecf3abaaa6aee85e3d68938029a51c31bed6f2adbc4580452f42e97fe772b0` and
  `abac391946f84d08a09e5a8ea76a39fe22e203fddc05c7c263ed8c350b1e664c`.
- Owned state is three running A100-40GB jobs (`1219931/1220056/1222348`),
  no A100-80GB/L40S. Quota is home `52.0%`, scratch `37.7%`; Kombuys remains
  untouched, Sheet E/F/G blank, trusted base `16/16`, seed-42 NER terminal
  valid `4/11` (`3/3` Stage-A plus `1/8` Stage-B), frozen winners `0/8`,
  held-out `0`, Mono not started.

## Three corrected callbacks active — 14:50 SAST

- Stage-B `b0` job `1218997` completed exact epoch-14 validation loss
  `0.40557230555878254`; `b1` job `1219931` completed exact epoch-9 loss
  `0.39639591940273583`; `b2` job `1220056` completed exact epoch-7 loss
  `0.44957441025063893`. All three are in corrected task-native generation.
  Their step-`7574`/`4869`/`3787` artifacts are absent, so no checkpoint,
  patience, pruning, or cross-candidate decision was made from loss.
- Targeted scans show no traceback, CUDA/OOM, NCCL, non-finite, manifest, or
  coverage fault. Jobs `1218997/1219931/1220056` remain running on
  `srvrocgpu010`, one A100-40GB `gpu:ampere` each, with approximately
  `7:37/13:04/15:18` wall time remaining at the initial check. `b2` is
  tentatively due around `14:55--15:05`; b0/b1 around `15:05--15:25`, all
  output-dependent.
- Owned state remains exactly three A100-40GB jobs with no A100-80GB/L40S
  work. Quota is home `52.0%`, scratch `37.6%`; Kombuys remains untouched,
  Sheet E/F/G blank, trusted base `16/16`, Stage-A NER terminal `3/3`, frozen
  winners `0/8`, held-out adapter evaluations `0`, and Mono not started.

## Stage-B b1 improves at epoch 8; b0 retains epoch 12 — 14:19 SAST

- Stage-B `b1` job `1219931` produced a valid epoch-8 step-`4328` artifact.
  Tsn/Xho/Zul span micro-F1 is
  `0.5355132672205342/0.5267403314916632/0.5558674901740096`; arithmetic mean
  `0.5393736962954022` matches trainer-state `0.5393736962954023` to floating
  precision. Its `0.011490621822435987` gain over epoch 7 exceeds frozen
  `0.001`, so `checkpoint-4328` is retained, patience reset, and epoch 9
  resumed near step `4759/8115`.
- Stage-B `b0` job `1218997` produced valid epoch-13 step-`7033` mean F1
  `0.6475630097288816` from Tsn/Xho/Zul
  `0.6366782006919915/0.6511131279696262/0.6548977005250269`. This is
  `0.0013083612682588397` below epoch 12, so `checkpoint-6492` and best F1
  `0.6488713709971404` correctly remain retained; the first non-improving
  epoch under patience `2` did not stop training, and epoch 14 resumed near
  step `7391/8115`.
- Both artifacts have exactly `192` rows, `64` per language, finite metrics,
  and no literal empty `raw_prediction` strings. Whitespace-only outputs that
  normalize empty are `58/192` in each artifact; this behavior also exists in
  earlier corrected artifacts and is scored by the frozen evaluator, not a
  coverage or prompt-contract fault. b0 debug/state SHA-256 values are
  `ce7cfa407a82240b2b1c6828cd6469e94ebfb3f1b6224fc8b9a2be62742c51b2` and
  `52ec29b52a6a052c1ec208071a358a178410fd1c37982c1f6827436cb6360079`;
  b1 values are
  `af2434eb806503d4615dc0a8d7b5780522928fc564b0045244b16b351abe19b3` and
  `caf0f02c83e903dac4c0ca27ac979816cda3ba607c0a91828e00340fef3fc1e7`.
- `b2` job `1220056` completed exact epoch-7 validation loss
  `0.44957441025063893` and entered corrected generation at `14:12` SAST;
  its step-`3787` artifact is absent, so no patience decision was made from
  loss. All three jobs remain healthy on `srvrocgpu010`, one A100-40GB each,
  with no targeted faults and no owned A100-80GB/L40S work. Quota is home
  `52.0%`, scratch `37.6%`; Kombuys remains untouched, Sheet E/F/G blank,
  trusted base `16/16`, frozen winners `0/8`, held-out `0`, Mono not started.

## Stage-B b0 improves at epoch 12; b1/b2 generation active — 13:18 SAST

- Stage-B `b0` job `1218997` produced a scientifically valid epoch-12
  step-`6492` artifact. Tsn/Xho/Zul span micro-F1 is
  `0.6443671394165945/0.6459540673963873/0.6562929061784397`; exact mean
  `0.6488713709971404` matches newly retained `checkpoint-6492` exactly.
  The gain over epoch 11 is `0.0049355574309944`, above frozen `0.001`, so
  resetting patience and continuing is correct. This remains within-candidate
  evidence; no candidate is frozen, pruned, or ranked.
- Independent checks found exactly `192` rows, `64` per language, and all raw
  predictions nonempty. Tsn/Xho/Zul unique raw-output counts are `39/54/36`.
  Debug and trainer-state SHA-256 values are
  `88a43ed1456e8b680c7f218a1f1dc813b3519ca0c0931f66289f9505031be3ff` and
  `f8319343dad3391f9c46bfce128c98f4b4bc49b157ac17b9f9639acd75e21fed`.
  The job resumed healthy epoch-13 training.
- `b1` job `1219931` completed exact epoch-8 validation loss
  `0.39547758988731413`; `b2` job `1220056` completed exact epoch-6 loss
  `0.46967894827123025`. Both are in corrected generation without their next
  F1 artifacts, so no checkpoint or patience decision was made from loss.
  Targeted fault scans remain empty.
- Owned state remains three running A100-40GB jobs, exactly at the cap, with
  no A100-80GB/L40S work. Quota is home `52.0%`, scratch `37.6%`; Kombuys is
  read-only/untouched, Sheet E/F/G remain blank, trusted base `16/16`, frozen
  winners `0/8`, held-out `0`, and Mono not started.

## Stage-B b1 improves at epoch 7; b0 generation active — 12:48 SAST

- Stage-B `b1` job `1219931` produced a scientifically valid epoch-7
  step-`3787` artifact. Tsn/Xho/Zul span micro-F1 is
  `0.5178216509870972/0.5036417001847531/0.5621858722470486`; exact mean
  `0.5278830744729662` matches newly retained `checkpoint-3787` exactly.
  The gain over epoch 6 is `0.0241741561402960`, above frozen `0.001`, so
  resetting patience and continuing is correct. This remains within-candidate
  evidence; no candidate is frozen, pruned, or ranked.
- Independent checks found exactly `192` rows, `64` per language, and all raw
  predictions nonempty. Tsn/Xho/Zul unique raw-output counts are `40/55/38`.
  Debug and trainer-state SHA-256 values are
  `712c9e3a54d0287fe6a91338f2a65c0c7e252d8c0d1da6189bd1f68fb6bd92e1` and
  `d20882ca92fecc5b7d8e895e50be071d97522c04633126b241990591266f9f06`.
  The job resumed healthy epoch-8 training near step `3919/8115`.
- `b0` job `1218997` remains in epoch-12 corrected generation after exact
  loss `0.3988653133349791`; its step-6492 F1 artifact does not exist yet, so
  no checkpoint or patience decision was made from loss. `b2` job `1220056`
  remains healthy in epoch-6 training near step `3223/8115`. Targeted fault
  scans remain empty.
- Owned state remains three running A100-40GB jobs, exactly at the cap, with
  no A100-80GB/L40S work. Quota is home `52.0%`, scratch `37.6%`; Kombuys is
  read-only/untouched, Sheet E/F/G remain blank, trusted base `16/16`, frozen
  winners `0/8`, held-out `0`, and Mono not started.

## Stage-B b2 improves at epoch 5; b0/b1 generation active — 12:23 SAST

- Stage-B `b2` job `1220056` produced a scientifically valid epoch-5
  step-`2705` artifact. Tsn/Xho/Zul span micro-F1 is
  `0.3512463985969688/0.33180474880255884/0.36860655033090517`; exact mean
  `0.3505525659101443` matches newly retained `checkpoint-2705` exactly.
  The gain over epoch 4 is `0.07628408463817873`, above frozen `0.001`, so
  resetting patience and continuing is correct. This remains within-candidate
  evidence; no candidate is frozen, pruned, or ranked.
- Independent checks found exactly `192` rows, `64` per language, and all raw
  predictions nonempty. Tsn/Xho/Zul unique raw-output counts are `38/57/35`.
  Debug and trainer-state SHA-256 values are
  `74aeda3e5472d018b551b514a5fc941dad7362854d956f2dba76ca915a95f475` and
  `a15c9c10198c2bd915c0ded0bd1e13182e1c82863ec2523deaef8de6da73678d`.
  The job resumed healthy epoch-6 training.
- `b0` job `1218997` completed exact epoch-12 validation loss
  `0.3988653133349791`; `b1` job `1219931` completed exact epoch-7 loss
  `0.393537137056372`. Both are in corrected generation without their next
  F1 artifacts, so no checkpoint or patience decision was made from loss.
  Targeted fault scans across all three jobs remain empty.
- Owned state remains three running A100-40GB jobs, exactly at the cap, with
  no A100-80GB/L40S work. Quota is home `52.0%`, scratch `37.6%`; Kombuys is
  read-only/untouched, Sheet E/F/G remain blank, trusted base `16/16`, frozen
  winners `0/8`, held-out `0`, and Mono not started.

## Stage-B b0 epoch 11 and b1 epoch 6 improve — 11:48 SAST

- Stage-B `b0` job `1218997` produced a scientifically valid epoch-11
  step-`5951` artifact. Tsn/Xho/Zul span micro-F1 is
  `0.6350053361792457/0.6418292292932025/0.6549728752259898`; exact mean
  `0.643935813566146` matches newly retained `checkpoint-5951` exactly.
  The gain over epoch 10 is `0.005867249566207`, above frozen `0.001`, so
  resetting patience and continuing is correct.
- Its artifact has exactly `192` rows, `64` per language, and all raw
  predictions nonempty; Tsn/Xho/Zul unique raw-output counts are `39/54/36`.
  Debug and trainer-state SHA-256 values are
  `5f81d7aedfa13673da7b2ffebe31da5d1dae75bda20f6548b986f6ddccc8ffbd` and
  `1fea1e7289c2d24058d67680fe53a44e598e3f41db9343e12cbeb09b19954aa7`.
  The job resumed healthy epoch-12 training near step `6111/8115`.
- Stage-B `b1` job `1219931` produced a scientifically valid epoch-6
  step-`3246` artifact. Tsn/Xho/Zul span micro-F1 is
  `0.49645390070917006/0.4963758803268919/0.5182969739619486`; exact mean
  `0.5037089183326702` matches newly retained `checkpoint-3246` exactly.
  The gain over epoch 5 is `0.0234528348096561`, above frozen `0.001`, so
  resetting patience and continuing is correct.
- Its artifact also has exactly `192` rows, `64` per language, and all raw
  predictions nonempty; Tsn/Xho/Zul unique raw-output counts are `40/55/37`.
  Debug and trainer-state SHA-256 values are
  `7c8aed471c679c26435c2a563db918cf98a791c717263349bdd9cfc4c4f2c5f1` and
  `338f58227bbab51fa6d615694bc77fec9dd06de76f3ab5d705413d383e5bb0dc`.
  The job resumed healthy epoch-7 training near step `3645/8115`.
- Both remain within-candidate evidence; no candidate is frozen, pruned, or
  ranked. `b2` job `1220056` completed exact epoch-5 validation loss
  `0.5249240350546004` and is in corrected generation without its step-2705
  F1 artifact, so no decision was made from loss. Targeted fault scans remain
  empty.
- Owned state remains three running A100-40GB jobs, exactly at the cap, with
  no A100-80GB/L40S work. Quota is home `52.0%`, scratch `37.6%`; Kombuys is
  read-only/untouched, Sheet E/F/G remain blank, trusted base `16/16`, frozen
  winners `0/8`, held-out `0`, and Mono not started.

## Stage-B b2 improves at epoch 4; b0/b1 generation active — 11:18 SAST

- Stage-B `b2` job `1220056` produced a scientifically valid epoch-4
  step-`2164` artifact. Tsn/Xho/Zul span micro-F1 is
  `0.26803909336655674/0.25188246097332045/0.3028838894760195`; exact mean
  `0.27426848127196557` matches newly retained `checkpoint-2164` exactly.
  The gain over epoch 3 is `0.10095193978780567`, above frozen `0.001`, so
  resetting patience and continuing is correct. This remains within-candidate
  evidence; no candidate is frozen, pruned, or ranked.
- Independent checks found exactly `192` rows, `64` per language, and all raw
  predictions nonempty. Tsn/Xho/Zul unique raw-output counts are `42/58/39`.
  Debug and trainer-state SHA-256 values are
  `e08e539c3e5213f03f233f322ea131fa40c9807e40e8404b4861f72720af6c4b` and
  `8a2f35e46ef99244bb3064c2db954298e192d5490c84646e5f5c5f9f97ad9445`.
  The job resumed healthy epoch-5 training near step `2380/8115`.
- `b0` job `1218997` completed exact epoch-11 validation loss
  `0.39432813226068775`; `b1` job `1219931` completed exact epoch-6 loss
  `0.3903183947708527`. Both are in corrected generation without their next
  F1 artifacts, so no checkpoint or patience decision was made from loss.
  Targeted fault scans across all three jobs remain empty.
- Owned state remains three running A100-40GB jobs, exactly at the cap, with
  no A100-80GB/L40S work. Quota is home `52.0%`, scratch `37.6%`; Kombuys is
  read-only/untouched, Sheet E/F/G remain blank, trusted base `16/16`, frozen
  winners `0/8`, held-out `0`, and Mono not started.

## Stage-B b0 improves at epoch 10; b1/b2 generation active — 10:48 SAST

- Stage-B `b0` job `1218997` produced a scientifically valid epoch-10
  step-`5410` artifact. Tsn/Xho/Zul span micro-F1 is
  `0.627621685793381/0.6371529638760356/0.6494310423304006`; exact mean
  `0.638068563999939` matches newly retained `checkpoint-5410` exactly.
  The gain over epoch 9 is `0.0089627700204281`, above frozen `0.001`, so
  resetting patience and continuing is correct. This remains within-candidate
  evidence; no candidate is frozen, pruned, or ranked.
- Independent checks found exactly `192` rows, `64` per language, and all raw
  predictions nonempty. Tsn/Xho/Zul unique raw-output counts are `39/54/36`.
  Debug and trainer-state SHA-256 values are
  `665ce314e4ac451621d89aba78e03fc7a0de25f01bf4507cce2088e93d5a77e1` and
  `107ab9affbc3d3c29d9883b5af15ebb2768f988b2717e37d6e7ee890894ea3a6`.
  The job resumed healthy epoch-11 training near step `5757/8115`.
- `b1` job `1219931` is in epoch-6 corrected generation. `b2` job `1220056`
  completed exact epoch-4 validation loss `0.7704469021368204` and is also in
  corrected generation. Neither next F1 artifact exists yet, so no checkpoint
  or patience decision was made from loss. Targeted fault scans remain empty.
- Owned state remains three running A100-40GB jobs, exactly at the cap, with
  no A100-80GB/L40S work. Quota is home `52.0%`, scratch `37.6%`; Kombuys is
  read-only/untouched, Sheet E/F/G remain blank, trusted base `16/16`, frozen
  winners `0/8`, held-out `0`, and Mono not started.

## Stage-B b1 epoch 5 and b2 epoch 3 improve — 10:18 SAST

- Stage-B `b1` job `1219931` produced a scientifically valid epoch-5
  step-`2705` artifact. Tsn/Xho/Zul span micro-F1 is
  `0.47900346731727383/0.4651410324640268/0.4966237507877415`; exact mean
  `0.4802560835230141` matches newly retained `checkpoint-2705` exactly.
  The gain over epoch 4 is `0.06356442360022315`, above frozen `0.001`, so
  resetting patience and continuing is correct.
- Its artifact has exactly `192` rows, `64` per language, and all raw
  predictions nonempty; Tsn/Xho/Zul unique raw-output counts are `42/56/37`.
  Debug and trainer-state SHA-256 values are
  `2f0c43262fc184b3fc3ba1997dbb12953dbae75144376001f3dd1b89f86efd29` and
  `01a5c15f6459059af99ec2e1928671a6b97eb1c024610440103e017635cc2bd6`.
  The job resumed healthy epoch-6 training near step `2819/8115`.
- Stage-B `b2` job `1220056` produced a scientifically valid epoch-3
  step-`1623` artifact. Tsn/Xho/Zul span micro-F1 is
  `0.16267858708033106/0.16801662355281158/0.18925441381933708`; exact mean
  `0.1733165414841599` matches newly retained `checkpoint-1623` exactly.
  The gain over epoch 2 is `0.0094945849198557`, above frozen `0.001`, so
  resetting patience and continuing is correct.
- Its artifact also has exactly `192` rows, `64` per language, and all raw
  predictions nonempty; Tsn/Xho/Zul unique raw-output counts are `60/60/54`.
  Debug and trainer-state SHA-256 values are
  `f02f04d0cda0fc305f9f84b55b5223cc74d555f43f95997192fa186787c40787` and
  `4c54f28493b1018a761f0c83a98c433d2dbe2f132c32f0252caaf0f97024072e`.
  The job resumed healthy epoch-4 training near step `2154/8115`.
- Both remain within-candidate evidence; no candidate is frozen, pruned, or
  ranked. `b0` job `1218997` completed exact epoch-10 validation loss
  `0.38368800280262544` and entered corrected generation without its
  step-5410 F1 artifact, so no decision was made from loss. Targeted fault
  scans across all three jobs remain empty.
- Owned state remains three running A100-40GB jobs, exactly at the cap, with
  no A100-80GB/L40S work. Quota is home `52.0%`, scratch `37.6%`; Kombuys is
  read-only/untouched, Sheet E/F/G remain blank, trusted base `16/16`, frozen
  winners `0/8`, held-out `0`, and Mono not started.

## Stage-B b0 improves at epoch 9; b1/b2 generation active — 09:48 SAST

- Stage-B `b0` job `1218997` produced a scientifically valid epoch-9
  step-`4869` artifact. Tsn/Xho/Zul span micro-F1 is
  `0.6252717391303849/0.6267364821542567/0.635309160653891`; exact mean
  `0.6291057939795109` matches newly retained `checkpoint-4869` exactly.
  The gain over epoch 8 is `0.0087232233184932`, above frozen `0.001`, so
  resetting patience and continuing is correct. This remains within-candidate
  evidence; no candidate is frozen, pruned, or ranked.
- Independent checks found exactly `192` rows, `64` per language, and all raw
  predictions nonempty. Tsn/Xho/Zul unique raw-output counts are `39/54/36`.
  Debug and trainer-state SHA-256 values are
  `10010a538a76647ab3c122e88f75d534331c153b60ad2d9896591074d82dbfd6` and
  `27b87619ac1fd046737167dc074c3240fbc628b03a47b3c3dad5b865ecdb7afa`.
- `b1` job `1219931` completed exact epoch-5 validation loss
  `0.39540263463130226` and remains in corrected generation. `b2` job
  `1220056` remains in epoch-3 corrected generation after exact loss
  `1.1244368429077602`. Neither next F1 artifact exists yet, so no checkpoint
  or patience decision was made from loss. Targeted fault scans remain empty.
- Owned state remains three running A100-40GB jobs, exactly at the cap, with
  no A100-80GB/L40S work. Quota is home `52.0%`, scratch `37.6%`; Kombuys is
  read-only/untouched, Sheet E/F/G remain blank, trusted base `16/16`, frozen
  winners `0/8`, held-out `0`, and Mono not started.

## Stage-B b1 improves at epoch 4; b0/b2 generation active — 09:18 SAST

- Stage-B `b1` job `1219931` produced a scientifically valid epoch-4
  step-`2164` artifact. Tsn/Xho/Zul span micro-F1 is
  `0.40733008582690516/0.4091173054587189/0.43362758848274874`; exact mean
  `0.41669165992279095` matches newly retained `checkpoint-2164` exactly.
  The gain over epoch 3 is `0.09695622609201065`, above frozen `0.001`, so
  resetting patience and continuing is correct. This remains within-candidate
  evidence; no candidate is frozen, pruned, or ranked.
- Independent checks found exactly `192` rows, `64` per language, and all raw
  predictions nonempty. Tsn/Xho/Zul unique raw-output counts are `40/56/36`.
  Debug and trainer-state SHA-256 values are
  `259253bb1a1446983addd232d722d39fd4c541c5e6fe1c84cbeeab2695ad40bf` and
  `4c3e60e2dfe5f8eedb998dc1e6db6bd6ed1a55d696ccb128671a911034c0cbb4`.
  The job resumed healthy epoch-5 training near step `2566/8115`.
- `b0` job `1218997` remains in epoch-9 corrected generation after exact loss
  `0.3812508735514928`. `b2` job `1220056` completed exact epoch-3 validation
  loss `1.1244368429077602` and entered corrected generation. Neither next F1
  artifact exists yet, so no checkpoint or patience decision was made from
  loss. Targeted fault scans across all three jobs remain empty.
- Owned state remains three running A100-40GB jobs, exactly at the cap, with
  no A100-80GB/L40S work. Quota is home `52.0%`, scratch `37.6%`; Kombuys is
  read-only/untouched, Sheet E/F/G remain blank, trusted base `16/16`, frozen
  winners `0/8`, held-out `0`, and Mono not started.

## Stage-B b2 improves at epoch 2; b0/b1 generation active — 08:53 SAST

- Stage-B `b2` job `1220056` produced a scientifically valid epoch-2
  step-`1082` artifact. Tsn/Xho/Zul span micro-F1 is
  `0.17885024840307345/0.13693256118059663/0.17568306010924253`; exact mean
  `0.1638219565643042` matches newly retained `checkpoint-1082` exactly.
  The gain over epoch 1 is `0.11986505071325975`, above frozen `0.001`, so
  resetting patience and continuing is correct. This remains within-candidate
  evidence; no candidate is frozen, pruned, or ranked.
- Independent checks found exactly `192` rows, `64` per language, and all raw
  predictions nonempty. Tsn/Xho/Zul unique raw-output counts are `55/54/50`.
  Debug and trainer-state SHA-256 values are
  `9a12280888d3356569dd88cd58248a88088e9109a6e256760982a6b5cc5ad69d` and
  `b69bb8bb196f6854e43647e3922b34644d2ddad20d7ec3c54fafd633774ae681`.
  The job resumed healthy epoch-3 training near step `1421/8115`.
- `b0` job `1218997` completed exact epoch-9 validation loss
  `0.3812508735514928`; `b1` job `1219931` completed exact epoch-4 validation
  loss `0.438888345597845`. Both are in corrected generation without their
  next task-native F1 artifacts, so no checkpoint or patience decision was
  made from loss. Targeted fault scans across all three jobs remain empty.
- Owned state remains three running A100-40GB jobs, exactly at the cap, with
  no A100-80GB/L40S work. Quota is home `52.0%`, scratch `37.6%`; Kombuys is
  read-only/untouched, Sheet E/F/G remain blank, trusted base `16/16`, frozen
  winners `0/8`, held-out `0`, and Mono not started.

## Stage-B b0 epoch 8 and b1 epoch 3 improve — 08:11 SAST

- Stage-B `b0` job `1218997` produced a scientifically valid epoch-8
  step-`4328` artifact. Tsn/Xho/Zul span micro-F1 is
  `0.6221399285808257/0.6079748163693102/0.631032967032917`; exact mean
  `0.6203825706610177` matches newly retained `checkpoint-4328` exactly.
  The gain over epoch 7 is `0.0047530834308320`, above frozen `0.001`, so
  resetting patience and continuing is correct.
- Its artifact has exactly `192` rows, `64` per language, and all raw
  predictions nonempty; Tsn/Xho/Zul unique raw-output counts are `41/54/38`.
  Debug and trainer-state SHA-256 values are
  `46de0d74fedcbc5458383e2cb86d73b8bf97e275ec6381a98cc59e8d330ac3c5` and
  `16fb1dfc7520c123a2c5c76ac5ca4f66babcbf2dd1cb6fa142e13cb1ceb1ab2a`.
  The job resumed epoch-9 training.
- Stage-B `b1` job `1219931` produced a scientifically valid epoch-3
  step-`1623` artifact. Tsn/Xho/Zul span micro-F1 is
  `0.31807711567346336/0.30517226784046586/0.3359569179784117`; exact mean
  `0.3197354338307803` matches newly retained `checkpoint-1623` exactly.
  The gain over epoch 2 is `0.02923439043360694`, above frozen `0.001`, so
  resetting patience and continuing is correct.
- Its artifact also has exactly `192` rows, `64` per language, and all raw
  predictions nonempty; Tsn/Xho/Zul unique raw-output counts are `43/58/37`.
  Debug and trainer-state SHA-256 values are
  `29bfb39c414a631d59923b347de94a7155dada418c59fbeb4411b4aea0ec2141` and
  `510e461be0f842f432b9c6b4495dc54ca025f00800ff1cb4d39c775d5748adf1`.
  The job resumed epoch-4 training.
- Both remain within-candidate evidence; no candidate is frozen or ranked.
  `b2` job `1220056` completed exact epoch-2 validation loss
  `1.3991323931952835` and is in corrected generation without its step-1082
  F1 artifact, so no decision was made from loss. Targeted fault scans are
  empty.
- Owned state remains three running A100-40GB jobs, exactly at the cap, with
  no A100-80GB/L40S work. Quota is home `52.0%`, scratch `37.6%`; Kombuys is
  read-only/untouched, Sheet E/F/G remain blank, trusted base `16/16`, frozen
  winners `0/8`, held-out `0`, and Mono not started.

## Stage-B b2 epoch 1 reconciles; b0/b1 generation active — 07:41 SAST

- Stage-B `b2` job `1220056` produced a scientifically valid epoch-1
  step-`541` artifact. Tsn/Xho/Zul span micro-F1 is
  `0.05142857142852147/0.028593442342240153/0.05184870378237173`; exact mean
  `0.04395690585104445` matches retained `checkpoint-541` exactly. This is
  only an initial within-candidate result; the frozen protocol forbids
  low-fidelity pruning or cross-candidate selection.
- Independent checks found exactly `192` rows, `64` per language, and all raw
  predictions nonempty. Tsn/Xho/Zul unique raw-output counts are `31/8/13`.
  Debug and trainer-state SHA-256 values are
  `6347350aba222257ef5550d74397321a3972e7484b0fbd494e890529d98b36ad` and
  `ac30c40b05f14b851610e0a372824c8c319be9d40862e999b6050407ea29c03e`.
  The job resumed healthy epoch-2 training near step `931/8115`.
- `b0` job `1218997` completed exact epoch-8 validation loss
  `0.37637544653229554` and is in corrected generation without its step-4328
  F1 artifact. `b1` job `1219931` remains in epoch-3 corrected generation
  without its step-1623 F1 artifact. No decisions were made from loss;
  targeted fault scans remain empty.
- Owned state remains three running A100-40GB jobs, exactly at the cap, with
  no A100-80GB/L40S work. Quota is home `52.0%`, scratch `37.6%`; Kombuys is
  read-only/untouched, Sheet E/F/G remain blank, trusted base `16/16`, frozen
  winners `0/8`, held-out `0`, and Mono not started.

## Stage-B b0 improves at epoch 7; b1/b2 generation active — 07:11 SAST

- Stage-B `b0` job `1218997` produced a scientifically valid epoch-7
  step-`3787` artifact. Tsn/Xho/Zul span micro-F1 is
  `0.6145573770491303/0.609097594285064/0.6232334903563628`; exact mean
  `0.6156294872301857` matches newly retained `checkpoint-3787` exactly.
  The gain over epoch 6 is `0.0171293534161445`, above the frozen `0.001`
  threshold, so resetting patience and continuing is correct. This remains
  within-candidate evidence; no candidate is frozen or pruned.
- Independent checks found exactly `192` rows, `64` per language, and all raw
  predictions nonempty. Tsn/Xho/Zul unique raw-output counts are `40/57/38`.
  Debug and trainer-state SHA-256 values are
  `694fa03f272247eee1c5035fccbc8b41b35244873ecf2369a0feb48397c78140` and
  `4b622bfb8b8f6b3f1eb21bcb6dadccae7b9d133631319f29a4d35a1d565e1729`.
  The job resumed healthy epoch-8 training near step `4010/8115`.
- `b1` job `1219931` completed exact epoch-3 validation loss
  `0.723133299873664` and entered corrected generation without its step-1623
  F1 artifact yet. `b2` job `1220056` completed exact epoch-1 validation loss
  `1.9991583501539265` and also entered corrected generation without its
  step-541 F1 artifact. No checkpoint or patience decision was made from
  either validation loss. Targeted fault scans remain empty.
- Owned state remains three running A100-40GB jobs, exactly at the cap, with
  no A100-80GB/L40S work. Quota is home `52.0%`, scratch `37.6%`; Kombuys is
  read-only/untouched, Sheet E/F/G remain blank, trusted base `16/16`, frozen
  winners `0/8`, held-out `0`, and Mono not started.

## Stage-B b1 improves at epoch 2 — 06:41 SAST

- Stage-B `b1` job `1219931` produced a scientifically valid epoch-2
  step-`1082` artifact. Tsn/Xho/Zul span micro-F1 is
  `0.30320395936436784/0.2633148262388972/0.3049843445882552`; exact mean
  `0.29050104339717336` matches newly retained `checkpoint-1082` exactly.
  The gain over epoch 1 is `0.11547740388034549`, above the frozen `0.001`
  threshold, so resetting patience and continuing is correct. This remains
  within-candidate evidence; no candidate is frozen or pruned.
- Independent checks found exactly `192` rows, `64` per language, and all raw
  predictions nonempty. Tsn/Xho/Zul unique raw-output counts are `39/57/35`.
  Debug and trainer-state SHA-256 values are
  `e275e4e5149d56d3b1c9134004e58d751f782511ce2c519988fac084ae182135` and
  `fd2fb9a24ed09d40552c41eabe84916a889095db12eb32ebd5f1085ef6210475`.
  The job resumed healthy epoch-3 training near step `1389/8115`.
- `b0` job `1218997` completed exact epoch-7 validation loss
  `0.3684530038372735` and is in corrected generation without its step-3787
  F1 artifact yet. `b2` job `1220056` remains healthy in epoch-1 training.
  Targeted scans across all three jobs are empty.
- Owned state remains three running A100-40GB jobs, exactly at the cap, with
  no A100-80GB/L40S work. Quota is home `52.0%`, scratch `37.6%`; Kombuys is
  read-only/untouched, Sheet E/F/G remain blank, trusted base `16/16`, frozen
  winners `0/8`, held-out `0`, and Mono not started.

## Stage-B b0 improves at epoch 6; b2 starts cleanly — 06:11 SAST

- Stage-B `b0` job `1218997` produced a scientifically valid epoch-6
  step-`3246` artifact. Tsn/Xho/Zul span micro-F1 is
  `0.592918131592115/0.5861427094104983/0.6164395604395104`; exact mean
  `0.5985001338140412` matches newly retained `checkpoint-3246` exactly.
  The gain over epoch 5 is `0.0155290351307471`, above the frozen `0.001`
  threshold, so resetting patience and continuing is correct. This remains a
  within-candidate result; no candidate is frozen or pruned.
- Independent checks found exactly `192` rows, `64` per language, and all raw
  predictions nonempty. Tsn/Xho/Zul unique raw-output counts are `40/54/37`.
  Debug and trainer-state SHA-256 values are
  `8e2f4ae7890c6f8176c0b24163a9346be8ba6a7eb105fd83c23d8a662034c170` and
  `be46397071404c4ee0e85265ff2a080e79873fc3755e42c9a405ce98a3697d47`.
  The job resumed healthy epoch-7 training.
- Fixed Stage-B `b2` job `1220056` started at `06:06:29 SAST` on
  `srvrocgpu010` A100-40GB and verified all `694` immutable files. Its
  execution-manifest and trial SHA-256 values are
  `00a0a053eb475721a72b2046e7a05b7f7b5c1e01b5d9d0627b0b1ec4a5b54abe` and
  `e97b1ce27840b5d423442aa7cc548b1b16544fad72e2bb7d7989703f2492626e`.
  The hashed trial confirms validation-only seed 42, LR
  `0.00003617950373518279`, rank/alpha `8/16`, dropout
  `0.019731278717517856`, warmup `0.09582568347454071`, and registry SHA-256
  `8fdd6ea5f192a1d2ef2267353162b9a640ccd09ae0d889cb5a999645198bb726`.
  It loaded the canonical pure GDN in BF16 with fast path available, prepared
  `4,323` train/`10,760` validation rows, and was healthy near step `60/8115`.
- `b1` job `1219931` completed exact epoch-2 validation loss
  `1.0695723636442844` and entered corrected generation without its step-1082
  F1 artifact yet. Targeted scans across all three jobs are empty. Owned state
  is now three running A100-40GB jobs, exactly at the cap, with no
  A100-80GB/L40S work. Quota is home `52.0%`, scratch `37.6%`; Kombuys is
  read-only/untouched, Sheet E/F/G remain blank, trusted base `16/16`, frozen
  winners `0/8`, held-out `0`, and Mono not started.

## Stage-B b0 epoch-6 generation active — 05:41 SAST

- Stage-B `b0` job `1218997` completed exact `10,760`-row epoch-6 validation
  loss `0.35770683430384526` and entered its final corrected-generation
  segment. No step-`3246` F1 artifact exists yet, so no checkpoint, patience,
  pruning, or cross-candidate decision was made from validation loss. Retained
  within-candidate best remains epoch-5 F1 `0.5829710986832941`; the next F1
  artifact is tentatively due around `05:45--05:55 SAST`, output-dependent.
- `b1` job `1219931` remains healthy after its reconciled epoch-1 F1 and has
  entered epoch-2 validation preparation. Fixed `b2` job `1220056` remains
  `PENDING (Priority)`, dynamically projected for `12:27 SAST`. Targeted
  scans show no traceback, CUDA/OOM, NCCL, non-finite, manifest, or coverage
  fault.
- Owned state remains two running plus one pending A100-40GB jobs, no
  A100-80GB/L40S; quota home `52.0%`, scratch `37.6%`. Kombuys remains
  read-only/untouched, Sheet `GDN Results` E/F/G remain blank, trusted base
  `16/16`, frozen Multilingual winners `0/8`, held-out adapter evaluations
  `0`, and Mono not started.

## Stage-B b1 epoch 1 reconciles cleanly — 05:11 SAST

- Enhanced Stage-B `b1` job `1219931` produced a scientifically valid
  epoch-1 step-`541` artifact. Tsn/Xho/Zul span micro-F1 is
  `0.1639749192475316/0.15426869006605212/0.20682730923689988`; exact mean
  `0.17502363951682787` matches retained `checkpoint-541` exactly. This is
  only an initial within-candidate result: the frozen protocol forbids
  low-fidelity pruning or cross-candidate selection before all 11 seed-42
  candidates terminate validly.
- Independent checks found exactly `192` rows, `64` per language, and all raw
  predictions nonempty. Tsn/Xho/Zul unique raw-output counts are `55/59/50`.
  Debug and trainer-state SHA-256 values are
  `fc851a41c4092aaefdc2c31527c087a50a17102a6fcb16b70b543b8804ec1b31` and
  `751efbd2251309ec11bc70ec03c05f40ba5a250f5d2b91fb9978cc5227e9b96f`.
  The job resumed healthy epoch-2 training.
- `b0` job `1218997` remains healthy in epoch-6 validation after retaining
  epoch-5 F1 `0.5829710986832941`. Fixed `b2` job `1220056` remains
  `PENDING (Priority)` with a dynamic Slurm projection of `12:27 SAST`.
  Targeted fault scans are empty.
- Owned state remains two running plus one pending A100-40GB jobs, no
  A100-80GB/L40S; quota home `52.0%`, scratch `37.6%`. Kombuys remains
  read-only/untouched, Sheet `GDN Results` E/F/G remain blank, trusted base
  `16/16`, frozen Multilingual winners `0/8`, held-out adapter evaluations
  `0`, and Mono not started.

## Stage-B b0 improves at epoch 5; b1 epoch-1 generation active — 04:42 SAST

- Enhanced Stage-B `b0` job `1218997` produced a scientifically valid
  epoch-5 step-`2705` artifact. Tsn/Xho/Zul span micro-F1 is
  `0.5850447604001606/0.5677351550183422/0.5961333806313797`; exact mean
  `0.5829710986832941` matches newly retained `checkpoint-2705` exactly.
  The gain over epoch 4 is `0.0275625934877679`, above the frozen `0.001`
  threshold, so resetting patience and continuing is correct. This remains a
  within-candidate result; no candidate is frozen or pruned.
- Independent checks found exactly `192` rows, `64` per language, and all raw
  predictions nonempty. Tsn/Xho/Zul unique raw-output counts are `41/56/39`.
  Debug and trainer-state SHA-256 values are
  `de189383053c69ce455d1572ede0147364a8decc76894fbb840c217539ca665c` and
  `50f76630c6a1c9f786c3101bc33518b8d337d86f6a306b616bdabaf0423f263e`.
  The job resumed healthy epoch-6 training.
- Stage-B `b1` job `1219931` completed exact `10,760`-row epoch-1 validation
  loss `1.4194149882376859` and entered corrected generation; its first F1
  artifact does not exist yet, so no checkpoint or patience decision has been
  made. Fixed `b2` job `1220056` remains `PENDING (Priority)` with a dynamic
  Slurm projection of `12:27 SAST`.
- Targeted fault scans are empty. Owned state remains two running plus one
  pending A100-40GB jobs, no A100-80GB/L40S; quota home `52.0%`, scratch
  `37.5%`. Kombuys remains read-only/untouched, Sheet `GDN Results` E/F/G
  remain blank, trusted base `16/16`, frozen Multilingual winners `0/8`,
  held-out adapter evaluations `0`, and Mono not started.

## Stage-A complete; b1 running and b2 submitted — 04:12 SAST

- Stage-A NER LR `1.5e-4` job `1218345` completed cleanly `0:0` at
  `03:52:01 SAST` after the frozen patience condition was satisfied at epoch
  14. Its scientifically valid epoch-14 step-`7574` F1 is
  `0.678259542642304` (Tsn/Xho/Zul
  `0.6660465116278571/0.6835103529663702/0.6852217633326847`). This is
  `0.0012630339910531` below retained epoch-12 best
  `0.6795225766333571`, the second consecutive non-improving epoch after the
  epoch-12 reset. The run correctly stopped and restored checkpoint `6492`.
- Independent epoch-14 checks found exactly `192` rows, `64` per language,
  and all raw predictions nonempty. Tsn/Xho/Zul unique raw-output counts are
  `40/55/36`. The debug artifact SHA-256 is
  `7548e8d0535c7192252293b3363a9bddc4bf475285fbdf762e7864513d05c8ca`.
  Retained trainer-state, execution-manifest, final-adapter weight, and
  final-adapter config SHA-256 values are
  `c0e54e9467e217855c8648926fc35fc876ae4bb481389d6d995cfac0db73fb66`,
  `3181df56896df904fc81808eaaff9cdbd75d410d93b59343f3dd20f2e08ebeb1`,
  `f82ad9baa7faa4f99b048abe03fa7ec9f4e2a0c270acaf1afa29118ad812e337`,
  and `d085fe1a35e196c55094c57f58491a088e2b71b27d396756f8c37e1e1fedc826`.
- All three Stage-A candidates are now terminal and scientifically valid:
  `a0`/job `1218343` F1 `0.500502987595103`, `a1`/job `1218344` F1
  `0.6173130857642662`, and `a2`/job `1218345` F1
  `0.6795225766333571`. Candidate `a2` is only the provisional Stage-A
  leader; no NER recipe is frozen until all eight Stage-B candidates terminate
  validly and the preregistered seed-confirmation stage is complete.
- The released A100-40GB started fixed Stage-B `b1` job `1219931` at
  `03:52:01 SAST`. It verified all `694` immutable files and passed pure-GDN,
  BF16, fast-kernel, `4,323` train/`10,760` validation-row, and validation-only
  startup gates. Execution-manifest and trial SHA-256 values are
  `f67827a869206788a812485fc2e0707a76d112eb8dc936cf1818b1bbf4b8dd22` and
  `cc5b4391f138143bfc4f5678addd2e3d29f969fe0182c2812fc6b48634b8b88e`.
  Its frozen seed-42 configuration is LR `0.000030329114402234482`, rank/alpha
  `32/64`, dropout `0.08691746592521668`, warmup
  `0.029702768474817273`; it was healthy near step `350/8115`. Its first F1
  artifact is tentatively due around `04:50--05:10`, output-dependent.
- After verifying exactly two owned running jobs, no A100-80GB/L40S work, no
  existing `b2` artifact, and the immutable launcher/registry hashes, fixed
  Stage-B `b2` was submitted as job `1220056`. Its frozen seed-42
  configuration is LR `0.00003617950373518279`, rank/alpha `8/16`, dropout
  `0.019731278717517856`, and warmup `0.09582568347454071`. Slurm verified
  `nlpgroup/a100/nlpgroup`, one `gpu:ampere`, 24 hours, eight CPUs, one node,
  and `/home/lmbanr001/masters/sallm`; it is `PENDING (Priority)` with no
  output or metric.
- Stage-B `b0` job `1218997` remains healthy in epoch-5 training with retained
  within-candidate best F1 `0.5554085051955262`; its next artifact is
  tentatively due around `05:00--05:20`. Owned state is two running plus one
  pending A100-40GB jobs, exactly at the three-job cap, with no A100-80GB or
  L40S work. Quota is home `52.0%`, scratch `37.5%`. Kombuys remains
  read-only/untouched, Sheet `GDN Results` E/F/G remain blank, trusted base
  `16/16`, frozen Multilingual winners `0/8`, held-out evaluations `0`, and
  Mono not started.

## Stage-B b0 improves at epoch 4; Stage-A epoch-14 generation active — 03:44 SAST

- Enhanced Stage-B `b0` job `1218997` produced a scientifically valid
  epoch-4 step-`2164` artifact. Tsn/Xho/Zul span micro-F1 is
  `0.5657188841201217/0.5305276251683962/0.5699790062980606`; exact mean
  `0.5554085051955262` matches newly retained `checkpoint-2164` exactly.
  The gain over epoch 3 is `0.07601672733558295`, above the frozen `0.001`
  threshold, so resetting patience and continuing is correct. This remains a
  within-candidate result; no candidate is frozen or pruned.
- Independent checks found exactly `192` rows, `64` per language, and all raw
  predictions nonempty. Tsn/Xho/Zul unique raw-output counts are `41/55/38`.
  Debug and trainer-state SHA-256 values are
  `0036f193e8a43a735b4f05787e4de599bcd7357563067eafd9e25f3271dd5783` and
  `17b023d47d3d8ae00439728baea1d477d13e32390763971392630d314ebda191`.
  The job resumed healthy epoch-5 training and had reached at least step
  `2494/8115`.
- Stage-A LR `1.5e-4` job `1218345` completed exact `10,760`-row epoch-14
  validation loss `0.3901939760796643` and entered corrected generation;
  generation segments began at `03:17:07/03:24:27/03:39:05 SAST`. No
  step-`7574` F1 artifact exists yet, so retained best remains epoch-12 F1
  `0.6795225766333571`. The frozen patience state remains one non-improving
  epoch until the epoch-14 F1 artifact is reconciled. The artifact is
  output-dependent and tentatively due around `03:50--04:00`.
- Stage-B `b1` job `1219931` remains pending, now `PENDING (Resources)`, with
  no output or metric. Targeted fault scans are empty and batch accounting is
  live. Owned state remains two running plus one pending A100-40GB jobs, no
  A100-80GB/L40S; quota home `52.0%`, scratch `37.5%`. Kombuys remains
  read-only/untouched, Sheet `GDN Results` E/F/G remain blank, trusted base
  `16/16`, frozen Multilingual winners `0/8`, held-out adapter evaluations
  `0`, and Mono not started.

## Stage-B b0 epoch-4 final generation active — 03:14 SAST

- Enhanced Stage-B `b0` job `1218997` completed exact `10,760`-row epoch-4
  health-only validation with finite loss `0.3470055222068134`. Corrected
  generation began its three language segments at
  `02:46:24/02:54:14/03:09:15 SAST`; the step-`2164` F1 artifact is not yet
  complete. No checkpoint, patience, pruning, or cross-candidate decision was
  made from validation loss. The artifact is output-dependent and tentatively
  expected around `03:22--03:32`.
- Stage-A LR `1.5e-4` job `1218345` reached the epoch-14 step-`7574` boundary
  and entered validation. No epoch-14 loss or F1 artifact exists yet, so
  retained provisional best remains epoch-12 F1 `0.6795225766333571` and the
  frozen patience state remains one non-improving epoch.
- Stage-B `b1` job `1219931` remains `PENDING (Priority)`. Targeted fault
  scans are empty and batch accounting is live. Owned state remains two
  running plus one pending A100-40GB jobs, no A100-80GB/L40S; quota home
  `52.0%`, scratch `37.5%`. Kombuys remains read-only/untouched, Sheet `GDN
  Results` E/F/G remain blank, trusted base `16/16`, frozen Multilingual
  winners `0/8`, held-out adapter evaluations `0`, and Mono not started.

## Stage-A LR-2 epoch 13 valid but non-improving — 02:42 SAST

- Stage-A NER LR `1.5e-4` job `1218345` produced a scientifically valid
  epoch-13 step-`7033` artifact. Tsn/Xho/Zul span micro-F1 is
  `0.6644535918204255/0.6800707736850073/0.6854468544684947`; exact mean
  `0.6766570733246425`. This is below the retained epoch-12 best
  `0.6795225766333571` by `0.0028655033087146`; `checkpoint-6492` correctly
  remains best and `checkpoint-7033/trainer_state.json` records that path and
  metric. This is the first non-improving epoch after the epoch-12 reset, so
  the run correctly continues under the frozen patience rule.
- Independent checks found exactly `192` rows, `64` per language, and all raw
  predictions nonempty. Tsn/Xho/Zul unique raw-output counts are `41/55/36`.
  Debug and epoch-13 trainer-state SHA-256 values are
  `baaa452714ef7691eaba51be29b06b8fd22e89dbba9a7761e41547fb5f773327` and
  `1129608194f8d2f80495c7489a18b60be53e9f7a98147741ad41a371c76d20b7`.
  The job resumed healthy epoch-14 training.
- Enhanced Stage-B `b0` job `1218997` reached its epoch-4 boundary at step
  `2164` and entered full validation. No epoch-4 loss or F1 artifact exists
  yet, so retained within-candidate best remains epoch-3 F1
  `0.47939177785994325` and no patience decision has been made.
- Stage-B `b1` job `1219931` remains `PENDING (Priority)`. Targeted fault
  scans are empty. Owned state remains two running plus one pending
  A100-40GB jobs, no A100-80GB/L40S; quota home `52.0%`, scratch `37.5%`.
  Kombuys remains read-only/untouched, Sheet `GDN Results` E/F/G remain
  blank, trusted base `16/16`, frozen Multilingual winners `0/8`, held-out
  adapter evaluations `0`, and Mono not started.

## Stage-B b0 improves at epoch 3 — 02:12 SAST

- Enhanced Stage-B `b0` job `1218997` produced a scientifically valid
  epoch-3 step-`1623` artifact. Tsn/Xho/Zul span micro-F1 is
  `0.4617483232919825/0.4673776054289369/0.5090494048589103`; exact mean
  `0.47939177785994325` matches newly retained `checkpoint-1623` exactly.
  The gain over epoch 2 is `0.12854471703682655`, above the frozen `0.001`
  threshold, so resetting patience and continuing is correct. This remains a
  within-candidate result; no candidate is frozen or pruned.
- Independent checks found exactly `192` rows, `64` per language, and all raw
  predictions nonempty. Tsn/Xho/Zul unique raw-output counts are `43/57/37`.
  Debug and trainer-state SHA-256 values are
  `5c95a258789f235e1de981339beba9d927734b98fe3c9db08d60594770746fa2` and
  `ed55337ea3b868c551b96c0c90e534fc70481aaaee3bd902f0fae236f882d90c`.
  The job remained running normally after checkpoint creation.
- Stage-A LR `1.5e-4` job `1218345` completed exact `10,760`-row epoch-13
  validation loss `0.38740929812746866` and entered corrected generation at
  `02:06:12 SAST`. No step-`7033` F1 artifact exists yet, so the retained
  provisional best remains epoch-12 F1 `0.6795225766333571`; no patience
  decision was made from validation loss. The next artifact is
  output-dependent and tentatively due around `02:35--02:50`.
- Stage-B `b1` job `1219931` remains `PENDING (Priority)`. Targeted fault
  scans are empty and batch accounting is live. Owned state remains two
  running plus one pending A100-40GB jobs, no A100-80GB/L40S; quota home
  `52.0%`, scratch `37.5%`. Kombuys remains read-only/untouched, Sheet `GDN
  Results` E/F/G remain blank, trusted base `16/16`, frozen Multilingual
  winners `0/8`, held-out adapter evaluations `0`, and Mono not started.

## Stage-A LR-2 improves at epoch 12 — 01:42 SAST

- Stage-A NER LR `1.5e-4` job `1218345` produced a scientifically valid
  epoch-12 step-`6492` artifact. Tsn/Xho/Zul span micro-F1 is
  `0.6625962304220365/0.685531574740158/0.6904399247378767`; exact mean
  `0.6795225766333571` matches newly retained `checkpoint-6492` exactly.
  The gain over epoch 11 is `0.0022997008228423`, above the frozen `0.001`
  threshold, so resetting patience and continuing is correct. This remains a
  provisional Stage-A result; NER is not frozen.
- Independent checks found exactly `192` rows, `64` per language, and all raw
  predictions nonempty. Tsn/Xho/Zul unique raw-output counts are `41/54/36`.
  Debug and trainer-state SHA-256 values are
  `f1b48ab554e8ffd84fb2673a78a4bdf78fa4442ee8ed80b3107e8c76d319adee` and
  `c0e54e9467e217855c8648926fc35fc876ae4bb481389d6d995cfac0db73fb66`.
  The job resumed healthy epoch-13 training and reached at least step
  `6686/8115`.
- Enhanced Stage-B `b0` job `1218997` completed exact `10,760`-row epoch-3
  validation loss `0.3786283528494569` and entered corrected generation at
  `01:31:46 SAST`. No step-`1623` F1 artifact exists yet, so its retained
  within-candidate best remains epoch-2 F1 `0.3508470608231167`; no selection
  or patience decision was made from validation loss. The next F1 artifact
  is tentatively due around `02:05--02:20`, output-dependent.
- Stage-B `b1` job `1219931` remains `PENDING (Priority)`. Targeted fault
  scans are empty and batch accounting is live. Owned state remains two
  running plus one pending A100-40GB jobs, no A100-80GB/L40S; quota home
  `52.0%`, scratch `37.5%`. Kombuys remains read-only/untouched, Sheet `GDN
  Results` E/F/G remain blank, trusted base `16/16`, frozen Multilingual
  winners `0/8`, held-out adapter evaluations `0`, and Mono not started.

## Stage-B b0 epoch 2 reconciles cleanly — 01:12 SAST

- Enhanced Stage-B `b0` job `1218997` produced a scientifically valid
  epoch-2 step-`1082` artifact. Tsn/Xho/Zul span micro-F1 is
  `0.34924399644228965/0.3375142531356411/0.36578293289141917`; exact mean
  `0.3508470608231167` matches newly retained `checkpoint-1082` exactly.
  The gain over epoch 1 is `0.24554592951990876`, well above the frozen
  `0.001` threshold, so resetting patience and continuing is correct. This is
  still only a within-candidate result; no candidate is frozen or pruned.
- Independent checks found exactly `192` rows, `64` per language, and all raw
  predictions nonempty. Tsn/Xho/Zul unique raw-output counts are `41/56/35`.
  Debug and trainer-state SHA-256 values are
  `1742612f543a642a15d99e8009fb3209dcc6ca1877c12453c5bfe6fbdbb0937b` and
  `3385bd0db11608ffe242dbade52563cdc8cdac6446e3b772a93ecda2fc3ef168`.
  The job resumed healthy epoch-3 training and had reached at least step
  `1367/8115`.
- Stage-A LR `1.5e-4` job `1218345` completed exact `10,760`-row epoch-12
  validation loss `0.38388843181851184` and entered corrected generation;
  generation segments began at `00:55:48/01:03:26 SAST`. No step-`6492` F1
  artifact exists yet, so the reconciled provisional best remains epoch-11
  F1 `0.6772228758105148` and no patience decision has been made.
- Stage-B `b1` job `1219931` remains `PENDING (Priority)`. Targeted fault
  scans are empty and batch accounting is live. Owned state remains two
  running plus one pending A100-40GB jobs, no A100-80GB/L40S; quota home
  `52.0%`, scratch `37.5%`. Kombuys remains read-only/untouched, Sheet `GDN
  Results` E/F/G remain blank, trusted base `16/16`, frozen Multilingual
  winners `0/8`, held-out adapter evaluations `0`, and Mono not started.

## Stage-B b0 epoch-2 final generation segment active — 00:46 SAST

- Enhanced Stage-B `b0` job `1218997` completed exact `10,760`-row epoch-2
  health-only validation with finite loss `0.4526454996442263`. Corrected
  task-native generation began its three language segments at
  `00:16:24/00:25:47/00:40:52 SAST`; the step-`1082` F1 artifact is not yet
  complete. No checkpoint, patience, pruning, or cross-candidate decision was
  made from validation loss. A scientifically usable artifact is tentatively
  expected around `00:53--01:05`, but remains output-dependent.
- Stage-A LR `1.5e-4` job `1218345` remains healthy in epoch-12 training at
  step `6463/8115`, retaining its reconciled provisional best F1
  `0.6772228758105148`. Stage-B `b1` job `1219931` remains
  `PENDING (Priority)` with no output or metric.
- Batch accounting is live and targeted fault scans remain empty. Owned state
  is two running plus one pending A100-40GB jobs, at the three-job cap, with
  no A100-80GB or L40S work. Quota remains home `52.0%`, scratch `37.5%`.
  Kombuys remains read-only and untouched; Sheet `GDN Results` E/F/G remain
  blank. Trusted base is `16/16`, frozen Multilingual winners `0/8`, held-out
  adapter evaluations `0`, and Monolingual work has not started.

## Stage-A LR-2 improves at epoch 11 — 00:21 SAST

- Stage-A NER LR `1.5e-4` job `1218345` produced a scientifically valid
  epoch-11 step-`5951` artifact. Tsn/Xho/Zul span micro-F1 is
  `0.6594354417145344/0.6821315944197481/0.6901015912972618`; exact mean
  `0.6772228758105148` matches newly retained `checkpoint-5951` exactly.
  The gain over epoch 10 is `0.0034844431694402`, above the frozen `0.001`
  threshold, so resetting patience and continuing is correct. This remains a
  provisional Stage-A result; NER is not frozen.
- Independent checks found exactly `192` rows, `64` per language, and all raw
  predictions nonempty. Tsn/Xho/Zul unique raw-output counts are `41/55/36`.
  Debug and trainer-state SHA-256 values are
  `afc4907de5e0c2eee003d73c5537a1c079d4300b2d8231afb84042b3f7a0c78f` and
  `351fb3a21ba8472a332a1cad79e886beb2bb491b54ed0699bd7a59c1cfe8df6a`.
  The job resumed healthy epoch-12 training immediately after the callback.
- Stage-B `b0` job `1218997` remains healthy in its epoch-2 callback with no
  step-`1082` F1 artifact yet. Stage-B `b1` job `1219931` remains
  priority-pending. No low-fidelity pruning or cross-candidate selection was
  performed, and no held-out metric was accessed.

## Epoch-11/epoch-2 callbacks healthy; Stage-B b1 submitted — 00:16 SAST

- Stage-A NER LR `1.5e-4` job `1218345` remains healthy on one A100-40GB.
  It completed exact `10,760`-row epoch-11 validation loss
  `0.38024836543767426` and entered corrected generation. The step-`5951`
  F1 artifact is not complete yet, so the retained provisional best remains
  epoch-10 mean F1 `0.6737384326410746`; no patience or selection decision
  was made. The next artifact is tentatively due around `00:20--00:35 SAST`.
- Enhanced Stage-B `b0` job `1218997` remains healthy on one A100-40GB. It
  reached epoch 2 at step `1082`, completed the five-example callback, and
  entered exact validation/generation. The step-`1082` F1 artifact is not
  complete yet, so its epoch-1 mean F1 `0.10530113130320794` remains only a
  within-candidate point. The preregistered no-low-fidelity-pruning rule keeps
  it running. Its next artifact is tentatively due around `00:45--01:05`.
- Targeted scans found no actual traceback, CUDA/OOM, NCCL, non-finite,
  manifest, or coverage fault. Batch accounting was live for both running
  jobs. No held-out split was accessed.
- After verifying exactly two owned jobs, no A100-80GB/L40S work, no existing
  `b1` artifact or duplicate, and the immutable launcher/registry hashes
  `f92329a9f510edddf7a5cf521cd52808309511d8a67d942838dc3d7bf44b4e59` /
  `8fdd6ea5f192a1d2ef2267353162b9a640ccd09ae0d889cb5a999645198bb726`,
  fixed Stage-B candidate `b1` was submitted as job `1219931`. Its frozen
  seed-42 configuration is LR `0.000030329114402234482`, rank/alpha `32/64`,
  dropout `0.08691746592521668`, and warmup `0.029702768474817273`.
  Slurm verified account/partition/QOS `nlpgroup/a100/nlpgroup`, exactly one
  `gpu:ampere`, 24 hours, eight CPUs, one node, and working directory
  `/home/lmbanr001/masters/sallm`. It is `PENDING (Priority)` with no output
  or metric; startup time is scheduler-dependent.
- Owned state is two running plus one pending A100-40GB jobs, exactly at the
  three-job cap, with no owned A100-80GB or L40S allocation. Quota is home
  `52.0%`, scratch `37.5%`. Kombuys was not accessed and remains read-only
  with RTX 5090 untouched.
- Operational and scientifically trusted base progress remains `16/16`.
  Frozen Multilingual winners remain `0/8`, held-out adapter evaluations `0`,
  and Monolingual work has not started. The verified Sheet remains unchanged:
  `GDN Results` E/F/G are blank. No Sheet or Hugging Face write occurred.
  Immediate blockers are terminal validation-only results for all Stage-A/B
  NER candidates and the preregistered seed-confirmation stage.
# Stage-B b2 improves at epoch 6 — 13:49 SAST

- Stage-B `b2` job `1220056` produced an exact 192-row epoch-6 step-`3246`
  artifact with 64 Tsn/Xho/Zul rows and finite span micro-F1
  `0.3792564071711693/0.3698532409294241/0.4096468561584341`. Their exact
  arithmetic mean `0.3862521680863425` matches newly retained
  `checkpoint-3246` and improves epoch 5 by `0.03569960217619822`, above the
  frozen `0.001` threshold, so the run correctly reset patience and resumed
  epoch 7 near step `3529/8115`.
- All `192/192` raw strings are nonempty, while `58/192` are whitespace-only
  and normalize empty: Tsn/Xho/Zul `23/7/28`. This behavior also occurs in
  earlier corrected NER artifacts and is measured by the preregistered
  empty-prediction metric; it is not the former evaluator-wide prompt-EOS
  fault. Declared/evaluated coverage remains exact and the primary metrics
  are finite. Debug/state SHA-256 values are
  `8eebe2ac749b092bbf5082cc0868768f832e870849198693baf2d4f77d146d7f` and
  `66bdf88c88547a2a0b75cf6ed99a55b0326a4d81e5f2db72d1849b927449f6d2`.
- `b0` job `1218997` remains in epoch-13 corrected generation after loss
  `0.40321707459630574`; `b1` job `1219931` remains in epoch-8 corrected
  generation after loss `0.39547758988731413`. Their step-`7033`/`4328`
  artifacts are absent. Targeted fault scans remain empty. All three jobs are
  healthy on `srvrocgpu010`, one A100-40GB each; quota is home `52.0%`,
  scratch `37.6%`. There is no owned A100-80GB/L40S work, Kombuys remains
  untouched, Sheet E/F/G remain blank, trusted base is `16/16`, frozen
  winners `0/8`, held-out adapter evaluations `0`, and Mono has not started.
