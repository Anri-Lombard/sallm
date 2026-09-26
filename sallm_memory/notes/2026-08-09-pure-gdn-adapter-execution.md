# Pure-GDN adapter execution — 2026-08-09

- At 00:26 SAST, General LR-0/LR-1/LR-2 validation jobs
  `1204261/1204262/1207524` were healthy at steps
  `4415/3593/512` of `13640`. Current training losses and gradients were
  finite and all three logs had zero runtime fault markers. LR-0/LR-1 retain
  their verified epoch-1 `checkpoint-2728` validation-loss artifacts; LR-2
  has no validation artifact yet. Conditional next validation windows remain
  approximately `02:15--02:20`, `03:40--03:45`, and `04:10--04:20 SAST`,
  respectively. Exactly three owned A100-40GB `gpu:ampere` jobs were active
  on `srvrocgpu010`, with no A100-80GB/L40S overlap. HEX quota was `/home`
  `3/10 GB` (`32.6%`) and `/scratch` `108/300 GB` (`36.0%`). Kombuys remained
  read-only and idle (RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch
  `61%`; only pre-existing `tailscale-kombuys` tmux). Base remains `16/16`
  and validation-frozen family winners `7/8`; no held-out adapter test, Sheet
  write, Hugging Face action, or new submission occurred. General terminal
  validation and validation-only winner freezing remain the sole blocker.

- At 00:56 SAST, General LR-0/LR-1/LR-2 jobs
  `1204261/1204262/1207524` remained healthy at steps
  `4697/3877/793` of `13640`, with current logs, finite training records, and
  zero fault markers. No new checkpoint or validation metric existed. Updated
  conditional validation windows are approximately `02:15--02:25`,
  `03:40--03:50`, and `04:15--04:25 SAST`. Exactly three owned A100-40GB
  `gpu:ampere` jobs remained active on `srvrocgpu010`, with no A100-80GB/L40S
  overlap. HEX quota remained `/home` `3/10 GB` (`32.6%`) and `/scratch`
  `108/300 GB` (`36.0%`). Kombuys remained read-only and idle (RTX 5090
  `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `61%`; only pre-existing
  `tailscale-kombuys` tmux). Base remains `16/16`, frozen family winners
  `7/8`; no held-out adapter test, Sheet write, Hugging Face action, or new
  submission occurred.

- At 01:26 SAST, General LR-0/LR-1/LR-2 jobs
  `1204261/1204262/1207524` remained healthy at steps
  `4983/4163/1084` of `13640`, with current finite training records and zero
  fault markers. No new validation artifact existed. Conditional validation
  windows remain approximately `02:15--02:25`, `03:40--03:50`, and
  `04:15--04:25 SAST`; LR-0 is now about one hour from its epoch-2 boundary.
  Exactly three owned A100-40GB `gpu:ampere` jobs remained active on
  `srvrocgpu010`, with no A100-80GB/L40S overlap. HEX quota remained `/home`
  `3/10 GB` (`32.6%`) and `/scratch` `108/300 GB` (`36.0%`). Kombuys remained
  read-only and idle (RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch
  `61%`; only pre-existing `tailscale-kombuys` tmux). Base remains `16/16`,
  frozen family winners `7/8`; no held-out adapter test, Sheet write, Hugging
  Face action, or new submission occurred.

- At 01:56 SAST, General LR-0/LR-1/LR-2 jobs
  `1204261/1204262/1207524` remained healthy at steps
  `5273/4454/1371` of `13640`, with current finite training records and zero
  fault markers. No new validation artifact existed. LR-0 was about 20
  minutes from its epoch-2 training boundary, with validation expected near
  `02:20--02:25 SAST`; LR-1/LR-2 remained on track for approximately
  `03:40--03:50` and `04:15--04:25 SAST`. Exactly three owned A100-40GB
  `gpu:ampere` jobs remained active on `srvrocgpu010`, with no A100-80GB/L40S
  overlap. HEX quota remained `/home` `3/10 GB` (`32.6%`) and `/scratch`
  `108/300 GB` (`36.0%`). Kombuys remained read-only and idle (RTX 5090
  `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `61%`; only pre-existing
  `tailscale-kombuys` tmux). Base remains `16/16`, frozen family winners
  `7/8`; no held-out adapter test, Sheet write, Hugging Face action, or new
  submission occurred.

- At 02:26 SAST, General LR-0 job `1204261` produced a valid epoch-2
  `checkpoint-5456` with validation loss `13.807909396570615`, worse than its
  epoch-1 best `13.603466245839952`. The saved trainer state correctly retains
  `checkpoint-2728` as `best_model_checkpoint` and records
  `best_metric=13.603466245839952`; LR-0 resumed epoch 3 at step
  `5473/13640` without a fault marker. Checkpoint-5456 adapter/config/trainer
  state SHA-256 values are
  `ad8ee9a8cd8ad0927b7bccc311ec14d9b6d5bf4a6751fa8070a21bc4987a1e4a` /
  `f2d56f5fed78c09f0d4e6a5e96a374cc46fc45c91c579aacc8c222ce75af419a` /
  `b2763115d5d2268220092250444e64b9f34069176d3b27217a5ffd5bfba2d0a2`.
  General LR-1/LR-2 jobs `1204262/1207524` remained healthy at steps
  `4741/1662`, with no new validation artifact and conditional validation
  windows near `03:45--03:50` and `04:20--04:25 SAST`. If LR-0's epoch-3
  validation is also non-improving, the frozen patience-2 rule can stop it
  near `07:15 SAST`; otherwise it continues without intervention. Exactly
  three owned A100-40GB `gpu:ampere` jobs remained active on `srvrocgpu010`,
  with no A100-80GB/L40S overlap. HEX quota was `/home` `3/10 GB` (`32.6%`)
  and `/scratch` `108/300 GB` (`36.1%`). Kombuys remained read-only and idle
  (RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `61%`; only
  pre-existing `tailscale-kombuys` tmux). Base remains `16/16`, frozen family
  winners `7/8`; no held-out adapter test, Sheet write, Hugging Face action,
  or new submission occurred.

- At 02:56 SAST, General LR-0/LR-1/LR-2 jobs
  `1204261/1204262/1207524` remained healthy at steps
  `5759/5027/1953` of `13640`, with current finite training records and zero
  fault markers. No new validation artifact existed. LR-1 was about 45
  minutes from its epoch-2 boundary with validation still expected near
  `03:45--03:50 SAST`; LR-2 remained on track for its first validation near
  `04:20--04:25 SAST`. LR-0's conditional patience-stop window remains near
  `07:15 SAST` if epoch 3 is also non-improving. Exactly three owned
  A100-40GB `gpu:ampere` jobs remained active on `srvrocgpu010`, with no
  A100-80GB/L40S overlap. HEX quota remained `/home` `3/10 GB` (`32.6%`) and
  `/scratch` `108/300 GB` (`36.1%`). Kombuys remained read-only and idle (RTX
  5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `61%`; only pre-existing
  `tailscale-kombuys` tmux). Base remains `16/16`, frozen family winners
  `7/8`; no held-out adapter test, Sheet write, Hugging Face action, or new
  submission occurred.

- At 03:26 SAST, General LR-0/LR-1/LR-2 jobs
  `1204261/1204262/1207524` remained healthy at steps
  `6046/5319/2241` of `13640`, with current finite training records and zero
  fault markers. No new validation artifact existed. LR-1 was about 15
  minutes from its epoch-2 training boundary with validation expected by
  roughly `03:50 SAST`; LR-2 was about 50 minutes from its first training
  boundary with validation expected near `04:20--04:25 SAST`. LR-0's
  conditional patience-stop window remains near `07:15 SAST` if epoch 3 is
  also non-improving. Exactly three owned A100-40GB `gpu:ampere` jobs remained
  active on `srvrocgpu010`, with no A100-80GB/L40S overlap. HEX quota remained
  `/home` `3/10 GB` (`32.6%`) and `/scratch` `108/300 GB` (`36.1%`). Kombuys
  remained read-only and idle (RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`,
  scratch `61%`; only pre-existing `tailscale-kombuys` tmux). Base remains
  `16/16`, frozen family winners `7/8`; no held-out adapter test, Sheet write,
  Hugging Face action, or new submission occurred.

- At 03:56 SAST, General LR-1 job `1204262` produced a valid epoch-2
  `checkpoint-5456` with validation loss `13.76491213184439`, worse than its
  epoch-1 best `13.732546259124712`. The saved trainer state correctly retains
  `checkpoint-2728` as `best_model_checkpoint` and records
  `best_metric=13.732546259124712`; LR-1 resumed epoch 3 at step
  `5520/13640` without a fault marker. Checkpoint-5456 adapter/config/trainer
  state SHA-256 values are
  `39a6da6f464fcce3d14c424b92f0d32e008e9e0f75f06ed9b2674fd89e5c0863` /
  `b8a895382a87dc48f5a5ef06c2aab19f33b6875ad50e453b3252487ebe268a6b` /
  `44f3c2a933ec615d66df2c0514750672d64b09194f2d2d6657065ee63325566e`.
  General LR-0 `1204261` remained healthy at step `6332`; LR-2 `1207524`
  remained healthy at step `2525`, about 20 minutes from its first training
  boundary with validation expected near `04:20--04:25 SAST`. Conditional
  patience-stop windows are near `07:15 SAST` for LR-0 and `08:40 SAST` for
  LR-1 if each epoch-3 validation also fails to improve. Exactly three owned
  A100-40GB `gpu:ampere` jobs remained active on `srvrocgpu010`, with no
  A100-80GB/L40S overlap. HEX quota remained `/home` `3/10 GB` (`32.6%`) and
  `/scratch` `108/300 GB` (`36.1%`). Kombuys remained read-only and idle (RTX
  5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `61%`; only pre-existing
  `tailscale-kombuys` tmux). Base remains `16/16`, frozen family winners
  `7/8`; no held-out adapter test, Sheet write, Hugging Face action, or new
  submission occurred.

- At 04:26 SAST, General LR-2 job `1207524` produced its valid epoch-1
  `checkpoint-2728` with validation loss `13.611691059151418`. Its trainer
  state records that checkpoint as `best_model_checkpoint` with
  `best_metric=13.611691059151418`; LR-2 resumed epoch 2 at step
  `2738/13640` without a fault marker. Checkpoint-2728 adapter/config/trainer
  state SHA-256 values are
  `dd5753ca8312fd048c09878954bb4a8b9d6f80bb3a5591c2aaef65a49e07e9d7` /
  `5c812370bb2b5738618a2be94a828971aeea60bf764989cf97ef3e479512c2af` /
  `5a53d4459dee08eb4a3406f88b5be9ba64b9382939e329b8d0281d7411df879d`.
  The interim validation-loss ordering is LR-0 `13.603466245839952`, LR-2
  `13.611691059151418`, LR-1 `13.732546259124712`; this is not a frozen winner
  because all three trials must terminate first. LR-0/LR-1 jobs
  `1204261/1204262` remained healthy at steps `6616/5809`. Conditional next
  validation windows are near `07:15 SAST` for LR-0, `08:40 SAST` for LR-1,
  and `09:15 SAST` for LR-2. Exactly three owned A100-40GB `gpu:ampere` jobs
  remained active on `srvrocgpu010`, with no A100-80GB/L40S overlap. HEX
  quota remained `/home` `3/10 GB` (`32.6%`) and `/scratch` `108/300 GB`
  (`36.1%`). Kombuys remained read-only and idle (RTX 5090 `10 MiB/0%`, RTX
  3080 Ti `1 MiB/0%`, scratch `61%`; only pre-existing `tailscale-kombuys`
  tmux). Base remains `16/16`, frozen family winners `7/8`; no held-out
  adapter test, Sheet write, Hugging Face action, or new submission occurred.

- At 04:56 SAST, General LR-0/LR-1/LR-2 jobs
  `1204261/1204262/1207524` remained healthy at steps
  `6903/6098/3026` of `13640`, with current finite training records and zero
  fault markers. No new validation artifact existed. Conditional next
  validation windows remain near `07:15 SAST`, `08:40 SAST`, and
  `09:15 SAST`, respectively. Exactly three owned A100-40GB `gpu:ampere`
  jobs remained active on `srvrocgpu010`, with no A100-80GB/L40S overlap.
  HEX quota remained `/home` `3/10 GB` (`32.6%`) and `/scratch` `108/300 GB`
  (`36.1%`). Kombuys remained read-only and idle (RTX 5090 `10 MiB/0%`, RTX
  3080 Ti `1 MiB/0%`, scratch `61%`; only pre-existing `tailscale-kombuys`
  tmux). Base remains `16/16`, frozen family winners `7/8`; no held-out
  adapter test, Sheet write, Hugging Face action, or new submission occurred.

- At 05:26 SAST, General LR-0/LR-1/LR-2 jobs
  `1204261/1204262/1207524` remained healthy at steps
  `7189/6383/3307` of `13640`, with current finite training records and zero
  fault markers. No new validation artifact existed. Conditional next
  validation windows remain near `07:15 SAST`, `08:40 SAST`, and
  `09:15 SAST`, respectively. Exactly three owned A100-40GB `gpu:ampere`
  jobs remained active on `srvrocgpu010`, with no A100-80GB/L40S overlap.
  HEX quota remained `/home` `3/10 GB` (`32.6%`) and `/scratch` `108/300 GB`
  (`36.1%`). Kombuys remained read-only and idle (RTX 5090 `10 MiB/0%`, RTX
  3080 Ti `1 MiB/0%`, scratch `61%`; only pre-existing `tailscale-kombuys`
  tmux). Base remains `16/16`, frozen family winners `7/8`; no held-out
  adapter test, Sheet write, Hugging Face action, or new submission occurred.

- At 05:56 SAST, General LR-0/LR-1/LR-2 jobs
  `1204261/1204262/1207524` remained healthy at steps
  `7484/6681/3605` of `13640`, with current log writes and zero fault markers.
  No new validation artifact existed. Conditional next validation windows
  remain near `07:15--07:20`, `08:40`, and `09:15 SAST`, respectively.
  Exactly three owned A100-40GB `gpu:ampere` jobs remained active on
  `srvrocgpu010`, with no A100-80GB/L40S overlap. HEX quota remained `/home`
  `3/10 GB` (`32.6%`) and `/scratch` `108/300 GB` (`36.1%`). Kombuys remained
  read-only and idle (RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch
  `61%`; only pre-existing `tailscale-kombuys` tmux). Base remains `16/16`,
  frozen family winners `7/8`; no held-out adapter test, Sheet write, Hugging
  Face action, or new submission occurred.

- At 06:26 SAST, General LR-0/LR-1/LR-2 jobs
  `1204261/1204262/1207524` remained healthy at steps
  `7760/6958/3883` of `13640`, with current log writes and zero fault markers.
  Their checkpoint sets remained exactly `2728/5456`, `2728/5456`, and
  `2728`, so no new validation artifact existed. Conditional validation
  windows remain near `07:15--07:20`, `08:40`, and `09:15 SAST`.
  Exactly three owned A100-40GB `gpu:ampere` jobs remained active on
  `srvrocgpu010`, with no A100-80GB/L40S overlap. HEX quota remained `/home`
  `3/10 GB` (`32.6%`) and `/scratch` `108/300 GB` (`36.1%`). Kombuys remained
  read-only and idle (RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch
  `61%`; only pre-existing `tailscale-kombuys` tmux). Base remains `16/16`,
  frozen family winners `7/8`; no held-out adapter test, Sheet write, Hugging
  Face action, or new submission occurred.

- At 06:56 SAST, General LR-0/LR-1/LR-2 jobs
  `1204261/1204262/1207524` remained healthy at steps
  `8046/7247/4170` of `13640`, with current log writes and zero fault markers.
  Their checkpoint sets remained exactly `2728/5456`, `2728/5456`, and
  `2728`; LR-0 was about 15 raw minutes from its epoch-3 boundary, preserving
  a validation/conditional patience-stop window near `07:15--07:20 SAST`.
  LR-1/LR-2 remain on track near `08:40` and `09:15 SAST`. Exactly three
  owned A100-40GB `gpu:ampere` jobs remained active on `srvrocgpu010`, with
  no A100-80GB/L40S overlap. HEX quota remained `/home` `3/10 GB` (`32.6%`)
  and `/scratch` `108/300 GB` (`36.1%`). Kombuys remained read-only and idle
  (RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `61%`; only
  pre-existing `tailscale-kombuys` tmux). Base remains `16/16`, frozen family
  winners `7/8`; no held-out adapter test, Sheet write, Hugging Face action,
  or new submission occurred.

- General LR-0 job `1204261` completed cleanly `0:0` at `07:18:51 SAST`
  after `14:41:51`. Epoch-3 `checkpoint-8184` validation loss was
  `13.80606733455681`, non-improving versus epoch-1
  `13.603466245839952`; the frozen patience-2 rule therefore stopped the run
  after epoch 3 and retained epoch-1 `checkpoint-2728`. The retained
  checkpoint is the only checkpoint directory. A read-only exact tensor
  comparison verified all `424/424` final-adapter tensors equal the retained
  checkpoint, with zero missing, extra, or mismatched keys. Final adapter,
  config, retained checkpoint weights/config/trainer-state, and run README
  SHA-256 are respectively
  `50ebbeddd2a45206f3d73c1eda62d188cf2d269848027e7227cffb4309381b8c`,
  `f2d56f5fed78c09f0d4e6a5e96a374cc46fc45c91c579aacc8c222ce75af419a`,
  `5c1c08009f5c83bba61a40c998b5e6733f754068dd46168fca81ba1f5bcd1b5d`,
  `f2d56f5fed78c09f0d4e6a5e96a374cc46fc45c91c579aacc8c222ce75af419a`,
  `b5dad80258441a283823304241b94d16e604c46c1e1a6d9db62e2bab473a9bed`,
  and `66c0acb0603e1fb2ab523e63c013df81f2e544f04e020b388fd1de474ab804c3`.
  At 07:26 SAST, LR-1/LR-2 jobs `1204262/1207524` remained healthy at
  `7532/4459` of `13640` with zero fault markers and conditional validation
  windows near `08:40` and `09:15 SAST`. They were the only two owned GPU
  jobs, both A100-40GB `gpu:ampere` on `srvrocgpu010`; no A100-80GB/L40S
  overlap existed. HEX quota was `/home` `3/10 GB` (`32.6%`) and `/scratch`
  `108/300 GB` (`36.2%`). Kombuys remained read-only and idle (RTX 5090
  `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `61%`; only pre-existing
  `tailscale-kombuys` tmux). Base remains `16/16`, frozen family winners
  `7/8`; General cannot be frozen until LR-1/LR-2 terminate, so no held-out
  adapter test, Sheet write, Hugging Face action, or new submission occurred.

- At 07:56 SAST, General LR-1/LR-2 jobs `1204262/1207524` remained healthy
  at `7819/4747` of `13640`, with current log writes and zero fault markers.
  Their checkpoint sets remained exactly `2728/5456` and `2728`, so no new
  validation artifact existed. Conditional validation/patience-stop windows
  remain near `08:40` and `09:15--09:20 SAST`. These were the only two owned
  GPU jobs, both A100-40GB `gpu:ampere` on `srvrocgpu010`; no A100-80GB/L40S
  overlap existed. HEX quota remained `/home` `3/10 GB` (`32.6%`) and
  `/scratch` `108/300 GB` (`36.2%`). Kombuys remained read-only and idle
  (RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `61%`; only
  pre-existing `tailscale-kombuys` tmux). Base remains `16/16`, frozen family
  winners `7/8`; General, held-out adapter evaluation, Sheet writes, Hugging
  Face publication, and any new submission remain blocked until both trials
  terminate and the validation-only winner is frozen.

- At 08:19 SAST, General LR-1/LR-2 jobs `1204262/1207524` remained healthy
  at `8048/4969` of `13640`, with current log writes and zero fault markers.
  Their checkpoint sets remained exactly `2728/5456` and `2728`, so no new
  validation artifact existed. LR-1 was about 15 raw minutes from its epoch-3
  boundary, preserving a validation/conditional patience-stop window near
  `08:40 SAST`; LR-2 remained on track near `09:15--09:20 SAST`. These were
  the only two owned GPU jobs, both A100-40GB `gpu:ampere` on
  `srvrocgpu010`; no A100-80GB/L40S overlap existed. HEX quota remained
  `/home` `3/10 GB` (`32.6%`) and `/scratch` `108/300 GB` (`36.2%`). Kombuys
  remained read-only and idle (RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`,
  scratch `61%`; only pre-existing `tailscale-kombuys` tmux). Base remains
  `16/16`, frozen family winners `7/8`; General, held-out adapter evaluation,
  Sheet writes, Hugging Face publication, and new submissions remain blocked
  until both trials terminate and the validation-only winner is frozen.

- General LR-1 job `1204262` completed cleanly `0:0` at `08:42:25 SAST`
  after `14:38:37`. Epoch-3 validation loss was `13.776524366234797`,
  non-improving versus epoch-1 `13.732546259124712`; the frozen patience-2
  rule therefore stopped the run after epoch 3 and retained epoch-1
  `checkpoint-2728` as its only checkpoint directory. A read-only exact
  tensor comparison verified all `424/424` final-adapter tensors equal the
  retained checkpoint, with zero missing, extra, or mismatched keys. Final
  adapter, config, retained checkpoint weights/config/trainer-state, and run
  README SHA-256 are respectively
  `c15d6079785714ea40881dbeba5f3be7e318b789cf7297f180ae6af8de328fb1`,
  `b8a895382a87dc48f5a5ef06c2aab19f33b6875ad50e453b3252487ebe268a6b`,
  `ac4fbca21f6dd2b0df0072faed6d6553813de6e5903ca5bcc50a9091623fd4b9`,
  `b8a895382a87dc48f5a5ef06c2aab19f33b6875ad50e453b3252487ebe268a6b`,
  `94a12d58edd30c3ca54d0185f8e0375071783625592c07a31f5b6b6a141013cb`,
  and `6ec920984b9ecb8997bec8fa0863370158071eb074da37c8537b2bdebdb5e12f`.
  At 08:56 SAST, LR-2 job `1207524` was the sole owned GPU job and remained
  healthy at `5320/13640` with zero fault markers. It was about 15 raw
  minutes from epoch 2, preserving a validation window near
  `09:15--09:20 SAST`. The job remained on `srvrocgpu010` A100-40GB
  `gpu:ampere`; no A100-80GB/L40S overlap existed. HEX quota remained `/home`
  `3/10 GB` (`32.6%`) and `/scratch` `108/300 GB` (`36.2%`). Kombuys remained
  read-only and idle (RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch
  `61%`; only pre-existing `tailscale-kombuys` tmux). Base remains `16/16`,
  frozen family winners `7/8`; General cannot be frozen until LR-2
  terminates, so no Monolingual submission, held-out adapter evaluation,
  Sheet write, or Hugging Face action occurred.

- General LR-2 job `1207524` improved at epoch 2: retained
  `checkpoint-5456` has validation loss `13.511420388596335`, better than its
  epoch-1 `13.611691059151418` and better than the completed LR-0/LR-1 bests
  `13.603466245839952/13.732546259124712`. LR-2 is therefore the provisional
  General leader, but the family remains unfrozen until this trial terminates
  under the preregistered patience rule. The saved trainer state records
  `best_global_step=5456`, `best_metric=13.511420388596335`, and
  `best_model_checkpoint=checkpoint-5456`; retained checkpoint
  weights/config/trainer-state SHA-256 are
  `31fd7c1c46bc0496f62577a5b4b447fdd249200a630db24e962cd0fa701f8f11`,
  `5c812370bb2b5738618a2be94a828971aeea60bf764989cf97ef3e479512c2af`,
  and `ae3b8c0f7e05ac0c3e694f7b4fe489c75ec44d06a87d4699252365825d42a938`.
  At 09:26 SAST it had resumed epoch 3 at `5519/13640` with current log
  writes and zero fault markers; the next validation is conditionally near
  `14:10--14:15 SAST`. It was the sole owned GPU job on `srvrocgpu010`
  A100-40GB `gpu:ampere`, with no A100-80GB/L40S overlap. HEX quota remained
  `/home` `3/10 GB` (`32.6%`) and `/scratch` `108/300 GB` (`36.2%`). Kombuys
  remained read-only and idle (RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`,
  scratch `61%`; only pre-existing `tailscale-kombuys` tmux). Base remains
  `16/16`, frozen family winners `7/8`; no Monolingual submission, held-out
  adapter evaluation, Sheet write, Hugging Face action, or new submission
  occurred.

- At 09:56 SAST, General LR-2 job `1207524` remained healthy at
  `5815/13640`, with current log writes and zero fault markers. Retained
  `checkpoint-5456` and its best validation loss `13.511420388596335`
  remained unchanged; the epoch-3 validation window remains near
  `14:10--14:15 SAST`. It was the sole owned GPU job on `srvrocgpu010`
  A100-40GB `gpu:ampere`, with no A100-80GB/L40S overlap. HEX quota remained
  `/home` `3/10 GB` (`32.6%`) and `/scratch` `108/300 GB` (`36.2%`). Kombuys
  remained read-only and idle (RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`,
  scratch `61%`; only pre-existing `tailscale-kombuys` tmux). Base remains
  `16/16`, frozen family winners `7/8`; no Monolingual submission, held-out
  adapter evaluation, Sheet write, Hugging Face action, or new submission
  occurred.

- At 10:26 SAST, General LR-2 job `1207524` remained healthy at
  `6101/13640`, with current log writes and zero fault markers. Retained
  `checkpoint-5456` and best validation loss `13.511420388596335` remained
  unchanged; epoch-3 validation remains expected near `14:10--14:15 SAST`.
  It was the sole owned GPU job on `srvrocgpu010` A100-40GB `gpu:ampere`,
  with no A100-80GB/L40S overlap. HEX quota remained `/home` `3/10 GB`
  (`32.6%`) and `/scratch` `108/300 GB` (`36.2%`). Kombuys remained
  read-only and idle (RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch
  `61%`; only pre-existing `tailscale-kombuys` tmux). Base remains `16/16`,
  frozen family winners `7/8`; no Monolingual submission, held-out adapter
  evaluation, Sheet write, Hugging Face action, or new submission occurred.

- At 10:56 SAST, General LR-2 job `1207524` remained healthy at
  `6399/13640`, with current log writes and zero fault markers. Retained
  `checkpoint-5456` and best validation loss `13.511420388596335` remained
  unchanged; epoch-3 validation remains expected near `14:10--14:15 SAST`.
  It was the sole owned GPU job on `srvrocgpu010` A100-40GB `gpu:ampere`,
  with no A100-80GB/L40S overlap. HEX quota remained `/home` `3/10 GB`
  (`32.6%`) and `/scratch` `108/300 GB` (`36.2%`). Kombuys remained
  read-only and idle (RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch
  `61%`; only pre-existing `tailscale-kombuys` tmux). Base remains `16/16`,
  frozen family winners `7/8`; no Monolingual submission, held-out adapter
  evaluation, Sheet write, Hugging Face action, or new submission occurred.
