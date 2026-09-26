# Pure-GDN validation-only HPO — 2026-08-20

## AfriHG b0 checkpoint 12328 becomes retained validation best — 22:54 SAST

- Job `1248575` completed its step-12328 exact callback and resumed healthy
  training. The artifact has exactly `128` rows split `64/64` Xho/Zul, with
  chrF `23.470367754445988/25.368293786591423` and registered mean
  `24.419330770518705`. This supersedes b0 checkpoint 9246 within the run;
  `trainer_state.json` retains `checkpoint-12328` with the same best metric.
- Integrity checks pass: zero empty/debug-empty predictions, `64/64` unique
  predictions per language, all prompts begin `[BOS]` and end
  `[EOS]<|assistant|>`, and zero EOS tokens occur after the assistant marker.
  Artifact SHA-256 is
  `22a35faa403babfc22c2ed2aee1bd48820485888a9b762b8934d77cbb10fdd50`;
  trainer-state SHA-256 is
  `728f1f52d605dea12b33c36ef23ae2e59a81c86ef1143c3f244c6be3c62add37`.
- This remains within-run validation-only evidence, not a terminal-valid trial
  or frozen family winner. At `22:54 SAST`, b0 was healthy near
  `12405/15410`; jobs `1248576/1248577` remained pending. AfriHG stays `3/11`
  terminal-valid and the global freeze `0/8`. HEX quota is home `88.6%`,
  scratch `40.4%`; hardware remains A100-40GB only, Kombuys is read-only,
  held-out access remains `0`, and Sheet E/F/G remain blank.

## AfriHG b0 checkpoint 9246 becomes retained validation best — 19:24 SAST

- Job `1248575` completed its step-9246 exact callback and resumed healthy
  training. The artifact has exactly `128` rows split `64/64` Xho/Zul, with
  chrF `23.769019260468166/24.8046626683949` and registered mean
  `24.286840964431534`. This supersedes b0 checkpoint 6164 within the run;
  `trainer_state.json` retains `checkpoint-9246` with the same best metric.
- Integrity checks pass: zero empty/debug-empty predictions, `64/64` unique
  predictions per language, all prompts begin `[BOS]` and end
  `[EOS]<|assistant|>`, and zero EOS tokens occur after the assistant marker.
  Artifact SHA-256 is
  `47b9b7c714b128b63c6fe2d7d33f9e86d776c332e658b13320999cfca76f6ef8`;
  trainer-state SHA-256 is
  `635bea2131920a4ccd7715d68d9ba2e4452d84e7a85a7aa51ef2511b621dfb91`.
- This remains within-run validation-only evidence, not a terminal-valid trial
  or frozen family winner. At `19:24 SAST`, b0 was healthy near
  `9573/15410`; jobs `1248576/1248577` remained pending. AfriHG stays `3/11`
  terminal-valid and the global freeze `0/8`. HEX quota is home `88.6%`,
  scratch `40.4%`; hardware remains A100-40GB only, Kombuys is read-only,
  held-out access remains `0`, and Sheet E/F/G remain blank.

## AfriHG b0 checkpoint 6164 becomes retained validation best — 15:54 SAST

- Job `1248575` completed its step-6164 exact callback and resumed training.
  The artifact has exactly `128` rows split `64/64` Xho/Zul, with chrF
  `22.863993326854846/24.03782125622953` and registered mean
  `23.450907291542187`. This improves b0's prior step-3082 mean
  `22.11669022677217`; checkpoint 6164 is the within-run retained best.
- Integrity checks pass: no empty/debug-empty predictions, `64/64` unique
  predictions per language, all prompts begin `[BOS]` and end
  `[EOS]<|assistant|>`, and zero EOS tokens occur after the assistant marker.
  Artifact SHA-256 is
  `6cc6ea0b24c8506a2494bc3d585db08643b2a0e995aee14d1d73aa311834ba7e`;
  trainer-state SHA-256 is
  `698b45e8fbcdf6e43d1f2032580fb919e0314cf71303d313fe0d4c9165571c84`.
- This is eligible validation-only evidence, not a terminal-valid trial or
  frozen family winner. AfriHG remains `3/11` terminal-valid and the global
  freeze remains `0/8`; jobs `1248576/1248577` remain pending. Hardware is
  A100-40GB only, quota remains `88.6%/40.4%`, held-out access remains `0`,
  Kombuys is read-only, and Sheet E/F/G remain blank.

## AfriHG b0 reaches second validation boundary — 14:24 SAST

- Stage-B b0 job `1248575` reached step `6164/15410` and completed full
  declared validation over all `3,082` rows at health-only loss
  `2.214256766862021` in `138.8222 s`. The frozen exact generation callback
  started with automatic batch size 64 at `14:22:32`; no step-6164 exact
  artifact exists yet, so this does not change selector ranking.
- Job `1248575` remains healthy on `srvrocgpu010` A100-40GB. Jobs
  `1248576/1248577` remain pending (`Resources`/`Priority`), with b1's
  provisional start `18:31:54 SAST`. Quota is home `88.6%`, scratch `40.4%`;
  no A100-80GB/L40S work overlaps, held-out access remains `0`, and the
  scientific state remains AfriHG `3/11` terminal-valid with global freeze
  `0/8`. Kombuys remains read-only and Sheet E/F/G remain blank.

## GDN result-sheet presentation normalized — 13:53 SAST

- Removed the operational progress dashboard from `GDN Results!A46:J59` so
  the tab retains the same result-only structure as the other architecture
  tabs. Adapter columns E/F/G remain blank pending verified one-time held-out
  artifacts.
- Replaced artifact paths in `GDN Results!J2:J44` with one representative
  `Input / Gold / Pure GDN` example for every applicable result row, sourced
  from the corrected frozen base artifacts. Row 40 remains blank and AfriHG
  English is explicitly marked structurally inapplicable.
- Artifact paths, record selectors, and prior visible provenance were retained
  in cell notes. Verification found 42 populated example cells, zero visible
  scratch/JSON paths, zero missing provenance notes, unchanged wrapped/top
  formatting, and zero values in E/F/G.

## AfriHG b0 produces first exact selector artifact — 11:43 SAST

- Job `1248575` completed its frozen step-3082 exact callback at `11:38:08`
  and resumed training. The artifact has exactly `128` rows split `64/64`
  Xho/Zul. Xho/Zul chrF is
  `21.196932693799432/23.036447759744906`, giving the registered mean
  `22.11669022677217`; `trainer_state.json` records this as `best_metric` at
  retained `checkpoint-3082`.
- Artifact integrity checks pass: zero empty or whitespace-only predictions,
  `64/64` unique predictions per language, every prompt starts with `[BOS]`
  and ends with `[EOS]<|assistant|>`, and no targeted fault marker is present.
  Artifact SHA-256 is
  `825e6a931dfc37a728e739e7df7bf725c1e1e2c1cace17e461b258f0b9d26a4c`;
  trainer-state SHA-256 is
  `848ac3e516b906bf15c6cca308297654a9376e61a3469bd4e79fd40429d00779`.
- This is scientifically eligible validation-only selector evidence, but b0
  is not terminal-valid until the full run and terminal reconciliation pass.
  AfriHG therefore remains `3/11` seed-42 terminal-valid, with b0's first
  eligible checkpoint now available; no family winner is frozen and the
  global freeze remains `0/8`. Jobs `1248576/1248577` remain pending and all
  owned work is A100-40GB only. Quota remains `88.6%/40.4%`, held-out access
  remains `0`, and Kombuys remains read-only with RTX 5090 untouched.

## AfriHG b0 reaches first validation boundary — 11:07 SAST

- Job `1248575` reached exactly step `3082/15410` and completed full declared
  validation over all `3,082` rows at health-only loss
  `2.269708057876379` in `139.0663 s`. Language coverage is preserved at
  `1,305` Xho and `1,777` Zul rows. This loss is health evidence only, not
  selector evidence.
- The frozen exact generation callback started with automatic batch size 64;
  both language segments have begun and the job remains running on
  `srvrocgpu010` A100-40GB with no targeted fault marker. The step-3082 exact
  JSONL is not yet present, so AfriHG remains scientifically `3/11`
  terminal-valid and the global family freeze remains `0/8`.
- Jobs `1248576/1248577` remain pending (`Resources`/`Priority`), with b1's
  provisional start still `18:31:54 SAST` and no b2 ETA. HEX quota remains
  home `88.6%`, scratch `40.4%`; held-out access remains `0`, Kombuys is
  read-only with RTX 5090 untouched, and no A100-80GB/L40S work overlaps.

## AfriHG Stage-B b0 starts — 08:08 SAST

- B0 job `1248575` started at `07:46:38 SAST` on `srvrocgpu010` with one
  A100-40GB `gpu:ampere`. At `08:08` it was healthy near `370/15410` after
  about 21 minutes, with zero targeted fault markers. The fast GatedDeltaNet
  path passed, declared data is `24,649/3,082` train/validation rows, and the
  run uses the frozen validation-only b0 recipe: LR
  `0.00015156541821567134`, rank/alpha `8/16`, dropout
  `0.028929591178894043`, and warmup `0.061286738514900206`.
- Immutable source and configuration provenance reconcile. Execution-manifest
  SHA-256 is
  `630e9a5506288e57a8b6440e30941b1d1786d3870a039f26b39ec29b915da36a`,
  trial-record SHA-256 is
  `c3067b07bc81719ca08eae359e2c8131fd195f61821a4b921e40ef399b1a0fcc`,
  and the frozen registry SHA-256 is
  `8fdd6ea5f192a1d2ef2267353162b9a640ccd09ae0d889cb5a999645198bb726`.
- At the observed `~3.1 s/step`, b0's first step-3082 training boundary is
  tentatively near `10:25--10:40 SAST`; full validation and its frozen exact
  callback put the first selector artifact roughly near `11:00--11:30`.
  These are operational estimates, not selection evidence.
- Jobs `1248576/1248577` for b1/b2 remain pending on A100-40GB only. B1 has
  provisional start `18:31:54 SAST`; b2 has no estimate. Scientifically
  AfriHG remains `3/11` terminal-valid at seed 42 and no family winner is
  frozen (`0/8` globally). HEX quota is home `88.6%`, scratch `40.4%`;
  Kombuys remains read-only with RTX 5090 untouched, held-out access remains
  `0`, and General/Monolingual/publication remain blocked. The live Sheet
  tracker was updated and verified with formatting intact and E/F/G blank.

## AfriHG Stage A closes; Stage B starts — 07:42 SAST

- A2 job `1246940` completed `0:0` at `07:20:44 SAST` after `18:54:18`
  on `srvrocgpu010` A100-40GB. Its terminal step-15410 exact artifact has
  exactly `128` rows, split `64/64` Xho/Zul, with Xho/Zul chrF
  `23.800277043809448/26.021454194337124` and mean
  `24.910865619073284`. It has no exact-empty, whitespace-only,
  normalized-empty, or debug-empty prediction and `64/64` unique predictions
  per language. Every prompt starts with `[BOS]`, ends with
  `[EOS]<|assistant|>`, and has zero EOS tokens after the assistant marker.
- Terminal `trainer_state.json` records global step `15410`, best metric
  `24.910865619073284`, and retained `checkpoint-15410`. The retained BIN and
  final safetensors contain the same `424` keys and all `71,762,560` tensor
  values compare exactly. SHA-256 values are terminal artifact
  `81c58c78143bdc8d653ed444446f2d2ef09eb5d712f14da2ce080cfdb8569162`,
  trainer state
  `0049ef1189713b034e45a6ce6833940710de03c694f70af13d1fc5698bffb902`,
  retained weights
  `c55fd9c466389862013889cace952a8e1750fbced6d3a20cbc129dbb2f07c52e`,
  final weights
  `0dbac82ff84ad77936bb35f36eba44456d88b2ab20ebeef0a34da7613adf960a`,
  adapter config
  `04ca6881ae6b2d040d1315a33882080a84673433be9030752efc01742d0f8985`,
  execution manifest
  `03ac17c961a2f9ca96644f11b8bd3058f70b6f37fa57b777cb6c4d4b910e5d3d`,
  and trial record
  `a85ed5f5e3e4d15f8e3ffb00157fafc0236cd1363d88797ad764ab505c07ab11`.
- A2 is terminal-valid and the strongest Stage-A candidate, closing Stage A
  at `3/3`. It is not a frozen AfriHG winner: the preregistration requires all
  eight Stage-B candidates at seed 42, ranking all eleven candidates, and
  three-seed confirmation of the top two. AfriHG is therefore `3/11`
  terminal-valid at seed 42 and the global final-family freeze remains `0/8`.
- With no owned jobs and absent b0/b1/b2 output roots, immutable-snapshot
  verification passed and Stage-B b0/b1/b2 were submitted once as jobs
  `1248575/1248576/1248577`. All request `nlpgroup/a100/nlpgroup`, one
  A100-40GB `gpu:ampere`, 24 hours, eight CPUs, and the canonical home working
  directory. All are pending without model or data access; b0 has a
  provisional Slurm start estimate of `18:31:54 SAST`, while b1/b2 have no
  estimate yet.
- No owned job is running and no A100-80GB/L40S work is owned. HEX quota is
  home `88.6%`, scratch `40.2%`; Kombuys remains read-only with RTX 5090
  untouched, and held-out adapter access remains `0`. The live Sheet tracker
  was re-read, corrected to the full Stage-B/Stage-C protocol, written, and
  verified with formatting intact and E/F/G blank. General, Monolingual, and
  publication remain blocked.

## AfriHG a2 enters terminal exact callback — 06:37 SAST

- A2 job `1246940` reached exactly `15410/15410` at `06:12:58 SAST`, then
  completed terminal full declared validation over all `3,082` rows at
  health-only loss `2.26641358530885` in `139.8180 s`. This loss is not
  selector evidence; checkpoint 12328 remains the eligible validation-only
  best at mean chrF `24.78084394766916` until terminal reconciliation.
- The frozen terminal exact callback established automatic generation batch
  size 64 for its first segment at `06:19:19 SAST` and remained healthy in
  that segment at `06:37:41`, with no targeted fault marker. The terminal
  step-15410 JSONL and final adapter remain absent. Prior callback timing
  keeps terminal reconciliation near `07:15--07:30 SAST`.
- Operationally jobs `1246938/1246939` are complete and `1246940` is in its
  terminal callback; scientifically AfriHG remains `2/3` terminal-valid and
  the global freeze remains `0/8`. No AfriHG winner is frozen and no
  held-out artifact has been touched.
- A2 is the only owned job and uses one A100-40GB `gpu:ampere`; no
  A100-80GB/L40S work overlaps. HEX quota is home `88.6%`, scratch `40.2%`;
  Kombuys remains read-only with RTX 5090 untouched, Sheet E/F/G remain
  blank, and General/Monolingual/publication remain blocked.

## AfriHG a2 checkpoint 12328 becomes retained validation best — 05:39 SAST

- A2 job `1246940` completed both frozen exact callbacks missed between
  monitor cycles and resumed healthy, fault-free training to about
  `14717/15410` after `17:12:36` on `srvrocgpu010` A100-40GB. The step-9246
  artifact has exactly `128` rows, split `64/64` Xho/Zul, with Xho/Zul chrF
  `23.896595514862387/25.517543978828154` and mean
  `24.70706974684527`.
- The newer step-12328 artifact also has exactly `128` rows, split `64/64`
  Xho/Zul, with Xho/Zul chrF
  `23.738394297462413/25.823293597875907` and mean
  `24.78084394766916`. This exactly matches `trainer_state.json`
  `best_metric`, `best_global_step=12328`, and retained
  `checkpoint-12328`, superseding a2 checkpoint 9246 within the same run.
- Both artifacts have no exact-empty, whitespace-only, normalized-empty, or
  debug-empty output and `64/64` unique predictions per language. All 256
  prompts start with `[BOS]`, end with `[EOS]<|assistant|>`, and have zero
  EOS tokens after the assistant marker. SHA-256 values are step 9246
  `3f986111f3945846f47ea3b502edebd9d7a37f31e421668a6b7138b0062eb87f`,
  step 12328
  `d9285b011c04c1b14cdff110a8929786f3e6d761d414a51b3cd4e0a1e23cd5b0`,
  retained trainer state
  `2a06be19720b2b9ff0c4a06fb15f02fb351a043058cea451ace2b9f07dec43ef`,
  and adapter config
  `04ca6881ae6b2d040d1315a33882080a84673433be9030752efc01742d0f8985`.
- This is valid within-run validation evidence only: a2 is not terminal-valid
  and no AfriHG winner is frozen. Operationally jobs `1246938/1246939` are
  complete and `1246940` is running; scientifically AfriHG remains `2/3`
  terminal-valid and the global freeze remains `0/8`. A2 is about 36 minutes
  from its step-15410 boundary, followed by full validation and the terminal
  exact callback; terminal reconciliation is tentatively due around
  `07:15--07:30 SAST`.
- A2 is the only owned job and requests exactly one A100-40GB
  `gpu:ampere`; no A100-80GB/L40S work overlaps. HEX quota is home `88.6%`,
  scratch `40.2%`; Kombuys remains read-only with RTX 5090 untouched,
  held-out adapter evaluation is `0`, Sheet E/F/G remain blank, and
  General/Monolingual/publication remain blocked.
