# Pure-GDN downstream scientific audit — 2026-08-09

## Superseding prompt-contract audit — 17:15 SAST

The earlier `14/16` and provisional `15/16` trust counts are withdrawn. The
first correction removed a terminal EOS but mistakenly certified token
equality against the fallback chat template rather than equality with the
canonical pretraining contract.

Read-only inspection of `anrilombard/mzansi-text-tokenized` shows inspected
validation documents begin in `[BOS]` (`0`) and end in `[EOS]` (`1`). The
fallback chat template has no BOS, while base lm-eval forces
`add_bos_token=false`. The base tokenizer has no atomic/special system, user,
or assistant markers, and pure-GDN pretraining used plain documents rather
than SALLM chat conversations. Corrected T2X job `1210865` therefore evaluated
a missing-BOS prompt containing untrained chat-role syntax. Its artifact and
score remain provenance but are not scientific evidence of model quality.

This applies across the matrix. Fourteen lanes used chat wrapping or chat
generation; SIB and SA-general avoided chat wrapping but still omitted BOS.
The conservative state is now `16/16` operational and `0/16` scientifically
trusted. All Sheet column-D results are quarantined; E/F/G remain blank.

Fine-tuning adds/resizes/trains the chat-marker token rows, but its shared chat
template also omitted BOS. All eight Multilingual HPO grids must therefore be
rerun validation-only after the BOS-correct template is verified. Existing
News/SIB/Intent reconciliations remain preserved provenance, not frozen
winners. General retains every additional coverage/aggregation correction.

The second prospective implementation correction was frozen before corrected
metrics at SHA-256
`0f84b1f50df5d713164fc3ff62c7e9bc61c5be22505116287956f93c7ddb86d8`
(`2026-08-09-pure-gdn-prompt-contract-correction-preregistration.md`). It is
determined from tokenizer/pretraining source facts, not from held-out score
selection. No HPO or held-out adapter test is authorized until it passes
regression tests, validation-only canary, immutable deployment, and one clean
base rerun of every lane.

Implementation is locally clean at `112 passed`; focused Ruff and launcher
checks pass. The hashed dataset/tokenizer audit is
`327d10cadaa48a3718c102366b80a64ebcc7acfa4c553e045688455eda450203`.
The final read-only HEX source snapshot has `691/691` hashes at source-set
SHA-256
`5a72784950be3b09317ecf563242751da5d0d9ae79a4ebd406d4485725ac583b`
and deployment-manifest SHA-256
`9d6fb4bee2b84f4764e17c2a82b3a241e526c09ade38072343e4f7d80d8dfd9c`.

A100 prompt canary `1211314` failed before model/data access because `$SCRATCH`
was not exported; `1211315` then failed before model/data access because the
strict guard assumed an SXM SKU while `srvrocgpu010` reports
`NVIDIA A100-PCIE-40GB`. Neither produced a metric. The corrected exact-SKU
retry `1211316` started on `gpu:ampere` and verified its 691-file execution
manifest. At 17:24 SAST it was healthy with no fault marker. Running AfriHG job
`1210866` remains provenance only and is not altered or trusted by this audit.

## Conclusion

The workstream is operationally advanced but is not ready for held-out adapter
evaluation or publication. All 16 frozen base lanes produced artifacts and the
canonical Sheet hashes reconcile, but only 14 lanes are scientifically trusted.
T2X and AfriHG base results require implementation-correction reruns. The
adapter screen ran all eight families operationally, with General LR-2 still
running for provenance, but **zero of eight** recipes may pass the global
scientific freeze gate yet. News, SIB, and Intent are recoverable by a hashed
validation-only reconciliation; NER, POS, T2X, AfriHG, and General require
corrected validation-only reruns.

No held-out adapter test has run. Do not use held-out metrics to select a
recipe, checkpoint, prompt, retry, or implementation correction.

The correction design is now frozen before any corrected metric in
`2026-08-09-pure-gdn-hpo-correction-preregistration.md`, SHA-256
`a27f55cd35a48bb1d22c5ca101ec536ffb89a64229559ce6883adb6fe744a04c`.
Corrected HPO is **no-go** until its tests and validation-only Kombuys canary
pass.

## Correction-gate update — 14:05 SAST

The implementation and validation-only Kombuys canary gates now pass. This
does **not** ratify an HPO winner or authorize held-out evaluation; it clears
the code for immutable deployment to HEX and the preregistered correction
sequence.

- The complete local suite is `109 passed`; selected Ruff checks are clean.
  The exact historical fallback template is centralized in
  `src/main/sallm/chat_template.py`. Training, generation, classification,
  constrained POS, lm-eval, and evaluation-harness paths all consume that
  single contract. Regression tests compare exact token IDs, require rendered
  retokenization with `add_special_tokens=False`, and reject terminal EOS.
- Preserve Kombuys attempt 1 unchanged under
  `/scratch/alombard/sallm/results/pure_gdn_hpo_correction_canary/2026-08-09/attempt-1`.
  It failed after model loading because the constrained scorer's tokenizer had
  no chat template. This was an implementation-path failure, not an HPO
  result.
- Attempt 2 passed under
  `/scratch/alombard/sallm/results/pure_gdn_hpo_correction_canary/2026-08-09/attempt-2`.
  The execution manifest verified `690` source/config files and has SHA-256
  `6b2164891e94eeca2c4b2764837acc6df53c42ba0a43ba47922a1f77b3eb4b00`.
  The result artifact independently passes `sha256sum -c` and has SHA-256
  `2c309c9f3f2fe0110f00bfc0a78973ff8acb55be52acef691b1560208b153ad5`.
- GPU visibility was exactly one device, `NVIDIA GeForce RTX 3080 Ti`.
  Canonical model identity passed: pure `GatedDeltaNetForCausalLM`,
  `attn=None`, exactly `127,425,448` parameters, BF16 present, no PEFT adapter,
  and no merge. RTX 5090 remained at `10 MiB/0%`; both GPUs were idle after
  completion.
- NER audited all `10,760` rows (`2,152` for each P1--P5), parsed `19,255`
  spans, and preserved all four punctuation/substring fixtures. POS evaluated
  exactly 12 Tsn/Xho/Zul by P1--P4 cells under
  `closed_label_tuple_mean_logprob_v1`; its bounded canary accuracy was
  `0.0227272727`, and a correct prefix plus one extra label scored `2/3`, not
  full credit. This accuracy is implementation evidence, not recipe selection.
- General coverage is exactly `22,167`: SIB `2,970`, News `3,095`, NER
  `10,760`, POS `1,800`, AfriHG `3,082`, and T2X `460`, with both AfriHG
  language labels present (`1,305` Xhosa and `1,777` Zulu). The manual toy
  equal-family assistant-token NLL gives family NLL `1.0` and macro NLL `1.0`.
- NER/T2X/AfriHG bounded generation probes all produced `<<<<<<<<`. The canary
  explicitly preregistered that output non-emptiness or quality is not a pass
  criterion; this only proves the corrected generation path executes without
  the terminal-EOS defect.
- The correction preregistration remains byte-stable at SHA-256
  `a27f55cd35a48bb1d22c5ca101ec536ffb89a64229559ce6883adb6fe744a04c`.
  The deterministic News/SIB/Intent validation-only reconciliation is stored
  at SHA-256
  `497ad12de2d397b82132fe39cb1e0f7cb07ad14f1f3449aaf17fbf4c8ef68839`.

At this point the recovery order remains the right one. The strongest
remaining counterargument is that a passing bounded canary cannot establish
downstream quality; the repetitive generation probes reinforce that point.
Therefore it would be scientifically wrong to shrink the corrected grids or
promote a canary metric. The next gate is immutable HEX deployment, followed
by one correction rerun of base lanes 14--15 and the full validation-only
NER/POS/T2X/AfriHG/General grids.

### Provenance hardening — 14:15 SAST

The first passing correction canary was not accepted blindly. Comparing its
complete manifest with the intended local/HEX snapshot exposed 13 hashed-file
differences, including `launch_finetune.sh`,
`run_pure_gdn_validation_trial.sh`, and `disk.py`. Attempt 2 remains preserved
as a passing implementation diagnostic, but it is not the deployment
equivalence artifact.

A new clean Kombuys source snapshot was populated byte-for-byte from local and
compared as full dictionaries: `690/690` source/config hashes matched with zero
differences. Canary attempt 3 then passed against that exact snapshot. Its
artifacts are:

- source-set SHA-256:
  `5b8992861abaa6fe90904feafffc45552fef59ccaefecbaf99bd6672dc3228e8`;
- execution-manifest SHA-256:
  `26647ba8d43d9e3fb3b7d5ea13a0f7e69bcb80033c7e06097faf559a92fc7fe3`;
- result SHA-256:
  `ea386ef7a8e4fc8acb82fc09a2caaaeae4ae05bdb7ae055e08ab2b0df5ad682b`.

Attempt 3 reproduced all material attempt-2 checks and bounded values exactly:
canonical model identity, four token contracts, NER coverage/fixtures, POS
12-cell coverage and `0.0227272727` bounded accuracy, prefix-plus-extra score
`2/3`, General `22,167` coverage, and the same repetitive `<<<<<<<<`
generation probes. Both before/after GPU records show RTX 5090 at
`10 MiB/0%` and RTX 3080 Ti at `1 MiB/0%`.

The exact snapshot is now deployed read-only on HEX at
`/home/lmbanr001/masters/sallm_snapshots/pure-gdn-correction-20260809-5b899286`.
Its deployment-manifest SHA-256 is
`c408a10ab5264b81962025c96f76c907a392abb2844d09688215021c3ccae99f`.
The HEX manifest and attempt-3 manifest each contain 690 files with zero hash
differences. This closes the immutable-provenance gate. The prior HEX candidate
whose name ends in `eea08da7` is preserved but quarantined because it predates
the exact-source rerun.

### Corrected base lanes active — 14:23 SAST

After confirming no corrected summary and no active duplicate, the two
quarantined base lanes were submitted individually from a runnable copy of the
exact immutable source snapshot:

| Lane | Task | Job | Run-manifest SHA-256 | Output root |
| ---: | --- | ---: | --- | --- |
| 14 | T2X Xhosa | `1210825` | `427d4ca61681d350b2fc542c7bd1a13b6a7082f9cbd3d4d829226d005f7ef90e` | `/scratch/lmbanr001/masters/sallm/results/eval/pure_gdn_base_0shot_correction_20260809_r2_t2x_xho_r1` |
| 15 | AfriHG Xhosa/Zulu | `1210826` | `57d9f86c3b6d94fa61956e08559ebf199d219b3c9cbb007aa9bdf5d9db93c0b4` | `/scratch/lmbanr001/masters/sallm/results/eval/pure_gdn_base_0shot_correction_20260809_r2_afrihg_all_r1` |

Each run manifest verified all 690 source/config files at source-set SHA-256
`5b8992861abaa6fe90904feafffc45552fef59ccaefecbaf99bd6672dc3228e8`
and hashes the six canonical model artifacts. Slurm settings are
`nlpgroup/a100/nlpgroup`, one `gpu:ampere`, 24 hours, eight CPUs, and
`--chdir=/home/lmbanr001/masters/sallm`. The effective protocol is the frozen
canonical checkpoint, BF16, zero-shot, `peft_adapter=null`, `merge_lora=false`,
few-shot count zero, FLA backend dispatch disabled, and the unchanged
task-specific beam contracts.

At 14:23 SAST both jobs were running on `srvrocgpu010`, had loaded the
canonical local checkpoint, and had entered their held-out task data paths.
There was no traceback, OOM, token-equality/EOS contract failure, CUDA fault,
or NCCL marker. Based only on prior wall times, conditional ETAs are about
17:05 SAST for T2X and 23:55 SAST for AfriHG; auto-batch probing can move those
estimates.

Provenance-only invalid General LR-2 `1207524` remains active as the third
A100-40GB job. It reached step `8327/13640`; its displayed training ETA is
essentially equal to the remaining Slurm wall time, so terminal validation no
longer fits and a timeout near 23:29 SAST is likely. It remains unusable for
General selection regardless of terminal state.

HEX quota is home `3/10 GB` (`32.9%`) and scratch `108/300 GB` (`36.2%`).
There is no owned A100-80GB or L40S work. Kombuys is idle with RTX 5090 at
`10 MiB/0%` and RTX 3080 Ti at `1 MiB/0%`. The canonical Sheet is unchanged:
column D remains operationally populated, rows 41--43 are quarantined, and
adapter columns E/F/G are blank.

### Live correction-rerun check — 14:29 SAST

Jobs `1210825` and `1210826` remained `RUNNING` on `srvrocgpu010` after
`00:07:55`, each on one A100-40GB `gpu:ampere`. Their logs still showed the
canonical checkpoint load followed by the frozen T2X/AfriHG test data paths;
dataset expansion and filtering completed and no traceback, OOM, token/EOS
contract failure, CUDA, or NCCL marker appeared. Neither corrected
`evaluation_summary.json` existed yet, so no result audit or Sheet replacement
was possible.

Invalid provenance-only General job `1207524` remained the third owned
A100-40GB job at `14:59:16`, reaching step `8383/13640` with a displayed
training ETA of roughly nine further hours. Home/scratch quota remained
`3/10 GB` (`32.9%`) and `108/300 GB` (`36.2%`). There was no owned A100-80GB
or L40S work, and no new job or Sheet write was made.

### Shared HEX node failure — 14:39 SAST

At `2026-08-09 14:39:35 SAST`, `srvrocgpu010` became
`DOWN+NOT_RESPONDING`. Slurm marked all three colocated owned jobs
`NODE_FAIL` at exactly that timestamp: corrected T2X `1210825`, corrected
AfriHG `1210826`, and invalid provenance-only General `1207524`. The two base
jobs had run for only `00:18:26`; the last log records are automatic batch-size
selection, not model/evaluator failures. Neither corrected base
`evaluation_summary.json` exists. The General job ended at step `8445/13640`
after `15:09:47` and remains invalid provenance only; it must not be restarted
or used for selection.

This is a shared infrastructure failure independent of metrics. A single
unchanged retry of base lanes 14 and 15 is scientifically admissible only from
the same immutable source/model/protocol, using new output prefixes so the
`1210825/1210826` directories and logs remain preserved. No held-out score was
produced or consulted in making that retry decision.

After rechecking that there was no active duplicate and neither failed output
directory contained a summary, the unchanged infrastructure retries were
submitted as T2X job `1210850` and AfriHG job `1210851`. They use new `r3`
result/run prefixes, the same immutable `5b899286` source snapshot, canonical
model, BF16 zero-shot no-adapter/no-merge protocol, and the required
`nlpgroup/a100/nlpgroup`, `gpu:ampere:1`, 24-hour, 8-CPU Slurm envelope. At
14:59 SAST both were pending with
`ReqNodeNotAvail, UnavailableNodes:srvrocgpu010`; their ETA is therefore
unknown until the node recovers. No third job was submitted.

Quota remained home `3/10 GB` (`32.9%`) and scratch `108/300 GB` (`36.2%`).
No owned A100-80GB or L40S work exists. A read-only Kombuys check showed both
GPUs idle: RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`; scratch remained
`61%` used and only the pre-existing `tailscale-kombuys` tmux session was
present. Sheet state remained unchanged.

At 15:14 SAST the controller still reported `srvrocgpu010` as
`DOWN+NOT_RESPONDING`, with the only exposed reason `Not responding
[slurm@2026-08-09T14:39:35]`; four jobs on the node were marked `NODE_FAIL`
in that failure window. This supports a shared node/Slurm-daemon outage, but
does not distinguish power, network, kernel, or hardware failure. The other
A100-40GB node `srvrocgpu009` was idle but exposes the different frozen GRES
class `gpu:amperemk`; `srvrocgpu011` is A100-80GB `gpu:ampere80` and is
disallowed. Keep `1210850/1210851` queued on `gpu:ampere`. Moving to
`amperemk` would require a prospective hardware-fallback amendment and
replacement—not duplicate—jobs before any result is observed.

### Hardware fallback rejected by Slurm policy — 15:20 SAST

With explicit user approval, a prospective A100-40GB-only fallback amendment
was frozen before any replacement action at SHA-256
`1f3ab735f34a5e79a4b25321dffba12d57ae6621852833bbbcb82391d9f53708`.
A byte-identical copy was stored outside the immutable source tree under the
HEX correction artifact root. Pending jobs `1210850/1210851`, which had never
started and produced no summaries, were then cancelled to prevent duplicate
execution.

Slurm rejected the first `gpu:amperemk:1` replacement at submission with
`AssocGrpGRES`; because the submission shell was fail-fast, the second was not
attempted. No replacement job ID or `r4` output directory was created.
Read-only accounting configuration explains the rejection: account/QOS
`nlpgroup` permits `gres/gpu:ampere=4` but explicitly sets
`gres/gpu:amperemk=0`; `a100free` also sets `amperemk=0`, and `lmbanr001` has
no association with that account. Thus idle `srvrocgpu009` is not allocatable
under any authorized association. The amendment remains preserved but unused.

The clean available action is to restore replacement jobs to the original
`gpu:ampere` queue under new `r5` prefixes so they begin automatically if
`srvrocgpu010` recovers. Do not bypass Slurm policy, use A100-80GB/L40S, or
disturb Kombuys.

The original-hardware queue was restored at 15:22 SAST as T2X job `1210865`
and AfriHG job `1210866`. Both are pending on
`ReqNodeNotAvail, UnavailableNodes:srvrocgpu010` with the required
`nlpgroup/a100/nlpgroup`, `gpu:ampere:1`, 24-hour, one-node, 8-CPU allocation,
and `/home/lmbanr001/masters/sallm` working directory. They use new `r5`
output prefixes and the same immutable source/model/protocol; neither output
summary exists. ETA remains unknown until `srvrocgpu010` returns. These are
the only owned jobs, and the invalid General run was not restarted.

### Original A100-40GB node recovered — 15:49 SAST

`srvrocgpu010` returned to service and Slurm started T2X `1210865` and AfriHG
`1210866` together at `15:49:41 SAST`. At 15:56 both were healthy on one
A100-40GB `gpu:ampere` each. Their execution-manifest SHA-256 values are
`5eed5d2b7e2d46f097df3fc8474cbf722d01fd4232f05ca05ae52c127a2cccff`
and `8e4b6b71a789944a8405c1af50617210fda28c7c77ac86385f4e018cb6c6f18c`.
Reaching the evaluator proves each preceding manifest verification completed.

Both logs identify the canonical `final_model`, `peft_adapter=None`, and
`merge_lora=False`. T2X entered the frozen test path with 378 prepared
examples; AfriHG entered Xhosa test generation with 1,305 prepared examples.
Neither log contained a traceback, error, OOM, CUDA, NCCL, token-contract, or
terminal-EOS marker, and neither summary existed yet. Conditional historical
runtime ETAs are roughly 18:33 SAST for T2X and 01:23 SAST on 10 August for
AfriHG; automatic batch probing can move them.

Quota remained home `3/10 GB` (`32.9%`) and scratch `108/300 GB` (`36.2%`).
No owned A100-80GB or L40S work exists. Kombuys remained read-only and idle at
15:57: RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `61%` used.
Sheet rows 41--43 remain quarantined and adapter columns E/F/G blank.

### Corrected base T2X verified — 16:39 SAST

T2X job `1210865` completed `0:0` in `00:49:51` on one A100-40GB
`gpu:ampere`. Its execution manifest reverified and has SHA-256
`5eed5d2b7e2d46f097df3fc8474cbf722d01fd4232f05ca05ae52c127a2cccff`;
the corrected `evaluation_summary.json` SHA-256 is
`87627ec268bba04a34790f43378f0189e9302126f834f19d007815712ffdfbf7`.
The immutable six-artifact canonical model hashes are present, config remains
pure `gated_deltanet` with `attn=null`, and the runtime is canonical
`final_model`, zero-shot, `peft_adapter=None`, `merge_lora=False`, BF16 under
the frozen corrected evaluator.

Full 378-row prediction audit:

- metric: Xhosa/all chrF `1.729633871926085`; ROUGE-1/2/L and BLEU are `0.0`;
- examples SHA-256:
  `cdf2e945c1eaeebe735a5d0ed411b9fd498e4335dbf555abda9b55bd03e83c89`;
- metrics SHA-256:
  `fc849f8d87808f046c47cf0a52c35fae37bad25a67cc77d199a3f445d5321f71`;
- `378/378` outputs are nonempty and all language fields are `xho`, proving
  generation no longer stops at the erroneous terminal EOS;
- only 34 exact predictions are unique; every output contains
  `<|assistant|>`, 347 contain `<|assisted|>`, and all 378 contain zero
  alphanumeric content after control-marker stripping;
- output lengths are 1,069--1,136 characters (mean 1,119.77), dominated by
  repeated control markers, with no target-language sentence content.

This is an implementation-valid but severely degenerate zero-shot model
result. The preregistered correction executed correctly; output quality is not
an artifact-validity gate and cannot trigger another prompt, decoder, or
metric-driven rerun. Lane 14 is therefore scientifically trusted as a negative
result, moving base trust to `15/16`, but its Sheet cell remains quarantined
until corrected AfriHG completes and the joint base-closeout write is verified.

## 1. Generation evaluator defect

`src/main/sallm/evaluation/generation_metrics.py` renders the chat template
with `tokenize=False`, then retokenizes the rendered string with the tokenizer
default `add_special_tokens=True`. Direct chat-template tokenization is
equivalent to retokenization with `add_special_tokens=False`; the current path
adds BOS and, critically, EOS after `<|assistant|>`. Generation therefore
starts from an already terminated prompt.

Validation debug artifacts provide direct evidence:

- NER: every LR/epoch file is `192/192` empty.
- POS: LR-0 epoch 1 is `0/192` empty but `192/192` parse failures; every other
  LR/epoch file is `192/192` empty.
- T2X: every LR/epoch file is `64/64` empty.
- AfriHG: every LR/epoch file is `64/64` empty.

A read-only tokenizer audit on Kombuys independently confirmed that direct
chat-template tokenization exactly equals rendered-text tokenization with
`add_special_tokens=False`, while the current default adds two tokens: BOS at
the start and EOS after the assistant marker. Bounded RTX 3080 Ti A/B probes
changed NER, POS, T2X, and AfriHG generation when this was corrected: NER,
T2X, and AfriHG changed from empty to non-empty but repetitive output; POS
changed but remained invalid. RTX 5090 was untouched. Non-empty output proves
path sensitivity, not task quality.

The separate completion callback uses direct chat-template tokenization and
produced non-empty, repetitive output. This establishes evaluator-path
inconsistency rather than a model-wide inability to emit tokens.

Scientific consequence: NER, POS, T2X, and AfriHG validation screens must be
rerun in full after the implementation correction because invalid metrics
controlled early stopping and checkpoint retention. Existing winners cannot be
rescored safely from the final adapters alone.

The base T2X and AfriHG lanes also require documented implementation-correction
reruns under the otherwise frozen protocol:

- T2X: all `378` predictions begin with the same Afrikaans phrase and only
  `22` predictions are unique.
- AfriHG Xhosa/Zulu: every prediction has the same prefix, with only `15/14`
  unique predictions.

Until corrected artifacts exist, treat the base gate as operationally `16/16`
but scientifically trusted `14/16`. Quarantine canonical Sheet rows 41--43;
do not delete the original values or failure provenance.

## 2. NER parser defect

The current NER metric corrupts valid entity values in two independent ways:

- `_tags_to_spans` splits entity text on `". "` and `", "` instead of only on
  explicit entity delimiters.
- `_normalize_ner_prediction` performs substring replacement over the entire
  string rather than exact normalization of the label field.

Direct fixtures demonstrate the failure: `PER: David A. Gross` becomes
`PER: david`; `LOC: Kazan, Russia` becomes two locations;
`The Bomb Shelter Film Company` is altered by replacing `company`; and
`Stimela` is altered because `time` is replaced inside the entity name.

A full validation-reference audit found misparsing in Xhosa `8/817`, Zulu
`8/836`, and Tswana `4/499`: `20/2,152` raw references, or `100/10,760`
P1--P5 prompt-expanded examples. Current parsing yields `3,862` spans;
delimiter-faithful parsing yields `3,851`. Corrected HPO requires an explicit
label-aware `$`/`$$` parser and full-grid rerun; changing tokenization alone is
insufficient.

## 3. POS selection-contract mismatch

The free-generation POS metric gives a correct prefix plus arbitrary extra tags
full credit: gold `NOUN VERB` and prediction `NOUN VERB X` score `1.0`. More
importantly, it does not match the established cross-architecture final POS
contract.

Corrected POS HPO must use closed-label continuation-logprob scoring, the tuple
contract, mean label-token score, exactly one UPOS label per input token, token
accuracy as the primary metric, and canonical prompts P1--P4. Fixing only the
free-generation denominator would still leave selection mismatched with final
evaluation. The three-LR POS grid therefore requires a full validation-only
rerun.

## 4. Classification aggregation mismatch

`src/main/sallm/evaluation/classification_metrics.py` reports support-weighted
F1 as `all_f1`. The preregistration requires the mean of per-language macro-F1.
Validation-only W&B histories contain sufficient confusion data to recompute
the correct metric without touching held-out data. The winner is unchanged in
all three affected families:

| Family | Logged weighted F1 | Correct macro-F1 | Validation-only winner |
|---|---:|---:|---|
| News | `0.1081157355` | `0.0736133409` | job `1183134`, LR `3e-5`, earliest checkpoint |
| SIB | `0.1018246986` | `0.0576036866` | job `1189730`, checkpoint `526` |
| Intent | `0.0025551335` | `0.0025019580` | job `1192267`, LR `8e-5`, checkpoint `2643` |

W&B run IDs used for the validation-only reconciliation:

- News: `ldtnnu63`, `d46ssqps`, `y4go6aeh`.
- SIB: `86xxffgr`, `v2b6kqxe`, `web3ll2q`.
- Intent: `t7smhrtn`, `r9qqor27`, `ug1czuc8`.

These three winners are recoverable without retraining, but they must not be
ratified until the recomputation, inputs, aggregation rule, and output hash are
stored in a dated reconciliation artifact.

The preregistered early-stopping threshold remains `0.001`. Intent LR-2's
epoch-2 corrected macro-F1 improvement was smaller than that threshold, so it
correctly did not reset patience under the frozen rule. Changing the threshold
to strict zero now would be post-hoc and would require rerunning the affected
Intent grid; that change is not part of this correction.

## 5. General validation coverage and objective defects

HEX used an older `afrihg.py` that does not attach `lang` when Xhosa and Zulu
are loaded together. `CustomSFTTrainer` excludes rows whose language is null.
The General validation job therefore declared `12,209` examples but evaluated
only `9,127`; the missing `3,082` rows are exactly the complete AfriHG
component.

Consequently, these are not valid six-family General validation losses:

- LR-0 job `1204261`: `13.603466245839952`.
- LR-1 job `1204262`: `13.732546259124712`.
- LR-2 job `1207524`: provisional best `13.511420388596335` at
  `checkpoint-5456`.

General must be rerun validation-only after preserving language labels, adding
an explicit `task_name`, and asserting coverage. Current processed composition
is SIB `2,970`, News `3,095`, NER `2,152`, POS `450`, AfriHG `3,082`, and T2X
`460`, total `12,209`. In addition to omitting AfriHG, `CustomSFTTrainer`
multiplies each batch mean loss by batch size, so the result is sample/batch
weighted rather than assistant-token weighted.

The frozen replacement selection view expands NER over P1--P5 and POS over the
canonical P1--P4, for exact processed coverage `22,167`. It computes summed
valid assistant-token NLL divided by valid assistant-token count within each
family, then takes the arithmetic mean of the six family NLL values. This is
equal-family selection with token weighting inside family and matches the
intent of the token-balanced training mixture. Counts, summed NLL, and
per-family values must be persisted at every checkpoint; missing coverage is a
hard failure.

## 6. Execution provenance defect

The HEX working copy has no valid Git `HEAD` and zero tracked files. Critical
source drift was observed:

- `afrihg.py`: local SHA begins `c3784f`; HEX SHA begins `44af40`.
- `classification_metrics.py`: local SHA begins `a72630`; HEX SHA begins
  `906aca`.

Future correction runs require either an immutable commit deployed to HEX or a
complete imported-source, launcher, config, and environment hash manifest.
Hashing only the top-level runner is insufficient.

## 7. Required correction sequence

1. Implement and regression-test generation tokenization, NER parsing, POS
   constrained selection, true macro-F1, General coverage/token aggregation,
   and provenance.
2. Pass a validation-only Kombuys canary on RTX 3080 Ti; leave RTX 5090
   untouched.
3. Deploy an immutable code snapshot or complete execution manifest to HEX.
4. Rerun base lanes 14--15 once under the otherwise frozen base protocol.
5. Write and hash the validation-only News/SIB/Intent reconciliation.
6. Rerun the NER, POS, T2X, AfriHG, and General validation grids.
7. Freeze all eight valid winners in a manifest, then train/select Monolingual
   models on validation only.
8. Touch each applicable Mono/Multi/General held-out test once and populate
   Sheet columns E/F/G only from verified artifacts.

## Live operational state at 13:09 SAST

- General LR-0 `1204261`: completed `0:0`, retained `checkpoint-2728`.
- General LR-1 `1204262`: completed `0:0`, retained `checkpoint-2728`.
- General LR-2 `1207524`: running at step `7693/13640` on `srvrocgpu010`
  A100-40GB `gpu:ampere`; finite loss/gradient and zero runtime fault markers.
  Retained `checkpoint-5456`; next invalid/provenance-only validation is
  estimated near `14:10--14:20 SAST`. Preserve the run, but do not ratify its
  result as a six-family General selection.
- HEX quota: home `3/10 GB` (`32.6%`), scratch `108/300 GB` (`36.2%`).
- GPU-family state: one owned A100-40GB job; no owned A100-80GB or L40S job.
- Kombuys diagnostics are complete and both GPUs are now idle: RTX 5090
  `10 MiB`, `0%`; RTX 3080 Ti `1 MiB`, `0%`; scratch `61%`; only
  `tailscale-kombuys` tmux. RTX 5090 was not used.
- Sheet: base column D is populated, adapter columns E/F/G remain blank. No
  Sheet write was made during the audit; rows 41--43 require quarantine.
- Hugging Face publication remains blocked.

## Prompt-contract canary closeout and corrected base restart — 18:14 SAST

- A100 prompt-contract canary `1211316` completed `0:0` in `00:09:14` on
  `srvrocgpu010`, one A100-40GB `gpu:ampere`. `canary_result.json` SHA-256 is
  `79beda323ddab1d8c14dc755756ac4f7d607e2f324439bbf7287dd00081ca20e`;
  execution-manifest SHA-256 is
  `6104df2251b5f6f09d361a189405cd466c09bf7a91851c4fe9c5c813a40163b3`.
  It passed canonical model identity, exactly one BOS/no terminal EOS, raw
  base prompts without chat markers, corrected NER/POS, and exact General
  coverage `22,167`. The canary loaded no held-out split and selected no
  metric or prompt.
- Immutable execution exposed three orchestration assumptions before any
  model/data access: the generic launcher tried to create `.venv` inside the
  read-only snapshot; the batch launcher resolved a relative script from the
  mandated home working directory; then Hydra tried to write bookkeeping
  under the read-only source tree. Root correction separates immutable source
  from mutable runtime/working directory and uses an explicit snapshot batch
  path. Local suite remains `112 passed`; focused Ruff, shell syntax, and
  out-of-tree dry-run checks pass.
- Preserve failed jobs `1211606/1211607`, `1211637/1211638`,
  `1211649/1211650`, and `1211661`. All failed before model/data evaluation,
  produced no metric and no `evaluation_summary.json`, so recovery is
  implementation-driven rather than held-out-metric-driven.
- Final read-only execution snapshot is
  `/home/lmbanr001/masters/sallm_snapshots/pure-gdn-prompt-correction-20260809-693fe42c`.
  All `691/691` source/config hashes match local; deployment-manifest SHA-256
  is `0396a24cb9cd4f83e80406ec06fbf5d4f2532a245f3b2cc09f85acd3eacbbbb1`.
- Corrected base lane 0 MasakhaNews `1211671_0` and lane 3 SIB-200 `1211675_3`
  are running healthy under unique prefix
  `pure_gdn_base_0shot_bosraw_20260809r5`. Both use canonical `final_model`,
  BF16, zero-shot, `peft_adapter=None`, `merge_lora=False`, raw lm-eval,
  `apply_chat_template=False`, and `add_bos_token=True`. Both entered real
  loglikelihood evaluation without fault markers; no summary exists yet.
- Old AfriHG `1210866` remains healthy on Xhosa generation but is provenance
  only because it used the superseded chat/no-BOS contract. At this pass the
  owned state is exactly three A100-40GB jobs; no A100-80GB or L40S work.
  Quota is home `33.1%`, scratch `36.3%`. Historical runtime suggests lane 3
  should finish near 18:18 SAST and lane 0 near 18:21 SAST; AfriHG provenance
  remains roughly 01:23 SAST on 10 August, subject to generation batching.
- Kombuys remains read-only and idle: RTX 5090 `10 MiB/0%`, RTX 3080 Ti
  `1 MiB/0%`, scratch `61%`, only `tailscale-kombuys` tmux. The Sheet was not
  changed; all column-D values remain quarantined and E/F/G remain blank.
  Scientifically trusted status remains base `0/16`, frozen HPO winners `0/8`,
  held-out adapter tests `0`.

## First corrected base artifacts verified — 18:32 SAST

- Lane 3 SIB-200 job `1211675_3` completed `0:0` in `00:07:39` on one
  A100-40GB `gpu:ampere`. Summary SHA-256 is
  `e16b70d127d44ce812b4596a37a9791b70bcec47f7ea0279e2ff75f84cc485b6`;
  raw results SHA-256 is
  `80ce8d8afe6386d960c6933d9016aa4a08b489886508c8ae73403da663b40c0c`.
- Lane 0 MasakhaNews job `1211671_0` completed `0:0` in `00:11:28` on the
  same GPU family. Summary SHA-256 is
  `52679f0d7c021324c29fed2f56b161de9ccb5cf8d770d21f7e93c3f6a473f2c3`;
  raw results SHA-256 is
  `b7ac32a5af44450b0ecc1ef0cec7c1447e553df2960c9d228c5898bf0c3a22a3`.
- Both artifacts passed canonical model, BF16, zero-shot, no-adapter/no-merge,
  raw prompt, `apply_chat_template=false`, `add_bos_token=true`, task-count,
  summary/result hash, and fault-marker checks. They are the first two
  scientifically trusted corrected base lanes. Trusted base progress is now
  `2/16`; frozen HPO winners remain `0/8`; no adapter held-out test has run.
- After exact live reads, canonical Sheet rows 2--3 and 10--15 were updated
  from these artifacts only. A post-write read verified `9 Aug` dates,
  headline values, notes containing all prompt values/means/ranges, job IDs,
  artifact paths and hashes, immutable snapshot/deployment hash, descriptive
  best-prompt warning, and preserved cell/date formatting. All other base
  rows remain quarantined and adapter columns E/F/G remain blank.
- Corrected lane 1 MasakhaNER job `1211758_1` and lane 2 MasakhaPOS job
  `1211759_2` started at 18:30:50 SAST on A100-40GB. Both log canonical
  `final_model`, BF16, zero-shot, `peft_adapter=None`, `merge_lora=False`,
  `apply_chat_template=False`, `add_bos_token=True`, and no fault markers.
  Historical runtimes place POS near 21:37 SAST and NER near 00:53 SAST on
  10 August, subject to corrected-token throughput.
- Provenance-only AfriHG `1210866` remains running unchanged; total owned
  state is exactly three A100-40GB jobs, no A100-80GB or L40S. HEX quota is
  home `33.1%`, scratch `36.4%`. Kombuys is read-only and idle: RTX 5090
  `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `61%`, only the Tailscale tmux.

## Corrected base monitoring — 18:59 SAST

- Corrected MasakhaNER `1211758_1` and MasakhaPOS `1211759_2` remain healthy
  on `srvrocgpu010`, one A100-40GB `gpu:ampere` each. Fresh evaluator progress
  was `1057/14980` and `1025/7216`; projected remaining runtimes were about
  `5h53m` and `2h42m`. Neither job has a fault marker or summary yet.
- Superseded AfriHG `1210866` remains `RUNNING` on the third A100-40GB and is
  provenance-only. Its metric must never enter the corrected base gate.
- Owned work is exactly three A100-40GB jobs, with no A100-80GB or L40S
  overlap. Quota is home `33.1%`, scratch `36.4%`. Trusted progress remains
  base `2/16`, frozen Multilingual winners `0/8`, held-out adapter tests `0`.
- The degenerate earlier T2X result is not evidence to alter the corrected
  contract: it came from the superseded chat/no-BOS path. The raw-prompt,
  exactly-one-BOS contract passed canary `1211316`; HPO remains blocked until
  all 16 corrected base lanes are verified.

## Corrected base monitoring — 19:27 SAST

- Corrected NER `1211758_1` advanced to `2193/14980` with about `5h20m`
  remaining; corrected POS `1211759_2` advanced to `2137/7216` with about
  `2h06m` remaining. Both logs are fresh and have no fault marker or summary.
- Superseded AfriHG `1210866` remains provenance-only but active: it completed
  Xhosa with chrF `1.8003387196281417` and entered Zulu generation. This metric
  is quarantined and cannot influence the corrected gate or HPO.
- All three owned jobs remain on `srvrocgpu010` A100-40GB `gpu:ampere`; there
  is no A100-80GB or L40S overlap. Quota is home `33.5%`, scratch `36.4%`.
  Trusted state remains base `2/16`, Multilingual winners `0/8`, held-out
  adapter tests `0`; Sheet E/F/G remain blank.

## Corrected base monitoring — 19:57 SAST

- Corrected NER `1211758_1` is healthy at `3377/14980` with about `4h53m`
  remaining; corrected POS `1211759_2` is healthy at `3329/7216` with about
  `1h38m` remaining. Both logs are fresh, with no fault marker or summary.
- Provenance-only AfriHG `1210866` is still advancing through Zulu generation;
  its latest log event was a context-limit truncation warning at `19:52:02`,
  not a runtime fault. Its result remains scientifically quarantined.
- Exactly three owned A100-40GB `gpu:ampere` jobs remain on `srvrocgpu010`, so
  no corrected lane slot is open. No owned A100-80GB or L40S work exists.
  Quota remains home `33.5%`, scratch `36.4%`; trusted state remains `2/16`,
  HPO `0/8`, held-out adapter tests `0`, with Sheet E/F/G blank.

## Corrected POS verified; lanes 4--5 started — 22:02 SAST

- Corrected MasakhaPOS `1211759_2` completed `0:0` in `03:03:56` on
  A100-40GB `gpu:ampere`. Summary SHA-256 is
  `c7e512d1c873435fa3c230c47775f94df3a915732605c65ddacca0ee934c4daf`;
  raw results SHA-256 is
  `2b2ea84c3c108b8118d283de1ef910a945433680485e5c46f33e350f934aee33`.
  It passed canonical checkpoint, `127,425,448` parameters, BF16, zero-shot,
  no adapter/merge, raw prompt, `apply_chat_template=false`,
  `add_bos_token=true`, 12-task coverage, and fault-marker checks.
- Xhosa, Zulu, and Tswana P1--P4 token accuracies are all exactly `0.0`.
  This is a verified negative base result under the corrected contract, not a
  prompt-selection or retry trigger. Trusted corrected-base progress is now
  `3/16`; HPO remains `0/8` and held-out adapter tests remain `0`.
- Canonical Sheet rows 7--9 were reread, updated, and reread again. Dates are
  `9 Aug`; values retain the established token-accuracy headline; notes carry
  all prompt values, mean/range, job, paths, hashes, model/protocol identity,
  immutable deployment hash, and descriptive-best-prompt warning. Formatting
  is preserved and E/F/G remain blank.
- Superseded AfriHG `1210866` completed cleanly `0:0` in `05:51:20`, but its
  chat/no-BOS result remains provenance-only and scientifically quarantined.
- First lane-4/5 submissions `1213980_4/1213981_5` failed in one second before
  importing Hydra, model, or data because the mutable runtime-repo override
  was omitted and the read-only snapshot correctly rejected `.venv` creation.
  They produced no summary or metric and are preserved as infrastructure
  provenance.
- Recovery jobs `1213986_4` Intent and `1213987_5` SA-general started at
  `21:59:55` using the immutable source snapshot plus the established mutable
  runtime repo. Both log canonical checkpoint, BF16, zero-shot, no
  adapter/merge, `apply_chat_template=false`, and `add_bos_token=true`.
  Intent entered loglikelihood evaluation; SA-general initialized its first
  frozen pack without fault markers.
- Corrected NER `1211758_1` remains healthy, last observed at `8201/14980`
  with about `2h46m` remaining. Exactly three owned A100-40GB jobs are active,
  no A100-80GB or L40S work exists, and quota is home `33.5%`, scratch `36.4%`.
  Kombuys remains at its last verified read-only idle state.

## Corrected Intent verified; Belebele lane 6 queued — 23:00 SAST

- Corrected InjonGoIntent `1213986_4` completed `0:0` in `00:52:30` on
  A100-40GB `gpu:ampere`. Summary SHA-256 is
  `168cec2f9d755378edf08f62a31ab01f3d37d824c4a689cac76efce19f4b6b28`;
  raw results SHA-256 is
  `da138f24d5394f5a3e20420aa45378e5a7f2daf37f570557f31c6103e0a27569`.
  The artifact passed canonical checkpoint, `127,425,448` parameters, BF16,
  zero-shot, no adapter/merge, raw prompt, `apply_chat_template=false`,
  `add_bos_token=true`, exact 20-task coverage, and fault-marker checks.
- All P1--P5 prompts tie within language: English F1
  `0.001290205525708353`; Xhosa, Zulu, and Southern Sotho F1 each
  `0.0012195121951219512`. The descriptive tie did not change the frozen
  protocol. Trusted corrected-base progress is now `4/16`; HPO remains `0/8`
  and held-out adapter tests remain `0`.
- Canonical Sheet rows 16--19 were reread, updated, and verified with `9 Aug`
  dates, established F1 headlines, complete prompt values/means/ranges, exact
  job/artifact hashes, immutable model/protocol provenance, preserved
  formatting, and blank E/F/G cells.
- After checking that no corrected summary or active duplicate existed,
  corrected lane 6 Belebele-Afrikaans was submitted as `1214451_6` from the
  immutable snapshot with the established mutable runtime and frozen base
  overrides. It is resource-pending; no model or data access has begun.
- Corrected NER `1211758_1` remains healthy, last observed at `10593/14980`
  with about `1h49m` remaining. SA-general `1213987_5` is healthy at
  `2089/5000` in its first generation pack with about `1h15m` remaining for
  that pack. Both are on `srvrocgpu010` A100-40GB. No A100-80GB or L40S work
  exists; quota is home `33.5%`, scratch `36.6%`. Kombuys remains at its last
  verified read-only idle state.
