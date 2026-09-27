# Advisor Meeting Notes: Mamba Classification, Generation Gap, and Base-Model Decision

Date: 2026-05-21

## One-Sentence Summary

Mamba is not broadly broken: after fixing evaluation/task-template confounds it
performs strongly on MasakhaNews classification, but generation remains far
behind LLaMA because Mamba under-generates and has much worse matched base and
generation-task likelihood; after ruling out several implementation, decoding,
checkpoint-selection, and fine-tuning confounds, the next defensible step is
pure-Mamba2 base recipe/shape optimization.

## What We Need Advice On

- Are the validation controls sufficient to justify treating base-model quality
  as the main remaining root cause for the generation gap?
- Should the next base experiment stay pure Mamba2, or should we already include
  a hybrid Mamba/attention/MLP architecture as a separate comparison point?
- What is the minimum additional evidence needed before presenting the current
  Mamba generation gap as a base/pretraining issue rather than an evaluation or
  downstream fine-tuning issue?

## Current Position

We should avoid the simple claim "Mamba is bad." The evidence is more nuanced:

- Mamba can perform well on classification when the task/evaluation template is
  correct.
- Mamba generation can be improved by recipe recovery and checkpoint selection,
  so the early weak results were not final or fully optimized.
- Even after those fixes, Mamba remains much weaker than LLaMA on generation.
- The remaining gap is supported by matched likelihood audits, output-shape
  forensics, and negative continuation/recovery gates.

The current working hypothesis is:

> Mamba's downstream generation weakness is largely caused by weaker base-model
> quality and/or base pretraining recipe/shape, with an additional generation
> dynamics issue around length, coverage, and source-conditioned copying.

## How We Made Mamba Perform Well on Classification

### Problem

Early MasakhaNews mono-vs-multilingual results looked alarming. Monolingual
Mamba variants appeared weaker than expected, which could have implied that the
architecture or fine-tuning recipe was fundamentally poor even on
classification.

### What Was Actually Wrong

The News comparison had evaluation/task-template confounds:

- The monolingual task pack and multilingual task pack were not using the same
  chat-template path.
- Built-in MasakhaNews task definitions expected a `headline_text` field, while
  the local data/task path used `headline`/`text`.
- This made the comparison unfair: the model setups were not being evaluated
  through the same prompt/schema path.

### Fix

- Built local test-split task packs with the correct local schema.
- Put monolingual and multilingual/classification comparisons through the same
  chat-template path.
- Updated dependency hygiene around `lm-eval>=0.4.11,<0.5`.
- Re-ran the monolingual parity checks after task/schema/template correction.

### Classification Result After Fix

Best current Mamba monolingual parity results:

| Task | Setup | Metric | Score | Interpretation |
|---|---|---|---:|---|
| MasakhaNews Eng | Mamba monolingual adapter, all-chat parity | best F1 | `0.773` | strong |
| MasakhaNews Xho | Mamba monolingual adapter, all-chat parity | best F1 | `0.757` | strong |

Diagnostic means retained in the vault:

| Setup | Mean F1 | Mean accuracy | Cross-language mean F1 |
|---|---:|---:|---:|
| Eng mono adapter on all-chat pack | `0.7155` | `0.7188` | Xho `0.3964` |
| Xho mono adapter on all-chat pack | `0.6923` | `0.7091` | Eng `0.6362` |

### Classification Interpretation

- The scary "mono < multi/general" pattern was mostly a measurement/template
  bug for News.
- News is the strongest evidence that Mamba can fine-tune successfully in this
  project.
- This is important because it separates "Mamba cannot learn downstream tasks"
  from "Mamba struggles on generation tasks."

## Why Generation Is Still Bad

### Generation Tasks Checked

Main generation tasks investigated:

- T2X Xho translation-style generation.
- AfriHG Xho headline generation.
- AfriHG Zul headline generation.

### LLaMA Reference Gap

Even after Mamba recipe recovery and checkpoint selection, LLaMA remains far
ahead on generation:

| Task | Current best Mamba | LLaMA reference | Gap |
|---|---:|---:|---:|
| T2X Xho chrF | `33.1243` | about `53.98` | large |
| AfriHG Xho chrF | `11.5335` | about `20.22` | large |
| AfriHG Zul chrF | `13.5726` | about `23.00` | large |

Important nuance:

- Mamba T2X improved from weak reproductions around chrF `20` to `32+` after
  recipe recovery and checkpoint selection.
- AfriHG also improved over earlier weak Mamba runs.
- Therefore the final statement should be "Mamba generation is still behind
  LLaMA after meaningful recovery," not "Mamba was never optimized."

## Validation Steps and Evidence

### 1. Checkpoint Selection by Generation Metric

Problem:

- Some Mamba generation runs could not rely on trainer-side generation metrics
  because of cache/generation-path issues.
- Selecting final checkpoints or lowest eval-loss checkpoints was not reliable
  for generation quality.

Action:

- Evaluated saved checkpoints by validation generation metrics.
- Promoted validation-selected checkpoints to official test reruns.

Official test improvements:

| Task | Selected checkpoint | chrF | Prior Mamba best | Delta |
|---|---|---:|---:|---:|
| T2X Xho | `checkpoint-244` | `32.6125` | `32.1797` | `+0.4328` |
| AfriHG Xho | `checkpoint-656` | `11.5335` | `9.8976` | `+1.6359` |
| AfriHG Zul | `checkpoint-892` | `13.5726` | `12.0901` | `+1.4825` |

Interpretation:

- Checkpoint selection matters for Mamba generation.
- Eval loss/final checkpoint selection left performance on the table.
- This improved Mamba but did not close the LLaMA gap.

### 2. Longer Downstream Continuation

T2X continuation:

| Run | chrF | BLEU | ROUGE-L |
|---|---:|---:|---:|
| T2X checkpoint-selected `checkpoint-244` | `32.6125` | `0.0580` | `0.2966` |
| T2X `checkpoint-244` continue2 | `33.1243` | `0.0577` | `0.3015` |

Interpretation:

- Longer downstream training can still improve T2X slightly.
- The improvement did not fix output-shape problems.
- Mamba remained much shorter than the references and LLaMA.

AfriHG continuation:

| Task | Checkpoint-selected | Continue2 | Outcome |
|---|---:|---:|---|
| AfriHG Xho chrF | `11.5335` | `10.3470` | worse |
| AfriHG Zul chrF | `13.5726` | `12.6551` | worse |

Interpretation:

- Blind longer downstream continuation is not the general answer.
- For AfriHG, continuation shortened outputs and reduced chrF.

### 3. Output-Shape Forensics

We compared local generated examples and summary statistics.

| Task/model | Empty rate | Mean pred tokens | Mean ref tokens | Length ratio | Repeated bigram rate | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| Mamba T2X ckpt244 | `0.000` | `10.24` | `27.37` | `0.478` | `0.065` | too short, more repetition |
| LLaMA T2X best | `0.000` | `12.74` | `27.37` | `0.661` | `0.035` | still short, but better |
| Mamba AfriHG Xho ckpt656 | `0.006` | `2.70` | `5.56` | `0.527` | `0.0005` | too short |
| LLaMA AfriHG Xho best | `0.000` | `6.04` | `5.56` | `1.253` | `0.010` | close/slightly long |
| Mamba AfriHG Zul ckpt892 | `0.005` | `3.44` | `5.41` | `0.685` | `0.0058` | too short |
| LLaMA AfriHG Zul best | `0.000` | `6.12` | `5.41` | `1.223` | `0.030` | close/slightly long |

Interpretation:

- The generation gap is not just metric noise.
- Mamba under-generates relative to references and LLaMA.
- T2X also has more repetition for Mamba.
- This points to coverage/length/source-conditioning issues.

Meeting TODOs:

- [ ] Manually inspect AfriHG and T2X generated examples for coherence, not only
  metric scores: check whether outputs are semantically plausible, truncated,
  hallucinated, copied from the source, or locally fluent but incomplete.
- [ ] Add a minimal Mamba copy/seq-to-seq diagnostic task to test whether the
  model can copy an input sequence into the output reliably. Use this to
  separate a broad source-conditioned seq-to-seq weakness from task-specific
  AfriHG/T2X failures.
- [ ] Stress-test long outputs: force or request longer generations and check
  whether Mamba consistently degrades into repetition. Track the point where
  overgeneration begins, and compare against LLaMA under the same decoding
  settings.

### 4. Decoder-Only Output-Shape Rescue

We tested whether decoding controls could rescue Mamba without changing the
architecture:

- stronger length penalties,
- `min_new_tokens`,
- larger beams,
- sampling variants.

T2X validation:

| Decode setting | chrF | Mean pred tokens | Length ratio | Repeated bigram rate |
|---|---:|---:|---:|---:|
| beam3/lp1.2 | `31.9386` | `13.15` | `1.30` | `0.0754` |
| beam5/lp1.6 | `21.4976` | `96.41` | `9.82` | `0.3426` |
| beam5/lp1.6/min24 | `18.2950` | `131.32` | `13.71` | `0.4782` |
| sample/t0.8/top-p0.95/min24 | `26.6023` | `20.05` | `2.23` | `0.0242` |

T2X interpretation:

- Stronger length controls caused severe overgeneration and repetition.
- The T2X gap is not solved by a simple length knob.

AfriHG Xho validation:

| Decode setting | chrF | ROUGE-L | Empty rate | Length ratio |
|---|---:|---:|---:|---:|
| beam5/lp0.7 | `10.9517` | `0.0456` | `0.0625` | `0.49` |
| beam5/lp1.2 | `12.8140` | `0.0434` | `0.0625` | `0.75` |
| sample/t0.8/top-p0.95/min6 | `12.8566` | `0.0227` | `0.1250` | `0.83` |

AfriHG Zul validation:

| Decode setting | chrF | ROUGE-L | Empty rate | Length ratio |
|---|---:|---:|---:|---:|
| beam5/lp0.7 | `14.2068` | `0.0659` | `0.0000` | `0.74` |
| beam5/lp1.2 | `17.0020` | `0.0839` | `0.0000` | `1.03` |
| sample/t0.8/top-p0.95/min6 | `15.0534` | `0.0360` | `0.0000` | `0.88` |

Decoder-rescue interpretation:

- AfriHG benefits somewhat from beam5/lp1.2, especially Zul.
- T2X does not benefit; stronger length control hurts badly.
- Decoding explains part of the AfriHG output-shape problem but not the full
  generation gap.

### 5. Matched Base and Fine-Tuned Likelihood Audits

This is the strongest evidence that base quality matters.

We computed teacher-forced target likelihood/perplexity on the same validation
prompts and gold targets for Mamba and LLaMA, using the same tokenizer and the
same target token burden.

Clean base generation-loss audit:

| Task | Mamba base PPL | LLaMA base PPL | Ratio |
|---|---:|---:|---:|
| T2X Xho | `347.81` | `64.20` | `5.42x` |
| AfriHG Xho | `438.73` | `205.48` | `2.14x` |
| AfriHG Zul | `606.04` | `267.32` | `2.27x` |

Clean fine-tuned generation-loss audit:

| Task | Mamba selected checkpoint PPL | LLaMA task-specific PPL | LLaMA SA-general PPL | Interpretation |
|---|---:|---:|---:|---|
| T2X Xho | `18.73` | `5.31` | `14.77` | Mamba improves a lot but remains behind. |
| AfriHG Xho | `21.04` | `13.28` | `21.24` | Mamba roughly matches SA-general LLaMA, not task-specific LLaMA. |
| AfriHG Zul | `19.28` | `10.79` | `18.48` | Mamba close to SA-general LLaMA, still behind task-specific LLaMA. |

Held-out pretraining-style loss audit:

| Model | Weighted NLL | PPL | Validation tokens |
|---|---:|---:|---:|
| Mamba base | `3.8066` | `45.00` | `302278` |
| LLaMA base | `2.3011` | `9.98` | `302278` |

Interpretation:

- The tokenizer is shared, so this is not a tokenizer mismatch.
- Fine-tuning is not fundamentally broken: Mamba moves from base PPL hundreds
  to fine-tuned PPL around `19-21`.
- But Mamba starts from a much weaker base and remains behind task-specific
  LLaMA after fine-tuning.
- This supports base checkpoint quality/training recipe as the main remaining
  root cause.

## Why We Decided to Go Back to Base-Model Training

The decision was not immediate. We first ruled out or reduced several alternative
explanations:

1. **Task/evaluation bug?**
   - Yes for News classification; fixed.
   - After fixing, News became strong.
   - Therefore poor generation is not simply because the whole evaluation stack
     is broken.

2. **Bad checkpoint selection?**
   - Yes, partly.
   - Selecting checkpoints by validation generation metric improved T2X and
     AfriHG official test results.
   - But the LLaMA gap remained large.

3. **Not enough downstream training?**
   - Partly for T2X: continuation improved chrF from `32.61` to `33.12`.
   - Negative for AfriHG: continuation reduced chrF and shortened outputs.

4. **Decoding/output-shape issue?**
   - Partly for AfriHG, especially Zul.
   - Not enough for T2X, where length controls caused overgeneration/repetition.

5. **Implementation issue?**
   - Several implementation-control issues were found and repaired:
     - missing fallback chat template in loss audit,
     - Hydra `peft_adapter` override hygiene,
     - assistant-mask fallback in clean loss audit,
     - Mamba cache output hygiene in continued-pretraining eval,
     - tied-embedding safetensors save issue.
   - After repairs, successful artifact-producing runs still showed the same
     substantive pattern: Mamba base likelihood is much weaker.

6. **Tokenizer mismatch?**
   - Ruled out as the main between-model explanation because Mamba and LLaMA
     use the same tokenizer in the matched audits.
   - Tokenizer fertility remains a useful task difficulty variable, but not a
     mismatch.

After these checks, the remaining explanation with the strongest evidence is:

> The current Mamba base checkpoint is weaker than the LLaMA base checkpoint,
> and downstream generation tasks expose that weakness more strongly than
> classification.

## Base-Model Experiments Run So Far

### Continued Pretraining Recovery

Purpose:

- Test whether a cheap continued-pretraining pass could improve the existing
  Mamba base without committing to a full fresh base retrain.

Result:

| LR | Held-out eval loss | Clean generation-loss outcome |
|---:|---:|---|
| `5e-5` | `3.8625` | did not improve over starting Mamba base |
| `1e-4` | `3.8635` | nearly tied on pretraining loss, no downstream rescue |
| `2e-4` | `3.8902` | worse |

Clean generation-loss comparison for best LR `5e-5`:

| Task | Continued Mamba PPL | Starting Mamba base PPL |
|---|---:|---:|
| T2X Xho | `351.80` | `347.81` |
| AfriHG Xho | `446.38` | `438.73` |
| AfriHG Zul | `618.10` | `606.04` |

Conclusion:

- Cheap 300-step continued pretraining does not rescue the base.
- The next serious step should be fresh-base recipe/shape HPO rather than
  longer blind continuation.

### Fresh 3k Probe

Purpose:

- Test whether a fresh-from-config pure-Mamba2 run has a promising learning
  direction and whether LR `4e-4` or `2e-4` is better early.

Result:

| Run | Best eval loss | Train loss |
|---|---:|---:|
| LR `2e-4`, 3k steps | `7.2619` | `7.2631` |
| LR `4e-4`, 3k steps | `6.3838` | `6.6181` |

Conclusion:

- LR `4e-4` learns faster in the short probe.
- But 3k fresh-from-random checkpoints are far from usable as base replacements.

### Fresh 20k Pilot: Current Shape

Shape:

- `hidden_size=512`
- `num_hidden_layers=27`
- `expand=4`
- `state_size=64`
- about `126.4M` parameters

Result:

| Step | Eval loss |
|---:|---:|
| `5000` | `5.9383` |
| `10000` | `6.0468` |
| `15000` | `6.1576` |
| `20000` | `6.1647` |

Clean generation-loss remained orders of magnitude worse than the current Mamba
base and LLaMA base.

Conclusion:

- This exact fresh recipe is negative.
- The best point was already at 5k, then it worsened.
- Simply extending this exact run is not promising.

### Active Fresh 20k Pilot: Wide Pure-Mamba2 Shape

Reason:

- Current shape may allocate parameters poorly: narrow/deep, high expansion,
  smaller state.
- Public Mamba2-like configurations suggest wider/shallower, lower expansion,
  larger state may be worth testing.

New shape:

- `hidden_size=704`
- `num_hidden_layers=24`
- `expand=2`
- `state_size=128`
- `num_heads=22`
- `n_groups=2`
- `126,811,888` trainable parameters

Status:

- Submitted as jobs `852296` -> `852297`.
- One L40S GPU.
- Heartbeat: `sallm-mamba-wide-shape-gate-watch`.
- At the latest successful check, job `852296` was still running and no
  checkpoint had landed yet.
- Recent heartbeat checks had intermittent HEX/VPN timeouts, so state should be
  refreshed once connectivity is stable.

Decision rule:

- If this wide shape beats the failed current-shape best eval loss `5.9383` and
  improves clean generation-loss, it becomes the next pure-Mamba base candidate.
- If it is also weak, then simple shape reallocation is not enough; the next
  options are fuller pure-Mamba HPO, deeper optimizer/data audit, or an advisor
  decision on whether to introduce hybrid Mamba as a separate architecture.

## POS and NER Side Note

POS/NER remain important but are not the current main root-cause gate.

Current understanding:

- POS/NER failures are dominated by exact output-shape control under
  free-generation scoring.
- Mamba shows signs of teacher-forced learning, but autonomous generation often
  fails to produce exactly parseable, length-matched tag sequences.

Meeting TODO:

- [ ] For POS and NER, add an English-data control: get English POS/NER data and
  check whether Mamba still struggles when low-resource language/data scarcity is
  less confounded. If it still fails, the issue is more likely the tagging
  formulation, exact output-shape constraint, or model/decoder behavior rather
  than only the African-language data setting.
- [ ] For POS and NER, inspect generated outputs directly after the English-data
  control and any decoding/output-shape rescue: determine whether the outputs
  are coherent tag sequences, partially coherent but misaligned, or malformed
  free text.

Possible decoder-only fixes:

- constrained decoding over tag vocabulary,
- dynamic length control based on input tokens,
- per-token decoder-only loglikelihood tagging,
- delimiter-stable tag language,
- curriculum over short to long sequences.

Why paused:

- Generation/base-quality audits are currently more decisive for the
  Mamba-vs-LLaMA architecture comparison.
- Once base quality is less confounded, POS/NER output-shape rescue should be
  revisited with matched Mamba and LLaMA controls.

## Advisor-Facing Interpretation

The strongest defensible story is:

1. We discovered that some early Mamba weakness was not real; it came from
   task-template/schema mismatches, especially for MasakhaNews classification.
2. After fixing those issues, Mamba performs well on classification, proving
   downstream Mamba fine-tuning can work.
3. Generation remains substantially weaker than LLaMA even after recipe
   recovery, checkpoint selection, continuation, output forensics, and decoding
   diagnostics.
4. Clean teacher-forced likelihood audits show the Mamba base is much weaker
   than the LLaMA base under the same tokenizer, same prompts, and same target
   token burden.
5. Therefore the next scientifically defensible move is not more ad hoc
   downstream tuning, but pure-Mamba2 base recipe/shape optimization.

## Remaining Risks and Open Questions

- Are the LLaMA and Mamba base pretraining histories fully comparable in token
  budget, data ordering, optimizer stability, and checkpoint selection?
- Are the absolute PPL values fully publication-clean, or should they remain
  internal diagnostics until the audit implementation is frozen?
- Would a matched pure-Mamba2 base trained with a more public-Mamba2-like shape
  close enough of the base-loss gap to justify downstream reruns?
- If the wide pure-Mamba2 shape fails, should the next step be:
  - more pure-Mamba HPO,
  - a full fresh base run with a better-validated recipe,
  - or an explicit hybrid Mamba comparison arm?
- How should we fairly report decoder-only constrained/structured output
  variants for POS/NER without making them incomparable to the LLaMA baseline?

## Proposed Meeting Ask

Ask advisors for agreement on this decision boundary:

> If the active wide pure-Mamba2 base gate does not improve the base-loss and
> clean generation-loss profile, we should treat pure-Mamba base optimization as
> requiring a broader HPO/pretraining-design pass, and we should decide whether
> to spend that compute before moving to hybrid Mamba or xLSTM comparisons.

## Files and Evidence Trail

- High-level tracker:
  - `sallm_memory/sallm_progress.md`
- Gap plan:
  - `sallm_memory/notes/2026-05-18-mamba-gap-investigation.md`
- Main daily notes:
  - `sallm_memory/notes/2026-05-18.md`
  - `sallm_memory/notes/2026-05-19.md`
  - `sallm_memory/notes/2026-05-20.md`
  - `sallm_memory/notes/2026-05-21.md`
- Current final-results view:
  - `sallm_memory/final_results.html`
- Comprehensive optimized-result registry:
  - Google Sheet linked from `sallm_memory/sallm_progress.md`
