# Supervisor Meeting Notes: Mamba Defensibility And xLSTM Update

Date: 2026-05-28

Purpose: comprehensive speaking notes for the supervisor meeting. The goal is
to make the Mamba result defensible by walking through exactly what failed,
what was rescued, what controls were used, what examples show, and how the
literature supports or complicates the interpretation. The last section gives
the current xLSTM status.

## Executive Position

The most defensible Mamba claim is scoped:

> The current HF pure-Mamba2 SALLM path is weak and fragile for
> source-conditioned generation and decoder-only POS/NER under the tested data,
> budget, prompts, and recipes. This is not a universal claim that Mamba as an
> architecture cannot do these tasks.

The reason this is defensible is that the weakness survived multiple rescue
gates:

- output-format fixes;
- constrained tag generation;
- label-prior controls;
- decoding sweeps;
- full-data POS/NER atomic tag-sequence LoRA;
- source-preservation diagnostics for T2X;
- placeholder abstraction for T2X;
- prompt/checklist and repetition-control decoding;
- clean base-likelihood audits;
- fresh pure-Mamba2 base screens;
- shallow Mamba-2/attention hybrid screening;
- implementation hygiene checks.

The balanced conclusion is:

- Mamba is not globally broken. It can fine-tune and it can be improved on some
  generation tasks.
- The hard failures are concentrated in tasks that require source-conditioned
  preservation, copying/retrieval, exact sequence shape, or label-prior control.
- The literature supports this as a plausible failure mode for pure/fixed-state
  SSMs, especially on copying and retrieval, but it does not prove POS/NER must
  fail.
- xLSTM is now the cleaner next architecture comparison: its base likelihood
  is better than current Mamba and worse than LLaMA, so it is promising but not
  yet a win.

## One-Minute Spoken Version

If time is tight, say:

> I do not want to overclaim that Mamba cannot solve these tasks generally.
> What we can defend is that our HF pure-Mamba2 SALLM path remains weak after a
> lot of rescue attempts. It is not just one failed run. For POS/NER, we tried
> ordinary decoding, short/beam decoding, constrained tag-sequence outputs,
> label-prior controls, and full-data atomic tag-sequence LoRA. The best POS
> D4 validation token accuracy was only 0.196, and the NER D4 BIO F1 was 0.177
> with a 0.9375 all-O rate. For T2X, the B1 audit showed Mamba is
> disproportionately bad on source entity and value tokens, B2 showed
> placeholder abstraction did not rescue it, and B3 showed checklist/repetition
> prompting did not fix it. AfriHG is the nuance: beam5/length-penalty helped
> Mamba, so it is not globally broken, but the residual errors are still generic
> or wrong-focus headlines. Base audits also show Mamba starts much weaker than
> LLaMA, and cheap fresh-base rescue did not produce a credible replacement.
> This lines up with papers like Repeat After Me and Waleffe et al. on pure SSM
> copying/retrieval weaknesses, but papers on Mamba ICL prevent us from saying
> Mamba cannot do this generally.

## Main Meeting Story

### 1. Start With The Claim We Are Not Making

Do not say:

- "Mamba is bad."
- "Mamba cannot do POS/NER."
- "The architecture is proven unsuitable."
- "The base model explains everything."

Say:

- "The current HF pure-Mamba2 SALLM implementation and recipe are weak on these
  tasks."
- "The weakness is robust to several decoder-only rescue attempts."
- "The evidence points to a combination of base quality, copying/source
  preservation, output-shape control, and implementation fragility."
- "The literature makes these failures plausible, especially for copying and
  retrieval, but does not let us universalize the result."

### 2. Show That Fine-Tuning Is Not Fundamentally Broken

This matters because supervisors may ask whether the models simply did not
train.

Evidence from clean generation-loss audits:

| Task | Mamba base PPL | Mamba fine-tuned PPL | LLaMA task-specific PPL | Read |
| --- | ---: | ---: | ---: | --- |
| T2X Xho | `347.81` | `18.73` | `5.31` | Mamba learns a lot, but remains behind. |
| AfriHG Xho | `438.73` | `21.04` | `13.28` | Mamba improves, roughly around SA-general LLaMA, but not task-specific LLaMA. |
| AfriHG Zul | `606.04` | `19.28` | `10.79` | Same pattern: learning, but still behind. |

Held-out pretraining-style loss also favors LLaMA:

| Model | Weighted NLL | PPL | Validation tokens |
| --- | ---: | ---: | ---: |
| Mamba base | `3.8066` | `45.00` | `302278` |
| LLaMA base | `2.3011` | `9.98` | `302278` |

Interpretation:

- Fine-tuning is doing something meaningful.
- The base model is weaker before fine-tuning.
- The downstream failures are not just "loss never went down."
- But low or improved loss is not enough: output examples and task-specific
  diagnostics show persistent behavioral failures.

### 3. Explain The Base-Model Issue Carefully

The base is likely a major contributor, but it is not the whole story.

Clean base generation-loss audit:

| Task | Mamba base PPL | LLaMA base PPL | Ratio |
| --- | ---: | ---: | ---: |
| T2X Xho | `347.81` | `64.20` | `5.42x` |
| AfriHG Xho | `438.73` | `205.48` | `2.14x` |
| AfriHG Zul | `606.04` | `267.32` | `2.27x` |

So yes, the base model is weaker. But:

- AfriHG can be partially rescued by decoding, so the system is not globally
  broken.
- Fine-tuned Mamba PPL improves sharply, so task adaptation is not dead.
- POS/NER failures persist even when output shape is constrained and more task
  data is used.
- T2X failures are specifically concentrated on source entity/value
  preservation, not just general language quality.

Meeting wording:

> The base model is part of the explanation, but the downstream evidence is more
> specific than "bad base". It shows source preservation and output-shape
> control failures that survive several attempted repairs.

## POS/NER: Exact Experiments Tried

### Initial Observation

The original POS/NER failures were not just low accuracy. The model often did
not produce the requested structure at all.

POS examples:

| Input shape | Raw Mamba output | Parsed result | Read |
| --- | --- | --- | --- |
| Token list beginning `['Kuthintwa', 'okhulumela', 'i', '-', 'IFP', ...]` | mojibake/audio-like text such as `Audio: ...` | invalid | The model does not produce tag tuples. |
| Token list beginning `['Uthuthuva', 'luqale', ...]` | repeated `Audio:` or `(a)` patterns | invalid | Short/beam decoding reduces length but not shape obedience. |
| Token list with entities `KwaZulu - Natal ... Thembeka Mbele ...` | bracket/character streams | invalid | Not a near miss; it is output-format failure. |

NER examples:

| Text / target | Raw Mamba output | Filtered result | Read |
| --- | --- | --- | --- |
| `... kusho uShauwn.` target `PER: uShauwn` | `hofoza ... DATE PER MONEY` | empty | Emits stray words/labels, not the required entity line. |
| `NgoLwesihlanu ... ngoMsombuluko.` target two `DATE` entities | `hofoza Ngomgaqo-mali...` | empty | Generates unrelated sentence-like text. |
| `UNomfundo Zondo ... kuleli sonto...` target `PER` and `DATE` | `umise` | empty | Very short lexical collapse. |

### POS/NER Rescue Gate A: Decode-Defensibility Sweeps

Question: Is this just a bad decoding setting?

What was tried:

- short greedy decoding;
- short beam decoding;
- reduced generation lengths;
- beam controls.

Result:

| Run | Samples | Invalid / empty outputs | Weird/raw collapse | Main metric |
| --- | ---: | ---: | ---: | ---: |
| POS short greedy | `1200` | `100%` invalid parsed outputs | `83.3%` | token accuracy `0.0000` |
| POS original | `1800` | `100%` invalid parsed outputs | `75.1%` | token accuracy `0.0000` |
| NER short greedy | `1500` | `70.5%` empty filtered outputs | `30.5%` | mean F1 `0.0134` |
| NER short beam3 | `1500` | `82.6%` empty filtered outputs | `30.3%` | mean F1 `0.0191` |

Interpretation:

- Decoding alone did not rescue POS/NER.
- The issue is not simply that the model generated too long.

### POS/NER Rescue Gate A1/A2: Constrained Tag-Sequence Controls

Question: If we force the output shape, does Mamba recover?

What was tried:

- constrained generation over allowed tag labels;
- exact-length/prefix-constrained tag outputs;
- matched Mamba and LLaMA controls where feasible.

Result:

- Parseability could be forced to `1.0`.
- But label quality collapsed.
- POS collapsed toward dominant labels such as `cconj`.
- NER flooded or collapsed into labels such as `b-date`, `i-date`, and `i-org`.
- LLaMA controls were also weak under some decoder-only formulations, so some
  of this is formulation difficulty rather than a clean Mamba-only indictment.

Interpretation:

- Output shape is not the whole problem.
- Forcing a parseable sequence does not mean the model learned the tagging
  decision.
- This weakens any claim that the original problem was just metric parsing.

### POS/NER Rescue Gate A3: NER Label-Prior / All-O Controls

Question: Is NER just collapsing to `O` because that label dominates?

What was tried:

- label-prior bias controls;
- non-`O` recall checks;
- all-`O` rate tracking.

Result:

- NER continued to show label-prior collapse tendencies.
- Even when the model can produce tag strings, it does not reliably recover
  named entities.

Interpretation:

- The NER problem is not only an output parser problem.
- It is also a label-prior/entity-recall problem.

### POS/NER Rescue Gate D4: Full-Data Atomic Tag-Sequence LoRA

Question: Does the best decoder-only atomic tag recipe recover if we use more
task data?

This is one of the most important supervisor-facing gates.

What was tried:

- full-data POS/NER atomic tag-sequence LoRA;
- best prior canary recipes;
- multiple learning-rate/epoch settings.

Results:

| D4 run | Validation result | Interpretation |
| --- | ---: | --- |
| POS `lr3e-4/e80` | token accuracy `0.0970`, exact length `0.0333` | poor |
| POS `lr2e-4/e120` | token accuracy `0.1960`, exact length `0.0400`, overgeneration `1.1486` | best D4 POS, still weak and shape-poor |
| NER `lr1e-4/e60` | token accuracy `0.4252`, non-`O` recall `0.1772`, BIO F1 `0.1769`, all-`O` rate `0.9375` | no material rescue |

Interpretation:

> POS/NER weakness survived decoding fixes, constrained output space, label
> prior controls, and full-data task adaptation. That is why it is defensible
> to say this current decoder-only pure-Mamba path is not rescued for POS/NER.

What this does not prove:

- It does not prove a classifier head, CRF, encoder model, encoder-decoder
  model, or a different Mamba implementation could not solve POS/NER.
- It does not prove official `mamba_ssm` would behave identically.
- It does not prove all Mamba-like architectures fail.

## T2X: Exact Experiments Tried

### Initial Observation

T2X failures look like source-conditioned preservation failures: Mamba often
repeats source fragments, drops relation structure, omits values, or produces
very short outputs.

Representative example:

| Field | Text |
| --- | --- |
| Triple | `Abilene Regional Airport` / `cityServed` / `Abilene, Texas` |
| Mamba | `Abasebenzi beAbakwa-Abilene, e-Abilene, eAbilene.` |
| LLaMA | `Abilene Regional Airport ifumaneka e Abilene, Texas.` |
| Reference | `Isikhululo seenqwelo moya sase-Abilene sisebenzela isixeko sase-Abilene eTexas.` |
| Read | LLaMA keeps the entity/value almost literally but uses the wrong relation nuance. Mamba repeats Abilene-like fragments and loses the airport/relation structure. |

Other examples:

- `Adolfo Suarez Madrid-Barajas Airport` / `location` / `Madrid, Paracuellos de
  Jarama, San Sebastian de los Reyes and Alcobendas` produced mojibake-like
  repeated `Addis...` fragments and lost most of the value list.
- `Adolfo Suarez Madrid-Barajas Airport` / `runwayName` / `18L/36R` preserved
  only part of the runway value.
- `Afonso Pena International Airport` / `cityServed` / `Curitiba` became
  `iCuritiba yeCuritiba yiCuritiba`, preserving the value but collapsing into
  repetition instead of verbalising the relation.

### T2X Gate: Local Output Forensics

Question: Are the examples anecdotal, or is there a pattern?

Result over matched examples:

| Run | n | Avg pred tokens | Avg ref tokens | Length ratio | Repetition | Entity word preserved | Value word preserved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mamba T2X checkpoint-244 | `378` | `6.77` | `18.64` | `0.363` | `0.119` | `0.646` | `0.484` |
| Mamba T2X continue2 | `378` | `6.90` | `18.64` | `0.370` | `0.127` | `0.664` | `0.524` |
| LLaMA T2X parity recheck | `378` | `7.06` | `18.64` | `0.379` | `0.043` | `0.942` | `0.780` |

Interpretation:

- Both models are shorter than references, so length alone does not explain the
  gap.
- Mamba is worse at preserving source entities and values.
- Mamba is more repetitive.
- Continuation improved chrF slightly but did not fix the behavior.

### T2X Gate B1: Teacher-Forced Token-Class Loss Audit

Question: Is Mamba specifically worse on source entity/value tokens?

Result:

| Token class | Mamba disadvantage |
| --- | ---: |
| Entity tokens | NLL gap `+2.7656`, PPL ratio `15.89x` |
| Value tokens | NLL gap `+2.3065`, PPL ratio `10.04x` |
| Other tokens | PPL ratio `3.25x` |

Interpretation:

- The gap is not uniform language modelling weakness.
- Mamba is disproportionately worse on exactly the tokens we care about for
  source-conditioned T2X.
- This is one of the strongest bridges to the copying/retrieval literature.

### T2X Gate B2: Placeholder / Reinsertion Diagnostic

Question: If we remove direct copy pressure by replacing source values with
placeholders, does Mamba recover?

Result:

- Mamba generated zero expected placeholders in `216` examples where
  placeholders were expected.
- Reinserted chrF was only `8.2455`.
- LLaMA was much more usable under this diagnostic.

Interpretation:

- The failure is not simply "copying exact strings is hard."
- Mamba also struggled to follow the abstraction contract.
- This is a LLaMA-only or negative result for Mamba, not a Mamba rescue.

### T2X Gate B3: Source-Checklist And Repetition-Control Decoding

Question: Can prompting/decoding fix source preservation?

Result:

| Variant | chrF |
| --- | ---: |
| Mamba baseline | `28.064` |
| Source-checklist prompt | `28.148` |
| Repetition-control variants | `20.141` / `21.737` |

Matched preservation comparison:

| Model | Entity coverage | Value coverage |
| --- | ---: | ---: |
| Mamba | `0.410` | `0.320` |
| LLaMA | `0.763` | `0.594` |

Interpretation:

- Checklist prompting does not materially fix T2X.
- Repetition controls can make output worse.
- The issue looks like source-conditioned preservation/planning, not just
  greedy decoding.

What this does not prove:

- It does not prove no prompt could help.
- It does not prove no larger Mamba could do T2X.
- It supports a scoped claim: under current model/recipe, T2X source
  preservation remains weak after targeted diagnostics.

## AfriHG: Exact Experiments Tried

AfriHG is important because it prevents an over-simple "Mamba cannot generate"
story.

### Initial Observation

Mamba AfriHG outputs are often short, generic, or wrong-focus.

Representative Xho examples:

| Article hook | Mamba | LLaMA | Reference | Read |
| --- | --- | --- | --- | --- |
| Proteas batting issues and AB de Villiers absence after England series | `UMzantsi Afrika ifikile` | `UDe Villiers ubethe iNgilane ngo'3-1` | `Ingxubakaxaka ngo-AB de Villiers` | Mamba is generic and misses the hook; LLaMA is imperfect but at least uses De Villiers/England. |
| Banyana Banyana squad for Women's Africa Cup of Nations | `UMzantsi Afrika usendleleni` | `Umqeqeshi weBanyana Banyana, uDesiree Ellis...` | `ABanyana baya kwitumente ye-Afrika` | Mamba collapses to generic South Africa movement; LLaMA captures Banyana/Ellis but is too long. |
| Hlaudi Motsoeneng launches political party | `UMotsoeneng uyaphikisa uMotsoeneng` | `neeMbumba zeMpuma Koloni` | `Hlaudi: Kudala abantu bendifuna` | Both weak; Mamba repeats the entity and invents a vague relation. |

Representative Zul examples:

| Article hook | Mamba | LLaMA | Reference | Read |
| --- | --- | --- | --- | --- |
| SAFTAs planning and event dates | `I-Saftas noKhozi ngoMeyi 21` | `I-SA Film and Television Awards izokwethula amaSaftas` | `Aseqalile amalungiselelo amaSaftas` | Both identify SAFTAs; LLaMA is cleaner, Mamba adds odd wording. |
| Zandile Gumede corruption case moved to high court | `Owe-ANC umbango weTheku` | `BUKA: BUKA: BUKA...` | `Liya enkantolo enkulu elowayeyimeya yeTheku` | Mamba captures ANC/Theku context but misses the court action; LLaMA collapses here. |
| Eric Tinkler criticizes SuperSport players | `UTinkulu kowePSL` | `U-Eric Tinkler ukhale ngabadlale` | `UTinkler ukhala ngokuphela komdlandla kubadlali bakhe` | Mamba corrupts the name and is vague; LLaMA keeps actor/action. |

### AfriHG Gate C1: Beam5 / Length-Penalty 1.2

Question: Is AfriHG decode-rescuable?

Result:

| Task | Prior Mamba checkpoint-selected | C1 beam5/lp1.2 | Delta |
| --- | ---: | ---: | ---: |
| AfriHG Xho chrF | `11.5335` | `15.1702` | `+3.6367` |
| AfriHG Zul chrF | `13.5726` | `17.1245` | `+3.5519` |

Interpretation:

- This is a real Mamba improvement.
- It proves the system is not globally broken.
- It should be kept as the current Mamba AfriHG recipe.

### AfriHG Gate C1b: Matched LLaMA Decode Control

Question: Is beam5/lp1.2 just a better protocol for everyone?

Result:

- LLaMA Xho beam5/lp1.2 chrF `13.1149`, below tracked LLaMA parity `14.2017`.
- LLaMA Zul beam5/lp1.2 chrF `20.8709`, below tracked LLaMA parity `21.5552`
  and older best around `23.0041`.

Interpretation:

- This is a Mamba-specific decode rescue, not a shared protocol improvement.
- Do not use it to claim Mamba is competitive with LLaMA; use it to show
  Mamba was under-optimized on AfriHG and can be improved.

### AfriHG Gate C2: Residual Error Analysis

Question: After C1, what is still wrong?

Result:

| Language | Main residual errors |
| --- | --- |
| Xho | off-topic/generic `37.2%`; too-short/generic `31.9%` |
| Zul | off-topic/generic `54.7%`; too-short/generic `14.9%` |

Interpretation:

- Remaining AfriHG weakness is not mostly copying exact strings.
- It is semantic focus/headline planning.

### AfriHG Gates C3/C4: Focus Prompt

Question: Can stricter focus prompting fix the wrong-focus issue?

Result:

| Task | C1 beam5/lp1.2 | C4 focus_v1 official test | Read |
| --- | ---: | ---: | --- |
| AfriHG Xho chrF | `15.1702` | `14.4983` | regressed |
| AfriHG Zul chrF | `17.1245` | `17.3358` | tiny improvement only |

Interpretation:

- Focus prompting looked useful on validation, especially Zul.
- Official test did not confirm cleanly.
- Do not adopt `focus_v1` as a final recipe.

Meeting wording:

> AfriHG is the nuance: Mamba can be partially rescued by decoding, but the
> remaining issue is content selection and focus. That is different from T2X,
> where source preservation remains weak after direct diagnostics.

## Pure-Mamba Base Rescue: Exact Experiments Tried

### Why This Was Necessary

Supervisors may ask whether the Mamba base was simply bad. The answer is:

- yes, the current base is weaker than LLaMA;
- no, cheap base-recipe fixes did not produce a better base candidate;
- therefore the thesis should not spend blindly on more small HF-Mamba screens
  unless advisors explicitly want that ablation table.

### Continued/Fresh Base Attempts

Base rescue attempts included:

- 300-step continued-pretraining probes;
- 3k fresh probes;
- 20k current-shape fresh run;
- D1 wide pure-Mamba2 20k with torch eval repair;
- D2 current-shape lower-LR/warmup 10k;
- D2 wide lower-LR/warmup 10k;
- D5 shallow Mamba-2/attention hybrid screen.

Final base-gate summary:

| Run | Result |
| --- | --- |
| 20k current-shape fresh run | negative; best eval loss around checkpoint 5000, then worse; clean generation-loss far worse than existing base |
| D1 wide pure-Mamba2 20k | clean-loss PPLs T2X `8470`, AfriHG Xho `11315`, AfriHG Zul `11720`; far worse than current Mamba base `348` / `439` / `606` |
| D2 current-shape lower-LR/warmup 10k | PPLs T2X `11925.1`, AfriHG Xho `26758.1`, AfriHG Zul `28854.2`; negative |
| D2 wide lower-LR/warmup 10k | best wide checkpoint PPLs T2X `9983.6`, AfriHG Xho `21768.7`, AfriHG Zul `20378.4`; negative |
| D5 shallow Mamba-2 hybrid | held-out eval improved, but downstream clean-loss PPLs still T2X `12364.7`, AfriHG Xho `14020.5`, AfriHG Zul `14634.8`; not a base candidate |

Interpretation:

- Cheap fresh-base rescue is negative.
- Lower LR/warmup did not rescue current shape.
- The tested wide shape did not become competitive.
- The tested shallow hybrid is not a usable base candidate.
- Do not present D5 as a broad negative result for all hybrids. It was one
  shallow 8.3%-attention custom implementation with caveats.

Meeting wording:

> We cannot say that all possible Mamba bases fail. We can say that the cheap
> HF pure-Mamba2 base rescue arms we tried did not produce a credible candidate,
> so continuing that path is lower priority than xLSTM or a deliberately scoped
> official-Mamba canary.

## Implementation Hygiene Experiments

### E1: HF-vs-Official Mamba Logits Parity

Question: Are we really testing official Mamba behavior?

Result:

- Parameter counts matched.
- State load had no missing/unexpected keys.
- But logits were not close under simple mapping.
- Max absolute logit deltas were about `9.31` to `16.01`.
- Greedy next-token choices differed on two of three prompts.

Interpretation:

- This does not invalidate HF-Mamba versus HF-LLaMA experiments, because the
  comparison is on the HF path.
- It does weaken broad architecture-family claims.
- Final claim should say "HF pure-Mamba2 SALLM path", not "Mamba universally".

### E2: Mamba Generation Cache/Batch Sensitivity

Question: Are Mamba generation results stable under normal eval settings?

Result:

- Mamba generation was not perfectly invariant to cache/batch settings.
- T2X exact match to batch1/cache-off was `7/8` under cache-on or batch4.
- AfriHG Xho was `6/8` to `7/8`.

Interpretation:

- Pin generation settings in final reporting.
- Treat this as evaluation hygiene, not as the primary scientific result.

## Literature Walkthrough

### Mamba And Mamba-2: Why The Baseline Was Reasonable

Primary papers:

- Gu and Dao, "Mamba: Linear-Time Sequence Modeling with Selective State
  Spaces": https://arxiv.org/abs/2312.00752
- Dao and Gu, "Transformers are SSMs: Generalized Models and Efficient
  Algorithms Through Structured State Space Duality": https://arxiv.org/abs/2405.21060

What they support:

- Mamba is a credible efficient sequence model.
- Selective SSMs address older SSM weaknesses on discrete/token data.
- Mamba can be competitive with Transformers in language modeling settings.
- Mamba-2/SSD gives a stronger and faster variant.

How to relate to our case:

- We were justified in testing Mamba.
- Our negative downstream results are not because Mamba was an obviously bad
  candidate.
- But success in general LM benchmarks does not guarantee success on small
  low-resource decoder-only POS/NER or source-conditioned generation tasks.

### Repeat After Me: Copying And Retrieval Limits

Paper:

- Jelassi, Brandfonbrener, Kakade, and Malach, "Repeat After Me: Transformers
  are Better than State Space Models at Copying":
  https://arxiv.org/abs/2402.01032

Exact finding to convey:

- The paper studies generalized state-space models with fixed-size latent
  states.
- It argues theoretically that Transformers can copy strings in ways that
  fixed-state SSMs are fundamentally limited at.
- It also shows empirically that Transformers outperform SSMs on synthetic
  copying tasks and pretrained-model copying/retrieval tasks.
- Their practical tasks include copying text, phone-book lookup, and retrieval
  from context.

How it supports us:

- T2X requires preserving source entities and values.
- B1 shows Mamba is disproportionately worse on entity/value tokens.
- B2 and B3 show placeholder abstraction and checklist prompting do not rescue
  source preservation.
- Therefore our T2X evidence lines up with the paper's copying/retrieval
  concern.

What it does not support:

- It does not prove our POS/NER failures are caused by the same mechanism.
- It does not prove larger or differently trained Mamba models cannot copy.
- It does not directly test South African low-resource languages.

Meeting wording:

> Repeat After Me supports the plausibility of our T2X source-preservation
> result. I would not use it as a proof of the POS/NER result.

### Waleffe et al.: Empirical Study Of Mamba-Based Language Models

Paper:

- Waleffe et al., "An Empirical Study of Mamba-based Language Models":
  https://arxiv.org/abs/2406.07887

Exact finding to convey:

- The paper compares 8B Mamba, Mamba-2, Transformer, and Mamba-2-Hybrid models
  trained on controlled data up to very large token budgets.
- Pure SSMs can match or exceed Transformers on many tasks.
- But pure Mamba/Mamba-2 lag on tasks requiring strong copying, in-context
  learning, Phonebook lookup, and long-context reasoning.
- Their Mamba-2-Hybrid uses about `43%` Mamba-2, `7%` attention, and `50%` MLP
  layers.
- The hybrid model exceeds the 8B Transformer average on the standard tasks
  they evaluated and does well on long-context tasks.
- At smaller scale, they report hybrid ablations where validation loss is
  minimized around `8%` attention layers.

How it supports us:

- It supports the idea that pure SSMs may be weaker on copying/ICL-like tasks.
- It supports moving from pure Mamba to hybrid or other recurrent alternatives
  after pure-Mamba rescue fails.
- It makes our D5 hybrid attempt conceptually motivated.

What it does not support:

- It does not mean our D5 shallow hybrid should have worked.
- Their successful hybrid has a specific block structure, attention placement,
  MLP mix, large scale, and training stack.
- Our D5 was one small custom screen, so a negative D5 is not a negative result
  for all hybrids.

Meeting wording:

> Waleffe et al. supports both sides of the story: pure Mamba can be strong in
> many settings, but it can lag on copying/ICL/Phonebook-style tasks; hybrids
> can recover much of that. That is exactly why I am not making a broad
> anti-Mamba claim.

### Is Mamba Capable Of In-Context Learning?

Paper:

- Grazzi et al., "Is Mamba Capable of In-Context Learning?":
  https://arxiv.org/abs/2402.03170

Exact finding to convey:

- The paper gives empirical evidence that Mamba can show ICL-like behavior.
- It evaluates both simple function approximation and more complex NLP-style
  problems.
- It reports that Mamba can closely match Transformer performance for ICL in
  those settings.

How it complicates our claim:

- We should not say "Mamba cannot do ICL" or "Mamba cannot use context."
- Our result is about our trained HF-Mamba2 model, our low-resource tasks, our
  decoder-only setup, and our compute budget.

Meeting wording:

> This paper is the caveat I want to include. It prevents overclaiming. It says
> Mamba can show ICL-like behavior, so our evidence must stay scoped to the
> current SALLM setup and observed task failures.

### xLSTM Literature

Primary papers:

- Beck et al., "xLSTM: Extended Long Short-Term Memory":
  https://arxiv.org/abs/2405.04517
- Beck et al., "xLSTM 7B: A Recurrent LLM for Fast and Efficient Inference":
  https://arxiv.org/abs/2503.13427

Exact finding to convey:

- xLSTM extends LSTM with exponential gating and modified memory structures.
- The architecture is recurrent/efficient, with linear compute scaling in
  sequence length.
- The 7B paper reports competitive downstream task performance for a recurrent
  LLM, with faster and more efficient inference than comparable Llama- and
  Mamba-based LLMs in their setup.

How it supports our next step:

- xLSTM is a credible non-Transformer follow-up.
- Our strict-125M HF xLSTM base already beats current Mamba on clean likelihood
  audits, though it remains behind LLaMA.
- Downstream fine-tuning is the right next test.

What it does not support yet:

- It does not mean our local HF xLSTM will beat LLaMA.
- Our local shape is SALLM-HF xLSTM, not a paper-faithful 125M xLSTM recipe.
- Base-only xLSTM downstream results are weak; the real comparison needs the
  fine-tuned phases.

## xLSTM Current Update

### xLSTM Base Result

Strict full-base run:

- Run id: `xlstm_h736_ctx2048_native_4gpu_ddp_llama_budget_20260524`.
- Architecture: HF xLSTM `h736_l12_h4_chunk64`.
- Parameters: `126,901,952`.
- Training job: `861849`, completed in `1-05:59:05`.
- Effective token-slot budget: `4,758,208,512`.
- Trainer loss logs are inflated by a Transformers/xLSTM 4-GPU DDP
  loss-scaling issue, so clean audits should be used rather than raw Trainer
  losses.

Clean generation-loss audit:

| Model | T2X Xho NLL/PPL | AfriHG Xho NLL/PPL | AfriHG Zul NLL/PPL |
| --- | ---: | ---: | ---: |
| xLSTM final | `5.6584` / `286.70` | `5.7048` / `300.30` | `5.8647` / `352.39` |
| Current Mamba base | `5.8515` / `347.77` | `6.0834` / `438.50` | `6.4066` / `605.85` |
| LLaMA base | `4.1620` / `64.20` | `5.3254` / `205.48` | `5.5885` / `267.32` |

Repaired pretrain-loss audit:

| Model | Weighted NLL/PPL |
| --- | ---: |
| xLSTM final | `2.9250` / `18.63` |
| Mamba base | `3.8065` / `44.99` |
| LLaMA base | `2.3010` / `9.98` |
| xLSTM checkpoint-30000 probe | `2.6888` / `14.71` |

Interpretation:

- xLSTM is better than current Mamba on clean likelihood.
- xLSTM is still worse than LLaMA.
- This is promising enough for downstream adaptation, not enough to declare a
  win.

### xLSTM Base Downstream Phase

Base phase completed cleanly after repair:

- Jobs `867562` to `867572` completed `0:0`.
- Long-pole `867570` (`masakhaner_all`) completed in `12:47:13`.
- `39` base evaluation summaries were pulled; `afrihg_eng` is invalid/skipped.

Base-only downstream is weak:

- Belebele roughly `0.23` to `0.25` acc_norm by language.
- AfriXNLI average around `0.334` accuracy.
- AfrMMLU around `0.232` accuracy.
- AfriMGSM exact match around `0.005`.
- MasakhaNER F1 `0.0` across `tsn/xho/zul/all`.
- MasakhaPOS token accuracy `0.0` across `tsn/xho/zul/all`.
- AfriHG BLEU `0.0`.
- T2X BLEU about `0.0003`.

Interpretation:

- Base-only xLSTM is not downstream-strong.
- That is not the final architecture result, because the fair comparison needs
  monolingual/multilingual/general fine-tuned phases.

### xLSTM Mono Status As Of 2026-05-28 13:48 SAST

Mono is not completed yet.

What happened:

1. Initial mono failed because Hydra rejected
   `finetune.training.save_total_limit=1`.
2. Submitter was repaired to use Hydra append/force overrides.
3. Repaired mono exposed another Hydra override issue for W&B group/name.
4. Submitter was repaired again.
5. Full mono retry exposed xLSTM native chunk assertion in fine-tuning batches:
   sequence lengths were not divisible by chunk size `64`.
6. Submitter was repaired with `++finetune.training.pad_to_multiple_of=64`.
7. Canary then passed training batches but failed in `ShowCompletionsCallback`
   because generation inputs were not padded to chunk size.
8. Submitters were repaired with `++finetune.training.show_completions=false`.
9. Canary then failed in `ClassificationMetricsCallback`, where label-choice
   forward scoring hit the same chunk-size assertion.
10. `classification_metrics.py` was patched so xLSTM label-choice scoring
    right-pads `input_ids` and `attention_mask` to chunk size while preserving
    label-score masks based on real lengths.

Current canary:

- Tag: `xlstm_downstream_initial_20260526_pad64_mono_canary_metricpad`.
- Fine-tune job: `870846`.
- Dependent eval job: `870847`.
- Scratch: `68GB/100GB` (`68.8%`).
- Status at 13:48 SAST: `870846` running on `srvrocgpu012` at about `01:43`,
  `870847` pending on dependency.
- Log shows the canary passed the epoch-1 and epoch-2 evaluation and
  classification callbacks that previously killed `870602`.
- Canary eval loss improved from `2.3224` at epoch 1 to `2.0280` at epoch 2.
- Latest observed progress was around step `108/520`.

Interpretation:

> The xLSTM mono pipeline is not completed, but the current patch set is
> canary-healthy beyond the previous failure windows. Do not say "mono is
> fixed" yet. Say "mono has a healthy canary after repairing Hydra overrides,
> xLSTM batch padding, show-completions generation, and classification metric
> padding."

## Answers To Likely Supervisor Questions

### Is The Perplexity/Loss Low During Fine-Tuning?

For Mamba generation tasks:

- Fine-tuning dramatically reduces clean generation-loss PPL from hundreds to
  around `18-21`.
- So fine-tuning is not dead.
- But Mamba remains behind task-specific LLaMA and still produces bad outputs
  on source preservation and tagging.

For xLSTM current mono canary:

- The canary eval loss is decreasing: `2.3224` at epoch 1, `2.0280` at epoch 2.
- Raw training losses in xLSTM can be hard to compare directly because of
  known loss-scaling/logging issues in the xLSTM/DDP path, so clean eval/loss
  audits and final metrics are more reliable than raw Trainer loss.

### Is The Base Model The Issue?

Partly, yes.

- Mamba base PPL is much worse than LLaMA under the same tokenizer and task
  target burden.
- xLSTM base PPL is better than Mamba and worse than LLaMA.

But base weakness does not explain everything:

- Fine-tuning improves Mamba a lot, yet outputs remain behaviorally weak.
- POS/NER failures persist after constrained outputs and full-data LoRA.
- T2X failures concentrate on source entities/values, as shown by B1.
- AfriHG can be partially rescued by decoding, which means the story differs
  by task.

### Is The Claim Defensible Enough?

Yes, if scoped correctly.

Defensible:

> This HF pure-Mamba2 SALLM path remains weak for decoder-only POS/NER and
> source-conditioned generation after targeted rescue attempts.

Not defensible:

> Mamba cannot do POS/NER.

Defensible:

> The observed T2X weakness is consistent with literature on pure SSM
> copying/retrieval limitations.

Not defensible:

> Repeat After Me proves our POS/NER failure is architectural.

Defensible:

> Cheap HF pure-Mamba base rescue attempts did not produce a credible base.

Not defensible:

> All Mamba base recipes or official Mamba implementations would fail.

### What Should We Ask Supervisors?

Ask:

1. Is this scoped Mamba claim acceptable for the thesis direction?
2. Do they require an English POS/NER control before we pause Mamba rescue?
3. Do they want a small official `mamba_ssm` viability canary, or is the HF
   implementation caveat sufficient?
4. Should xLSTM remain the mainline architecture comparison now that its base
   likelihood beats current Mamba?
5. If xLSTM downstream is promising, should the next architecture be HGRN2 /
   BabyHGRN, or should effort return to official Mamba/hybrid Mamba?

## Final Suggested Framing

End the Mamba section with:

> I am not claiming that Mamba is fundamentally incapable. I am claiming that
> for this SALLM setting, the current HF pure-Mamba2 path stayed weak after a
> sequence of rescue experiments. The failures are concentrated in
> source-conditioned preservation and decoder-only sequence labelling. That is
> consistent with the SSM copying/retrieval literature, but the ICL literature
> also tells us not to overgeneralize. That is why I think the next defensible
> step is to continue xLSTM downstream and keep official Mamba as a scoped
> caveat or small future control, not the main compute path.

## Local Evidence Files

- Root-cause matrix:
  `sallm_memory/mamba_root_cause_rescue_evidence_matrix.md`
- Supervisor forensics:
  `sallm_memory/notes/2026-05-21-supervisor-todos-mamba-output-forensics.md`
- Weekly advisor update:
  `sallm_memory/notes/2026-05-28-advisor-meeting-weekly-update.md`
- xLSTM daily status:
  `sallm_memory/notes/2026-05-28.md`
- Mamba literature notes:
  `sallm_memory/mamba_architecture_literature.md`
- xLSTM literature notes:
  `sallm_memory/xlstm_architecture_literature.md`

## Paper Links

- Mamba: https://arxiv.org/abs/2312.00752
- Mamba-2 / SSD: https://arxiv.org/abs/2405.21060
- Repeat After Me: https://arxiv.org/abs/2402.01032
- Waleffe et al. Mamba study: https://arxiv.org/abs/2406.07887
- Is Mamba Capable of In-Context Learning?: https://arxiv.org/abs/2402.03170
- xLSTM: https://arxiv.org/abs/2405.04517
- xLSTM 7B: https://arxiv.org/abs/2503.13427
