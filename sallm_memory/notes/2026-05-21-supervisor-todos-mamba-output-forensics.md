# Supervisor To-Dos: Mamba Output Forensics and Defensibility

This note turns the advisor/supervisor feedback into a concrete worklist for
next week. The goal is to answer why Mamba is weak on generation and structured
generation tasks without overclaiming architecture failure before ruling out
implementation, base quality, decoding, output control, and formulation issues.

## Current Best Answer

Mamba is not uniformly bad. It is strong on MasakhaNews after the template/schema
fix, which proves the downstream fine-tuning path can work. The remaining gap is
concentrated in tasks that need source-conditioned generation, coverage, copying,
or exact output shape: T2X, AfriHG, POS, and NER.

The best current explanation is a combination of:

- weaker Mamba base quality than LLaMA under the same tokenizer and target
  token burden;
- poorer source-conditioned coverage/copying in free generation;
- output-shape collapse on tag-sequence tasks;
- Mamba/HF generation and eval-kernel fragility that needs parity checks.

## To-Do List

### 1. Base Model Adequacy

Question: Is the Mamba base simply weaker before fine-tuning?

Already answered:

- Clean base-loss audits show Mamba base has much worse PPL than LLaMA base:
  - T2X Xho: Mamba `347.81` vs LLaMA `64.20`.
  - AfriHG Xho: Mamba `438.73` vs LLaMA `205.48`.
  - AfriHG Zul: Mamba `606.04` vs LLaMA `267.32`.
- Held-out pretraining PPL also favors LLaMA: Mamba `45.00` vs LLaMA `9.98`.
- Same tokenizer is used, so the gap is not explained by using different
  tokenizers.

Still to answer:

- Tokenizer fertility by task/language and by failed examples.
- Base zero/few-shot raw outputs for the exact same T2X/AfriHG prompts.
- Whether a cleaner fresh pure-Mamba2 base recipe/shape reduces the base-loss
  gap. Active gate: `854987 -> 854988 -> 854989`.

Update, 2026-05-21:

- The active wide-shape base gate is now a third repaired chain:
  `854987 -> 854988 -> 854989`.
- It keeps fused CUDA/Mamba2 training enabled and forces only evaluation through
  the HF torch path.
- The canary `854987` completed successfully after reducing eval memory
  pressure with `eval_batch_size=1` and `eval_max_length=1024`; the full 20k
  job `854988` is running.
- This means the current blocker is no longer "Mamba cannot run this shape";
  the remaining question is whether the wide pure-Mamba2 shape actually learns
  better once allowed to complete.

### 2. T2X Output Forensics

Question: What exactly is Mamba doing wrong on triple verbalisation?

Observed so far:

- Mamba often repeats entity fragments, drops relations, or omits values.
- Example: for `Abilene Regional Airport | cityServed | Abilene, Texas`, Mamba
  generated `Abasebenzi beAbakwa-Abilene, e-Abilene, eAbilene`, while the
  reference says the airport serves Abilene, Texas.
- Example: for `runwayName | 18L/36R`, Mamba got a partial relation but dropped
  `/36R`.

To do:

- Build a small T2X error table with columns:
  source entity, relation, value, Mamba output, LLaMA output, reference,
  error type.
- Error types:
  relation dropped, value dropped, entity repetition, mojibake/name corruption,
  hallucinated relation, too short, malformed sentence.
- Score exact preservation of source entity and value.
- Score relation lexicalisation success by relation type.

Answers so far, 2026-05-21:

Quantitative local output-forensics pass over the official/checkpoint-selected
Mamba T2X examples and the local LLaMA reference examples:

| Run | n | Avg pred tokens | Avg ref tokens | Length ratio | Repetition | Entity word preserved | Value word preserved |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mamba T2X checkpoint-244 | 378 | 6.77 | 18.64 | 0.363 | 0.119 | 0.646 | 0.484 |
| Mamba T2X continue2 | 378 | 6.90 | 18.64 | 0.370 | 0.127 | 0.664 | 0.524 |
| LLaMA T2X parity recheck | 378 | 7.06 | 18.64 | 0.379 | 0.043 | 0.942 | 0.780 |

Interpretation:

- Both models are shorter than the references on this T2X setup, but Mamba is
  worse at preserving the source entity and value.
- T2X continuation improved chrF a little, but did not fix the main behavior:
  it stayed short and slightly more repetitive.
- Relation-string preservation is low for both under this crude string test
  because relation labels are verbalized, but the qualitative examples show
  Mamba more often drops or distorts the relation/value.

Representative paired example:

| Field | Text |
|---|---|
| Triple | `Abilene Regional Airport` / `cityServed` / `Abilene, Texas` |
| Mamba | `Abasebenzi beAbakwa-Abilene, e-Abilene, eAbilene.` |
| LLaMA | `Abilene Regional Airport ifumaneka e Abilene, Texas.` |
| Reference | `Isikhululo seenqwelo moya sase-Abilene sisebenzela isixeko sase-Abilene eTexas.` |
| Read | LLaMA keeps the entity/value almost literally but uses the wrong relation nuance (`located at` vs `serves`). Mamba repeats Abilene-like fragments and loses the airport/relation structure. |

Other Mamba T2X examples:

- `Adolfo Suarez Madrid-Barajas Airport` / `location` / `Madrid, Paracuellos de
  Jarama, San Sebastian de los Reyes and Alcobendas` -> Mamba output becomes
  mojibake-like repeated `Addis...` fragments, losing most of the value list.
- `Adolfo Suarez Madrid-Barajas Airport` / `runwayName` / `18L/36R` -> Mamba
  says roughly `... yi18L`, preserving only half of the runway value.
- `Afonso Pena International Airport` / `cityServed` / `Curitiba` -> Mamba
  outputs `iCuritiba yeCuritiba yiCuritiba`, preserving the value but collapsing
  into repetition instead of verbalizing the relation.

### 3. AfriHG Output Forensics

Question: Why are headlines weak if they are fluent-ish?

Observed so far:

- AfriHG Mamba outputs are often too short and generic.
- Xho examples include outputs like `UMzantsi Afrika ifikile` or
  `UMzantsi Afrika usendleleni`, which are grammatical-ish but miss the news
  hook.
- Zul examples are sometimes closer, e.g. SAFTAs headline, but still less
  precise than references and LLaMA.

To do:

- Build an AfriHG error table:
  article hook, named entities in reference, Mamba headline, LLaMA headline,
  reference, error type.
- Error types:
  too generic, wrong focus, entity missing, date/event missing, repetition,
  ungrammatical, hallucinated entity.
- Add source-entity preservation and headline length-ratio metrics.

Answers so far, 2026-05-21:

Quantitative local output-forensics pass over checkpoint-selected Mamba AfriHG
examples and local LLaMA parity examples:

| Run | n | Avg pred tokens | Avg ref tokens | Length ratio | Repetition |
|---|---:|---:|---:|---:|---:|
| Mamba AfriHG Xho checkpoint-656 | 1305 | 2.42 | 4.92 | 0.492 | 0.014 |
| LLaMA AfriHG Xho parity recheck | 1305 | 8.44 | 4.92 | 1.716 | 0.053 |
| Mamba AfriHG Zul checkpoint-892 | 1776 | 2.95 | 4.94 | 0.596 | 0.026 |
| LLaMA AfriHG Zul parity recheck | 1776 | 4.54 | 4.94 | 0.920 | 0.032 |

Interpretation:

- Mamba's AfriHG problem is visibly different from T2X: it under-generates very
  short headlines, especially Xho.
- LLaMA Xho sometimes over-generates or includes artifacts like `xn`, but it
  usually keeps more article-specific content. LLaMA Zul is much closer to the
  target headline length.
- The AfriHG continuation gate regressed because it shortened outputs further,
  so blind longer fine-tuning is not the right fix for AfriHG.

Representative Xho examples:

| Article hook | Mamba | LLaMA | Reference | Read |
|---|---|---|---|---|
| Proteas batting issues and AB de Villiers absence after England series | `UMzantsi Afrika ifikile` | `UDe Villiers ubethe iNgilane ngo'3-1` | `Ingxubakaxaka ngo-AB de Villiers` | Mamba is generic and misses the hook; LLaMA is imperfect but at least uses De Villiers/England. |
| Banyana Banyana squad for Women's Africa Cup of Nations | `UMzantsi Afrika usendleleni` | `Umqeqeshi weBanyana Banyana, uDesiree Ellis...` | `ABanyana baya kwitumente ye-Afrika` | Mamba collapses to generic South Africa movement; LLaMA captures Banyana/Ellis but is too long. |
| Hlaudi Motsoeneng launches political party | `UMotsoeneng uyaphikisa uMotsoeneng` | `neeMbumba zeMpuma Koloni` | `Hlaudi: Kudala abantu bendifuna` | Both weak; Mamba repeats the entity and invents a vague relation. |

Representative Zul examples:

| Article hook | Mamba | LLaMA | Reference | Read |
|---|---|---|---|---|
| SAFTAs planning and event dates | `I-Saftas noKhozi ngoMeyi 21` | `I-SA Film and Television Awards izokwethula amaSaftas` | `Aseqalile amalungiselelo amaSaftas` | Both identify SAFTAs; LLaMA is cleaner, Mamba adds an odd `noKhozi`. |
| Zandile Gumede corruption case moved to high court | `Owe-ANC umbango weTheku` | `BUKA: BUKA: BUKA...` | `Liya enkantolo enkulu elowayeyimeya yeTheku` | Mamba captures ANC/Theku context but misses court-case action; LLaMA collapses into repetition here. |
| Eric Tinkler criticizes SuperSport players | `UTinkulu kowePSL` | `U-Eric Tinkler ukhale ngabadlale` | `UTinkler ukhala ngokuphela komdlandla kubadlali bakhe` | Mamba corrupts the name and becomes too vague; LLaMA keeps the actor/action. |

### 4. POS/NER Output-Shape Rescue

Question: Can decoder-only Mamba be rescued without switching to classifier
heads or CRFs?

Already answered:

- POS/NER failures are not just low task accuracy; they are output-shape
  failures.
- NER original extraction prompt often generates news-like text that filters to
  empty.
- BIO tag-sequence overfit can produce lexical/repetitive garbage instead of
  BIO tags.
- Atomic labels improve parseability but collapse mostly to `<ner_o>`.
- Best NER diagnostic is still weak:
  - token accuracy `0.670`;
  - strict token accuracy `0.278`;
  - exact length `0.055`;
  - all-O prediction `0.855`;
  - non-O recall `0.178`;
  - BIO F1 `0.177`.

To do:

- Constrained decoding over allowed tag labels only.
- Dynamic max-new-tokens from input token count.
- Per-token decoder-only loglikelihood tagging.
- Matched LLaMA controls under the same decoder-only formulation.
- Keep these as diagnostic variants unless they are also run for both
  architectures.

Answers so far, 2026-05-21:

Local sample-level inspection confirms that POS/NER are failing mainly because
free generation does not obey the requested output shape.

NER decode-defensibility summaries:

| Run | Samples | Empty filtered outputs | Weird/raw collapse | Mean F1 over prompt variants | Max prompt F1 |
|---|---:|---:|---:|---:|---:|
| NER short greedy | 1500 | 70.5% | 30.5% | 0.0134 | 0.0702 |
| NER short beam3 | 1500 | 82.6% | 30.3% | 0.0191 | 0.0936 |

NER examples:

| Text / target | Raw Mamba output | Filtered result | Read |
|---|---|---|---|
| `... kusho uShauwn.` target `PER: uShauwn` | `hofoza ... DATE PER MONEY` | empty | It emits stray words/labels, not the required entity line. |
| `NgoLwesihlanu ... ngoMsombuluko.` target two `DATE` entities | `hofoza Ngomgaqo-mali...` | empty | It generates unrelated sentence-like text, so extraction produces nothing. |
| `UNomfundo Zondo ... kuleli sonto...` target `PER` and `DATE` | `umise` | empty | Very short lexical collapse; no parseable NER output. |

POS decode-defensibility summaries:

| Run | Samples | Invalid parsed outputs | Weird/raw collapse | Mean token accuracy over prompt variants |
|---|---:|---:|---:|---:|
| POS short greedy | 1200 | 100% | 83.3% | 0.0000 |
| POS original | 1800 | 100% | 75.1% | 0.0000 |

POS examples:

| Input shape | Raw Mamba output | Parsed result | Read |
|---|---|---|---|
| Token list beginning `['Kuthintwa', 'okhulumela', 'i', '-', 'IFP', ...]` | mojibake/audio-like text such as `Audio: ...` | invalid | The model does not produce tag tuples at all. |
| Token list beginning `['Uthuthuva', 'luqale', ...]` | repeated `Audio:` or `(a)` patterns | invalid | Beam/short decoding reduces length but not shape obedience. |
| Token list with entities `KwaZulu - Natal ... Thembeka Mbele ...` | mojibake-like bracket/character streams | invalid | This is not a close tagging error; it is output-format failure. |

Interpretation:

- For POS/NER, normal decoding tweaks are not enough. We need either
  constrained label decoding, per-token loglikelihood classification, or a
  stricter decoder-only formulation where the model only chooses from allowed
  labels.
- To keep architecture comparison fair, any rescue that changes formulation
  should be run for both Mamba and LLaMA as a matched decoder-only diagnostic.

### 5. Implementation Parity

Question: Is the HF Mamba path itself hurting results?

Known issues:

- Trainer-side generation metrics previously broke on `Mamba2Cache.float`.
- Wide pure-Mamba2 shape hit HF eval-only `causal_conv1d` stride constraints.
- Eval/generation has required fallbacks and small batch caps.

To do:

- Run HF-vs-official `mamba_ssm` logits parity on a representative checkpoint
  and prompt.
- Run short generation parity if feasible.
- Confirm tokenizer, chat template, dtype, cache behavior, and eval fallback
  do not change metrics except speed/memory.

### 6. Base Recipe / Shape Optimization

Question: Is the base model bad because of architecture limits or because of
our base recipe/shape?

Already answered:

- 300-step continued pretraining did not rescue current Mamba base.
- 3k fresh probes showed LR `4e-4` learns faster than `2e-4`, but 3k is far
  from usable.
- 20k current-shape fresh run was negative: best eval loss at checkpoint 5000,
  then worsening, and clean generation-loss far worse than existing Mamba base.
- Wide shape is still scientifically unresolved because the first two attempts
  hit HF eval-kernel constraints before producing eval metrics.

Active:

- `854987 -> 854988 -> 854989`: wide pure-Mamba2 run with fused fast training
  retained and eval forced through HF torch path with cheaper eval
  (`eval_batch_size=1`, `eval_max_length=1024`).

Decision boundary:

- If the wide torch-eval gate improves base/eval loss and clean generation
  loss, use it as the next pure-Mamba base candidate.
- If it fails scientifically, prepare a small pure-Mamba base HPO matrix before
  deciding whether to spend on a full retrain or pivot to hybrid/xLSTM arms.

## Supervisor-Facing Deliverables for Next Meeting

- One table of final Mamba vs LLaMA task metrics.
- One table of T2X qualitative errors.
- One table of AfriHG qualitative errors.
- One table of POS/NER output-shape failures and attempted rescues.
- One paragraph on base-model likelihood evidence.
- One paragraph on implementation parity risks.
- One clear ask: whether to spend compute on broader pure-Mamba base HPO before
  moving to hybrid Mamba or xLSTM comparisons.
