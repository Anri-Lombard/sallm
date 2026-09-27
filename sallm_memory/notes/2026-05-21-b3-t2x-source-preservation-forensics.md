# B3 T2X Source-Preservation Forensics

Created: 2026-05-21.

Purpose: inspect whether the T2X gap is partly source entity/value preservation
rather than only general fluency. This is a local forensic pass over already
pulled outputs, not a new model run.

This is a decoder-only diagnostic. It does not change the architecture.

## Artifacts Used

Mamba current checkpoint-selected T2X test output:

- `outputs/eval/final_mamba/checkpoint_selected_20260518/mamba-t2x-xho-ckpt244/t2x_xho_hpo_selected_final/examples.jsonl`
- Metric: chrF `32.6125`, BLEU `0.0580`, ROUGE-L `0.2966`.

LLaMA current-stack parity T2X test output:

- `outputs/eval/diagnostics/llama_generation_parity_recheck_20260502/llama_t2x_xho_opt_chrf_fix2_fullft/t2x_xho/examples.jsonl`
- Metric: chrF `53.4796`, BLEU `0.2155`, ROUGE-L `0.5053`.

Important caveat:

- The source-preservation metric below is exact token overlap between source
  entity/value strings and the generated output.
- It undercounts legitimate isiXhosa translations or paraphrases, especially
  when a source name/title is translated in the reference.
- It is still useful as a relative diagnostic because both architectures are
  evaluated with the same crude source-token preservation rule.

## Aggregate Source Preservation

Both outputs have n=`378` examples.

| Model | Mean entity token coverage | Any entity token | Complete entity | Mean value token coverage | Any value token | Complete value | Repetition rate | Empty rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Mamba ckpt244 | `0.241` | `0.397` | `0.106` | `0.202` | `0.294` | `0.106` | `0.082` | `0.000` |
| LLaMA parity | `0.511` | `0.762` | `0.220` | `0.437` | `0.563` | `0.272` | `0.034` | `0.000` |

Reference exact source-token coverage for the same examples is low because
many references translate or inflect names/values:

| Field | Reference exact token coverage |
|---|---:|
| Entity | `0.364` |
| Value | `0.295` |

Interpretation:

- Mamba preserves substantially fewer source entity/value tokens than LLaMA.
- Mamba also repeats more often.
- Since LLaMA is strong on headline metric and source preservation under the
  same rough metric, this supports a real Mamba-specific T2X preservation gap.
- The exact-overlap metric is not enough by itself to prove the whole cause,
  because the references often translate source values; B1 teacher-forced
  token-class NLL is still needed for stronger evidence.

## Worst Relation Types by Mamba Value Coverage

Only relation types with at least three examples are shown.

| Relation type | n | Mamba value coverage | Mamba entity coverage |
|---|---:|---:|---:|
| areaTotal | 3 | `0.000` | `0.333` |
| award | 3 | `0.000` | `0.500` |
| cityServed | 3 | `0.000` | `0.000` |
| league | 3 | `0.000` | `0.000` |
| mediaType | 3 | `0.000` | `0.000` |
| status | 3 | `0.000` | `0.000` |
| activeYearsStartDate | 4 | `0.000` | `0.417` |
| birthDate | 4 | `0.000` | `0.125` |
| ethnicGroup | 4 | `0.000` | `0.250` |
| runwayLength | 5 | `0.000` | `0.150` |

This points toward weaknesses on structured values, dates, numeric fields,
airport/location triples, and some relation labels.

## Illustrative Mamba Examples

| # | Entity | Relation | Value | Mamba output | Reference | Entity cov | Value cov | Note |
|---|---|---|---|---|---|---:|---:|---|
| 1 | Abilene Regional Airport | cityServed | Abilene, Texas | Abasebenzi beAbakwa-Abilene, e-Abilene, eAbilene. | Isikhululo seenqwelo moya sase-Abilene sisebenzela isixeko sase-Abilene eTexas. | `0.00` | `0.00` | repeats Abilene-like forms but misses relation and exact value |
| 2 | Adolfo Suarez Madrid-Barajas Airport | location | Madrid, Paracuellos de Jarama, San Sebastian de los Reyes and Alcobendas | iAdolfi... i-Addis... | Isikhululo seenqwelo moya i-Adolfo Suarez Madrid-Barajas sifumaneka eMadrid... | `0.00` | `0.00` | severe copied-name corruption |
| 3 | Adolfo Suarez Madrid-Barajas Airport | runwayName | 18L/36R | Igama lerun le runway yaseAdolfo Suarez yi18L. | Igama lomgaqo ... yi-18L/36R. | `0.17` | `0.00` | partial entity, value truncated |
| 4 | Afonso Pena International Airport | ICAO Location Identifier | SBCT | I-SBCT ye-Af. | I-SBCT sisichongi sendawo ye-ICAO ... i-Afonso Pena. | `0.00` | `0.00` | value appears but tokenization metric misses; sentence incomplete |
| 5 | Afonso Pena International Airport | cityServed | Curitiba | iCuritiba yeCuritiba yiCuritiba. | ... sisebenzela isixeko saseCuritiba. | `0.00` | `0.00` | value loop without entity/relation |
| 6 | Al-Taqaddum Air Base | cityServed | Fallujah | i-Al-Taqadd ye-Al Taq yi-Air. | I-Al Taqaddum Air Base isebenza kwisixeko saseFallujah | `0.00` | `0.00` | entity fragment, missing value |
| 7 | Al-Taqaddum Air Base | runwayLength | 3684 | iLocation yaseAl-Al Taq yi3. | Ubude ... yi-3684.0. | `0.00` | `0.00` | numeric value lost |
| 8 | Alderney Airport | runwayName | 14/32 | Igama lerunway yaseAlderney yase-Alderney. | ... yi-14/32. | `0.00` | n/a | repeats entity, misses runway value |

## Current Decision

B3 is partially answered from existing outputs:

- Mamba's T2X gap is plausibly tied to source preservation and copy/value
  fidelity, not only fluent wording.
- LLaMA preserves source entity/value strings much more often under the same
  crude diagnostic and has much higher chrF.
- The next strong test should be B1: teacher-forced token-class conditional
  loss, because it can tell whether Mamba assigns disproportionately worse
  likelihood to source entity/value spans before generation errors compound.
- A later B2 delexicalized placeholder/reinsertion diagnostic is also well
  motivated: if placeholders improve Mamba substantially, then copying/value
  fidelity is a root lever.
