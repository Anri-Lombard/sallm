# Pure-GDN T2X and NER held-out recovery results

Date: 2026-09-03

The disclosed recovery produced the first terminal-valid official held-out
results for the previously missing T2X and NER families. These are recovery
artifacts, not claims that the original one-time jobs succeeded.

## T2X

- Job `1291767` completed `0:0` in `00:07:50` on A100-80GB.
- Exact coverage is `378` Xhosa test rows.
- Official held-out chrF is `51.73983564531445`.
- BLEU is `0.19635039005699184`, ROUGE-1 is `0.4996135720968379`,
  ROUGE-2 is `0.3025574075636826`, and ROUGE-L is
  `0.489791468767565`.
- Structural verification passed and contains no metric values.
- Sealed artifact-tree manifest SHA-256 is
  `4d008493f93d07c5f8451b3ea9ac5b01206d97d379c2abba58540d5dcdb67536`.

## NER

- Job `1291768` completed `0:0` in `01:04:16` on A100-80GB.
- All six Mono/Multi language arms passed structural verification with exact
  five-prompt coverage: Tswana `4,980`, Xhosa `5,000`, and Zulu `5,000`
  rows per mode.
- The actual task-source path and immutable snapshot were proven byte-exact
  across the 32 selected NER YAML/Python files. Equivalence artifact SHA-256
  is `046826b170ecba0e0b2db60d1f2d23160f183f82e57bdb69bf64f2399b0758f6`.
- Sealed NER artifact-tree manifest SHA-256 is
  `d1b2808b85af820c30edd51602a42f2b06eb06d90b967c26a32b2c1964d5e253`.

The five-prompt means below are descriptive aggregation across the complete
frozen prompt set, not held-out prompt selection:

| Mode | Language | Mean F1 | Prompt F1 range |
|---|---|---:|---:|
| Multi | Tswana | 0.780121633143 | 0.769865841073-0.784548784549 |
| Multi | Xhosa | 0.700863188515 | 0.692661792581-0.708795900939 |
| Multi | Zulu | 0.736678628778 | 0.731239092496-0.744493392070 |
| Mono | Tswana | 0.676808782625 | 0.664161257260-0.683973693319 |
| Mono | Xhosa | 0.526014117971 | 0.454095656954-0.556734693878 |
| Mono | Zulu | 0.498772346650 | 0.486790818536-0.509426393903 |

No held-out value was used to select, retry, alter, or reorder any arm.
Sheet E/F/G remains unchanged until all recovery families and exact sheet
readback verification are complete.
