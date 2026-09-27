# SALLM results and valid-improvement audit — 2026-08-30

This audit uses validation-only evidence. It does not open or use an official
held-out result, and it does not authorize Sheet E/F/G changes.

## Strongest defensible results

| Family | Status | Validation-only result | Interpretation |
| --- | --- | --- | --- |
| T2X | frozen, full protocol | b7 mean chrF `51.987594971037375` versus a2 `50.75818404404604`; sample SD `0.272831202123097` versus `0.5410846408801392` | Clearest full-protocol win. |
| POS | frozen, budget-limited | a2 mean constrained accuracy `0.8606543221600176` versus a1 `0.8463392380469843` | Largest classification margin, but only under the prospective three-candidate, two-seed close-out. |
| NER | frozen, full protocol | b7 mean span F1 `0.682516674203261` versus a2 `0.6778547247706191` | Valid winner; margin is smaller than b7's seed SD, so claim competitiveness rather than robust superiority. |
| AfriHG | frozen, reconciled full protocol | b7 mean chrF `25.24535303429672` versus a2 `25.0302068631814` | Valid but modest win after applying the original three-seed rule to unchanged evidence. |
| General | incomplete | a2 retained macro NLL `0.9284030074439155`; b0 retained `0.9451928643283297` | Promising evidence only. No winner may be named before all frozen trials and confirmations verify. |

The defensible global state is `4/8` frozen. News, SIB, and Intent have no
post-BOS candidates, and General is incomplete. Earlier `7/8` shorthand is
historical provenance only.

The separate post-hoc LLaMA-125M T2X lane has five terminal seed-42 candidates
so far. A2 retains validation chrF `48.52835434271314` and b0
`46.98977851830069`, both with exact adapter roundtrips. This lane is
incomplete and historically test-exposed, so it cannot support an architecture
winner claim.

## Highest-payoff valid improvements

1. Complete General b4--b7 and the exact same-trial b1, b2, then b3
   checkpoint-10912 recoveries. Rank only after all 11 seed-42 candidates are
   terminal-valid, then run the fixed four confirmations.
2. Run the missing post-BOS News, SIB, and Intent programs exactly as frozen:
   11 seed-42 candidates using corrected validation-only macro F1, followed by
   the fixed top-two confirmations at seeds 13 and 87.
3. Write and verify one hashed eight-family freeze manifest.
4. Complete the 21-arm Monolingual set: train and validation-select 20 new
   language adapters using exactly the learning-rate component of each final
   hashed family winner, and count the already selected T2X HPO winner as the
   single T2X Xhosa arm without retraining it. Keep rank 16, alpha 32, dropout
   0.05, and warmup ratio 0.03 fixed for the 20 new adapters; enhanced
   Multilingual winners' other hyperparameters do not transfer to Mono.
5. Only then run every applicable adapter official held-out arm once and fill
   Sheet E/F/G from verified artifacts and exact readback.
6. After the metric-free real FLA/GDN CUDA canary passes, rerun exactly the 14
   quarantined raw base lanes under the prospective correction amendment and
   update only Sheet column D from verified artifacts and exact readback.
7. Finish the separately labelled LLaMA T2X HPO lane before selecting its
   recipe. A2 remains provisional until all candidates and confirmations
   complete.

These actions improve the result set by finishing the frozen search and adding
language-specific adapters. They do not change a candidate, metric, prompt, or
checkpoint rule in response to observed performance. General continuation
authority is a disclosed post-start amendment triggered by administrative
wall-time, not by metric magnitude.

## Repository cleanup decision

Draft PR #128 was closed rather than merging a disconnected HPO provenance
framework. Its branch adds 1,319 lines but has no production caller; the
acceptance prototype added another 615 lines and still could not bind a run to
the actual config, architecture, checkpoint, tokenizer, and family metric.
Issue #43 now carries the smaller replacement: one integrated launch-to-rank
slice that rejects partial/TIMEOUT trials, requires full coverage, a final
adapter and sidecar-bound exact roundtrip proof, derives direction from the
bound family protocol, and rejects incomplete seed sets. The remote branch and
uncommitted review prototype are preserved as evidence; neither is used for
scientific ranking.

## Invalid or biased shortcuts

- Starting Monolingual training or official evaluation before the hashed 8/8
  freeze.
- Reusing pre-BOS News, SIB, or Intent winners.
- Extending POS with Stage-B or seed 87 after its prospective close-out.
- Cancelling, reordering, retrying, or adding General candidates from interim
  validation NLL.
- Restarting any General recovery from scratch, changing its checkpoint, or
  making a second scientific recovery for b1--b7 after any model, data,
  training, validation, or evaluator output. Only a separately hashed
  pre-payload execution correction may relaunch, never because of a metric.
- Reporting General as an unamended preregistered result. B1--b3 continuation
  authority was added after their timeouts and interim validation artifacts;
  the b4--b7 rule was frozen after start but before terminal outcomes. Both are
  metric-independent, same-state protocol amendments and must be disclosed.
- Changing prompts, decoding, selection metrics, early stopping, LoRA targets,
  or candidate grids after observing validation.
- Changing Mamba targets or bypassing PEFT under the failed activation. That
  would require a separately named new protocol, not a retry.
- Using incomplete post-hoc LLaMA validation to claim architecture
  superiority.
- Any new adapter held-out access before all applicable recipes and checkpoints
  are frozen. Historical base outputs remain excluded from selection, and the
  14 invalid raw lanes remain quarantined until their final correction rerun.

## 2 September non-General execution update

SIB Xhosa/Zulu and Intent Xhosa completed under their already-frozen family
recipes. Exact CPU verifiers `1288988`-`1288990` proved retained-to-final
equality across `424` keys and `71,762,560` values per adapter. This advances
Mono freeze to `19/21`, with SIB complete `6/6` and Intent `2/4`. Intent
Southern Sotho `1288991` was submitted once after metric-free frozen
preflight; Intent Zulu remains active. No held-out result informed training,
verification, submission order, or scheduling.

Intent Zulu subsequently completed under the unchanged frozen recipe, and
CPU verifier `1289009` proved exact checkpoint-`750` to final equality across
`424` keys and `71,762,560` values. Mono freeze is `20/21`; only Intent
Southern Sotho `1288991` remains active. No held-out result informed this
state transition.

Intent Southern Sotho and verifier `1289188` then completed cleanly, freezing
all `21/21` Mono adapters. The first SIB manifest/preflight chain remained
pre-payload but was rejected because its immutable snapshot lacked the
post-arm structural verifier. No held-out inference or metric existed. The
prospective correction is frozen under SHA-256
`1620c6559b0b937ef1a51862664b0801cbbd19fae184404f060bf2bff2e7e854`.
The corrected chain sealed and verified the full cache tree and twelve
resolved configs under jobs `1289206`-`1289208`. One-time SIB Mono/Multi job
`1289209` was submitted without inspecting any test metric.

SIB job `1289209` failed before payload because the immutable `v2` snapshot
used raw Linux kernel build strings during runtime verification. Its official
result root remained absent, so no prediction or metric existed. The final
prospective public-seam correction is frozen under SHA-256
`4c6fd9934109174f9da8fad2445de20e1f6d86ab68065ebf35eec9c88710b93f`.
Fresh immutable manifest/preflight jobs `1289210/1289211` completed `0:0`;
single replacement job `1289212` was submitted once without inspecting any
held-out metric.

SIB replacement `1289212` completed `0:0` in `00:18:42`. All twelve fixed
Mono/Multi arms and structural sidecars verified before any score was opened.
The sealed result-tree manifest SHA-256 is
`ad98b6c0bbf2c03d101fbc79232654d4bfdb125abb317840290a18fc227136d9`.
Report-only six-language macro held-out F1 is `0.1367692769759137` for Multi
and `0.07555764695434825` for Mono. No score informed another run or decision;
Sheet E/F/G remain blank pending the complete obtainable set.

## Source artifacts

- `2026-08-13-pure-gdn-t2x-confirmation-ranking.json`
- `2026-08-30-pure-gdn-pos-budget-closeout-ranking.json`
- `2026-08-18-pure-gdn-ner-confirmation-ranking.json`
- `2026-08-30-pure-gdn-afrihg-confirmation-ranking-reconciliation.json`
- `2026-08-30-pure-gdn-global-freeze-evidence-reconciliation.md`
- `2026-08-30-pure-gdn-general-stage-b-walltime-recovery-preregistration.md`
- `2026-08-30-pure-gdn-general-stage-b-later-timeout-continuation-amendment.md`
- `2026-08-30-pure-gdn-corrected-base-heldout-rerun-amendment.md`
- `2026-08-30-cross-architecture-kombuys-hpo-activation.md`

## 3 September obtainable non-General close-out

News official job `1290454` failed after opening the first held-out task but
before writing any example, prediction, result, or metric artifact. The
failure is terminal under the one-time protocol, so News remains missing.

Intent's pre-payload sealing defect was corrected prospectively without
changing any scientific input. Jobs `1290455`-`1290458` completed cleanly and
verified all eight fixed Mono/Multi arms. The sealed 32-file result-tree
manifest SHA-256 is
`44a89fde7b63361e586a2135e1dc46a756032a6fad9b8613ac55174832534576`.
Report-only macro held-out F1 is `0.002614876682742` for Mono and
`0.001161031235439` for Multi.

The obtainable current-adapter non-General result set is therefore complete:
Base `16/16`, plus SIB and Intent covering `10/21` Mono rows and `10/20` Multi
rows. News, T2X, NER, POS, and AfriHG are terminally missing. Exact API
readback of `GDN Results!C10:G19` confirmed that SIB and Intent E/F values and
dates match the sealed artifacts, Base D is untouched, formatting is intact,
and General G remains blank.
