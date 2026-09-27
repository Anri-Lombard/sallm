# Pure-GDN result interpretation audit

Date: 2026-09-03

This is a post-hoc, report-only audit of already frozen official held-out
artifacts. It did not select, retry, alter, or reorder any model, checkpoint,
prompt, recipe, or job.

## Live Sheet state

Exact Google Sheets readback confirmed that the latest terminal-valid News,
NER, and T2X values are present in `GDN Results`, `Variant Comparison`, and
the formula-backed `Variant Charts`. Qwen occupies columns P:S and GDN occupies
columns T:W in `Variant Comparison`; the values are not mixed. Pure-GDN
General remains blank, and pure-GDN POS Mono/Multi remain blank while their
official bundle is still running.

## MasakhaNews

The official task reports support-weighted F1, not macro-F1: the frozen task
YAML imports `weighted_f1_score` with `average: weighted`. Existing generic
`F1` Sheet labels remain accurate, but prose calling these official values
macro-F1 must be corrected.

The weak News result is a repeatable partial class collapse rather than random
test noise. On English prompt 1, the Mono adapter predicts only business,
entertainment, health, and politics, with only one business prediction and no
sports or technology predictions. It has zero sports and technology recall,
despite those classes contributing 200 and 123 of the 948 test rows. The
Multi adapter broadens the usable label space and sharply improves health and
politics recall, but still has zero sports recall and near-zero technology
recall. The same qualitative pattern repeats over all five prompts.

Across all five prompts, Multi exceeds Mono by `0.124963` weighted F1 for
English and `0.098213` for Xhosa. A fixed-seed paired document bootstrap over
the five-prompt mean gave 95% intervals `[0.104284, 0.147824]` for English and
`[0.027594, 0.170292]` for Xhosa. The result is therefore not a one-prompt or
small-sample fluctuation.

The likely training explanation is confounded rather than architecture-pure.
The multilingual News adapter trains on the concatenated 3,309 English and
1,032 Xhosa rows, whereas each Mono adapter sees only its own language. Under
the shared family recipe, Multi therefore receives more total examples and
optimizer updates per epoch, in addition to possible cross-lingual transfer.
No update-matched or target-exposure-matched control exists, so the current
result cannot distinguish those mechanisms.

Across architectures, pure GDN is not uniquely broken on News English. Its
best-prompt Mono F1 `0.2717` exceeds xLSTM `0.1700` and Qwen `0.2094`, but is
well below LLaMA `0.6350` and Mamba `0.7730`. The defensible claim is a broad
News weakness relative to the leading decoder baselines, not an English-only
failure.

## MasakhaNER

The strong NER Multi result is replicated across all three languages and all
five prompts. Paired document bootstraps of the five-prompt mean Multi-minus-
Mono span F1 gave approximate 95% intervals `0.086-0.122` for Tswana,
`0.151-0.198` for Xhosa, and `0.210-0.270` for Zulu; every sampled difference
was positive.

The gain is not a formatting-only artifact. Aggregating all five prompts,
Multi improves exact-span F1 for every entity type (DATE, LOC, ORG, and PER)
in all three languages and substantially reduces both false positives and
false negatives. For example, Xhosa falls from `4,940/3,818` FP/FN under Mono
to `3,099/2,322` under Multi; Zulu falls from `3,646/2,517` to `1,594/1,391`.

The strongest supported explanation is pooled task exposure plus transfer.
Each NER Mono arm has exactly 1,441 training rows; Multi concatenates all
three languages for 4,323 rows under one shared four-entity extraction schema.
NER is also a local span-copying task, which is a plausible fit for a recurrent
sequence model, but the present experiment does not isolate GDN's architecture
from data volume, update count, or multilingual transfer. Cross-architecture
winner claims remain weaker because the older architecture rows were not all
trained under the same HPO and seed protocol.

## POS and General

There is no official pure-GDN General NER or POS score yet. GDN General cells
are blank in both `GDN Results` and `Variant Comparison`. The visible strong
GDN NER values are Multi, not General. Pure-GDN POS Mono/Multi is also still
blank pending terminal verification. Any current explanation of a strong
pure-GDN POS or General test result would therefore be premature.

The same exposure confound will matter when POS completes: the frozen POS
Multi training set concatenates three 753-row language datasets (2,259 rows),
whereas a Mono arm sees 753. Interpret Multi-versus-Mono only after reporting
this unmatched update/data budget.
