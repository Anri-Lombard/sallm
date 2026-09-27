# Pure-GDN corrected base held-out rerun amendment — 2026-08-30

Status: prospective execution correction. Frozen before any corrected rerun
under this amendment and without opening a held-out artifact in this audit.

## Evidence boundary

Base held-out outputs predate the adapter HPO program and were explicitly
excluded from every adapter selection. Adapter held-out access in the current
program remains zero. The prior shorthand "held-out access zero" is therefore
superseded by the narrower and accurate statement "adapter held-out access
zero."

The raw lm-eval token-boundary audit found that the earlier base correction
could add both BOS and EOS, corrupting continuation slicing. The affected base
outputs are invalid implementation artifacts, not selection evidence. Preserve
them and their provenance; never use them to choose, retry, prompt, checkpoint,
or schedule any adapter.

## Frozen corrected-rerun scope

After all eight family recipes and all 21 Monolingual checkpoints are frozen,
rerun exactly these 14 base lanes once:

- News, NER, POS, SIB, Intent, and SA-general;
- Belebele Afrikaans, English, Southern Sotho, Swati, Tswana, Tsonga, Xhosa,
  and Zulu.

Do not rerun the unaffected corrected T2X or AfriHG generation lanes under this
amendment. Do not overwrite the invalid originals.

The execution snapshot must bind the merged and retested token-boundary,
classification-aggregation, NER/POS scorer, runtime-compatibility, and source
stability corrections before submission. Before consuming any corrected
held-out lane, a metric-free A100-80GB canary must load the exact frozen
pure-GDN checkpoint through the merged Transformers 5 and FLA CUDA path and
complete one forward/backward step plus generation without task or held-out
data. It must record exact model, tokenizer, prompt, evaluator, dependency,
dataset, and output hashes and write to a new correction-specific root.
Candidate recipes and adapter selection must already be immutable, so no base
result can affect them.

The 14 corrected base lanes belong to the same final reporting phase as the
one-time adapter official tests, but remain separately labelled as prospective
implementation-correction reruns. Corrected base values may update only Sheet
column D. Sheet E/F/G remain adapter-only. Fill or replace a column-D value only
from a fully verified corrected artifact and exact readback, retaining a link
to the superseded invalid artifact.

## Scientific boundary

No held-out value was consulted to choose this scope. The scope follows only
the shared faulty raw evaluation path. The first valid corrected artifact for
each lane is terminal. Only an implementation or infrastructure failure before
model, data, evaluator, prediction, or metric payload exists may receive a
separately hashed execution-only correction recorded before relaunch; no retry,
lane, or evaluator change may follow scientific output or a metric value.

## Independent-review tightening — 16:35 SAST

The original SHA-256 `078c19f5...c4784` is superseded before any use. The rule
above now closes the retry ambiguity, fixes corrected base writes to column D,
and makes the real FLA/GDN CUDA canary a hard predeployment gate. No held-out
artifact or job used the superseded wording.
