# Pure-GDN NER Mono offline-cache ID correction

Date: 2026-09-01
Scope: prospective pre-validation infrastructure correction only

## Failure boundary

NER Mono Tsn `1281964` and Xho `1281965` completed their immutable source,
runtime, GPU, base-model, and recipe checks, then failed before loading any
dataset row. No validation metric, checkpoint, or final adapter was produced.
The isolated failed roots and logs remain preserved and must not be reused.

The pinned MasakhaNER-X parquet revision is present in the offline cache, but
the current `datasets` runtime computes different config IDs from the runtime
that built that cache. An explicit-revision CPU probe `1281974` reproduced the
same Tsn lookup failure, and probe `1281976` established the corresponding
current Zulu ID. This is a cache-key compatibility failure, not a change to the
data or scientific recipe.

## Frozen correction

Create only these absent compatibility aliases inside the user-owned datasets
cache, with each alias pointing to the already-present exact cached config:

- Tsn: `default-21d7aeb4ecebf694` -> `default-d8b906c0392feca5`
- Xho: `default-9278b546d4159f3d` -> `default-0dda5814bfb8acb9`
- Zulu: `default-c511ba6223ac8e2c` -> `default-913bf988fbedf987`

Before any replacement GPU submission, an offline CPU compute-node probe must
load the aliased train and validation splits at pinned revision
`6aa65cdbfa22d66e5b4ed176ac525c364cda08d1` and verify exact row counts:
Tsn `1441/499`, Xho `1441/817`, and Zulu `1441/836`. The canonical cached Arrow
files and `dataset_info.json` files must retain their existing SHA-256 values.

If and only if that probe passes, Tsn and Xho may each receive one isolated
replacement submission with the unchanged frozen NER recipe, seed 42, base,
runtime, source snapshot, and validation-only selection protocol. Zulu follows
when an eligible card releases. The aliases may not be used to reopen official
held-out data or to alter any candidate, checkpoint, or result decision.
