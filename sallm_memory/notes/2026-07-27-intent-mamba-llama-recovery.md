# INJOngo Intent Mamba/LLaMA recovery — 2026-07-27

## Decision

The current Mamba and LLaMA decoder-only Intent rows are invalid as final
architecture evidence. Do not promote them or tune against their test scores.

## Root causes

1. The evaluator summed continuation log-probabilities although the 40 intent
   verbalizers span 1–6 tokenizer tokens, structurally favouring short labels.
   In a matched LLaMA artifact, `text`/`time` scored about `-10.06` while the
   correct longer `alarm` verbalizer scored `-31.25`.
2. Historical fine-tunes used 1,582 validation rows: all 622 English test rows
   plus the three 320-row Sot/Xho/Zul dev splits. Checkpoint selection therefore
   touched held-out test data.
3. All `622/622` English test texts occur in English train; one Xhosa test text
   occurs in Xhosa train.
4. Zulu dev contains shifted/corrupt labels, including a meal-suggestion text
   labelled `car_rental`.
5. Historical `run_llama_injongointent_all.yaml` defaults to the SA-general
   checkpoint, although matched job `1073400` overrode it correctly.

LLaMA reached teacher-forced validation loss about `0.1361`, so its chance-level
choice score is primarily a protocol failure. Mamba's old best validation loss
was about `1.7074`, so it may still need bounded optimization after the shared
repair.

## Shared local fix

Changed:

- `src/main/sallm/evaluation/classification_metrics.py`
  - mean token log-probability is the default choice score;
  - capped validation samples are balanced by prompt and gold label.
- `src/main/sallm/data/loaders/huggingface.py`
  - remove held-out test-text overlaps before splitting;
  - derive validation from clean train rather than upstream dev.
- `src/main/sallm/data/loaders/injongointent_split.py`
  - normalize held-out text matching;
  - include intent in stable split keys because upstream example IDs repeat.
- Focused tests cover scoring mode, balanced capping, decontamination, and
  repeated cross-label IDs.

Validation:

- `pytest`: 6 passed.
- `ruff check`: passed.

Expected clean train/validation counts from the audit are Eng `1046/111`, Sot
`2000/240`, Xho `1999/240`, and Zul `2000/240`; every split retains all 40
labels and clean-train/test text overlap is zero.

## Next dependency-safe work

Sync only the reviewed files to HEX, run clean validation controls for exact
Mamba/LLaMA bases on A100, retrain because old checkpoint selection was
contaminated, use bounded Mamba HPO only if needed, and evaluate held-out test
once per validation-selected recipe. This external recovery chain requires
fresh explicit approval after the exact risks and output writes were identified.
No final sheet write is allowed before that chain completes.
