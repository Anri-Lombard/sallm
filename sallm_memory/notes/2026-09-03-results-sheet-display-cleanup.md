# Results Sheet display cleanup

Date: 2026-09-03

At the user's request, prompt-summary text such as `best prompt`, `all prompts
tied`, prompt means, and prompt ranges was removed from result columns D:G in
`Transformer Results`, `Mamba Results`, `XLSTM Results`, `Qwen Results`, and
`GDN Results`. This changed 354 display cells and preserved the reported score
text, notes, evidence, formulas, formatting, and underlying metrics.

Exact post-write readback found no remaining prompt-summary pattern in those
result ranges. `Variant Charts` continued to resolve to the same values.

The GDN tab was also normalized to match the other architecture tabs: the
`0-shot` label remains on Base results only and was removed from 31 populated
Mono/Multi cells. General was included in the scoped check but is still blank.
Exact readback found 41 Base shot labels and zero adapted-model shot labels.

The display-only placeholder `Not applicable: no task-specific adapter` was
also cleared from 154 Mono/Multi cells across the Transformer, Mamba, xLSTM,
and Qwen result tabs. GDN and the dependent comparison/chart tabs were already
blank in those positions. Exact workbook-wide readback found no remaining
occurrence and no broken reference.
