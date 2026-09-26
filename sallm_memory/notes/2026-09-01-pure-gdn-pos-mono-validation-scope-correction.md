# Pure-GDN POS Mono validation-scope correction

Frozen prospectively on 1 September 2026 after job `1282672` failed, before
any corrected POS Mono run or metric existed. No official held-out data was
opened.

## Observed failure

POS Tswana Mono job `1282672` completed the first constrained validation pass
over all `750` declared rows, then exited `1:0` after `01:30:56`. The evaluator
raised `POS coverage mismatch`. It demanded Xhosa and Zulu prompt cells and
rejected the declared Tswana `lm_eval_p5` cell. The output root contains only
the immutable execution manifest. It has no checkpoint, final adapter, or
selection artifact. Preserve the job, log, and root. Do not use its validation
output.

The failure is an evaluator-scope defect. The frozen Multilingual POS HPO grid
uses Tswana, Xhosa, and Zulu with prompts `lm_eval_p1` through `lm_eval_p4`.
The existing Mono configurations use one language with prompts `lm_eval_p1`
through `lm_eval_p5`. The evaluator hardcoded the Multilingual grid for both.

## Prospective correction

The corrected evaluator keeps strict coverage but derives the expected
language by template cross-product from the declared validation dataset. This
preserves the exact 12-cell Multilingual contract and accepts the intended
five-cell Mono contract for each language. No model, base checkpoint, adapter
recipe, learning rate, seed, data seed, optimizer, epoch count, length limit,
dataset row, template, checkpoint rule, or metric formula changes.

The source SHA-256 changes from
`49ff23f8ecd8ea488079cbc31b0a67ac7bb8986c0aa27da3d40e126ddd536344` to
`bd40634152c40f95e7b06ddf92dac2c8a2b5df3b3f994324cfc69c753e5f3f93`.
The regression test SHA-256 is
`c171d8125e867fc274f3838583c9979cbd04f5891e4fa7af18c6797c94c19da6`.
The focused POS suite passes `8/8`; Ruff and formatting checks pass.

Because job `1282672` produced no checkpoint, the one isolated corrected
Tswana replacement must start from the unchanged pure-GDN base. It is an
implementation correction, not an added scientific trial. Use the corrected
evaluator prospectively for the unstarted Xhosa and Zulu Mono arms. Each run
still requires absent-output, no-duplicate, immutable-source, exact-data, and
active-cap checks.
