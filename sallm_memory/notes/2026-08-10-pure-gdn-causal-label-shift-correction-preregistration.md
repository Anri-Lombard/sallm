# Pure-GDN causal label-shift correction preregistration — 2026-08-10

Frozen at 19:52 SAST before implementation changes or any replacement-run
metric. No held-out adapter result was accessed.

## Trigger and scientific status

Transformers `4.57.3` applies causal shifting in its label smoother only when
the unwrapped model class name occurs in
`MODEL_FOR_CAUSAL_LM_MAPPING_NAMES`. External FLA
`GatedDeltaNetForCausalLM` is absent. With the shared launcher's frozen
`label_smoothing_factor=0.05`, labels are removed before model forward and the
smoother compares position-*t* logits with position-*t* labels. The intended
causal objective compares position-*t* logits with position-*t+1* labels.

This fact comes from the immutable launcher, execution manifest, installed
Transformers source, and model class. It does not depend on held-out scores.
NER jobs `1217444/1217445/1217446` and POS jobs
`1217737/1217738/1217739` are provenance-only and cannot select a recipe or
characterize the architecture. Frozen Multilingual winners are `0/8`.

## Frozen implementation correction

1. Register the external FLA class name as the `gated_deltanet` causal-LM
   entry in Transformers' causal-LM name mapping before any SALLM trainer is
   constructed. Do not change model weights, data, prompts, learning rates,
   smoothing factor, optimizer, LoRA recipe, evaluation metrics, stopping
   rules, or tie-break rules.
2. Add one focused regression test that exercises Transformers'
   `Trainer.compute_loss` with a dummy class named
   `GatedDeltaNetForCausalLM`. Its logits will make shifted and unshifted
   label-smoothed loss distinguishable; the test must equal the explicitly
   shifted reference and differ from the unshifted reference.
3. Run the focused test, complete local test suite, and focused lint. Deploy
   only through a new immutable source snapshot and hashed execution manifest.
4. Start every replacement adapter from the canonical pure-GDN base, never
   from an invalid adapter checkpoint. Rerun affected grids validation-only
   under their already frozen recipes. Existing artifacts remain untouched.

## Acceptance and selection

- The regression test must fail without the registration and pass with it.
- Training loss must use shifted label smoothing; corrected evaluation remains
  unchanged and independently causal/shifted.
- Validation coverage and artifact-hash gates remain unchanged.
- Recipe and checkpoint selection use only preregistered validation metrics
  and tie-break rules. No held-out adapter evaluation or Sheet E/F/G write is
  authorized until all eight Multilingual winners are validly frozen.

