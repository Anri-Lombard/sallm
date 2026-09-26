# Pure-GDN Intent pre-payload sealing correction

Official Intent job `1290452` failed `1:0` on 3 September 2026 before loading a
model, dataset, task, prediction, or metric. All source, base, cache, adapter,
and resolved-config hashes verified. The wrapper then failed closed because the
resolved configs were still writable. The preceding submission command had
stopped at an unavailable `rg` executable before reaching its sealing step.
The Intent result root contains zero files.

This is a prospective execution-only correction. Preserve job `1290452`, its
log, the `intent_v1` protocol, and the original snapshot. Bind the corrected
wrapper to a fresh `intent_v2` protocol and snapshot, generate fresh manifests
and resolved configs, seal them using a standalone command, and verify every
file is non-writable before one replacement submission. No model, adapter,
dataset, row, prompt, checkpoint, task pack, or metric changes.

Corrected wrapper SHA-256 is
`907f5da5c3da8d383e5ebef8141bc1cf145cd955287c8c5436d69242a8250224`.
Bash syntax and the focused wrapper test pass. The replacement remains the
same frozen eight-arm Intent bundle and becomes terminal when it opens any
Intent held-out payload.
