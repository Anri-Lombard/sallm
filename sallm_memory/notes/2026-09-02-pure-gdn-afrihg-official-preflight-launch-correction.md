# Pure-GDN AfriHG official preflight launch correction — 2026-09-02

CPU freeze job `1287732` failed `126:0` before the AfriHG wrapper executed.
The immutable wrapper had the intended read-only mode `0444`, so direct shell
execution returned permission denied. The job only wrote the six prospective
execution manifests; it did not resolve a config, load a model or dataset, or
open official held-out payloads. Preserve its job record and logs.

One corrected CPU preflight may invoke the unchanged wrapper through `bash`.
The wrapper remains byte-identical at SHA-256
`fb19cdc03f4adaa5c475b62f7bd770764431dedd162a3a1f173a92d86871e63a`.
No scientific input, arm, config, checkpoint, result path, or terminality rule
changes. The corrected preflight must verify the existing manifests and write
all four resolved configs before the one-time official GPU job is submitted.
