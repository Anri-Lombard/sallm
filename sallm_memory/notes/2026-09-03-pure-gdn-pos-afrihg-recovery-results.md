# Pure-GDN POS and AfriHG recovery results

Date: 2026-09-03

The two remaining familywise official-test recoveries are terminal-valid.
POS job `1291776` completed `0:0`; AfriHG continuation `1291874`
completed `0:0` after preserving the earlier Multi-Xhosa output from
`1291770`.

All ten structural sidecars and all POS/AfriHG execution-manifest sidecars
pass exact SHA-256 verification. Coverage is exact:

- POS: 2,408 Tswana rows and 2,404 Xhosa/Zulu rows for each of Mono and Multi,
  over all four frozen prompts.
- AfriHG: 1,305 Xhosa rows and 1,776 Zulu rows for each of Mono and Multi,
  under the single frozen template.

Verified family artifact-tree SHA-256 values are:

- POS: `cc26ce67a1752b41adf209b56bf92d657772be369b5d60b4db6e60a6133c13b4`
- AfriHG: `323bdbab403016bf2c91b5c35c0e427480b054f08031b8e436dbeb2664285feb`

The canonical Sheet uses the same descriptive best-prompt headline convention
as the other architectures. POS Mono/Multi token accuracy is
`0.9568/0.9767` for Tswana, `0.9717/0.9734` for Xhosa, and
`0.9734/0.9667` for Zulu. AfriHG Mono/Multi chrF is
`23.2563/24.5583` for Xhosa and `24.9258/25.8549` for Zulu. These are
official recovery test results, not validation values. No held-out value
informed selection, correction, retry, checkpoint, or HPO.

Exact Sheet readback verified `GDN Results`, `Comparison Data`, `Variant
Comparison`, `Variant Charts`, and the affected language-chart data.
During readback, all 160 language-chart GDN formulas were found to source the
Qwen block (`P:S`) despite the column being labelled GDN. They were corrected
to the GDN block (`T:W`). Post-write formula readback is 160/160 correct,
with zero remaining Qwen-sourced GDN formulas and zero `#REF!` errors.
General remains blank.

At the 18:38 UTC follow-up, all 28 files listed in the ten POS/AfriHG
structural verifiers' `artifact_sha256` maps passed direct SHA-256 checks.
This checks the actual result files as well as the previously checked sidecars.

Four AfriHG cell notes in `GDN Results!E42:F43` incorrectly labelled both
`1291770` and `1291874` as completed. Live Slurm readback confirmed `1291770`
failed `1:0` and `1291874` completed `0:0`. The notes now distinguish the
preserved Multi Xhosa inference from the three arms evaluated by the
continuation. Exact readback confirms only these four notes changed; scores,
formatting, and General blanks are unchanged. No inference was rerun.

At the 19:08 UTC follow-up, quota-first HEX readback found no owned running
or queued jobs. Home quota remained 89.5% and scratch 51.2%. The half-hour
`gdn-hpo-workstream-monitor` automation was deleted to stop idle checks.
All authorized non-General test work is complete. General remains paused
under the current protocol and its Sheet cells stay blank; stopping this
monitor does not declare the entire downstream program complete.
