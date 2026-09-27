# General b3 offline preflight queued

B3 no-launch GPU preflight 1312488 is pending with AssocGrpGRES. The shared
four-card nlpgroup80 limit is occupied by our b5/b6 jobs 1312084/1312231
and other-user jobs 1297472/1306119. None were changed. B5/b6 remain healthy,
observed at steps 11269 and 11127; no terminal artifacts yet.

The prospective b3 amendment is frozen at SHA-256
`2f12c3cf67b37eb5d1bf8a87c0af4170ad696b33dcf3807da452c87b4c112c5b`.
See `2026-09-06-general-b3-infrastructure-recovery-amendment.md` for the
explicit post-payload protocol exception and exact state/data/runtime binds.
Wrapper SHA-256 is
`d2359186a9ead265940efa63fc7da10484ad60f6d6578403ee93c7e94da3eddd`.
The original and failed recovery roots/logs are preserved in separate
read-only archives. All five latest complete checkpoint hashes still match
the 30 August preregistration. No held-out or cross-candidate score was used.

No b3 scientific replacement has been submitted. After 1312488 passes 0:0,
read its full check result and hash the log, then fresh no-duplicate,
absent-final/new-resume and owned-cap checks precede a single 24-hour
submission of the sealed wrapper. If the preflight fails, preserve evidence
and diagnose; do not silently launch through it. Existing failed resume
1279472 is the only allowed prior resume manifest. No extra trial is allowed.

Current quota: home 90.1%, scratch 51.5%. Existing Mono/Multi results and
Sheets are unchanged. General Stage-B remains 5/8 terminal-verified.
