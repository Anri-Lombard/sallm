# General b6 continuation

The b6 same-trial continuation is job 1312231, submitted once after GPU
preflight 1312227 completed 0:0 in 20 seconds. Preflight log SHA-256:
`2aa271753bf48c289a7446c928bf1a40d8dabe370467bdf1b71fb2163cd750f7`.
The immutable prospective amendment SHA-256 is
`a119efbad4b90d9d60c13846095cb483f815acf8f5ef5df99afada251deeedc6`;
see `2026-09-06-general-b6-offline-continuation-amendment.md` for all state,
archive, source, runtime and data bindings.

All five checkpoint-10912 hashes match the 31 August record. The complete
original root is archived read-only. Preflight verified original runtime,
695-file source inventories, the narrow cache-only AfriHG correction, exact
AfriHG record equality and all twelve train/validation digests against the
frozen successful offline preflight. No metric informed this continuation.
Fresh accounting, absent-final/resume and owned-cap checks preceded submission.

B5 1312084 remains healthy, observed at step 11003/13640. Both continuations
use only A100-80GB with one GPU each, eight CPUs, and 24 hours. No completed
Mono/Multi result or Sheet cell was changed. General b3 still needs its
separately disclosed infrastructure-recovery amendment before replacement.

At 16:51 UTC, b6 resumed directly to step 10913 and advanced through 10920.
Its resume execution manifest SHA-256 is
`7d1355af1b1c20dac151e1a9a0be340ecf3455aa0903be5bfcf057ea6d14bacd`.
B5 advanced through step 11063. Both jobs remain running on A100-80GB.

Next: monitor final artifacts and
22,167-row validation coverage for b5/b6, then exact adapter roundtrips.
Do not rank the incomplete candidate set. Home quota 90.1%, scratch 51.4%.
