# Pure-GDN AfriHG recovery verifier correction

Date: 2026-09-03

Recovery job `1291770` completed the frozen Multi Xhosa evaluation and wrote
its four result artifacts, then stopped before any later arm because the
structural verifier parsed JSONL with `str.splitlines()`. An output prediction
contains Unicode line separator U+2028. That character is valid inside the
JSON string and was correctly preserved by `json.dumps`, but `splitlines()`
incorrectly treated it as a record boundary and raised `JSONDecodeError`.
The scratch volume had ample capacity, the output file was closed, and the
two observed U+2028 byte sequences reproduce the failure exactly.

This prospective implementation correction changes the verifier to iterate
the UTF-8 file handle, which splits JSONL records only at actual newline
boundaries. A focused regression test reproduces the failure before the
change and passes afterward. The continuation wrapper must verify the already
written Multi Xhosa arm without rerunning it, then run only the three absent
arms: Multi Zulu, Mono Xhosa, and Mono Zulu. Job `1291770`, its log, and its
existing output remain preserved.

No model, adapter, task, prompt, split, example, generation setting, metric,
expected row count, or reporting rule changes. The partial metric printed by
job `1291770` was not used to choose or alter this correction and remains
unreportable until the full four-arm recovery bundle verifies and is sealed.

The immutable two-file overlay is
`/scratch/lmbanr001/masters/sallm/overlays/pure-gdn-afrihg-recovery-verifier-correction-20260903-2d58e73a`.
The wrapper SHA-256 is
`f745913a698e480eb80d34e66653e896cefc699fea4eb48a36b521443b10581a`,
the verifier SHA-256 is
`2bacd0a5c72a136ae4ef37f16cc8ef0ca1dc47b3f5783f8d5b268fb0948c997c`,
and the sealed overlay-manifest SHA-256 is
`6073a94fc3a84d0829153a2eef6f3cf437e4819b4abb82d20dc80feb95e49630`.

After absent-output and no-duplicate checks, continuation job `1291874` was
submitted once on A100-80GB. Its immutable batch script SHA-256 is
`abd6773b3dcbe982d1a07c2427a44a3bc9f3b31b58a720ec6bb933f5e21a4aa7`.
