# Pure-GDN T2X official-test snapshot overlay correction

Frozen before any T2X official held-out access.

The first immutable snapshot path
`/home/lmbanr001/masters/sallm_snapshots/pure-gdn-t2x-official-test-20260901-v1-daf6d8cf`
was copied from the read-only CUDA-gate snapshot, so its read-only `scripts`
directory rejected the two new official-test files during the overlay. No
model, dataset, evaluator, output directory, or metric was opened. Preserve
and exclude v1.

The one prospective replacement uses
`/home/lmbanr001/masters/sallm_snapshots/pure-gdn-t2x-official-test-20260901-v2-daf6d8cf`.
It copies the same verified v4 CUDA-gate source, makes only the destination
scripts directory temporarily owner-writable, overlays the already frozen
wrapper and structural verifier byte-for-byte, and returns the entire snapshot
to read-only mode. Wrapper and verifier hashes remain
`4bc7a937adea2d74470d695b45db02699e6c2e796e344eb1a1d05bede0a3d31b`
and `daf6d8cf004441cdf7d56a232d83b6e193b3ccbbacf93c88202541c0166985d9`.
All scientific settings, checkpoint and adapter paths, manifest checks,
official output path, and one-time reporting rule are unchanged.
