# Pure-GDN family-wise gate root-binding correction

Frozen before the replacement gate or any current adapter official held-out
access. Follow-up independent review found that runtime and artifact files were
hashed but the wrapper-supplied snapshot and checkpoint roots were not compared
to the roots recorded in the manifest. Pending job `1279752` was cancelled at
elapsed zero before payload and is preserved with its v3 snapshot.

The final replacement adds two fail-closed manifest assertions:

- the recorded repository root must equal the wrapper-supplied immutable
  snapshot root; and
- the artifact map must be non-empty and every recorded artifact must be a
  direct file under the wrapper-supplied exact checkpoint root.

Manifest verification still hashes every source/config and checkpoint file and
verifies the immutable runtime identity. The wrapper passes both expected roots
explicitly. This prevents a manifest for snapshot/checkpoint A from authorizing
execution of snapshot/checkpoint B.

The regression test first failed without these arguments, then all `13`
relevant tests passed and Ruff was clean. Superseding SHA-256 values are:

- execution-manifest utility:
  `4480a713e73d1e73a5b2b2cd6a50c078d29f7d6cd3059e1e07af7160aa7aa02d`;
- gate wrapper:
  `3d725220067253205e37870e502a518f3155799ba07dbc597a2cf73050f9e5ea`;
- manifest regression test:
  `4b5442749d3c7984d1253906c7a77468d3ac64c1b5ac987c6e04094ed2befed6`.

Jobs `1279742` and `1279752` and snapshots v1 through v3 are excluded. Only a
fresh v4 snapshot, manifest, result root, and Slurm job may execute.
