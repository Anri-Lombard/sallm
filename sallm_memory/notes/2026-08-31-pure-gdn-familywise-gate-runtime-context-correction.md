# Pure-GDN family-wise gate runtime-context correction

Frozen before any replacement CUDA-gate submission or current adapter official
held-out access. The first corrected immutable snapshot exposed a fail-closed
verification mismatch before submission: the manifest records creation-context
environment variables, including Slurm variables, that must legitimately differ
inside the later batch job.

The runtime comparison now verifies only immutable runtime identity: Python
version, Python executable, platform, and the complete installed-package
version mapping. It deliberately excludes recorded environment variables;
the wrapper already freezes the scientific paths and offline flags, while
Slurm job identifiers and similar execution-context variables cannot match
manifest creation. Source and exact checkpoint artifact hashes remain fully
verified.

The regression test now includes a deliberately different creation-context
Slurm value and still requires all immutable runtime fields to match. All `13`
relevant tests pass and Ruff is clean. This supersedes only the execution-
manifest utility and its test hashes in the earlier review correction:

- execution-manifest utility:
  `76c71c1412506ea1389c5eb407f9624a878658318a14f59f5852417bfaffa71f`;
- manifest regression test:
  `74c69cd570712e47da402aa2db9e41337d1581e93e442599788d0f7c5ce99679`.

Cancelled job `1279742` and immutable snapshots ending `0c242dcf` and
`v2-0c2958dc` remain preserved and may never be used. A fresh snapshot,
manifest, result root, and job are required.
