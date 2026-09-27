# Mamba-2 T2X training-input amendment — 2026-08-30

Recorded at 14:01 SAST before the metric-free preflight and before any Mamba
validation result.

The activation preflight and every Mamba HPO trial use only these frozen T2X
training files:

- `data/t2x_cache/train.data`: 3,859 Python `splitlines()` rows, SHA-256
  `b16dd121b83dc7f5d2e57bfa7191bd8c04ef48dd607de25e5e86f230dee4f964`
- `data/t2x_cache/train.text`: 3,859 Python `splitlines()` rows, SHA-256
  `c245b5995f23582f2a5aceb465593bd6ec096fc24a31d4aeb880ec50d9d1aa29`

The metric-free preflight uses row index 0 after the frozen T2X template is
applied. It may report the resulting row hash, but it must not read a
validation or test file, generate a task metric, or expose generated text.
