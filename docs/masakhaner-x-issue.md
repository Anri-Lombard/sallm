# Request: Convert masakhaner-x to Parquet format (remove loading script)

## Problem

The `masakhane/masakhaner-x` dataset uses a custom loading script (`masakhaner-x.py`), which is no longer supported by the HuggingFace `datasets` library as of version 4.0.0.

When trying to load the dataset with `datasets>=4.0.0`:

```python
from datasets import load_dataset
ds = load_dataset("masakhane/masakhaner-x", "xh")
```

This results in:
```
RuntimeError: Dataset scripts are no longer supported, but found masakhaner-x.py
```

This affects downstream tools like [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) which use this dataset for the MasakhaNER benchmark.

## Proposed Solution

Convert the dataset to Parquet format and remove the loading script. This is the recommended approach per HuggingFace's [migration guide](https://huggingface.co/docs/datasets/main/en/repository_structure).

I've created a working Parquet mirror at [`anrilombard/masakhaner-x-parquet`](https://huggingface.co/datasets/anrilombard/masakhaner-x-parquet) that can be used as a reference or starting point.

The structure is:
```
data/
  xh/
    train.parquet
    validation.parquet
    test.parquet
  zu/
    ...
  tn/
    ...
```

Each parquet file contains columns: `id`, `text`, `spans`, `target`

## Impact

This change would:
1. Allow users with modern `datasets` versions to load the dataset
2. Enable lm-eval-harness MasakhaNER benchmarks to work out-of-the-box
3. Remove the security warning about "arbitrary Python code execution"

## References

- HuggingFace deprecation notice: https://discuss.huggingface.co/t/dataset-scripts-are-no-longer-supported/163891
- Similar lm-eval issue: https://github.com/EleutherAI/lm-evaluation-harness/issues/3390
