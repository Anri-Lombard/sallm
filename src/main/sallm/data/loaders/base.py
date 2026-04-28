from __future__ import annotations

from datasets import Dataset, load_dataset

VALIDATION_ALIASES = ["validation", "dev", "val", "valid"]


def load_split_with_fallback(
    hf_name: str, name: str | None, split: str, revision: str | None = None
) -> Dataset:
    """Load a dataset split, falling back to alternative split names if needed."""
    try:
        return load_dataset(hf_name, name=name, split=split, revision=revision)
    except Exception as err:
        s = split.lower()
        if s in VALIDATION_ALIASES:
            candidates = VALIDATION_ALIASES
        elif s == "test":
            candidates = ["test"]
        else:
            candidates = [split]

        last_err = err
        for alt in candidates:
            if alt == split:
                continue
            try:
                return load_dataset(hf_name, name=name, split=alt, revision=revision)
            except Exception as e:
                last_err = e
        raise last_err from None
