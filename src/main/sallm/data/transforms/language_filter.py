from __future__ import annotations

from typing import Any

from datasets import Dataset

LANG_COLUMNS = ("lang", "language", "language_code")


def get_lang_column(ds: Dataset) -> str | None:
    """Get the language column name if present in the dataset.

    Args:
        ds: Dataset to check

    Returns:
        Column name if found, None otherwise
    """
    for col in LANG_COLUMNS:
        if col in ds.column_names:
            return col
    return None


def filter_by_language(ds: Dataset, languages: set[str]) -> Dataset:
    """Filter dataset to only include rows matching specified languages.

    Checks 'lang', 'language', and 'language_code' columns.

    Args:
        ds: Dataset to filter
        languages: Set of language codes to keep

    Returns:
        Filtered dataset
    """
    lang_col = get_lang_column(ds)
    if lang_col is None:
        return ds

    def _matches(ex: dict[str, Any]) -> bool:
        code = ex.get(lang_col)
        return code in languages

    return ds.filter(_matches)


def filter_by_single_language(ds: Dataset, lang_tag: str) -> Dataset:
    """Filter dataset to a single language.

    Args:
        ds: Dataset to filter
        lang_tag: Single language code to keep

    Returns:
        Filtered dataset
    """

    def _matches(ex: dict[str, Any]) -> bool:
        if "lang" in ex:
            return ex["lang"] == lang_tag
        if "language" in ex:
            return ex["language"] == lang_tag
        if "language_code" in ex:
            return ex["language_code"] == lang_tag
        return False

    return ds.filter(_matches)
