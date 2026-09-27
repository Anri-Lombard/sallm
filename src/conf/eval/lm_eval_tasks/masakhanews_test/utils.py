from lm_eval.utils import weighted_f1_score  # noqa: F401

LABEL_TO_INDEX = {
    "business": 0,
    "entertainment": 1,
    "health": 2,
    "politics": 3,
    "religion": 4,
    "sports": 5,
    "technology": 6,
}


def doc_to_target(doc: dict[str, object]) -> int:
    """Map a MasakhaNEWS category to its fixed multiple-choice index."""
    category = doc.get("category")
    if not isinstance(category, str) or category not in LABEL_TO_INDEX:
        raise ValueError(f"Unknown MasakhaNEWS category: {category!r}")
    return LABEL_TO_INDEX[category]
