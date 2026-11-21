import importlib.util
from pathlib import Path

from lm_eval.filters.transformation import SPANFilter

UTILS = (
    Path(__file__).resolve().parents[2]
    / "src/conf/eval/lm_eval_tasks/nchlt_ner_validation/utils.py"
)
spec = importlib.util.spec_from_file_location("nchlt_ner_utils", UTILS)
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

WITHOUT_MISC = [
    ["PER: Nelson Mandela $$ LOC: eMvezo", "person: Thabo, Sipho\nORG: none"],
    ["Location: Cape Town $$ date: 1994", "no entities here"],
]


def test_nchlt_filter_matches_lm_eval_span_filter_without_misc() -> None:
    assert utils.format_span_misc(WITHOUT_MISC, None) == SPANFilter().apply(
        WITHOUT_MISC, None
    )


def test_nchlt_filter_keeps_misc_entities_that_the_stock_filter_drops() -> None:
    resps = [["PER: Thabo $$ MISC: isiNdebele"]]

    assert utils.format_span_misc(resps, None) == [["per: thabo $ misc: isindebele"]]
    assert SPANFilter().apply(resps, None) == [["per: thabo"]]
