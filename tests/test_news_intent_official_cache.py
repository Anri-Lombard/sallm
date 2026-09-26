from pathlib import Path


def test_news_intent_cache_script_freezes_test_sources_and_counts() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/cache_news_intent_official_dataset.py"
    ).read_text(encoding="utf-8")

    assert '"masakhane/masakhanews", {"eng": 948, "xho": 297}' in script
    assert '"masakhane/InjongoIntent"' in script
    assert '{"eng": 622, "sot": 640, "xho": 640, "zul": 640}' in script
    assert 'split="test"' in script
    assert '"metrics_computed": False' in script
