from pathlib import Path


def test_news_intent_wrapper_freezes_all_twelve_arms() -> None:
    wrapper = (
        Path(__file__).resolve().parents[1]
        / "scripts/run_pure_gdn_news_intent_official_test.sh"
    ).read_text(encoding="utf-8")
    preflight, execution = wrapper.split(
        '[[ "$preflight_only" == 1 ]] && exit 0', maxsplit=1
    )

    assert "languages=(eng xho)" in wrapper
    assert "languages=(eng sot xho zul)" in wrapper
    assert "familywise_20260901/intent_v2" in wrapper
    assert "[eng]=4740 [xho]=1485" in wrapper
    assert "[eng]=3110 [sot]=3200 [xho]=3200 [zul]=3200" in wrapper
    assert "masakhanews_${language}_test_chat" in wrapper
    assert "injongointent_${language}" in wrapper
    assert "eval.evaluation.task_packs=[$task_pack]" in wrapper
    assert "--required-metric 'f1,none'" in execution
    assert "--cfg job --resolve" in preflight
    assert "--cfg job --resolve" not in execution
    assert '[[ ! -w "$protocol_root/${label}.resolved_config.yaml" ]]' in execution
