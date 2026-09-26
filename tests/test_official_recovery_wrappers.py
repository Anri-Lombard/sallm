from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WRAPPERS = (
    "run_pure_gdn_t2x_official_test.sh",
    "run_pure_gdn_ner_official_test.sh",
    "run_pure_gdn_pos_official_test.sh",
    "run_pure_gdn_afrihg_official_test.sh",
    "run_pure_gdn_news_intent_official_test.sh",
)


def test_missing_family_wrappers_use_isolated_recovery_roots() -> None:
    for name in WRAPPERS:
        wrapper = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "pure-gdn-heldout-recovery-20260903-v2" in wrapper
        assert "familywise_recovery_20260903_v2" in wrapper
        assert "official_recovery_runtime_cache/20260903_v2" in wrapper
        assert 'HF_HOME="$runtime_cache/hf"' in wrapper
