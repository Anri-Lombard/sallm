from pathlib import Path


def test_official_wrapper_pins_cache_and_reuses_resolved_configs() -> None:
    wrapper = (
        Path(__file__).resolve().parents[1]
        / "scripts/run_pure_gdn_afrihg_official_test.sh"
    ).read_text(encoding="utf-8")

    preflight, execution = wrapper.split(
        '[[ "$preflight_only" == 1 ]] && exit 0', maxsplit=1
    )
    assert "SALLM_AFRIHG_CACHE_ONLY=1" in wrapper
    assert "SALLM_AFRIHG_CACHE_DIR" in wrapper
    assert '--expected-artifact-root "$dataset_root"' in wrapper
    assert "--cfg job --resolve" in preflight
    assert "--cfg job --resolve" not in execution
    assert 'shasum -a 256 -c "${label}.resolved_config.yaml.sha256"' in preflight
    assert "expected_rows=(1305 1776 1305 1776)" in wrapper
    assert "SALLM_OFFICIAL_CONTINUE_AFTER_VERIFIER_FIX" in wrapper
    assert "SALLM_OFFICIAL_GENERATION_VERIFIER" in wrapper
    assert "overlay_sha256.txt" in wrapper
    assert '"$label" != multi_xho' in execution
