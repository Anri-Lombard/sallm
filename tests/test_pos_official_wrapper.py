from pathlib import Path


def test_execution_reuses_frozen_resolved_configs() -> None:
    wrapper = (
        Path(__file__).resolve().parents[1]
        / "scripts/run_pure_gdn_pos_official_test.sh"
    ).read_text(encoding="utf-8")

    preflight, execution = wrapper.split(
        '[[ "$preflight_only" == 1 ]] && exit 0', maxsplit=1
    )
    assert "--cfg job --resolve" in preflight
    assert "--cfg job --resolve" not in execution
    assert 'shasum -a 256 -c "${label}.resolved_config.yaml.sha256"' in preflight
    assert "pos_resolved_config_correction_v3" in wrapper
