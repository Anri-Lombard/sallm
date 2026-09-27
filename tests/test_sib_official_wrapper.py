from pathlib import Path


def test_sib_official_wrapper_freezes_twelve_language_arms() -> None:
    wrapper = (
        Path(__file__).resolve().parents[1]
        / "scripts/run_pure_gdn_sib_official_test.sh"
    ).read_text(encoding="utf-8")

    preflight, execution = wrapper.split(
        '[[ "$preflight_only" == 1 ]] && exit 0', maxsplit=1
    )
    assert "languages=(afr eng nso sot xho zul)" in wrapper
    assert '[[ "${#labels[@]}" == 12 ]]' in wrapper
    assert 'export HF_HOME="$dataset_cache/hf"' in wrapper
    assert '--expected-artifact-root "$dataset_cache"' in wrapper
    assert 'shasum -a 256 -c "$(basename "$cache_tree_manifest")"' in preflight
    assert '[[ -f "$cache_tree_manifest" && -f "$verifier" ]]' in preflight
    assert "--required-metric 'f1,none'" in wrapper
    assert "--expected-rows 1020" in wrapper
    assert "--cfg job --resolve" in preflight
    assert "--cfg job --resolve" not in execution
    assert 'shasum -a 256 -c "${label}.resolved_config.yaml.sha256"' in preflight
    assert '[[ ! -w "$protocol_root/${label}.resolved_config.yaml" ]]' in execution
    assert '"$python_bin" "$verifier"' in execution


def test_sib_cache_script_lists_the_six_frozen_configs() -> None:
    script = (
        Path(__file__).resolve().parents[1] / "scripts/cache_sib_official_dataset.py"
    ).read_text(encoding="utf-8")

    assert '"afr_Latn", "eng_Latn", "nso_Latn"' in script
    assert '"sot_Latn", "xho_Latn", "zul_Latn"' in script
    assert 'split="test"' in script
    assert '"metrics_computed": False' in script
    assert "if len(dataset) != 204" in script
