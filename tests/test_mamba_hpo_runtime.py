import importlib.util
from pathlib import Path

import pytest


def _load_verifier_module():
    path = Path(__file__).parents[1] / "scripts" / "verify_mamba_hpo_runtime.py"
    spec = importlib.util.spec_from_file_location("verify_mamba_hpo_runtime", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mamba_preflight_contract_is_fail_closed() -> None:
    verifier = _load_verifier_module()

    verifier._validate_contract(
        ("in_proj", "out_proj"),
        module_count=54,
        trainable_params=6_663_168,
    )
    verifier._assert_peak_memory(11 * 1024**3 - 1)

    with pytest.raises(ValueError, match="target modules"):
        verifier._validate_contract(
            ("in_proj", "x_proj"),
            module_count=54,
            trainable_params=6_663_168,
        )
    with pytest.raises(ValueError, match="54"):
        verifier._validate_contract(
            ("in_proj", "out_proj"),
            module_count=53,
            trainable_params=6_663_168,
        )
    with pytest.raises(ValueError, match="6,663,168"):
        verifier._validate_contract(
            ("in_proj", "out_proj"),
            module_count=54,
            trainable_params=1,
        )
    with pytest.raises(RuntimeError, match="11 GiB"):
        verifier._assert_peak_memory(11 * 1024**3)
