from types import SimpleNamespace

import pytest
from sallm.models import optional


def test_optional_registration_returns_false_only_when_fla_is_absent(
    monkeypatch,
) -> None:
    def missing_fla(_: str):
        raise ModuleNotFoundError("No module named 'fla'", name="fla")

    monkeypatch.setattr(optional, "import_module", missing_fla)

    assert optional.register_fla_gated_deltanet() is False


def test_optional_registration_imports_fla_module(monkeypatch) -> None:
    seen: list[str] = []
    monkeypatch.setattr(
        optional,
        "import_module",
        lambda name: seen.append(name) or SimpleNamespace(),
    )

    assert optional.register_fla_gated_deltanet() is True
    assert seen == ["fla.models.gated_deltanet"]


def test_optional_registration_reraises_transitive_import_errors(monkeypatch) -> None:
    error = ModuleNotFoundError("No module named 'triton'", name="triton")
    monkeypatch.setattr(
        optional,
        "import_module",
        lambda _: (_ for _ in ()).throw(error),
    )

    with pytest.raises(ModuleNotFoundError) as raised:
        optional.register_fla_gated_deltanet()

    assert raised.value is error
