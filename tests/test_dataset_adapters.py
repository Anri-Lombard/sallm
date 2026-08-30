from __future__ import annotations

from urllib.error import HTTPError, URLError

import pytest
from datasets import Dataset, DatasetDict
from sallm.config import FinetuneDatasetConfig, FinetuneTaskType
from sallm.data.adapters import (
    github,
    huggingface,
    masakhapos,
)
from sallm.data.adapters.masakhaner import masakhaner_data_files
from sallm.data.adapters.registry import load_raw_dataset, resolve_dataset_adapter
from sallm.data.formatters.instruction import format_instruction
from sallm.data.formatters.pos import format_pos
from sallm.data.loaders import mix


def _cfg(
    hf_name: str,
    *,
    subset: str | None = None,
    languages: list[str] | None = None,
    task: FinetuneTaskType = FinetuneTaskType.INSTRUCTION,
) -> FinetuneDatasetConfig:
    return FinetuneDatasetConfig(
        hf_name=hf_name,
        subset=subset,
        languages=languages,
        task=task,
        splits={"train": "train", "val": "validation"},
        max_seq_length=128,
        packing=False,
        assistant_only_loss=True,
    )


def test_resolves_dataset_specific_adapters() -> None:
    cases = {
        "github:francois-meyer/t2x": "github-t2x",
        "github:dadelani/AfriHG": "github-afrihg",
        "masakhane/masakhapos": "masakhapos",
        "masakhane/InjongoIntent": "injongointent",
        "masakhane/masakhaner2": "masakhaner",
        "Davlan/sib200": "huggingface",
    }

    for hf_name, adapter_name in cases.items():
        assert resolve_dataset_adapter(_cfg(hf_name)).name == adapter_name


def test_t2x_github_adapter_loads_instruction_sample(monkeypatch) -> None:
    monkeypatch.setattr(
        github,
        "load_t2x_from_github",
        lambda: DatasetDict(
            {
                "train": Dataset.from_list([{"source": "amanzi", "target": "water"}]),
                "validation": Dataset.from_list(
                    [{"source": "umlilo", "target": "fire"}]
                ),
            }
        ),
    )

    train_raw, val_raw = load_raw_dataset(_cfg("github:francois-meyer/t2x"))

    assert train_raw[0]["source"] == "amanzi"
    assert val_raw[0]["target"] == "fire"
    assert format_instruction(train_raw[0])[1]["content"] == "water"


def test_afrihg_github_adapter_uses_subset_as_language(monkeypatch) -> None:
    seen: dict[str, list[str] | None] = {}

    def fake_load_afrihg(languages=None, cache_dir=None):  # noqa: ANN001
        seen["languages"] = languages
        return DatasetDict(
            {
                "train": Dataset.from_list(
                    [{"text": "headline body", "title": "Headline", "lang": "xho"}]
                ),
                "dev": Dataset.from_list(
                    [{"text": "dev body", "title": "Dev", "lang": "xho"}]
                ),
            }
        )

    monkeypatch.setattr(github, "load_afrihg_from_github", fake_load_afrihg)

    train_raw, val_raw = load_raw_dataset(_cfg("github:dadelani/AfriHG", subset="xho"))

    assert seen["languages"] == ["xho"]
    assert val_raw[0]["title"] == "Dev"
    assert format_instruction(train_raw[0])[1]["content"] == "Headline"


def test_masakhapos_adapter_loads_and_formats_pos_sample(monkeypatch) -> None:
    def fake_load_split(lang_code: str, split: str) -> Dataset:
        return masakhapos.parse_masakhapos_conll(
            "Ndiyahamba VERB\n. PUNCT\n\n",
            lang_code,
        )

    monkeypatch.setattr(masakhapos, "load_masakhapos_split", fake_load_split)

    train_raw, val_raw = load_raw_dataset(
        _cfg(
            "masakhane/masakhapos",
            languages=["xho", "zul"],
            task=FinetuneTaskType.POS_TAGGING,
        )
    )

    assert len(train_raw) == 2
    assert {row["lang"] for row in train_raw} == {"xho", "zul"}
    assert val_raw[0]["upos"] == ["VERB", "PUNCT"]
    assert "VERB" in format_pos(train_raw[0])[1]["content"]


class _MasakhaPOSResponse:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self) -> bytes:
        return b"Ndiyahamba VERB\n\n"


@pytest.mark.parametrize(
    "transient_error",
    [
        HTTPError("url", 408, "timeout", None, None),
        HTTPError("url", 429, "rate limited", None, None),
        HTTPError("url", 599, "server error", None, None),
        URLError("temporary transport failure"),
        TimeoutError(),
    ],
)
def test_masakhapos_retries_transient_download_failures(
    monkeypatch, transient_error
) -> None:
    requests = []

    def fake_urlopen(request, timeout):  # noqa: ANN001
        assert timeout == 30
        requests.append(request)
        if len(requests) == 1:
            raise transient_error
        return _MasakhaPOSResponse()

    monkeypatch.setattr(masakhapos, "urlopen", fake_urlopen)
    monkeypatch.setattr(masakhapos, "sleep", lambda _seconds: None)

    dataset = masakhapos.load_masakhapos_split("xho", "train")

    assert dataset[0]["tokens"] == ["Ndiyahamba"]
    assert len(requests) == 2
    assert requests[0].full_url == requests[1].full_url


def test_masakhapos_stops_after_three_transient_failures(monkeypatch) -> None:
    calls = 0

    def fake_urlopen(request, timeout):  # noqa: ANN001
        nonlocal calls
        calls += 1
        raise HTTPError(request.full_url, 503, "unavailable", None, None)

    monkeypatch.setattr(masakhapos, "urlopen", fake_urlopen)
    monkeypatch.setattr(masakhapos, "sleep", lambda _seconds: None)

    with pytest.raises(HTTPError):
        masakhapos.load_masakhapos_split("xho", "train")

    assert calls == 3


def test_masakhapos_uses_pinned_source_and_preserves_filename_fallback(
    monkeypatch,
) -> None:
    requests = []

    def fake_urlopen(request, timeout):  # noqa: ANN001
        assert timeout == 30
        requests.append(request)
        if request.full_url.endswith(
            "/dev.txt?ref=376f4161f0425584d4bd7664122b56fa026926d3"
        ):
            raise HTTPError(request.full_url, 404, "missing", None, None)
        return _MasakhaPOSResponse()

    monkeypatch.setattr(masakhapos, "urlopen", fake_urlopen)

    dataset = masakhapos.load_masakhapos_split("xho", "validation")

    assert dataset[0]["upos"] == ["VERB"]
    assert [request.full_url for request in requests] == [
        "https://api.github.com/repos/masakhane-io/masakhane-pos/contents/data/"
        "xho/dev.txt?ref=376f4161f0425584d4bd7664122b56fa026926d3",
        "https://api.github.com/repos/masakhane-io/masakhane-pos/contents/data/"
        "xho/validation.txt?ref=376f4161f0425584d4bd7664122b56fa026926d3",
    ]
    assert all(
        request.headers["Accept"] == "application/vnd.github.raw+json"
        for request in requests
    )


def test_huggingface_adapter_filters_subset_after_default_fallback(monkeypatch) -> None:
    calls: list[str | None] = []

    def fake_load(hf_name, name, train_split, val_split):  # noqa: ANN001
        calls.append(name)
        if name == "xho_Latn":
            raise RuntimeError("missing config")
        return (
            Dataset.from_list(
                [
                    {"text": "xho train", "lang": "xho_Latn"},
                    {"text": "eng train", "lang": "eng_Latn"},
                ]
            ),
            Dataset.from_list(
                [
                    {"text": "xho val", "lang": "xho_Latn"},
                    {"text": "eng val", "lang": "eng_Latn"},
                ]
            ),
        )

    monkeypatch.setattr(
        huggingface,
        "load_train_val_with_revision_fallback",
        fake_load,
    )

    train_raw, val_raw = load_raw_dataset(_cfg("Davlan/sib200", subset="xho_Latn"))

    assert calls == ["xho_Latn", None]
    assert train_raw["lang"] == ["xho_Latn"]
    assert val_raw["text"] == ["xho val"]


def test_masakhaner_data_files_are_language_specific() -> None:
    assert masakhaner_data_files("xho")["train"] == "data/xh/train.parquet"

    try:
        masakhaner_data_files("eng")
    except ValueError as err:
        assert "Supported languages: tsn, xho, zul" in str(err)
    else:
        raise AssertionError("Expected unsupported MasakhaNER language to fail")


def test_mix_component_raw_loading_uses_shared_adapter(monkeypatch) -> None:
    seen: list[str] = []
    train = Dataset.from_list([{"source": "a", "target": "b"}])
    validation = Dataset.from_list([{"source": "c", "target": "d"}])

    def fake_load_raw_dataset(ds_cfg: FinetuneDatasetConfig):
        seen.append(ds_cfg.hf_name)
        return train, validation

    monkeypatch.setattr(mix, "load_raw_dataset", fake_load_raw_dataset)

    train_raw, val_raw = mix._load_component_raw(  # noqa: SLF001
        _cfg("github:francois-meyer/t2x")
    )

    assert seen == ["github:francois-meyer/t2x"]
    assert train_raw is train
    assert val_raw is validation
