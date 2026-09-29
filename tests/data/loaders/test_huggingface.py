from pathlib import Path
from urllib.error import HTTPError

from sallm.data.loaders import huggingface


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self) -> bytes:
        return b"stable"


def test_read_url_retries_transient_http_failure(monkeypatch) -> None:
    calls = 0

    def urlopen(_url, timeout):
        nonlocal calls
        assert timeout == 30
        calls += 1
        if calls == 1:
            raise HTTPError("url", 503, "temporary", None, None)
        return _Response()

    monkeypatch.setattr(huggingface, "urlopen", urlopen)
    monkeypatch.setattr(huggingface, "sleep", lambda _seconds: None)

    assert huggingface._read_url("url") == b"stable"
    assert calls == 2


def test_masakhapos_source_is_commit_pinned() -> None:
    assert len(huggingface.MASAKHAPOS_REVISION) == 40


def test_hf_training_sources_are_commit_pinned() -> None:
    assert huggingface.DATASET_REVISIONS == {
        "masakhane/masakhanews": "fa3b5fff8a91d187bf0c5900a39c4271d08cf7fe",
        "anrilombard/masakhaner-x-parquet": "6aa65cdbfa22d66e5b4ed176ac525c364cda08d1",
        "Davlan/sib200": "38977a667f6fc264d5c26ec57a01e16db040b358",
        "anrilombard/nchlt-ner-sa4": "833a02ee599b37cd393fa25f35a6766421e19c5a",
        "anrilombard/nchlt-pos-sa4": "0a760eb7900ce533f60d710dc1d71d39121ddc03",
    }
    assert "/resolve/main" not in huggingface.INJONGOINTENT_BASE_URL


def test_masakhapos_official_tasks_use_commit_pinned_sources() -> None:
    root = (
        Path(__file__).resolve().parents[3]
        / "src/conf/eval/lm_eval_tasks/masakhapos_test"
    )
    tasks = sorted(root.glob("sallm_masakhapos_*_prompt_*.yaml"))

    assert len(tasks) == 12
    for task in tasks:
        source = task.read_text(encoding="utf-8")
        assert f"/raw/{huggingface.MASAKHAPOS_REVISION}/" in source
        assert "/raw/main/" not in source


def test_masakhaner_validation_split_is_loaded_at_the_pinned_revision(
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    from datasets import Dataset

    calls = []

    def load_dataset(path, **kwargs):
        calls.append((path, kwargs))
        return Dataset.from_list([{"tokens": ["a"], "ner_tags": [0]}])

    monkeypatch.setattr(huggingface, "load_dataset", load_dataset)
    monkeypatch.setattr("sallm.data.loaders.base.load_dataset", load_dataset)
    cfg = SimpleNamespace(
        splits={"train": "train", "val": "validation"},
        languages=["xho"],
        subset=None,
        hf_name="masakhane/masakhaner2",
    )
    monkeypatch.setattr(huggingface, "_requested_languages", lambda *_: ["xho"])
    huggingface._load_masakhaner_dataset(cfg)
    val = [kw for _, kw in calls if kw.get("split") == "validation"]
    assert (
        val
        and val[0]["revision"]
        == huggingface.DATASET_REVISIONS[huggingface.MASAKHANER_PARQUET_DATASET]
    )
    assert val[0].get("name") is None


def test_read_url_caches_commit_pinned_sources(monkeypatch, tmp_path) -> None:
    calls = []
    monkeypatch.setattr(
        huggingface, "urlopen", lambda url, timeout: calls.append(url) or _Response()
    )
    monkeypatch.setenv("SALLM_SOURCE_CACHE_DIR", str(tmp_path))
    pinned = f"https://example.org/x?ref={huggingface.MASAKHAPOS_REVISION}"
    assert huggingface._read_url(pinned) == b"stable"
    assert huggingface._read_url(pinned) == b"stable"
    assert len(calls) == 1
    huggingface._read_url("https://example.org/main/x")
    huggingface._read_url("https://example.org/main/x")
    assert len(calls) == 3
