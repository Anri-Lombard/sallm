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
