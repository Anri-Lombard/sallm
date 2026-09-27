from pathlib import Path

import requests
from sallm.data.afrihg import load_afrihg_from_github


def test_cache_only_uses_existing_files_without_network(
    tmp_path: Path, monkeypatch
) -> None:
    for split in ("train", "dev", "test"):
        (tmp_path / f"xho_{split}.csv").write_text(
            "text,title\narticle,headline\n", encoding="utf-8"
        )

    def fail_network(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setenv("SALLM_AFRIHG_CACHE_ONLY", "1")
    monkeypatch.setattr(requests.Session, "get", fail_network)
    dataset = load_afrihg_from_github(["xho"], str(tmp_path))

    assert set(dataset) == {"train", "validation", "test"}
    assert all(len(dataset[split]) == 1 for split in dataset)
