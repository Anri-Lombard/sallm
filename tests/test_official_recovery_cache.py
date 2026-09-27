import importlib.util
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/cache_pure_gdn_official_recovery_assets.py"
)
SPEC = importlib.util.spec_from_file_location("official_recovery_cache", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
materialize = MODULE.materialize
count_pos_sentences = MODULE._count_pos_sentences


def test_materializes_only_frozen_recovery_assets(tmp_path: Path) -> None:
    news_cache = tmp_path / "news"
    metric_cache = tmp_path / "metrics"
    (news_cache / "hf").mkdir(parents=True)
    metric_modules = metric_cache / "hf" / "modules" / "evaluate_modules"
    metric_modules.mkdir(parents=True)
    (metric_modules / "__init__.py").write_text("")
    dataset_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    metric_calls: list[str] = []
    expected_lengths = [
        948,
        297,
        996,
        1000,
        1000,
        602,
        602,
        601,
        601,
        601,
        601,
    ]

    class FakeDataset:
        def __init__(
            self,
            length: int,
            lines: list[str] | None = None,
            cache_file: Path | None = None,
        ) -> None:
            self.length = length
            self.lines = lines
            self._fingerprint = f"fp-{length}"
            self.cache_files = (
                [{"filename": str(cache_file)}] if cache_file is not None else []
            )

        def __len__(self) -> int:
            return self.length

        def __getitem__(self, key: str) -> list[str]:
            assert key == "text" and self.lines is not None
            return self.lines

    def fake_dataset(*args, **kwargs):
        dataset_calls.append((args, kwargs))
        expected = expected_lengths[len(dataset_calls) - 1]
        if args[0] == "text":
            lines = [line for _ in range(expected) for line in ("word NOUN", "")]
            return FakeDataset(len(lines), lines)
        cache_file = None
        if args[0] == "anrilombard/masakhaner-x-parquet":
            cache_file = (
                tmp_path
                / "recovery/hf/datasets/ner"
                / f"online-{len(dataset_calls)}"
                / "0.0.0/hash/data.arrow"
            )
            cache_file.parent.mkdir(parents=True)
            cache_file.write_text("")
        return FakeDataset(expected, cache_file=cache_file)

    def fake_metric(name: str, **_kwargs) -> object:
        metric_calls.append(name)
        return object()

    result = materialize(
        tmp_path / "recovery",
        news_cache,
        metric_cache,
        fake_dataset,
        fake_metric,
    )

    assert len(dataset_calls) == 11
    assert metric_calls == ["bleu", "chrf", "f1"]
    assert result["metrics_computed"] is False
    assert set(result["datasets"]) == {
        "news/eng",
        "news/xho",
        "ner/tn",
        "ner/xh",
        "ner/zu",
        "pos/tsn",
        "pos/xho",
        "pos/zul",
    }


def test_materializes_from_a_sealed_news_cache(tmp_path: Path) -> None:
    news_cache = tmp_path / "news"
    metric_cache = tmp_path / "metrics"
    (news_cache / "hf").mkdir(parents=True)
    metric_modules = metric_cache / "hf" / "modules" / "evaluate_modules"
    metric_modules.mkdir(parents=True)
    (news_cache / "hf").chmod(0o555)
    lengths = iter([948, 297, 996, 1000, 1000, 602, 602, 601, 601, 601, 601])

    class FakeDataset:
        def __init__(
            self,
            length: int,
            lines: list[str] | None = None,
            cache_file: Path | None = None,
        ) -> None:
            self.length = length
            self.lines = lines
            self._fingerprint = f"fp-{length}"
            self.cache_files = (
                [{"filename": str(cache_file)}] if cache_file is not None else []
            )

        def __len__(self) -> int:
            return self.length

        def __getitem__(self, key: str) -> list[str]:
            assert key == "text" and self.lines is not None
            return self.lines

    calls = 0

    def fake_dataset(*args, **_kwargs):
        nonlocal calls
        calls += 1
        expected = next(lengths)
        if args[0] == "text":
            lines = [line for _ in range(expected) for line in ("word NOUN", "")]
            return FakeDataset(len(lines), lines)
        cache_file = None
        if args[0] == "anrilombard/masakhaner-x-parquet":
            cache_file = (
                tmp_path
                / "recovery/hf/datasets/ner"
                / f"online-{calls}"
                / "0.0.0/hash/data.arrow"
            )
            cache_file.parent.mkdir(parents=True)
            cache_file.write_text("")
        return FakeDataset(expected, cache_file=cache_file)

    materialize(
        tmp_path / "recovery",
        news_cache,
        metric_cache,
        fake_dataset,
        lambda *_args, **_kwargs: object(),
    )
    assert (tmp_path / "recovery" / "hf").stat().st_mode & 0o200


def test_counts_pos_sentences_like_the_frozen_task_loader() -> None:
    assert count_pos_sentences(["-DOCSTART-", "", "one NOUN", "two VERB", ""]) == 1
    assert count_pos_sentences(["one NOUN", "", "two VERB"]) == 2
