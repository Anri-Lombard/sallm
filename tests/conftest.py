from pathlib import Path

from huggingface_hub import snapshot_download

TOKENIZER_DIR = (
    Path(__file__).resolve().parents[1] / "tokenizer" / "sallm_bpe_tokenizer"
)


def pytest_configure() -> None:
    # The paper tokenizer is published with MzansiLM rather than committed.
    if not (TOKENIZER_DIR / "tokenizer.json").exists():
        snapshot_download(
            "uctnlp/mzansilm-125m",
            allow_patterns=[
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
            ],
            local_dir=TOKENIZER_DIR,
        )
