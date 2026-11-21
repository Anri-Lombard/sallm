"""Repository locations. The library runs from a checkout: conf/, recipes/
and tokenizer/ are not shipped in the wheel."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CONF_DIR = REPO_ROOT / "src" / "conf"
