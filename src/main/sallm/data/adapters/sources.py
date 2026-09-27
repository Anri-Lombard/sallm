from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from time import sleep
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def read_url(url: str | Request) -> bytes:
    """Read a source file, retrying only transient transport failures.

    With SALLM_SOURCE_CACHE_DIR set, commit-pinned URLs (immutable content) are
    read from and written to that directory (file name = sha256 of the URL).
    GitHub's API allows 60 unauthenticated requests per hour per IP, which
    parallel cluster jobs exhaust (HTTP 403 "rate limit exceeded").
    """
    full_url = url.full_url if isinstance(url, Request) else url
    cache_dir = os.environ.get("SALLM_SOURCE_CACHE_DIR")
    cache = None
    if cache_dir and re.search(r"[0-9a-f]{40}", full_url):
        cache = Path(cache_dir) / hashlib.sha256(full_url.encode()).hexdigest()
        if cache.is_file():
            return cache.read_bytes()
    data = _fetch_url(url)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_name(f"{cache.name}.{os.getpid()}.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, cache)
    return data


def _fetch_url(url: str | Request) -> bytes:
    for attempt in range(3):
        try:
            with urlopen(url, timeout=30) as response:
                return response.read()
        except HTTPError as err:
            transient = err.code in {408, 429} or 500 <= err.code < 600
            if not transient or attempt == 2:
                raise
        except (URLError, TimeoutError):
            if attempt == 2:
                raise
        sleep(2**attempt)
    raise AssertionError("unreachable")
