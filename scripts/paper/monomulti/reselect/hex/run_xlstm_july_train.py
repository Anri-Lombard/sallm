#!/usr/bin/env python3
"""Run sallm.main from the July xLSTM source copy with InjongoIntent read from the local hub snapshot.

The July loader fetched InjongoIntent JSONL files from
https://huggingface.co/datasets/masakhane/InjongoIntent/resolve/main/<lang>/<split>.jsonl
with urlopen. HEX jobs run offline, so the same files are read from the cached hub
snapshot through a file:// base URL. Nothing else is changed.
"""

from __future__ import annotations

import os
import runpy

import sallm.data.loaders.huggingface as hf_loader

hf_loader.INJONGOINTENT_BASE_URL = os.environ["SALLM_INJONGOINTENT_BASE_URL"]
runpy.run_module("sallm.main", run_name="__main__", alter_sys=True)
