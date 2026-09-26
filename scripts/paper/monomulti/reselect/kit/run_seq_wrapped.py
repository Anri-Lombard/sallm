#!/usr/bin/env python3
"""Run the frozen General sequence runner (v2) with two optional, recorded adjustments.

SEQ_BATCH1=1          force lm-eval batch_size=1 (xLSTM ignores the attention mask, so
                      left-padded generate batches leak pad tokens into its state).
SEQ_VALIDATION_PROMPTS=protocol
                      in the validation phase score only the General protocol prompts
                      (NER tsn P2, xho P5, zul P5; POS P3) instead of all prompts; used
                      only for checkpoint selection on the validation split.
SEQ_LANGUAGES=tsn,... restrict the languages scored (validation selection only).
Everything else is the unmodified runner (sha 9cad0519...).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

RUNNER = Path("/scratch/lmbanr001/masters/sallm/results/general_sequence_official_test_20260917_v2/control/run_general_sequence_eval_20260917_v2.py")
RUNNER_SHA = "9cad0519cf65b0cd95c9f453b7ec441a68a3e89dde456e84a39856365fe026f5"
PROTOCOL_PROMPTS = {"ner": {"tsn": 2, "xho": 5, "zul": 5}, "pos": {"tsn": 3, "xho": 3, "zul": 3}}

assert hashlib.sha256(RUNNER.read_bytes()).hexdigest() == RUNNER_SHA
spec = importlib.util.spec_from_file_location("general_sequence_v2", RUNNER)
runner = importlib.util.module_from_spec(spec)
sys.modules["general_sequence_v2"] = runner
spec.loader.exec_module(runner)

adjustments = {}
if os.environ.get("SEQ_BATCH1") == "1":
    original = runner.evaluator.simple_evaluate

    def simple_evaluate_batch1(*args, **kwargs):
        kwargs["batch_size"] = 1
        kwargs["max_batch_size"] = 1
        return original(*args, **kwargs)

    runner.evaluator.simple_evaluate = simple_evaluate_batch1
    adjustments["lm_eval_batch_size"] = 1

languages = os.environ.get("SEQ_LANGUAGES")
if languages:
    runner.LANGUAGES = tuple(languages.split(","))
    adjustments["languages"] = list(runner.LANGUAGES)

if os.environ.get("SEQ_VALIDATION_PROMPTS") == "protocol":
    original_chosen = runner.chosen_prompts

    def chosen_protocol(task, phase, selection):
        if phase != "validation":
            return original_chosen(task, phase, selection)
        return {lang: [PROTOCOL_PROMPTS[task][lang]] for lang in runner.LANGUAGES}

    runner.chosen_prompts = chosen_protocol
    adjustments["validation_prompts"] = "protocol"

original_atomic = runner.atomic_json


def atomic_json_with_adjustments(path, value):
    if isinstance(value, dict):
        value = {**value, "wrapper_adjustments": adjustments, "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    return original_atomic(path, value)


runner.atomic_json = atomic_json_with_adjustments
print("WRAPPER_ADJUSTMENTS", json.dumps(adjustments), flush=True)
runner.main()
