#!/usr/bin/env python3
"""Fail-closed one-time test entry point for the General NER/POS runner."""

from __future__ import annotations

import run_general_sequence_eval_20260914 as runner

ORIGINAL_PARSE_ARGS = runner.parse_args


def parse_test_args():  # type: ignore[no-untyped-def]
    args = ORIGINAL_PARSE_ARGS()
    if args.self_check:
        return args
    if args.phase != "test":
        raise ValueError("This entry point permits held-out test only")
    if args.selection is None or args.release is None:
        raise ValueError("Test requires frozen selection and task release")
    if args.limit is not None:
        raise ValueError("Official test forbids partial limits")
    return args


if __name__ == "__main__":
    runner.parse_args = parse_test_args
    runner.main()
