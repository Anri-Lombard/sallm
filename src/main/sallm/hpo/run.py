from __future__ import annotations

import argparse

from sallm.hpo.trial import resolve_tokenizer_path, run_trial


def main() -> None:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--base-config")
    parser.add_argument("--tokenizer-path", action="store_true")
    args = parser.parse_args()
    if args.tokenizer_path:
        try:
            path = resolve_tokenizer_path(None)
        except FileNotFoundError:
            return
        if path:
            print(path)
        return
    if not args.base_config:
        parser.error("--base-config is required unless --tokenizer-path is supplied")
    run_trial(args.base_config)


if __name__ == "__main__":
    main()
