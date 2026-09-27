#!/usr/bin/env python3
"""Run `sallm.main` from an immutable code copy with optional offline and check hooks.

Hooks (all opt-in through environment variables):
  SALLM_PRELOAD_RUNNER + SALLM_PRELOAD_FN   import a runner module by path and call one
                                            of its install functions (v15 T2X loader).
  SALLM_HEXV3_RUNNER                        hex_v3 historical runner: sealed train/validation
                                            files, extended to one extra adapter id.
  SALLM_INJONGO_OFFLINE_SNAPSHOT            serve InjongoIntent URL reads from the local
                                            HF hub snapshot instead of the network.
  SALLM_DATA_CHECK=1                        build model, datasets and trainer, print their
                                            fingerprints and exit before training.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import runpy
import sys
from pathlib import Path
from urllib.error import HTTPError

INJONGO_PREFIXES = (
    "https://huggingface.co/datasets/masakhane/InjongoIntent/resolve/main/",
    "https://huggingface.co/datasets/masakhane/InjongoIntent/resolve/"
    "fe4be3882a1614161dfe231ec793197bb74f4b44/",
)


def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _install_injongo_offline(snapshot: Path) -> None:
    import sallm.data.loaders.huggingface as hf_loader

    def offline_urlopen(url, timeout=None):
        target = url if isinstance(url, str) else url.full_url
        for prefix in INJONGO_PREFIXES:
            if target.startswith(prefix):
                path = snapshot / target[len(prefix) :]
                if not path.is_file():
                    raise HTTPError(target, 404, "not in local snapshot", None, None)
                return io.BytesIO(path.read_bytes())
        raise RuntimeError(f"Offline run refused a network read: {target}")

    hf_loader.urlopen = offline_urlopen


def _install_clean_injongo(split_module_path: str, snapshot: Path) -> None:
    """Replace the HF loader for InjongoIntent with the July clean split (Feb code only)."""
    from datasets import Dataset, concatenate_datasets

    import sallm.data.factory as factory
    import sallm.data.loaders.huggingface as hf_loader

    split = _load_module(split_module_path, "july_injongointent_split")
    original = hf_loader.load_hf_dataset

    def read_rows(lang: str, name: str) -> list[dict]:
        path = snapshot / lang / name
        return [
            {**json.loads(line), "lang": lang}
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def load_hf_dataset(ds_cfg):
        if ds_cfg.hf_name != "masakhane/InjongoIntent":
            return original(ds_cfg)
        langs = list(ds_cfg.languages or []) or [ds_cfg.subset]
        train_parts, val_parts = [], []
        for lang in langs:
            train_rows = split.exclude_heldout_texts(
                read_rows(lang, "train.jsonl"), read_rows(lang, "test.jsonl")
            )
            train_rows, val_rows = split.split_injongointent_rows(train_rows)
            train_parts.append(Dataset.from_list(train_rows))
            val_parts.append(Dataset.from_list(val_rows))
        if len(train_parts) == 1:
            return train_parts[0], val_parts[0], False
        return concatenate_datasets(train_parts), concatenate_datasets(val_parts), False

    hf_loader.load_hf_dataset = load_hf_dataset
    factory.load_hf_dataset = load_hf_dataset


def _install_hexv3(runner_path: str) -> None:
    runner = _load_module(runner_path, "hexv3_runner")
    adapter_id = os.environ["SALLM_RECOVERY_ADAPTER_ID"]
    extra = json.loads(os.environ["SALLM_HEXV3_EXTRA_RECORD"])
    if extra["id"] != adapter_id:
        raise RuntimeError("SALLM_HEXV3_EXTRA_RECORD does not match the adapter id")
    original = runner.adapter_record

    def adapter_record(manifest, requested):
        if requested == adapter_id:
            return extra
        return original(manifest, requested)

    runner.adapter_record = adapter_record
    runner.validate_file_bindings(runner.load_manifest())
    os.environ["SALLM_RECOVERY_DATA_OFFLINE"] = "1"
    runner.install_loader()


def _fingerprint(ds) -> str:
    digest = hashlib.sha256()
    for row in ds:
        digest.update(json.dumps(row, sort_keys=True, default=str).encode("utf-8"))
    return digest.hexdigest()


def _install_data_check() -> None:
    import sallm.fine_tune.run as ft_run
    import transformers.training_args as training_args_module

    # CPU login-node check only: let TrainingArguments accept bf16 so the real trainer builds.
    training_args_module.is_torch_bf16_gpu_available = lambda: True

    real_build_trainer = ft_run.build_trainer

    def check_build_trainer(config, model, tokenizer, train_ds, val_ds):
        report = {
            "train_rows": len(train_ds),
            "val_rows": len(val_ds),
            "train_sha256": _fingerprint(train_ds),
            "val_sha256": _fingerprint(val_ds),
            "train_first": train_ds[0],
            "trainable_params": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
            "vocab": len(tokenizer),
            "chat_template_sha256": hashlib.sha256(
                (tokenizer.chat_template or "").encode("utf-8")
            ).hexdigest(),
        }
        try:
            trainer = real_build_trainer(config, model, tokenizer, train_ds, val_ds)
            args = trainer.args
            report["trainer_args"] = {
                key: str(getattr(args, key))
                for key in sorted(args.to_dict())
                if not key.startswith(("hub_", "push_to_hub"))
            }
            loader = trainer.get_train_dataloader()
            report["micro_batches_per_epoch"] = len(loader)
            report["optimizer_steps_per_epoch"] = max(
                len(loader) // args.gradient_accumulation_steps, 1
            )
            report["callbacks"] = [
                type(cb).__name__ for cb in trainer.callback_handler.callbacks
            ]
        except Exception as err:  # noqa: BLE001 - CPU login node cannot build every trainer
            report["trainer_build_error"] = f"{type(err).__name__}: {err}"
        destination = os.environ.get("SALLM_DATA_CHECK_OUT")
        encoded = json.dumps(report, indent=1, sort_keys=True, default=str)
        if destination:
            Path(destination).write_text(encoded + "\n", encoding="utf-8")
        print("DATA_CHECK " + encoded)
        raise SystemExit(0)

    ft_run.build_trainer = check_build_trainer


def main() -> None:
    preload = os.environ.get("SALLM_PRELOAD_RUNNER")
    if preload:
        module = _load_module(preload, "preload_runner")
        getattr(module, os.environ["SALLM_PRELOAD_FN"])()
    hexv3 = os.environ.get("SALLM_HEXV3_RUNNER")
    if hexv3:
        _install_hexv3(hexv3)
    snapshot = os.environ.get("SALLM_INJONGO_OFFLINE_SNAPSHOT")
    clean_split = os.environ.get("SALLM_CLEAN_INJONGO_SPLIT_MODULE")
    if clean_split:
        _install_clean_injongo(clean_split, Path(snapshot or ""))
    elif snapshot:
        _install_injongo_offline(Path(snapshot))
    if os.environ.get("SALLM_DATA_CHECK") == "1":
        _install_data_check()
    runpy.run_module("sallm.main", run_name="__main__")


if __name__ == "__main__":
    main()
