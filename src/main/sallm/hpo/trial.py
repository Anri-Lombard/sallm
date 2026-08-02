from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import wandb
from omegaconf import DictConfig, OmegaConf, open_dict

from sallm.config import ExperimentConfig, to_resolved_dict
from sallm.fine_tune.run import run as run_finetune

_REPO_ROOT = Path(__file__).resolve().parents[4]
_TOKENIZER_SUBDIR = Path("tokenizer") / "sallm_bpe_tokenizer"


def _register_env_resolver() -> None:
    if not OmegaConf.has_resolver("oc.env"):
        OmegaConf.register_new_resolver(
            "oc.env", lambda key, default=None: os.environ.get(key, default or "")
        )


def _normalize_base_config_path(base_config: str) -> str:
    if base_config.endswith(".yaml"):
        return base_config
    return f"{base_config}.yaml"


def _candidate_base_config_paths(base_config: str) -> list[Path]:
    normalized = _normalize_base_config_path(base_config)
    return [Path(normalized), _REPO_ROOT / "src" / "conf" / normalized]


def load_base_config(base_config: str) -> DictConfig:
    _register_env_resolver()
    for path in _candidate_base_config_paths(base_config):
        if path.is_file():
            cfg = OmegaConf.load(path)
            schema = OmegaConf.structured(ExperimentConfig)
            merged = OmegaConf.merge(schema, cfg)
            if not isinstance(merged, DictConfig):
                raise TypeError("Base config must resolve to a mapping.")
            return merged
    raise FileNotFoundError(base_config)


def resolve_tokenizer_path(requested: str | None) -> str:
    candidates: list[Path] = []
    if requested:
        candidates.append(Path(requested))
    env_path = os.environ.get("TOKENIZER_PATH")
    if env_path:
        candidates.append(Path(env_path))
    home = os.environ.get("HOME")
    if home:
        candidates.append(Path(home) / "masters" / "sallm" / _TOKENIZER_SUBDIR)
    scratch = os.environ.get("SCRATCH")
    if scratch:
        candidates.append(Path(scratch) / "masters" / "sallm" / _TOKENIZER_SUBDIR)
    candidates.append(Path.cwd() / _TOKENIZER_SUBDIR)
    repo_candidate = _REPO_ROOT / _TOKENIZER_SUBDIR
    if repo_candidate not in candidates:
        candidates.append(repo_candidate)
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_dir():
            return str(candidate)
    raise FileNotFoundError(requested or "tokenizer path")


def _set_by_dotted_key(cfg: DictConfig, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node: Any = cfg
    for key in keys[:-1]:
        if key not in node or node[key] is None:
            node[key] = {}
        node = node[key]
    node[keys[-1]] = value


def _apply_training_paths(cfg: DictConfig, run_id: str | None) -> None:
    training = cfg.get("training")
    if training is None:
        return
    base_out = training.get("output_dir")
    base_log = training.get("logging_dir")
    if run_id and base_out:
        training["output_dir"] = os.path.join(base_out, run_id)
    if run_id and base_log:
        training["logging_dir"] = os.path.join(base_log, run_id)


def _apply_derived_updates(cfg: DictConfig, updates: dict[str, Any]) -> None:
    rank = updates.pop("hpo.lora_rank", None)
    if rank is not None:
        rank = int(rank)
        _set_by_dotted_key(cfg, "peft.kwargs.r", rank)
        _set_by_dotted_key(cfg, "peft.kwargs.lora_alpha", 2 * rank)

    effective_batch = updates.pop("hpo.effective_batch_size", None)
    if effective_batch is not None:
        microbatch = int(cfg.training.per_device_train_batch_size)
        effective_batch = int(effective_batch)
        if effective_batch % microbatch:
            raise ValueError(
                "hpo.effective_batch_size must be divisible by "
                "training.per_device_train_batch_size"
            )
        cfg.training.gradient_accumulation_steps = effective_batch // microbatch


def _apply_updates(
    cfg: DictConfig, updates: dict[str, Any], run_id: str | None
) -> DictConfig:
    updates = dict(updates)
    with open_dict(cfg):
        if run_id:
            if cfg.get("wandb") is None:
                cfg["wandb"] = {}
            cfg["wandb"]["id"] = f"sweep-{run_id}"
        derived_updates = {
            key: updates.pop(key)
            for key in ("hpo.lora_rank", "hpo.effective_batch_size")
            if key in updates
        }
        for dotted_key, value in updates.items():
            _set_by_dotted_key(cfg, dotted_key, value)
        _apply_derived_updates(cfg, derived_updates)
        _apply_training_paths(cfg, run_id)
        if cfg.get("hub") is not None:
            cfg["hub"]["enabled"] = False
            cfg["hub"]["push_adapter"] = False
            cfg["hub"]["push_merged"] = False
        token_path = OmegaConf.select(cfg, "tokenizer.path")
        if isinstance(token_path, str):
            resolved = resolve_tokenizer_path(token_path)
            cfg["tokenizer"]["path"] = resolved
    return cfg


def _to_experiment_config(cfg: DictConfig) -> ExperimentConfig:
    obj = OmegaConf.to_object(cfg)
    if isinstance(obj, ExperimentConfig):
        return obj
    raise TypeError("Failed to build ExperimentConfig")


def run_trial(base_config: str) -> None:
    cfg = load_base_config(base_config)
    run = wandb.init()
    sweep_values = dict(wandb.config)
    run_id = run.id if run else None
    prepared = _apply_updates(cfg, sweep_values, run_id)
    resolved = to_resolved_dict(prepared, name="prepared config")
    clean_cfg = OmegaConf.create(resolved)
    structured = OmegaConf.merge(OmegaConf.structured(ExperimentConfig), clean_cfg)
    if not isinstance(structured, DictConfig):
        raise TypeError("Structured sweep config must resolve to a mapping.")
    config_obj = _to_experiment_config(structured)
    if run:
        run.config.update(resolved)
    run_finetune(config_obj)
