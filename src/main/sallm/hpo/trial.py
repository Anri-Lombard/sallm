from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import wandb
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf, open_dict

from sallm.config import ExperimentConfig, to_resolved_dict
from sallm.fine_tune.run import run as run_finetune

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CONF_ROOT = _REPO_ROOT / "src" / "conf"
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
    return [Path(normalized), _CONF_ROOT / normalized]


def _config_target_from_conf_path(path: Path) -> str | None:
    try:
        relative_path = path.resolve().relative_to(_CONF_ROOT.resolve())
    except ValueError:
        return None
    return str(relative_path.with_suffix(""))


def _unwrap_config_group(cfg: DictConfig) -> DictConfig:
    keys_in_cfg = list(cfg.keys())
    if (
        len(keys_in_cfg) == 1
        and keys_in_cfg[0] not in ExperimentConfig.__dataclass_fields__
    ):
        unwrapped = cfg[keys_in_cfg[0]]
        if not isinstance(unwrapped, DictConfig):
            raise TypeError("Base config group must resolve to a mapping.")
        return unwrapped
    return cfg


def _merge_experiment_config(cfg: DictConfig) -> DictConfig:
    schema = OmegaConf.structured(ExperimentConfig)
    merged = OmegaConf.merge(schema, _unwrap_config_group(cfg))
    if not isinstance(merged, DictConfig):
        raise TypeError("Base config must resolve to a mapping.")
    return merged


def _load_hydra_config_target(base_config: str) -> DictConfig | None:
    config_target = None
    for path in _candidate_base_config_paths(base_config):
        if not path.is_file():
            continue
        config_target = _config_target_from_conf_path(path)
        if config_target is not None:
            break

    if config_target is None:
        return None

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONF_ROOT), version_base=None):
        cfg = compose(config_name=config_target)
    if not isinstance(cfg, DictConfig):
        raise TypeError("Base config must resolve to a mapping.")
    return _merge_experiment_config(cfg)


def load_base_config(base_config: str) -> DictConfig:
    _register_env_resolver()
    hydra_cfg = _load_hydra_config_target(base_config)
    if hydra_cfg is not None:
        return hydra_cfg

    for path in _candidate_base_config_paths(base_config):
        if path.is_file():
            cfg = OmegaConf.load(path)
            if not isinstance(cfg, DictConfig):
                raise TypeError("Base config must resolve to a mapping.")
            return _merge_experiment_config(cfg)
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
    if isinstance(training, dict):
        base_out = training.get("output_dir")
        base_log = training.get("logging_dir")
        if run_id and base_out:
            training["output_dir"] = os.path.join(base_out, run_id)
        if run_id and base_log:
            training["logging_dir"] = os.path.join(base_log, run_id)


def _apply_updates(
    cfg: DictConfig, updates: dict[str, Any], run_id: str | None
) -> DictConfig:
    with open_dict(cfg):
        if run_id:
            if cfg.get("wandb") is None:
                cfg["wandb"] = {}
            cfg["wandb"]["id"] = f"sweep-{run_id}"
        _apply_training_paths(cfg, run_id)
        for dotted_key, value in updates.items():
            _set_by_dotted_key(cfg, dotted_key, value)
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
