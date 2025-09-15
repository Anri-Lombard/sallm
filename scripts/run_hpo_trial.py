import argparse
import os
from typing import Any

import wandb
from omegaconf import DictConfig, OmegaConf, open_dict
from sallm.config import ExperimentConfig
from sallm.fine_tune.run import run as run_finetune

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _register_env_resolver() -> None:
    if not OmegaConf.has_resolver("oc.env"):
        OmegaConf.register_new_resolver(
            "oc.env", lambda key, default=None: os.environ.get(key, default or "")
        )


def _load_base_cfg(base_cfg: str) -> DictConfig:
    _register_env_resolver()
    candidates: list[str] = []
    if base_cfg.endswith(".yaml"):
        candidates.append(base_cfg)
    else:
        candidates.append(base_cfg + ".yaml")
    candidates.append(os.path.join("src", "conf", base_cfg + ".yaml"))
    for p in candidates:
        if os.path.isfile(p):
            cfg = OmegaConf.load(p)
            schema = OmegaConf.structured(ExperimentConfig)
            return OmegaConf.merge(schema, cfg)
    raise FileNotFoundError(base_cfg)


def _set_by_dotted_key(cfg: DictConfig, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = cfg
    for k in parts[:-1]:
        if k not in node or node[k] is None:
            node[k] = {}
        node = node[k]
    node[parts[-1]] = value


def main() -> None:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--base-config", required=True)
    args, _ = parser.parse_known_args()

    cfg = _load_base_cfg(args.base_config)

    run = wandb.init()
    sweep_conf = dict(wandb.config)

    wandb_id = run.id if run else None

    updates: dict[str, Any] = {}
    for k, v in sweep_conf.items():
        updates[k] = v

    with open_dict(cfg):
        if wandb_id:
            if "wandb" not in cfg or cfg["wandb"] is None:
                cfg["wandb"] = {}
            cfg["wandb"]["id"] = f"sweep-{wandb_id}"
        if cfg.get("training") and isinstance(cfg["training"], dict):
            base_out = cfg["training"].get("output_dir")
            base_log = cfg["training"].get("logging_dir")
            if wandb_id and base_out:
                cfg["training"]["output_dir"] = os.path.join(base_out, wandb_id)
            if wandb_id and base_log:
                cfg["training"]["logging_dir"] = os.path.join(base_log, wandb_id)
        for dotted_key, val in updates.items():
            _set_by_dotted_key(cfg, dotted_key, val)

        tok_path = OmegaConf.select(cfg, "tokenizer.path")
        if isinstance(tok_path, str):
            if not os.path.isdir(tok_path):
                candidates = []
                env_tok = os.environ.get("TOKENIZER_PATH")
                if env_tok:
                    candidates.append(env_tok)
                candidates.append(
                    os.path.join(repo_root, "tokenizer", "sallm_bpe_tokenizer")
                )
                existing = next((p for p in candidates if p and os.path.isdir(p)), None)
                if existing:
                    cfg["tokenizer"]["path"] = existing
                else:
                    raise FileNotFoundError(tok_path)

    resolved = OmegaConf.to_container(cfg, resolve=True)
    cfg = OmegaConf.create(resolved)

    if run is not None:
        run.config.update(OmegaConf.to_container(cfg, resolve=True))

    run_finetune(cfg)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
