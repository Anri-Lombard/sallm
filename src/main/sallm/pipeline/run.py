from __future__ import annotations
import tempfile, yaml, os
from pathlib import Path
from sallm.config import ExperimentConfig, load_experiment_config
from sallm.fine_tune.run import run as run_ft
from sallm.evaluation.run import run as run_ev
from sallm.templates import registry as tmpl


def _dump(cfg, tmp: tempfile.NamedTemporaryFile):
    yaml.safe_dump(cfg, tmp)
    tmp.flush()
    return tmp.name


def run(cfg: ExperimentConfig):
    pipe = cfg.pipeline
    for lang in pipe.languages:

        with open(pipe.finetune_base_cfg, "r") as f:
            ft_cfg = yaml.safe_load(f)

        ft_cfg["model"]["init_checkpoint"] = pipe.base_checkpoint
        ft_cfg["dataset"]["subset"] = lang
        ft_cfg["wandb"]["name"] = f"ft-{lang}"

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as tmp:
            ft_path = _dump(ft_cfg, tmp)

        run_ft(load_experiment_config(ft_path))
        ckpt_dir = Path(
            ft_cfg["training"]["output_dir"]
            or os.path.join(os.environ.get("SCRATCH", "/tmp"), "ft", lang)
        )
        final_ckpt = ckpt_dir / "final_model"

        with open(pipe.eval_stub_cfg, "r") as f:
            ev_cfg = yaml.safe_load(f)

        ev_cfg["eval_model"]["checkpoint"] = str(final_ckpt)
        ev_cfg["evaluation"]["task_packs"] = [f"masakhanews_{lang}"]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as tmp:
            ev_path = _dump(ev_cfg, tmp)

        run_ev(load_experiment_config(ev_path))
