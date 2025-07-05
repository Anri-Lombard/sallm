# Directory Structure
```
sallm/
  data/
    factory.py
    utils.py
  evaluation/
    config.py
    harness.py
    registry.py
    run.py
  fine_tune/
    run.py
  models/
    factory.py
    registry.py
  pipeline/
    run.py
  templates/
    registry.py
  training/
    factory.py
    run.py
    trainer.py
  __init__.py
  config.py
  main.py
  utils.py
```

# Files

## File: sallm/data/factory.py
```python
from __future__ import annotations

from typing import Optional, Tuple

from datasets import Dataset, DatasetDict, load_from_disk, load_dataset
from transformers import AutoTokenizer

from sallm.config import ExperimentConfig, RunMode
from sallm.data.utils import make_example_mapper


def build_datasets(
    config: ExperimentConfig, is_hpo: bool
) -> Tuple[Dataset, Dataset, Optional[Dataset]]:
    if config.mode == RunMode.FINETUNE:
        assert config.dataset, "Finetune mode requires a `dataset` block in the config."

        ds_cfg = config.dataset
        split_map = ds_cfg.splits

        # TODO: specify split in config rather
        train_raw = load_dataset(
            ds_cfg.hf_name,
            ds_cfg.subset,
            split=split_map["train"],
            trust_remote_code=True,
        )
        # TODO: specify split in config rather
        val_raw = load_dataset(
            ds_cfg.hf_name,
            ds_cfg.subset,
            split=split_map["val"],
            trust_remote_code=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.path)

        train_ds = _build_finetune_dataset(train_raw, config, tokenizer)
        val_ds = _build_finetune_dataset(val_raw, config, tokenizer)
        return train_ds, val_ds, None

    data_conf = config.data
    dataset_dict = load_from_disk(data_conf.path)

    if not isinstance(dataset_dict, DatasetDict):
        raise TypeError(
            f"Expected data at {data_conf.path} to be a DatasetDict, "
            f"but found {type(dataset_dict)}"
        )

    train_ds = dataset_dict[data_conf.train_split]
    val_ds = dataset_dict[data_conf.eval_split]

    test_ds = None
    if not is_hpo and data_conf.test_split and data_conf.test_split in dataset_dict:
        test_ds = dataset_dict[data_conf.test_split]

    return train_ds, val_ds, test_ds


def _build_finetune_dataset(
    raw_ds: Dataset,
    cfg: ExperimentConfig,
    tokenizer: AutoTokenizer,
) -> Dataset:
    ds_cfg = cfg.dataset
    mapper = make_example_mapper(ds_cfg, tokenizer)
    processed_ds = raw_ds.map(
        mapper,
        batched=False,
        remove_columns=raw_ds.column_names,
        desc="Mapping prompt+label → LM inputs",
    )

    if ds_cfg.subset:
        lang_column = [ds_cfg.subset] * len(processed_ds)
        processed_ds = processed_ds.add_column("lang", lang_column)

    return processed_ds
```

## File: sallm/data/utils.py
```python
from __future__ import annotations
from typing import Any, Dict, Callable, List
import random
from transformers import PreTrainedTokenizerBase
from sallm.config import FinetuneDatasetConfig, TemplateChoice
from sallm.templates import registry as tmpl


def make_example_mapper(
    ds_cfg: FinetuneDatasetConfig, tokenizer: PreTrainedTokenizerBase
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    templates = [tmpl.get(t.id) for t in ds_cfg.templates]
    weights = [t.weight for t in ds_cfg.templates]

    # if ds_cfg.template_choice == TemplateChoice.SINGLE:
    #     templates = [random.choices(templates, weights)[0]]

    numeric_keys = isinstance(next(iter(templates[0].label_mapping.keys())), int)

    def encode(ex: Dict[str, Any], template) -> Dict[str, Any]:
        prompt_kwargs = {col: ex[col] for col in ds_cfg.text_columns}
        prompt_text = template.prompt.format(**prompt_kwargs)
        raw_label = ex[ds_cfg.label_column]
        if numeric_keys:
            label_text = template.label_mapping[int(raw_label)]
        else:
            label_text = template.label_mapping[raw_label]
        full_text = f"{prompt_text} {label_text}{tokenizer.eos_token}"
        enc = tokenizer(
            full_text,
            truncation=True,
            max_length=ds_cfg.max_seq_length,
            padding="max_length",
        )
        labels = enc["input_ids"].copy()
        enc["labels"] = [
            (tok if tok != tokenizer.pad_token_id else -100) for tok in labels
        ]
        return enc

    if ds_cfg.template_choice == TemplateChoice.ALL:

        def mapper(ex):
            outs: List[Dict[str, Any]] = []
            for t in templates:
                outs.append(encode(ex, t))
            return outs

        return mapper

    if ds_cfg.template_choice == TemplateChoice.CYCLE:

        def mapper(ex, _counter=[0]):
            t = templates[_counter[0] % len(templates)]
            _counter[0] += 1
            return encode(ex, t)

        return mapper

    def mapper(ex):
        t = random.choices(templates, weights)[0]
        return encode(ex, t)

    return mapper
```

## File: sallm/evaluation/config.py
```python
from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field

__all__ = ["TaskPack"]


class TaskPack(BaseModel):
    name: str
    tasks: List[str]
    fewshot: int = 0
    batch_size: int = 8
    lm_eval_kwargs: Dict[str, Any] = Field(default_factory=dict)

    def to_lm_eval_kwargs(self) -> Dict[str, Any]:
        base = {
            "tasks": self.tasks,
            "batch_size": self.batch_size,
            "num_fewshot": self.fewshot,
        }
        base.update(self.lm_eval_kwargs)
        return base
```

## File: sallm/evaluation/harness.py
```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import lm_eval

from sallm.config import ModelEvalConfig
from sallm.evaluation.config import TaskPack

SUPPORTED_LANGS: List[str] = [
    "afr",
    "xho",
    "zul",
    "nso",
    "sot",
    "ssw",
    "tsn",
    "tso",
    "ven",
    "eng",
    "nbl",
]


def _filter_tasks_by_lang(task_names: List[str]) -> List[str]:
    return [t for t in task_names if t.split("_")[-1] in SUPPORTED_LANGS]


def evaluate_pack(
    pack: TaskPack,
    model_cfg: ModelEvalConfig,
    out_dir: Path,
    overrides: Dict[str, Dict],
) -> Dict:
    pack_over = overrides.get(pack.name, {})
    task_list = pack_over.get("tasks", pack.tasks)
    task_list = _filter_tasks_by_lang(task_list)

    fewshot = pack_over.get("fewshot", pack.fewshot)
    batch_size = pack_over.get("batch_size", pack.batch_size)

    model_args = f"pretrained={model_cfg.checkpoint},dtype={model_cfg.dtype},device={model_cfg.device}"

    eval_kwargs = {
        "model": model_cfg.adapter,
        "model_args": model_args,
        "tasks": task_list,
        "batch_size": batch_size,
        "num_fewshot": fewshot,
        "verbosity": "ERROR",
    }
    eval_kwargs.update(pack.lm_eval_kwargs)

    results = lm_eval.evaluate(**eval_kwargs)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pack.name}.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)

    return results
```

## File: sallm/evaluation/registry.py
```python
from __future__ import annotations

from pathlib import Path
from typing import Dict

import yaml
from pydantic import ValidationError

from sallm.evaluation.config import TaskPack

TASK_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent.parent
    / "conf"
    / "eval"
    / "tasks"
)

_CACHE: Dict[str, TaskPack] = {}


def load_task_pack(key: str) -> TaskPack:
    """Load a *TaskPack* by name and fail fast if YAML is incomplete or malformed."""
    if key in _CACHE:
        return _CACHE[key]

    yaml_path = TASK_DIR / f"{key}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Task‑pack YAML '{yaml_path}' not found.")

    with yaml_path.open("r") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise ValueError(f"Task‑pack YAML '{yaml_path}' is empty.")

    cfg["name"] = key

    try:
        pack = TaskPack(**cfg)
    except ValidationError as e:
        raise ValueError(f"Invalid Task‑pack YAML '{yaml_path}': {e}") from e

    _CACHE[key] = pack
    return pack
```

## File: sallm/evaluation/run.py
```python
from __future__ import annotations
import logging
import json
from pathlib import Path
from typing import Dict

from sallm.config import ExperimentConfig
from sallm.evaluation.harness import evaluate_pack
from sallm.evaluation.registry import load_task_pack

logger = logging.getLogger(__name__)


def run(config: ExperimentConfig) -> None:
    assert (
        config.evaluation and config.eval_model
    ), "`evaluation` and `eval_model` blocks required."

    eval_cfg = config.evaluation
    model_cfg = config.eval_model
    out_root = Path(eval_cfg.output_dir)
    overrides = eval_cfg.overrides or {}

    for pack_name in eval_cfg.task_packs:
        pack = load_task_pack(pack_name)
        logger.info(
            f"Evaluating task-pack '{pack_name}' with {len(pack.tasks)} tasks …"
        )
        results: Dict = evaluate_pack(pack, model_cfg, out_root, overrides)
        logger.info(json.dumps(results["results"], indent=2))

    logger.info("Evaluation done.")
```

## File: sallm/fine_tune/run.py
```python
from __future__ import annotations
import os
import logging
import torch

from sallm.config import ExperimentConfig
from sallm.data.factory import build_datasets
from sallm.models.factory import build_model, build_tokenizer
from sallm.training.factory import build_trainer

logger = logging.getLogger(__name__)


# TODO: improve naming
# TODO: no defaults for loraconfig, specify in config files
def _apply_peft_if_needed(model, peft_cfg):
    if not peft_cfg or peft_cfg.method == "none":
        return model
    import peft

    if peft_cfg.method.lower() in {"lora", "qlora"}:
        lora_conf = peft.LoraConfig(
            r=peft_cfg.kwargs.get("r", 64),
            lora_alpha=peft_cfg.kwargs.get("lora_alpha", 16),
            lora_dropout=peft_cfg.kwargs.get("lora_dropout", 0.05),
            target_modules=peft_cfg.kwargs.get("target_modules", ["q_proj", "v_proj"]),
            bias="none",
            task_type=peft.TaskType.CAUSAL_LM,
        )
        return peft.get_peft_model(model, lora_conf)
    raise ValueError(f"Unsupported PEFT method '{peft_cfg.method}'")


def run(config: ExperimentConfig) -> None:
    if config.wandb.project:
        os.environ["WANDB_PROJECT"] = config.wandb.project
    if config.wandb.name:
        os.environ["WANDB_RUN_NAME"] = config.wandb.name
    if config.wandb.id:
        os.environ["WANDB_RUN_ID"] = config.wandb.id
        os.environ["WANDB_RESUME"] = "allow"
    if config.training and config.wandb and config.wandb.name:
        config.training["run_name"] = config.wandb.name

    logger.info("Tokenizer …")
    tokenizer = build_tokenizer(config)

    if tokenizer.pad_token is None:
        logger.info("Tokenizer `pad_token` not set. Setting it to `eos_token`.")
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Model …")
    model = build_model(config, tokenizer)
    model = _apply_peft_if_needed(model, config.peft)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    logger.info("Datasets …")
    train_ds, val_ds, _ = build_datasets(config, is_hpo=False)
    logger.info(f"Samples: train={len(train_ds)}, val={len(val_ds)}")

    if train_ds:
        logger.info("--- Inspecting a single training sample ---")
        sample = train_ds[0]
        logger.info(f"Input IDs: {sample['input_ids']}")
        logger.info(f"Decoded Text: {tokenizer.decode(sample['input_ids'])}")
        # In Causal LM fine-tuning, labels are typically a copy of input_ids
        logger.info(f"Labels: {sample['labels']}")
        logger.info("-------------------------------------------")

    trainer = build_trainer(config, model, tokenizer, train_ds, val_ds)
    resume_ckpt = config.training.get("resume_from_checkpoint")

    logger.info("Fine-tuning start …")
    torch.autograd.set_detect_anomaly(mode=True, check_nan=True)
    trainer.train(resume_from_checkpoint=resume_ckpt)
    logger.info("Fine-tuning done.")

    final_path = os.path.join(trainer.args.output_dir, "final_model")
    trainer.save_model(final_path)
    logger.info(f"Final model saved → {final_path}")
```

## File: sallm/models/factory.py
```python
import logging

from transformers import AutoModelForCausalLM, AutoTokenizer

from sallm.config import ExperimentConfig
from sallm.models.registry import MODEL_CONFIG_REGISTRY, MODEL_CLASS_REGISTRY
from sallm.utils import count_trainable_parameters

logger = logging.getLogger(__name__)


def build_tokenizer(config: ExperimentConfig) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(config.tokenizer.path)


def build_model(
    config: ExperimentConfig, tokenizer: AutoTokenizer
) -> AutoModelForCausalLM:
    model_conf = config.model
    model_class = MODEL_CLASS_REGISTRY.get(model_conf.architecture)

    if not model_class:
        raise ValueError(f"Unsupported model architecture: {model_conf.architecture}")

    if getattr(model_conf, "init_checkpoint", None):
        logger.info(
            f"Loading model of type {model_class.__name__} from checkpoint: {model_conf.init_checkpoint}"
        )
        model = model_class.from_pretrained(model_conf.init_checkpoint)
        return model

    config_class = MODEL_CONFIG_REGISTRY[model_conf.architecture]
    if model_conf.config is None:
        raise ValueError(
            "`model.config` is required when `init_checkpoint` is not provided."
        )

    model_config_obj = config_class(**model_conf.config)
    model_config_obj.vocab_size = len(tokenizer)

    model = model_class(model_config_obj)

    if model_conf.param_validation:
        num_params = count_trainable_parameters(model)
        num_params_m = num_params / 1_000_000

        min_p = model_conf.param_validation.min_params_m
        max_p = model_conf.param_validation.max_params_m

        logger.info(f"Validating model size: {num_params_m:.2f}M parameters.")

        if not (min_p <= num_params_m <= max_p):
            raise ValueError(
                f"Model size validation failed! "
                f"Expected between {min_p}M and {max_p}M parameters, "
                f"but got {num_params_m:.2f}M."
            )
        logger.info("Model size validation passed.")

    return model
```

## File: sallm/models/registry.py
```python
from typing import Dict, Type

from transformers import (
    LlamaConfig,
    PretrainedConfig,
    Mamba2Config,
    LlamaForCausalLM,
    Mamba2ForCausalLM,
)

# TODO add mamba
# TODO consider mixtral?
MODEL_CONFIG_REGISTRY: Dict[str, Type[PretrainedConfig]] = {
    "llama": LlamaConfig,
    "mamba2": Mamba2Config,
}

MODEL_CLASS_REGISTRY: Dict[str, Type[PretrainedConfig]] = {
    "llama": LlamaForCausalLM,
    "mamba2": Mamba2ForCausalLM,
}
```

## File: sallm/pipeline/run.py
```python
from __future__ import annotations
import hydra

# from omegaconf import DictConfig

from sallm.config import ExperimentConfig
from sallm.fine_tune.run import run as run_ft
from sallm.evaluation.run import run as run_ev


def run(cfg: ExperimentConfig):
    pipe = cfg.pipeline
    for lang in pipe.languages:
        with hydra.initialize(
            config_path="../../conf", job_name=f"sallm-pipeline-{lang}"
        ):
            ft_cfg = hydra.compose(
                config_name=pipe.finetune_base_cfg,
                overrides=[
                    f"model.init_checkpoint={pipe.base_checkpoint}",
                    f"dataset.subset={lang}",
                    f"wandb.name=ft-{lang}",
                ],
            )
            run_ft(hydra.utils.instantiate(ft_cfg, _convert_="all"))

        with hydra.initialize(
            config_path="../../conf", job_name=f"sallm-pipeline-{lang}-eval"
        ):
            ev_cfg = hydra.compose(
                config_name=pipe.eval_stub_cfg,
                overrides=[
                    "eval_model.checkpoint=???",  # TODO: get from ft_cfg
                    f"evaluation.task_packs=[masakhanews_{lang}]",
                ],
            )
            run_ev(hydra.utils.instantiate(ev_cfg, _convert_="all"))
```

## File: sallm/templates/registry.py
```python
from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import yaml
from pydantic import BaseModel

_TEMPLATE_ROOT = (
    Path(__file__).resolve().parent.parent.parent.parent / "conf" / "templates"
)
_CACHE: Dict[str, "TemplateSpec"] = {}
_TASK_INDEX: Dict[str, List[str]] = {}


class TemplateSpec(BaseModel):
    id: str
    prompt: str
    label_mapping: Dict[int | str, str]
    task: str


def _load_all() -> None:
    if not _TEMPLATE_ROOT.exists():
        print(f"ERROR: Template directory not found at {_TEMPLATE_ROOT}")
        return

    for path in _TEMPLATE_ROOT.rglob("*.yaml"):
        with path.open("r") as f:
            cfg = yaml.safe_load(f)

        relative_path = path.relative_to(_TEMPLATE_ROOT)
        template_id = str(relative_path.with_suffix(""))

        cfg["id"] = template_id

        spec = TemplateSpec(**cfg)
        _CACHE[spec.id] = spec
        _TASK_INDEX.setdefault(spec.task, []).append(spec.id)


if not _CACHE:
    _load_all()


def get(template_id: str) -> TemplateSpec:
    if not _CACHE:
        raise RuntimeError(
            f"Template cache is empty. Check that the path '{_TEMPLATE_ROOT}' is correct "
            "and contains template YAML files."
        )
    return _CACHE[template_id]


def list_by_task(task: str) -> List[str]:
    return _TASK_INDEX.get(task, [])
```

## File: sallm/training/factory.py
```python
import logging
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
)
from datasets import Dataset

from sallm.config import ExperimentConfig
from sallm.training.trainer import CustomTrainer

logger = logging.getLogger(__name__)


def build_trainer(
    config: ExperimentConfig,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
) -> CustomTrainer:
    training_args = TrainingArguments(**config.training)

    if training_args.local_rank <= 0:  # Print only on the main process
        logger.info("--- Effective Training Arguments ---")
        # The __str__ method of TrainingArguments provides a nice, readable format.
        logger.info(training_args)
        logger.info("------------------------------------")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # TODO: depricated, so use a different method eventually
    # trainer.tokenizer = tokenizer

    return trainer
```

## File: sallm/training/run.py
```python
import logging
import os

from sallm.config import ExperimentConfig
from sallm.data.factory import build_datasets
from sallm.models.factory import build_model, build_tokenizer
from sallm.training.factory import build_trainer

logger = logging.getLogger(__name__)


def run(config: ExperimentConfig) -> None:
    is_hpo_run = "WANDB_SWEEP_ID" in os.environ

    if not is_hpo_run:
        if config.wandb.project:
            os.environ["WANDB_PROJECT"] = config.wandb.project
        if config.wandb.name:
            os.environ["WANDB_RUN_NAME"] = config.wandb.name
    if config.wandb.id:
        os.environ["WANDB_RUN_ID"] = config.wandb.id
        os.environ["WANDB_RESUME"] = "allow"
    if config.training and config.wandb and config.wandb.name:
        config.training["run_name"] = config.wandb.name

    tokenizer = build_tokenizer(config)
    model = build_model(config, tokenizer)
    train_ds, val_ds, test_ds = build_datasets(config, is_hpo=is_hpo_run)

    trainer = build_trainer(config, model, tokenizer, train_ds, val_ds)
    trainer.train(resume_from_checkpoint=config.training.get("resume_from_checkpoint"))

    if not is_hpo_run:
        out = os.path.join(trainer.args.output_dir, "final_model")
        trainer.save_model(out)
        logger.info(f"Saved model → {out}")

    # TODO: do per language
    if test_ds:
        res = trainer.predict(test_ds)
        logger.info(f"Test metrics: {res.metrics}")
```

## File: sallm/training/trainer.py
```python
import logging
import math
from pathlib import Path
import time
from typing import Dict, Optional

import torch
import wandb
from torch.utils.data import DataLoader
from transformers import Trainer
from transformers.utils import is_datasets_available

if is_datasets_available():
    import datasets

logger = logging.getLogger(__name__)


# TODO rename to something more appropriate since different models will use different trainers
class CustomTrainer(Trainer):
    def evaluate(
        self,
        eval_dataset: Optional[datasets.Dataset] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        if "lang" not in eval_dataset.features:
            logger.warning("Custom `evaluate` is exiting: 'lang' column not found.")
            return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        start_time = time.time()

        unique_languages = sorted(list(set(eval_dataset["lang"])))
        per_language_metrics_for_wandb = {}
        # TODO remove samples?
        grand_total_loss = 0.0
        grand_total_samples = 0

        model = self._wrap_model(self.model, training=False, dataloader=None)
        model.eval()

        if self._signature_columns is None:
            raise RuntimeError("Could not find model signature columns.")
        model_args = self._signature_columns

        for lang in unique_languages:
            lang_dataset = eval_dataset.filter(
                lambda x: x["lang"] == lang, load_from_cache_file=False
            )
            if len(lang_dataset) == 0:
                continue

            dataloader: DataLoader = self.get_eval_dataloader(lang_dataset)
            total_loss, num_samples = 0.0, 0

            with torch.no_grad():
                for batch in dataloader:
                    batch = self._prepare_inputs(batch)
                    model_inputs = {k: v for k, v in batch.items() if k in model_args}
                    outputs = model(**model_inputs)
                    loss = outputs.loss
                    batch_size = batch["input_ids"].size(0)
                    total_loss += loss.item() * batch_size
                    num_samples += batch_size

            all_losses = self.accelerator.gather(
                torch.tensor(total_loss, device=self.args.device)
            )
            all_samples = self.accelerator.gather(
                torch.tensor(num_samples, device=self.args.device)
            )

            total_loss_agg = torch.sum(all_losses).item()
            total_samples_agg = torch.sum(all_samples).item()

            grand_total_loss += total_loss_agg
            grand_total_samples += total_samples_agg

            if total_samples_agg > 0:
                avg_loss = total_loss_agg / total_samples_agg
                try:
                    perplexity = math.exp(avg_loss)
                except OverflowError:
                    perplexity = float("inf")

                per_language_metrics_for_wandb[f"eval/{lang}_loss"] = avg_loss
                per_language_metrics_for_wandb[f"eval/{lang}_perplexity"] = perplexity

        if self.is_world_process_zero() and per_language_metrics_for_wandb:
            wandb.log(per_language_metrics_for_wandb, step=self.state.global_step)

        metrics_to_return = {}
        if grand_total_samples > 0:
            overall_loss = grand_total_loss / grand_total_samples
            metrics_to_return[f"{metric_key_prefix}_loss"] = overall_loss

        runtime = time.time() - start_time
        metrics_to_return[f"{metric_key_prefix}_runtime"] = runtime
        metrics_to_return[f"{metric_key_prefix}_samples_per_second"] = (
            grand_total_samples / runtime
        )

        self.log(metrics_to_return)

        return metrics_to_return

    def save_model(self, output_dir=None, _internal_call=False):
        out = output_dir or self.args.output_dir
        Path(out).mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(out, safe_serialization=False)

        if hasattr(self, "tokenizer"):
            self.tokenizer.save_pretrained(out)
```

## File: sallm/__init__.py
```python
# TODO change factory.py to not be duplicated in directories
# TODO develop way for hpo to continue from crashed trial if hpc cuts short
```

## File: sallm/config.py
```python
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from omegaconf import MISSING

from sallm.utils import RunMode


@dataclass
class ScriptArguments:
    config_path: str = field(
        metadata={"help": "Path to the main YAML experiment config file."}
    )
    wandb_run_id: Optional[str] = field(
        default=None,
        metadata={"help": "Wandb run ID to resume a specific crashed trial."},
    )


@dataclass
class ParamRangeConfig:
    min_params_m: float = MISSING
    max_params_m: float = MISSING


@dataclass
class WandbConfig:
    project: str = MISSING
    entity: Optional[str] = None
    group: Optional[str] = None
    name: Optional[str] = MISSING
    id: Optional[str] = None


@dataclass
class EvaluationConfig:
    task_packs: List[str] = field(default_factory=list)
    output_dir: str = MISSING
    overrides: Dict[str, Any] = field(default_factory=dict)
    wandb: Optional[WandbConfig] = MISSING


@dataclass
class ModelEvalConfig:
    checkpoint: str = MISSING
    adapter: str = "hf"
    dtype: str = "bfloat16"
    device: str = "cuda:0"


@dataclass
class ModelConfig:
    architecture: str = MISSING
    config: Optional[Dict[str, Any]] = None
    init_checkpoint: Optional[str] = None
    param_validation: Optional[ParamRangeConfig] = None

    def __post_init__(self):
        if self.config is None and self.init_checkpoint is None:
            raise ValueError(
                "Either `config` or `init_checkpoint` must be provided inside `model`."
            )


@dataclass
class DataConfig:
    path: str = MISSING
    train_split: str = "train"
    eval_split: str = "validation"
    test_split: Optional[str] = "test"


@dataclass
class TokenizerConfig:
    path: str = MISSING


class TemplateChoice(str, Enum):
    CYCLE = "cycle"
    RANDOM = "random"
    ALL = "all"


@dataclass
class TemplateRef:
    id: str = MISSING
    weight: float = 1.0


@dataclass
class FinetuneDatasetConfig:
    hf_name: str = MISSING
    subset: Optional[str] = None
    text_columns: List[str] = field(default_factory=list)
    label_column: str = MISSING
    splits: Dict[str, str] = field(default_factory=dict)
    templates: List[TemplateRef] = field(default_factory=list)
    template_choice: TemplateChoice = TemplateChoice.CYCLE
    max_seq_length: int = MISSING


@dataclass
class PipelineConfig:
    base_checkpoint: str = MISSING
    languages: List[str] = field(default_factory=list)
    task_name: str = MISSING
    finetune_base_cfg: str = MISSING
    eval_stub_cfg: str = MISSING
    slurm_array: bool = False


@dataclass
class TemplateConfig:
    prompt: str = MISSING
    label_mapping: Dict[Union[int, str], str] = field(default_factory=dict)


@dataclass
class PeftConfig:
    method: str = "qlora"
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    mode: RunMode
    wandb: WandbConfig
    model: Optional[ModelConfig] = None
    data: Optional[DataConfig] = None
    tokenizer: Optional[TokenizerConfig] = None
    training: Optional[Dict[str, Any]] = None
    evaluation: Optional[Dict[str, Any]] = None
    eval_model: Optional[ModelEvalConfig] = None
    dataset: Optional[FinetuneDatasetConfig] = None
    peft: Optional[PeftConfig] = None
    template: Optional[TemplateConfig] = None
    pipeline: Optional[PipelineConfig] = None
```

## File: sallm/main.py
```python
from __future__ import annotations
import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from sallm.config import ExperimentConfig
from sallm.utils import RunMode

from sallm.training.run import run as run_train
from sallm.fine_tune.run import run as run_fine_tune
from sallm.evaluation.run import run as run_eval
from sallm.pipeline.run import run as run_orch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    unwrapped_cfg = cfg

    # Dynamically detect if the config is nested inside a group key.
    keys_in_cfg = list(cfg.keys())
    if (
        len(keys_in_cfg) == 1
        and keys_in_cfg[0] not in ExperimentConfig.__dataclass_fields__
    ):
        group_name = keys_in_cfg[0]
        logger.info(
            f"Detected nested config group '{group_name}'. Unwrapping configuration."
        )
        unwrapped_cfg = cfg[group_name]

    # Now, perform the merge with the (potentially unwrapped) config.
    schema = OmegaConf.structured(ExperimentConfig)
    config = OmegaConf.merge(schema, unwrapped_cfg)

    logger.info(f"Run mode: {config.mode.value}")

    if config.mode == RunMode.TRAIN:
        run_train(config)
    elif config.mode == RunMode.FINETUNE:
        run_fine_tune(config)
    elif config.mode == RunMode.EVALUATE:
        run_eval(config)
    elif config.mode == RunMode.ORCHESTRATE:
        run_orch(config)
    else:
        raise ValueError(f"Unsupported mode {config.mode!r}")


if __name__ == "__main__":
    main()
```

## File: sallm/utils.py
```python
import logging
from enum import Enum

from torch import nn

logger = logging.getLogger(__name__)


class RunMode(str, Enum):
    TRAIN = "train"
    FINETUNE = "finetune"
    EVALUATE = "evaluate"
    ORCHESTRATE = "orchestrate"


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```
