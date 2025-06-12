import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Type, TypeVar
import os

T = TypeVar("T")


@dataclass
class DataConfig:
    dataset_name: str = "masakhane/masakhaner2"
    pad_token_id: int = -100
    max_seq_length: int = 512


@dataclass
class ModelConfig:
    tokenizer_path: str = (
        "/home/lmbanr001/masters/sallm/tokenizer/tokenizer/tokenizer.json"
    )
    pretrained_model_path: str = (
        "/scratch/lmbanr001/masters/sallm/trained_models/final/checkpoints_bash_20250423_155507/checkpoint_best_loss.pt"
    )
    vocab_size: int = 50257
    model_dim: int = 512
    num_heads: int = 8
    num_layers: int = 6


@dataclass
class TrainingConfig:
    output_dir: str = "./finetuning_results"
    wandb_project: str = "sallm-ner-finetuning"
    wandb_entity: str = "anri-m-lombard"
    batch_size: int = 16
    learning_rate: float = 5e-5
    head_lr_multiplier: float = 5.0
    num_train_epochs: int = 10
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06


@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    hpo: bool = False

    def save_hparams(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self.training), f, indent=4)

    def load_hparams(self, path: str):
        with open(path, "r") as f:
            hparams = json.load(f)
        self.training = TrainingConfig(**hparams)

    def update_from_wandb(self, wandb_config):
        training_dict = asdict(self.training)
        for key, value in wandb_config.items():
            if key in training_dict:
                setattr(self.training, key, value)


def _parse_args_into_config(config_instance):
    parser = argparse.ArgumentParser(description="Fine-tuning script.")
    parser.add_argument(
        "--hpo", action="store_true", help="Enable Hyperparameter Optimization mode."
    )

    args, unknown = parser.parse_known_args()
    config_instance.hpo = args.hpo

    for arg in unknown:
        if arg.startswith("--"):
            key_str, val_str = arg.lstrip("-").split("=", 1)
            keys = key_str.split(".")

            try:
                val = json.loads(val_str)
            except json.JSONDecodeError:
                val = val_str

            d = config_instance
            for key in keys[:-1]:
                d = getattr(d, key)
            setattr(d, keys[-1], val)


def get_config(config_class: Type[T]) -> T:
    config = config_class()
    _parse_args_into_config(config)

    hparams_path = os.path.join(config.training.output_dir, "best_hparams.json")

    if not config.hpo and os.path.exists(hparams_path):
        print(f"Loading best hyperparameters from {hparams_path}")
        config.load_hparams(hparams_path)
    elif not config.hpo and not os.path.exists(hparams_path):
        print(
            f"Warning: Hyperparameter file not found at {hparams_path}. Using defaults."
        )

    return config
