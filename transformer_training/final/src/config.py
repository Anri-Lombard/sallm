import argparse
import glob
import torch
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    train_files: str
    val_files: str
    batch_size: int
    seq_len: int
    epochs: int
    cooldown_frac: float
    val_loss_every: int
    head_lr: float
    embed_lr: float
    scalar_lr: float
    hidden_lr: float
    momentum_start: float
    momentum_end: float
    adam_beta1: float
    adam_beta2: float
    adam_eps: float
    vocab_size: int
    num_layers: int
    num_heads: int
    model_dim: int
    save_checkpoint: bool
    wandb_project: str
    wandb_entity: str
    wandb_name: str
    checkpoint_dir: str
    save_epochs: str
    save_best_checkpoint: bool
    train_tokens: int = 0
    num_iterations: int = 0


def get_config() -> TrainingConfig:
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_files", type=str, required=True)
    parser.add_argument("--val_files", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64 * 1024)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--cooldown_frac", type=float, default=0.4)
    parser.add_argument("--val_loss_every", type=int, default=2000)
    parser.add_argument("--head_lr", type=float, default=0.008)
    parser.add_argument("--embed_lr", type=float, default=0.6)
    parser.add_argument("--scalar_lr", type=float, default=0.04)
    parser.add_argument("--hidden_lr", type=float, default=0.025)
    parser.add_argument("--momentum_start", type=float, default=0.85)
    parser.add_argument("--momentum_end", type=float, default=0.95)
    parser.add_argument("--adam_beta1", type=float, default=0.8)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--adam_eps", type=float, default=1e-10)
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--model_dim", type=int, default=512)
    parser.add_argument("--save_checkpoint", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="sallm-pretraining")
    parser.add_argument("--wandb_entity", type=str, default="anri-m-lombard")
    parser.add_argument("--wandb_name", type=str, default="pretrain-run-1")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--save_epochs", type=str, default="")
    parser.add_argument("--save_best_checkpoint", action="store_true")

    args = parser.parse_args()
    config = TrainingConfig(**vars(args))

    train_files_list = glob.glob(config.train_files)
    total_tokens = 0
    for file in train_files_list:
        header = torch.from_file(str(file), False, 256, dtype=torch.int32)
        if header[0] != 20240520:
            raise ValueError(f"Invalid magic number in {file}")
        if header[1] != 1:
            raise ValueError(f"Unsupported version in {file}")
        total_tokens += int(header[2])

    config.train_tokens = (total_tokens // 8192) * 8192
    config.num_iterations = (config.train_tokens * config.epochs) // config.batch_size

    return config
