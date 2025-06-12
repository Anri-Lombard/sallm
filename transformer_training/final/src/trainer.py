import os
import sys
import time
import uuid
import glob
import wandb
import torch
import torch.distributed as dist
from torch import nn
from pathlib import Path
from collections import defaultdict
from functools import lru_cache

from .config import TrainingConfig
from .data import distributed_data_generator, _load_data_shard
from .model import GPT, next_multiple_of_n
from .optimizer import Muon


class Trainer:
    def __init__(self, config: TrainingConfig, rank: int, world_size: int, device):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.master_process = rank == 0
        self.logfile = None

        self._setup_logging_and_wandb()
        self._setup_environment()

        self.train_loader = distributed_data_generator(
            config.train_files, config.batch_size, rank, world_size
        )
        self.model = self._setup_model()
        self.optimizers, self.schedulers = self._setup_optimizers()

        self.model = torch.compile(self.model)

    def _setup_logging_and_wandb(self):
        if self.master_process:
            run_id = uuid.uuid4()
            os.makedirs("logs", exist_ok=True)
            self.logfile = f"logs/{run_id}.txt"
            print(f"Log file at: {self.logfile}")
            wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=self.config.wandb_name,
                config={
                    k: v
                    for k, v in self.config.__dict__.items()
                    if k not in ["train_files", "val_files", "save_checkpoint"]
                },
            )

    def _setup_environment(self):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        torch._dynamo.config.suppress_errors = True
        torch._inductor.config.coordinate_descent_tuning = True
        backends = torch._inductor.config.max_autotune_gemm_backends
        torch._inductor.config.max_autotune_gemm_backends = "ATEN," + backends

    def _setup_model(self):
        model = GPT(
            vocab_size=self.config.vocab_size,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            model_dim=self.config.model_dim,
        ).to(self.device)
        for m in model.modules():
            if isinstance(m, nn.Embedding):
                m.bfloat16()
        for param in model.parameters():
            dist.broadcast(param.detach(), 0)
        return model

    def _setup_optimizers(self):
        hidden_matrix_params = [
            p for p in self.model.blocks.parameters() if p.ndim >= 2
        ]
        embed_params = [
            self.model.embed.weight,
            *self.model.value_embeds.parameters(),
        ]
        scalar_params = [p for p in self.model.parameters() if p.ndim < 2]
        head_params = [self.model.lm_head.weight]

        adam_params = [
            dict(params=head_params, lr=self.config.head_lr),
            dict(params=embed_params, lr=self.config.embed_lr),
            dict(params=scalar_params, lr=self.config.scalar_lr),
        ]
        optimizer1 = torch.optim.Adam(
            adam_params,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            fused=True,
            eps=self.config.adam_eps,
        )
        optimizer2 = Muon(
            hidden_matrix_params,
            lr=self.config.hidden_lr,
            momentum=self.config.momentum_end,
            rank=self.rank,
            world_size=self.world_size,
        )
        optimizers = [optimizer1, optimizer2]

        def get_lr(it: int):
            t = 1 - it / self.config.num_iterations
            assert 1 >= t >= 0
            w = min(t / self.config.cooldown_frac, 1.0)
            return w * 1.0 + (1 - w) * 0.1

        schedulers = [
            torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers
        ]
        return optimizers, schedulers

    def _log(self, message, console=False):
        if self.master_process:
            if console:
                print(message)
            with open(self.logfile, "a") as f:
                print(message, file=f)

    def train(self):
        best_val_loss = float("inf")
        steps_per_epoch = self.config.train_tokens // self.config.batch_size
        save_epochs_set = (
            set(map(int, self.config.save_epochs.split(",")))
            if self.config.save_epochs
            else set()
        )

        if self.master_process:
            os.makedirs(self.config.checkpoint_dir, exist_ok=True)
            self._log(f"Checkpoints will be saved in: {self.config.checkpoint_dir}")

        running_loss = 0.0
        running_accuracy = 0.0
        training_time_ms = 0
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        for step in range(self.config.num_iterations + 1):
            last_step = step == self.config.num_iterations
            if step == 10:
                training_time_ms = 0
                t0 = time.perf_counter()

            current_epoch = (step // steps_per_epoch) + 1 if steps_per_epoch > 0 else 1
            window_size = next_multiple_of_n(
                1728 * step / self.config.num_iterations, n=128
            )

            if last_step or (
                self.config.val_loss_every > 0
                and step > 0
                and step % self.config.val_loss_every == 0
            ):
                torch.cuda.synchronize()
                training_time_ms += 1000 * (time.perf_counter() - t0)
                val_metrics = self._evaluate(window_size)
                avg_loss = val_metrics["overall/loss"]

                if self.master_process:
                    is_best = (
                        self.config.save_best_checkpoint and avg_loss < best_val_loss
                    )
                    if is_best:
                        best_val_loss = avg_loss
                        self._log(
                            f"*** New best validation loss {avg_loss:.4f} at step {step} ***",
                            console=True,
                        )
                        self._save_checkpoint(step, current_epoch, "best_loss")

                    if current_epoch in save_epochs_set:
                        self._save_checkpoint(
                            step, current_epoch, f"epoch_{current_epoch}"
                        )

                    wandb.log(val_metrics, step=step)

                self.model.train()
                torch.cuda.synchronize()
                t0 = time.perf_counter()

            if last_step:
                if self.master_process and self.config.save_checkpoint:
                    self._save_checkpoint(step, current_epoch, f"final_step{step}")
                break

            inputs, targets = next(self.train_loader)
            for input_seq, target_seq in zip(
                inputs.split(self.config.seq_len), targets.split(self.config.seq_len)
            ):
                loss, accuracy = self.model(
                    input_seq, target_seq, self._sw_num_blks(window_size)
                )
                loss.backward()

                if self.master_process:
                    loss_item, accuracy_item = loss.item(), accuracy.item()
                    running_loss = (
                        running_loss * step / (step + 1) + loss_item / (step + 1)
                        if step > 0
                        else loss_item
                    )
                    running_accuracy = (
                        running_accuracy * step / (step + 1)
                        + accuracy_item / (step + 1)
                        if step > 0
                        else accuracy_item
                    )
                    wandb.log(
                        {
                            "train/loss": loss_item,
                            "train/running_loss": running_loss,
                            "train/accuracy": accuracy_item,
                            "train/running_accuracy": running_accuracy,
                        },
                        step=step,
                    )

            for param in self.model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

            frac = min(step / 300, 1)
            for group in self.optimizers[1].param_groups:
                group["momentum"] = (
                    1 - frac
                ) * self.config.momentum_start + frac * self.config.momentum_end

            for opt, sched in zip(self.optimizers, self.schedulers):
                opt.step()
                sched.step()

            self.model.zero_grad(set_to_none=True)

        if self.master_process:
            wandb.finish()
        self._log(
            f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB"
        )

    def _evaluate(self, window_size: int):
        self.model.eval()
        lang_stats = self._compute_validation_stats(window_size)
        metrics = {}

        if self.master_process:
            total_loss = sum(stats.total_loss for stats in lang_stats.values())
            total_tokens = sum(stats.total_tokens for stats in lang_stats.values())
            avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
            perplexity = torch.exp(torch.tensor(avg_loss)).item()

            metrics["overall/loss"] = avg_loss
            metrics["overall/perplexity"] = perplexity

            for lang, stats in lang_stats.items():
                metrics[f"val/{lang}/loss"] = stats.get_average_loss()
                metrics[f"val/{lang}/perplexity"] = stats.get_perplexity()

            self._log(f"Validation loss: {avg_loss:.4f}", console=True)
        return metrics

    def _compute_validation_stats(self, window_size: int):
        val_files = sorted(glob.glob(self.config.val_files))
        lang_files = defaultdict(list)
        for file in val_files:
            lang = self._get_language_from_filename(file)
            lang_files[lang].append(file)

        lang_stats = {lang: LanguageValidationStats() for lang in lang_files}
        for lang, files in lang_files.items():
            for file in files:
                val_bs = self.world_size * self.config.seq_len
                tokens = _load_data_shard(Path(file))
                num_batches = len(tokens) // val_bs
                with torch.no_grad():
                    for i in range(num_batches):
                        start_idx = i * val_bs + self.rank * self.config.seq_len
                        end_idx = start_idx + self.config.seq_len + 1
                        if end_idx <= len(tokens):
                            batch = tokens[start_idx:end_idx]
                            x = batch[:-1].to(self.device, dtype=torch.int32)
                            y = batch[1:].to(self.device, dtype=torch.int64)
                            loss, _ = self.model(x, y, self._sw_num_blks(window_size))
                            dist.all_reduce(loss, op=dist.ReduceOp.AVG)
                            lang_stats[lang].update(loss.item(), self.config.seq_len)
        return lang_stats

    def _save_checkpoint(self, step: int, epoch: int, name: str):
        path = os.path.join(self.config.checkpoint_dir, f"checkpoint_{name}.pt")
        self._log(f"Saving checkpoint to {path}", console=True)
        log_data = {
            "step": step,
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizers": [opt.state_dict() for opt in self.optimizers],
        }
        torch.save(log_data, path)

    @lru_cache(1)
    def _sw_num_blks(self, window_size: int):
        return torch.tensor(window_size // 128, dtype=torch.int32).to(self.device)

    def _get_language_from_filename(self, filename: str) -> str:
        parts = Path(filename).stem.split("_")
        return parts[2]


class LanguageValidationStats:
    def __init__(self):
        self.total_loss = 0.0
        self.total_tokens = 0

    def update(self, loss: float, num_tokens: int):
        self.total_loss += loss * num_tokens
        self.total_tokens += num_tokens

    def get_average_loss(self):
        return self.total_loss / self.total_tokens if self.total_tokens > 0 else 0.0

    def get_perplexity(self):
        return torch.exp(torch.tensor(self.get_average_loss())).item()
