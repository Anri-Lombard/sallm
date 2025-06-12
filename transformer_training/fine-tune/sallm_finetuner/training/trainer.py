import torch
import wandb
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import numpy as np
import os
import json
from dataclasses import asdict

from ..config import AppConfig, TrainingConfig
from ..models.heads import GPTForTokenClassification
from ..data.processing import get_per_language_loaders
from .metrics import MetricsComputer


class FineTuner:
    def __init__(
        self,
        config: AppConfig,
        model,
        tokenizer,
        train_dl,
        val_dl,
        test_dl,
        metrics_computer,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl
        self.metrics_computer = metrics_computer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Using device: {self.device}")

    def train(self) -> str:
        training_config = self.config.training
        output_dir = training_config.output_dir
        hparams_path = os.path.join(output_dir, "best_hparams.json")

        body_params = [
            p
            for n, p in self.model.named_parameters()
            if not n.startswith("classifier") and p.requires_grad
        ]
        head_params = [
            p
            for n, p in self.model.named_parameters()
            if n.startswith("classifier") and p.requires_grad
        ]

        optimizer_grouped_parameters = [
            {"params": body_params, "lr": training_config.learning_rate},
            {
                "params": head_params,
                "lr": training_config.learning_rate
                * training_config.head_lr_multiplier,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters, weight_decay=training_config.weight_decay
        )

        num_training_steps = training_config.num_train_epochs * len(self.train_dl)
        num_warmup_steps = int(num_training_steps * training_config.warmup_ratio)
        lr_scheduler = get_scheduler(
            "cosine", optimizer, num_warmup_steps, num_training_steps
        )

        best_val_f1 = -1.0
        best_model_path = None
        global_step = 0
        os.makedirs(output_dir, exist_ok=True)

        for epoch in range(training_config.num_train_epochs):
            self.model.train()
            progress_bar = tqdm(self.train_dl, desc=f"Training Epoch {epoch+1}")
            for batch in progress_bar:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs["loss"]
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.set_postfix(loss=loss.item())
                wandb.log({"train/loss": loss.item()}, step=global_step)
                global_step += 1

            print(f"\n--- Evaluating Epoch {epoch+1} ---")
            overall_metrics = self.evaluate(self.val_dl, "overall_val")
            wandb.log(
                {f"val/overall/{k}": v for k, v in overall_metrics.items()},
                step=global_step,
            )

            current_f1 = overall_metrics.get("f1-score", -1.0)
            if current_f1 > best_val_f1:
                best_val_f1 = current_f1
                best_model_path = os.path.join(output_dir, "best_model.pt")
                print(
                    f"New best model with F1-score: {best_val_f1:.4f}. Saving to {best_model_path}"
                )
                torch.save(
                    {"epoch": epoch, "model_state_dict": self.model.state_dict()},
                    best_model_path,
                )
                if self.config.hpo:
                    print(f"Saving new best hyperparameters to {hparams_path}")
                    self.config.save_hparams(hparams_path)

        return best_model_path

    def evaluate(self, dataloader, eval_name):
        self.model.eval()
        all_preds, all_labels = [], []
        for batch in tqdm(dataloader, desc=f"Evaluating on {eval_name}"):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)
            logits = outputs["logits"]
            predictions = torch.argmax(logits, dim=-1)
            all_preds.extend(predictions.cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())

        metrics = self.metrics_computer.compute(all_preds, all_labels)
        print(f"Results for {eval_name}: {metrics}")
        return metrics

    def final_evaluation(self, best_model_path: str):
        if not best_model_path or not os.path.exists(best_model_path):
            print(
                "\nSkipping final test evaluation as no best model was saved or found."
            )
            return

        print(f"\n--- Final Evaluation on Test Set ---")
        print(f"Loading best model from {best_model_path} for final evaluation.")
        checkpoint = torch.load(best_model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        print("\n--- Overall Test Set Evaluation ---")
        overall_test_metrics = self.evaluate(self.test_dl, "overall_test")
        wandb.log({f"test/overall/{k}": v for k, v in overall_test_metrics.items()})

        print("\n--- Per-Language Test Set Evaluation ---")
        tokenization_fn = self.metrics_computer.tokenization_function
        per_lang_test_loaders = get_per_language_loaders(
            split="test",
            dataset_name=self.config.data.dataset_name,
            language_subsets=self.config.data.test_language_subsets,
            tokenizer=self.tokenizer,
            tokenization_function=tokenization_fn,
            batch_size=self.config.training.batch_size,
        )
        for lang, lang_test_loader in per_lang_test_loaders.items():
            lang_metrics = self.evaluate(lang_test_loader, f"test_{lang}")
            wandb.log({f"test/{lang}/{k}": v for k, v in lang_metrics.items()})
