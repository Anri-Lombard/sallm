from __future__ import annotations
import logging
import torch
import wandb
import os
from omegaconf import OmegaConf

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
        peft_kwargs = OmegaConf.to_container(peft_cfg.kwargs, resolve=True)
        lora_conf = peft.LoraConfig(
            r=peft_kwargs.get("r", 64),
            lora_alpha=peft_kwargs.get("lora_alpha", 16),
            lora_dropout=peft_kwargs.get("lora_dropout", 0.05),
            target_modules=peft_kwargs.get("target_modules", ["q_proj", "v_proj"]),
            bias="none",
            task_type=peft.TaskType.CAUSAL_LM,
        )

        return peft.get_peft_model(model, lora_conf)
    raise ValueError(f"Unsupported PEFT method '{peft_cfg.method}'")


def run(config: ExperimentConfig) -> None:
    if config.wandb and config.wandb.project:
        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            group=config.wandb.group,
            name=config.wandb.name,
            id=config.wandb.id,
            config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
        )
        if config.wandb.id:
            wandb.config.update({"resume": "allow"})

    logger.info("Tokenizer …")
    tokenizer = build_tokenizer(config)

    logger.info("Model …")
    model = build_model(config, tokenizer)
    model = _apply_peft_if_needed(model, config.peft)

    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    logger.info("Datasets …")
    train_ds, val_ds, _ = build_datasets(config, tokenizer, is_hpo=False)
    logger.info(f"Samples: train={len(train_ds)}, val={len(val_ds)}")

    if train_ds and len(train_ds) > 0:
        sample = train_ds[0]
        logger.info("--- Inspecting a single training sample ---")
        logger.info(f"Prompt:\n{sample['prompt']}")
        logger.info(f"Completion: '{sample['completion'].strip()}'")
        logger.info("-------------------------------------------")

    trainer = build_trainer(config, model, tokenizer, train_ds, val_ds)
    resume_ckpt = config.training.get("resume_from_checkpoint")

    logger.info("Fine-tuning start …")
    torch.autograd.set_detect_anomaly(mode=True, check_nan=True)
    trainer.train(resume_from_checkpoint=resume_ckpt)
    logger.info("Fine-tuning done.")

    output_dir = os.path.join(trainer.args.output_dir, "final_model")
    trainer.save_model(output_dir)
    logger.info(f"Saved final model/adapter → {output_dir}")
