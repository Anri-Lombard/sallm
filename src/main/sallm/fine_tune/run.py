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
import peft

logger = logging.getLogger(__name__)


# TODO: improve naming
# TODO: no defaults for loraconfig, specify in config files
def _apply_peft_if_needed(model, peft_cfg):
    if not peft_cfg or peft_cfg.method == "none":
        return model

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

    if tokenizer.chat_template is None:
        # TODO: move this template to it's own file
        tokenizer.chat_template = """{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|system|>\\n' + message['content'] + '<|end|>\\n' }}{% elif message['role'] == 'user' %}{{ '<|user|>\\n' + message['content'] + '<|end|>\\n' }}{% elif message['role'] == 'assistant' %}{{ '<|assistant|>\\n' + message['content'] + '<|end|>\\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\\n' }}{% else %}{{ eos_token }}{% endif %}"""
        logger.info(
            "Tokenizer chat template not found. Applying custom template provided."
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(
            f"tokenizer.pad_token was not set, setting it to eos_token: {tokenizer.eos_token}"
        )

    model = _apply_peft_if_needed(model, config.peft)

    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    logger.info("Datasets …")
    train_ds, val_ds, _ = build_datasets(config, tokenizer, is_hpo=False)

    assert "messages" in train_ds.column_names, (
        "Training dataset is missing the required 'messages' column. "
        "Please ensure your data processing pipeline is creating the correct conversational format."
    )
    if val_ds:
        assert "messages" in val_ds.column_names, (
            "Validation dataset is missing the required 'messages' column. "
            "Please ensure your data processing pipeline is creating the correct conversational format."
        )

    logger.info(f"Samples: train={len(train_ds)}, val={len(val_ds)}")

    if train_ds and len(train_ds) > 0:
        sample = train_ds[0]
        logger.info("--- Inspecting a single training sample ---")
        logger.info(f"Messages:\n{sample['messages']}")
        logger.info("-------------------------------------------")

    model.tokenizer = tokenizer

    trainer = build_trainer(config, model, tokenizer, train_ds, val_ds)
    resume_ckpt = config.training.get("resume_from_checkpoint")

    logger.info("Fine-tuning start …")
    torch.autograd.set_detect_anomaly(mode=True, check_nan=True)
    trainer.train(resume_from_checkpoint=resume_ckpt)
    logger.info("Fine-tuning done.")

    output_dir = os.path.join(trainer.args.output_dir, "final_model")
    trainer.save_model(output_dir)
    logger.info(f"Saved final model/adapter → {output_dir}")
