from __future__ import annotations

import logging
import os
import textwrap

import peft
import torch
import wandb
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

    if peft_cfg.method.lower() in {"lora", "qlora"}:
        peft_kwargs = OmegaConf.to_container(peft_cfg.kwargs, resolve=True)
        lora_conf = peft.LoraConfig(
            r=peft_kwargs.get("r", 64),
            lora_alpha=peft_kwargs.get("lora_alpha", 16),
            lora_dropout=peft_kwargs.get("lora_dropout", 0.05),
            target_modules=peft_kwargs.get("target_modules", ["q_proj", "v_proj"]),
            bias="none",
            task_type=peft.TaskType.CAUSAL_LM,
            modules_to_save=["embed_tokens", "lm_head"],
        )

        return peft.get_peft_model(model, lora_conf)
    raise ValueError(f"Unsupported PEFT method '{peft_cfg.method}'")


def run(config: ExperimentConfig) -> None:
    sel = OmegaConf.select(config, "runtime.is_main")
    i_am_main = bool(True if sel is None else sel)

    # Ensure each process sets its local CUDA device to avoid NCCL warnings
    try:
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        if local_rank != -1:
            visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            if visible:
                devs = [int(x) for x in visible.split(",") if x != ""]
                if local_rank < len(devs):
                    torch.cuda.set_device(devs[local_rank])
                else:
                    torch.cuda.set_device(local_rank)
            else:
                torch.cuda.set_device(local_rank)
    except Exception:
        pass

    if config.wandb and config.wandb.project and i_am_main:
        settings = wandb.Settings(init_timeout=120)
        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            group=config.wandb.group,
            name=config.wandb.name,
            id=config.wandb.id,
            config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
            settings=settings,
        )
        if config.wandb.id:
            wandb.config.update({"resume": "allow"})

    logger.info("Tokenizer …")
    tokenizer = build_tokenizer(config)

    logger.info("Model …")
    model = build_model(config, tokenizer)

    logger.info("Adding special tokens and resizing model embeddings.")
    special_tokens_dict = {
        "additional_special_tokens": [
            "<|system|>",
            "<|user|>",
            "<|assistant|>",
        ]
    }
    num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)

    if num_added_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))

    if tokenizer.chat_template is None:
        # TODO: move this template to its own file
        tokenizer.chat_template = textwrap.dedent(
            """
            {%- if system_message %}
            <|system|>
            {{ system_message }}{{ eos_token }}
            {%- endif %}
            {%- for message in messages %}
                {%- if message['role'] == 'user' %}
                    <|user|>
                    {{ message['content'] }}{{ eos_token }}
                {%- elif message['role'] == 'assistant' %}
                    {%- generation -%}
                    <|assistant|>
                    {{ message['content'] }}{{ eos_token }}
                    {%- endgeneration -%}
                {%- endif %}
            {%- endfor %}
            {%- if add_generation_prompt %}<|assistant|>{%- endif %}
            """
        ).lstrip()

        logger.info("Tokenizer chat template not found. Applying default template.")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(
            "tokenizer.pad_token was not set, setting it to eos_token: %s",
            tokenizer.eos_token,
        )

    model = _apply_peft_if_needed(model, config.peft)

    if i_am_main and hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    logger.info("Datasets …")
    train_ds, val_ds, _ = build_datasets(config, tokenizer, is_hpo=False)

    assert "messages" in train_ds.column_names, (
        "Training dataset is missing the required 'messages' column. "
        "Please ensure your data processing pipeline is creating the correct "
        "conversational format."
    )
    if val_ds:
        assert "messages" in val_ds.column_names, (
            "Validation dataset is missing the required 'messages' column. "
            "Please ensure your data processing pipeline is creating the correct "
            "conversational format."
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

    # TODO: rather just save adapters and merge at eval time to save space
    if hasattr(model, "merge_and_unload"):
        merged_model = model.merge_and_unload()
    else:
        merged_model = model

    output_dir = os.path.join(trainer.args.output_dir, "final_merged_model")
    if hasattr(merged_model, "save_pretrained"):
        merged_model.save_pretrained(output_dir)
    else:
        os.makedirs(output_dir, exist_ok=True)
        torch.save(
            merged_model.state_dict(), os.path.join(output_dir, "pytorch_model.bin")
        )
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Saved final MERGED model to → {output_dir}")
