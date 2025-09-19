from __future__ import annotations

import logging
import os
import textwrap
from typing import Any, cast

import peft
import torch
import wandb
from omegaconf import OmegaConf
from sallm.config import ExperimentConfig
from sallm.data.factory import build_datasets
from sallm.models.factory import build_model, build_tokenizer
from sallm.training.factory import build_trainer

logger = logging.getLogger(__name__)


def _is_hpo_run(config: ExperimentConfig) -> bool:
    wandb_id = OmegaConf.select(config, "wandb.id")
    return isinstance(wandb_id, str) and "sweep" in wandb_id


# TODO: improve naming
# TODO: no defaults for loraconfig, specify in config files
def _apply_peft_if_needed(model: Any, peft_cfg: Any) -> Any:
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
    is_hpo_run = _is_hpo_run(config)
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
    tokenizer_any = cast(Any, tokenizer)

    logger.info("Model …")
    model = build_model(config, tokenizer)
    model_any = cast(Any, model)

    logger.info("Adding special tokens and resizing model embeddings.")
    special_tokens_dict = {
        "additional_special_tokens": [
            "<|system|>",
            "<|user|>",
            "<|assistant|>",
        ]
    }
    num_added_tokens = tokenizer_any.add_special_tokens(special_tokens_dict)

    if num_added_tokens > 0:
        vocab_size = len(cast(Any, tokenizer).get_vocab())
        model_any.resize_token_embeddings(vocab_size)

    if getattr(tokenizer_any, "chat_template", None) is None:
        # TODO: move this template to its own file
        template_text = textwrap.dedent(
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

        tokenizer_any.chat_template = template_text
        logger.info("Tokenizer chat template not found. Applying default template.")

    if getattr(tokenizer_any, "pad_token", None) is None:
        eos_token = getattr(tokenizer_any, "eos_token", None)
        tokenizer_any.pad_token = eos_token
        logger.info(
            "tokenizer.pad_token was not set, setting it to eos_token: %s",
            eos_token,
        )

    model = _apply_peft_if_needed(model, config.peft)
    model_any = cast(Any, model)

    if i_am_main and hasattr(model_any, "print_trainable_parameters"):
        model_any.print_trainable_parameters()

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

    model_any.tokenizer = tokenizer

    trainer = build_trainer(config, model, tokenizer, train_ds, val_ds)
    training_params: dict[str, Any] = config.training or {}
    resume_ckpt = training_params.get("resume_from_checkpoint")

    logger.info("Fine-tuning start …")
    torch.autograd.set_detect_anomaly(mode=True, check_nan=True)
    trainer.train(resume_from_checkpoint=resume_ckpt)
    logger.info("Fine-tuning done.")

    if is_hpo_run:
        return

    # Prefer saving adapters when training with PEFT; otherwise merge weights.
    if (
        config.peft
        and config.peft.method
        and config.peft.method.lower() in {"lora", "qlora"}
        and isinstance(model, peft.PeftModel)
    ):
        base_checkpoint = getattr(config.model, "init_checkpoint", None)
        if not base_checkpoint:
            raise ValueError(
                "Saving PEFT adapters requires `model.init_checkpoint` to be set."
            )
        output_dir = os.path.join(trainer.args.output_dir, "final_adapter")
        os.makedirs(output_dir, exist_ok=True)
        active_adapter = getattr(model, "active_adapter", "default")
        peft_cfg = model.peft_config.get(active_adapter)
        if peft_cfg:
            peft_cfg.base_model_name_or_path = base_checkpoint
        if hasattr(model, "base_model_name_or_path"):
            model.base_model_name_or_path = base_checkpoint
        model.save_pretrained(output_dir)
        tokenizer_any.save_pretrained(output_dir)
        logger.info(f"Saved PEFT adapter to → {output_dir}")
        return

    if hasattr(model_any, "merge_and_unload"):
        merged_model = model_any.merge_and_unload()
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
    tokenizer_any.save_pretrained(output_dir)
    logger.info(f"Saved final MERGED model to → {output_dir}")
