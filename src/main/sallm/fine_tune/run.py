from __future__ import annotations

import logging
import os
import shutil
import textwrap
from pathlib import Path

import peft
import torch
import wandb
from omegaconf import OmegaConf
from sallm.config import ExperimentConfig
from sallm.data.factory import build_conversation_dataset, build_datasets
from sallm.models.factory import build_model, build_tokenizer
from sallm.training.factory import build_trainer

logger = logging.getLogger(__name__)


def _is_hpo_run(config: ExperimentConfig) -> bool:
    wandb_id = OmegaConf.select(config, "wandb.id")
    return isinstance(wandb_id, str) and "sweep" in wandb_id


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


def _sync_weight_tying_flag(model) -> None:
    config = getattr(model, "config", None)
    if config is None or not hasattr(config, "tie_word_embeddings"):
        return
    get_input = getattr(model, "get_input_embeddings", None)
    get_output = getattr(model, "get_output_embeddings", None)
    if not callable(get_input) or not callable(get_output):
        return
    input_emb = get_input()
    output_emb = get_output()
    if input_emb is None or output_emb is None:
        return
    input_weight = getattr(input_emb, "weight", None)
    output_weight = getattr(output_emb, "weight", None)
    if input_weight is None or output_weight is None:
        return
    shared = input_weight is output_weight
    if not shared:
        shared = input_weight.data_ptr() == output_weight.data_ptr()
    if shared and not bool(config.tie_word_embeddings):
        logger.info(
            "Detected tied embeddings; updating config.tie_word_embeddings to "
            "True before saving."
        )
        config.tie_word_embeddings = True
    if not shared and bool(config.tie_word_embeddings):
        logger.info(
            "Detected untied embeddings; updating config.tie_word_embeddings to "
            "False before saving."
        )
        config.tie_word_embeddings = False


def _save_tokenizer_with_fallback(
    tokenizer,
    output_dir: str,
    source_path: str | None,
) -> None:
    tokenizer.save_pretrained(output_dir)
    output_root = Path(output_dir)
    required = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    )
    missing = [name for name in required if not (output_root / name).exists()]
    if not missing:
        return
    if source_path is None:
        missing_display = ", ".join(missing)
        raise FileNotFoundError(f"Tokenizer save missing files: {missing_display}")
    source_root = Path(source_path)
    if not source_root.exists():
        missing_display = ", ".join(missing)
        raise FileNotFoundError(
            f"Tokenizer save missing files: {missing_display}. "
            f"Fallback path '{source_path}' does not exist."
        )
    for name in missing:
        candidate = source_root / name
        if candidate.exists():
            shutil.copy2(candidate, output_root / name)
    remaining = [name for name in required if not (output_root / name).exists()]
    if remaining:
        remaining_display = ", ".join(remaining)
        raise FileNotFoundError(
            f"Tokenizer save missing files after fallback copy: " f"{remaining_display}"
        )
    logger.info(
        "Copied tokenizer files %s from %s to %s",
        ", ".join(missing),
        source_root,
        output_root,
    )


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
    tokenizer_source_path = getattr(config.tokenizer, "path", None)
    if tokenizer_source_path is not None:
        tokenizer_source_path = os.path.expanduser(str(tokenizer_source_path))

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

    if "messages" not in train_ds.column_names:
        logger.warning(
            "Training dataset lacks 'messages'; applying "
            "conversational formatter fallback."
        )
        train_ds = build_conversation_dataset(train_ds, config)
    if val_ds and "messages" not in val_ds.column_names:
        logger.warning(
            "Validation dataset lacks 'messages'; applying "
            "conversational formatter fallback."
        )
        val_ds = build_conversation_dataset(val_ds, config)

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

    if is_hpo_run:
        return

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
        _save_tokenizer_with_fallback(
            tokenizer,
            output_dir,
            tokenizer_source_path,
        )
        logger.info(f"Saved PEFT adapter to → {output_dir}")
        return

    if hasattr(model, "merge_and_unload"):
        merged_model = model.merge_and_unload()
    else:
        merged_model = model

    output_dir = os.path.join(trainer.args.output_dir, "final_merged_model")
    if hasattr(merged_model, "save_pretrained"):
        _sync_weight_tying_flag(merged_model)
        safe_serialization = True
        if hasattr(trainer, "args") and hasattr(trainer.args, "save_safetensors"):
            safe_serialization = bool(trainer.args.save_safetensors)
        try:
            merged_model.save_pretrained(
                output_dir, safe_serialization=safe_serialization
            )
        except RuntimeError as exc:
            message = str(exc)
            if safe_serialization and "shared tensors" in message:
                logger.warning(
                    "Retrying save_pretrained with safe_serialization=False due "
                    "to shared tensors."
                )
                merged_model.save_pretrained(output_dir, safe_serialization=False)
            else:
                raise
    else:
        os.makedirs(output_dir, exist_ok=True)
        torch.save(
            merged_model.state_dict(), os.path.join(output_dir, "pytorch_model.bin")
        )
    _save_tokenizer_with_fallback(
        tokenizer,
        output_dir,
        tokenizer_source_path,
    )
    logger.info(f"Saved final MERGED model to → {output_dir}")
