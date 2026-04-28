from __future__ import annotations

import logging
import os
import re
import shutil
import textwrap
from pathlib import Path
from typing import Any, cast

import peft
import torch
import wandb
from datasets import Dataset
from omegaconf import OmegaConf
from sallm.config import ExperimentConfig, to_resolved_dict
from sallm.data.factory import (
    build_conversation_dataset,
    build_datasets,
    resolve_eval_template_choice,
)
from sallm.models.factory import build_model, build_tokenizer
from sallm.training.factory import build_trainer
from tokenizers import AddedToken

logger = logging.getLogger(__name__)


def _is_hpo_run(config: ExperimentConfig) -> bool:
    wb = getattr(config, "wandb", None)
    wb_id = getattr(wb, "id", None) if wb is not None else None
    return isinstance(wb_id, str) and "sweep" in wb_id


# TODO: improve naming
# TODO: no defaults for loraconfig, specify in config files
def _apply_peft_if_needed(model, peft_cfg):
    if not peft_cfg or peft_cfg.method == "none":
        return model

    if peft_cfg.method.lower() in {"lora", "qlora"}:
        kwargs_obj = getattr(peft_cfg, "kwargs", {})
        if kwargs_obj is None:
            peft_kwargs: dict[str, Any] = {}
        else:
            peft_kwargs = to_resolved_dict(kwargs_obj, name="peft kwargs")
        lora_conf = peft.LoraConfig(
            r=peft_kwargs.get("r", 64),
            lora_alpha=peft_kwargs.get("lora_alpha", 16),
            lora_dropout=peft_kwargs.get("lora_dropout", 0.05),
            target_modules=peft_kwargs.get("target_modules", ["q_proj", "v_proj"]),
            bias="none",
            task_type=peft.TaskType.CAUSAL_LM,
            modules_to_save=peft_kwargs.get("modules_to_save"),
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
            f"Tokenizer save missing files after fallback copy: {remaining_display}"
        )
    logger.info(
        "Copied tokenizer files %s from %s to %s",
        ", ".join(missing),
        source_root,
        output_root,
    )


def _is_main_process() -> bool:
    try:
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    except Exception:
        return True
    return local_rank in (-1, 0)


def _sanitize_hf_repo_component(value: str) -> str:
    """Normalize dynamic strings so they are safe for HF Hub repo names."""
    component = str(value).strip()
    component = component.replace("mix:", "")
    component = component.replace("github:", "github-")
    component = component.replace("/", "-")
    component = re.sub(r"[^A-Za-z0-9._-]+", "-", component)
    component = re.sub(r"-{2,}", "-", component)
    component = component.strip("-.")
    if not component:
        return "unknown"
    return component.lower()


def _build_hub_repo_id(config: ExperimentConfig, *, merged: bool = False) -> str:
    org = str(config.hub.organization).strip() if config.hub else "anrilombard"
    arch = _sanitize_hf_repo_component(
        config.model.architecture if config.model else "unknown"
    )
    hf_name = config.dataset.hf_name if config.dataset else "unknown"
    task = _sanitize_hf_repo_component(hf_name)

    if config.dataset and config.dataset.languages:
        raw_langs = "-".join(str(lang) for lang in config.dataset.languages)
    elif config.dataset and getattr(config.dataset, "subset", None) not in (
        None,
        "null",
    ):
        raw_langs = str(config.dataset.subset)
    else:
        raw_langs = "all"
    langs = _sanitize_hf_repo_component(raw_langs)

    suffix = "-merged" if merged else ""
    repo_name = f"sallm-{arch}-{task}-{langs}{suffix}"
    repo_name = re.sub(r"-{2,}", "-", repo_name).strip("-.")
    if len(repo_name) > 96:
        repo_name = repo_name[:96].rstrip("-.")

    return f"{org}/{repo_name}"


def run(config: ExperimentConfig) -> None:
    if config.dataset is None:
        raise ValueError("Fine-tuning requires a `dataset` config block.")

    is_hpo_run = _is_hpo_run(config)
    i_am_main = _is_main_process()

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

    if config.wandb and config.wandb.project and i_am_main and (not is_hpo_run):
        settings = wandb.Settings(init_timeout=120)
        cfg_for_wandb = to_resolved_dict(
            OmegaConf.structured(config), name="wandb config"
        )
        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            group=config.wandb.group,
            name=config.wandb.name,
            id=config.wandb.id,
            config=cfg_for_wandb,
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
    special_tokens_dict: dict[str, list[str | AddedToken]] = {
        "additional_special_tokens": [
            "<|system|>",
            "<|user|>",
            "<|assistant|>",
        ]
    }
    num_added_tokens = tokenizer.add_special_tokens(cast(Any, special_tokens_dict))

    if num_added_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))

        # TEMP FIX: Mamba2 resize bug - lm_head not resized
        # See: https://github.com/huggingface/transformers/issues/43206
        # TODO: Remove once transformers fix is released
        if hasattr(model, "lm_head") and hasattr(model, "backbone"):
            model_any = cast(Any, model)
            expected_vocab = len(tokenizer)
            actual_vocab = int(model_any.lm_head.weight.shape[0])
            if actual_vocab != expected_vocab:
                import torch.nn as nn

                logger.warning(
                    "Mamba2 resize bug: lm_head has %d, expected %d. Fixing...",
                    actual_vocab,
                    expected_vocab,
                )
                model_any.lm_head = nn.Linear(
                    model_any.config.hidden_size, expected_vocab, bias=False
                )
                if hasattr(model_any.backbone, "embeddings"):
                    model_any.lm_head.weight = model_any.backbone.embeddings.weight

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

    def _has_messages(ds) -> bool:
        if hasattr(ds, "column_names"):
            return "messages" in ds.column_names
        try:
            ex = ds[0]
        except Exception:
            try:
                ex = next(iter(ds))
            except Exception:
                return False
        return isinstance(ex, dict) and ("messages" in ex)

    if not _has_messages(train_ds):
        if isinstance(train_ds, Dataset):
            logger.warning("Training dataset lacks 'messages'; applying formatter.")
            train_ds = build_conversation_dataset(train_ds, config)
        else:
            raise ValueError(
                "Training dataset lacks 'messages' and cannot be auto-formatted."
            )

    if val_ds and not _has_messages(val_ds):
        if isinstance(val_ds, Dataset):
            logger.warning("Validation dataset lacks 'messages'; applying formatter.")
            val_ds = build_conversation_dataset(
                val_ds,
                config,
                template_choice_override=resolve_eval_template_choice(config.dataset),
            )
        else:
            raise ValueError(
                "Validation dataset lacks 'messages' and cannot be auto-formatted."
            )

    def _safe_len(ds):
        try:
            return len(ds)
        except Exception:
            return None

    n_train = _safe_len(train_ds)
    n_val = _safe_len(val_ds)
    logger.info(
        "Samples: train=%s, val=%s",
        n_train if n_train is not None else "?",
        n_val if n_val is not None else "?",
    )

    sample = None
    try:
        sample = train_ds[0]
    except Exception:
        try:
            sample = next(iter(train_ds))
        except Exception:
            sample = None
    if sample:
        logger.info("--- Inspecting a single training sample ---")
        logger.info(f"Messages:\n{sample['messages']}")
        logger.info("-------------------------------------------")

    cast(Any, model).tokenizer = tokenizer

    trainer = build_trainer(config, model, tokenizer, train_ds, val_ds)
    resume_ckpt = (config.training or {}).get("resume_from_checkpoint")

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
        base_checkpoint = config.model.init_checkpoint if config.model else None
        if not base_checkpoint:
            raise ValueError(
                "Saving PEFT adapters requires `model.init_checkpoint` to be set."
            )
        trainer_output_dir = str(trainer.args.output_dir)
        output_dir = os.path.join(trainer_output_dir, "final_adapter")
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

        if config.hub and config.hub.enabled and config.hub.push_adapter and i_am_main:
            repo_id = _build_hub_repo_id(config, merged=False)

            try:
                # Update base model reference to HF model ID before pushing
                if config.hub.base_model_id:
                    active_adapter = getattr(model, "active_adapter", "default")
                    peft_cfg = model.peft_config.get(active_adapter)
                    if peft_cfg:
                        peft_cfg.base_model_name_or_path = config.hub.base_model_id
                        logger.info(
                            f"Set base_model reference to: {config.hub.base_model_id}"
                        )

                logger.info(f"Pushing adapter to HuggingFace Hub: {repo_id}")
                cast(Any, model).push_to_hub(repo_id, private=config.hub.private)
                tokenizer.push_to_hub(repo_id, private=config.hub.private)
                logger.info(f"Successfully pushed to {repo_id}")

                if config.hub.collection_slug:
                    try:
                        from huggingface_hub import add_collection_item

                        add_collection_item(
                            collection_slug=config.hub.collection_slug,
                            item_id=repo_id,
                            item_type="model",
                            exists_ok=True,
                        )
                        logger.info(
                            f"Added to collection: {config.hub.collection_slug}"
                        )
                    except Exception as e:
                        logger.warning(f"Could not add to collection: {e}")

                checkpoint_dir = trainer_output_dir
                if os.path.exists(checkpoint_dir) and os.path.isdir(checkpoint_dir):
                    try:
                        shutil.rmtree(checkpoint_dir)
                        logger.info(
                            f"Deleted local checkpoint to free space: {checkpoint_dir}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to delete checkpoint {checkpoint_dir}: {e}"
                        )
            except Exception as e:
                logger.warning(f"Hub push failed, keeping local adapter artifact: {e}")

        return

    if hasattr(model, "merge_and_unload"):
        merged_model: Any = cast(Any, model).merge_and_unload()
    else:
        merged_model = model

    output_dir = os.path.join(str(trainer.args.output_dir), "final_merged_model")
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

    if config.hub and config.hub.enabled and config.hub.push_merged and i_am_main:
        repo_id = _build_hub_repo_id(config, merged=True)
        try:
            logger.info(f"Pushing merged model to HuggingFace Hub: {repo_id}")
            merged_model.push_to_hub(repo_id, private=config.hub.private)
            tokenizer.push_to_hub(repo_id, private=config.hub.private)
        except Exception as e:
            logger.warning(f"Hub push failed, keeping local merged artifact: {e}")
