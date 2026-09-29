from __future__ import annotations

import logging
import os
import re
from typing import Any, cast

import torch
import wandb
from datasets import Dataset, IterableDataset
from omegaconf import OmegaConf
from sallm.chat_template import install_canonical_chat_template
from sallm.config import ExperimentConfig, FinetuneTaskType, to_resolved_dict
from sallm.data.factory import (
    build_conversation_dataset,
    build_datasets,
    resolve_eval_template_choice,
)
from sallm.models.factory import build_model, build_tokenizer
from sallm.training.factory import build_trainer
from tokenizers import AddedToken

logger = logging.getLogger(__name__)


def _configure_chat_template(tokenizer: Any) -> bool:
    installed = install_canonical_chat_template(tokenizer)
    if installed:
        logger.info("Tokenizer chat template not found. Applying default template.")
    return installed


def _is_hpo_run(config: ExperimentConfig) -> bool:
    wb = getattr(config, "wandb", None)
    wb_id = getattr(wb, "id", None) if wb is not None else None
    return isinstance(wb_id, str) and "sweep" in wb_id


def _configure_task_truncation(*, tokenizer, task_type) -> None:
    """Preserve assistant answers when fine-tuning examples exceed the limit."""
    if task_type in (
        FinetuneTaskType.CLASSIFICATION,
        FinetuneTaskType.INSTRUCTION,
    ):
        tokenizer.truncation_side = "left"


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


def _build_hub_repo_id(config: ExperimentConfig) -> str:
    if config.hub and config.hub.repo_id:
        repo_id = config.hub.repo_id.strip().rstrip("/")
        if repo_id.count("/") != 1:
            raise ValueError("hub.repo_id must use the 'owner/model' format")
        return repo_id

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

    repo_name = f"sallm-{arch}-{task}-{langs}"
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
    task_type = config.dataset.task if config.dataset is not None else None
    _configure_task_truncation(tokenizer=tokenizer, task_type=task_type)

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

    _configure_chat_template(tokenizer)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(
            "tokenizer.pad_token was not set, setting it to eos_token: %s",
            tokenizer.eos_token,
        )

    logger.info("Datasets …")
    train_ds, val_ds, _ = build_datasets(config, tokenizer, is_hpo=False)
    if isinstance(train_ds, IterableDataset) or isinstance(val_ds, IterableDataset):
        raise TypeError("Fine-tuning requires map-style HuggingFace datasets.")

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
    trainer.train(resume_from_checkpoint=resume_ckpt)
    logger.info("Fine-tuning done.")

    if is_hpo_run:
        return

    output_dir = os.path.join(str(trainer.args.output_dir), "final_model")
    if hasattr(model, "save_pretrained"):
        _sync_weight_tying_flag(model)
        safe_serialization = True
        if hasattr(trainer, "args") and hasattr(trainer.args, "save_safetensors"):
            safe_serialization = bool(trainer.args.save_safetensors)
        try:
            model.save_pretrained(output_dir, safe_serialization=safe_serialization)
        except RuntimeError as exc:
            message = str(exc)
            if safe_serialization and "shared tensors" in message:
                logger.warning(
                    "Retrying save_pretrained with safe_serialization=False due "
                    "to shared tensors."
                )
                model.save_pretrained(output_dir, safe_serialization=False)
            else:
                raise
    else:
        os.makedirs(output_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Saved final model to → {output_dir}")

    if config.hub and config.hub.enabled and i_am_main:
        repo_id = _build_hub_repo_id(config)
        try:
            logger.info(f"Pushing model to HuggingFace Hub: {repo_id}")
            cast(Any, model).push_to_hub(repo_id, private=config.hub.private)
            tokenizer.push_to_hub(repo_id, private=config.hub.private)
            if config.hub.collection_slug:
                from huggingface_hub import add_collection_item

                add_collection_item(
                    collection_slug=config.hub.collection_slug,
                    item_id=repo_id,
                    item_type="model",
                    exists_ok=True,
                )
        except Exception as e:
            logger.warning(f"Hub push failed, keeping the local model: {e}")
