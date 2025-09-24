from __future__ import annotations

import json
import logging
from pathlib import Path

import lm_eval
import numpy as np
import torch
from lm_eval.models.huggingface import HFLM
from peft import PeftModel
from tokenizers.decoders import ByteLevel
from transformers import AutoTokenizer
from transformers.modeling_utils import PreTrainedModel

from sallm.config import ModelEvalConfig
from sallm.evaluation.config import TaskPack

# TODO figure out how to not have these bespoke patches
if not hasattr(PreTrainedModel, "_sallm_resize_token_embeddings"):
    _original_resize_token_embeddings = PreTrainedModel.resize_token_embeddings

    def _sallm_resize_token_embeddings(
        self: PreTrainedModel,
        new_num_tokens: int | None = None,
        pad_to_multiple_of: int | None = None,
        mean_resizing: bool = True,
    ):
        output_embeddings = self.get_output_embeddings()
        if output_embeddings is not None:
            try:
                del output_embeddings.weight
            except AttributeError:
                parameters = getattr(output_embeddings, "_parameters", None)
                if (
                    isinstance(parameters, dict)
                    and parameters.get("weight") is not None
                ):
                    parameters.pop("weight")
        return _original_resize_token_embeddings(
            self,
            new_num_tokens,
            pad_to_multiple_of,
            mean_resizing,
        )

    PreTrainedModel.resize_token_embeddings = _sallm_resize_token_embeddings
    PreTrainedModel._sallm_resize_token_embeddings = _original_resize_token_embeddings

if not hasattr(PreTrainedModel, "_sallm_tie_or_clone_weights"):
    _original_tie_or_clone_weights = PreTrainedModel._tie_or_clone_weights

    def _sallm_tie_or_clone_weights(self, output_embeddings, input_embeddings):
        if output_embeddings is not None:
            try:
                del output_embeddings.weight
            except AttributeError:
                parameters = getattr(output_embeddings, "_parameters", None)
                if (
                    isinstance(parameters, dict)
                    and parameters.get("weight") is not None
                ):
                    parameters.pop("weight")
        try:
            return _original_tie_or_clone_weights(
                self, output_embeddings, input_embeddings
            )
        except KeyError as exc:
            if output_embeddings is not None and input_embeddings is not None:
                params = getattr(output_embeddings, "_parameters", None)
                if isinstance(params, dict):
                    params["weight"] = input_embeddings.weight
                output_embeddings.__dict__["weight"] = input_embeddings.weight
                return None
            raise exc

    PreTrainedModel._tie_or_clone_weights = _sallm_tie_or_clone_weights
    PreTrainedModel._sallm_tie_or_clone_weights = _original_tie_or_clone_weights

logger = logging.getLogger(__name__)


def _safe_json_encoder(obj):
    if isinstance(obj, np.integer) or isinstance(obj, np.floating):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.tolist() if obj.ndim else obj.item()
    if isinstance(obj, torch.dtype):
        return str(obj)
    return str(obj)


def _load_tokenizer_and_pretrained(
    checkpoint: str, trust_remote_code: bool = True, adapter_path: str | None = None
):
    checkpoint_path = Path(checkpoint)
    pretrained_id = checkpoint
    if checkpoint_path.exists():
        try:
            pretrained_id = str(checkpoint_path.resolve())
        except Exception:
            pretrained_id = str(checkpoint_path)
    adapter_tokenizer = None
    if adapter_path:
        candidate = Path(adapter_path)
        if candidate.exists() and candidate.is_dir():
            tokenizer_files = (
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
                "tokenizer.model",
            )
            if any((candidate / name).exists() for name in tokenizer_files):
                adapter_tokenizer = candidate
    tokenizer_root = adapter_tokenizer
    if tokenizer_root is None and checkpoint_path.exists():
        tokenizer_root = checkpoint_path
    if tokenizer_root is not None and tokenizer_root.exists():
        try:
            tokenizer_resolved = str(tokenizer_root.resolve())
        except Exception:
            tokenizer_resolved = str(tokenizer_root)
        source_label = (
            "adapter checkpoint"
            if adapter_tokenizer is not None and tokenizer_root == adapter_tokenizer
            else "local checkpoint"
        )
        logger.info("Loading tokenizer from %s: %s", source_label, tokenizer_resolved)
        tok = AutoTokenizer.from_pretrained(
            tokenizer_resolved,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        )
        return tok, pretrained_id
    logger.info("Loading tokenizer from HF hub or identifier: %s", checkpoint)
    tok = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=trust_remote_code)
    return tok, pretrained_id


def evaluate_pack(
    pack: TaskPack,
    model_cfg: ModelEvalConfig,
    out_dir: Path,
    overrides: dict[str, dict],
) -> dict:
    pack_over = overrides.get(pack.name, {})
    task_list = pack_over.get("tasks", pack.tasks)
    fewshot = int(pack_over.get("fewshot", pack.fewshot))
    batch_size = int(pack_over.get("batch_size", pack.batch_size))
    apply_chat_template = pack_over.get("apply_chat_template", pack.apply_chat_template)

    tok, pretrained_id = _load_tokenizer_and_pretrained(
        model_cfg.checkpoint,
        trust_remote_code=True,
        adapter_path=model_cfg.peft_adapter,
    )
    tok.backend_tokenizer.decoder = ByteLevel()

    hf_model = HFLM(
        pretrained=pretrained_id,
        tokenizer=tok,
        peft=model_cfg.peft_adapter,
        device=model_cfg.device,
        dtype=model_cfg.dtype,
        trust_remote_code=True,
    )

    genconf = hf_model.model.generation_config
    genconf.eos_token_id = tok.eos_token_id
    genconf.pad_token_id = tok.pad_token_id or tok.eos_token_id

    if model_cfg.merge_lora and isinstance(hf_model.model, PeftModel):
        hf_model._model = hf_model.model.merge_and_unload()

    try:
        results = lm_eval.simple_evaluate(
            model=hf_model,
            tasks=task_list,
            num_fewshot=fewshot,
            batch_size=batch_size,
            apply_chat_template=apply_chat_template,
            log_samples=True,
            write_out=True,
        )
    except TypeError:
        raise

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pack.name}.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2, default=_safe_json_encoder)

    return results
