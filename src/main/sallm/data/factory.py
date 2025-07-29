from __future__ import annotations

from typing import Optional, Tuple, Any, Dict, List

from datasets import Dataset, DatasetDict, load_from_disk, load_dataset
from transformers import AutoTokenizer

from sallm.config import ExperimentConfig, RunMode, FinetuneTaskType
from sallm.templates import registry as tmpl


def _reconstruct_entities_from_iob(
    tokens: List[str], ner_tag_ids: List[int], tag_map: List[str]
) -> List[str]:
    entities = []
    current_entity_tokens = []
    current_entity_label = None

    for token, tag_id in zip(tokens, ner_tag_ids):
        tag_name = tag_map[tag_id]

        if tag_name.startswith("B-"):
            if current_entity_tokens:
                entity_text = " ".join(current_entity_tokens)
                entities.append(f"{current_entity_label}: {entity_text}")

            current_entity_tokens = [token]
            current_entity_label = tag_name[2:]
        elif tag_name.startswith("I-"):
            if current_entity_label == tag_name[2:]:
                current_entity_tokens.append(token)
            else:
                if current_entity_tokens:
                    entity_text = " ".join(current_entity_tokens)
                    entities.append(f"{current_entity_label}: {entity_text}")
                current_entity_tokens = []
                current_entity_label = None
        else:  # O tag
            if current_entity_tokens:
                entity_text = " ".join(current_entity_tokens)
                entities.append(f"{current_entity_label}: {entity_text}")
            current_entity_tokens = []
            current_entity_label = None

    if current_entity_tokens:
        entity_text = " ".join(current_entity_tokens)
        entities.append(f"{current_entity_label}: {entity_text}")

    return entities


def build_datasets(
    config: ExperimentConfig, tokenizer: AutoTokenizer, is_hpo: bool
) -> Tuple[Dataset, Dataset, Optional[Dataset]]:
    if config.mode == RunMode.FINETUNE:
        assert config.dataset, "Finetune mode requires a `dataset` block in the config."

        ds_cfg = config.dataset
        split_map = ds_cfg.splits

        train_raw = load_dataset(
            ds_cfg.hf_name,
            ds_cfg.subset,
            split=split_map["train"],
            trust_remote_code=True,
            # streaming=True,
        )
        val_raw = load_dataset(
            ds_cfg.hf_name,
            ds_cfg.subset,
            split=split_map["val"],
            trust_remote_code=True,
            # streaming=True,
        )

        train_ds = _build_finetune_dataset(train_raw, config)
        val_ds = _build_finetune_dataset(val_raw, config)
        return train_ds, val_ds, None

    data_conf = config.data
    dataset_dict = load_from_disk(data_conf.path)

    if not isinstance(dataset_dict, DatasetDict):
        raise TypeError(
            f"Expected data at {data_conf.path} to be a DatasetDict, "
            f"but found {type(dataset_dict)}"
        )

    train_ds = dataset_dict[data_conf.train_split]
    val_ds = dataset_dict[data_conf.eval_split]

    test_ds = None
    if not is_hpo and data_conf.test_split and data_conf.test_split in dataset_dict:
        test_ds = dataset_dict[data_conf.test_split]

    return train_ds, val_ds, test_ds


def _build_finetune_dataset(
    raw_ds: Dataset,
    cfg: ExperimentConfig,
) -> Dataset:
    ds_cfg = cfg.dataset
    assert (
        ds_cfg.task is not None
    ), "A `dataset.task` must be specified for fine-tuning."

    if ds_cfg.task == FinetuneTaskType.INSTRUCTION:

        def to_messages(ex):
            return {
                "messages": [
                    {"role": "user", "content": ex["instruction"]},
                    {"role": "assistant", "content": ex["output"]},
                ]
            }

        map_function, desc = to_messages, "Format AfriInstruct to chat"

    elif ds_cfg.task == FinetuneTaskType.NAMED_ENTITY_RECOGNITION:
        # TODO don't index into templates[0] if multiple templates are used
        template_spec = tmpl.get(ds_cfg.templates[0].id)

        tag_map = template_spec.ner_tags
        if not tag_map:
            raise ValueError(
                f"Template '{template_spec.id}' is for an NER task but is missing the 'ner_tags' list."
            )

        def to_messages_format_ner(
            ex: Dict[str, Any],
        ) -> Dict[str, List[Dict[str, str]]]:
            text_input = " ".join(ex["tokens"])
            user_prompt = template_spec.prompt.format(text=text_input)

            reconstructed_entities = _reconstruct_entities_from_iob(
                ex["tokens"], ex["ner_tags"], tag_map
            )
            assistant_response = " $ ".join(reconstructed_entities)

            return {
                "messages": [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_response},
                ]
            }

        map_function = to_messages_format_ner
        desc = "Formatting dataset for NER"

    elif ds_cfg.task == FinetuneTaskType.CLASSIFICATION:
        template_spec = tmpl.get(ds_cfg.templates[0].id)
        label_column = getattr(ds_cfg, "label_column", "label")
        if not template_spec.label_mapping:
            raise ValueError(
                f"Template '{template_spec.id}' is for a classification task but is missing 'label_mapping'."
            )
        numeric_keys = isinstance(next(iter(template_spec.label_mapping.keys())), int)
        str_to_int_key_map = {}
        if numeric_keys:
            str_to_int_key_map = {
                str(v).lower(): k for k, v in template_spec.label_mapping.items()
            }

        def to_messages_format_classification(
            ex: Dict[str, Any],
        ) -> Dict[str, List[Dict[str, str]]]:
            user_prompt = template_spec.prompt.format(**ex)
            raw_label = ex[label_column]

            key_to_use = None
            if numeric_keys:
                if isinstance(raw_label, str):
                    try:
                        key_to_use = int(raw_label)
                    except ValueError:
                        key_to_use = str_to_int_key_map.get(raw_label.lower())
                        if key_to_use is None:
                            raise ValueError(
                                f"Cannot map string label '{raw_label}' to an integer key."
                            )
                else:
                    key_to_use = int(raw_label)
            else:
                key_to_use = str(raw_label)

            assistant_response = template_spec.label_mapping[key_to_use]

            return {
                "messages": [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_response},
                ]
            }

        map_function = to_messages_format_classification
        desc = "Formatting dataset into 'messages' format"
    else:
        raise ValueError(f"Unsupported task type: {ds_cfg.task}")

    processed_ds = raw_ds.map(
        map_function,
        batched=False,
        remove_columns=raw_ds.column_names,
        desc=desc,
    )

    if ds_cfg.subset:
        processed_ds = processed_ds.add_column(
            "lang", [ds_cfg.subset] * len(processed_ds)
        )

    return processed_ds
