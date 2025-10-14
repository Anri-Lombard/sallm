from __future__ import annotations

import logging
from math import isfinite
from typing import Any

from datasets import (
    Dataset,
    DatasetDict,
    concatenate_datasets,
    get_dataset_config_names,
    load_dataset,
    load_from_disk,
)
from transformers import AutoTokenizer

from sallm.config import (
    ExperimentConfig,
    FinetuneDatasetConfig,
    FinetuneTaskType,
    RunMode,
    TemplateChoice,
    TemplateRef,
)
from sallm.data.afrihg import load_afrihg_from_github
from sallm.data.multitask import TaskComponent, WeightedMultiTaskDataset
from sallm.data.t2x import load_t2x_from_github
from sallm.templates import registry as tmpl

# TODO cleanup this file - way too complicated right now


logger = logging.getLogger(__name__)


def _reconstruct_entities_from_iob(
    tokens: list[str], ner_tag_ids: list[int], tag_map: list[str]
) -> list[str]:
    entities = []
    current_entity_tokens = []
    current_entity_label = None

    for token, tag_id in zip(tokens, ner_tag_ids, strict=False):
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


def _safe_format_prompt(prompt: str, values: dict[str, Any]) -> str:
    safe: dict[str, str] = {}
    for k, v in values.items():
        try:
            if v is None:
                safe[k] = ""
            else:
                safe[k] = str(v).strip()
        except Exception:
            safe[k] = str(v)
    return prompt.format(**safe)


def build_datasets(
    config: ExperimentConfig, tokenizer: AutoTokenizer, is_hpo: bool
) -> tuple[Dataset, Dataset, Dataset | None]:
    def _load_split_with_fallback(
        hf_name: str, name: str | None, split: str
    ) -> Dataset:
        try:
            return load_dataset(hf_name, name=name, split=split, trust_remote_code=True)
        except Exception as err:
            s = split.lower()
            if s in ("validation", "valid", "val"):
                candidates = ["validation", "dev", "val", "valid", "test"]
            elif s == "dev":
                candidates = ["dev", "validation", "val", "valid", "test"]
            elif s == "test":
                candidates = ["test", "validation", "dev", "val", "valid"]
            else:
                candidates = [split, "validation", "dev", "val", "valid", "test"]

            seen: set[str] = set()
            last_err = err
            for alt in candidates:
                if alt in seen:
                    continue
                seen.add(alt)
                try:
                    return load_dataset(
                        hf_name, name=name, split=alt, trust_remote_code=True
                    )
                except Exception as e:
                    last_err = e
                    continue
            raise last_err from None

    if config.mode == RunMode.FINETUNE:
        assert config.dataset, "Finetune mode requires a `dataset` block."
        ds_cfg = config.dataset
        splits = ds_cfg.splits
        lang_tag = ds_cfg.subset
        lang_list = set(ds_cfg.languages or [])

        if isinstance(ds_cfg.hf_name, str) and ds_cfg.hf_name.startswith("mix:"):
            mix_name = ds_cfg.hf_name[len("mix:") :].strip().lower()

            def _cfg_like(base: FinetuneDatasetConfig):
                class _C:
                    pass

                c = _C()
                c.dataset = base
                return c

            def _load_component_raw(
                comp: FinetuneDatasetConfig,
            ) -> tuple[Dataset, Dataset]:
                if isinstance(comp.hf_name, str) and comp.hf_name.startswith("github:"):
                    gh_ref = comp.hf_name[len("github:") :]
                    if "francois-meyer/t2x" in gh_ref or gh_ref.strip().endswith(
                        "/t2x"
                    ):
                        ds_from_github = load_t2x_from_github()
                    else:
                        ds_from_github = load_afrihg_from_github(
                            languages=comp.languages
                        )
                    if isinstance(ds_from_github, DatasetDict):
                        tr = (
                            ds_from_github["train"]
                            if "train" in ds_from_github
                            else ds_from_github[next(iter(ds_from_github.keys()))]
                        )
                        if "validation" in ds_from_github:
                            va = ds_from_github["validation"]
                        elif "dev" in ds_from_github:
                            va = ds_from_github["dev"]
                        elif "test" in ds_from_github:
                            va = ds_from_github["test"]
                        else:
                            va = tr
                    else:
                        tr = ds_from_github
                        va = ds_from_github
                    return tr, va

                available_configs = get_dataset_config_names(
                    comp.hf_name, trust_remote_code=True
                )
                load_name = comp.subset
                lang_list_cfg = list(comp.languages or [])
                if lang_list_cfg:
                    can_multi_load = all(
                        lang_code in available_configs for lang_code in lang_list_cfg
                    )
                    if can_multi_load:
                        train_parts: list[Dataset] = []
                        val_parts: list[Dataset] = []
                        for lang_code in lang_list_cfg:
                            tr = load_dataset(
                                comp.hf_name,
                                name=lang_code,
                                split=comp.splits["train"],
                                trust_remote_code=True,
                            )
                            va = _load_split_with_fallback(
                                comp.hf_name, lang_code, comp.splits["val"]
                            )
                            if "lang" not in tr.column_names:
                                tr = tr.add_column("lang", [lang_code] * len(tr))
                            if "lang" not in va.column_names:
                                va = va.add_column("lang", [lang_code] * len(va))
                            train_parts.append(tr)
                            val_parts.append(va)
                        return concatenate_datasets(train_parts), concatenate_datasets(
                            val_parts
                        )
                tr = load_dataset(
                    comp.hf_name,
                    name=load_name,
                    split=comp.splits["train"],
                    trust_remote_code=True,
                )
                va = _load_split_with_fallback(
                    comp.hf_name, load_name, comp.splits["val"]
                )
                return tr, va

            def _process_component(
                comp: FinetuneDatasetConfig,
            ) -> tuple[Dataset, Dataset]:
                tr_raw, va_raw = _load_component_raw(comp)
                tr = build_conversation_dataset(tr_raw, _cfg_like(comp))
                va = build_conversation_dataset(va_raw, _cfg_like(comp))
                return tr, va

            if mix_name in ("sa_general", "sa-general", "sa_all", "sa-all"):
                common_seq_len = ds_cfg.max_seq_length
                common_packing = ds_cfg.packing
                common_asst_only = ds_cfg.assistant_only_loss

                mix_components: list[tuple[str, FinetuneDatasetConfig]] = [
                    (
                        "sib",
                        FinetuneDatasetConfig(
                            hf_name="Davlan/sib200",
                            subset=None,
                            languages=[
                                "afr_Latn",
                                "eng_Latn",
                                "nso_Latn",
                                "sot_Latn",
                                "xho_Latn",
                                "zul_Latn",
                            ],
                            task=FinetuneTaskType.CLASSIFICATION,
                            splits={"train": "train", "val": "validation"},
                            templates=[
                                TemplateRef(
                                    id="sib_topic_classification/lm_eval_p1",
                                    weight=1.0,
                                ),
                                TemplateRef(
                                    id="sib_topic_classification/lm_eval_p2",
                                    weight=1.0,
                                ),
                                TemplateRef(
                                    id="sib_topic_classification/lm_eval_p3",
                                    weight=1.0,
                                ),
                                TemplateRef(
                                    id="sib_topic_classification/lm_eval_p4",
                                    weight=1.0,
                                ),
                                TemplateRef(
                                    id="sib_topic_classification/lm_eval_p5",
                                    weight=1.0,
                                ),
                            ],
                            template_choice=TemplateChoice.CYCLE,
                            label_column="category",
                            max_seq_length=common_seq_len,
                            packing=common_packing,
                            assistant_only_loss=common_asst_only,
                        ),
                    ),
                    (
                        "news",
                        FinetuneDatasetConfig(
                            hf_name="masakhane/masakhanews",
                            subset=None,
                            languages=["eng", "xho"],
                            task=FinetuneTaskType.CLASSIFICATION,
                            splits={"train": "train", "val": "validation"},
                            templates=[
                                TemplateRef(
                                    id="masakhane_news_classification/lm_eval_p1",
                                    weight=1.0,
                                ),
                                TemplateRef(
                                    id="masakhane_news_classification/lm_eval_p2",
                                    weight=1.0,
                                ),
                                TemplateRef(
                                    id="masakhane_news_classification/lm_eval_p3",
                                    weight=1.0,
                                ),
                                TemplateRef(
                                    id="masakhane_news_classification/lm_eval_p4",
                                    weight=1.0,
                                ),
                                TemplateRef(
                                    id="masakhane_news_classification/lm_eval_p5",
                                    weight=1.0,
                                ),
                            ],
                            template_choice=TemplateChoice.CYCLE,
                            label_column="label",
                            max_seq_length=common_seq_len,
                            packing=common_packing,
                            assistant_only_loss=common_asst_only,
                        ),
                    ),
                    (
                        "ner",
                        FinetuneDatasetConfig(
                            hf_name="masakhane/masakhaner2",
                            subset=None,
                            languages=["tsn", "xho", "zul"],
                            task=FinetuneTaskType.NAMED_ENTITY_RECOGNITION,
                            splits={"train": "train", "val": "validation"},
                            templates=[
                                TemplateRef(
                                    id="masakhane_named_entity_recognition/lm_eval_p1",
                                    weight=1.0,
                                ),
                                TemplateRef(
                                    id="masakhane_named_entity_recognition/lm_eval_p2",
                                    weight=1.0,
                                ),
                                TemplateRef(
                                    id="masakhane_named_entity_recognition/lm_eval_p3",
                                    weight=1.0,
                                ),
                                TemplateRef(
                                    id="masakhane_named_entity_recognition/lm_eval_p4",
                                    weight=1.0,
                                ),
                                TemplateRef(
                                    id="masakhane_named_entity_recognition/lm_eval_p5",
                                    weight=1.0,
                                ),
                            ],
                            template_choice=TemplateChoice.CYCLE,
                            max_seq_length=common_seq_len,
                            packing=common_packing,
                            assistant_only_loss=common_asst_only,
                        ),
                    ),
                    (
                        "pos",
                        FinetuneDatasetConfig(
                            hf_name="masakhane/masakhapos",
                            subset=None,
                            languages=["tsn", "xho", "zul"],
                            task=FinetuneTaskType.POS_TAGGING,
                            splits={"train": "train", "val": "validation"},
                            templates=[
                                TemplateRef(
                                    id="masakhane_pos_tagging/lm_eval_p1",
                                    weight=1.0,
                                ),
                                TemplateRef(
                                    id="masakhane_pos_tagging/lm_eval_p2",
                                    weight=1.0,
                                ),
                                TemplateRef(
                                    id="masakhane_pos_tagging/lm_eval_p3",
                                    weight=1.0,
                                ),
                                TemplateRef(
                                    id="masakhane_pos_tagging/lm_eval_p4",
                                    weight=1.0,
                                ),
                                TemplateRef(
                                    id="masakhane_pos_tagging/lm_eval_p5",
                                    weight=1.0,
                                ),
                            ],
                            template_choice=TemplateChoice.CYCLE,
                            max_seq_length=common_seq_len,
                            packing=common_packing,
                            assistant_only_loss=common_asst_only,
                        ),
                    ),
                    (
                        "afrihg",
                        FinetuneDatasetConfig(
                            hf_name="github:dadelani/AfriHG",
                            subset=None,
                            languages=["xho", "zul"],
                            task=FinetuneTaskType.INSTRUCTION,
                            splits={"train": "train", "val": "validation"},
                            templates=[
                                TemplateRef(id="afrihg_headline/v1", weight=1.0)
                            ],
                            template_choice=TemplateChoice.CYCLE,
                            max_seq_length=common_seq_len,
                            packing=common_packing,
                            assistant_only_loss=common_asst_only,
                        ),
                    ),
                    (
                        "t2x",
                        FinetuneDatasetConfig(
                            hf_name="github:francois-meyer/t2x",
                            subset="xho",
                            languages=None,
                            task=FinetuneTaskType.INSTRUCTION,
                            splits={"train": "train", "val": "validation"},
                            templates=[
                                TemplateRef(id="t2x_verbalisation/v1", weight=1.0)
                            ],
                            template_choice=TemplateChoice.CYCLE,
                            max_seq_length=(
                                1024 if common_seq_len > 1024 else common_seq_len
                            ),
                            packing=common_packing,
                            assistant_only_loss=common_asst_only,
                        ),
                    ),
                ]

                weight_map = dict(ds_cfg.mix_weights)
                missing_weights = [
                    name for name, _ in mix_components if name not in weight_map
                ]
                if missing_weights:
                    raise ValueError(
                        f"Missing mix weights for components: {sorted(missing_weights)}"
                    )

                train_components: list[TaskComponent] = []
                val_parts: list[Dataset] = []
                for name, comp_cfg in mix_components:
                    train_processed, val_processed = _process_component(comp_cfg)
                    train_components.append(
                        TaskComponent(
                            name=name,
                            dataset=train_processed,
                            weight=weight_map[name],
                        )
                    )
                    val_parts.append(val_processed)

                epoch_size_cfg = ds_cfg.mix_epoch_size
                if isinstance(epoch_size_cfg, str):
                    epoch_size_value = None if epoch_size_cfg == "sum" else None
                elif isinstance(epoch_size_cfg, int):
                    epoch_size_value = epoch_size_cfg
                else:
                    epoch_size_value = None

                seed_value = 0
                training_cfg = config.training
                if training_cfg is not None:
                    try:
                        seed_candidate = training_cfg.get("seed", 0)
                        if seed_candidate is not None:
                            seed_value = int(seed_candidate)
                    except Exception:
                        seed_value = 0

                train_mix = WeightedMultiTaskDataset(
                    train_components,
                    seed=seed_value,
                    temperature=ds_cfg.mix_temperature,
                    epoch_size=epoch_size_value,
                    min_prob=ds_cfg.mix_min_prob,
                    max_prob=ds_cfg.mix_max_prob,
                )

                logger.info("SA General mix distribution | %s", train_mix.describe())

                # Materialize one epoch worth of samples into an in-memory HF Dataset
                materialized = [train_mix[i] for i in range(len(train_mix))]
                hf_ep_dataset = Dataset.from_list(materialized)

                return (
                    hf_ep_dataset,
                    concatenate_datasets(val_parts),
                    None,
                )

            raise ValueError(f"Unsupported dataset mix '{mix_name}'")

        filter_after_load = False
        # TODO: improve this logic
        if isinstance(ds_cfg.hf_name, str) and ds_cfg.hf_name.startswith("github:"):
            gh_ref = ds_cfg.hf_name[len("github:") :]
            if "francois-meyer/t2x" in gh_ref or gh_ref.strip().endswith("/t2x"):
                ds_from_github = load_t2x_from_github()
            else:
                langs = [ds_cfg.subset] if ds_cfg.subset else ds_cfg.languages
                ds_from_github = load_afrihg_from_github(languages=langs)
            # load_afrihg_from_github returns a DatasetDict. Select concrete
            # Dataset splits to pass to the finetune builder.
            if isinstance(ds_from_github, DatasetDict):
                if "train" in ds_from_github:
                    train_raw = ds_from_github["train"]
                else:
                    first_split = next(iter(ds_from_github.keys()))
                    train_raw = ds_from_github[first_split]

                if "validation" in ds_from_github:
                    val_raw = ds_from_github["validation"]
                elif "dev" in ds_from_github:
                    val_raw = ds_from_github["dev"]
                elif "test" in ds_from_github:
                    val_raw = ds_from_github["test"]
                else:
                    val_raw = train_raw
            else:
                train_raw = ds_from_github
                val_raw = ds_from_github
        else:
            available_configs = get_dataset_config_names(
                ds_cfg.hf_name, trust_remote_code=True
            )
            load_name = ds_cfg.subset
            filter_after_load = False
            lang_list_cfg = list(ds_cfg.languages or [])

            if lang_list_cfg:
                can_multi_load = all(
                    lang_code in available_configs for lang_code in lang_list_cfg
                )
                if can_multi_load:
                    train_parts: list[Dataset] = []
                    val_parts: list[Dataset] = []
                    for lang_code in lang_list_cfg:
                        tr = load_dataset(
                            ds_cfg.hf_name,
                            name=lang_code,
                            split=splits["train"],
                            trust_remote_code=True,
                        )
                        va = _load_split_with_fallback(
                            ds_cfg.hf_name, lang_code, splits["val"]
                        )
                        if "lang" not in tr.column_names:
                            tr = tr.add_column("lang", [lang_code] * len(tr))
                        if "lang" not in va.column_names:
                            va = va.add_column("lang", [lang_code] * len(va))
                        train_parts.append(tr)
                        val_parts.append(va)
                    train_raw = concatenate_datasets(train_parts)
                    val_raw = concatenate_datasets(val_parts)
                    load_name = None
                    filter_after_load = False
                else:
                    load_name = None
                    filter_after_load = True
            else:
                if ds_cfg.subset not in available_configs:
                    load_name = None
                    filter_after_load = True

                train_raw = load_dataset(
                    ds_cfg.hf_name,
                    name=load_name,
                    split=splits["train"],
                    trust_remote_code=True,
                )
                val_raw = _load_split_with_fallback(
                    ds_cfg.hf_name, load_name, splits["val"]
                )

        if filter_after_load and lang_tag:

            def _matches_lang(ex: dict[str, Any]) -> bool:
                # Prefer 'lang', else fallback to 'language' or 'language_code'
                if "lang" in ex:
                    return ex["lang"] == lang_tag
                if "language" in ex:
                    return ex["language"] == lang_tag
                if "language_code" in ex:
                    return ex["language_code"] == lang_tag
                return False

            train_raw = train_raw.filter(_matches_lang)
            val_raw = val_raw.filter(_matches_lang)

        # Optional multi-language filtering if a languages list is provided
        # Only apply when a language indicator column exists, otherwise skip.
        if lang_list:

            def _in_lang_list(ex: dict[str, Any]) -> bool:
                code = ex.get("lang") or ex.get("language_code") or ex.get("language")
                return code in lang_list

            has_lang_col = any(
                col in train_raw.column_names
                for col in ("lang", "language_code", "language")
            )
            if has_lang_col:
                train_raw = train_raw.filter(_in_lang_list)
                val_raw = val_raw.filter(_in_lang_list)

        train_ds = build_conversation_dataset(train_raw, config)
        val_ds = build_conversation_dataset(val_raw, config)
        return train_ds, val_ds, None

    data_conf = config.data
    dataset_dict = load_from_disk(data_conf.path)

    if not isinstance(dataset_dict, DatasetDict):
        raise TypeError(
            f"Expected DatasetDict at {data_conf.path}, found {type(dataset_dict)}"
        )

    train_ds = dataset_dict[data_conf.train_split]
    val_ds = dataset_dict[data_conf.eval_split]

    test_ds = None
    if not is_hpo and data_conf.test_split and data_conf.test_split in dataset_dict:
        test_ds = dataset_dict[data_conf.test_split]

    return train_ds, val_ds, test_ds


def build_conversation_dataset(
    raw_ds: Dataset,
    cfg: ExperimentConfig,
) -> Dataset:
    ds_cfg = cfg.dataset
    if ds_cfg.task is None:
        raise ValueError("A `dataset.task` must be specified for fine-tuning.")

    if "messages" in raw_ds.column_names:
        return raw_ds

    # TODO: fix this logic
    def _extract_instruction_pair(ex: dict[str, Any]) -> tuple[str, str]:
        """Return (user_prompt, assistant_response) for instruction-style datasets.

        Supports multiple common schemas:
        - {instruction, output}
        - {inputs, targets}  # Aya/xP3x
        - {prompt, response}
        - {query, response}
        """
        # Preferred keys
        if "instruction" in ex and "output" in ex:
            return str(ex["instruction"]), str(ex["output"])
        # Aya / xP3x style
        if "inputs" in ex and "targets" in ex:
            return str(ex["inputs"]), str(ex["targets"])
        # Common data-to-text style (T2X)
        if "source" in ex and "target" in ex:
            return str(ex["source"]), str(ex["target"])
        if "data" in ex and "text" in ex:
            return str(ex["data"]), str(ex["text"])
        if "triple" in ex and "verbalisation" in ex:
            return str(ex["triple"]), str(ex["verbalisation"])
        # Other common aliases
        if "prompt" in ex and "response" in ex:
            return str(ex["prompt"]), str(ex["response"])
        if "query" in ex and "response" in ex:
            return str(ex["query"]), str(ex["response"])
        # Headline-style generation datasets: article/text -> headline/title
        if "text" in ex and "title" in ex:
            return str(ex["text"]), str(ex["title"])
        if "article" in ex and "headline" in ex:
            return str(ex["article"]), str(ex["headline"])
        if "text" in ex and "headline" in ex:
            return str(ex["text"]), str(ex["headline"])

        # Try to find likely fields as last resort
        candidates_user = [
            "instruction",
            "inputs",
            "prompt",
            "query",
            "input",
            "source",
            "data",
            "triple",
        ]
        candidates_assistant = [
            "output",
            "targets",
            "response",
            "target",
            "answer",
            "text",
            "headline",
            "verbalisation",
        ]
        user_val = next((ex[k] for k in candidates_user if k in ex), None)
        asst_val = next((ex[k] for k in candidates_assistant if k in ex), None)
        if user_val is not None and asst_val is not None:
            return str(user_val), str(asst_val)
        raise KeyError(
            "Unable to locate instruction/response fields. Expected one of: "
            "(instruction, output) or (inputs, targets) or (prompt/query, response)."
        )

    def _format_instruction(ex: dict[str, Any]) -> list[dict[str, str]]:
        user_text, assistant_text = _extract_instruction_pair(ex)
        return [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]

    def _format_classification(
        ex: dict[str, Any],
        template_id: str,
        label_column: str,
    ) -> list[dict[str, str]]:
        template_spec = tmpl.get(template_id)
        if not template_spec.label_mapping:
            raise ValueError(
                f"Template '{template_spec.id}' is for a classification task "
                "but is missing 'label_mapping'."
            )
        user_prompt = _safe_format_prompt(template_spec.prompt, ex)
        raw_label = ex[label_column]

        numeric_keys = isinstance(next(iter(template_spec.label_mapping.keys())), int)
        if numeric_keys:
            if isinstance(raw_label, str):
                try:
                    key_to_use = int(raw_label)
                except ValueError as err:
                    str_to_int_key_map = {
                        str(v).lower(): k
                        for k, v in template_spec.label_mapping.items()
                    }
                    key_to_use = str_to_int_key_map.get(raw_label.lower())
                    if key_to_use is None:
                        raise ValueError("Cannot map label to integer key") from err
            else:
                key_to_use = int(raw_label)

            if (
                key_to_use not in template_spec.label_mapping
                and (key_to_use - 1) in template_spec.label_mapping
            ):
                key_to_use -= 1
        else:
            key_to_use = str(raw_label)

        assistant_response = template_spec.label_mapping[key_to_use]
        return [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response},
        ]

    def _format_ner(ex: dict[str, Any], template_id: str) -> list[dict[str, str]]:
        template_spec = tmpl.get(template_id)
        tag_map = template_spec.ner_tags
        if not tag_map:
            raise ValueError(
                f"Template '{template_spec.id}' is for an NER task "
                "but is missing the 'ner_tags' list."
            )
        text_input = " ".join(ex["tokens"])
        user_prompt = _safe_format_prompt(template_spec.prompt, {"text": text_input})
        reconstructed_entities = _reconstruct_entities_from_iob(
            ex["tokens"], ex["ner_tags"], tag_map
        )
        assistant_response = " $$ ".join(reconstructed_entities)
        return [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response},
        ]

    if ds_cfg.template_choice == TemplateChoice.ALL:
        template_ids = [t.id for t in ds_cfg.templates]
        if len(template_ids) == 0:
            raise ValueError(
                "dataset.template_choice is 'ALL' but no templates were provided."
            )

        if ds_cfg.task == FinetuneTaskType.NAMED_ENTITY_RECOGNITION:
            for t_id in template_ids:
                if not tmpl.get(t_id).ner_tags:
                    raise ValueError(
                        f"Template '{t_id}' selected for NER but missing 'ner_tags'."
                    )
        if ds_cfg.task == FinetuneTaskType.CLASSIFICATION:
            for t_id in template_ids:
                if not tmpl.get(t_id).label_mapping:
                    raise ValueError(
                        f"Template '{t_id}' selected for classification "
                        "but missing 'label_mapping'."
                    )
            if ds_cfg.label_column not in raw_ds.column_names:
                raise ValueError(
                    f"Classification label column '{ds_cfg.label_column}' "
                    "not found in dataset."
                )

        upos_class_names = None
        if ds_cfg.task == FinetuneTaskType.POS_TAGGING:
            try:
                upos_feat = raw_ds.features.get("upos")
                if hasattr(upos_feat, "feature") and hasattr(
                    upos_feat.feature, "names"
                ):
                    upos_class_names = list(upos_feat.feature.names)
            except Exception:
                upos_class_names = None

        def _format_pos(ex: dict[str, Any], template_id: str) -> list[dict[str, str]]:
            tokens: list[str] = ex["tokens"]
            raw_upos = ex["upos"]
            if raw_upos and isinstance(raw_upos[0], int) and upos_class_names:
                upos_tags = [upos_class_names[i] for i in raw_upos]
            else:
                upos_tags = [str(t) for t in raw_upos]
            if len(tokens) != len(upos_tags):
                raise ValueError(
                    "MasakhaPOS sample length mismatch: "
                    f"tokens={len(tokens)} vs upos={len(upos_tags)}"
                )
            tuple_list_str = (
                "["
                + ", ".join(
                    f"({repr(tok)}, {repr(tag)})"
                    for tok, tag in zip(tokens, upos_tags, strict=False)
                )
                + "]"
            )
            spec = tmpl.get(template_id)
            tokens_repr = "[" + ", ".join(repr(t) for t in tokens) + "]"
            user_prompt = _safe_format_prompt(spec.prompt, {"tokens": tokens_repr})
            return [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": tuple_list_str},
            ]

        def _expand_batch(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
            out_messages: list[list[dict[str, str]]] = []
            out_template_ids: list[str] = []
            out_langs: list[str] = []

            n = len(next(iter(batch.values()))) if batch else 0
            for i in range(n):
                ex = {k: v[i] for k, v in batch.items()}

                for tref in ds_cfg.templates:
                    t_id = tref.id
                    w = int(tref.weight) if isfinite(tref.weight) else 1
                    w = max(w, 1)

                    if ds_cfg.task == FinetuneTaskType.INSTRUCTION:
                        msgs = _format_instruction(ex)
                    elif ds_cfg.task == FinetuneTaskType.CLASSIFICATION:
                        msgs = _format_classification(
                            ex, t_id, ds_cfg.label_column or "label"
                        )
                    elif ds_cfg.task == FinetuneTaskType.NAMED_ENTITY_RECOGNITION:
                        msgs = _format_ner(ex, t_id)
                    elif ds_cfg.task == FinetuneTaskType.POS_TAGGING:
                        msgs = _format_pos(ex, t_id)
                    else:
                        raise ValueError(f"Unsupported task type: {ds_cfg.task}")

                    for _ in range(w):
                        out_messages.append(msgs)
                        out_template_ids.append(t_id)
                        lang_val = ex.get("lang") if isinstance(ex, dict) else None
                        if lang_val:
                            out_langs.append(str(lang_val))
                        elif ds_cfg.subset:
                            out_langs.append(ds_cfg.subset)

            out: dict[str, list[Any]] = {
                "messages": out_messages,
                "template_id": out_template_ids,
            }
            if out_langs:
                out["lang"] = out_langs
            return out

        processed_ds = raw_ds.map(
            _expand_batch,
            batched=True,
            remove_columns=raw_ds.column_names,
            desc="Expanding ALL templates",
        )

        # Preserve language column
        if ds_cfg.subset and "lang" not in processed_ds.column_names:
            processed_ds = processed_ds.add_column(
                "lang", [ds_cfg.subset] * len(processed_ds)
            )
        processed_ds = processed_ds.shuffle(seed=42)
        return processed_ds

    # Cycle strategy: assign exactly one template per example in round-robin
    if ds_cfg.template_choice == TemplateChoice.CYCLE and ds_cfg.templates:
        cycle_template_ids = [t.id for t in ds_cfg.templates]
        if len(cycle_template_ids) == 0:
            raise ValueError(
                "dataset.template_choice is 'CYCLE' but no templates were provided."
            )

        if ds_cfg.task == FinetuneTaskType.INSTRUCTION:

            def _cycle_instruction(ex: dict[str, Any], idx: int) -> dict[str, Any]:
                t_id = cycle_template_ids[idx % len(cycle_template_ids)]
                user_text, assistant_text = _extract_instruction_pair(ex)
                try:
                    spec = tmpl.get(t_id)
                    user_prompt = _safe_format_prompt(spec.prompt, ex)
                except Exception:
                    user_prompt = user_text
                msgs = [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_text},
                ]
                out: dict[str, Any] = {"messages": msgs, "template_id": t_id}
                lang_val = ex.get("lang") if isinstance(ex, dict) else None
                if lang_val:
                    out["lang"] = str(lang_val)
                elif ds_cfg.subset:
                    out["lang"] = ds_cfg.subset
                return out

            processed_ds = raw_ds.map(
                _cycle_instruction,
                batched=False,
                with_indices=True,
                remove_columns=raw_ds.column_names,
                desc="Cycling instruction templates",
            )

            return processed_ds

        if ds_cfg.task == FinetuneTaskType.NAMED_ENTITY_RECOGNITION:
            for t_id in cycle_template_ids:
                if not tmpl.get(t_id).ner_tags:
                    raise ValueError(
                        f"Template '{t_id}' selected for NER but missing 'ner_tags'."
                    )

            def _cycle_ner(ex: dict[str, Any], idx: int) -> dict[str, Any]:
                t_id = cycle_template_ids[idx % len(cycle_template_ids)]
                msgs = _format_ner(ex, t_id)
                out: dict[str, Any] = {"messages": msgs, "template_id": t_id}
                lang_val = ex.get("lang") if isinstance(ex, dict) else None
                if lang_val:
                    out["lang"] = str(lang_val)
                elif ds_cfg.subset:
                    out["lang"] = ds_cfg.subset
                return out

            processed_ds = raw_ds.map(
                _cycle_ner,
                batched=False,
                with_indices=True,
                remove_columns=raw_ds.column_names,
                desc="Cycling NER templates",
            )

        elif ds_cfg.task == FinetuneTaskType.CLASSIFICATION:
            for t_id in cycle_template_ids:
                if not tmpl.get(t_id).label_mapping:
                    raise ValueError(
                        f"Template '{t_id}' selected for classification "
                        "but missing 'label_mapping'."
                    )
            if ds_cfg.label_column not in raw_ds.column_names:
                raise ValueError(
                    f"Classification label column '{ds_cfg.label_column}' "
                    "not found in dataset."
                )

            def _cycle_classification(ex: dict[str, Any], idx: int) -> dict[str, Any]:
                t_id = cycle_template_ids[idx % len(cycle_template_ids)]
                msgs = _format_classification(ex, t_id, ds_cfg.label_column or "label")
                out: dict[str, Any] = {"messages": msgs, "template_id": t_id}
                lang_val = ex.get("lang") if isinstance(ex, dict) else None
                if lang_val:
                    out["lang"] = str(lang_val)
                elif ds_cfg.subset:
                    out["lang"] = ds_cfg.subset
                return out

            processed_ds = raw_ds.map(
                _cycle_classification,
                batched=False,
                with_indices=True,
                remove_columns=raw_ds.column_names,
                desc="Cycling classification templates",
            )

        elif ds_cfg.task == FinetuneTaskType.POS_TAGGING:
            upos_class_names = None
            try:
                upos_feat = raw_ds.features.get("upos")
                if hasattr(upos_feat, "feature") and hasattr(
                    upos_feat.feature, "names"
                ):
                    upos_class_names = list(upos_feat.feature.names)
            except Exception:
                upos_class_names = None

            def _render_prompt(tokens_list: list[str], t_id: str) -> str:
                tokens_repr = "[" + ", ".join(repr(t) for t in tokens_list) + "]"
                spec = tmpl.get(t_id)
                return _safe_format_prompt(spec.prompt, {"tokens": tokens_repr})

            def _cycle_pos(ex: dict[str, Any], idx: int) -> dict[str, Any]:
                t_id = cycle_template_ids[idx % len(cycle_template_ids)]
                tokens: list[str] = ex["tokens"]
                raw_upos = ex["upos"]
                if raw_upos and isinstance(raw_upos[0], int) and upos_class_names:
                    upos_tags = [upos_class_names[i] for i in raw_upos]
                else:
                    upos_tags = [str(t) for t in raw_upos]
                if len(tokens) != len(upos_tags):
                    raise ValueError(
                        "MasakhaPOS sample length mismatch: "
                        f"tokens={len(tokens)} vs upos={len(upos_tags)}"
                    )
                tuple_list_str = (
                    "["
                    + ", ".join(
                        f"({repr(tok)}, {repr(tag)})"
                        for tok, tag in zip(tokens, upos_tags, strict=False)
                    )
                    + "]"
                )
                user_prompt = _render_prompt(tokens, t_id)
                out: dict[str, Any] = {
                    "messages": [
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": tuple_list_str},
                    ],
                    "template_id": t_id,
                }
                lang_val = ex.get("lang") if isinstance(ex, dict) else None
                if lang_val:
                    out["lang"] = str(lang_val)
                elif ds_cfg.subset:
                    out["lang"] = ds_cfg.subset
                return out

            processed_ds = raw_ds.map(
                _cycle_pos,
                batched=False,
                with_indices=True,
                remove_columns=raw_ds.column_names,
                desc="Cycling POS templates",
            )

        else:
            raise ValueError(f"Unsupported task type for CYCLE: {ds_cfg.task}")

        return processed_ds

    if ds_cfg.task == FinetuneTaskType.INSTRUCTION:

        def to_messages(ex):
            user_text, assistant_text = _extract_instruction_pair(ex)
            return {
                "messages": [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": assistant_text},
                ]
            }

        map_function, desc = to_messages, "Format instruction data"

    elif ds_cfg.task == FinetuneTaskType.NAMED_ENTITY_RECOGNITION:
        template_spec = tmpl.get(ds_cfg.templates[0].id)
        tag_map = template_spec.ner_tags
        if not tag_map:
            raise ValueError(
                f"Template '{template_spec.id}' is for an NER task "
                "but is missing the 'ner_tags' list."
            )

        def to_messages(ex):
            text_input = " ".join(ex["tokens"])
            user_prompt = _safe_format_prompt(
                template_spec.prompt, {"text": text_input}
            )
            reconstructed_entities = _reconstruct_entities_from_iob(
                ex["tokens"], ex["ner_tags"], tag_map
            )
            assistant_response = " $$ ".join(reconstructed_entities)
            return {
                "messages": [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_response},
                ]
            }

        map_function, desc = to_messages, "Format NER data"

    elif ds_cfg.task == FinetuneTaskType.CLASSIFICATION:
        template_spec = tmpl.get(ds_cfg.templates[0].id)
        label_column = getattr(ds_cfg, "label_column", "label")
        if not template_spec.label_mapping:
            raise ValueError(
                "Template '" + template_spec.id + "' is for a classification task "
                "but is missing 'label_mapping'."
            )
        numeric_keys = isinstance(next(iter(template_spec.label_mapping.keys())), int)

        def to_messages(ex):
            user_prompt = template_spec.prompt.format(**ex)
            raw_label = ex[label_column]
            if numeric_keys:
                if isinstance(raw_label, str):
                    try:
                        key_to_use = int(raw_label)
                    except ValueError as err:
                        str_to_int_key_map = {
                            str(v).lower(): k
                            for k, v in template_spec.label_mapping.items()
                        }
                        key_to_use = str_to_int_key_map.get(raw_label.lower())
                        if key_to_use is None:
                            raise ValueError("Cannot map label to integer key") from err
                else:
                    key_to_use = int(raw_label)

                if (
                    key_to_use not in template_spec.label_mapping
                    and (key_to_use - 1) in template_spec.label_mapping
                ):
                    key_to_use -= 1
            else:
                key_to_use = str(raw_label)

            assistant_response = template_spec.label_mapping[key_to_use]
            return {
                "messages": [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_response},
                ]
            }

        map_function = to_messages
        desc = "Formatting dataset into 'messages' format"

    # TODO: extend support to other pos tasks if we go beyond just masakhanepos
    elif ds_cfg.task == FinetuneTaskType.POS_TAGGING:
        upos_class_names = None
        try:
            upos_feat = raw_ds.features.get("upos")
            if hasattr(upos_feat, "feature") and hasattr(upos_feat.feature, "names"):
                upos_class_names = list(upos_feat.feature.names)
        except Exception:
            upos_class_names = None

        if ds_cfg.templates and len(ds_cfg.templates) > 0:
            template_spec = tmpl.get(ds_cfg.templates[0].id)

            def render_prompt(tokens_list):
                tokens_repr = "[" + ", ".join(repr(t) for t in tokens_list) + "]"
                return _safe_format_prompt(
                    template_spec.prompt, {"tokens": tokens_repr}
                )

        else:

            def render_prompt(tokens_list):
                tokens_repr = "[" + ", ".join(repr(t) for t in tokens_list) + "]"
                return (
                    "Please provide UPOS tags for each token as a list of (token, TAG) "
                    "tuples.\nSentence: "
                    f"{tokens_repr}\nOutput: "
                )

        def to_messages_format_pos(
            ex: dict[str, Any],
        ) -> dict[str, list[dict[str, str]]]:
            tokens: list[str] = ex["tokens"]
            raw_upos = ex["upos"]

            if raw_upos and isinstance(raw_upos[0], int) and upos_class_names:
                upos_tags = [upos_class_names[i] for i in raw_upos]
            else:
                upos_tags = [str(t) for t in raw_upos]

            if len(tokens) != len(upos_tags):
                raise ValueError(
                    "MasakhaPOS sample length mismatch: "
                    f"tokens={len(tokens)} vs upos={len(upos_tags)}"
                )

            tuple_list_str = (
                "["
                + ", ".join(
                    f"({repr(tok)}, {repr(tag)})"
                    for tok, tag in zip(tokens, upos_tags, strict=False)
                )
                + "]"
            )

            user_prompt = render_prompt(tokens)
            return {
                "messages": [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": tuple_list_str},
                ]
            }

        map_function = to_messages_format_pos
        desc = "Formatting dataset for POS tagging (MasakhaPOS)"

    else:
        raise ValueError(f"Unsupported task type: {ds_cfg.task}")

    processed_ds = raw_ds.map(
        map_function,
        batched=False,
        remove_columns=raw_ds.column_names,
        desc=desc,
    )
    if "lang" not in processed_ds.column_names:
        if ds_cfg.subset:
            processed_ds = processed_ds.add_column(
                "lang", [ds_cfg.subset] * len(processed_ds)
            )
        elif "language_code" in raw_ds.column_names:
            processed_ds = processed_ds.add_column(
                "lang", [v for v in raw_ds["language_code"]]
            )
        elif "language" in raw_ds.column_names:
            processed_ds = processed_ds.add_column(
                "lang", [v for v in raw_ds["language"]]
            )
        elif "lang" in raw_ds.column_names:
            processed_ds = processed_ds.add_column("lang", [v for v in raw_ds["lang"]])

    return processed_ds
