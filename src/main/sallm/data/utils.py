from __future__ import annotations
from typing import Any, Dict, Callable, List
import random
from transformers import PreTrainedTokenizerBase
from sallm.config import FinetuneDatasetConfig, TemplateChoice
from sallm.templates import registry as tmpl


def make_example_mapper(
    ds_cfg: FinetuneDatasetConfig, tokenizer: PreTrainedTokenizerBase
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    templates = [tmpl.get(t.id) for t in ds_cfg.templates]
    weights = [t.weight for t in ds_cfg.templates]

    # if ds_cfg.template_choice == TemplateChoice.SINGLE:
    #     templates = [random.choices(templates, weights)[0]]

    numeric_keys = isinstance(next(iter(templates[0].label_mapping.keys())), int)

    def encode(ex: Dict[str, Any], template) -> Dict[str, Any]:
        prompt_kwargs = {col: ex[col] for col in ds_cfg.text_columns}
        prompt_text = template.prompt.format(**prompt_kwargs)
        raw_label = ex[ds_cfg.label_column]
        if numeric_keys:
            label_text = template.label_mapping[int(raw_label)]
        else:
            label_text = template.label_mapping[raw_label]
        full_text = f"{prompt_text} {label_text}{tokenizer.eos_token}"
        enc = tokenizer(
            full_text,
            truncation=True,
            max_length=ds_cfg.max_seq_length,
            padding="max_length",
        )
        labels = enc["input_ids"].copy()
        enc["labels"] = [
            (tok if tok != tokenizer.pad_token_id else -100) for tok in labels
        ]
        return enc

    if ds_cfg.template_choice == TemplateChoice.ALL:

        def mapper(ex):
            outs: List[Dict[str, Any]] = []
            for t in templates:
                outs.append(encode(ex, t))
            return outs

        return mapper

    if ds_cfg.template_choice == TemplateChoice.CYCLE:

        def mapper(ex, _counter=[0]):
            t = templates[_counter[0] % len(templates)]
            _counter[0] += 1
            return encode(ex, t)

        return mapper

    def mapper(ex):
        t = random.choices(templates, weights)[0]
        return encode(ex, t)

    return mapper
