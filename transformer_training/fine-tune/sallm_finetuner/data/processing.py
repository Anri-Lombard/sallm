from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import DataCollatorForTokenClassification, PreTrainedTokenizerFast
from typing import List, Dict, Any


def load_and_combine_datasets(
    dataset_name: str, language_subsets: List[str]
) -> DatasetDict:
    all_splits = {"train": [], "validation": [], "test": []}
    for lang in language_subsets:
        dataset = load_dataset(dataset_name, lang, trust_remote_code=True)
        for split_name in all_splits.keys():
            if split_name in dataset:
                all_splits[split_name].append(dataset[split_name])

    return DatasetDict(
        {
            split: concatenate_datasets(dsets)
            for split, dsets in all_splits.items()
            if dsets
        }
    )


def create_tokenization_function(tokenizer, label_to_id, max_seq_length, pad_token_id):
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            max_length=max_seq_length,
        )
        labels = []
        for i, ner_tags_for_example in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(pad_token_id)
                elif word_idx != previous_word_idx:
                    label_ids.append(
                        label_to_id.get(ner_tags_for_example[word_idx], pad_token_id)
                    )
                else:
                    label_ids.append(pad_token_id)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    return tokenize_and_align_labels


def process_dataset(raw_datasets, tokenization_function):
    column_names = (
        raw_datasets["train"].column_names
        if isinstance(raw_datasets, DatasetDict)
        else list(raw_datasets.features)
    )
    return raw_datasets.map(
        tokenization_function,
        batched=True,
        remove_columns=column_names,
        desc="Tokenizing and aligning labels",
    )


def get_per_language_loaders(
    split: str,
    dataset_name: str,
    language_subsets: List[str],
    tokenizer: PreTrainedTokenizerFast,
    tokenization_function: callable,
    batch_size: int,
) -> Dict[str, DataLoader]:
    loaders = {}
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer, padding="longest"
    )

    for lang in language_subsets:
        raw_dataset = load_dataset(dataset_name, lang, trust_remote_code=True)
        if split not in raw_dataset:
            print(f"Warning: No '{split}' split found for language {lang}.")
            continue

        processed_dataset = process_dataset(raw_dataset, tokenization_function)
        loaders[lang] = DataLoader(
            processed_dataset[split],
            collate_fn=data_collator,
            batch_size=batch_size,
        )
    return loaders
