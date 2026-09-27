#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import re
from itertools import cycle
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, Qwen3NextConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiny BabyLM GDN training smoke.")
    parser.add_argument(
        "--output-dir", default="/scratch/alombard/babylm-gdn/outputs/gdn-tiny-smoke"
    )
    parser.add_argument(
        "--dataset-name", default="BabyLM-community/BabyLM-2026-Strict-Small"
    )
    parser.add_argument("--tokenizer-name", default="gpt2")
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--seq-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-hidden-layers", type=int, default=2)
    parser.add_argument("--num-attention-heads", type=int, default=4)
    parser.add_argument("--full-attention-every", type=int, default=0)
    parser.add_argument("--empty-state-emphasis-repeat", type=int, default=0)
    parser.add_argument(
        "--empty-state-emphasis-regex",
        default=r"\b(nothing|empty|emptied|none|nobody|no one|without)\b",
    )
    parser.add_argument("--synthetic-entity-examples", type=int, default=0)
    parser.add_argument("--synthetic-entity-empty-rate", type=float, default=0.5)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--packed", action="store_true")
    return parser.parse_args()


def build_config(
    args: argparse.Namespace, vocab_size: int, dtype: torch.dtype
) -> Qwen3NextConfig:
    from transformers import Qwen3NextConfig

    head_dim = args.hidden_size // args.num_attention_heads
    layer_types = ["linear_attention"] * args.num_hidden_layers
    if args.full_attention_every > 0:
        layer_types = [
            "full_attention"
            if (idx + 1) % args.full_attention_every == 0
            else "linear_attention"
            for idx in range(args.num_hidden_layers)
        ]

    # ponytail: only size knobs needed for the first pilot; no sweep system yet.
    config = Qwen3NextConfig(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.hidden_size * 3,
        moe_intermediate_size=max(64, args.hidden_size // 4),
        shared_expert_intermediate_size=max(64, args.hidden_size // 4),
        num_experts=1,
        num_experts_per_tok=1,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=1,
        head_dim=head_dim,
        linear_key_head_dim=head_dim,
        linear_value_head_dim=head_dim,
        linear_num_key_heads=args.num_attention_heads,
        linear_num_value_heads=args.num_attention_heads,
        linear_conv_kernel_dim=4,
        max_position_embeddings=args.seq_length,
        layer_types=layer_types,
        tie_word_embeddings=True,
        use_cache=False,
    )
    config.dtype = dtype
    return config


class PackedTextDataset:
    def __init__(self, input_ids: torch.Tensor, metadata: dict | None = None):
        import torch

        self.input_ids = input_ids
        self.attention_mask = torch.ones_like(input_ids)
        self.metadata = metadata or {}

    def __len__(self) -> int:
        return int(self.input_ids.size(0))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        input_ids = self.input_ids[index]
        return {
            "input_ids": input_ids,
            "attention_mask": self.attention_mask[index],
            "labels": input_ids.clone(),
        }


def text_repeat_count(
    text: str, regex: re.Pattern[str] | None, extra_repeats: int
) -> int:
    return 1 + extra_repeats if regex is not None and regex.search(text) else 1


OBJECTS = [
    "the apple",
    "the ball",
    "the bell",
    "the book",
    "the bottle",
    "the camera",
    "the coat",
    "the cup",
    "the hat",
    "the key",
    "the knife",
    "the letter",
    "the mirror",
    "the paper",
    "the shell",
    "the string",
]


def synthetic_entity_texts(count: int, empty_rate: float, seed: int) -> list[str]:
    if count <= 0:
        return []
    if not 0.0 <= empty_rate <= 1.0:
        raise ValueError("--synthetic-entity-empty-rate must be between 0 and 1.")

    rng = random.Random(seed)
    empty_count = round(count * empty_rate)
    texts = []
    for idx in range(count):
        box = rng.randint(1, 6)
        other_box = box % 6 + 1
        obj = OBJECTS[idx % len(OBJECTS)]
        distractor = OBJECTS[(idx * 7 + 3) % len(OBJECTS)]
        if idx < empty_count:
            texts.append(
                f"Box {box} contains {obj}. Box {box} is emptied. "
                f"Box {box} contains nothing."
            )
        else:
            texts.append(
                f"Box {box} contains {obj}. Box {other_box} contains {distractor}. "
                f"Box {box} contains {obj}."
            )
    rng.shuffle(texts)
    return texts


def build_dataset(args: argparse.Namespace, tokenizer: AutoTokenizer) -> Dataset:
    import torch
    from datasets import load_dataset

    split = "train" if args.train_samples <= 0 else f"train[:{args.train_samples}]"
    dataset = load_dataset(args.dataset_name, split=split)

    if args.packed:
        eos_id = tokenizer.eos_token_id
        regex = (
            re.compile(args.empty_state_emphasis_regex, re.IGNORECASE)
            if args.empty_state_emphasis_repeat > 0
            else None
        )
        token_ids: list[int] = []
        matched_texts = 0
        emitted_texts = 0
        synthetic_texts = synthetic_entity_texts(
            args.synthetic_entity_examples,
            args.synthetic_entity_empty_rate,
            args.seed,
        )
        for text in dataset["text"]:
            if isinstance(text, str) and text.strip():
                repeats = text_repeat_count(
                    text, regex, args.empty_state_emphasis_repeat
                )
                matched_texts += int(repeats > 1)
                emitted_texts += repeats
                encoded = tokenizer.encode(text, add_special_tokens=False)
                for _ in range(repeats):
                    token_ids.extend(encoded)
                    token_ids.append(eos_id)
        for text in synthetic_texts:
            if isinstance(text, str) and text.strip():
                repeats = text_repeat_count(
                    text, regex, args.empty_state_emphasis_repeat
                )
                matched_texts += int(repeats > 1)
                emitted_texts += repeats
                encoded = tokenizer.encode(text, add_special_tokens=False)
                for _ in range(repeats):
                    token_ids.extend(encoded)
                    token_ids.append(eos_id)
        block_count = len(token_ids) // args.seq_length
        if block_count == 0:
            raise ValueError(
                "Packed dataset has zero full blocks; reduce --seq-length or add data."
            )
        tensor = torch.tensor(
            token_ids[: block_count * args.seq_length], dtype=torch.long
        )
        metadata = {
            "empty_state_emphasis_repeat": args.empty_state_emphasis_repeat,
            "empty_state_emphasis_regex": args.empty_state_emphasis_regex,
            "empty_state_matched_texts": matched_texts,
            "emitted_texts": emitted_texts,
            "synthetic_entity_examples": len(synthetic_texts),
            "synthetic_entity_empty_rate": args.synthetic_entity_empty_rate,
        }
        return PackedTextDataset(tensor.view(block_count, args.seq_length), metadata)

    def tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        texts = [
            text if isinstance(text, str) and text.strip() else tokenizer.eos_token
            for text in batch["text"]
        ]
        encoded = tokenizer(
            texts,
            max_length=args.seq_length,
            padding="max_length",
            truncation=True,
        )
        encoded["labels"] = [
            [token if mask else -100 for token, mask in zip(ids, masks, strict=False)]
            for ids, masks in zip(
                encoded["input_ids"], encoded["attention_mask"], strict=False
            )
        ]
        return encoded

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    tokenized.set_format(type="torch")
    return tokenized


def main() -> None:
    args = parse_args()
    if args.self_check:
        regex = re.compile(args.empty_state_emphasis_regex, re.IGNORECASE)
        assert text_repeat_count("The shelf was empty.", regex, 2) == 3
        assert text_repeat_count("The shelf was full.", regex, 2) == 1
        synthetic = synthetic_entity_texts(10, 0.5, 13)
        assert len(synthetic) == 10
        assert sum("contains nothing." in text for text in synthetic) == 5
        print("SELF_CHECK_OK")
        return

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3NextForCausalLM

    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = build_dataset(args, tokenizer)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    config = build_config(args, len(tokenizer), dtype)
    model = Qwen3NextForCausalLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    losses: list[float] = []
    model.train()
    for step, batch in zip(range(1, args.max_steps + 1), cycle(loader)):
        batch = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=dtype, enabled=device.type == "cuda"
        ):
            loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        if step == 1 or step % 10 == 0 or step == args.max_steps:
            print(f"step={step} loss={loss_value:.4f}", flush=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    reloaded = AutoModelForCausalLM.from_pretrained(
        output_dir, trust_remote_code=True
    ).to(device)
    reloaded.eval()
    with torch.no_grad():
        probe = next(iter(loader))
        probe = {key: value[:1].to(device) for key, value in probe.items()}
        reload_loss = float(reloaded(**probe).loss.detach().cpu())

    metrics = {
        "output_dir": str(output_dir),
        "steps": args.max_steps,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "reload_loss": reload_loss,
        "device": str(device),
        "dtype": str(dtype),
        "packed": args.packed,
        "train_blocks_or_rows": len(train_dataset),
        "dataset_metadata": getattr(train_dataset, "metadata", {}),
    }
    (output_dir / "smoke_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
