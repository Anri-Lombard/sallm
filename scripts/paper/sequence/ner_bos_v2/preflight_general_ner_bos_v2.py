#!/usr/bin/env python3
import hashlib
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

from run_general_sequence_eval_20260917_v2 import build_ner_tasks, chosen_prompts


protocol = json.loads(Path(sys.argv[1]).read_text())
selection = json.loads(Path(sys.argv[2]).read_text())
_, tasks, _ = build_ner_tasks(
    Path(protocol["source_snapshot"]["path"]),
    "test",
    chosen_prompts("ner", "test", selection),
)

prompts = []
for task in tasks:
    task.build_all_requests(
        limit=1,
        rank=0,
        world_size=1,
        cache_requests=False,
        rewrite_requests_cache=False,
        system_instruction=None,
        apply_chat_template=False,
        fewshot_as_multiturn=False,
        chat_template=None,
        tokenizer_name="fixed-bos-v2",
    )
    prompt = task.instances[0].arguments[0]
    assert prompt.startswith("[BOS]        <|user|>\n        ")
    assert prompt.endswith("[EOS]<|assistant|>")
    prompts.append(prompt)

token_hashes = {}
for architecture, binding in protocol["models"].items():
    tokenizer = AutoTokenizer.from_pretrained(
        binding["adapter_path"], trust_remote_code=True, local_files_only=True
    )
    assert tokenizer.convert_tokens_to_ids("[BOS]") == 0
    assert tokenizer.convert_tokens_to_ids("[EOS]") == 1
    encoded = [tokenizer(prompt, add_special_tokens=False)["input_ids"] for prompt in prompts]
    token_hashes[architecture] = hashlib.sha256(
        json.dumps(encoded, separators=(",", ":")).encode()
    ).hexdigest()

assert len(set(token_hashes.values())) == 1, token_hashes
print("GENERAL_NER_BOS_V2_PREFLIGHT_OK", token_hashes)
