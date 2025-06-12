import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from .core import GPTModel, rms_norm
from ..config import AppConfig


class GPTForTokenClassification(nn.Module):
    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self.num_labels = len(config.data.ner_tags)

        self.transformer = GPTModel(config.model)
        self.classifier = nn.Linear(config.model.model_dim, self.num_labels)

        self.load_pretrained_weights()

    def load_pretrained_weights(self):
        print("--- Loading Pre-trained Weights ---")
        path = self.config.model.pretrained_model_path
        if not path:
            print("No pretrained model path provided. Skipping weight loading.")
            return
        try:
            pretrained_checkpoint = torch.load(path, map_location="cpu")
            pretrained_state_dict = pretrained_checkpoint["model"]
        except Exception as e:
            print(f"Error loading checkpoint file: {e}")
            return

        unwrapped_state_dict = {}
        for k, v in pretrained_state_dict.items():
            if k.startswith("_orig_mod."):
                unwrapped_state_dict[k[len("_orig_mod.") :]] = v
            else:
                unwrapped_state_dict[k] = v

        transformer_state_dict = self.transformer.state_dict()
        filtered_state_dict = {
            k: v for k, v in unwrapped_state_dict.items() if k in transformer_state_dict
        }

        missing_keys, unexpected_keys = self.transformer.load_state_dict(
            filtered_state_dict, strict=False
        )

        print("Weight loading summary:")
        if unexpected_keys:
            print(
                f"  - Ignored {len(unexpected_keys)} unexpected keys from checkpoint."
            )
        if missing_keys:
            print(
                f"  - WARNING: {len(missing_keys)} keys were missing from checkpoint and not loaded: {missing_keys}"
            )
        if not missing_keys:
            print("  - Successfully loaded all core transformer weights.")

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        sequence_output = self.transformer(input_ids=input_ids)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {"loss": loss, "logits": logits}
