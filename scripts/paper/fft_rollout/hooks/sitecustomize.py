"""Loaded at interpreter start when this directory is on PYTHONPATH (rollout.py puts it there for Mamba-2 jobs).

SALLM_FLA_MAMBA2=1 registers FLA's Mamba-2 classes for model_type "mamba2" (exist_ok=True overrides transformers'),
so every AutoModelForCausalLM load, in training, lm-eval and the sallm harness, gets FLA's Mamba2ForCausalLM.
transformers' Mamba-2 normalises the gated RMSNorm over all channels in its eval path instead of per group.
"""
import os

if os.environ.get("SALLM_FLA_MAMBA2") == "1":
    import fla.models.mamba2  # noqa: F401
