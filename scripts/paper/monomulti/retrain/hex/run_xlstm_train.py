#!/usr/bin/env python3
"""Run sallm.main with the xLSTM inference-mode patch for in-training generation metrics.

The patch is copied from historical_selected_adapter_recovery_20260917_v3/
run_historical_selected_train_validation_only_20260915.py (install_loader), which
produced the xLSTM Mono NER adapters. Training-mode xLSTM cannot generate on
sequences whose length is not a multiple of the chunk size.
"""

from __future__ import annotations

import runpy
from typing import Any


def install_generation_patch() -> None:
    import sallm.evaluation.generation_metrics as generation_metrics

    original_evaluate = generation_metrics.GenerationEvaluator.evaluate

    def evaluate_with_xlstm_inference(
        self: Any, model: Any, *args: Any, **kwargs: Any
    ) -> Any:
        config = getattr(model, "config", None)
        if getattr(config, "model_type", None) != "xlstm":
            return original_evaluate(self, model, *args, **kwargs)
        config_type = type(config)
        previous_qk_head_dim = config_type.qk_head_dim
        previous_v_head_dim = config_type.v_head_dim
        previous_mode = config.mode
        previous_use_cache = config.use_cache
        config_type.qk_head_dim = property(
            lambda current: int(current.hidden_size * current.qk_dim_factor)
            // current.num_heads
        )
        config_type.v_head_dim = property(
            lambda current: int(current.hidden_size * current.v_dim_factor)
            // current.num_heads
        )
        config.mode = "inference"
        config.use_cache = True
        try:
            return original_evaluate(self, model, *args, **kwargs)
        finally:
            config_type.qk_head_dim = previous_qk_head_dim
            config_type.v_head_dim = previous_v_head_dim
            config.mode = previous_mode
            config.use_cache = previous_use_cache

    generation_metrics.GenerationEvaluator.evaluate = evaluate_with_xlstm_inference


if __name__ == "__main__":
    install_generation_patch()
    runpy.run_module("sallm.main", run_name="__main__")
