"""Write configs/<arch>_<schedule>.json for the matched run (4 architectures x {cosine, wsd})."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "configs"
SPECIAL = {"bos_token_id": 0, "eos_token_id": 1, "pad_token_id": 2}  # tokenizer [BOS]/[EOS]/[PAD]; fixed IDs for all four

COMMON = {
    "seq_len": 2048,
    "global_batch_seqs": 192,  # 393,216 tokens/step (MzansiLM/Mamba-2 original global batch)
    "micro_batch": 8,  # hardware knob only (GA = 192 / (8 x GPUs)); --micro-batch may override
    "budget_epochs": 1,  # tokens per epoch = n_blocks x 2048 of the packed train.bin (set by the data, not here)
    "seed": 20260926,  # model init + epoch>=1 block permutations; document order is fixed in train.bin (pack.py seed)
    "optim": {"betas": [0.9, 0.95], "eps": 1e-8, "weight_decay": 0.1, "grad_clip": 1.0},
    "eval_every_tokens": 100_000_000,  # up to 1B tokens
    "eval_every_tokens_after_1b": 250_000_000,
    "weights_at_tokens": [100_000_000, 250_000_000, 500_000_000, 1_000_000_000, 2_000_000_000, 3_000_000_000],
    "resume_every_steps": 500,
    "eval_parquet": "${MP_VAL_PARQUET}",  # raw-text held-out split with a `lang` column (bench: uctnlp/mzansi-text validation)
    "tokenizer": "${MP_TOKENIZER}",
    "wandb_project": "sallm-matched-pretrain",
}
SCHEDULES = {
    # warmup 4% ~= MzansiLM's 2,000 / 48,400 steps (4.13%); floor 10% of peak for both shapes
    "cosine": {"type": "cosine", "warmup_frac": 0.04, "min_lr_ratio": 0.1},
    # Hagele et al. 2024: constant peak, (1-sqrt) cooldown over the final 20% of the budget
    "wsd": {"type": "wsd", "warmup_frac": 0.04, "min_lr_ratio": 0.1, "decay_frac": 0.2},
}
ARCHS = {
    "mzansilm": {
        "expected_params": 125_008_384,
        "attn_implementation": "sdpa",
        "peak_lr": 4e-4, "peak_lr_source": "W&B anri-lombard/sallm-llama/2mkkmx3d config learning_rate (src/conf/base/llama_125m.yaml)",
        "model_config": {"vocab_size": 65536, "hidden_size": 512, "intermediate_size": 1536, "num_hidden_layers": 30,
                         "num_attention_heads": 9, "num_key_value_heads": 3, "head_dim": 56, "hidden_act": "silu",
                         "max_position_embeddings": 2048, "rms_norm_eps": 1e-5, "rope_theta": 10000.0,
                         "attention_bias": False, "mlp_bias": False, "initializer_range": 0.02,
                         "tie_word_embeddings": True, "use_cache": False, **SPECIAL},
    },
    "mamba2": {
        "expected_params": 126_427_168,
        "peak_lr": 4e-4, "peak_lr_source": "W&B anri-lombard/sallm-mamba/7c3t63pc (Slurm 407979) config learning_rate (src/conf/base/mamba_125m.yaml)",
        # FLA Mamba2 (per-group gated RMSNorm, as trained); shape from the retained base config.json.
        # Init hyper-parameters are FLA defaults (initializer_range 0.02, rescale_prenorm_residual True), not the
        # HF-class defaults of the original run (0.1, False).
        "model_config": {"vocab_size": 65536, "hidden_size": 512, "state_size": 64, "num_hidden_layers": 27,
                         "expand": 4, "head_dim": 64, "n_groups": 4, "conv_kernel": 4, "chunk_size": 256,
                         "use_bias": False, "use_conv_bias": True, "hidden_act": "silu", "norm_eps": 1e-5,
                         "residual_in_fp32": True, "rmsnorm": True, "dt_min": 0.001, "dt_max": 0.1,
                         "dt_init_floor": 1e-4, "tie_word_embeddings": True, "use_cache": False,
                         "fuse_cross_entropy": False, **SPECIAL},
    },
    "xlstm": {
        "expected_params": 126_901_952,
        "peak_lr": 4e-4, "peak_lr_source": "HEX ~/masters/sallm/slurm-880318.out launch command --learning-rate 4e-4",
        # retained base config.json (h736_l12_h4_chunk64), native chunkwise kernels, fp32 master weights
        "model_config": {"vocab_size": 65536, "hidden_size": 736, "embedding_dim": 736, "num_hidden_layers": 12,
                         "num_blocks": 12, "num_heads": 4, "qk_dim_factor": 0.5, "v_dim_factor": 1.0,
                         "ffn_proj_factor": 2.667, "ffn_round_up_to_multiple_of": 64, "chunk_size": 64,
                         "gate_soft_cap": 15.0, "output_logit_soft_cap": 30.0, "use_bias": False,
                         "weight_mode": "single", "add_out_norm": True, "norm_eps": 1e-6, "eps": 1e-6,
                         "norm_reduction_force_float32": True, "mode": "train",
                         "chunkwise_kernel": "chunkwise--native_autograd", "sequence_kernel": "native_sequence__native",
                         "step_kernel": "native", "autocast_kernel_dtype": "bfloat16",
                         "inference_state_dtype": "float32", "max_inference_chunksize": 16384,
                         "return_last_states": True, "tie_word_embeddings": True, "use_cache": False, **SPECIAL},
    },
    "gdn": {
        "expected_params": 127_425_448,
        "peak_lr": 4e-4, "peak_lr_source": "src/conf/base/gated_deltanet_125m_pure.yaml learning_rate (run a10080-3epoch-20260802)",
        "model_config": {"vocab_size": 65536, "hidden_size": 512, "num_hidden_layers": 21, "intermediate_size": 1536,
                         "hidden_ratio": None, "num_heads": 4, "head_dim": 128, "expand_v": 2, "num_v_heads": None,
                         "attn": None, "attn_mode": "chunk", "use_gate": True, "use_short_conv": True, "conv_size": 4,
                         "allow_neg_eigval": False, "hidden_act": "swish", "max_position_embeddings": 2048,
                         "norm_eps": 1e-6, "initializer_range": 0.02, "fuse_norm": True, "fuse_swiglu": True,
                         "fuse_cross_entropy": False, "fuse_linear_cross_entropy": False, "use_l2warp": False,
                         "tie_word_embeddings": True, "use_cache": False, **SPECIAL},
    },
}

OUT.mkdir(exist_ok=True)
for arch, spec in ARCHS.items():
    for sname, sched in SCHEDULES.items():
        c = {"arch": arch, **COMMON, "schedule": sched}
        c["optim"] = {"peak_lr": spec["peak_lr"], **COMMON["optim"]}
        c.update({k: v for k, v in spec.items() if k != "peak_lr"})
        (OUT / f"{arch}_{sname}.json").write_text(json.dumps(c, indent=2) + "\n")
        print(OUT / f"{arch}_{sname}.json")
