#!/usr/bin/env python3
"""
Generate xLSTM finetune configs from existing Mamba configs.

This script reads Mamba monolingual finetune configs and transforms them
into corresponding xLSTM configs with appropriate architectural changes.
"""

from pathlib import Path

import yaml

# Task to language mapping
TASKS = {
    "ner": ["xho", "zul", "tsn"],
    "pos": ["xho", "zul", "tsn"],
    "news": ["eng", "xho"],
    "sib": ["afr", "eng", "nso", "sot", "xho", "zul"],
}


def map_target_modules(mamba_modules: list[str]) -> list[str]:
    """
    Map Mamba target modules to xLSTM target modules.

    Mamba uses: ["in_proj", "x_proj"] or ["in_proj", "x_proj", "embeddings"]
    xLSTM uses: ["q", "k", "v", "out_proj"] or ["q", "k", "v", "out_proj", "embeddings"]
    """
    if len(mamba_modules) == 2:
        return ["q", "k", "v", "out_proj"]
    elif len(mamba_modules) == 3:
        return ["q", "k", "v", "out_proj", "embeddings"]
    else:
        raise ValueError(f"Unexpected number of target modules: {len(mamba_modules)}")


def transform_config(mamba_config: dict, task: str, lang: str) -> dict:
    """Transform a Mamba config into an xLSTM config."""

    # Deep copy to avoid modifying original
    xlstm_config = yaml.safe_load(yaml.dump(mamba_config))

    # 1. Update model architecture and checkpoint
    xlstm_config["model"]["architecture"] = "xlstm"
    xlstm_config["model"]["init_checkpoint"] = (
        "${oc.env:SCRATCH}/masters/sallm/checkpoints/sallm-xlstm-125m/final_model"
    )

    # 2. Update PEFT target modules
    if "peft" in xlstm_config and "kwargs" in xlstm_config["peft"]:
        if "target_modules" in xlstm_config["peft"]["kwargs"]:
            mamba_modules = xlstm_config["peft"]["kwargs"]["target_modules"]
            xlstm_config["peft"]["kwargs"]["target_modules"] = map_target_modules(
                mamba_modules
            )

    # 3. Update wandb name
    if "wandb" in xlstm_config and "name" in xlstm_config["wandb"]:
        wandb_name = xlstm_config["wandb"]["name"]
        # Replace "mamba-125m" with "xlstm-125m"
        xlstm_config["wandb"]["name"] = wandb_name.replace("mamba-125m", "xlstm-125m")

    # 4. Update training directories
    if "training" in xlstm_config:
        # output_dir
        if "output_dir" in xlstm_config["training"]:
            output_dir = xlstm_config["training"]["output_dir"]
            xlstm_config["training"]["output_dir"] = output_dir.replace(
                "ft_mamba_125m", "ft_xlstm_125m"
            )

        # logging_dir
        if "logging_dir" in xlstm_config["training"]:
            logging_dir = xlstm_config["training"]["logging_dir"]
            xlstm_config["training"]["logging_dir"] = logging_dir.replace(
                "ft_mamba_125m", "ft_xlstm_125m"
            )

        # run_name
        if "run_name" in xlstm_config["training"]:
            run_name = xlstm_config["training"]["run_name"]
            xlstm_config["training"]["run_name"] = run_name.replace(
                "mamba-125m", "xlstm-125m"
            )

    # 5. Update HuggingFace base_model_id (if hub section exists)
    if "hub" in xlstm_config:
        # Note: anrilombard/sallm-xlstm-125m doesn't exist yet on HF
        # Setting it anyway for when it gets created/pushed
        xlstm_config["hub"]["base_model_id"] = "anrilombard/sallm-xlstm-125m"

    return xlstm_config


def main():
    """Generate all xLSTM configs from Mamba templates."""

    repo_root = Path(__file__).parent.parent
    conf_dir = repo_root / "src" / "conf" / "finetune"

    if not conf_dir.exists():
        print(f"❌ Config directory not found: {conf_dir}")
        return 1

    configs_created = []
    configs_failed = []

    for task, languages in TASKS.items():
        for lang in languages:
            mamba_config_name = f"mamba_{task}_{lang}.yaml"
            xlstm_config_name = f"xlstm_{task}_{lang}.yaml"

            mamba_config_path = conf_dir / mamba_config_name
            xlstm_config_path = conf_dir / xlstm_config_name

            # Check if Mamba config exists
            if not mamba_config_path.exists():
                msg = f"⚠️  Skipping {xlstm_config_name}: template not found"
                print(msg)
                configs_failed.append(xlstm_config_name)
                continue

            # Read Mamba config
            try:
                with open(mamba_config_path) as f:
                    mamba_config = yaml.safe_load(f)
            except Exception as e:
                print(f"❌ Error reading {mamba_config_name}: {e}")
                configs_failed.append(xlstm_config_name)
                continue

            # Transform to xLSTM config
            try:
                xlstm_config = transform_config(mamba_config, task, lang)
            except Exception as e:
                print(f"❌ Error transforming {mamba_config_name}: {e}")
                configs_failed.append(xlstm_config_name)
                continue

            # Write xLSTM config
            try:
                with open(xlstm_config_path, "w") as f:
                    yaml.dump(
                        xlstm_config, f, default_flow_style=False, sort_keys=False
                    )
                print(f"✓ Created {xlstm_config_name}")
                configs_created.append(xlstm_config_name)
            except Exception as e:
                print(f"❌ Error writing {xlstm_config_name}: {e}")
                configs_failed.append(xlstm_config_name)
                continue

    # Summary
    print(f"\n{'=' * 60}")
    print(f"✓ Successfully created {len(configs_created)} configs")
    if configs_failed:
        print(f"❌ Failed to create {len(configs_failed)} configs:")
        for config in configs_failed:
            print(f"   - {config}")
    print(f"{'=' * 60}\n")

    return 0 if not configs_failed else 1


if __name__ == "__main__":
    exit(main())
