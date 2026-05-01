from pathlib import Path

import sallm.config as legacy_config
import sallm.configs as domain_config
from omegaconf import OmegaConf

CONF_ROOT = Path(__file__).resolve().parents[1] / "src" / "conf"


def test_legacy_config_reexports_domain_config_types() -> None:
    assert legacy_config.ExperimentConfig is domain_config.ExperimentConfig
    assert legacy_config.ModelConfig is domain_config.ModelConfig
    assert legacy_config.FinetuneDatasetConfig is domain_config.FinetuneDatasetConfig
    assert legacy_config.ModelEvalConfig is domain_config.ModelEvalConfig
    assert legacy_config.DecodingConfig is domain_config.DecodingConfig
    assert legacy_config.RunMode is domain_config.RunMode


def test_to_resolved_dict_available_from_both_import_paths() -> None:
    cfg = OmegaConf.create({"answer": 42})

    assert legacy_config.to_resolved_dict(cfg) == {"answer": 42}
    assert domain_config.to_resolved_dict(cfg) == {"answer": 42}


def test_experiment_schema_merges_representative_finetune_config() -> None:
    schema = OmegaConf.structured(domain_config.ExperimentConfig)
    raw_cfg = OmegaConf.load(CONF_ROOT / "finetune" / "llama_t2x_xho.yaml")

    merged = OmegaConf.merge(schema, raw_cfg)

    assert merged.mode == domain_config.RunMode.FINETUNE
    assert merged.model.architecture == "llama"
    assert merged.dataset.hf_name == "github:francois-meyer/t2x"
    assert merged.generation_decoding.strategy == "beam"


def test_experiment_schema_merges_representative_eval_config() -> None:
    schema = OmegaConf.structured(domain_config.ExperimentConfig)
    raw_cfg = OmegaConf.load(CONF_ROOT / "eval" / "run_llama_t2x_xho.yaml")

    merged = OmegaConf.merge(schema, raw_cfg)

    assert merged.mode == domain_config.RunMode.EVALUATE
    assert merged.eval_model.adapter == "hf"
    assert merged.evaluation.generation_tasks[0].id == "t2x_xho"
    assert merged.evaluation.generation_tasks[0].decoding.strategy == "beam"
