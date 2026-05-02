from pathlib import Path

import sallm.config as legacy_config
import sallm.configs as domain_config
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from sallm.hpo.trial import load_base_config

CONF_ROOT = Path(__file__).resolve().parents[1] / "src" / "conf"


def compose_config_target(config_target: str):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_ROOT), version_base=None):
        cfg = compose(config_name=config_target)

    keys_in_cfg = list(cfg.keys())
    if (
        len(keys_in_cfg) == 1
        and keys_in_cfg[0] not in domain_config.ExperimentConfig.__dataclass_fields__
    ):
        return cfg[keys_in_cfg[0]]
    return cfg


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
    raw_cfg = compose_config_target("finetune/llama_t2x_xho")

    merged = OmegaConf.merge(schema, raw_cfg)

    assert merged.mode == domain_config.RunMode.FINETUNE
    assert merged.model.architecture == "llama"
    assert merged.dataset.hf_name == "github:francois-meyer/t2x"
    assert merged.generation_decoding.strategy == "beam"


def test_representative_sib_finetune_configs_share_dataset_defaults() -> None:
    schema = OmegaConf.structured(domain_config.ExperimentConfig)
    expected_configs = {
        "finetune/llama_sib_xho": {
            "architecture": "llama",
            "subset": "xho_Latn",
            "languages": None,
            "max_seq_length": 2048,
        },
        "finetune/mamba_sib_all": {
            "architecture": "mamba2",
            "subset": None,
            "languages": [
                "afr_Latn",
                "eng_Latn",
                "nso_Latn",
                "sot_Latn",
                "xho_Latn",
                "zul_Latn",
            ],
            "max_seq_length": 1024,
        },
        "finetune/xlstm_sib_xho": {
            "architecture": "xlstm",
            "subset": "xho_Latn",
            "languages": None,
            "max_seq_length": 1024,
        },
    }

    for config_target, expected in expected_configs.items():
        raw_cfg = compose_config_target(config_target)
        merged = OmegaConf.merge(schema, raw_cfg)

        assert merged.mode == domain_config.RunMode.FINETUNE
        assert merged.model.architecture == expected["architecture"]
        assert merged.dataset.hf_name == "Davlan/sib200"
        assert merged.dataset.task == domain_config.FinetuneTaskType.CLASSIFICATION
        assert merged.dataset.splits == {"train": "train", "val": "validation"}
        assert [template.id for template in merged.dataset.templates] == [
            "sib_topic_classification/lm_eval_p1",
            "sib_topic_classification/lm_eval_p2",
            "sib_topic_classification/lm_eval_p3",
            "sib_topic_classification/lm_eval_p4",
            "sib_topic_classification/lm_eval_p5",
        ]
        assert merged.dataset.template_choice == domain_config.TemplateChoice.CYCLE
        assert merged.dataset.label_column == "category"
        assert merged.dataset.packing is False
        assert merged.dataset.assistant_only_loss is True
        assert merged.dataset.subset == expected["subset"]
        assert merged.dataset.languages == expected["languages"]
        assert merged.dataset.max_seq_length == expected["max_seq_length"]


def test_representative_injongointent_finetune_configs_share_dataset_defaults() -> None:
    schema = OmegaConf.structured(domain_config.ExperimentConfig)
    expected_configs = {
        "finetune/llama_injongointent_xho": {
            "architecture": "llama",
            "subset": "xho",
            "languages": None,
            "max_seq_length": 2048,
        },
        "finetune/mamba_injongointent_xho": {
            "architecture": "mamba2",
            "subset": "xho",
            "languages": None,
            "max_seq_length": 1024,
        },
        "finetune/xlstm_injongointent_all": {
            "architecture": "xlstm",
            "subset": None,
            "languages": ["eng", "sot", "xho", "zul"],
            "max_seq_length": 2048,
        },
    }

    for config_target, expected in expected_configs.items():
        raw_cfg = compose_config_target(config_target)
        merged = OmegaConf.merge(schema, raw_cfg)

        assert merged.mode == domain_config.RunMode.FINETUNE
        assert merged.model.architecture == expected["architecture"]
        assert merged.dataset.hf_name == "masakhane/InjongoIntent"
        assert merged.dataset.task == domain_config.FinetuneTaskType.CLASSIFICATION
        assert merged.dataset.splits == {"train": "train", "val": "validation"}
        assert [template.id for template in merged.dataset.templates] == [
            "injongointent_intent_classification/lm_eval_p1",
            "injongointent_intent_classification/lm_eval_p2",
            "injongointent_intent_classification/lm_eval_p3",
            "injongointent_intent_classification/lm_eval_p4",
            "injongointent_intent_classification/lm_eval_p5",
        ]
        assert merged.dataset.template_choice == domain_config.TemplateChoice.CYCLE
        assert merged.dataset.label_column == "intent"
        assert merged.dataset.packing is False
        assert merged.dataset.assistant_only_loss is True
        assert merged.dataset.subset == expected["subset"]
        assert merged.dataset.languages == expected["languages"]
        assert merged.dataset.max_seq_length == expected["max_seq_length"]


def test_experiment_schema_merges_representative_eval_config() -> None:
    schema = OmegaConf.structured(domain_config.ExperimentConfig)
    raw_cfg = compose_config_target("eval/run_llama_t2x_xho")

    merged = OmegaConf.merge(schema, raw_cfg)

    assert merged.mode == domain_config.RunMode.EVALUATE
    assert merged.eval_model.adapter == "hf"
    assert merged.evaluation.generation_tasks[0].id == "t2x_xho"
    assert merged.evaluation.generation_tasks[0].decoding.strategy == "beam"


def test_representative_eval_configs_share_run_defaults() -> None:
    schema = OmegaConf.structured(domain_config.ExperimentConfig)
    expected_configs = {
        "eval/run_llama_sib_xho": {
            "checkpoint_suffix": "ft_llama_125m_sa_general_all/final_merged_model",
            "task_pack": "sib_xho",
            "wandb_name": "eval-ft-llama-125m-sib-xho",
            "merge_lora": False,
        },
        "eval/run_mamba_sib_xho": {
            "checkpoint": "anrilombard/sallm-mamba-sib_xho",
            "task_pack": "sib_xho",
            "wandb_name": "eval-ft-mamba-125m-sib-xho",
            "merge_lora": False,
        },
        "eval/run_xlstm_sib_xho": {
            "checkpoint": "anrilombard/sallm-xlstm-125m",
            "task_pack": "sib_xho",
            "wandb_name": "eval-xlstm-125m-sib-xho",
            "merge_lora": None,
        },
    }

    for config_target, expected in expected_configs.items():
        raw_cfg = compose_config_target(config_target)
        merged = OmegaConf.merge(schema, raw_cfg)

        assert merged.mode == domain_config.RunMode.EVALUATE
        assert merged.eval_model.adapter == "hf"
        assert merged.eval_model.dtype == "bfloat16"
        assert merged.eval_model.device == "cuda:0"
        assert merged.eval_model.merge_lora is expected["merge_lora"]
        eval_model = OmegaConf.to_container(merged.eval_model, resolve=False)
        if "checkpoint" in expected:
            assert merged.eval_model.checkpoint == expected["checkpoint"]
        else:
            assert str(eval_model["checkpoint"]).endswith(expected["checkpoint_suffix"])
        assert list(merged.evaluation.task_packs) == [expected["task_pack"]]
        assert merged.wandb.project == "sallm-eval"
        assert merged.wandb.name == expected["wandb_name"]
        assert merged.training is None
        assert merged.dataset is None


def test_hpo_base_config_loader_composes_hydra_defaults() -> None:
    cfg = load_base_config("finetune/xlstm_sib_xho")

    assert cfg.mode == domain_config.RunMode.FINETUNE
    assert cfg.model.architecture == "xlstm"
    assert cfg.dataset.hf_name == "Davlan/sib200"
    assert cfg.dataset.task == domain_config.FinetuneTaskType.CLASSIFICATION
    assert cfg.dataset.subset == "xho_Latn"
    assert cfg.dataset.max_seq_length == 1024
    assert [template.id for template in cfg.dataset.templates] == [
        "sib_topic_classification/lm_eval_p1",
        "sib_topic_classification/lm_eval_p2",
        "sib_topic_classification/lm_eval_p3",
        "sib_topic_classification/lm_eval_p4",
        "sib_topic_classification/lm_eval_p5",
    ]


def test_hpo_base_config_loader_accepts_conf_file_paths() -> None:
    cfg = load_base_config("src/conf/finetune/xlstm_sib_xho.yaml")

    assert cfg.mode == domain_config.RunMode.FINETUNE
    assert cfg.model.architecture == "xlstm"
    assert cfg.dataset.hf_name == "Davlan/sib200"
    assert cfg.dataset.subset == "xho_Latn"
