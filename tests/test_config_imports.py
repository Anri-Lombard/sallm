from pathlib import Path

import pytest
import sallm.configs as domain_config
import yaml
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from sallm.hpo.trial import load_base_config

CONF_ROOT = Path(__file__).resolve().parents[1] / "src" / "conf"


def compose_config_target(config_target: str, overrides: list[str] | None = None):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_ROOT), version_base=None):
        cfg = compose(config_name=config_target, overrides=overrides or [])

    keys_in_cfg = list(cfg.keys())
    if (
        len(keys_in_cfg) == 1
        and keys_in_cfg[0] not in domain_config.ExperimentConfig.__dataclass_fields__
    ):
        return cfg[keys_in_cfg[0]]
    return cfg


def test_to_resolved_dict_resolves_omegaconf_mapping() -> None:
    cfg = OmegaConf.create({"answer": 42})

    assert domain_config.to_resolved_dict(cfg) == {"answer": 42}


def test_experiment_schema_merges_representative_finetune_config() -> None:
    schema = OmegaConf.structured(domain_config.ExperimentConfig)
    raw_cfg = compose_config_target(
        "finetune/run",
        overrides=[
            "finetune/defaults/arch@_global_=llama",
            "finetune/defaults/datasets@_global_=t2x",
            "dataset.subset=xho",
        ],
    )

    merged = OmegaConf.merge(schema, raw_cfg)

    assert merged.mode == domain_config.RunMode.FINETUNE
    assert merged.model.architecture == "llama"
    assert merged.dataset.hf_name == "github:francois-meyer/t2x"
    assert merged.generation_decoding.strategy == "beam"


@pytest.mark.parametrize(
    "arch", ["llama", "mamba2", "xlstm", "gated_deltanet", "hybrid"]
)
@pytest.mark.parametrize(
    ("dataset", "expected_hf_name", "expected_task"),
    [
        ("sib", "Davlan/sib200", domain_config.FinetuneTaskType.CLASSIFICATION),
        (
            "injongointent",
            "masakhane/InjongoIntent",
            domain_config.FinetuneTaskType.CLASSIFICATION,
        ),
        (
            "masakhanews",
            "masakhane/masakhanews",
            domain_config.FinetuneTaskType.CLASSIFICATION,
        ),
        (
            "masakhaner",
            "masakhane/masakhaner2",
            domain_config.FinetuneTaskType.NAMED_ENTITY_RECOGNITION,
        ),
        (
            "nchlt_ner",
            "anrilombard/nchlt-ner-sa4",
            domain_config.FinetuneTaskType.NAMED_ENTITY_RECOGNITION,
        ),
        (
            "masakhapos",
            "masakhane/masakhapos",
            domain_config.FinetuneTaskType.POS_TAGGING,
        ),
        (
            "afrihg",
            "github:dadelani/AfriHG",
            domain_config.FinetuneTaskType.INSTRUCTION,
        ),
        (
            "t2x",
            "github:francois-meyer/t2x",
            domain_config.FinetuneTaskType.INSTRUCTION,
        ),
        (
            "sa_multitask",
            "mix:sa_multitask",
            domain_config.FinetuneTaskType.INSTRUCTION,
        ),
    ],
)
def test_composable_finetune_run_covers_all_architectures_and_tasks(
    arch: str,
    dataset: str,
    expected_hf_name: str,
    expected_task: domain_config.FinetuneTaskType,
) -> None:
    schema = OmegaConf.structured(domain_config.ExperimentConfig)
    overrides = [
        f"finetune/defaults/arch@_global_={arch}",
        f"finetune/defaults/datasets@_global_={dataset}",
        f"wandb.name={arch}-{dataset}",
    ]
    if arch in ("gated_deltanet", "hybrid"):
        overrides.append(f"model.init_checkpoint=/tmp/{arch}-base")
    if dataset == "t2x":
        overrides.append("dataset.subset=xho")
    raw_cfg = compose_config_target("finetune/run", overrides=overrides)
    merged = OmegaConf.merge(schema, raw_cfg)

    assert merged.mode == domain_config.RunMode.FINETUNE
    assert merged.model.architecture == arch
    assert merged.dataset.hf_name == expected_hf_name
    assert merged.dataset.task == expected_task
    assert merged.dataset.packing is False
    assert merged.dataset.assistant_only_loss is True
    assert merged.training["optim"] == "adamw_torch_fused"
    assert merged.training["tf32"] is True
    assert merged.hub.enabled is False
    if dataset in ("afrihg", "masakhanews", "nchlt_ner", "sa_multitask"):
        assert merged.dataset.max_seq_length == 2048
    if dataset == "afrihg":
        assert merged.training["metric_for_best_model"] == "eval_all_chrf"
        assert merged.generation_decoding.strategy == "beam"
    if dataset == "t2x":
        assert merged.dataset.subset == "xho"
    if dataset == "sa_multitask":
        assert merged.dataset.mix_cap == 3000
        assert merged.training["general_selection"] is True
        assert merged.training["num_train_epochs"] == 10


def test_sweep_base_configs_resolve() -> None:
    sweeps = sorted((CONF_ROOT / "sweeps").glob("*.yaml"))
    checked = 0
    for path in sweeps:
        data = yaml.safe_load(path.read_text()) or {}
        command = data.get("command") or []
        if "--base-config" not in command:
            continue

        base_config = command[command.index("--base-config") + 1]
        overrides = [
            command[idx + 1]
            for idx, token in enumerate(command)
            if token == "--override" and idx + 1 < len(command)
        ]
        cfg = load_base_config(base_config, overrides=overrides)

        assert cfg.mode == domain_config.RunMode.FINETUNE
        assert cfg.dataset is not None
        checked += 1

    assert checked == len(sweeps)


def test_experiment_schema_merges_representative_eval_config() -> None:
    schema = OmegaConf.structured(domain_config.ExperimentConfig)
    raw_cfg = compose_config_target("eval/generate_t2x")

    merged = OmegaConf.merge(schema, raw_cfg)

    assert merged.mode == domain_config.RunMode.EVALUATE
    assert merged.evaluation.generation_tasks[0].id == "t2x_xho"
    assert merged.evaluation.generation_tasks[0].decoding.strategy == "beam"


def test_generic_eval_config_takes_checkpoint_and_task_packs() -> None:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_ROOT), version_base=None):
        cfg = compose(
            config_name="eval/run",
            overrides=[
                "eval_model.checkpoint=uctnlp/mzansilm-125m",
                "evaluation.task_packs=[sib_xho]",
                "wandb.name=eval-mzansilm-sib-xho",
            ],
        )
    merged = OmegaConf.merge(OmegaConf.structured(domain_config.ExperimentConfig), cfg)

    assert merged.mode == domain_config.RunMode.EVALUATE
    assert list(merged.evaluation.task_packs) == ["sib_xho"]
    assert merged.wandb.project == "sallm-eval"
    assert merged.training is None
    output_dir = OmegaConf.to_container(merged.evaluation, resolve=False)["output_dir"]
    assert output_dir.endswith("/results/eval/${..wandb.name}")


def test_hpo_base_config_loader_composes_hydra_defaults() -> None:
    cfg = load_base_config(
        "finetune/run",
        overrides=[
            "finetune/defaults/arch@_global_=xlstm",
            "finetune/defaults/datasets@_global_=sib",
            "dataset.subset=xho_Latn",
        ],
    )

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
    cfg = load_base_config(
        "src/conf/finetune/run.yaml",
        overrides=[
            "finetune/defaults/arch@_global_=xlstm",
            "finetune/defaults/datasets@_global_=sib",
            "dataset.subset=xho_Latn",
        ],
    )

    assert cfg.mode == domain_config.RunMode.FINETUNE
    assert cfg.model.architecture == "xlstm"
    assert cfg.dataset.hf_name == "Davlan/sib200"
    assert cfg.dataset.subset == "xho_Latn"


def test_every_entry_config_merges_into_the_schema() -> None:
    schema = OmegaConf.structured(domain_config.ExperimentConfig)
    targets = [
        f"{group}/{path.stem}"
        for group in ("finetune", "eval")
        for path in sorted((CONF_ROOT / group).glob("*.yaml"))
    ]

    for target in targets:
        OmegaConf.merge(schema, compose_config_target(target))

    assert targets
