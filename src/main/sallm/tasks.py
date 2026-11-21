"""Per-task fine-tuning specifications in a single table."""

from dataclasses import dataclass

from sallm.config import FinetuneTaskType


@dataclass(frozen=True)
class TaskSpec:
    # Default metric_for_best_model under early stopping; greater is better.
    # Evaluators name keys "<prefix>/all_<metric>"; the metrics callbacks add
    # the "eval_" and "/" -> "_" aliases that these names select.
    selection_metric: str
    # Truncate from the left so the assistant answer survives long examples.
    truncate_left: bool
    # Template field a template must define to serve this task.
    required_template_field: str | None


TASK_SPECS: dict[FinetuneTaskType, TaskSpec] = {
    FinetuneTaskType.CLASSIFICATION: TaskSpec(
        selection_metric="eval_classification/all_macro_f1",
        truncate_left=True,
        required_template_field="label_mapping",
    ),
    FinetuneTaskType.NAMED_ENTITY_RECOGNITION: TaskSpec(
        selection_metric="eval_all_f1",
        truncate_left=False,
        required_template_field="ner_tags",
    ),
    FinetuneTaskType.POS_TAGGING: TaskSpec(
        selection_metric="eval_all_token_accuracy",
        truncate_left=False,
        required_template_field=None,
    ),
    FinetuneTaskType.INSTRUCTION: TaskSpec(
        selection_metric="eval_all_rougeL",
        truncate_left=True,
        required_template_field=None,
    ),
}
