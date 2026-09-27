import pytest
from sallm.data.multitask import TaskComponent, WeightedMultiTaskDataset


def _component(name: str, size: int, weight: float = 1.0) -> TaskComponent:
    rows = [{"text": f"{name}-{i}"} for i in range(size)]
    return TaskComponent(name=name, dataset=rows, weight=weight)


def test_weights_alone_set_probabilities_without_temperature() -> None:
    mix = WeightedMultiTaskDataset([_component("a", 10, 3.0), _component("b", 90)])

    assert mix.probabilities == pytest.approx([0.75, 0.25])
    assert len(mix) == 100


def test_temperature_one_is_proportional_to_size() -> None:
    mix = WeightedMultiTaskDataset(
        [_component("a", 10), _component("b", 90)], temperature=1.0
    )

    assert mix.probabilities == pytest.approx([0.1, 0.9])


def test_probability_bounds_clamp_and_renormalise() -> None:
    mix = WeightedMultiTaskDataset(
        [_component("a", 1), _component("b", 1), _component("c", 98)],
        temperature=1.0,
        min_prob=0.2,
    )

    assert mix.probabilities == pytest.approx([0.2, 0.2, 0.6])


def test_infeasible_lower_bound_is_rejected() -> None:
    with pytest.raises(ValueError, match="infeasible"):
        WeightedMultiTaskDataset([_component("a", 5), _component("b", 5)], min_prob=0.6)


def test_samples_are_deterministic_per_epoch_and_tagged_with_task() -> None:
    mix = WeightedMultiTaskDataset(
        [_component("a", 5), _component("b", 5)], seed=7, epoch_size=20
    )

    first = [mix[i] for i in range(20)]
    assert [mix[i] for i in range(20)] == first
    assert {row["task_name"] for row in first} == {"a", "b"}
    assert all(row["text"].startswith(row["task_name"]) for row in first)

    mix.set_epoch(1)
    assert [mix[i] for i in range(20)] != first
