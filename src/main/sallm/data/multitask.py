from __future__ import annotations

import bisect
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from random import Random

from torch.utils.data import Dataset as TorchDataset


@dataclass(frozen=True)
class TaskComponent:
    name: str
    dataset: Sequence
    weight: float

    @property
    def size(self) -> int:
        size = len(self.dataset)
        if size <= 0:
            raise ValueError(f"Dataset '{self.name}' is empty")
        return size


class WeightedMultiTaskDataset(TorchDataset):
    def __init__(
        self,
        components: Iterable[TaskComponent],
        *,
        seed: int = 0,
        temperature: float = 0.0,
        epoch_size: int | None = None,
        min_prob: float | None = None,
        max_prob: float | None = None,
    ) -> None:
        components = list(components)
        if not components:
            raise ValueError("At least one component is required")
        positives = [c for c in components if c.weight > 0]
        if not positives:
            raise ValueError("All mix weights are non-positive")
        self._components = components
        self._base_seed = seed
        self._rng = Random(seed)
        self._temperature = temperature
        self._epoch_size = epoch_size or sum(c.size for c in components)
        if self._epoch_size <= 0:
            raise ValueError("epoch_size must be positive")
        self._probabilities = self._compute_probabilities(min_prob, max_prob)
        self._cdf = self._build_cdf(self._probabilities)
        self._expected = [p * self._epoch_size for p in self._probabilities]

    def _compute_probabilities(
        self, min_prob: float | None, max_prob: float | None
    ) -> list[float]:
        adjusted = []
        for component in self._components:
            if self._temperature:
                size_factor = component.size**self._temperature
            else:
                size_factor = 1.0
            adjusted.append(component.weight * size_factor)
        total = sum(adjusted)
        if total <= 0:
            raise ValueError("Mixture weights collapsed to zero")
        base = [x / total for x in adjusted]
        return self._apply_bounds(base, min_prob, max_prob)

    def _apply_bounds(
        self, base: list[float], min_prob: float | None, max_prob: float | None
    ) -> list[float]:
        n = len(base)
        lower = min_prob if min_prob is not None else 0.0
        upper = max_prob if max_prob is not None else 1.0
        if lower * n > 1 + 1e-9:
            raise ValueError("Lower probability bound is infeasible")
        if upper * n < 1 - 1e-9:
            raise ValueError("Upper probability bound is infeasible")
        assigned = [0.0] * n
        remaining = set(range(n))
        remaining_mass = 1.0
        while remaining:
            total_base = sum(base[i] for i in remaining)
            if total_base == 0:
                for idx in remaining:
                    assigned[idx] = remaining_mass / len(remaining)
                break
            scale = remaining_mass / total_base
            progressed = False
            to_remove = []
            for idx in remaining:
                candidate = base[idx] * scale
                if candidate < lower - 1e-12:
                    assigned[idx] = lower
                    remaining_mass -= lower
                    to_remove.append(idx)
                    progressed = True
                elif candidate > upper + 1e-12:
                    assigned[idx] = upper
                    remaining_mass -= upper
                    to_remove.append(idx)
                    progressed = True
            if not progressed:
                for idx in remaining:
                    assigned[idx] = base[idx] * scale
                break
            for idx in to_remove:
                remaining.remove(idx)
        total_assigned = sum(assigned)
        if abs(total_assigned - 1.0) > 1e-6:
            if total_assigned == 0:
                raise ValueError("Failed to allocate probability mass")
            correction = 1.0 / total_assigned
            assigned = [x * correction for x in assigned]
        return assigned

    def _build_cdf(self, probabilities: Sequence[float]) -> list[float]:
        cdf = []
        acc = 0.0
        for p in probabilities:
            acc += p
            cdf.append(acc)
        cdf[-1] = 1.0
        return cdf

    def __len__(self) -> int:
        return self._epoch_size

    def __getitem__(self, _: int) -> dict:
        choice = self._draw_component()
        component = self._components[choice]
        index = self._rng.randrange(component.size)
        example = component.dataset[index]
        if not isinstance(example, dict):
            raise TypeError("Expected dict samples from component datasets")
        result = dict(example)
        if "task_name" not in result:
            result["task_name"] = component.name
        return result

    def _draw_component(self) -> int:
        value = self._rng.random()
        return bisect.bisect_left(self._cdf, value)

    def set_epoch(self, epoch: int) -> None:
        self._rng = Random(self._base_seed + epoch)

    @property
    def probabilities(self) -> list[float]:
        return list(self._probabilities)

    @property
    def expected_counts(self) -> list[float]:
        return list(self._expected)

    @property
    def component_names(self) -> list[str]:
        return [c.name for c in self._components]

    def describe(self) -> str:
        parts = []
        for name, prob, expected in zip(
            self.component_names,
            self._probabilities,
            self._expected,
            strict=False,
        ):
            parts.append(f"{name}: p={prob:.4f}, expected={expected:.1f}")
        return " | ".join(parts)
