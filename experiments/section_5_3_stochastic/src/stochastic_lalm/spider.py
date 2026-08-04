"""Projected SPIDER estimator with exact same-sample difference counting."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
from numpy.typing import NDArray

from .mlalm import Array, MLALMProblem, OracleCounts


@dataclass(frozen=True)
class SPIDERConfig:
    checkpoint_batch: int
    period: int
    difference_batch: int
    projection_radius: float
    seed: int

    def __post_init__(self) -> None:
        if self.checkpoint_batch < 1:
            raise ValueError("checkpoint_batch must be positive")
        if self.period < 1:
            raise ValueError("period must be positive")
        if self.difference_batch < 1:
            raise ValueError("difference_batch must be positive")
        if self.projection_radius <= 0.0:
            raise ValueError("projection_radius must be positive")


@dataclass(frozen=True)
class SPIDERStep:
    raw: Array
    projected: Array
    checkpoint: bool
    batch_size: int
    component_calls: int
    indices: NDArray[np.int64]


def project_ball(vector: Array, radius: float) -> Array:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= radius:
        return vector.copy()
    return vector * (radius / norm)


class ProjectedSPIDER:
    def __init__(
        self,
        problem: MLALMProblem,
        config: SPIDERConfig,
        counts: OracleCounts,
    ) -> None:
        self.problem = problem
        self.config = config
        self.counts = counts
        self.rng = np.random.default_rng(config.seed)
        self._hash = hashlib.sha256()

    def _sample(self, size: int, iteration: int, checkpoint: bool) -> NDArray[np.int64]:
        indices = self.rng.integers(
            0, self.problem.num_components, size=size, dtype=np.int64
        )
        header = np.asarray(
            [iteration, int(checkpoint), size], dtype="<i8"
        ).tobytes()
        self._hash.update(header)
        self._hash.update(np.asarray(indices, dtype="<i8").tobytes())
        return indices

    @property
    def stream_sha256(self) -> str:
        return self._hash.hexdigest()

    def step(
        self,
        iteration: int,
        x: Array,
        previous_x: Array | None,
        previous_raw: Array | None,
    ) -> SPIDERStep:
        checkpoint = iteration % self.config.period == 0
        if checkpoint:
            size = self.config.checkpoint_batch
            indices = self._sample(size, iteration, True)
            raw = np.asarray(
                self.problem.component_gradient(x, indices), dtype=float
            )
            calls = size
        else:
            if previous_x is None or previous_raw is None:
                raise ValueError("SPIDER difference step lacks previous state")
            size = self.config.difference_batch
            indices = self._sample(size, iteration, False)
            current = np.asarray(
                self.problem.component_gradient(x, indices), dtype=float
            )
            previous = np.asarray(
                self.problem.component_gradient(previous_x, indices), dtype=float
            )
            raw = np.asarray(previous_raw, dtype=float) + current - previous
            calls = 2 * size
        self.counts.objective_component_gradients += calls
        projected = project_ball(raw, self.config.projection_radius)
        return SPIDERStep(
            raw=raw,
            projected=projected,
            checkpoint=checkpoint,
            batch_size=size,
            component_calls=calls,
            indices=indices,
        )


def spider_total_calls(horizon: int, batch_scale: int = 1) -> int:
    """Return exact calls for B=K, Q=ceil(sqrt(K)), b=batch_scale*Q."""

    return spider_total_calls_scaled(
        horizon,
        checkpoint_batch_scale=1,
        difference_batch_scale=batch_scale,
    )


def spider_total_calls_scaled(
    horizon: int,
    checkpoint_batch_scale: int = 1,
    difference_batch_scale: int = 1,
) -> int:
    """Return exact calls for constant-scaled theorem-order SPIDER batches.

    The schedule is ``B=checkpoint_batch_scale*K``,
    ``Q=ceil(sqrt(K))``, and ``b=difference_batch_scale*Q``.  The scales
    alter only the constant factors in the manuscript's ``B=Theta(K)`` and
    ``b=Theta(sqrt(K))`` schedule.
    """

    if (
        horizon < 2
        or checkpoint_batch_scale < 1
        or difference_batch_scale < 1
    ):
        raise ValueError("invalid horizon or SPIDER batch scale")
    period = int(np.ceil(np.sqrt(horizon)))
    checkpoints = (horizon - 1) // period + 1
    difference_steps = horizon - checkpoints
    checkpoint_batch = checkpoint_batch_scale * horizon
    difference_batch = difference_batch_scale * period
    return checkpoints * checkpoint_batch + 2 * difference_steps * difference_batch


def largest_horizon_for_budget(budget: int, batch_scale: int = 1) -> int:
    """Find the largest theorem-schedule horizon not exceeding a call budget."""

    return largest_horizon_for_scaled_budget(
        budget,
        checkpoint_batch_scale=1,
        difference_batch_scale=batch_scale,
    )


def largest_horizon_for_scaled_budget(
    budget: int,
    checkpoint_batch_scale: int = 1,
    difference_batch_scale: int = 1,
) -> int:
    """Find the largest scaled-schedule horizon within a call budget."""

    def calls(horizon: int) -> int:
        return spider_total_calls_scaled(
            horizon,
            checkpoint_batch_scale=checkpoint_batch_scale,
            difference_batch_scale=difference_batch_scale,
        )

    if budget < calls(2):
        raise ValueError("budget is too small for a two-step SPIDER run")
    low, high = 2, 2
    while calls(high) <= budget:
        low = high
        high *= 2
    while low + 1 < high:
        middle = (low + high) // 2
        if calls(middle) <= budget:
            low = middle
        else:
            high = middle
    return low
