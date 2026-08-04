"""Equation-level implementation of MLALM Algorithm 1.

The implementation follows equations (4)--(6) of Shi, Wang, and Wang.
Only objective component gradients are stochastic.  Constraint values and
Jacobians are deterministic and are therefore excluded from the stochastic
oracle counter.  Evaluation-only full gradients are also kept in a separate
counter so that plotting cannot silently consume the algorithmic budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from time import perf_counter
from typing import Callable, Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray


Array: TypeAlias = NDArray[np.float64]
Schedule: TypeAlias = float | Callable[[int], float]


class MLALMProblem(Protocol):
    """Minimal interface required by the equality/inequality MLALM solver."""

    n: int
    num_components: int
    equality_mask: NDArray[np.bool_]

    def component_gradient(self, x: Array, indices: NDArray[np.int64]) -> Array:
        """Return the mean objective gradient over ``indices``."""

    def constraints(self, x: Array) -> Array:
        """Return deterministic constraint values, ordered as in the mask."""

    def jacobian(self, x: Array) -> Array:
        """Return the n-by-m column-gradient Jacobian."""

    def prox(self, point: Array, eta: float) -> Array:
        """Return the equation-(5) proximal map at ``point``."""


@dataclass
class OracleCounts:
    """Algorithmic and evaluator work are deliberately separated."""

    objective_component_gradients: int = 0
    constraint_values: int = 0
    constraint_jacobians: int = 0
    evaluator_full_gradient_components: int = 0
    evaluator_constraint_values: int = 0
    evaluator_constraint_jacobians: int = 0
    evaluator_multiplier_solves: int = 0


@dataclass(frozen=True)
class MLALMConfig:
    iterations: int
    batch_size: int = 1
    eta: Schedule = 1.0e-2
    beta: Schedule = 1.0
    rho: Schedule = 0.5
    alpha: Schedule = 0.5
    seed: int = 1
    output_seed: int | None = None
    output_seeds: tuple[int, ...] = ()
    x0: Array | None = None
    lambda0: Array | None = None
    record_every: int = 1
    store_internal: bool = False
    enforce_rho_lt_beta: bool = True

    def validate(self) -> None:
        if self.iterations < 1:
            raise ValueError("iterations must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.record_every < 1:
            raise ValueError("record_every must be positive")


@dataclass
class MLALMResult:
    x: Array
    multiplier: Array
    counts: OracleCounts
    trace: list[dict[str, float]] = field(default_factory=list)
    x_history: list[Array] = field(default_factory=list)
    multiplier_history: list[Array] = field(default_factory=list)
    direction_history: list[Array] = field(default_factory=list)
    sample_history: list[NDArray[np.int64]] = field(default_factory=list)
    parameter_warnings: list[str] = field(default_factory=list)
    random_output_index: int | None = None
    random_output_x: Array | None = None
    random_output_multiplier: Array | None = None
    random_output_metrics: dict[str, float] | None = None
    random_output_indices: list[int] = field(default_factory=list)
    random_output_xs: list[Array] = field(default_factory=list)
    random_output_multipliers: list[Array] = field(default_factory=list)
    random_output_metrics_list: list[dict[str, float]] = field(default_factory=list)
    wall_algorithm_seconds: float = 0.0
    wall_evaluator_seconds: float = 0.0
    sample_stream_sha256: str = ""


def _schedule_value(schedule: Schedule, iteration: int, name: str) -> float:
    value = float(schedule(iteration) if callable(schedule) else schedule)
    if not np.isfinite(value):
        raise ValueError(f"{name}[{iteration}] is not finite")
    return value


def _augmented_gradient(
    problem: MLALMProblem,
    x: Array,
    multiplier: Array,
    beta: float,
    counts: OracleCounts,
) -> Array:
    values = np.asarray(problem.constraints(x), dtype=float)
    counts.constraint_values += 1
    counts.constraint_jacobians += 1
    equality_mask = np.asarray(problem.equality_mask, dtype=bool)
    weights = np.empty_like(multiplier)
    weights[equality_mask] = (
        multiplier[equality_mask] + beta * values[equality_mask]
    )
    inequality_mask = ~equality_mask
    weights[inequality_mask] = np.maximum(
        multiplier[inequality_mask] + beta * values[inequality_mask], 0.0
    )
    adjoint = getattr(problem, "adjoint_action", None)
    if callable(adjoint):
        return np.asarray(adjoint(x, weights), dtype=float)
    jacobian = np.asarray(problem.jacobian(x), dtype=float)
    return jacobian @ weights


def mlalm_step_direction(
    current_gradient: Array,
    previous_same_sample_gradient: Array | None,
    previous_direction: Array | None,
    alpha_previous: float | None,
) -> Array:
    """Evaluate equation (4) without hiding the same-sample requirement."""

    current_gradient = np.asarray(current_gradient, dtype=float)
    if previous_direction is None:
        if previous_same_sample_gradient is not None or alpha_previous is not None:
            raise ValueError("first MLALM direction received previous-state inputs")
        return current_gradient.copy()
    if previous_same_sample_gradient is None or alpha_previous is None:
        raise ValueError("recursive MLALM direction requires all previous-state inputs")
    if not 0.0 <= alpha_previous <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    return current_gradient + (1.0 - alpha_previous) * (
        np.asarray(previous_direction, dtype=float)
        - np.asarray(previous_same_sample_gradient, dtype=float)
    )


def update_multipliers(
    multiplier: Array,
    constraint_values: Array,
    equality_mask: NDArray[np.bool_],
    beta: float,
    rho: float,
) -> Array:
    """Evaluate equation (6) for mixed equalities and inequalities."""

    multiplier = np.asarray(multiplier, dtype=float)
    constraint_values = np.asarray(constraint_values, dtype=float)
    equality_mask = np.asarray(equality_mask, dtype=bool)
    if beta <= 0.0 or rho <= 0.0:
        raise ValueError("beta and rho must be positive")
    if multiplier.shape != constraint_values.shape:
        raise ValueError("multiplier and constraint vector shapes differ")
    updated = multiplier.copy()
    updated[equality_mask] += rho * constraint_values[equality_mask]
    inequality_mask = ~equality_mask
    updated[inequality_mask] += rho * np.maximum(
        -multiplier[inequality_mask] / beta,
        constraint_values[inequality_mask],
    )
    return updated


def run_mlalm(
    problem: MLALMProblem,
    config: MLALMConfig,
    evaluator: Callable[[Array, Array, OracleCounts], dict[str, float]] | None = None,
) -> MLALMResult:
    """Run Algorithm 1 with exact equation-(4) sample pairing.

    The schedule index is zero based in code: index zero represents paper
    iteration t=1.  At all recursive iterations the newly sampled batch is
    evaluated at both the current and previous primal/dual/penalty states.
    """

    config.validate()
    equality_mask = np.asarray(problem.equality_mask, dtype=bool)
    num_constraints = int(equality_mask.size)
    x = (
        np.zeros(problem.n, dtype=float)
        if config.x0 is None
        else np.asarray(config.x0, dtype=float).copy()
    )
    multiplier = (
        np.zeros(num_constraints, dtype=float)
        if config.lambda0 is None
        else np.asarray(config.lambda0, dtype=float).copy()
    )
    if x.shape != (problem.n,):
        raise ValueError("x0 has the wrong shape")
    if multiplier.shape != (num_constraints,):
        raise ValueError("lambda0 has the wrong shape")
    if np.any(multiplier[~equality_mask] < 0.0):
        raise ValueError("initial inequality multipliers must be nonnegative")

    counts = OracleCounts()
    rng = np.random.default_rng(config.seed)
    sample_digest = hashlib.sha256()
    trace: list[dict[str, float]] = []
    x_history: list[Array] = [x.copy()] if config.store_internal else []
    multiplier_history: list[Array] = (
        [multiplier.copy()] if config.store_internal else []
    )
    direction_history: list[Array] = []
    sample_history: list[NDArray[np.int64]] = []
    warnings: list[str] = []
    evaluator_seconds = 0.0
    start = perf_counter()
    output_seeds = (
        config.output_seeds
        if config.output_seeds
        else (() if config.output_seed is None else (config.output_seed,))
    )
    output_indices = [
        int(np.random.default_rng(seed).integers(1, config.iterations + 1))
        for seed in output_seeds
    ]
    selected_xs: list[Array | None] = [None] * len(output_indices)
    selected_multipliers: list[Array | None] = [None] * len(output_indices)

    previous_x: Array | None = None
    previous_multiplier: Array | None = None
    previous_beta: float | None = None
    previous_direction: Array | None = None
    previous_alpha: float | None = None

    if evaluator is not None:
        before = perf_counter()
        row = dict(evaluator(x, multiplier, counts))
        evaluator_seconds += perf_counter() - before
        row.update(iteration=0.0, component_calls=0.0)
        trace.append(row)

    for iteration in range(config.iterations):
        eta = _schedule_value(config.eta, iteration, "eta")
        beta = _schedule_value(config.beta, iteration, "beta")
        rho = _schedule_value(config.rho, iteration, "rho")
        alpha = _schedule_value(config.alpha, iteration, "alpha")
        if eta <= 0.0 or beta <= 0.0 or rho <= 0.0:
            raise ValueError("eta, beta, and rho must be positive")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")
        if rho >= beta:
            message = (
                f"paper parameter-domain condition rho < beta fails at "
                f"iteration {iteration + 1}: rho={rho:.16g}, beta={beta:.16g}"
            )
            if config.enforce_rho_lt_beta:
                raise ValueError(message)
            if not warnings:
                warnings.append(message)

        indices = rng.integers(
            0, problem.num_components, size=config.batch_size, dtype=np.int64
        )
        sample_digest.update(
            np.asarray([iteration, config.batch_size], dtype="<i8").tobytes()
        )
        sample_digest.update(np.asarray(indices, dtype="<i8").tobytes())
        current_objective_gradient = np.asarray(
            problem.component_gradient(x, indices), dtype=float
        )
        counts.objective_component_gradients += config.batch_size
        current_gradient = current_objective_gradient + _augmented_gradient(
            problem, x, multiplier, beta, counts
        )

        previous_same_sample_gradient: Array | None = None
        if previous_x is not None:
            if previous_multiplier is None or previous_beta is None:
                raise RuntimeError("incomplete previous MLALM state")
            previous_objective_gradient = np.asarray(
                problem.component_gradient(previous_x, indices), dtype=float
            )
            counts.objective_component_gradients += config.batch_size
            previous_same_sample_gradient = (
                previous_objective_gradient
                + _augmented_gradient(
                    problem,
                    previous_x,
                    previous_multiplier,
                    previous_beta,
                    counts,
                )
            )

        direction = mlalm_step_direction(
            current_gradient,
            previous_same_sample_gradient,
            previous_direction,
            previous_alpha,
        )
        next_x = np.asarray(problem.prox(x - eta * direction, eta), dtype=float)
        next_values = np.asarray(problem.constraints(next_x), dtype=float)
        counts.constraint_values += 1
        next_multiplier = update_multipliers(
            multiplier, next_values, equality_mask, beta, rho
        )

        previous_x = x
        previous_multiplier = multiplier
        previous_beta = beta
        previous_direction = direction
        previous_alpha = alpha
        x = next_x
        multiplier = next_multiplier

        for output_position, output_index in enumerate(output_indices):
            if output_index == iteration + 1:
                selected_xs[output_position] = x.copy()
                selected_multipliers[output_position] = multiplier.copy()

        if config.store_internal:
            x_history.append(x.copy())
            multiplier_history.append(multiplier.copy())
            direction_history.append(direction.copy())
            sample_history.append(indices.copy())

        should_record = (
            evaluator is not None
            and (
                (iteration + 1) % config.record_every == 0
                or iteration + 1 == config.iterations
            )
        )
        if should_record:
            before = perf_counter()
            row = dict(evaluator(x, multiplier, counts))
            evaluator_seconds += perf_counter() - before
            row.update(
                iteration=float(iteration + 1),
                component_calls=float(counts.objective_component_gradients),
                eta=eta,
                beta=beta,
                rho=rho,
                alpha=alpha,
            )
            trace.append(row)

    complete_xs = [
        np.asarray(point, dtype=float)
        for point in selected_xs
        if point is not None
    ]
    complete_multipliers = [
        np.asarray(dual, dtype=float)
        for dual in selected_multipliers
        if dual is not None
    ]
    if len(complete_xs) != len(output_indices) or len(complete_multipliers) != len(
        output_indices
    ):
        raise RuntimeError("independent MLALM output was not captured")
    random_metrics_list: list[dict[str, float]] = []
    if evaluator is not None:
        for point, dual in zip(complete_xs, complete_multipliers):
            before = perf_counter()
            random_metrics_list.append(dict(evaluator(point, dual, counts)))
            evaluator_seconds += perf_counter() - before
    wall_total = perf_counter() - start
    return MLALMResult(
        x=x,
        multiplier=multiplier,
        counts=counts,
        trace=trace,
        x_history=x_history,
        multiplier_history=multiplier_history,
        direction_history=direction_history,
        sample_history=sample_history,
        parameter_warnings=warnings,
        random_output_index=output_indices[0] if output_indices else None,
        random_output_x=complete_xs[0] if complete_xs else None,
        random_output_multiplier=(
            complete_multipliers[0] if complete_multipliers else None
        ),
        random_output_metrics=(
            random_metrics_list[0] if random_metrics_list else None
        ),
        random_output_indices=output_indices,
        random_output_xs=complete_xs,
        random_output_multipliers=complete_multipliers,
        random_output_metrics_list=random_metrics_list,
        wall_algorithm_seconds=max(0.0, wall_total - evaluator_seconds),
        wall_evaluator_seconds=evaluator_seconds,
        sample_stream_sha256=sample_digest.hexdigest(),
    )
