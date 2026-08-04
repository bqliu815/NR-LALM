"""Fixed-parameter stochastic NR-LALM and NR-LALM+SOC."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable

import numpy as np

from .logistic import ConstrainedLogistic, evaluate_pair
from .mlalm import Array, OracleCounts
from .spider import ProjectedSPIDER, SPIDERConfig


@dataclass(frozen=True)
class NRLALMConfig:
    iterations: int
    rho: float
    beta: float
    spider: SPIDERConfig
    output_seed: int
    output_seeds: tuple[int, ...] = ()
    use_soc: bool = False
    required_linear_residual: float = 1.0e-10
    evaluation_states: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.iterations < 2:
            raise ValueError("stochastic theorem-facing run needs at least two iterations")
        if self.rho <= 0.0 or self.beta <= 0.0:
            raise ValueError("rho and beta must be positive")
        if self.required_linear_residual <= 0.0:
            raise ValueError("required_linear_residual must be positive")
        if any(state < 0 or state > self.iterations for state in self.evaluation_states):
            raise ValueError("evaluation state is outside 0:iterations")


@dataclass
class NRWorkCounts:
    primal_solves: int = 0
    correction_solves: int = 0
    constraint_values: int = 0
    constraint_jacobians: int = 0


@dataclass
class NRLALMResult:
    method: str
    status: str
    message: str
    x: Array
    multiplier: Array
    random_output_x: Array
    random_output_multiplier: Array
    random_output_index: int
    random_output_metrics: dict[str, float]
    random_output_xs: list[Array]
    random_output_multipliers: list[Array]
    random_output_indices: list[int]
    random_output_metrics_list: list[dict[str, float]]
    oracle_counts: OracleCounts
    work_counts: NRWorkCounts
    sample_stream_sha256: str
    wall_algorithm_seconds: float
    wall_evaluator_seconds: float
    trace: list[dict[str, float]] = field(default_factory=list)
    max_primal_relative_residual: float = 0.0
    max_correction_relative_residual: float = 0.0


def _primal_step(
    problem: ConstrainedLogistic,
    x: Array,
    rhs: Array,
    beta: float,
    rho: float,
) -> tuple[Array, float, float]:
    gram = problem.gram_matrix(x)
    reduced = gram + (beta / rho) * np.eye(problem.num_constraints)
    coefficient = np.linalg.solve(
        reduced, problem.jacobian_action(x, rhs)
    )
    step = (rhs - problem.adjoint_action(x, coefficient)) / beta
    residual = (
        beta * step
        + rho
        * problem.adjoint_action(
            x, problem.jacobian_action(x, step)
        )
        - rhs
    )
    relative = float(np.linalg.norm(residual) / max(1.0, np.linalg.norm(rhs)))
    return np.asarray(step, dtype=float), relative, float(np.linalg.cond(reduced))


def _soc_correction(
    problem: ConstrainedLogistic,
    trial: Array,
    defect: Array,
) -> tuple[Array, float, float]:
    gram = problem.gram_matrix(trial)
    coefficient = np.linalg.solve(gram, defect)
    correction = -problem.adjoint_action(trial, coefficient)
    residual = problem.jacobian_action(trial, correction) + defect
    relative = float(
        np.linalg.norm(residual) / max(1.0, np.linalg.norm(defect))
    )
    return (
        np.asarray(correction, dtype=float),
        relative,
        float(np.linalg.cond(gram)),
    )


def run_nr_lalm(
    problem: ConstrainedLogistic,
    x0: Array,
    lambda0: Array,
    config: NRLALMConfig,
    evaluator: Callable[[Array, Array, OracleCounts], dict[str, float]] | None = None,
) -> NRLALMResult:
    """Run the manuscript iteration with a projected-SPIDER objective oracle."""

    x = np.asarray(x0, dtype=float).copy()
    multiplier = np.asarray(lambda0, dtype=float).copy()
    if x.shape != (problem.n,):
        raise ValueError("x0 has the wrong shape")
    if multiplier.shape != (problem.num_constraints,):
        raise ValueError("lambda0 has the wrong shape")

    oracle_counts = OracleCounts()
    work_counts = NRWorkCounts()
    estimator = ProjectedSPIDER(problem, config.spider, oracle_counts)
    output_seeds = config.output_seeds or (config.output_seed,)
    output_indices = [
        int(np.random.default_rng(seed).integers(1, config.iterations))
        for seed in output_seeds
    ]
    selected_xs: list[Array | None] = [None] * len(output_indices)
    selected_multipliers: list[Array | None] = [None] * len(output_indices)
    previous_x: Array | None = None
    previous_raw: Array | None = None
    trace: list[dict[str, float]] = []
    evaluator_seconds = 0.0
    max_primal_residual = 0.0
    max_correction_residual = 0.0
    start = perf_counter()

    if evaluator is None:

        def evaluate(
            point: Array, dual: Array, counts: OracleCounts
        ) -> dict[str, float]:
            return evaluate_pair(problem, point, dual, counts)

    else:
        evaluate = evaluator

    def record(state: int, point: Array, dual: Array) -> None:
        nonlocal evaluator_seconds
        before = perf_counter()
        metrics = evaluate(point, dual, oracle_counts)
        evaluator_seconds += perf_counter() - before
        row = {key: float(value) for key, value in metrics.items()}
        row.update(
            state=float(state),
            component_calls=float(oracle_counts.objective_component_gradients),
            primal_solves=float(work_counts.primal_solves),
            correction_solves=float(work_counts.correction_solves),
        )
        trace.append(row)

    evaluation_states = set(config.evaluation_states)
    if 0 in evaluation_states:
        record(0, x, multiplier)

    status = "completed"
    message = "fixed-horizon run completed"
    for iteration in range(config.iterations):
        spider_step = estimator.step(
            iteration, x, previous_x, previous_raw
        )
        constraint = problem.constraints(x)
        work_counts.constraint_values += 1
        work_counts.constraint_jacobians += 1
        rhs = -spider_step.projected - problem.adjoint_action(
            x, multiplier + config.rho * constraint
        )
        try:
            step, primal_residual, _ = _primal_step(
                problem, x, rhs, config.beta, config.rho
            )
        except np.linalg.LinAlgError as error:
            status = "primal_solve_failure"
            message = str(error)
            break
        work_counts.primal_solves += 1
        max_primal_residual = max(max_primal_residual, primal_residual)
        if primal_residual > config.required_linear_residual:
            status = "primal_residual_failure"
            message = f"relative primal residual {primal_residual:.3e}"
            break

        linearized = constraint + problem.jacobian_action(x, step)
        trial = x + step
        trial_constraint = problem.constraints(trial)
        work_counts.constraint_values += 1
        next_x = trial
        next_constraint = trial_constraint
        if config.use_soc:
            defect = trial_constraint - linearized
            try:
                correction, correction_residual, _ = _soc_correction(
                    problem, trial, defect
                )
            except np.linalg.LinAlgError as error:
                status = "correction_solve_failure"
                message = str(error)
                break
            work_counts.correction_solves += 1
            work_counts.constraint_jacobians += 1
            max_correction_residual = max(
                max_correction_residual, correction_residual
            )
            if correction_residual > config.required_linear_residual:
                status = "correction_residual_failure"
                message = f"relative correction residual {correction_residual:.3e}"
                break
            next_x = trial + correction
            next_constraint = problem.constraints(next_x)
            work_counts.constraint_values += 1

        next_multiplier = multiplier + config.rho * next_constraint
        previous_x = x
        previous_raw = spider_step.raw
        x = np.asarray(next_x, dtype=float)
        multiplier = np.asarray(next_multiplier, dtype=float)
        state = iteration + 1
        for output_position, output_index in enumerate(output_indices):
            if iteration == output_index:
                selected_xs[output_position] = x.copy()
                selected_multipliers[output_position] = multiplier.copy()
        if state in evaluation_states:
            record(state, x, multiplier)

    wall_total = perf_counter() - start
    algorithm_seconds = max(0.0, wall_total - evaluator_seconds)
    missing_output = False
    for position in range(len(output_indices)):
        if selected_xs[position] is None or selected_multipliers[position] is None:
            selected_xs[position] = x.copy()
            selected_multipliers[position] = multiplier.copy()
            missing_output = True
    if missing_output and status == "completed":
        status = "random_output_missing"
        message = "at least one selected random output was not generated"
    complete_xs = [np.asarray(point, dtype=float) for point in selected_xs]
    complete_multipliers = [
        np.asarray(dual, dtype=float) for dual in selected_multipliers
    ]
    random_metrics_list: list[dict[str, float]] = []
    for selected_x, selected_multiplier in zip(
        complete_xs, complete_multipliers
    ):
        before = perf_counter()
        random_metrics_list.append(
            evaluate(selected_x, selected_multiplier, oracle_counts)
        )
        evaluator_seconds += perf_counter() - before
    return NRLALMResult(
        method="NR-LALM+SOC" if config.use_soc else "NR-LALM",
        status=status,
        message=message,
        x=x,
        multiplier=multiplier,
        random_output_x=complete_xs[0],
        random_output_multiplier=complete_multipliers[0],
        random_output_index=output_indices[0],
        random_output_metrics=random_metrics_list[0],
        random_output_xs=complete_xs,
        random_output_multipliers=complete_multipliers,
        random_output_indices=output_indices,
        random_output_metrics_list=random_metrics_list,
        oracle_counts=oracle_counts,
        work_counts=work_counts,
        sample_stream_sha256=estimator.stream_sha256,
        wall_algorithm_seconds=algorithm_seconds,
        wall_evaluator_seconds=evaluator_seconds,
        trace=trace,
        max_primal_relative_residual=max_primal_residual,
        max_correction_relative_residual=max_correction_residual,
    )
