"""Matrix-free reduced-system NR-LALM, NR-LALM+SOC, and L-AL."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
from time import perf_counter
from typing import Literal

import numpy as np

from .problem import Array, SparseConstrainedLogistic

Method = Literal["nr_lalm", "nr_lalm_soc", "l_al", "ipopt"]


@dataclass(frozen=True)
class DynamicBetaConfig:
    rho: float
    beta_floor: float
    beta_initial: float
    beta_ceiling: float
    mu: float = 2.0
    acceptance_eta: float = 0.05
    decrease_ratio: float = 1.0
    decrease_factor: float = 0.5
    max_iterations: int = 200
    max_backtracks: int = 60
    target_residual_squared: float = 1.0e-14
    required_linear_residual: float = 1.0e-10
    maximum_linear_refinements: int = 0

    def __post_init__(self) -> None:
        if self.rho <= 0.0:
            raise ValueError("rho must be positive")
        if not (
            0.0
            < self.beta_floor
            <= self.beta_initial
            <= self.beta_ceiling
        ):
            raise ValueError("invalid beta safeguards")
        if self.mu <= 1.0:
            raise ValueError("mu must exceed one")
        if not (0.0 <= self.acceptance_eta < 1.0):
            raise ValueError("acceptance_eta must lie in [0,1)")
        if not (0.0 < self.decrease_factor <= 1.0):
            raise ValueError("decrease_factor must lie in (0,1]")
        if self.max_iterations <= 0 or self.max_backtracks < 0:
            raise ValueError("invalid iteration limits")
        if self.maximum_linear_refinements < 0:
            raise ValueError("maximum_linear_refinements must be nonnegative")


@dataclass
class SolverRun:
    method: Method
    status: str
    message: str
    trace: list[dict[str, float | int | None]]
    final_x: Array
    final_multiplier: Array
    counters: dict[str, int]
    config: dict[str, object]
    metadata: dict[str, object] = field(default_factory=dict)

    def to_jsonable(self) -> dict[str, object]:
        return {
            "method": self.method,
            "status": self.status,
            "message": self.message,
            "trace": self.trace,
            "counters": self.counters,
            "config": self.config,
            "metadata": {
                **self.metadata,
                "final_x_sha256": hashlib.sha256(
                    np.asarray(self.final_x, dtype="<f8").tobytes()
                ).hexdigest(),
                "final_multiplier": self.final_multiplier.tolist(),
            },
        }


def _augmented(
    objective: float,
    constraint: Array,
    multiplier: Array,
    rho: float,
) -> float:
    return float(
        objective
        + multiplier @ constraint
        + 0.5 * rho * float(constraint @ constraint)
    )


def _reduced_step(
    problem: SparseConstrainedLogistic,
    x: Array,
    rhs: Array,
    *,
    beta: float,
    rho: float,
) -> tuple[Array, float, float]:
    gram = problem.gram_matrix(x)
    reduced = gram + (beta / rho) * np.eye(problem.constraints)
    coefficient = np.linalg.solve(
        reduced, problem.jacobian_action(x, rhs)
    )
    step = (
        rhs - problem.adjoint_action(x, coefficient)
    ) / beta
    _, relative = _reduced_system_residual(
        problem,
        x,
        step,
        rhs,
        beta=beta,
        rho=rho,
    )
    return (
        np.asarray(step, dtype=np.float64),
        relative,
        float(np.linalg.cond(reduced)),
    )


def _reduced_system_residual(
    problem: SparseConstrainedLogistic,
    x: Array,
    step: Array,
    rhs: Array,
    *,
    beta: float,
    rho: float,
) -> tuple[Array, float]:
    residual = (
        beta * step
        + rho
        * problem.adjoint_action(
            x, problem.jacobian_action(x, step)
        )
        - rhs
    )
    relative = float(
        np.linalg.norm(residual) / max(1.0, np.linalg.norm(rhs))
    )
    return np.asarray(residual, dtype=np.float64), relative


def _refined_reduced_step(
    problem: SparseConstrainedLogistic,
    x: Array,
    rhs: Array,
    *,
    beta: float,
    rho: float,
    required_residual: float,
    maximum_refinements: int,
) -> tuple[Array, float, float, float, int]:
    """Solve the same reduced system, refining only when its gate fails.

    The returned residuals are the final and pre-refinement relative
    residuals.  The final integer counts every additional Woodbury solve,
    including a correction that is rejected because it does not improve the
    full-system residual.
    """

    step, reported_residual, reduced_condition = _reduced_step(
        problem, x, rhs, beta=beta, rho=rho
    )
    residual, current_residual = _reduced_system_residual(
        problem,
        x,
        step,
        rhs,
        beta=beta,
        rho=rho,
    )
    pre_refinement_residual = float(current_residual)
    maximum_condition = float(reduced_condition)
    refinement_solves = 0

    for _ in range(maximum_refinements):
        if current_residual <= required_residual:
            break
        correction, _, correction_condition = _reduced_step(
            problem,
            x,
            -residual,
            beta=beta,
            rho=rho,
        )
        refinement_solves += 1
        maximum_condition = max(
            maximum_condition, float(correction_condition)
        )
        candidate = np.asarray(step + correction, dtype=np.float64)
        candidate_residual, candidate_relative = (
            _reduced_system_residual(
                problem,
                x,
                candidate,
                rhs,
                beta=beta,
                rho=rho,
            )
        )
        if (
            not np.isfinite(candidate_relative)
            or candidate_relative >= current_residual
        ):
            break
        step = candidate
        residual = candidate_residual
        current_residual = candidate_relative

    if not np.isclose(
        reported_residual,
        pre_refinement_residual,
        rtol=8.0 * np.finfo(float).eps,
        atol=8.0 * np.finfo(float).eps,
    ):
        pre_refinement_residual = max(
            float(reported_residual), pre_refinement_residual
        )
    return (
        np.asarray(step, dtype=np.float64),
        float(current_residual),
        maximum_condition,
        pre_refinement_residual,
        refinement_solves,
    )


def _soc_correction(
    problem: SparseConstrainedLogistic,
    trial: Array,
    defect: Array,
) -> tuple[Array, float, float]:
    gram = problem.gram_matrix(trial)
    coefficient = np.linalg.solve(gram, defect)
    correction = -problem.adjoint_action(trial, coefficient)
    residual = (
        problem.jacobian_action(trial, correction) + defect
    )
    relative = float(
        np.linalg.norm(residual) / max(1.0, np.linalg.norm(defect))
    )
    return (
        np.asarray(correction, dtype=np.float64),
        relative,
        float(np.linalg.cond(gram)),
    )


def independent_record(
    problem: SparseConstrainedLogistic,
    x: Array,
    multiplier: Array,
    *,
    iteration: int,
    native_iteration: int | None,
    elapsed: float,
    beta: float | None,
    primal_solves: int,
    correction_solves: int,
    backtracks: int,
    model_ratio: float | None,
    step_norm: float | None,
    correction_norm: float | None,
    solve_relative_residual: float | None,
    correction_relative_residual: float | None,
) -> dict[str, float | int | None]:
    gradient = problem.gradient(x)
    constraint = problem.constraint(x)
    stationarity = gradient + problem.adjoint_action(x, multiplier)
    stationarity_squared = float(stationarity @ stationarity)
    feasibility_squared = float(constraint @ constraint)
    return {
        "iteration": iteration,
        "native_iteration": native_iteration,
        "wall_seconds": elapsed,
        "objective": problem.objective(x),
        "stationarity_2_squared": stationarity_squared,
        "feasibility_2_squared": feasibility_squared,
        "pair_residual_squared": (
            stationarity_squared + feasibility_squared
        ),
        "train_accuracy": problem.accuracy(x, split="train"),
        "test_accuracy": problem.accuracy(x, split="test"),
        "accepted_beta": beta,
        "cumulative_primal_solves": primal_solves,
        "cumulative_correction_solves": correction_solves,
        "backtracks": backtracks,
        "model_ratio": model_ratio,
        "step_norm": step_norm,
        "correction_norm": correction_norm,
        "linear_solve_relative_residual": solve_relative_residual,
        "correction_solve_relative_residual": (
            correction_relative_residual
        ),
    }


def _finish(
    *,
    method: Method,
    status: str,
    message: str,
    config: DynamicBetaConfig,
    trace: list[dict[str, float | int | None]],
    x: Array,
    multiplier: Array,
    primal_solves: int,
    correction_solves: int,
    refinement_solves: int,
    rejected_trials: int,
    metadata: dict[str, object],
) -> SolverRun:
    return SolverRun(
        method=method,
        status=status,
        message=message,
        trace=trace,
        final_x=x.copy(),
        final_multiplier=multiplier.copy(),
        counters={
            "accepted_iterations": len(trace) - 1,
            "primal_linear_solves": primal_solves,
            "linear_refinement_solves": refinement_solves,
            "correction_linear_solves": correction_solves,
            "rejected_trials": rejected_trials,
        },
        config=asdict(config),
        metadata=metadata,
    )


def solve_nr_lalm(
    problem: SparseConstrainedLogistic,
    config: DynamicBetaConfig,
    x0: Array,
    lambda0: Array,
    *,
    use_soc: bool,
) -> SolverRun:
    method: Method = "nr_lalm_soc" if use_soc else "nr_lalm"
    x = np.asarray(x0, dtype=np.float64).copy()
    multiplier = np.asarray(lambda0, dtype=np.float64).copy()
    problem.check_shapes(x)
    objective = problem.objective(x)
    constraint = problem.constraint(x)
    beta_candidate = config.beta_initial
    primal_solves = 0
    refinement_solves = 0
    correction_solves = 0
    rejected_trials = 0
    max_primal_residual = 0.0
    max_pre_refinement_residual = 0.0
    max_correction_residual = 0.0
    max_reduced_condition = 0.0
    max_correction_condition = 0.0
    start = perf_counter()
    trace = [
        independent_record(
            problem,
            x,
            multiplier,
            iteration=0,
            native_iteration=0,
            elapsed=0.0,
            beta=None,
            primal_solves=0,
            correction_solves=0,
            backtracks=0,
            model_ratio=None,
            step_norm=None,
            correction_norm=0.0,
            solve_relative_residual=None,
            correction_relative_residual=None,
        )
    ]

    for outer in range(config.max_iterations):
        if (
            float(trace[-1]["pair_residual_squared"])
            <= config.target_residual_squared
        ):
            return _finish(
                method=method,
                status="target_reached",
                message="exact pair-residual target reached",
                config=config,
                trace=trace,
                x=x,
                multiplier=multiplier,
                primal_solves=primal_solves,
                correction_solves=correction_solves,
                refinement_solves=refinement_solves,
                rejected_trials=rejected_trials,
                metadata={
                    "linear_solver": "matrix_free_woodbury",
                    "max_primal_linear_residual": max_primal_residual,
                    "max_pre_refinement_linear_residual": (
                        max_pre_refinement_residual
                    ),
                    "max_correction_linear_residual": (
                        max_correction_residual
                    ),
                    "max_reduced_condition": max_reduced_condition,
                    "max_correction_condition": (
                        max_correction_condition
                    ),
                },
            )
        gradient = problem.gradient(x)
        rhs = -gradient - problem.adjoint_action(
            x, multiplier + config.rho * constraint
        )
        current_al = _augmented(
            objective, constraint, multiplier, config.rho
        )
        trial_beta = float(
            np.clip(
                beta_candidate, config.beta_floor, config.beta_ceiling
            )
        )
        accepted_data: tuple[object, ...] | None = None

        for inner in range(config.max_backtracks + 1):
            try:
                (
                    step,
                    solve_residual,
                    reduced_condition,
                    pre_refinement_residual,
                    refinement_count,
                ) = _refined_reduced_step(
                    problem,
                    x,
                    rhs,
                    beta=trial_beta,
                    rho=config.rho,
                    required_residual=(
                        config.required_linear_residual
                    ),
                    maximum_refinements=(
                        config.maximum_linear_refinements
                    ),
                )
            except np.linalg.LinAlgError as error:
                return _finish(
                    method=method,
                    status="linear_solve_failure",
                    message=str(error),
                    config=config,
                    trace=trace,
                    x=x,
                    multiplier=multiplier,
                    primal_solves=primal_solves,
                    correction_solves=correction_solves,
                    refinement_solves=refinement_solves,
                    rejected_trials=rejected_trials,
                    metadata={"linear_solver": "matrix_free_woodbury"},
                )
            primal_solves += 1 + refinement_count
            refinement_solves += refinement_count
            max_primal_residual = max(
                max_primal_residual, solve_residual
            )
            max_pre_refinement_residual = max(
                max_pre_refinement_residual,
                pre_refinement_residual,
            )
            max_reduced_condition = max(
                max_reduced_condition, reduced_condition
            )
            if solve_residual > config.required_linear_residual:
                return _finish(
                    method=method,
                    status="linear_residual_failure",
                    message=(
                        f"primal relative residual {solve_residual:.3e}"
                    ),
                    config=config,
                    trace=trace,
                    x=x,
                    multiplier=multiplier,
                    primal_solves=primal_solves,
                    correction_solves=correction_solves,
                    refinement_solves=refinement_solves,
                    rejected_trials=rejected_trials,
                    metadata={"linear_solver": "matrix_free_woodbury"},
                )
            linearized = constraint + problem.jacobian_action(x, step)
            base_trial = x + step
            base_constraint = problem.constraint(base_trial)
            base_defect = base_constraint - linearized
            correction = np.zeros_like(step)
            correction_residual = 0.0
            correction_condition = 0.0
            if use_soc:
                try:
                    (
                        correction,
                        correction_residual,
                        correction_condition,
                    ) = _soc_correction(
                        problem, base_trial, base_defect
                    )
                except np.linalg.LinAlgError as error:
                    return _finish(
                        method=method,
                        status="correction_solve_failure",
                        message=str(error),
                        config=config,
                        trace=trace,
                        x=x,
                        multiplier=multiplier,
                        primal_solves=primal_solves,
                        correction_solves=correction_solves,
                        refinement_solves=refinement_solves,
                        rejected_trials=rejected_trials,
                        metadata={
                            "linear_solver": "matrix_free_woodbury"
                        },
                    )
                correction_solves += 1
                max_correction_residual = max(
                    max_correction_residual, correction_residual
                )
                max_correction_condition = max(
                    max_correction_condition, correction_condition
                )
            trial = base_trial + correction
            trial_objective = problem.objective(trial)
            trial_constraint = problem.constraint(trial)
            predicted = 0.5 * float(step @ rhs)
            actual = current_al - _augmented(
                trial_objective,
                trial_constraint,
                multiplier,
                config.rho,
            )
            threshold = (
                64.0
                * np.finfo(float).eps
                * max(1.0, abs(current_al))
            )
            if predicted > threshold:
                ratio = float(actual / predicted)
            elif actual >= -threshold:
                ratio = float("inf")
            else:
                ratio = float("-inf")
            accepted = bool(
                np.all(np.isfinite(trial))
                and (
                    (
                        predicted > threshold
                        and actual >= config.acceptance_eta * predicted
                    )
                    or (
                        predicted <= threshold
                        and actual >= -threshold
                    )
                )
            )
            if accepted:
                accepted_data = (
                    step,
                    trial,
                    trial_objective,
                    trial_constraint,
                    correction,
                    solve_residual,
                    correction_residual,
                    ratio,
                    inner,
                )
                break
            rejected_trials += 1
            if trial_beta >= config.beta_ceiling:
                break
            trial_beta = min(
                trial_beta * config.mu, config.beta_ceiling
            )

        if accepted_data is None:
            return _finish(
                method=method,
                status="backtracking_limit",
                message=f"no acceptable beta at outer iteration {outer}",
                config=config,
                trace=trace,
                x=x,
                multiplier=multiplier,
                primal_solves=primal_solves,
                correction_solves=correction_solves,
                refinement_solves=refinement_solves,
                rejected_trials=rejected_trials,
                metadata={"linear_solver": "matrix_free_woodbury"},
            )
        (
            step,
            x,
            objective,
            constraint,
            correction,
            solve_residual,
            correction_residual,
            ratio,
            inner,
        ) = accepted_data
        x = np.asarray(x, dtype=np.float64)
        constraint = np.asarray(constraint, dtype=np.float64)
        multiplier = multiplier + config.rho * constraint
        if int(inner) == 0 and float(ratio) >= config.decrease_ratio:
            beta_candidate = max(
                trial_beta * config.decrease_factor, config.beta_floor
            )
        else:
            beta_candidate = trial_beta
        trace.append(
            independent_record(
                problem,
                x,
                multiplier,
                iteration=outer + 1,
                native_iteration=outer + 1,
                elapsed=perf_counter() - start,
                beta=trial_beta,
                primal_solves=primal_solves,
                correction_solves=correction_solves,
                backtracks=int(inner),
                model_ratio=(
                    float(ratio) if np.isfinite(ratio) else None
                ),
                step_norm=float(np.linalg.norm(step)),
                correction_norm=float(np.linalg.norm(correction)),
                solve_relative_residual=float(solve_residual),
                correction_relative_residual=(
                    float(correction_residual) if use_soc else None
                ),
            )
        )

    return _finish(
        method=method,
        status="iteration_limit",
        message="accepted-iteration limit reached",
        config=config,
        trace=trace,
        x=x,
        multiplier=multiplier,
        primal_solves=primal_solves,
        correction_solves=correction_solves,
        refinement_solves=refinement_solves,
        rejected_trials=rejected_trials,
        metadata={
            "linear_solver": "matrix_free_woodbury",
            "max_primal_linear_residual": max_primal_residual,
            "max_pre_refinement_linear_residual": (
                max_pre_refinement_residual
            ),
            "max_correction_linear_residual": max_correction_residual,
            "max_reduced_condition": max_reduced_condition,
            "max_correction_condition": max_correction_condition,
        },
    )


def solve_lal(
    problem: SparseConstrainedLogistic,
    config: DynamicBetaConfig,
    x0: Array,
    lambda0: Array,
) -> SolverRun:
    method: Method = "l_al"
    x = np.asarray(x0, dtype=np.float64).copy()
    multiplier = np.asarray(lambda0, dtype=np.float64).copy()
    problem.check_shapes(x)
    objective = problem.objective(x)
    constraint = problem.constraint(x)
    previous_step = np.zeros_like(x)
    beta_previous = config.beta_initial
    beta_candidate = config.beta_initial
    current_potential = _augmented(
        objective, constraint, multiplier, config.rho
    )
    primal_solves = 0
    refinement_solves = 0
    rejected_trials = 0
    max_primal_residual = 0.0
    max_pre_refinement_residual = 0.0
    max_reduced_condition = 0.0
    start = perf_counter()
    trace = [
        independent_record(
            problem,
            x,
            multiplier,
            iteration=0,
            native_iteration=0,
            elapsed=0.0,
            beta=None,
            primal_solves=0,
            correction_solves=0,
            backtracks=0,
            model_ratio=None,
            step_norm=None,
            correction_norm=0.0,
            solve_relative_residual=None,
            correction_relative_residual=None,
        )
    ]

    for outer in range(config.max_iterations):
        if (
            float(trace[-1]["pair_residual_squared"])
            <= config.target_residual_squared
        ):
            return _finish(
                method=method,
                status="target_reached",
                message="exact pair-residual target reached",
                config=config,
                trace=trace,
                x=x,
                multiplier=multiplier,
                primal_solves=primal_solves,
                correction_solves=0,
                refinement_solves=refinement_solves,
                rejected_trials=rejected_trials,
                metadata={
                    "linear_solver": "matrix_free_woodbury",
                    "beta_policy": "published_algorithm_1",
                    "max_primal_linear_residual": max_primal_residual,
                    "max_pre_refinement_linear_residual": (
                        max_pre_refinement_residual
                    ),
                    "max_reduced_condition": max_reduced_condition,
                },
            )
        gradient = problem.gradient(x)
        rhs = -gradient - problem.adjoint_action(
            x, multiplier + config.rho * constraint
        )
        trial_beta = max(beta_candidate, config.beta_floor)
        accepted_data: tuple[object, ...] | None = None

        for inner in range(config.max_backtracks + 1):
            if trial_beta > config.beta_ceiling:
                break
            try:
                (
                    step,
                    solve_residual,
                    reduced_condition,
                    pre_refinement_residual,
                    refinement_count,
                ) = _refined_reduced_step(
                    problem,
                    x,
                    rhs,
                    beta=trial_beta,
                    rho=config.rho,
                    required_residual=(
                        config.required_linear_residual
                    ),
                    maximum_refinements=(
                        config.maximum_linear_refinements
                    ),
                )
            except np.linalg.LinAlgError as error:
                return _finish(
                    method=method,
                    status="linear_solve_failure",
                    message=str(error),
                    config=config,
                    trace=trace,
                    x=x,
                    multiplier=multiplier,
                    primal_solves=primal_solves,
                    correction_solves=0,
                    refinement_solves=refinement_solves,
                    rejected_trials=rejected_trials,
                    metadata={"linear_solver": "matrix_free_woodbury"},
                )
            primal_solves += 1 + refinement_count
            refinement_solves += refinement_count
            max_primal_residual = max(
                max_primal_residual, solve_residual
            )
            max_pre_refinement_residual = max(
                max_pre_refinement_residual,
                pre_refinement_residual,
            )
            max_reduced_condition = max(
                max_reduced_condition, reduced_condition
            )
            if solve_residual > config.required_linear_residual:
                return _finish(
                    method=method,
                    status="linear_residual_failure",
                    message=(
                        f"primal relative residual {solve_residual:.3e}"
                    ),
                    config=config,
                    trace=trace,
                    x=x,
                    multiplier=multiplier,
                    primal_solves=primal_solves,
                    correction_solves=0,
                    refinement_solves=refinement_solves,
                    rejected_trials=rejected_trials,
                    metadata={"linear_solver": "matrix_free_woodbury"},
                )
            linearized = (
                constraint + problem.jacobian_action(x, step)
            )
            multiplier_trial = multiplier + config.rho * linearized
            x_trial = x + step
            objective_trial = problem.objective(x_trial)
            constraint_trial = problem.constraint(x_trial)
            potential_trial = _augmented(
                objective_trial,
                constraint_trial,
                multiplier_trial,
                config.rho,
            ) + 0.25 * trial_beta * float(step @ step)
            delta_multiplier = multiplier_trial - multiplier
            right_side = (
                1.5
                / config.rho
                * float(delta_multiplier @ delta_multiplier)
                - 0.25 * trial_beta * float(step @ step)
                - 0.25
                * beta_previous
                * float(previous_step @ previous_step)
            )
            scale = max(
                1.0, abs(current_potential), abs(potential_trial)
            )
            accepted = bool(
                potential_trial - current_potential
                <= right_side
                + 64.0 * np.finfo(float).eps * scale
            )
            if accepted:
                accepted_data = (
                    step,
                    x_trial,
                    multiplier_trial,
                    objective_trial,
                    constraint_trial,
                    potential_trial,
                    solve_residual,
                    inner,
                )
                break
            rejected_trials += 1
            trial_beta *= config.mu

        if accepted_data is None:
            return _finish(
                method=method,
                status="backtracking_limit",
                message=f"no acceptable beta at outer iteration {outer}",
                config=config,
                trace=trace,
                x=x,
                multiplier=multiplier,
                primal_solves=primal_solves,
                correction_solves=0,
                refinement_solves=refinement_solves,
                rejected_trials=rejected_trials,
                metadata={"linear_solver": "matrix_free_woodbury"},
            )
        (
            step,
            x,
            multiplier,
            objective,
            constraint,
            current_potential,
            solve_residual,
            inner,
        ) = accepted_data
        x = np.asarray(x, dtype=np.float64)
        multiplier = np.asarray(multiplier, dtype=np.float64)
        constraint = np.asarray(constraint, dtype=np.float64)
        previous_step = np.asarray(step, dtype=np.float64)
        beta_previous = trial_beta
        beta_candidate = max(
            trial_beta / config.mu, config.beta_floor
        )
        trace.append(
            independent_record(
                problem,
                x,
                multiplier,
                iteration=outer + 1,
                native_iteration=outer + 1,
                elapsed=perf_counter() - start,
                beta=trial_beta,
                primal_solves=primal_solves,
                correction_solves=0,
                backtracks=int(inner),
                model_ratio=None,
                step_norm=float(np.linalg.norm(step)),
                correction_norm=0.0,
                solve_relative_residual=float(solve_residual),
                correction_relative_residual=None,
            )
        )

    return _finish(
        method=method,
        status="iteration_limit",
        message="accepted-iteration limit reached",
        config=config,
        trace=trace,
        x=x,
        multiplier=multiplier,
        primal_solves=primal_solves,
        correction_solves=0,
        refinement_solves=refinement_solves,
        rejected_trials=rejected_trials,
        metadata={
            "linear_solver": "matrix_free_woodbury",
            "beta_policy": "published_algorithm_1",
            "max_primal_linear_residual": max_primal_residual,
            "max_pre_refinement_linear_residual": (
                max_pre_refinement_residual
            ),
            "max_reduced_condition": max_reduced_condition,
        },
    )
