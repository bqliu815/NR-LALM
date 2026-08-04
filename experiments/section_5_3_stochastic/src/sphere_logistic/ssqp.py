"""Equation-level S-SQP baseline for deterministic equality constraints."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from time import perf_counter
from typing import Callable

import numpy as np

from stochastic_lalm.mlalm import OracleCounts

from .evaluator import evaluate_metrics
from .problem import Array, MulticlassSphereLogistic


@dataclass(frozen=True)
class SSQPConfig:
    iterations: int
    batch_size: int = 1024
    seed: int = 1
    tau_initial: float = 1.0
    epsilon: float = 1.0e-6
    sigma: float = 0.5
    xi_initial: float = 1.0
    beta: float = 1.0
    theta: float = 10.0
    objective_lipschitz: float | None = None
    constraint_lipschitz_sum: float | None = None
    record_every: int = 1
    sample_without_replacement: bool = True
    required_linear_residual: float = 1.0e-10

    def validate(self, num_components: int) -> None:
        if self.iterations < 1 or self.batch_size < 1:
            raise ValueError("iterations and batch_size must be positive")
        if self.sample_without_replacement and self.batch_size > num_components:
            raise ValueError("without-replacement batch exceeds the data set")
        if self.tau_initial <= 0.0 or self.xi_initial <= 0.0:
            raise ValueError("tau_initial and xi_initial must be positive")
        if not 0.0 < self.epsilon < 1.0:
            raise ValueError("epsilon must lie in (0,1)")
        if not 0.0 < self.sigma < 1.0:
            raise ValueError("sigma must lie in (0,1)")
        if not 0.0 < self.beta <= 1.0 or self.theta < 0.0:
            raise ValueError("beta must lie in (0,1] and theta be nonnegative")
        if self.record_every < 1 or self.required_linear_residual <= 0.0:
            raise ValueError("invalid recording or residual tolerance")


@dataclass
class SSQPResult:
    method: str
    status: str
    message: str
    x: Array
    multiplier: Array
    counts: OracleCounts
    linear_solves: int
    trace: list[dict[str, float]] = field(default_factory=list)
    sample_stream_sha256: str = ""
    max_linear_relative_residual: float = 0.0
    wall_algorithm_seconds: float = 0.0
    wall_evaluator_seconds: float = 0.0


def _identity_sqp_direction(
    problem: MulticlassSphereLogistic,
    x: Array,
    gradient: Array,
    constraint: Array,
) -> tuple[Array, Array, float]:
    gram = problem.gram_matrix(x)
    rhs = constraint - problem.jacobian_action(x, gradient)
    multiplier = np.linalg.solve(gram, rhs)
    direction = -gradient - problem.adjoint_action(x, multiplier)
    primal_residual = (
        direction + problem.adjoint_action(x, multiplier) + gradient
    )
    constraint_residual = problem.jacobian_action(x, direction) + constraint
    residual = np.concatenate((primal_residual, constraint_residual))
    scale = max(
        1.0,
        float(np.linalg.norm(gradient)),
        float(np.linalg.norm(constraint)),
    )
    return direction, multiplier, float(np.linalg.norm(residual) / scale)


def run_ssqp(
    problem: MulticlassSphereLogistic,
    x0: Array,
    config: SSQPConfig,
    evaluator: Callable[
        [MulticlassSphereLogistic, Array, Array | None, object | None],
        dict[str, float],
    ] = evaluate_metrics,
) -> SSQPResult:
    """Run Algorithm 3.1 of Berahas et al. with H_k=I exactly."""

    config.validate(problem.num_components)
    x = np.asarray(x0, dtype=float).copy()
    if x.shape != (problem.n,):
        raise ValueError("x0 has the wrong shape")
    counts = OracleCounts()
    rng = np.random.default_rng(config.seed)
    stream_hash = hashlib.sha256()
    tau = float(config.tau_initial)
    xi = float(config.xi_initial)
    objective_lipschitz = float(
        problem.objective_gradient_lipschitz_bound
        if config.objective_lipschitz is None
        else config.objective_lipschitz
    )
    constraint_lipschitz_sum = float(
        problem.constraint_gradient_lipschitz_sum
        if config.constraint_lipschitz_sum is None
        else config.constraint_lipschitz_sum
    )
    if objective_lipschitz <= 0.0 or constraint_lipschitz_sum <= 0.0:
        raise ValueError("Lipschitz inputs must be positive")

    trace: list[dict[str, float]] = []
    evaluator_seconds = 0.0
    linear_solves = 0
    maximum_residual = 0.0
    multiplier = np.zeros(problem.num_constraints, dtype=float)
    status = "completed"
    message = "fixed-horizon run completed"
    start_time = perf_counter()

    def record(iteration: int) -> None:
        nonlocal evaluator_seconds
        before = perf_counter()
        metrics = evaluator(problem, x, None, counts)
        evaluator_seconds += perf_counter() - before
        metrics.update(
            iteration=float(iteration),
            component_calls=float(counts.objective_component_gradients),
            tau=tau,
            xi=xi,
            linear_solves=float(linear_solves),
        )
        trace.append(metrics)

    record(0)
    for iteration in range(config.iterations):
        indices = rng.choice(
            problem.num_components,
            size=config.batch_size,
            replace=not config.sample_without_replacement,
        ).astype(np.int64, copy=False)
        stream_hash.update(
            np.asarray(
                [iteration, config.batch_size, int(config.sample_without_replacement)],
                dtype="<i8",
            ).tobytes()
        )
        stream_hash.update(np.asarray(indices, dtype="<i8").tobytes())
        gradient = problem.component_gradient(x, indices)
        counts.objective_component_gradients += config.batch_size
        constraint = problem.constraints(x)
        counts.constraint_values += 1
        counts.constraint_jacobians += 1
        try:
            direction, multiplier, relative = _identity_sqp_direction(
                problem, x, gradient, constraint
            )
        except np.linalg.LinAlgError as error:
            status = "linear_solve_failure"
            message = str(error)
            break
        linear_solves += 1
        maximum_residual = max(maximum_residual, relative)
        if relative > config.required_linear_residual:
            status = "linear_residual_failure"
            message = f"relative S-SQP residual {relative:.3e}"
            break

        direction_sq = float(direction @ direction)
        if direction_sq <= np.finfo(float).tiny:
            if (iteration + 1) % config.record_every == 0:
                record(iteration + 1)
            continue
        constraint_l1 = float(np.linalg.norm(constraint, ord=1))
        directional = float(gradient @ direction)
        trial_denominator = directional + max(direction_sq, 0.0)
        if trial_denominator <= 0.0:
            tau_trial = float("inf")
        else:
            tau_trial = (
                (1.0 - config.sigma) * constraint_l1 / trial_denominator
            )
        if tau > tau_trial:
            tau = (1.0 - config.epsilon) * tau_trial
        model_reduction = -tau * (
            directional + 0.5 * max(direction_sq, 0.0)
        ) + constraint_l1
        xi_trial = model_reduction / (tau * direction_sq)
        if not np.isfinite(xi_trial) or xi_trial <= 0.0:
            status = "nonpositive_model_ratio"
            message = f"invalid xi trial {xi_trial:.16g}"
            break
        if xi > xi_trial:
            xi = (1.0 - config.epsilon) * xi_trial

        denominator = (
            tau * objective_lipschitz + constraint_lipschitz_sum
        ) * direction_sq
        alpha_hat_initial = config.beta * model_reduction / denominator
        alpha_tilde_initial = (
            alpha_hat_initial - 4.0 * constraint_l1 / denominator
        )
        lower = (
            config.beta
            * xi
            * tau
            / (tau * objective_lipschitz + constraint_lipschitz_sum)
        )
        upper = lower + config.theta * config.beta**2
        alpha_hat = float(np.clip(alpha_hat_initial, lower, upper))
        alpha_tilde = float(np.clip(alpha_tilde_initial, lower, upper))
        if alpha_hat < 1.0:
            alpha = alpha_hat
        elif alpha_tilde <= 1.0 <= alpha_hat:
            alpha = 1.0
        else:
            alpha = alpha_tilde
        if not np.isfinite(alpha) or alpha <= 0.0:
            status = "invalid_stepsize"
            message = f"invalid alpha {alpha:.16g}"
            break
        x = x + alpha * direction
        if not np.all(np.isfinite(x)):
            status = "nonfinite_iterate"
            message = "S-SQP produced a nonfinite iterate"
            break
        if (
            (iteration + 1) % config.record_every == 0
            or iteration + 1 == config.iterations
        ):
            record(iteration + 1)

    total_seconds = perf_counter() - start_time
    return SSQPResult(
        method="S-SQP",
        status=status,
        message=message,
        x=x,
        multiplier=np.asarray(multiplier, dtype=float),
        counts=counts,
        linear_solves=linear_solves,
        trace=trace,
        sample_stream_sha256=stream_hash.hexdigest(),
        max_linear_relative_residual=maximum_residual,
        wall_algorithm_seconds=max(0.0, total_seconds - evaluator_seconds),
        wall_evaluator_seconds=evaluator_seconds,
    )
