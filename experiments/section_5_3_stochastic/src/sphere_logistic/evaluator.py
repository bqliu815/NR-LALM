"""Independent full-data evaluator for the multiclass sphere benchmark."""

from __future__ import annotations

import numpy as np

from .problem import Array, MulticlassSphereLogistic


def least_squares_multiplier(
    problem: MulticlassSphereLogistic,
    x: Array,
    gradient: Array,
) -> tuple[Array, float, float]:
    gram = problem.gram_matrix(x)
    rhs = -problem.jacobian_action(x, gradient)
    multiplier = np.linalg.lstsq(gram, rhs, rcond=1.0e-12)[0]
    residual = gram @ multiplier - rhs
    relative = float(np.linalg.norm(residual) / max(1.0, np.linalg.norm(rhs)))
    sigma = float(np.sqrt(max(0.0, np.linalg.eigvalsh(gram)[0])))
    return np.asarray(multiplier, dtype=float), relative, sigma


def evaluate_metrics(
    problem: MulticlassSphereLogistic,
    x: Array,
    multiplier: Array | None = None,
    counts: object | None = None,
) -> dict[str, float]:
    objective, gradient = problem.objective_and_full_gradient(x)
    constraint = problem.constraints(x)
    optimized, normal_relative, sigma = least_squares_multiplier(
        problem, x, gradient
    )
    supplied = optimized if multiplier is None else np.asarray(multiplier, dtype=float)
    stationarity = gradient + problem.adjoint_action(x, supplied)
    optimized_stationarity = gradient + problem.adjoint_action(x, optimized)
    if counts is not None:
        counts.evaluator_full_gradient_components += problem.num_components
        counts.evaluator_constraint_values += 1
        counts.evaluator_constraint_jacobians += 1
        counts.evaluator_multiplier_solves += 1
    feasibility_sq = float(constraint @ constraint)
    stationarity_sq = float(stationarity @ stationarity)
    optimized_stationarity_sq = float(
        optimized_stationarity @ optimized_stationarity
    )
    return {
        "objective": float(objective),
        "training_accuracy": problem.accuracy(x),
        "feasibility_norm": float(np.linalg.norm(constraint)),
        "feasibility_linf": float(np.max(np.abs(constraint))),
        "feasibility_sq": feasibility_sq,
        "stationarity_norm": float(np.linalg.norm(stationarity)),
        "stationarity_linf": float(np.max(np.abs(stationarity))),
        "stationarity_sq": stationarity_sq,
        "pair_residual_sq": stationarity_sq + feasibility_sq,
        "optimized_stationarity_norm": float(
            np.linalg.norm(optimized_stationarity)
        ),
        "optimized_stationarity_linf": float(
            np.max(np.abs(optimized_stationarity))
        ),
        "optimized_stationarity_sq": optimized_stationarity_sq,
        "optimized_pair_residual_sq": (
            optimized_stationarity_sq + feasibility_sq
        ),
        "optimized_multiplier_norm": float(np.linalg.norm(optimized)),
        "optimized_normal_relative_residual": normal_relative,
        "jacobian_sigma_min": sigma,
    }
