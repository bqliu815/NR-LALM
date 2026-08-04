from __future__ import annotations

import numpy as np
from scipy import sparse

from sphere_logistic import MulticlassSphereLogistic, evaluate_metrics


def make_problem(seed: int = 3) -> MulticlassSphereLogistic:
    rng = np.random.default_rng(seed)
    samples = 36
    raw = rng.normal(size=(samples, 4))
    features = np.column_stack((raw, np.ones(samples)))
    labels = np.arange(samples, dtype=np.int64) % 3
    return MulticlassSphereLogistic(
        name="toy",
        features=sparse.csr_matrix(features),
        label_indices=labels,
        class_labels=(0, 1, 2),
        evaluation_chunk_size=7,
    )


def test_softmax_component_gradient_matches_finite_difference() -> None:
    problem = make_problem()
    rng = np.random.default_rng(7)
    x = rng.normal(size=problem.n) / np.sqrt(problem.feature_dimension)
    direction = rng.normal(size=problem.n)
    direction /= np.linalg.norm(direction)
    indices = np.asarray([1, 4, 4, 8, 17, 21], dtype=np.int64)
    gradient = problem.component_gradient(x, indices)
    step = 1.0e-6
    plus = float(np.mean(problem.component_values(x + step * direction, indices)))
    minus = float(np.mean(problem.component_values(x - step * direction, indices)))
    finite_difference = (plus - minus) / (2.0 * step)
    assert np.isclose(gradient @ direction, finite_difference, rtol=2e-6, atol=2e-8)


def test_full_gradient_matches_all_component_gradient() -> None:
    problem = make_problem()
    x = problem.feasible_initial_point()
    indices = np.arange(problem.num_components, dtype=np.int64)
    assert np.allclose(
        problem.full_gradient(x), problem.component_gradient(x, indices), atol=1e-13
    )


def test_constraint_jacobian_adjoint_and_gram_identities() -> None:
    problem = make_problem()
    rng = np.random.default_rng(11)
    x = rng.normal(size=problem.n)
    direction = rng.normal(size=problem.n)
    vector = rng.normal(size=problem.num_constraints)
    step = 1.0e-7
    finite_difference = (
        problem.constraints(x + step * direction)
        - problem.constraints(x - step * direction)
    ) / (2.0 * step)
    action = problem.jacobian_action(x, direction)
    assert np.allclose(action, finite_difference, rtol=1e-7, atol=1e-8)
    assert np.isclose(
        action @ vector,
        direction @ problem.adjoint_action(x, vector),
        rtol=1e-13,
        atol=1e-13,
    )
    jacobian = problem.jacobian(x)
    assert np.allclose(problem.gram_matrix(x), jacobian.T @ jacobian, atol=1e-13)


def test_neutral_start_is_exactly_feasible_and_regular() -> None:
    problem = make_problem()
    x0 = problem.feasible_initial_point()
    assert np.array_equal(problem.constraints(x0), np.zeros(problem.num_constraints))
    assert np.array_equal(problem.gram_matrix(x0), np.eye(problem.num_constraints))
    assert np.isclose(
        problem.objective(x0),
        np.log(problem.num_classes),
        rtol=0.0,
        atol=5.0 * np.finfo(float).eps,
    )
    assert problem.accuracy(x0) == 1.0 / problem.num_classes


def test_independent_evaluator_uses_least_squares_multiplier() -> None:
    problem = make_problem()
    metrics = evaluate_metrics(problem, problem.feasible_initial_point())
    assert metrics["feasibility_norm"] == 0.0
    assert metrics["jacobian_sigma_min"] == 1.0
    assert metrics["optimized_normal_relative_residual"] < 1.0e-14
    assert np.isfinite(metrics["optimized_pair_residual_sq"])
