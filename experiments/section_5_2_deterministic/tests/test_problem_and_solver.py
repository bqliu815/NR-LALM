from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import sparse

from highdim_logistic.ipopt import IpoptConfig
from highdim_logistic.problem import (
    SparseBinaryData,
    dimension_adaptive_affine_shape,
    make_instance,
)
from highdim_logistic.solver import (
    DynamicBetaConfig,
    _reduced_step,
    solve_nr_lalm,
)


def test_ipopt_default_matches_paper_backend() -> None:
    assert IpoptConfig().linear_solver == "pardisomkl"


def test_paper_configuration_uses_pardisomkl() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "paper_stage_b_v2.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["ipopt"]["linear_solver"] == "pardisomkl"


def toy_data(seed: int = 1, dimension: int = 80) -> SparseBinaryData:
    rng = np.random.default_rng(seed)
    train = sparse.random(
        100,
        dimension,
        density=0.15,
        random_state=rng,
        data_rvs=lambda count: rng.normal(size=count),
        format="csr",
    )
    test = sparse.random(
        40,
        dimension,
        density=0.15,
        random_state=rng,
        data_rvs=lambda count: rng.normal(size=count),
        format="csr",
    )
    for matrix in (train, test):
        norms = np.sqrt(
            np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel()
        )
        inverse_norms = np.ones_like(norms)
        inverse_norms[norms > 0.0] = 1.0 / norms[norms > 0.0]
        matrix.data *= np.repeat(inverse_norms, np.diff(matrix.indptr))
    return SparseBinaryData(
        name="toy",
        train_features=train,
        train_labels=np.where(
            rng.normal(size=train.shape[0]) >= 0.0, 1.0, -1.0
        ),
        test_features=test,
        test_labels=np.where(
            rng.normal(size=test.shape[0]) >= 0.0, 1.0, -1.0
        ),
        source_sha256="toy",
        source_path="toy",
    )


def instance():
    return make_instance(
        toy_data(),
        seed=4,
        affine_constraints=4,
        affine_support_size=5,
        affine_rhs_norm=0.5,
    )


def test_derivative_and_adjoint_identities() -> None:
    item = instance()
    problem = item.problem
    rng = np.random.default_rng(5)
    x = rng.normal(size=problem.dimension)
    p = rng.normal(size=problem.dimension)
    y = rng.normal(size=problem.constraints)
    p /= np.linalg.norm(p)
    h = 1.0e-6
    objective_difference = (
        problem.objective(x + h * p)
        - problem.objective(x - h * p)
    ) / (2.0 * h)
    constraint_difference = (
        problem.constraint(x + h * p)
        - problem.constraint(x - h * p)
    ) / (2.0 * h)
    assert np.isclose(
        objective_difference,
        problem.gradient(x) @ p,
        rtol=3.0e-7,
        atol=3.0e-9,
    )
    assert np.allclose(
        constraint_difference,
        problem.jacobian_action(x, p),
        rtol=3.0e-7,
        atol=3.0e-9,
    )
    assert np.isclose(
        y @ problem.jacobian_action(x, p),
        p @ problem.adjoint_action(x, y),
        rtol=2.0e-14,
        atol=2.0e-14,
    )


def test_feasible_start_and_licq() -> None:
    item = instance()
    assert np.linalg.norm(
        item.problem.constraint(item.x0), ord=np.inf
    ) < 1.0e-13
    eigenvalues = np.linalg.eigvalsh(
        item.problem.gram_matrix(item.x0)
    )
    assert np.isclose(np.sqrt(eigenvalues[0]), 1.0 / np.sqrt(2.0))


def test_dimension_adaptive_shape_preserves_a_tangent_direction() -> None:
    assert dimension_adaptive_affine_shape(2) == (0, 0)
    assert dimension_adaptive_affine_shape(3) == (1, 2)
    assert dimension_adaptive_affine_shape(2560) == (10, 255)
    assert dimension_adaptive_affine_shape(2561) == (10, 256)
    with np.testing.assert_raises(ValueError):
        dimension_adaptive_affine_shape(1)


def test_two_dimensional_instance_uses_only_the_sphere() -> None:
    item = make_instance(
        toy_data(dimension=2),
        seed=4,
        affine_constraints=0,
        affine_support_size=0,
        affine_rhs_norm=0.5,
    )
    assert item.problem.affine_constraints == 0
    assert item.problem.constraints == 1
    assert item.problem.affine_rhs.size == 0
    assert item.metadata["affine_rhs_norm"] == 0.0
    np.testing.assert_allclose(
        item.metadata["initial_jacobian_sigma_min"], 1.0, atol=2.0e-14
    )
    assert np.linalg.norm(
        item.problem.constraint(item.x0), ord=np.inf
    ) < 1.0e-13


def test_matrix_free_woodbury_matches_explicit() -> None:
    item = instance()
    problem = item.problem
    rng = np.random.default_rng(6)
    x = rng.normal(size=problem.dimension)
    rhs = rng.normal(size=problem.dimension)
    beta = 0.7
    rho = 3.0
    step, relative, _ = _reduced_step(
        problem, x, rhs, beta=beta, rho=rho
    )
    rows, columns = problem.jacobian_structure()
    values = problem.jacobian_values(x)
    jacobian = sparse.coo_matrix(
        (values, (rows, columns)),
        shape=(problem.constraints, problem.dimension),
    ).toarray()
    direct = np.linalg.solve(
        beta * np.eye(problem.dimension)
        + rho * jacobian.T @ jacobian,
        rhs,
    )
    assert np.allclose(step, direct, rtol=8.0e-13, atol=8.0e-13)
    assert relative < 1.0e-12


def test_nr_and_soc_smoke() -> None:
    item = instance()
    config = DynamicBetaConfig(
        rho=1.0,
        beta_floor=1.0e-4,
        beta_initial=1.0,
        beta_ceiling=1.0e6,
        max_iterations=3,
        target_residual_squared=1.0e-30,
    )
    for use_soc in (False, True):
        run = solve_nr_lalm(
            item.problem,
            config,
            item.x0,
            item.lambda0,
            use_soc=use_soc,
        )
        assert run.status in {"iteration_limit", "target_reached"}
        assert len(run.trace) >= 2
        assert np.isfinite(run.trace[-1]["pair_residual_squared"])
