from __future__ import annotations

import numpy as np
from scipy import sparse

from sphere_logistic import (
    MulticlassSphereLogistic,
    SSQPConfig,
    evaluate_metrics,
    run_ssqp,
)
from stochastic_lalm import (
    MLALMConfig,
    NRLALMConfig,
    SPIDERConfig,
    run_mlalm,
    run_nr_lalm,
)


def make_problem(seed: int = 9) -> MulticlassSphereLogistic:
    rng = np.random.default_rng(seed)
    samples = 48
    features = np.column_stack((rng.normal(size=(samples, 5)), np.ones(samples)))
    labels = np.arange(samples, dtype=np.int64) % 3
    return MulticlassSphereLogistic(
        name="algorithm-toy",
        features=sparse.csr_matrix(features),
        label_indices=labels,
        class_labels=(0, 1, 2),
        evaluation_chunk_size=16,
    )


def test_s_sqp_smoke_and_counter() -> None:
    problem = make_problem()
    result = run_ssqp(
        problem,
        problem.feasible_initial_point(),
        SSQPConfig(
            iterations=8,
            batch_size=12,
            seed=17,
            record_every=2,
        ),
    )
    assert result.status == "completed"
    assert result.counts.objective_component_gradients == 8 * 12
    assert result.linear_solves == 8
    assert result.max_linear_relative_residual < 1.0e-12
    assert len(result.trace) == 5
    assert all(np.isfinite(row["objective"]) for row in result.trace)


def test_nr_pair_has_identical_sample_stream_and_soc_executes() -> None:
    problem = make_problem()
    x0 = problem.feasible_initial_point()
    lambda0 = np.zeros(problem.num_constraints)
    spider = SPIDERConfig(
        checkpoint_batch=8,
        period=3,
        difference_batch=4,
        projection_radius=problem.objective_gradient_bound(),
        seed=23,
    )
    common = dict(
        iterations=8,
        rho=1.0,
        beta=5.0,
        spider=spider,
        output_seed=29,
        evaluation_states=(0, 4, 8),
    )
    base = run_nr_lalm(problem, x0, lambda0, NRLALMConfig(**common))
    corrected = run_nr_lalm(
        problem, x0, lambda0, NRLALMConfig(**common, use_soc=True)
    )
    assert base.status == "completed"
    assert corrected.status == "completed"
    assert base.sample_stream_sha256 == corrected.sample_stream_sha256
    assert corrected.work_counts.correction_solves == 8
    assert corrected.max_correction_relative_residual < 1.0e-12
    assert all(
        np.isfinite(row["minimum_norm_pair_residual_sq"])
        for row in (*base.trace, *corrected.trace)
    )


def test_mlalm_runs_with_exact_deterministic_constraints() -> None:
    problem = make_problem()
    x0 = problem.feasible_initial_point()
    result = run_mlalm(
        problem,
        MLALMConfig(
            iterations=8,
            batch_size=6,
            eta=0.005,
            beta=1.0,
            rho=0.1,
            alpha=0.5,
            seed=31,
            x0=x0,
            lambda0=np.zeros(problem.num_constraints),
            record_every=2,
        ),
        evaluator=lambda x, multiplier, counts: evaluate_metrics(
            problem, x, multiplier, counts
        ),
    )
    assert result.counts.objective_component_gradients == 6 * (2 * 8 - 1)
    assert len(result.trace) == 5
    assert all(np.isfinite(row["objective"]) for row in result.trace)
