"""Common affine-plus-sphere equality-constrained Logistic model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.special import expit

from .mlalm import Array, OracleCounts


@dataclass(frozen=True)
class ConstrainedLogistic:
    name: str
    features: sparse.csr_matrix
    labels: NDArray[np.float64]
    affine_matrix: sparse.csr_matrix
    affine_rhs: NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.features.shape[0] != self.labels.size:
            raise ValueError("feature and label counts differ")
        if self.affine_matrix.shape[1] != self.features.shape[1]:
            raise ValueError("affine and feature dimensions differ")
        if self.affine_matrix.shape[0] != self.affine_rhs.size:
            raise ValueError("affine row and right-hand-side counts differ")
        if not np.all(np.isin(self.labels, (-1.0, 1.0))):
            raise ValueError("labels must be encoded as -1/+1")

    @property
    def n(self) -> int:
        return int(self.features.shape[1])

    @property
    def num_components(self) -> int:
        return int(self.features.shape[0])

    @property
    def num_constraints(self) -> int:
        return int(self.affine_matrix.shape[0] + 1)

    @property
    def equality_mask(self) -> NDArray[np.bool_]:
        return np.ones(self.num_constraints, dtype=bool)

    def component_gradient(
        self, x: Array, indices: NDArray[np.int64]
    ) -> Array:
        indices = np.asarray(indices, dtype=np.int64).ravel()
        if indices.size == 0:
            raise ValueError("component batch is empty")
        matrix = self.features[indices]
        labels = self.labels[indices]
        margins = labels * np.asarray(matrix @ x).ravel()
        weights = -labels * expit(-margins)
        gradient = matrix.T @ weights
        return np.asarray(gradient, dtype=float).ravel() / indices.size

    def component_values(
        self, x: Array, indices: NDArray[np.int64]
    ) -> NDArray[np.float64]:
        indices = np.asarray(indices, dtype=np.int64).ravel()
        matrix = self.features[indices]
        margins = self.labels[indices] * np.asarray(matrix @ x).ravel()
        return np.logaddexp(0.0, -margins)

    def objective(self, x: Array) -> float:
        margins = self.labels * np.asarray(self.features @ x).ravel()
        return float(np.mean(np.logaddexp(0.0, -margins)))

    def full_gradient(self, x: Array) -> Array:
        margins = self.labels * np.asarray(self.features @ x).ravel()
        weights = -self.labels * expit(-margins)
        gradient = self.features.T @ weights
        return np.asarray(gradient, dtype=float).ravel() / self.num_components

    def constraints(self, x: Array) -> Array:
        affine = np.asarray(self.affine_matrix @ x).ravel() - self.affine_rhs
        sphere = np.array([0.5 * (float(x @ x) - 1.0)])
        return np.concatenate((affine, sphere))

    def jacobian(self, x: Array) -> Array:
        """Return the dense n-by-m column-gradient Jacobian for small tests."""

        return np.column_stack((self.affine_matrix.toarray().T, x))

    def jacobian_action(self, x: Array, direction: Array) -> Array:
        return np.concatenate(
            (
                np.asarray(self.affine_matrix @ direction).ravel(),
                np.array([float(x @ direction)]),
            )
        )

    def adjoint_action(self, x: Array, vector: Array) -> Array:
        affine = self.affine_matrix.T @ np.asarray(vector[:-1], dtype=float)
        return np.asarray(affine, dtype=float).ravel() + x * float(vector[-1])

    def gram_matrix(self, x: Array) -> Array:
        affine_x = np.asarray(self.affine_matrix @ x).ravel()
        rows = self.affine_matrix.shape[0]
        gram = np.empty((rows + 1, rows + 1), dtype=float)
        gram[:-1, :-1] = (self.affine_matrix @ self.affine_matrix.T).toarray()
        gram[:-1, -1] = affine_x
        gram[-1, :-1] = affine_x
        gram[-1, -1] = float(x @ x)
        return gram

    def prox(self, point: Array, eta: float) -> Array:
        del eta
        return np.asarray(point, dtype=float)


@dataclass(frozen=True)
class LogisticInstance:
    problem: ConstrainedLogistic
    x0: Array
    lambda0: Array
    target: Array
    metadata: dict[str, object]


def _orthonormal_sparse_rows(
    n: int,
    rows: int,
    support_size: int,
    rng: np.random.Generator,
) -> sparse.csr_matrix:
    if rows < 1 or rows * support_size > n:
        raise ValueError("invalid sparse affine-row dimensions")
    columns = rng.choice(n, size=rows * support_size, replace=False)
    columns = columns.reshape(rows, support_size)
    row_indices = np.repeat(np.arange(rows), support_size)
    signs = rng.choice(np.array([-1.0, 1.0]), size=rows * support_size)
    values = signs / np.sqrt(float(support_size))
    matrix = sparse.csr_matrix(
        (values, (row_indices, columns.ravel())), shape=(rows, n), dtype=float
    )
    matrix.sort_indices()
    return matrix


def _normalized_null_vector(
    affine_matrix: sparse.csr_matrix,
    rng: np.random.Generator,
) -> Array:
    vector = rng.normal(size=affine_matrix.shape[1])
    vector -= np.asarray(
        affine_matrix.T @ (affine_matrix @ vector)
    ).ravel()
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise RuntimeError("failed to generate affine-null vector")
    return vector / norm


def make_synthetic_logistic(
    *,
    n: int,
    samples: int,
    affine_constraints: int,
    affine_support_size: int,
    affine_rhs_norm: float,
    label_noise: float,
    seed: int,
    initial_scale: float = 1.0,
    feature_support_size: int | None = None,
) -> LogisticInstance:
    """Generate one frozen, row-normalized stochastic Logistic instance."""

    if not 0.0 <= label_noise:
        raise ValueError("label_noise must be nonnegative")
    if not 0.0 <= affine_rhs_norm < 1.0:
        raise ValueError("affine_rhs_norm must lie in [0,1)")
    if not 0.0 < initial_scale <= 1.0:
        raise ValueError("initial_scale must lie in (0,1]")
    if feature_support_size is not None and not 1 <= feature_support_size <= n:
        raise ValueError("feature_support_size must lie in 1:n")
    rng = np.random.default_rng(seed)
    affine_matrix = _orthonormal_sparse_rows(
        n, affine_constraints, affine_support_size, rng
    )
    affine_rhs = np.zeros(affine_constraints, dtype=float)
    affine_rhs[0] = affine_rhs_norm
    affine_part = np.asarray(affine_matrix.T @ affine_rhs).ravel()
    null_scale = np.sqrt(1.0 - affine_rhs_norm**2)
    start_null = _normalized_null_vector(affine_matrix, rng)
    target_null = _normalized_null_vector(affine_matrix, rng)
    target_null -= float(target_null @ start_null) * start_null
    target_null /= np.linalg.norm(target_null)
    feasible_start = affine_part + null_scale * start_null
    x0 = initial_scale * feasible_start
    target = affine_part + null_scale * target_null

    if feature_support_size is None or feature_support_size == n:
        dense_features = rng.normal(size=(samples, n))
        row_norms = np.linalg.norm(dense_features, axis=1)
        dense_features /= row_norms[:, None]
        features = sparse.csr_matrix(dense_features)
    else:
        support = int(feature_support_size)
        columns = np.empty(samples * support, dtype=np.int32)
        for row in range(samples):
            columns[row * support : (row + 1) * support] = np.sort(
                rng.choice(n, size=support, replace=False)
            )
        values = rng.normal(size=samples * support)
        values = values.reshape(samples, support)
        values /= np.linalg.norm(values, axis=1)[:, None]
        features = sparse.csr_matrix(
            (
                values.ravel(),
                columns,
                np.arange(0, (samples + 1) * support, support, dtype=np.int64),
            ),
            shape=(samples, n),
            dtype=float,
        )
        features.sort_indices()
    row_norms = np.sqrt(features.multiply(features).sum(axis=1)).A1
    scores = np.asarray(features @ target).ravel() + label_noise * rng.normal(
        size=samples
    )
    labels = np.where(scores >= 0.0, 1.0, -1.0)
    problem = ConstrainedLogistic(
        name=f"synthetic-logistic-n{n}-N{samples}-noise{label_noise:g}-s{seed}",
        features=features,
        labels=np.asarray(labels, dtype=float),
        affine_matrix=affine_matrix,
        affine_rhs=affine_rhs,
    )
    gram_eigenvalues = np.linalg.eigvalsh(problem.gram_matrix(x0))
    return LogisticInstance(
        problem=problem,
        x0=x0,
        lambda0=np.zeros(problem.num_constraints, dtype=float),
        target=target,
        metadata={
            "seed": seed,
            "dimension": n,
            "samples": samples,
            "label_noise": label_noise,
            "affine_constraints": affine_constraints,
            "affine_support_size": affine_support_size,
            "affine_rhs_norm": affine_rhs_norm,
            "initial_scale": initial_scale,
            "feature_support_size": (
                n if feature_support_size is None else feature_support_size
            ),
            "feature_nonzeros": int(features.nnz),
            "initial_feasibility": float(np.linalg.norm(problem.constraints(x0))),
            "initial_sigma_min": float(np.sqrt(gram_eigenvalues[0])),
            "maximum_row_norm_error": float(
                np.max(np.abs(row_norms - 1.0))
            ),
        },
    )


def evaluate_pair(
    problem: ConstrainedLogistic,
    x: Array,
    multiplier: Array,
    counts: OracleCounts | None = None,
) -> dict[str, float]:
    gradient = problem.full_gradient(x)
    constraint = problem.constraints(x)
    stationarity = gradient + problem.adjoint_action(x, multiplier)
    gram = problem.gram_matrix(x)
    normal_rhs = -problem.jacobian_action(x, gradient)
    minimum_norm_multiplier = np.linalg.lstsq(
        gram, normal_rhs, rcond=1.0e-12
    )[0]
    minimum_norm_sigma = float(
        np.sqrt(max(0.0, np.linalg.eigvalsh(gram)[0]))
    )
    normal_residual = gram @ minimum_norm_multiplier - normal_rhs
    minimum_norm_stationarity = gradient + problem.adjoint_action(
        x, minimum_norm_multiplier
    )
    if counts is not None:
        counts.evaluator_full_gradient_components += problem.num_components
        counts.evaluator_constraint_values += 1
        counts.evaluator_constraint_jacobians += 1
        counts.evaluator_multiplier_solves += 1
    stationarity_sq = float(stationarity @ stationarity)
    feasibility_sq = float(constraint @ constraint)
    minimum_norm_stationarity_sq = float(
        minimum_norm_stationarity @ minimum_norm_stationarity
    )
    return {
        "objective": problem.objective(x),
        "stationarity_sq": stationarity_sq,
        "feasibility_sq": feasibility_sq,
        "pair_residual_sq": stationarity_sq + feasibility_sq,
        "multiplier_norm": float(np.linalg.norm(multiplier)),
        "minimum_norm_stationarity_sq": minimum_norm_stationarity_sq,
        "minimum_norm_pair_residual_sq": (
            minimum_norm_stationarity_sq + feasibility_sq
        ),
        "minimum_norm_multiplier_norm": float(
            np.linalg.norm(minimum_norm_multiplier)
        ),
        "minimum_norm_normal_relative_residual": float(
            np.linalg.norm(normal_residual) / max(1.0, np.linalg.norm(normal_rhs))
        ),
        "minimum_norm_sigma": minimum_norm_sigma,
    }
