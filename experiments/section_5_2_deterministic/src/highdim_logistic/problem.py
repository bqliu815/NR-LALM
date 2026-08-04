"""Sparse real-data logistic objectives with matrix-free equalities."""

from __future__ import annotations

import bz2
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.special import expit

Array = np.ndarray


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class SparseBinaryData:
    name: str
    train_features: sparse.csr_matrix
    train_labels: Array
    test_features: sparse.csr_matrix
    test_labels: Array
    source_sha256: str
    source_path: str

    @property
    def dimension(self) -> int:
        return int(self.train_features.shape[1])


def _stratified_split(
    matrix: sparse.csr_matrix,
    labels: Array,
    *,
    test_fraction: float,
    seed: int,
) -> tuple[sparse.csr_matrix, Array, sparse.csr_matrix, Array]:
    if not (0.0 < test_fraction < 1.0):
        raise ValueError("test_fraction must lie in (0,1)")
    rng = np.random.default_rng(seed)
    train_parts: list[Array] = []
    test_parts: list[Array] = []
    for label in (-1.0, 1.0):
        indices = np.flatnonzero(labels == label)
        if indices.size == 0:
            raise ValueError("both binary classes must be present")
        rng.shuffle(indices)
        test_count = max(1, int(round(test_fraction * indices.size)))
        test_parts.append(indices[:test_count])
        train_parts.append(indices[test_count:])
    train_indices = np.sort(np.concatenate(train_parts))
    test_indices = np.sort(np.concatenate(test_parts))
    return (
        matrix[train_indices].tocsr(),
        np.asarray(labels[train_indices], dtype=np.float64),
        matrix[test_indices].tocsr(),
        np.asarray(labels[test_indices], dtype=np.float64),
    )


def load_libsvm_bz2(
    path: Path,
    *,
    dataset_name: str,
    expected_dimension: int,
    split_seed: int,
    test_fraction: float = 0.2,
) -> SparseBinaryData:
    """Parse a bzip2-compressed LIBSVM binary-class file without sklearn."""

    path = path.resolve()
    labels: list[float] = []
    indptr = [0]
    indices: list[int] = []
    values: list[float] = []
    maximum_column = -1
    with bz2.open(path, mode="rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            fields = line.split()
            if not fields:
                continue
            raw_label = float(fields[0])
            labels.append(1.0 if raw_label > 0.0 else -1.0)
            previous_column = -1
            for item in fields[1:]:
                index_text, value_text = item.split(":", maxsplit=1)
                column = int(index_text) - 1
                if column < 0 or column >= expected_dimension:
                    raise ValueError(
                        f"feature index {column + 1} on line {line_number} "
                        f"is outside 1:{expected_dimension}"
                    )
                if column <= previous_column:
                    raise ValueError(
                        f"feature indices are not increasing on line "
                        f"{line_number}"
                    )
                previous_column = column
                indices.append(column)
                values.append(float(value_text))
                maximum_column = max(maximum_column, column)
            indptr.append(len(indices))
    if not labels or maximum_column < 0:
        raise ValueError(f"no samples found in {path}")
    matrix = sparse.csr_matrix(
        (
            np.asarray(values, dtype=np.float64),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=(len(labels), expected_dimension),
        dtype=np.float64,
    )
    matrix.sort_indices()
    row_norms = np.sqrt(
        np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel()
    )
    nonzero_rows = row_norms > 0.0
    inverse_norms = np.ones_like(row_norms)
    inverse_norms[nonzero_rows] = 1.0 / row_norms[nonzero_rows]
    matrix = sparse.diags(inverse_norms) @ matrix
    matrix = matrix.tocsr()
    label_array = np.asarray(labels, dtype=np.float64)
    train_x, train_y, test_x, test_y = _stratified_split(
        matrix,
        label_array,
        test_fraction=test_fraction,
        seed=split_seed,
    )
    return SparseBinaryData(
        name=dataset_name,
        train_features=train_x,
        train_labels=train_y,
        test_features=test_x,
        test_labels=test_y,
        source_sha256=sha256_file(path),
        source_path=str(path),
    )


@dataclass(frozen=True)
class SparseConstrainedLogistic:
    """Full-batch logistic loss with affine rows and one sphere equality."""

    name: str
    data: SparseBinaryData
    affine_matrix: sparse.csr_matrix
    affine_rhs: Array

    @property
    def dimension(self) -> int:
        return self.data.dimension

    @property
    def affine_constraints(self) -> int:
        return int(self.affine_matrix.shape[0])

    @property
    def constraints(self) -> int:
        return self.affine_constraints + 1

    def objective(self, x: Array) -> float:
        x = np.asarray(x, dtype=np.float64)
        margins = self.data.train_labels * (
            self.data.train_features @ x
        )
        return float(np.mean(np.logaddexp(0.0, -margins)))

    def gradient(self, x: Array) -> Array:
        x = np.asarray(x, dtype=np.float64)
        margins = self.data.train_labels * (
            self.data.train_features @ x
        )
        weights = -self.data.train_labels * expit(-margins)
        gradient = self.data.train_features.T @ weights
        return np.asarray(gradient, dtype=np.float64).ravel() / (
            self.data.train_features.shape[0]
        )

    def constraint(self, x: Array) -> Array:
        x = np.asarray(x, dtype=np.float64)
        affine = np.asarray(self.affine_matrix @ x).ravel()
        return np.concatenate(
            (
                affine - self.affine_rhs,
                np.array([0.5 * (float(x @ x) - 1.0)]),
            )
        )

    def jacobian_action(self, x: Array, p: Array) -> Array:
        x = np.asarray(x, dtype=np.float64)
        p = np.asarray(p, dtype=np.float64)
        return np.concatenate(
            (
                np.asarray(self.affine_matrix @ p).ravel(),
                np.array([float(x @ p)]),
            )
        )

    def adjoint_action(self, x: Array, y: Array) -> Array:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        affine_part = self.affine_matrix.T @ y[:-1]
        return np.asarray(affine_part, dtype=np.float64).ravel() + x * y[-1]

    def gram_matrix(self, x: Array) -> Array:
        x = np.asarray(x, dtype=np.float64)
        affine_x = np.asarray(self.affine_matrix @ x).ravel()
        gram = np.empty(
            (self.constraints, self.constraints), dtype=np.float64
        )
        gram[:-1, :-1] = np.eye(self.affine_constraints)
        gram[:-1, -1] = affine_x
        gram[-1, :-1] = affine_x
        gram[-1, -1] = float(x @ x)
        return gram

    def jacobian_structure(self) -> tuple[Array, Array]:
        matrix = self.affine_matrix
        affine_rows = np.repeat(
            np.arange(self.affine_constraints, dtype=np.int32),
            np.diff(matrix.indptr),
        )
        sphere_rows = np.full(
            self.dimension, self.affine_constraints, dtype=np.int32
        )
        sphere_columns = np.arange(self.dimension, dtype=np.int32)
        return (
            np.concatenate((affine_rows, sphere_rows)),
            np.concatenate((matrix.indices.astype(np.int32), sphere_columns)),
        )

    def jacobian_values(self, x: Array) -> Array:
        return np.concatenate(
            (
                np.asarray(self.affine_matrix.data, dtype=np.float64),
                np.asarray(x, dtype=np.float64),
            )
        )

    def accuracy(self, x: Array, *, split: str) -> float:
        if split == "train":
            matrix = self.data.train_features
            labels = self.data.train_labels
        elif split == "test":
            matrix = self.data.test_features
            labels = self.data.test_labels
        else:
            raise ValueError("split must be 'train' or 'test'")
        scores = matrix @ np.asarray(x, dtype=np.float64)
        predictions = np.where(scores >= 0.0, 1.0, -1.0)
        return float(np.mean(predictions == labels))

    def check_shapes(self, x: Array) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (self.dimension,):
            raise ValueError("x has the wrong shape")
        if self.gradient(x).shape != x.shape:
            raise ValueError("gradient has the wrong shape")
        if self.constraint(x).shape != (self.constraints,):
            raise ValueError("constraint has the wrong shape")
        probe = np.ones_like(x)
        if self.jacobian_action(x, probe).shape != (self.constraints,):
            raise ValueError("Jacobian action has the wrong shape")
        if self.adjoint_action(
            x, np.ones(self.constraints)
        ).shape != x.shape:
            raise ValueError("adjoint action has the wrong shape")


@dataclass(frozen=True)
class HighDimInstance:
    problem: SparseConstrainedLogistic
    x0: Array
    lambda0: Array
    metadata: dict[str, object]


def _sample_unique_indices(
    rng: np.random.Generator,
    *,
    population: int,
    count: int,
) -> Array:
    selected: set[int] = set()
    while len(selected) < count:
        needed = count - len(selected)
        candidates = rng.integers(
            0, population, size=max(2 * needed, 32)
        )
        selected.update(map(int, candidates))
    sampled = np.fromiter(selected, dtype=np.int64)
    rng.shuffle(sampled)
    return sampled[:count]


def dimension_adaptive_affine_shape(n: int) -> tuple[int, int]:
    """Return the prospectively frozen affine-row/support dimensions."""

    if n < 2:
        raise ValueError("feature dimension must be at least two")
    affine_constraints = min(10, n - 2)
    if affine_constraints == 0:
        return 0, 0
    support_size = min(256, (n - 1) // affine_constraints)
    if support_size <= 0:
        raise ValueError("adaptive support size is not positive")
    return affine_constraints, support_size


def make_instance(
    data: SparseBinaryData,
    *,
    seed: int,
    affine_constraints: int = 10,
    affine_support_size: int = 256,
    affine_rhs_norm: float = 0.5,
) -> HighDimInstance:
    """Create sparse orthonormal affine rows and an exact feasible start."""

    n = data.dimension
    r = affine_constraints
    if r < 0 or r >= n:
        raise ValueError("invalid affine constraint count")
    if r == 0 and affine_support_size != 0:
        raise ValueError("zero affine rows require zero support size")
    if r > 0 and (
        affine_support_size <= 0 or r * affine_support_size > n
    ):
        raise ValueError("invalid affine support size")
    if not (0.0 <= affine_rhs_norm < 1.0):
        raise ValueError("affine_rhs_norm must lie in [0,1)")
    rng = np.random.default_rng(seed)
    if r == 0:
        affine_matrix = sparse.csr_matrix((0, n), dtype=np.float64)
    else:
        selected = _sample_unique_indices(
            rng, population=n, count=r * affine_support_size
        ).reshape(r, affine_support_size)
        row_indices = np.repeat(np.arange(r), affine_support_size)
        column_indices = selected.ravel()
        signs = rng.choice(
            np.array([-1.0, 1.0]),
            size=r * affine_support_size,
        )
        values = signs / np.sqrt(float(affine_support_size))
        affine_matrix = sparse.csr_matrix(
            (values, (row_indices, column_indices)),
            shape=(r, n),
            dtype=np.float64,
        )
    affine_matrix.sort_indices()
    affine_rhs = np.zeros(r, dtype=np.float64)
    effective_rhs_norm = 0.0 if r == 0 else affine_rhs_norm
    if r > 0:
        affine_rhs[0] = effective_rhs_norm
    null_vector = rng.normal(size=n)
    null_vector -= np.asarray(
        affine_matrix.T @ (affine_matrix @ null_vector)
    ).ravel()
    null_vector /= np.linalg.norm(null_vector)
    x0 = (
        np.asarray(affine_matrix.T @ affine_rhs).ravel()
        + np.sqrt(1.0 - effective_rhs_norm**2) * null_vector
    )
    problem = SparseConstrainedLogistic(
        name=(
            f"{data.name}-n{n}-r{r}-s{seed}"
        ),
        data=data,
        affine_matrix=affine_matrix,
        affine_rhs=affine_rhs,
    )
    lambda0 = np.zeros(problem.constraints, dtype=np.float64)
    initial_constraint = problem.constraint(x0)
    gram_eigenvalues = np.linalg.eigvalsh(problem.gram_matrix(x0))
    row_norms = np.sqrt(
        np.asarray(
            data.train_features.multiply(
                data.train_features
            ).sum(axis=1)
        ).ravel()
    )
    test_row_norms = np.sqrt(
        np.asarray(
            data.test_features.multiply(
                data.test_features
            ).sum(axis=1)
        ).ravel()
    )
    nonzero_train_rows = row_norms > 0.0
    return HighDimInstance(
        problem=problem,
        x0=np.asarray(x0, dtype=np.float64),
        lambda0=lambda0,
        metadata={
            "dataset": data.name,
            "dimension": n,
            "training_samples": int(data.train_features.shape[0]),
            "test_samples": int(data.test_features.shape[0]),
            "training_nonzeros": int(data.train_features.nnz),
            "test_nonzeros": int(data.test_features.nnz),
            "affine_constraints": r,
            "total_constraints": problem.constraints,
            "affine_support_size": affine_support_size,
            "affine_rhs_norm": effective_rhs_norm,
            "seed": seed,
            "dataset_sha256": data.source_sha256,
            "dataset_source_path": data.source_path,
            "initial_feasibility_inf": float(
                np.linalg.norm(initial_constraint, ord=np.inf)
            ),
            "initial_jacobian_sigma_min": float(
                np.sqrt(gram_eigenvalues[0])
            ),
            "expected_initial_jacobian_sigma_min": float(
                1.0
                if r == 0
                else np.sqrt(1.0 - effective_rhs_norm)
            ),
            "zero_feature_rows": int(
                np.count_nonzero(~nonzero_train_rows)
                + np.count_nonzero(test_row_norms == 0.0)
            ),
            "maximum_train_nonzero_row_norm_error": float(
                np.max(np.abs(row_norms[nonzero_train_rows] - 1.0))
            ),
            "feature_preprocessing": (
                "per-sample l2 normalization of nonzero rows; "
                "official zero-feature rows retained"
            ),
            "split": "stratified 80/20 split of the official training file",
        },
    )
