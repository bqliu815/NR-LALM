"""Softmax finite sum with deterministic class-wise sphere equalities."""

from __future__ import annotations

import bz2
from dataclasses import dataclass, field
import hashlib
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.special import logsumexp, softmax


Array = NDArray[np.float64]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class MulticlassData:
    name: str
    features: sparse.csr_matrix
    labels: NDArray[np.int64]
    classes: tuple[int, ...]
    source_path: str
    source_sha256: str
    full_sample_count: int
    selection_seed: int | None
    per_class_limit: int | None
    raw_dimension: int
    bias_appended: bool

    @property
    def feature_dimension(self) -> int:
        return int(self.features.shape[1])

    @property
    def class_count(self) -> int:
        return len(self.classes)


def load_libsvm_multiclass(
    path: Path,
    *,
    dataset_name: str,
    expected_raw_dimension: int,
    expected_classes: int,
    expected_full_samples: int | None = None,
    expected_sha256: str | None = None,
    per_class_limit: int | None = None,
    selection_seed: int = 20260802,
    append_bias: bool = True,
) -> MulticlassData:
    """Load an official compressed LIBSVM multiclass archive without sklearn."""

    if expected_raw_dimension < 1 or expected_classes < 2:
        raise ValueError("invalid expected data dimensions")
    if per_class_limit is not None and per_class_limit < 1:
        raise ValueError("per_class_limit must be positive")
    path = path.resolve()
    actual_sha256 = sha256_file(path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(
            f"source hash mismatch: {actual_sha256} != {expected_sha256}"
        )

    labels: list[int] = []
    indptr = [0]
    indices: list[int] = []
    values: list[float] = []
    with bz2.open(path, mode="rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            fields = line.split()
            if not fields:
                continue
            labels.append(int(float(fields[0])))
            previous_column = -1
            for item in fields[1:]:
                index_text, value_text = item.split(":", maxsplit=1)
                column = int(index_text) - 1
                if not 0 <= column < expected_raw_dimension:
                    raise ValueError(
                        f"feature {column + 1} on line {line_number} is "
                        f"outside 1:{expected_raw_dimension}"
                    )
                if column <= previous_column:
                    raise ValueError(
                        f"feature indices are not increasing on line {line_number}"
                    )
                previous_column = column
                indices.append(column)
                values.append(float(value_text))
            indptr.append(len(indices))

    label_array = np.asarray(labels, dtype=np.int64)
    full_sample_count = int(label_array.size)
    if (
        expected_full_samples is not None
        and full_sample_count != expected_full_samples
    ):
        raise ValueError(
            f"expected {expected_full_samples} samples, observed {full_sample_count}"
        )
    classes = tuple(int(value) for value in np.unique(label_array))
    if len(classes) != expected_classes:
        raise ValueError(
            f"expected {expected_classes} classes, observed {classes}"
        )
    matrix = sparse.csr_matrix(
        (
            np.asarray(values, dtype=np.float64),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=(full_sample_count, expected_raw_dimension),
        dtype=np.float64,
    )
    matrix.sort_indices()

    effective_seed: int | None = None
    if per_class_limit is not None:
        rng = np.random.default_rng(selection_seed)
        selected_parts: list[NDArray[np.int64]] = []
        for class_label in classes:
            class_indices = np.flatnonzero(label_array == class_label)
            if class_indices.size < per_class_limit:
                raise ValueError(
                    f"class {class_label} has only {class_indices.size} samples"
                )
            selected_parts.append(
                np.sort(
                    rng.choice(
                        class_indices, size=per_class_limit, replace=False
                    )
                )
            )
        selected = np.sort(np.concatenate(selected_parts))
        matrix = matrix[selected].tocsr()
        label_array = label_array[selected]
        effective_seed = selection_seed

    if append_bias:
        bias = sparse.csr_matrix(
            np.ones((matrix.shape[0], 1), dtype=np.float64)
        )
        matrix = sparse.hstack((matrix, bias), format="csr")
        matrix.sort_indices()

    return MulticlassData(
        name=dataset_name,
        features=matrix,
        labels=label_array,
        classes=classes,
        source_path=str(path),
        source_sha256=actual_sha256,
        full_sample_count=full_sample_count,
        selection_seed=effective_seed,
        per_class_limit=per_class_limit,
        raw_dimension=expected_raw_dimension,
        bias_appended=append_bias,
    )


@dataclass(frozen=True)
class MulticlassSphereLogistic:
    """A vector-valued softmax problem with exact block-sphere constraints."""

    name: str
    features: sparse.csr_matrix
    label_indices: NDArray[np.int64]
    class_labels: tuple[int, ...]
    evaluation_chunk_size: int = 8192
    _maximum_feature_norm: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        features = sparse.csr_matrix(self.features, dtype=np.float64)
        features.sort_indices()
        labels = np.asarray(self.label_indices, dtype=np.int64).ravel()
        if features.shape[0] != labels.size:
            raise ValueError("feature and label counts differ")
        if features.shape[1] < 1 or len(self.class_labels) < 2:
            raise ValueError("invalid problem dimensions")
        if np.any(labels < 0) or np.any(labels >= len(self.class_labels)):
            raise ValueError("label index lies outside the class range")
        if self.evaluation_chunk_size < 1:
            raise ValueError("evaluation_chunk_size must be positive")
        row_norms = np.sqrt(features.multiply(features).sum(axis=1)).A1
        if row_norms.size == 0 or not np.all(np.isfinite(row_norms)):
            raise ValueError("feature matrix is empty or nonfinite")
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "label_indices", labels)
        object.__setattr__(
            self, "_maximum_feature_norm", float(np.max(row_norms))
        )

    @classmethod
    def from_data(cls, data: MulticlassData) -> "MulticlassSphereLogistic":
        class_to_index = {
            label: index for index, label in enumerate(data.classes)
        }
        indices = np.asarray(
            [class_to_index[int(label)] for label in data.labels],
            dtype=np.int64,
        )
        return cls(
            name=data.name,
            features=data.features,
            label_indices=indices,
            class_labels=data.classes,
        )

    @property
    def feature_dimension(self) -> int:
        return int(self.features.shape[1])

    @property
    def num_classes(self) -> int:
        return len(self.class_labels)

    @property
    def n(self) -> int:
        return self.feature_dimension * self.num_classes

    @property
    def num_components(self) -> int:
        return int(self.features.shape[0])

    @property
    def num_constraints(self) -> int:
        return self.num_classes

    @property
    def equality_mask(self) -> NDArray[np.bool_]:
        return np.ones(self.num_classes, dtype=bool)

    @property
    def maximum_feature_norm(self) -> float:
        return self._maximum_feature_norm

    @property
    def objective_gradient_lipschitz_bound(self) -> float:
        return 0.5 * self.maximum_feature_norm**2

    @property
    def constraint_gradient_lipschitz_sum(self) -> float:
        return float(self.num_classes)

    def objective_gradient_bound(self) -> float:
        return float(np.sqrt(2.0) * self.maximum_feature_norm)

    def _weights(self, x: Array) -> Array:
        vector = np.asarray(x, dtype=np.float64)
        if vector.shape != (self.n,):
            raise ValueError(f"expected vector shape {(self.n,)}, got {vector.shape}")
        return vector.reshape(self.num_classes, self.feature_dimension)

    def _scores(self, matrix: sparse.csr_matrix, weights: Array) -> Array:
        return np.asarray(matrix @ weights.T, dtype=np.float64)

    def component_gradient(
        self, x: Array, indices: NDArray[np.int64]
    ) -> Array:
        indices = np.asarray(indices, dtype=np.int64).ravel()
        if indices.size == 0:
            raise ValueError("component batch is empty")
        if np.any(indices < 0) or np.any(indices >= self.num_components):
            raise ValueError("component index lies outside the data set")
        matrix = self.features[indices]
        weights = self._weights(x)
        probabilities = softmax(self._scores(matrix, weights), axis=1)
        probabilities[np.arange(indices.size), self.label_indices[indices]] -= 1.0
        gradient = np.asarray(matrix.T @ probabilities, dtype=np.float64).T
        return (gradient / indices.size).ravel()

    def component_values(
        self, x: Array, indices: NDArray[np.int64]
    ) -> Array:
        indices = np.asarray(indices, dtype=np.int64).ravel()
        matrix = self.features[indices]
        scores = self._scores(matrix, self._weights(x))
        return logsumexp(scores, axis=1) - scores[
            np.arange(indices.size), self.label_indices[indices]
        ]

    def objective_and_full_gradient(self, x: Array) -> tuple[float, Array]:
        weights = self._weights(x)
        gradient = np.zeros_like(weights)
        loss_sum = 0.0
        for start in range(0, self.num_components, self.evaluation_chunk_size):
            stop = min(start + self.evaluation_chunk_size, self.num_components)
            matrix = self.features[start:stop]
            labels = self.label_indices[start:stop]
            scores = self._scores(matrix, weights)
            loss_sum += float(
                np.sum(
                    logsumexp(scores, axis=1)
                    - scores[np.arange(stop - start), labels]
                )
            )
            probabilities = softmax(scores, axis=1)
            probabilities[np.arange(stop - start), labels] -= 1.0
            gradient += np.asarray(
                matrix.T @ probabilities, dtype=np.float64
            ).T
        scale = 1.0 / self.num_components
        return loss_sum * scale, (gradient * scale).ravel()

    def objective(self, x: Array) -> float:
        return self.objective_and_full_gradient(x)[0]

    def full_gradient(self, x: Array) -> Array:
        return self.objective_and_full_gradient(x)[1]

    def accuracy(self, x: Array) -> float:
        weights = self._weights(x)
        correct = 0
        for start in range(0, self.num_components, self.evaluation_chunk_size):
            stop = min(start + self.evaluation_chunk_size, self.num_components)
            scores = self._scores(self.features[start:stop], weights)
            correct += int(
                np.sum(np.argmax(scores, axis=1) == self.label_indices[start:stop])
            )
        return float(correct / self.num_components)

    def constraints(self, x: Array) -> Array:
        weights = self._weights(x)
        return 0.5 * (np.sum(weights * weights, axis=1) - 1.0)

    def jacobian(self, x: Array) -> Array:
        weights = self._weights(x)
        jacobian = np.zeros((self.n, self.num_classes), dtype=np.float64)
        for class_index in range(self.num_classes):
            start = class_index * self.feature_dimension
            jacobian[start : start + self.feature_dimension, class_index] = (
                weights[class_index]
            )
        return jacobian

    def jacobian_action(self, x: Array, direction: Array) -> Array:
        weights = self._weights(x)
        directions = self._weights(direction)
        return np.sum(weights * directions, axis=1)

    def adjoint_action(self, x: Array, vector: Array) -> Array:
        weights = self._weights(x)
        vector = np.asarray(vector, dtype=np.float64)
        if vector.shape != (self.num_classes,):
            raise ValueError("constraint-space vector has the wrong shape")
        return (vector[:, None] * weights).ravel()

    def gram_matrix(self, x: Array) -> Array:
        weights = self._weights(x)
        return np.diag(np.sum(weights * weights, axis=1))

    def prox(self, point: Array, eta: float) -> Array:
        del eta
        return np.asarray(point, dtype=np.float64)

    def feasible_initial_point(self, mode: str = "shared_bias_basis") -> Array:
        if mode != "shared_bias_basis":
            raise ValueError(f"unsupported initialization mode: {mode}")
        weights = np.zeros(
            (self.num_classes, self.feature_dimension), dtype=np.float64
        )
        weights[:, -1] = 1.0
        return weights.ravel()
