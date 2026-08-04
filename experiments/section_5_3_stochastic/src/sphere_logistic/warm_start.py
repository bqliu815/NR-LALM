"""Deterministic method-independent warm start used in Section 5.3."""

from __future__ import annotations

import hashlib

import numpy as np

from .problem import MulticlassSphereLogistic


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(array, dtype="<f8").tobytes(order="C")
    ).hexdigest()


def normalize_blocks(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 2:
        raise ValueError("block normalization requires a matrix")
    norms = np.linalg.norm(weights, axis=1, keepdims=True)
    if np.any(norms <= np.finfo(float).eps):
        raise ValueError("warm-start rule produced a zero class block")
    return weights / norms


def build_common_warm_start(
    problem: MulticlassSphereLogistic,
    specification: dict[str, object],
    dataset_selection_seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    subset_size = int(specification["subset_size"])
    if not 1 <= subset_size <= problem.num_components:
        raise ValueError("invalid warm-start subset size")
    subset_seed = dataset_selection_seed + int(
        specification["subset_seed_offset"]
    )
    rng = np.random.default_rng(subset_seed)
    indices = np.sort(
        rng.choice(problem.num_components, size=subset_size, replace=False)
    ).astype(np.int64, copy=False)
    problem_class = type(problem)
    subset = problem_class(
        name=f"{problem.name}-warm-start-subset",
        features=problem.features[indices],
        label_indices=problem.label_indices[indices],
        class_labels=problem.class_labels,
        evaluation_chunk_size=problem.evaluation_chunk_size,
    )

    step_size = float(specification["projected_gradient_step_size"])
    steps = int(specification["projected_gradient_steps"])
    if step_size <= 0.0 or steps < 0:
        raise ValueError("invalid projected-gradient warm-start parameters")
    bias = subset.feasible_initial_point(
        str(specification["base"])
    ).reshape(subset.num_classes, subset.feature_dimension)
    direction = -subset.full_gradient(bias.ravel()).reshape(bias.shape)
    weights = normalize_blocks(direction)
    for _ in range(steps):
        _, gradient_flat = subset.objective_and_full_gradient(weights.ravel())
        gradient = gradient_flat.reshape(weights.shape)
        tangent = gradient - np.sum(
            gradient * weights, axis=1, keepdims=True
        ) * weights
        weights = normalize_blocks(weights - step_size * tangent)
    component_cost = (steps + 1) * subset_size
    base_seed = None
    initialization_family = "independent_spheres"
    manifold_error = float(
        np.max(np.abs(np.linalg.norm(weights, axis=1) - 1.0))
    )

    radial_scale = float(specification["radial_scale"])
    if not 0.5 < radial_scale < 1.0:
        raise ValueError(
            "warm-start radial scale must be mildly infeasible and regular"
        )
    x0 = (radial_scale * weights).ravel()
    if component_cost != int(specification["component_gradient_cost"]):
        raise ValueError("declared warm-start component cost is inconsistent")
    metadata = {
        "subset_seed": subset_seed,
        "base_seed": base_seed,
        "subset_size": subset_size,
        "subset_indices_sha256": hashlib.sha256(
            np.asarray(indices, dtype="<i8").tobytes(order="C")
        ).hexdigest(),
        "projected_gradient_steps": steps,
        "projected_gradient_step_size": step_size,
        "radial_scale": radial_scale,
        "component_gradient_cost": component_cost,
        "initialization_family": initialization_family,
        "pre_scale_manifold_error": manifold_error,
        "x0_sha256": array_sha256(x0),
    }
    return x0, metadata
