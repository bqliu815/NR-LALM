#!/usr/bin/env python3
"""Shared runner for the four Section 5.3 methods."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR / "src"))

import numpy as np  # noqa: E402

from sphere_logistic import (  # noqa: E402
    MulticlassSphereLogistic,
    SSQPConfig,
    evaluate_metrics,
    load_libsvm_multiclass,
    run_ssqp,
)
from stochastic_lalm import (  # noqa: E402
    MLALMConfig,
    NRLALMConfig,
    SPIDERConfig,
    largest_horizon_for_budget,
    largest_horizon_for_scaled_budget,
    run_mlalm,
    run_nr_lalm,
    spider_total_calls,
    spider_total_calls_scaled,
)


METHODS = ("NR-LALM", "NR-LALM+SOC", "MLALM", "S-SQP")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(array, dtype="<f8").tobytes(order="C")
    ).hexdigest()


def array_assignment(
    array_index: int, dataset_count: int, repeat_count: int
) -> tuple[int, int]:
    if not 0 <= array_index < dataset_count * repeat_count:
        raise ValueError("array index is outside the formal task range")
    return divmod(array_index, repeat_count)


def curve_states(iterations: int, curve_points: int) -> tuple[int, ...]:
    if iterations < 1 or curve_points < 2:
        raise ValueError("invalid formal curve dimensions")
    states = {
        int(round(index * iterations / (curve_points - 1)))
        for index in range(curve_points)
    }
    states.update((0, iterations))
    return tuple(sorted(states))


def load_problem(config: dict[str, object], dataset_index: int):
    specification = config["datasets"][dataset_index]
    path = Path(specification["path"])
    if not path.is_absolute():
        path = EXPERIMENT_DIR / path
    data = load_libsvm_multiclass(
        path,
        dataset_name=str(specification["name"]),
        expected_raw_dimension=int(specification["expected_raw_dimension"]),
        expected_classes=int(specification["expected_classes"]),
        expected_full_samples=int(specification["expected_full_samples"]),
        expected_sha256=str(specification["source_sha256"]),
        per_class_limit=None,
        append_bias=True,
    )
    if config.get("constraint_model") != "independent_spheres":
        raise ValueError("Section 5.3 requires independent sphere constraints")
    return data, MulticlassSphereLogistic.from_data(data)


def metrics_are_finite(metrics: dict[str, float]) -> bool:
    return all(math.isfinite(float(value)) for value in metrics.values())


def normalize_trace(trace: list[dict[str, float]]) -> list[dict[str, float]]:
    normalized: list[dict[str, float]] = []
    for row in trace:
        item = {key: float(value) for key, value in row.items()}
        if "feasibility_norm" not in item and "feasibility_sq" in item:
            item["feasibility_norm"] = math.sqrt(max(0.0, item["feasibility_sq"]))
        if (
            "optimized_stationarity_norm" not in item
            and "minimum_norm_stationarity_sq" in item
        ):
            item["optimized_stationarity_norm"] = math.sqrt(
                max(0.0, item["minimum_norm_stationarity_sq"])
            )
        if "optimized_pair_residual_sq" not in item:
            if "minimum_norm_pair_residual_sq" in item:
                item["optimized_pair_residual_sq"] = item[
                    "minimum_norm_pair_residual_sq"
                ]
            elif "optimized_stationarity_norm" in item:
                item["optimized_pair_residual_sq"] = (
                    item["optimized_stationarity_norm"] ** 2
                    + item["feasibility_norm"] ** 2
                )
        if (
            "optimized_stationarity_norm" not in item
            and "optimized_pair_residual_sq" in item
            and "feasibility_sq" in item
        ):
            item["optimized_stationarity_norm"] = math.sqrt(
                max(
                    0.0,
                    item["optimized_pair_residual_sq"] - item["feasibility_sq"],
                )
            )
        normalized.append(item)
    return normalized


def base_record(
    *,
    config: dict[str, object],
    config_sha256: str,
    dataset_index: int,
    repeat: int,
    method: str,
    data,
    problem: MulticlassSphereLogistic,
    x0: np.ndarray,
    initial_metrics: dict[str, float],
    stream_seed: int,
) -> dict[str, object]:
    task_index = (dataset_index * int(config["oracle_repeats"]) + repeat) * len(
        METHODS
    ) + METHODS.index(method)
    return {
        "protocol": config["protocol"],
        "protocol_status": config["protocol_status"],
        "config_sha256": config_sha256,
        "task": {
            "task_index": task_index,
            "dataset_index": dataset_index,
            "dataset_name": data.name,
            "repeat": repeat,
            "method": method,
        },
        "stream_seed": stream_seed,
        "target_component_calls": int(config["target_component_calls"]),
        "data": {
            "source_path": data.source_path,
            "source_sha256": data.source_sha256,
            "full_sample_count": data.full_sample_count,
            "optimization_sample_count": problem.num_components,
            "feature_dimension": problem.feature_dimension,
            "class_count": problem.num_classes,
            "variable_dimension": problem.n,
            "constraint_count": problem.num_constraints,
        },
        "initial_x_sha256": array_sha256(x0),
        "initial_metrics": initial_metrics,
    }


def run_method(
    *,
    config: dict[str, object],
    config_sha256: str,
    dataset_index: int,
    repeat: int,
    method: str,
    data,
    problem: MulticlassSphereLogistic,
    x0: np.ndarray,
    lambda0: np.ndarray,
    initial_metrics: dict[str, float],
) -> dict[str, object]:
    stream_seed = int(config["oracle_seed_base"]) + 10_000 * dataset_index + repeat
    record = base_record(
        config=config,
        config_sha256=config_sha256,
        dataset_index=dataset_index,
        repeat=repeat,
        method=method,
        data=data,
        problem=problem,
        x0=x0,
        initial_metrics=initial_metrics,
        stream_seed=stream_seed,
    )
    target_calls = int(config["target_component_calls"])
    curve_points = int(config["curve_points"])

    if method in ("NR-LALM", "NR-LALM+SOC"):
        candidate = config["proposed"]
        difference_batch_scale = int(config["proposed_batch_scale"])
        checkpoint_batch_scale = int(config.get("proposed_checkpoint_scale", 1))
        if checkpoint_batch_scale == 1:
            horizon = largest_horizon_for_budget(
                target_calls, difference_batch_scale
            )
        else:
            horizon = largest_horizon_for_scaled_budget(
                target_calls,
                checkpoint_batch_scale=checkpoint_batch_scale,
                difference_batch_scale=difference_batch_scale,
            )
        period = int(np.ceil(np.sqrt(horizon)))

        def evaluator(point, dual, counts):
            return evaluate_metrics(problem, point, dual, counts)

        result = run_nr_lalm(
            problem,
            x0,
            lambda0,
            NRLALMConfig(
                iterations=horizon,
                rho=float(candidate["rho"]),
                beta=float(candidate["beta"]),
                spider=SPIDERConfig(
                    checkpoint_batch=checkpoint_batch_scale * horizon,
                    period=period,
                    difference_batch=difference_batch_scale * period,
                    projection_radius=problem.objective_gradient_bound(),
                    seed=stream_seed,
                ),
                output_seed=stream_seed + 1_000_000,
                use_soc=method.endswith("+SOC"),
                required_linear_residual=float(config["required_linear_residual"]),
                evaluation_states=curve_states(horizon, curve_points),
            ),
            evaluator=evaluator,
        )
        endpoint = evaluate_metrics(problem, result.x, result.multiplier)
        expected_calls = spider_total_calls_scaled(
            horizon,
            checkpoint_batch_scale=checkpoint_batch_scale,
            difference_batch_scale=difference_batch_scale,
        )
        record["output"] = {
            "status": result.status,
            "message": result.message,
            "candidate": candidate,
            "effective_iterations": horizon,
            "effective_spider_schedule": {
                "checkpoint_batch_scale": checkpoint_batch_scale,
                "difference_batch_scale": difference_batch_scale,
                "checkpoint_batch": checkpoint_batch_scale * horizon,
                "period": period,
                "difference_batch": difference_batch_scale * period,
            },
            "expected_component_calls": expected_calls,
            "actual_component_calls": result.oracle_counts.objective_component_gradients,
            "endpoint": endpoint,
            "trace": normalize_trace(result.trace),
            "sample_stream_sha256": result.sample_stream_sha256,
            "oracle_counts": asdict(result.oracle_counts),
            "work_counts": asdict(result.work_counts),
            "maximum_primal_relative_residual": result.max_primal_relative_residual,
            "maximum_correction_relative_residual": result.max_correction_relative_residual,
            "wall_algorithm_seconds": result.wall_algorithm_seconds,
            "wall_evaluator_seconds": result.wall_evaluator_seconds,
        }
    elif method == "MLALM":
        candidate = config["mlalm"]
        batch_size = int(config["mlalm_batch_size"])
        iterations = max(1, int((target_calls // batch_size + 1) // 2))
        root = iterations**0.25
        beta = float(candidate["beta_coefficient"]) * root
        rho = float(candidate["rho_over_beta"]) * beta
        eta = float(candidate["eta_coefficient"]) / root

        def evaluator(point, dual, counts):
            return evaluate_metrics(problem, point, dual, counts)

        result = run_mlalm(
            problem,
            MLALMConfig(
                iterations=iterations,
                batch_size=batch_size,
                eta=eta,
                beta=beta,
                rho=rho,
                alpha=float(candidate["alpha"]),
                seed=stream_seed,
                output_seed=stream_seed + 1_000_000,
                x0=x0,
                lambda0=lambda0,
                record_every=max(1, iterations // (curve_points - 1)),
                enforce_rho_lt_beta=True,
            ),
            evaluator=evaluator,
        )
        endpoint = evaluate_metrics(problem, result.x, result.multiplier)
        status = "completed" if metrics_are_finite(endpoint) else "nonfinite"
        record["output"] = {
            "status": status,
            "message": "fixed-horizon run completed" if status == "completed" else "nonfinite endpoint",
            "candidate": candidate,
            "effective_parameters": {
                "iterations": iterations,
                "batch_size": batch_size,
                "eta": eta,
                "beta": beta,
                "rho": rho,
                "alpha": float(candidate["alpha"]),
            },
            "actual_component_calls": result.counts.objective_component_gradients,
            "endpoint": endpoint,
            "trace": normalize_trace(result.trace),
            "sample_stream_sha256": result.sample_stream_sha256,
            "oracle_counts": asdict(result.counts),
            "parameter_warnings": result.parameter_warnings,
            "wall_algorithm_seconds": result.wall_algorithm_seconds,
            "wall_evaluator_seconds": result.wall_evaluator_seconds,
        }
    elif method == "S-SQP":
        specification = config["s_sqp"]
        batch_size = int(specification["batch_size"])
        iterations = max(1, target_calls // batch_size)
        result = run_ssqp(
            problem,
            x0,
            SSQPConfig(
                iterations=iterations,
                batch_size=batch_size,
                seed=stream_seed,
                tau_initial=float(specification["tau_initial"]),
                epsilon=float(specification["epsilon"]),
                sigma=float(specification["sigma"]),
                xi_initial=float(specification["xi_initial"]),
                beta=float(specification["beta"]),
                theta=float(specification["theta"]),
                record_every=max(1, iterations // (curve_points - 1)),
                sample_without_replacement=bool(specification["sample_without_replacement"]),
                required_linear_residual=float(config["required_linear_residual"]),
            ),
        )
        endpoint = evaluate_metrics(problem, result.x, result.multiplier)
        record["output"] = {
            "status": result.status,
            "message": result.message,
            "candidate": specification,
            "effective_iterations": iterations,
            "actual_component_calls": result.counts.objective_component_gradients,
            "endpoint": endpoint,
            "trace": normalize_trace(result.trace),
            "sample_stream_sha256": result.sample_stream_sha256,
            "oracle_counts": asdict(result.counts),
            "linear_solves": result.linear_solves,
            "maximum_linear_relative_residual": result.max_linear_relative_residual,
            "wall_algorithm_seconds": result.wall_algorithm_seconds,
            "wall_evaluator_seconds": result.wall_evaluator_seconds,
        }
    else:
        raise ValueError(f"unknown method {method}")

    output = record["output"]
    output["endpoint_all_finite"] = metrics_are_finite(output["endpoint"])
    output["objective_decreased"] = (
        float(output["endpoint"]["objective"]) < float(initial_metrics["objective"])
    )
    output["jacobian_regular"] = (
        float(output["endpoint"]["jacobian_sigma_min"])
        >= float(config["formal_rule"]["minimum_jacobian_sigma"])
    )
    return record


def write_atomic(path: Path, record: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def validate_existing(
    path: Path, task_index: int, config_sha256: str, method: str
) -> None:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("config_sha256") != config_sha256:
        raise ValueError(f"existing result has wrong config hash: {path}")
    task = record.get("task", {})
    if int(task.get("task_index", -1)) != task_index or task.get("method") != method:
        raise ValueError(f"existing result has wrong task identity: {path}")
