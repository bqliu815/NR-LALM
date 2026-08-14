#!/usr/bin/env python3
"""Validate and summarize the balanced LIBSVM timing experiment."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sidecar(path: Path) -> tuple[bool, str, str]:
    sidecar = path.with_suffix(".sha256")
    if not sidecar.is_file():
        return False, "", ""
    fields = sidecar.read_text(encoding="utf-8").split()
    expected = fields[0] if fields else ""
    actual = sha256(path)
    return expected == actual, expected, actual


def first_hit(
    trace: list[dict[str, Any]], threshold: float
) -> dict[str, Any] | None:
    for point in trace:
        value = point.get("pair_residual_squared")
        if value is not None and float(value) <= threshold:
            return point
    return None


def residual_identity_deviation(
    point: dict[str, Any],
) -> tuple[float, float]:
    """Return absolute and relative error in R^2 = stationarity^2 + feasibility^2."""
    pair = float(point["pair_residual_squared"])
    components = float(point["stationarity_2_squared"]) + float(
        point["feasibility_2_squared"]
    )
    absolute = abs(pair - components)
    scale = max(
        abs(pair),
        abs(components),
        float(np.finfo(np.float64).tiny),
    )
    return absolute, absolute / scale


def threshold_key(threshold: float) -> str:
    exponent = -int(f"{threshold:.0e}".split("e")[1])
    return f"r2_1e_minus_{exponent}"


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "median": None,
            "q1": None,
            "q3": None,
            "minimum": None,
            "maximum": None,
            "mean": None,
            "standard_deviation": None,
            "coefficient_of_variation": None,
        }
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    standard_deviation = (
        float(np.std(array, ddof=1)) if len(values) > 1 else 0.0
    )
    return {
        "count": len(values),
        "median": float(np.median(array)),
        "q1": float(np.quantile(array, 0.25)),
        "q3": float(np.quantile(array, 0.75)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "mean": mean,
        "standard_deviation": standard_deviation,
        "coefficient_of_variation": (
            standard_deviation / mean if mean > 0.0 else None
        ),
    }


def trace_deviation(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
) -> tuple[float, float]:
    if len(reference) != len(candidate):
        return math.inf, math.inf
    maximum_absolute = 0.0
    maximum_relative = 0.0
    for reference_point, candidate_point in zip(reference, candidate):
        if set(reference_point) != set(candidate_point):
            return math.inf, math.inf
        for field, reference_value in reference_point.items():
            if field == "wall_seconds":
                continue
            candidate_value = candidate_point[field]
            if reference_value is None or candidate_value is None:
                if reference_value != candidate_value:
                    return math.inf, math.inf
                continue
            if isinstance(reference_value, bool):
                if reference_value != candidate_value:
                    return math.inf, math.inf
                continue
            difference = abs(
                float(reference_value) - float(candidate_value)
            )
            scale = max(1.0, abs(float(reference_value)))
            maximum_absolute = max(maximum_absolute, difference)
            maximum_relative = max(
                maximum_relative, difference / scale
            )
    return maximum_absolute, maximum_relative


def add_issue(
    issues: list[dict[str, Any]],
    *,
    dataset_index: int,
    order_index: int,
    position: int,
    method: str,
    field: str,
    actual: Any,
    expected: Any,
) -> None:
    issues.append(
        {
            "dataset_index": dataset_index,
            "order_index": order_index,
            "position": position,
            "method": method,
            "field": field,
            "actual": actual,
            "expected": expected,
        }
    )


def read_process_status(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    return [
        {
            "order_index": int(row["order_index"]),
            "position": int(row["position"]),
            "method": str(row["method"]),
            "exit_status": int(row["exit_status"]),
        }
        for row in rows
    ]


def geometric_mean(values: list[float]) -> float | None:
    if not values:
        return None
    if any(value <= 0.0 or not math.isfinite(value) for value in values):
        return None
    return float(math.exp(sum(math.log(value) for value in values) / len(values)))


def compare_methods(
    timing_rows: list[dict[str, Any]],
    method_a: str,
    method_b: str,
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    details: list[dict[str, Any]] = []
    speedups: list[float] = []
    for row in timing_rows:
        a_count = int(row[f"{method_a}_successes"])
        b_count = int(row[f"{method_b}_successes"])
        a_time = row[f"{method_a}_median_seconds"]
        b_time = row[f"{method_b}_median_seconds"]
        if a_count > b_count:
            outcome = f"{method_a}_win"
        elif b_count > a_count:
            outcome = f"{method_b}_win"
        elif a_count < 8:
            outcome = "both_incomplete"
        elif float(a_time) < float(b_time):
            outcome = f"{method_a}_win"
        elif float(a_time) > float(b_time):
            outcome = f"{method_b}_win"
        else:
            outcome = "tie"
        counts[outcome] += 1
        speedup = None
        if a_count == 8 and b_count == 8:
            speedup = float(b_time) / float(a_time)
            speedups.append(speedup)
        details.append(
            {
                "dataset": row["dataset"],
                f"{method_a}_successes": a_count,
                f"{method_b}_successes": b_count,
                f"{method_a}_median_seconds": a_time,
                f"{method_b}_median_seconds": b_time,
                f"{method_b}_over_{method_a}_speedup": speedup,
                "outcome": outcome,
            }
        )
    return {
        "metric": (
            "all-repeat success dominance, then median first-hit wall "
            "time at the primary residual threshold"
        ),
        "counts": dict(sorted(counts.items())),
        "geometric_mean_speedup_b_over_a_on_joint_successes": (
            geometric_mean(speedups)
        ),
        "joint_success_datasets": len(speedups),
        "detail": details,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    args = parser.parse_args()

    config_path = args.config.resolve()
    results_root = args.results_root.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_hash = sha256(config_path)
    methods = [str(method) for method in config["methods"]]
    orders = [
        [str(method) for method in order]
        for order in config["timing_method_orders"]
    ]
    datasets = list(config["datasets"])
    thresholds = [
        float(config["reporting"][
            "primary_residual_squared_threshold"
        ]),
        *[
            float(value)
            for value in config["reporting"][
                "secondary_residual_squared_thresholds"
            ]
        ],
    ]
    primary_key = threshold_key(thresholds[0])
    issues: list[dict[str, Any]] = []
    raw_records: list[dict[str, Any]] = []
    traces: dict[tuple[int, str], list[list[dict[str, Any]]]] = (
        defaultdict(list)
    )
    final_hashes: defaultdict[tuple[int, str], set[str]] = defaultdict(set)
    hostnames: defaultdict[int, set[str]] = defaultdict(set)
    maximum_residual_identity_absolute = 0.0
    maximum_residual_identity_relative = 0.0

    for dataset_index, dataset in enumerate(datasets):
        dataset_dir = results_root / f"dataset_{dataset_index}"
        status_path = dataset_dir / "process_status.tsv"
        status_rows = read_process_status(status_path)
        status_lookup = {
            (
                int(row["order_index"]),
                int(row["position"]),
                str(row["method"]),
            ): int(row["exit_status"])
            for row in status_rows
        }
        if len(status_rows) != len(orders) * len(methods):
            add_issue(
                issues,
                dataset_index=dataset_index,
                order_index=-1,
                position=-1,
                method="__dataset__",
                field="process_status.rows",
                actual=len(status_rows),
                expected=len(orders) * len(methods),
            )
        if len(status_lookup) != len(status_rows):
            add_issue(
                issues,
                dataset_index=dataset_index,
                order_index=-1,
                position=-1,
                method="__dataset__",
                field="process_status.unique_rows",
                actual=len(status_lookup),
                expected=len(status_rows),
            )
        for order_index, order in enumerate(orders):
            for position, method in enumerate(order):
                key = (order_index, position, method)
                exit_status = status_lookup.get(key)
                result = dataset_dir / (
                    f"order_{order_index}_position_{position}_"
                    f"{method}_raw.json"
                )
                partial_result = result.with_suffix(
                    result.suffix + ".part"
                )
                base_record: dict[str, Any] = {
                    "dataset_index": dataset_index,
                    "dataset": str(dataset["name"]),
                    "display_name": str(dataset["display_name"]),
                    "family": str(dataset["family"]),
                    "dimension": int(dataset["dimension"]),
                    "order_index": order_index,
                    "position": position,
                    "method": method,
                    "exit_status": exit_status,
                    "result_path": str(result),
                    "result_file_present": result.is_file(),
                    "result_valid": False,
                    "solver_status": None,
                    "final_pair_residual_squared": None,
                }
                if partial_result.is_file():
                    add_issue(
                        issues,
                        dataset_index=dataset_index,
                        order_index=order_index,
                        position=position,
                        method=method,
                        field="partial_result_file",
                        actual=str(partial_result),
                        expected="absent after process completion",
                    )
                for threshold in thresholds:
                    threshold_name = threshold_key(threshold)
                    base_record[f"{threshold_name}_wall_seconds"] = None
                    base_record[f"{threshold_name}_iteration"] = None
                    base_record[f"{threshold_name}_primal_solves"] = None
                    base_record[f"{threshold_name}_correction_solves"] = None
                    base_record[f"{threshold_name}_total_solves"] = None
                if exit_status is None:
                    add_issue(
                        issues,
                        dataset_index=dataset_index,
                        order_index=order_index,
                        position=position,
                        method=method,
                        field="process_status.exit_status",
                        actual=None,
                        expected="0 or 124",
                    )
                    base_record["audit_status"] = "missing_process_status"
                    raw_records.append(base_record)
                    continue
                if exit_status != 0:
                    if result.exists():
                        add_issue(
                            issues,
                            dataset_index=dataset_index,
                            order_index=order_index,
                            position=position,
                            method=method,
                            field="timeout_result_file",
                            actual=str(result),
                            expected="absent",
                        )
                    if exit_status != 124:
                        add_issue(
                            issues,
                            dataset_index=dataset_index,
                            order_index=order_index,
                            position=position,
                            method=method,
                            field="process_status.exit_status",
                            actual=exit_status,
                            expected="0 or documented timeout status 124",
                        )
                    base_record["audit_status"] = (
                        "external_timeout_1800s"
                        if exit_status == 124
                        else f"process_exit_{exit_status}"
                    )
                    raw_records.append(base_record)
                    continue
                if not result.is_file():
                    add_issue(
                        issues,
                        dataset_index=dataset_index,
                        order_index=order_index,
                        position=position,
                        method=method,
                        field="result_file",
                        actual="absent",
                        expected=str(result),
                    )
                    base_record["audit_status"] = "missing_result_file"
                    raw_records.append(base_record)
                    continue
                sidecar_valid, expected_hash, actual_hash = verify_sidecar(
                    result
                )
                if not sidecar_valid:
                    add_issue(
                        issues,
                        dataset_index=dataset_index,
                        order_index=order_index,
                        position=position,
                        method=method,
                        field="result_sha256",
                        actual=actual_hash,
                        expected=expected_hash or "valid sidecar",
                    )
                payload = json.loads(result.read_text(encoding="utf-8"))
                checks = {
                    "complete": (payload.get("complete"), True),
                    "config_sha256": (
                        payload.get("config_sha256"),
                        config_hash,
                    ),
                    "execution.dataset_index": (
                        payload.get("execution", {}).get(
                            "dataset_index"
                        ),
                        dataset_index,
                    ),
                    "execution.method_order": (
                        payload.get("execution", {}).get("method_order"),
                        [method],
                    ),
                    "environment.slurm_array_task_id": (
                        payload.get("environment", {}).get(
                            "slurm_array_task_id"
                        ),
                        str(dataset_index),
                    ),
                }
                for field, (actual, expected) in checks.items():
                    if actual != expected:
                        add_issue(
                            issues,
                            dataset_index=dataset_index,
                            order_index=order_index,
                            position=position,
                            method=method,
                            field=field,
                            actual=actual,
                            expected=expected,
                        )
                payload_records = payload.get("records", [])
                if (
                    len(payload_records) != 1
                    or payload_records[0].get("method") != method
                ):
                    add_issue(
                        issues,
                        dataset_index=dataset_index,
                        order_index=order_index,
                        position=position,
                        method=method,
                        field="payload.records",
                        actual=[
                            record.get("method")
                            for record in payload_records
                        ],
                        expected=[method],
                    )
                    base_record["audit_status"] = "invalid_record_set"
                    raw_records.append(base_record)
                    continue
                record = payload_records[0]
                problem_metadata = record.get("problem", {})
                if problem_metadata.get("dataset") != str(dataset["name"]):
                    add_issue(
                        issues,
                        dataset_index=dataset_index,
                        order_index=order_index,
                        position=position,
                        method=method,
                        field="record.problem.dataset",
                        actual=problem_metadata.get("dataset"),
                        expected=str(dataset["name"]),
                    )
                trace = record["run"]["trace"]
                if not trace:
                    add_issue(
                        issues,
                        dataset_index=dataset_index,
                        order_index=order_index,
                        position=position,
                        method=method,
                        field="record.run.trace",
                        actual=[],
                        expected="at least one trace point",
                    )
                    base_record["audit_status"] = "empty_trace"
                    raw_records.append(base_record)
                    continue
                previous_iteration = -1
                previous_wall_seconds = -math.inf
                previous_primal_solves = -1
                previous_correction_solves = -1
                for trace_index, point in enumerate(trace):
                    iteration = int(point["iteration"])
                    wall_seconds = float(point["wall_seconds"])
                    primal_solves = int(
                        point["cumulative_primal_solves"]
                    )
                    correction_solves = int(
                        point["cumulative_correction_solves"]
                    )
                    if (
                        iteration <= previous_iteration
                        or wall_seconds < previous_wall_seconds
                        or primal_solves < previous_primal_solves
                        or correction_solves
                        < previous_correction_solves
                    ):
                        add_issue(
                            issues,
                            dataset_index=dataset_index,
                            order_index=order_index,
                            position=position,
                            method=method,
                            field=f"record.run.trace[{trace_index}]",
                            actual={
                                "iteration": iteration,
                                "wall_seconds": wall_seconds,
                                "primal_solves": primal_solves,
                                "correction_solves": correction_solves,
                            },
                            expected=(
                                "strictly increasing iteration and "
                                "nondecreasing time/solve counters"
                            ),
                        )
                    previous_iteration = iteration
                    previous_wall_seconds = wall_seconds
                    previous_primal_solves = primal_solves
                    previous_correction_solves = correction_solves
                    identity_absolute, identity_relative = (
                        residual_identity_deviation(point)
                    )
                    maximum_residual_identity_absolute = max(
                        maximum_residual_identity_absolute,
                        identity_absolute,
                    )
                    maximum_residual_identity_relative = max(
                        maximum_residual_identity_relative,
                        identity_relative,
                    )
                    if identity_relative > 1.0e-10:
                        add_issue(
                            issues,
                            dataset_index=dataset_index,
                            order_index=order_index,
                            position=position,
                            method=method,
                            field=(
                                "pair_residual_squared_identity_"
                                f"trace_{trace_index}"
                            ),
                            actual=identity_relative,
                            expected="relative deviation <= 1e-10",
                        )
                traces[(dataset_index, method)].append(trace)
                final_hashes[(dataset_index, method)].add(
                    str(record["run"]["metadata"]["final_x_sha256"])
                )
                hostnames[dataset_index].add(
                    str(payload["environment"]["hostname"])
                )
                base_record.update(
                    {
                        "audit_status": "valid",
                        "result_valid": True,
                        "result_sha256": actual_hash,
                        "hostname": str(
                            payload["environment"]["hostname"]
                        ),
                        "solver_status": record["run"].get("status"),
                        "final_pair_residual_squared": float(
                            trace[-1]["pair_residual_squared"]
                        ),
                        "trace_points": len(trace),
                        "final_x_sha256": str(
                            record["run"]["metadata"]["final_x_sha256"]
                        ),
                    }
                )
                for threshold in thresholds:
                    threshold_name = threshold_key(threshold)
                    hit = first_hit(trace, threshold)
                    if hit is None:
                        continue
                    primal = int(hit["cumulative_primal_solves"])
                    correction = int(
                        hit["cumulative_correction_solves"]
                    )
                    base_record[
                        f"{threshold_name}_wall_seconds"
                    ] = float(hit["wall_seconds"])
                    base_record[
                        f"{threshold_name}_iteration"
                    ] = int(hit["iteration"])
                    base_record[
                        f"{threshold_name}_primal_solves"
                    ] = primal
                    base_record[
                        f"{threshold_name}_correction_solves"
                    ] = correction
                    base_record[
                        f"{threshold_name}_total_solves"
                    ] = primal + correction
                raw_records.append(base_record)

    maximum_trace_absolute = 0.0
    maximum_trace_relative = 0.0
    trace_audit: list[dict[str, Any]] = []
    for dataset_index, dataset in enumerate(datasets):
        if len(hostnames[dataset_index]) != 1:
            add_issue(
                issues,
                dataset_index=dataset_index,
                order_index=-1,
                position=-1,
                method="__dataset__",
                field="exclusive_node_hostnames",
                actual=sorted(hostnames[dataset_index]),
                expected="exactly one hostname",
            )
        for method in methods:
            group = traces[(dataset_index, method)]
            group_absolute = 0.0
            group_relative = 0.0
            if group:
                reference = group[0]
                for candidate in group[1:]:
                    absolute, relative = trace_deviation(
                        reference, candidate
                    )
                    group_absolute = max(group_absolute, absolute)
                    group_relative = max(group_relative, relative)
            maximum_trace_absolute = max(
                maximum_trace_absolute, group_absolute
            )
            maximum_trace_relative = max(
                maximum_trace_relative, group_relative
            )
            if group_relative > 1.0e-10:
                add_issue(
                    issues,
                    dataset_index=dataset_index,
                    order_index=-1,
                    position=-1,
                    method=method,
                    field="cross_repeat_trace_relative_deviation",
                    actual=group_relative,
                    expected="<= 1e-10",
                )
            if len(final_hashes[(dataset_index, method)]) > 1:
                add_issue(
                    issues,
                    dataset_index=dataset_index,
                    order_index=-1,
                    position=-1,
                    method=method,
                    field="cross_repeat_final_x_hashes",
                    actual=sorted(
                        final_hashes[(dataset_index, method)]
                    ),
                    expected="at most one unique hash",
                )
            trace_audit.append(
                {
                    "dataset_index": dataset_index,
                    "dataset": str(dataset["name"]),
                    "method": method,
                    "valid_traces": len(group),
                    "unique_final_x_hashes": len(
                        final_hashes[(dataset_index, method)]
                    ),
                    "maximum_absolute_deviation": group_absolute,
                    "maximum_relative_deviation": group_relative,
                }
            )

    timing_rows: list[dict[str, Any]] = []
    method_summaries: dict[str, Any] = {}
    grouped = {
        (dataset_index, method): [
            record
            for record in raw_records
            if record["dataset_index"] == dataset_index
            and record["method"] == method
        ]
        for dataset_index in range(len(datasets))
        for method in methods
    }
    for dataset_index, dataset in enumerate(datasets):
        row: dict[str, Any] = {
            "dataset_index": dataset_index,
            "dataset": str(dataset["name"]),
            "display_name": str(dataset["display_name"]),
            "family": str(dataset["family"]),
            "dimension": int(dataset["dimension"]),
            "used_samples": int(dataset["used_samples"]),
            "hostname": (
                next(iter(hostnames[dataset_index]))
                if len(hostnames[dataset_index]) == 1
                else None
            ),
        }
        for method in methods:
            records = grouped[(dataset_index, method)]
            for threshold in thresholds:
                threshold_name = threshold_key(threshold)
                values = [
                    float(record[
                        f"{threshold_name}_wall_seconds"
                    ])
                    for record in records
                    if record.get(
                        f"{threshold_name}_wall_seconds"
                    )
                    is not None
                ]
                time_summary = summarize(values)
                for statistic in (
                    "count",
                    "median",
                    "q1",
                    "q3",
                    "minimum",
                    "maximum",
                    "coefficient_of_variation",
                ):
                    row[
                        f"{method}_{threshold_name}_{statistic}"
                    ] = time_summary[statistic]
            primary_successes = int(
                row[f"{method}_{primary_key}_count"]
            )
            row[f"{method}_successes"] = primary_successes
            row[f"{method}_median_seconds"] = row[
                f"{method}_{primary_key}_median"
            ]
            primary_records = [
                record
                for record in records
                if record.get(f"{primary_key}_wall_seconds")
                is not None
            ]
            unique_work = sorted(
                {
                    (
                        int(record[f"{primary_key}_iteration"]),
                        int(record[f"{primary_key}_primal_solves"]),
                        int(record[f"{primary_key}_correction_solves"]),
                        int(record[f"{primary_key}_total_solves"]),
                    )
                    for record in primary_records
                }
            )
            row[f"{method}_unique_primary_work"] = json.dumps(
                unique_work, separators=(",", ":")
            )
            if len(unique_work) > 1:
                add_issue(
                    issues,
                    dataset_index=dataset_index,
                    order_index=-1,
                    position=-1,
                    method=method,
                    field="cross_repeat_primary_work",
                    actual=unique_work,
                    expected="at most one unique tuple",
                )
        timing_rows.append(row)

    for method in methods:
        method_rows = [
            record for record in raw_records if record["method"] == method
        ]
        dataset_successes = {
            row["dataset"]: int(row[f"{method}_successes"])
            for row in timing_rows
        }
        family_rates: defaultdict[str, list[float]] = defaultdict(list)
        for row in timing_rows:
            family_rates[str(row["family"])].append(
                dataset_successes[str(row["dataset"])] / len(orders)
            )
        family_means = {
            family: float(np.mean(values))
            for family, values in sorted(family_rates.items())
        }
        method_summaries[method] = {
            "expected_processes": len(datasets) * len(orders),
            "valid_result_files": sum(
                record["result_valid"] for record in method_rows
            ),
            "external_timeouts": sum(
                record["audit_status"] == "external_timeout_1800s"
                for record in method_rows
            ),
            "complete_success_datasets": sum(
                count == len(orders)
                for count in dataset_successes.values()
            ),
            "success_repeats": sum(dataset_successes.values()),
            "expected_repeats": len(datasets) * len(orders),
            "family_success_rates": family_means,
            "family_balanced_success_rate": float(
                np.mean(list(family_means.values()))
            ),
        }

    position_values: defaultdict[tuple[str, int], list[float]] = (
        defaultdict(list)
    )
    for record in raw_records:
        method = str(record["method"])
        value = record.get(f"{primary_key}_wall_seconds")
        if value is None:
            continue
        timing_row = timing_rows[int(record["dataset_index"])]
        median = timing_row[f"{method}_median_seconds"]
        if median is not None and float(median) > 0.0:
            position_values[(method, int(record["position"]))].append(
                float(value) / float(median)
            )
    position_effect = {
        method: {
            str(position): summarize(
                position_values[(method, position)]
            )
            for position in range(len(methods))
        }
        for method in methods
    }

    tau_grid = np.geomspace(1.0, 100.0, num=101)
    profile = {
        method: {f"{tau:.12g}": 0.0 for tau in tau_grid}
        for method in methods
    }
    ratios: dict[tuple[str, str], float] = {}
    for row in timing_rows:
        finite_times = {
            method: float(row[f"{method}_median_seconds"])
            for method in methods
            if int(row[f"{method}_successes"]) == len(orders)
        }
        best = min(finite_times.values()) if finite_times else math.inf
        for method in methods:
            value = finite_times.get(method, math.inf)
            ratios[(str(row["dataset"]), method)] = (
                value / best if math.isfinite(value) else math.inf
            )
    for method in methods:
        for tau in tau_grid:
            profile[method][f"{tau:.12g}"] = sum(
                ratios[(str(row["dataset"]), method)] <= tau
                for row in timing_rows
            ) / len(timing_rows)

    summary = {
        "schema": "libsvm_binary_suite_summary_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "config_sha256": config_hash,
        "results_root": str(results_root),
        "expected_datasets": len(datasets),
        "expected_orders": len(orders),
        "expected_methods": len(methods),
        "expected_processes": len(datasets) * len(orders) * len(methods),
        "primary_residual_squared_threshold": thresholds[0],
        "secondary_residual_squared_thresholds": thresholds[1:],
        "timing_policy": config["reporting"]["time_summary"],
        "failure_policy": config["reporting"]["failure_policy"],
        "valid_result_files": sum(
            record["result_valid"] for record in raw_records
        ),
        "external_timeouts": sum(
            record["audit_status"] == "external_timeout_1800s"
            for record in raw_records
        ),
        "audit_status_counts": dict(
            sorted(
                Counter(
                    str(record["audit_status"])
                    for record in raw_records
                ).items()
            )
        ),
        "maximum_cross_repeat_trace_absolute_deviation": (
            maximum_trace_absolute
        ),
        "maximum_cross_repeat_trace_relative_deviation": (
            maximum_trace_relative
        ),
        "maximum_residual_identity_absolute_deviation": (
            maximum_residual_identity_absolute
        ),
        "maximum_residual_identity_relative_deviation": (
            maximum_residual_identity_relative
        ),
        "method_summaries": method_summaries,
        "comparisons": {
            "nr_lalm_vs_l_al": compare_methods(
                timing_rows, "nr_lalm", "l_al"
            ),
            "nr_lalm_soc_vs_nr_lalm": compare_methods(
                timing_rows, "nr_lalm_soc", "nr_lalm"
            ),
            "nr_lalm_vs_ipopt": compare_methods(
                timing_rows, "nr_lalm", "ipopt"
            ),
        },
        "position_effect": position_effect,
        "performance_profile_tau_grid": [
            float(tau) for tau in tau_grid
        ],
        "performance_profile": profile,
        "timing_rows": timing_rows,
        "trace_audit": trace_audit,
        "raw_records": raw_records,
        "issue_count": len(issues),
        "passed": (
            not issues
            and len(raw_records)
            == len(datasets) * len(orders) * len(methods)
        ),
        "issues": issues,
    }
    output_json = args.output_json.resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_csv.resolve(), timing_rows)
    print(
        json.dumps(
            {
                "output_json": str(output_json),
                "output_csv": str(args.output_csv.resolve()),
                "expected_processes": summary["expected_processes"],
                "valid_result_files": summary["valid_result_files"],
                "external_timeouts": summary["external_timeouts"],
                "issue_count": summary["issue_count"],
                "passed": summary["passed"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
