from __future__ import annotations

from pathlib import Path
import csv
import hashlib
import json
import math
import subprocess
import sys

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from analyze_libsvm_suite_stage_b import (
    compare_methods,
    residual_identity_deviation,
    summarize,
    trace_deviation,
)


def test_summarize_uses_eight_repeat_median_and_quartiles() -> None:
    result = summarize([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    assert result["count"] == 8
    assert result["median"] == pytest.approx(4.5)
    assert result["q1"] == pytest.approx(2.75)
    assert result["q3"] == pytest.approx(6.25)


def test_trace_deviation_ignores_only_wall_time() -> None:
    reference = [
        {
            "iteration": 0,
            "wall_seconds": 0.0,
            "pair_residual_squared": 1.0,
        },
        {
            "iteration": 1,
            "wall_seconds": 1.0,
            "pair_residual_squared": 0.25,
        },
    ]
    changed_time = [
        {**reference[0], "wall_seconds": 0.1},
        {**reference[1], "wall_seconds": 2.0},
    ]
    assert trace_deviation(reference, changed_time) == (0.0, 0.0)

    changed_result = [
        reference[0],
        {**reference[1], "pair_residual_squared": 0.5},
    ]
    absolute, relative = trace_deviation(reference, changed_result)
    assert absolute == pytest.approx(0.25)
    assert relative == pytest.approx(0.25)


def test_residual_identity_deviation_detects_corruption() -> None:
    exact = {
        "pair_residual_squared": 0.3,
        "stationarity_2_squared": 0.1,
        "feasibility_2_squared": 0.2,
    }
    absolute, relative = residual_identity_deviation(exact)
    assert absolute <= 1.0e-16
    assert relative <= 1.0e-15

    corrupt = {**exact, "pair_residual_squared": 0.4}
    absolute, relative = residual_identity_deviation(corrupt)
    assert absolute == pytest.approx(0.1)
    assert relative == pytest.approx(0.25)


def test_compare_methods_keeps_failures_in_denominator() -> None:
    rows = [
        {
            "dataset": "all_success",
            "a_successes": 8,
            "b_successes": 8,
            "a_median_seconds": 2.0,
            "b_median_seconds": 4.0,
        },
        {
            "dataset": "coverage_dominates",
            "a_successes": 8,
            "b_successes": 7,
            "a_median_seconds": 3.0,
            "b_median_seconds": 1.0,
        },
        {
            "dataset": "both_incomplete",
            "a_successes": 7,
            "b_successes": 7,
            "a_median_seconds": 2.0,
            "b_median_seconds": 1.0,
        },
    ]
    comparison = compare_methods(rows, "a", "b")
    assert comparison["counts"] == {
        "a_win": 2,
        "both_incomplete": 1,
    }
    assert comparison["joint_success_datasets"] == 1
    assert math.isclose(
        comparison[
            "geometric_mean_speedup_b_over_a_on_joint_successes"
        ],
        2.0,
    )


def test_end_to_end_stage_b_audit_on_balanced_synthetic_results(
    tmp_path: Path,
) -> None:
    methods = ["nr_lalm", "nr_lalm_soc", "l_al", "ipopt"]
    orders = [
        methods[offset:] + methods[:offset]
        for offset in range(4)
        for _ in range(2)
    ]
    config = {
        "methods": methods,
        "timing_method_orders": orders,
        "datasets": [
            {
                "name": "synthetic",
                "display_name": "Synthetic",
                "family": "test",
                "dimension": 3000,
                "used_samples": 100,
            }
        ],
        "reporting": {
            "primary_residual_squared_threshold": 1.0e-8,
            "secondary_residual_squared_thresholds": [
                1.0e-10,
                1.0e-12,
            ],
            "time_summary": "median [Q1,Q3]",
            "failure_policy": "failures remain in denominator",
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(config, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    dataset_dir = tmp_path / "raw" / "dataset_0"
    dataset_dir.mkdir(parents=True)
    status_path = dataset_dir / "process_status.tsv"
    with status_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(
            ["order_index", "position", "method", "exit_status"]
        )
        for order_index, order in enumerate(orders):
            for position, method in enumerate(order):
                writer.writerow([order_index, position, method, 0])
                method_scale = {
                    "nr_lalm": 1.0,
                    "nr_lalm_soc": 1.2,
                    "l_al": 2.0,
                    "ipopt": 3.0,
                }[method]
                trace = [
                    {
                        "iteration": 0,
                        "wall_seconds": 0.0,
                        "cumulative_primal_solves": 0,
                        "cumulative_correction_solves": 0,
                        "pair_residual_squared": 1.0,
                        "stationarity_2_squared": 0.75,
                        "feasibility_2_squared": 0.25,
                    },
                    {
                        "iteration": 1,
                        "wall_seconds": method_scale
                        * (1.0 + 0.01 * order_index),
                        "cumulative_primal_solves": 1,
                        "cumulative_correction_solves": (
                            1 if method == "nr_lalm_soc" else 0
                        ),
                        "pair_residual_squared": 1.0e-14,
                        "stationarity_2_squared": 0.75e-14,
                        "feasibility_2_squared": 0.25e-14,
                    },
                ]
                payload = {
                    "complete": True,
                    "config_sha256": config_hash,
                    "execution": {
                        "dataset_index": 0,
                        "method_order": [method],
                    },
                    "environment": {
                        "hostname": "test-node",
                        "slurm_array_task_id": "0",
                    },
                    "records": [
                        {
                            "method": method,
                            "problem": {"dataset": "synthetic"},
                            "run": {
                                "status": "converged",
                                "metadata": {
                                    "final_x_sha256": (
                                        f"fixed-{method}-hash"
                                    )
                                },
                                "trace": trace,
                            },
                        }
                    ],
                }
                result_path = dataset_dir / (
                    f"order_{order_index}_position_{position}_"
                    f"{method}_raw.json"
                )
                result_path.write_text(
                    json.dumps(payload, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                digest = hashlib.sha256(
                    result_path.read_bytes()
                ).hexdigest()
                result_path.with_suffix(".sha256").write_text(
                    f"{digest}  {result_path.name}\n",
                    encoding="utf-8",
                )

    output_json = tmp_path / "summary.json"
    output_csv = tmp_path / "summary.csv"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_DIR / "analyze_libsvm_suite_stage_b.py"),
            "--config",
            str(config_path),
            "--results-root",
            str(tmp_path / "raw"),
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(output_json.read_text(encoding="utf-8"))
    assert summary["passed"] is True
    assert summary["issue_count"] == 0
    assert summary["expected_processes"] == 32
    assert summary["valid_result_files"] == 32
    assert (
        summary["method_summaries"]["nr_lalm"][
            "complete_success_datasets"
        ]
        == 1
    )
    assert summary["comparisons"]["nr_lalm_vs_l_al"]["counts"] == {
        "nr_lalm_win": 1
    }
