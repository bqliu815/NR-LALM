#!/usr/bin/env python3
"""Run the four paper methods for one data-set/stream replicate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR / "scripts"))
sys.path.insert(0, str(EXPERIMENT_DIR / "src"))

import numpy as np  # noqa: E402

from runner_core import (  # noqa: E402
    METHODS,
    array_assignment,
    base_record,
    evaluate_metrics,
    load_problem,
    run_method,
    sha256_file,
    validate_existing,
    write_atomic,
)
from sphere_logistic.warm_start import build_common_warm_start  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--array-index", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("protocol_status") != "PAPER_FROZEN":
        raise ValueError("runner requires the frozen paper protocol")
    if tuple(config.get("displayed_methods", ())) != METHODS:
        raise ValueError("configured method order differs from the four-method runner")
    dataset_index, repeat = array_assignment(
        args.array_index,
        len(config["datasets"]),
        int(config["oracle_repeats"]),
    )
    config_sha256 = sha256_file(config_path)
    data, problem = load_problem(config, dataset_index)
    dataset_specification = config["datasets"][dataset_index]
    x0, warm_metadata = build_common_warm_start(
        problem,
        config["initialization"],
        int(dataset_specification["selection_seed"]),
    )
    lambda0 = np.zeros(problem.num_constraints, dtype=float)
    initial_metrics = evaluate_metrics(problem, x0)
    if int(warm_metadata["component_gradient_cost"]) != int(
        config["initialization"]["component_gradient_cost"]
    ):
        raise ValueError("warm-start component cost is inconsistent")
    if float(initial_metrics["jacobian_sigma_min"]) < float(
        config["formal_rule"]["minimum_jacobian_sigma"]
    ):
        raise ValueError("common warm start fails the Jacobian regularity check")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    exceptions = 0
    for method_index, method in enumerate(METHODS):
        task_index = (
            dataset_index * int(config["oracle_repeats"]) + repeat
        ) * len(METHODS) + method_index
        output_path = args.output_dir / f"task_{task_index:04d}.json"
        if output_path.exists():
            validate_existing(output_path, task_index, config_sha256, method)
            continue
        try:
            record = run_method(
                config=config,
                config_sha256=config_sha256,
                dataset_index=dataset_index,
                repeat=repeat,
                method=method,
                data=data,
                problem=problem,
                x0=x0,
                lambda0=lambda0,
                initial_metrics=initial_metrics,
            )
        except Exception as error:
            exceptions += 1
            stream_seed = (
                int(config["oracle_seed_base"])
                + 10_000 * dataset_index
                + repeat
            )
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
            record["output"] = {
                "status": "exception",
                "message": str(error),
                "traceback": traceback.format_exc(),
                "endpoint_all_finite": False,
                "objective_decreased": False,
                "jacobian_regular": False,
            }
        record["warm_start"] = warm_metadata
        record["common_preprocessing_component_gradients"] = int(
            warm_metadata["component_gradient_cost"]
        )
        write_atomic(output_path, record)

    print(
        json.dumps(
            {
                "array_index": args.array_index,
                "dataset_index": dataset_index,
                "dataset": data.name,
                "repeat": repeat,
                "methods": list(METHODS),
                "new_exceptions": exceptions,
                "config_sha256": config_sha256,
                "warm_start_x0_sha256": warm_metadata["x0_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
