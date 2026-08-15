#!/usr/bin/env python3
"""Run high-dimensional constrained-logistic experiments."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter
from typing import Any

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from highdim_logistic.ipopt import IpoptConfig, solve_ipopt
from highdim_logistic.problem import (
    dimension_adaptive_affine_shape,
    load_libsvm_bz2,
    make_instance,
)
from highdim_logistic.solver import (
    LALMConfig,
    solve_lal,
    solve_nr_lalm,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_checkpoint(
    output_path: Path, payload: dict[str, Any], *, complete: bool
) -> None:
    payload["checkpointed_utc"] = datetime.now(timezone.utc).isoformat()
    payload["complete"] = complete
    temporary = output_path.with_suffix(output_path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    output_path.with_suffix(".sha256").write_text(
        f"{sha256(output_path)}  {output_path.name}\n",
        encoding="utf-8",
    )


def lalm_config(values: dict[str, Any]) -> LALMConfig:
    return LALMConfig(
        rho=float(values["rho"]),
        beta_floor=float(values["beta_floor"]),
        beta_initial=float(values["beta_initial"]),
        beta_ceiling=float(values["beta_ceiling"]),
        mu=float(values.get("mu", 2.0)),
        acceptance_eta=float(values.get("acceptance_eta", 0.05)),
        decrease_ratio=float(values.get("decrease_ratio", 1.0)),
        decrease_factor=float(values.get("decrease_factor", 0.5)),
        max_iterations=int(values.get("max_iterations", 200)),
        max_backtracks=int(values.get("max_backtracks", 60)),
        target_residual_squared=float(
            values.get("target_residual_squared", 1.0e-14)
        ),
        required_linear_residual=float(
            values.get("required_linear_residual", 1.0e-10)
        ),
        maximum_linear_refinements=int(
            values.get("maximum_linear_refinements", 0)
        ),
    )


def run_method(
    method: str,
    problem: Any,
    x0: np.ndarray,
    lambda0: np.ndarray,
    solver_config: LALMConfig,
    ipopt_values: dict[str, Any],
) -> Any:
    if method == "nr_lalm":
        return solve_nr_lalm(
            problem, solver_config, x0, lambda0, use_soc=False
        )
    if method == "nr_lalm_soc":
        return solve_nr_lalm(
            problem, solver_config, x0, lambda0, use_soc=True
        )
    if method == "l_al":
        return solve_lal(problem, solver_config, x0, lambda0)
    if method == "ipopt":
        return solve_ipopt(
            problem,
            IpoptConfig(
                max_iterations=int(
                    ipopt_values.get("max_iterations", 200)
                ),
                tolerance=float(ipopt_values.get("tolerance", 1.0e-10)),
                acceptable_tolerance=float(
                    ipopt_values.get("acceptable_tolerance", 1.0e-8)
                ),
                max_wall_seconds=float(
                    ipopt_values.get("max_wall_seconds", 1800.0)
                ),
                print_level=int(ipopt_values.get("print_level", 0)),
                linear_solver=str(
                    ipopt_values.get("linear_solver", "pardisomkl")
                ),
            ),
            x0,
            lambda0,
        )
    raise ValueError(f"unknown method {method}")


def resolve_method_order(
    config: dict[str, Any], order_index: int | None
) -> tuple[list[str], int | None]:
    methods = [str(method) for method in config["methods"]]
    if len(set(methods)) != len(methods):
        raise ValueError("config methods must be distinct")
    if order_index is None:
        return methods, None
    orders = config.get("timing_method_orders")
    if orders is None:
        raise ValueError(
            "--method-order-index requires timing_method_orders"
        )
    if order_index < 0 or order_index >= len(orders):
        raise IndexError("method order index is outside the config")
    order = [str(method) for method in orders[order_index]]
    if len(order) != len(methods) or set(order) != set(methods):
        raise ValueError(
            f"timing order {order_index} is not a method permutation"
        )
    return order, order_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset-index", type=int)
    parser.add_argument("--method-order-index", type=int)
    parser.add_argument(
        "--method",
        choices=["nr_lalm", "nr_lalm_soc", "l_al", "ipopt"],
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    output_path = args.output.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset_specs = config["datasets"]
    if args.dataset_index is not None:
        dataset_specs = [dataset_specs[args.dataset_index]]
    if args.method is None:
        method_order, method_order_index = resolve_method_order(
            config, args.method_order_index
        )
    else:
        if args.method_order_index is not None:
            raise ValueError(
                "--method and --method-order-index are mutually exclusive"
            )
        if args.method not in config["methods"]:
            raise ValueError("requested method is not in config")
        method_order = [args.method]
        method_order_index = None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    finals_dir = output_path.parent / "finals"
    save_final_states = bool(config.get("save_final_states", True))
    if save_final_states:
        finals_dir.mkdir(parents=True, exist_ok=True)
    solver_config = lalm_config(config["parameters"])
    records: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "schema": "highdim_sparse_constrained_logistic_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "hostname": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get(
                "SLURM_ARRAY_TASK_ID"
            ),
            "numpy": np.__version__,
        },
        "config": config,
        "execution": {
            "dataset_index": args.dataset_index,
            "method_order_index": method_order_index,
            "method_order": method_order,
        },
        "records": records,
    }

    for specification in dataset_specs:
        data_path = Path(specification["path"])
        if not data_path.is_absolute():
            data_path = PACKAGE_ROOT / data_path
        data = load_libsvm_bz2(
            data_path,
            dataset_name=str(specification["name"]),
            expected_dimension=int(specification["dimension"]),
            split_seed=int(config["split_seed"]),
            test_fraction=float(config["test_fraction"]),
        )
        instance_values = config["instance"]
        if bool(instance_values.get("dimension_adaptive", False)):
            affine_constraints, affine_support_size = (
                dimension_adaptive_affine_shape(data.dimension)
            )
        else:
            affine_constraints = int(
                instance_values["affine_constraints"]
            )
            affine_support_size = int(
                instance_values["affine_support_size"]
            )
        instance = make_instance(
            data,
            seed=int(instance_values["seed"]),
            affine_constraints=affine_constraints,
            affine_support_size=affine_support_size,
            affine_rhs_norm=float(
                instance_values["affine_rhs_norm"]
            ),
        )
        for execution_position, method in enumerate(method_order):
            method_start = perf_counter()
            run = run_method(
                str(method),
                instance.problem,
                instance.x0,
                instance.lambda0,
                solver_config,
                config.get("ipopt", {}),
            )
            runner_wall_seconds = perf_counter() - method_start
            final_path: Path | None = None
            final_state_sha256: str | None = None
            if save_final_states:
                order_tag = (
                    ""
                    if method_order_index is None
                    else f"__order{method_order_index}"
                )
                final_path = (
                    finals_dir
                    / (
                        f"{instance.problem.name}{order_tag}"
                        f"__{method}_final.npz"
                    )
                )
                np.savez_compressed(
                    final_path,
                    x=np.asarray(run.final_x, dtype=np.float64),
                    multiplier=np.asarray(
                        run.final_multiplier, dtype=np.float64
                    ),
                )
                final_state_sha256 = sha256(final_path)
            records.append(
                {
                    "problem_id": instance.problem.name,
                    "problem": instance.metadata,
                    "method": method,
                    "execution_position": execution_position,
                    "runner_wall_seconds": runner_wall_seconds,
                    "run": run.to_jsonable(),
                    "final_state_path": (
                        str(final_path) if final_path is not None else None
                    ),
                    "final_state_sha256": final_state_sha256,
                }
            )
            write_checkpoint(output_path, payload, complete=False)
            final = run.trace[-1]
            print(
                json.dumps(
                    {
                        "problem_id": instance.problem.name,
                        "method": method,
                        "status": run.status,
                        "iterations": len(run.trace) - 1,
                        "primal_solves": run.counters.get(
                            "primal_linear_solves"
                        ),
                        "final_r2": final["pair_residual_squared"],
                        "wall_seconds": final["wall_seconds"],
                        "test_accuracy": final["test_accuracy"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    write_checkpoint(output_path, payload, complete=True)
    print(json.dumps({"output": str(output_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
