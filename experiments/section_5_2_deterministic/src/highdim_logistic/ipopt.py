"""Sparse-Jacobian IPOPT adapter for million-feature logistic models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import numpy as np

from .problem import Array, SparseConstrainedLogistic
from .solver import SolverRun, independent_record


@dataclass(frozen=True)
class IpoptConfig:
    max_iterations: int = 200
    tolerance: float = 1.0e-10
    acceptable_tolerance: float = 1.0e-8
    max_wall_seconds: float = 1800.0
    print_level: int = 0
    linear_solver: str = "mumps"


class _Callbacks:
    def __init__(
        self,
        problem: SparseConstrainedLogistic,
        x0: Array,
        lambda0: Array,
    ) -> None:
        self.problem = problem
        self.nlp: Any = None
        self.start = perf_counter()
        self.previous_x = np.asarray(x0, dtype=np.float64).copy()
        self.previous_multiplier = np.asarray(
            lambda0, dtype=np.float64
        ).copy()
        self.counts = {
            "objective_evaluations": 0,
            "gradient_evaluations": 0,
            "constraint_evaluations": 0,
            "jacobian_evaluations": 0,
            "intermediate_callbacks": 0,
            "independent_evaluator_calls": 1,
        }
        self.trace = [
            independent_record(
                problem,
                self.previous_x,
                self.previous_multiplier,
                iteration=0,
                native_iteration=0,
                elapsed=0.0,
                beta=None,
                primal_solves=0,
                correction_solves=0,
                backtracks=0,
                model_ratio=None,
                step_norm=None,
                correction_norm=0.0,
                solve_relative_residual=None,
                correction_relative_residual=None,
            )
        ]

    def objective(self, x: Array) -> float:
        self.counts["objective_evaluations"] += 1
        return self.problem.objective(x)

    def gradient(self, x: Array) -> Array:
        self.counts["gradient_evaluations"] += 1
        return self.problem.gradient(x)

    def constraints(self, x: Array) -> Array:
        self.counts["constraint_evaluations"] += 1
        return self.problem.constraint(x)

    def jacobianstructure(self) -> tuple[Array, Array]:
        return self.problem.jacobian_structure()

    def jacobian(self, x: Array) -> Array:
        self.counts["jacobian_evaluations"] += 1
        return self.problem.jacobian_values(x)

    def _append(
        self,
        x: Array,
        multiplier: Array,
        *,
        native_iteration: int,
    ) -> None:
        x = np.asarray(x, dtype=np.float64)
        multiplier = np.asarray(multiplier, dtype=np.float64)
        if np.array_equal(x, self.previous_x) and np.array_equal(
            multiplier, self.previous_multiplier
        ):
            return
        self.counts["independent_evaluator_calls"] += 1
        self.trace.append(
            independent_record(
                self.problem,
                x,
                multiplier,
                iteration=len(self.trace),
                native_iteration=native_iteration,
                elapsed=perf_counter() - self.start,
                beta=None,
                primal_solves=len(self.trace),
                correction_solves=0,
                backtracks=0,
                model_ratio=None,
                step_norm=float(np.linalg.norm(x - self.previous_x)),
                correction_norm=0.0,
                solve_relative_residual=None,
                correction_relative_residual=None,
            )
        )
        self.previous_x = x.copy()
        self.previous_multiplier = multiplier.copy()

    def intermediate(self, *args: Any) -> bool:
        self.counts["intermediate_callbacks"] += 1
        native_iteration = int(args[1]) if len(args) > 1 else len(self.trace)
        try:
            current = self.nlp.get_current_iterate(scaled=False)
        except (AttributeError, RuntimeError):
            return True
        if not current or "x" not in current or "mult_g" not in current:
            return True
        x = np.asarray(current["x"], dtype=np.float64)
        multiplier = np.asarray(current["mult_g"], dtype=np.float64)
        if (
            x.shape == (self.problem.dimension,)
            and multiplier.shape == (self.problem.constraints,)
        ):
            self._append(
                x, multiplier, native_iteration=native_iteration
            )
        return True


def solve_ipopt(
    problem: SparseConstrainedLogistic,
    config: IpoptConfig,
    x0: Array,
    lambda0: Array,
) -> SolverRun:
    try:
        import cyipopt
    except ImportError as error:  # pragma: no cover - cluster dependency
        raise RuntimeError("cyipopt is required for IPOPT") from error

    x0 = np.asarray(x0, dtype=np.float64)
    lambda0 = np.asarray(lambda0, dtype=np.float64)
    problem.check_shapes(x0)
    callbacks = _Callbacks(problem, x0, lambda0)
    infinity = 1.0e19
    nlp = cyipopt.Problem(
        n=problem.dimension,
        m=problem.constraints,
        problem_obj=callbacks,
        lb=np.full(problem.dimension, -infinity),
        ub=np.full(problem.dimension, infinity),
        cl=np.zeros(problem.constraints),
        cu=np.zeros(problem.constraints),
    )
    callbacks.nlp = nlp
    options: dict[str, float | int | str] = {
        "max_iter": config.max_iterations,
        "tol": config.tolerance,
        "acceptable_tol": config.acceptable_tolerance,
        "max_wall_time": config.max_wall_seconds,
        "print_level": config.print_level,
        "hessian_approximation": "limited-memory",
        "linear_solver": config.linear_solver,
    }
    for key, value in options.items():
        nlp.add_option(key, value)
    try:
        solution, info = nlp.solve(x0, lagrange=lambda0)
        status = "completed"
        message = info.get("status_msg", "")
        if isinstance(message, bytes):
            message = message.decode(errors="replace")
    except Exception as error:  # pragma: no cover - backend dependent
        solution = callbacks.previous_x
        info = {
            "mult_g": callbacks.previous_multiplier,
            "status": -999,
        }
        status = "solver_exception"
        message = f"{type(error).__name__}: {error}"
    solution = np.asarray(solution, dtype=np.float64)
    final_multiplier = np.asarray(info["mult_g"], dtype=np.float64)
    callbacks._append(
        solution,
        final_multiplier,
        native_iteration=max(
            int(
                callbacks.trace[-1]["native_iteration"]
                if callbacks.trace[-1]["native_iteration"] is not None
                else len(callbacks.trace) - 1
            ),
            len(callbacks.trace) - 1,
        ),
    )
    return SolverRun(
        method="ipopt",
        status=status,
        message=str(message),
        trace=callbacks.trace,
        final_x=solution,
        final_multiplier=final_multiplier,
        counters={
            **callbacks.counts,
            "accepted_iterations": len(callbacks.trace) - 1,
            "primal_linear_solves": len(callbacks.trace) - 1,
            "correction_linear_solves": 0,
            "rejected_trials": 0,
        },
        config=asdict(config),
        metadata={
            "native_status": int(info.get("status", -999)),
            "options": options,
            "jacobian_nonzeros": int(
                problem.affine_matrix.nnz + problem.dimension
            ),
            "stored_pairs": len(callbacks.trace),
            "trace_evaluator": "online independent pair residual",
        },
    )
