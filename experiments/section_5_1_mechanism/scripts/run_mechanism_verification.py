#!/usr/bin/env python3
"""Run the paper-facing unified mechanism-verification experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 cluster compatibility
    import tomli as tomllib
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

METHOD_COLORS = {
    "base": "#0072B2",
    "corrected": "#D55E00",
}
NEUTRAL_GRAY = "#666666"

AXIS_LABELS = {
    "certificate_x": r"model-step bound $\Delta$",
    "certificate_y": r"multiplier bound $\Lambda$",
    "order_x": r"step norm $\Vert p\Vert$",
    "order_y": r"error norm $\Vert d\Vert$",
}

PANEL_TITLES = {
    "certificate": "(a) Sufficient parameter regions",
    "order": "(b) Constraint linearization error",
}

FIGURE_CAPTION = r"""\caption{Mechanism verification on one fixed
\((n,m)=(100,20)\) problem.  \textup{(a)} Sufficient parameter regions for
the base and corrected methods.  \textup{(b)} Norms of the base and SOC
constraint linearization errors over 32 directions; bands show interquartile
ranges.}
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_tree_sha256(project_root: Path) -> str:
    """Hash the computational Python source with stable relative paths."""
    source_files = [
        project_root / "scripts" / "run_mechanism_verification.py",
        *sorted((project_root / "src").rglob("*.py")),
    ]
    digest = hashlib.sha256()
    for path in source_files:
        digest.update(
            path.relative_to(project_root).as_posix().encode("utf-8")
        )
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def archive_source_tree(
    project_root: Path, snapshot_root: Path
) -> None:
    """Archive exactly the Python files covered by source_tree_sha256."""
    source_files = [
        project_root / "scripts" / "run_mechanism_verification.py",
        *sorted((project_root / "src").rglob("*.py")),
    ]
    for source_path in source_files:
        relative_path = source_path.relative_to(project_root)
        target_path = snapshot_root / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)


def array_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array, dtype=np.float64)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def canonical_qr(raw: np.ndarray) -> np.ndarray:
    q_matrix, r_matrix = np.linalg.qr(raw, mode="reduced")
    signs = np.sign(np.diag(r_matrix))
    signs[signs == 0.0] = 1.0
    return q_matrix * signs


def scalar_feasible_root() -> float:
    root = 0.45
    for _ in range(20):
        root -= (2.0 * root - np.cos(root)) / (
            2.0 + np.sin(root)
        )
    return float(root)


def fit_log_slope(
    step_scales: np.ndarray, errors: np.ndarray
) -> dict[str, float]:
    log_steps = np.log(step_scales)
    log_errors = np.log(errors)
    slope, intercept = np.polyfit(log_steps, log_errors, deg=1)
    prediction = slope * log_steps + intercept
    centered = log_errors - np.mean(log_errors)
    total = float(centered @ centered)
    residual = float(
        (log_errors - prediction) @ (log_errors - prediction)
    )
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": 1.0 if total == 0.0 else 1.0 - residual / total,
    }


def certificate_mask(
    deltas: np.ndarray,
    lambda_radii: np.ndarray,
    *,
    beta: float,
    rho: float,
    initial_lambda_norm: float,
    variant: str,
) -> np.ndarray:
    gradient_bound = 1.0
    jacobian_bound = 3.0
    objective_lipschitz = 1.0
    jacobian_lipschitz = 1.0
    licq = 1.0

    delta = deltas[:, None]
    lambda_radius = lambda_radii[None, :]
    kappa_linearization = jacobian_lipschitz / 2.0
    kappa_correction = jacobian_lipschitz / (2.0 * licq)
    kappa_fourth = jacobian_lipschitz**3 / (8.0 * licq**2)

    if variant == "base":
        gamma_q = np.zeros_like(delta)
        gamma_d = np.full_like(delta, kappa_linearization)
        displacement = np.ones_like(delta)
    elif variant == "corrected":
        gamma_q = np.full_like(delta, kappa_correction)
        gamma_d = kappa_fourth * delta**2
        displacement = 1.0 + kappa_correction * delta
    else:
        raise ValueError(f"unknown certificate variant: {variant}")

    multiplier_condition = lambda_radius >= (
        gradient_bound
        + beta * delta
        + rho * jacobian_bound * gamma_d * delta**2
    ) / licq
    step_condition = (
        gradient_bound / beta
        + 3.0
        * jacobian_bound
        * lambda_radius
        / (beta + rho * licq**2)
        <= delta
    )
    model_constant = (
        gradient_bound * gamma_q
        + 0.5 * objective_lipschitz * displacement**2
        + gamma_d
        * (3.0 * lambda_radius + rho * jacobian_bound * delta)
        + 0.5 * rho * gamma_d**2 * delta**2
    )
    descent_condition = model_constant <= 3.0 * beta / 8.0
    a_constant = beta + rho * jacobian_bound * gamma_d * delta
    b_constant = (
        a_constant
        + (objective_lipschitz + jacobian_lipschitz * lambda_radius)
        * displacement
    )
    multiplier_increment = (
        4.0 / licq**2 * np.maximum(a_constant**2, b_constant**2)
    )
    penalty_condition = rho >= 8.0 * multiplier_increment / beta
    initialization_condition = lambda_radius >= initial_lambda_norm
    return (
        multiplier_condition
        & step_condition
        & descent_condition
        & penalty_condition
        & initialization_condition
    )


def certificate_point_metrics(
    *,
    delta: float,
    lambda_radius: float,
    beta: float,
    rho: float,
    initial_lambda_norm: float,
    variant: str,
) -> dict[str, float | bool]:
    gradient_bound = 1.0
    jacobian_bound = 3.0
    objective_lipschitz = 1.0
    jacobian_lipschitz = 1.0
    licq = 1.0
    kappa_linearization = jacobian_lipschitz / 2.0
    kappa_correction = jacobian_lipschitz / (2.0 * licq)
    kappa_fourth = jacobian_lipschitz**3 / (8.0 * licq**2)
    if variant == "base":
        gamma_q = 0.0
        gamma_d = kappa_linearization
        displacement = 1.0
    elif variant == "corrected":
        gamma_q = kappa_correction
        gamma_d = kappa_fourth * delta**2
        displacement = 1.0 + kappa_correction * delta
    else:
        raise ValueError(f"unknown certificate variant: {variant}")

    lambda_lower_bound = (
        gradient_bound
        + beta * delta
        + rho * jacobian_bound * gamma_d * delta**2
    ) / licq
    step_left_side = (
        gradient_bound / beta
        + 3.0
        * jacobian_bound
        * lambda_radius
        / (beta + rho * licq**2)
    )
    model_constant = (
        gradient_bound * gamma_q
        + 0.5 * objective_lipschitz * displacement**2
        + gamma_d
        * (3.0 * lambda_radius + rho * jacobian_bound * delta)
        + 0.5 * rho * gamma_d**2 * delta**2
    )
    a_constant = beta + rho * jacobian_bound * gamma_d * delta
    b_constant = (
        a_constant
        + (objective_lipschitz + jacobian_lipschitz * lambda_radius)
        * displacement
    )
    multiplier_increment = (
        4.0 / licq**2 * max(a_constant**2, b_constant**2)
    )
    rho_lower_bound = 8.0 * multiplier_increment / beta
    passed = (
        lambda_radius >= lambda_lower_bound
        and step_left_side <= delta
        and model_constant <= 3.0 * beta / 8.0
        and rho >= rho_lower_bound
        and lambda_radius >= initial_lambda_norm
    )
    return {
        "passed": passed,
        "lambda_lower_bound": lambda_lower_bound,
        "step_left_side": step_left_side,
        "model_constant": model_constant,
        "model_upper_bound": 3.0 * beta / 8.0,
        "rho_lower_bound": rho_lower_bound,
        "initial_lambda_norm": initial_lambda_norm,
    }


def boundaries(
    deltas: np.ndarray,
    lambda_radii: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.full(deltas.shape, np.nan)
    upper = np.full(deltas.shape, np.nan)
    for index, row in enumerate(mask):
        feasible = lambda_radii[row]
        if feasible.size:
            lower[index] = feasible[0]
            upper[index] = feasible[-1]
    return lower, upper


def quantile_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "minimum": float(np.min(values)),
        "q1": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q3": float(np.quantile(values, 0.75)),
        "maximum": float(np.max(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    protocol_status = str(config.get("status", ""))
    if protocol_status != "FROZEN":
        raise RuntimeError("public runner requires status = FROZEN")

    problem_config = config["problem"]
    certificate_config = config["certificate"]
    order_config = config["order"]
    display_config = config["display"]
    base_method_label = str(display_config["base_method"]).strip()
    corrected_method_label = str(
        display_config["corrected_method"]
    ).strip()
    if not base_method_label or not corrected_method_label:
        raise ValueError("display method labels must be nonempty")
    n = int(problem_config["dimension"])
    m = int(problem_config["constraints"])
    if not 0 < m <= n:
        raise ValueError("require 0 < constraints <= dimension")

    problem_rng = np.random.default_rng(
        int(problem_config["problem_seed"])
    )
    column_basis = canonical_qr(problem_rng.normal(size=(n, m)))
    mixing = column_basis.T
    output_rotation = canonical_qr(
        problem_rng.normal(size=(m, m))
    )
    row_orthogonality_error = float(
        np.linalg.norm(mixing @ mixing.T - np.eye(m), ord=2)
    )
    rotation_error = float(
        np.linalg.norm(
            output_rotation.T @ output_rotation - np.eye(m),
            ord=2,
        )
    )

    root = scalar_feasible_root()
    base_coordinates = np.full(m, root)
    x0 = mixing.T @ base_coordinates
    initial_lambda_norm = float(
        certificate_config.get("initial_lambda_norm", 1049.0)
    )
    lambda0 = initial_lambda_norm * output_rotation[:, 0]
    constraint_x0 = output_rotation @ (
        2.0 * base_coordinates - np.cos(base_coordinates)
    )

    beta = float(certificate_config["beta"])
    rho = float(certificate_config["rho"])
    strict_delta = float(certificate_config["strict_delta"])
    strict_lambda = float(certificate_config["strict_lambda"])
    common_delta_raw = certificate_config.get("common_delta")
    common_lambda_raw = certificate_config.get("common_lambda")
    if (common_delta_raw is None) != (common_lambda_raw is None):
        raise ValueError(
            "common_delta and common_lambda must appear together"
        )
    common_delta = (
        None if common_delta_raw is None else float(common_delta_raw)
    )
    common_lambda = (
        None if common_lambda_raw is None else float(common_lambda_raw)
    )
    deltas = np.linspace(
        float(certificate_config["delta_min"]),
        float(certificate_config["delta_max"]),
        int(certificate_config["delta_points"]),
    )
    lambda_radii = np.linspace(
        float(certificate_config["lambda_min"]),
        float(certificate_config["lambda_max"]),
        int(certificate_config["lambda_points"]),
    )
    base_mask = certificate_mask(
        deltas,
        lambda_radii,
        beta=beta,
        rho=rho,
        initial_lambda_norm=initial_lambda_norm,
        variant="base",
    )
    corrected_mask = certificate_mask(
        deltas,
        lambda_radii,
        beta=beta,
        rho=rho,
        initial_lambda_norm=initial_lambda_norm,
        variant="corrected",
    )
    if np.any(base_mask & ~corrected_mask):
        raise RuntimeError("certificate containment failed")
    strict_corrected_membership = bool(
        certificate_mask(
            np.array([strict_delta]),
            np.array([strict_lambda]),
            beta=beta,
            rho=rho,
            initial_lambda_norm=initial_lambda_norm,
            variant="corrected",
        )[0, 0]
    )
    strict_base_membership = bool(
        certificate_mask(
            np.array([strict_delta]),
            np.array([strict_lambda]),
            beta=beta,
            rho=rho,
            initial_lambda_norm=initial_lambda_norm,
            variant="base",
        )[0, 0]
    )
    common_base_membership = None
    common_corrected_membership = None
    if common_delta is not None and common_lambda is not None:
        common_base_membership = bool(
            certificate_mask(
                np.array([common_delta]),
                np.array([common_lambda]),
                beta=beta,
                rho=rho,
                initial_lambda_norm=initial_lambda_norm,
                variant="base",
            )[0, 0]
        )
        common_corrected_membership = bool(
            certificate_mask(
                np.array([common_delta]),
                np.array([common_lambda]),
                beta=beta,
                rho=rho,
                initial_lambda_norm=initial_lambda_norm,
                variant="corrected",
            )[0, 0]
        )
    base_lower, base_upper = boundaries(
        deltas, lambda_radii, base_mask
    )
    corrected_lower, corrected_upper = boundaries(
        deltas, lambda_radii, corrected_mask
    )

    strict_point_metrics = {
        variant: certificate_point_metrics(
            delta=strict_delta,
            lambda_radius=strict_lambda,
            beta=beta,
            rho=rho,
            initial_lambda_norm=initial_lambda_norm,
            variant=variant,
        )
        for variant in ("base", "corrected")
    }
    common_point_metrics = None
    if common_delta is not None and common_lambda is not None:
        common_point_metrics = {
            variant: certificate_point_metrics(
                delta=common_delta,
                lambda_radius=common_lambda,
                beta=beta,
                rho=rho,
                initial_lambda_norm=initial_lambda_norm,
                variant=variant,
            )
            for variant in ("base", "corrected")
        }
    step_scales = np.geomspace(
        float(order_config["step_min"]),
        float(order_config["step_max"]),
        int(order_config["step_count"]),
    )
    direction_rng = np.random.default_rng(
        int(problem_config["direction_seed"])
    )
    direction_count = int(problem_config["direction_count"])
    reduced_directions: list[np.ndarray] = []
    base_error_rows: list[np.ndarray] = []
    corrected_error_rows: list[np.ndarray] = []
    base_fits: list[dict[str, float]] = []
    corrected_fits: list[dict[str, float]] = []
    solve_residual_rows: list[np.ndarray] = []

    for _ in range(direction_count):
        reduced_direction = direction_rng.normal(size=m)
        reduced_direction /= np.linalg.norm(reduced_direction)
        direction = mixing.T @ reduced_direction
        reduced_directions.append(reduced_direction)
        base_errors: list[float] = []
        corrected_errors: list[float] = []
        solve_residuals: list[float] = []

        for step_scale in step_scales:
            step = step_scale * direction
            reduced_step = mixing @ step
            base_remainder = output_rotation @ (
                np.cos(base_coordinates)
                - np.cos(base_coordinates + reduced_step)
                - np.sin(base_coordinates) * reduced_step
            )
            trial_jacobian = (
                output_rotation
                @ np.diag(
                    2.0 + np.sin(base_coordinates + reduced_step)
                )
                @ mixing
            )
            correction = -trial_jacobian.T @ np.linalg.solve(
                trial_jacobian @ trial_jacobian.T,
                base_remainder,
            )
            reduced_correction = mixing @ correction
            second_remainder = output_rotation @ (
                np.cos(base_coordinates + reduced_step)
                - np.cos(
                    base_coordinates
                    + reduced_step
                    + reduced_correction
                )
                - np.sin(base_coordinates + reduced_step)
                * reduced_correction
            )
            corrected_remainder = (
                base_remainder
                + trial_jacobian @ correction
                + second_remainder
            )
            base_norm = float(np.linalg.norm(base_remainder))
            base_errors.append(base_norm)
            corrected_errors.append(
                float(np.linalg.norm(corrected_remainder))
            )
            solve_residuals.append(
                float(
                    np.linalg.norm(
                        base_remainder
                        + trial_jacobian @ correction
                    )
                    / base_norm
                )
            )

        base_array = np.asarray(base_errors)
        corrected_array = np.asarray(corrected_errors)
        base_error_rows.append(base_array)
        corrected_error_rows.append(corrected_array)
        solve_residual_rows.append(np.asarray(solve_residuals))
        base_fits.append(fit_log_slope(step_scales, base_array))
        corrected_fits.append(
            fit_log_slope(step_scales, corrected_array)
        )

    base_errors = np.stack(base_error_rows)
    corrected_errors = np.stack(corrected_error_rows)
    solve_residuals = np.stack(solve_residual_rows)
    base_slopes = np.array([fit["slope"] for fit in base_fits])
    corrected_slopes = np.array(
        [fit["slope"] for fit in corrected_fits]
    )
    base_r_squared = np.array(
        [fit["r_squared"] for fit in base_fits]
    )
    corrected_r_squared = np.array(
        [fit["r_squared"] for fit in corrected_fits]
    )
    median_base = np.median(base_errors, axis=0)
    median_corrected = np.median(corrected_errors, axis=0)
    median_base_fit = fit_log_slope(step_scales, median_base)
    median_corrected_fit = fit_log_slope(
        step_scales, median_corrected
    )
    noise_scale = max(
        1.0,
        float(np.linalg.norm(base_coordinates)),
        float(np.max(step_scales)),
    )
    arithmetic_floor = (
        float(order_config["noise_multiplier"])
        * np.finfo(float).eps
        * noise_scale
    )

    passed = bool(
        strict_corrected_membership
        and not strict_base_membership
        and np.any(base_mask)
        and (
            common_base_membership is None
            or (
                common_base_membership
                and common_corrected_membership
            )
        )
        and np.max(np.abs(constraint_x0)) <= 1.0e-12
        and abs(np.linalg.norm(lambda0) - initial_lambda_norm)
        <= 1.0e-10
        and np.min(base_slopes)
        >= float(order_config["base_slope_min"])
        and np.max(base_slopes)
        <= float(order_config["base_slope_max"])
        and np.min(corrected_slopes)
        >= float(order_config["corrected_slope_min"])
        and np.max(corrected_slopes)
        <= float(order_config["corrected_slope_max"])
        and np.min(base_r_squared)
        >= float(order_config["minimum_r_squared"])
        and np.min(corrected_r_squared)
        >= float(order_config["minimum_r_squared"])
        and np.max(solve_residuals)
        <= float(order_config["maximum_solve_relative_residual"])
        and np.min(corrected_errors) > arithmetic_floor
    )

    raw = {
        "status": "PASS" if passed else "FAIL",
        "protocol_status": protocol_status,
        "problem": {
            "dimension": n,
            "constraints": m,
            "problem_seed": int(problem_config["problem_seed"]),
            "direction_seed": int(problem_config["direction_seed"]),
            "direction_count": direction_count,
            "root": root,
            "root_residual": abs(2.0 * root - np.cos(root)),
            "row_orthogonality_error": row_orthogonality_error,
            "output_rotation_error": rotation_error,
            "constraint_x0_norm": float(
                np.linalg.norm(constraint_x0)
            ),
            "lambda0_norm": float(np.linalg.norm(lambda0)),
            "problem_digest": array_sha256(
                mixing, output_rotation, x0, lambda0
            ),
            "global_constants": {
                "G": 1.0,
                "L_f": 1.0,
                "L_c": 1.0,
                "sigma": 1.0,
                "M": 3.0,
                "f_lower": -1.0,
            },
        },
        "certificate": {
            "beta": beta,
            "rho": rho,
            "initial_lambda_norm": initial_lambda_norm,
            "strict_delta": strict_delta,
            "strict_lambda": strict_lambda,
            "strict_base_membership": strict_base_membership,
            "strict_corrected_membership": (
                strict_corrected_membership
            ),
            "strict_point_metrics": strict_point_metrics,
            "common_delta": common_delta,
            "common_lambda": common_lambda,
            "common_base_membership": common_base_membership,
            "common_corrected_membership": (
                common_corrected_membership
            ),
            "common_point_metrics": common_point_metrics,
            "base_feasible_grid_points": int(np.sum(base_mask)),
            "corrected_feasible_grid_points": int(
                np.sum(corrected_mask)
            ),
            "delta_grid": deltas.tolist(),
            "base_lambda_lower": [
                None if np.isnan(value) else float(value)
                for value in base_lower
            ],
            "base_lambda_upper": [
                None if np.isnan(value) else float(value)
                for value in base_upper
            ],
            "corrected_lambda_lower": [
                None if np.isnan(value) else float(value)
                for value in corrected_lower
            ],
            "corrected_lambda_upper": [
                None if np.isnan(value) else float(value)
                for value in corrected_upper
            ],
        },
        "order": {
            "step_scales": step_scales.tolist(),
            "reduced_directions": [
                direction.tolist() for direction in reduced_directions
            ],
            "base_errors": base_errors.tolist(),
            "corrected_errors": corrected_errors.tolist(),
            "solve_relative_residuals": solve_residuals.tolist(),
            "base_fits": base_fits,
            "corrected_fits": corrected_fits,
            "base_slope_summary": quantile_summary(base_slopes),
            "corrected_slope_summary": quantile_summary(
                corrected_slopes
            ),
            "base_r_squared_summary": quantile_summary(
                base_r_squared
            ),
            "corrected_r_squared_summary": quantile_summary(
                corrected_r_squared
            ),
            "median_base_fit": median_base_fit,
            "median_corrected_fit": median_corrected_fit,
            "maximum_solve_relative_residual": float(
                np.max(solve_residuals)
            ),
            "minimum_corrected_error": float(
                np.min(corrected_errors)
            ),
            "arithmetic_floor": arithmetic_floor,
        },
    }

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.0,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "lines.linewidth": 1.2,
            "axes.linewidth": 1.0,
            "axes.edgecolor": "black",
            "axes.labelcolor": "black",
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.minor.width": 1.0,
            "ytick.minor.width": 1.0,
            "xtick.color": "black",
            "ytick.color": "black",
            "text.color": "black",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "hatch.linewidth": 1.0,
        }
    )
    figure, axes = plt.subplots(
        1, 2, figsize=(5.125, 2.45), facecolor="white"
    )

    ax = axes[0]
    corrected_finite = np.isfinite(corrected_lower)
    ax.fill_between(
        deltas[corrected_finite],
        corrected_lower[corrected_finite],
        corrected_upper[corrected_finite],
        color=METHOD_COLORS["corrected"],
        alpha=0.18,
        label=corrected_method_label,
    )
    ax.plot(
        deltas[corrected_finite],
        corrected_lower[corrected_finite],
        color=METHOD_COLORS["corrected"],
    )
    if np.any(base_mask):
        base_finite = np.isfinite(base_lower)
        ax.fill_between(
            deltas[base_finite],
            base_lower[base_finite],
            base_upper[base_finite],
            facecolor="none",
            edgecolor=METHOD_COLORS["base"],
            hatch="////",
            linewidth=1.0,
            label=base_method_label,
        )
    else:
        ax.text(
            0.052,
            1130.0,
            f"{base_method_label} certificate is empty\n"
            "on this parameter slice",
            ha="center",
            va="top",
            color=NEUTRAL_GRAY,
            fontsize=7.0,
        )
    ax.set_xlim(
        float(certificate_config["delta_min"]),
        float(certificate_config["delta_max"]),
    )
    ax.set_ylim(
        float(certificate_config["lambda_min"]),
        float(certificate_config["lambda_max"]),
    )
    if "delta_ticks" in certificate_config:
        ax.set_xticks(
            [float(value) for value in certificate_config["delta_ticks"]]
        )
    if "lambda_ticks" in certificate_config:
        ax.set_yticks(
            [
                float(value)
                for value in certificate_config["lambda_ticks"]
            ]
        )
    ax.set_xlabel(AXIS_LABELS["certificate_x"])
    ax.set_ylabel(AXIS_LABELS["certificate_y"])
    ax.set_title(PANEL_TITLES["certificate"])
    region_handles, region_labels = ax.get_legend_handles_labels()
    handle_by_label = dict(zip(region_labels, region_handles))
    ordered_region_labels = [
        label
        for label in (base_method_label, corrected_method_label)
        if label in handle_by_label
    ]
    region_legend = ax.legend(
        [handle_by_label[label] for label in ordered_region_labels],
        ordered_region_labels,
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=1.0,
        borderpad=0.25,
    )
    region_legend.get_frame().set_linewidth(1.0)
    ax.grid(color="#E0E0E0", linewidth=1.0)

    ax = axes[1]
    base_q1 = np.quantile(base_errors, 0.25, axis=0)
    base_q3 = np.quantile(base_errors, 0.75, axis=0)
    corrected_q1 = np.quantile(corrected_errors, 0.25, axis=0)
    corrected_q3 = np.quantile(corrected_errors, 0.75, axis=0)
    ax.fill_between(
        step_scales,
        base_q1,
        base_q3,
        color=METHOD_COLORS["base"],
        alpha=0.15,
    )
    ax.fill_between(
        step_scales,
        corrected_q1,
        corrected_q3,
        color=METHOD_COLORS["corrected"],
        alpha=0.15,
    )
    ax.loglog(
        step_scales,
        median_base,
        "o",
        color=METHOD_COLORS["base"],
        markerfacecolor="none",
        label=base_method_label,
    )
    ax.loglog(
        step_scales,
        np.exp(median_base_fit["intercept"])
        * step_scales ** median_base_fit["slope"],
        color=METHOD_COLORS["base"],
    )
    ax.loglog(
        step_scales,
        median_corrected,
        "s",
        color=METHOD_COLORS["corrected"],
        markerfacecolor="none",
        label=corrected_method_label,
    )
    ax.loglog(
        step_scales,
        np.exp(median_corrected_fit["intercept"])
        * step_scales ** median_corrected_fit["slope"],
        color=METHOD_COLORS["corrected"],
    )
    ax.set_xlabel(AXIS_LABELS["order_x"])
    ax.set_ylabel(AXIS_LABELS["order_y"])
    ax.set_title(PANEL_TITLES["order"])
    order_legend = ax.legend(
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=1.0,
        borderpad=0.25,
    )
    order_legend.get_frame().set_linewidth(1.0)
    ax.grid(which="both", color="#E0E0E0", linewidth=1.0)
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.tight_layout(w_pad=1.2)

    raw_path = output_dir / "raw.json"
    raw_path.write_text(
        json.dumps(raw, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pdf_path = output_dir / "mechanism_verification.pdf"
    png_path = output_dir / "mechanism_verification.png"
    figure.savefig(pdf_path, bbox_inches="tight", pad_inches=0.08)
    figure.savefig(
        png_path, dpi=300, bbox_inches="tight", pad_inches=0.08
    )
    plt.close(figure)

    caption_path = output_dir / "figure_caption.tex"
    caption_path.write_text(
        FIGURE_CAPTION.strip() + "\n",
        encoding="utf-8",
    )
    archived_config = output_dir / "config.toml"
    shutil.copy2(config_path, archived_config)
    source_snapshot = output_dir / "source_snapshot"
    archive_source_tree(PROJECT_ROOT, source_snapshot)
    command_path = output_dir / "command.txt"
    command_path.write_text(
        " ".join([sys.executable, *sys.argv]) + "\n",
        encoding="utf-8",
    )
    environment_path = output_dir / "environment.json"
    environment_path.write_text(
        json.dumps(
            {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "numpy": np.__version__,
                "matplotlib": matplotlib.__version__,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "protocol_status": protocol_status,
        "display_labels": {
            "base": base_method_label,
            "corrected": corrected_method_label,
        },
        "method_colors": METHOD_COLORS,
        "axis_labels": AXIS_LABELS,
        "panel_titles": PANEL_TITLES,
        "status": raw["status"],
        "config_sha256": sha256(archived_config),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "source_tree_sha256": source_tree_sha256(PROJECT_ROOT),
        "source_snapshot": source_snapshot.name,
        "source_snapshot_sha256": source_tree_sha256(
            source_snapshot
        ),
        "raw_sha256": sha256(raw_path),
        "pdf_sha256": sha256(pdf_path),
        "png_sha256": sha256(png_path),
        "caption_sha256": sha256(caption_path),
        "environment_sha256": sha256(environment_path),
        "command_sha256": sha256(command_path),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(
        {
            "output_dir": str(output_dir),
            "status": raw["status"],
            "protocol_status": protocol_status,
            "base_slope_summary": raw["order"][
                "base_slope_summary"
            ],
            "corrected_slope_summary": raw["order"][
                "corrected_slope_summary"
            ],
            "base_feasible_grid_points": raw["certificate"][
                "base_feasible_grid_points"
            ],
            "corrected_feasible_grid_points": raw["certificate"][
                "corrected_feasible_grid_points"
            ],
            "strict_base_membership": strict_base_membership,
            "strict_corrected_membership": (
                strict_corrected_membership
            ),
            "common_base_membership": common_base_membership,
            "common_corrected_membership": (
                common_corrected_membership
            ),
        },
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
