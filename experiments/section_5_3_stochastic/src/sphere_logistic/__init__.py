"""Multiclass logistic problem and common KKT evaluator for Section 5.3."""

from .evaluator import evaluate_metrics, least_squares_multiplier
from .problem import (
    MulticlassData,
    MulticlassSphereLogistic,
    load_libsvm_multiclass,
    sha256_file,
)
from .ssqp import SSQPConfig, SSQPResult, run_ssqp

__all__ = [
    "MulticlassData",
    "MulticlassSphereLogistic",
    "SSQPConfig",
    "SSQPResult",
    "evaluate_metrics",
    "least_squares_multiplier",
    "load_libsvm_multiclass",
    "run_ssqp",
    "sha256_file",
]
