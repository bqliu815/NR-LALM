"""Stochastic algorithms used in the Section 5.3 comparison."""

from .mlalm import MLALMConfig, MLALMResult, OracleCounts, run_mlalm
from .nr_lalm import NRLALMConfig, NRLALMResult, NRWorkCounts, run_nr_lalm
from .spider import (
    ProjectedSPIDER,
    SPIDERConfig,
    largest_horizon_for_budget,
    largest_horizon_for_scaled_budget,
    spider_total_calls,
    spider_total_calls_scaled,
)

__all__ = [
    "MLALMConfig",
    "MLALMResult",
    "NRLALMConfig",
    "NRLALMResult",
    "NRWorkCounts",
    "OracleCounts",
    "ProjectedSPIDER",
    "SPIDERConfig",
    "largest_horizon_for_budget",
    "largest_horizon_for_scaled_budget",
    "run_mlalm",
    "run_nr_lalm",
    "spider_total_calls",
    "spider_total_calls_scaled",
]
