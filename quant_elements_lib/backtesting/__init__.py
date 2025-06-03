# quant_elements_lib/backtesting/__init__.py

"""
The backtesting module provides tools for evaluating trading strategies and signals,
including frameworks for cross-validation, walk-forward analysis, and evaluation
metrics like Rademacher Anti-Serum (RAS) bounds.
"""

from .framework import generate_k_fold_cv_indices_ts, generate_walk_forward_indices
from .ras_evaluation import (
    calculate_empirical_rademacher_complexity,
    calculate_ras_lower_bound_ic,
    calculate_ras_lower_bound_sr,
)

__all__ = [
    "generate_k_fold_cv_indices_ts",
    "generate_walk_forward_indices",
    "calculate_empirical_rademacher_complexity",
    "calculate_ras_lower_bound_ic",
    "calculate_ras_lower_bound_sr",
]
