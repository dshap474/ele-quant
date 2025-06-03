"""
The ele_quant.utils package provides various utility functions and classes
for quantitative finance, covering areas like regression, Kalman filtering,
matrix operations, linear algebra, data preprocessing, and covariance estimation.
"""

from .regression import ordinary_least_squares, weighted_least_squares
from .kalman_filter import KalmanFilter
from .matrix_ops import (
    shrink_eigenvalues_spiked_model,
    shrink_eigenvalues_linear,
    align_eigenvectors_procrustes,
)
from .linalg import (
    woodbury_sherman_morrison_inverse,
    matrix_determinant_lemma,
)
from .preprocessing import winsorize_series
from .covariance_estimation import (
    calculate_ledoit_wolf_shrinkage,
    calculate_newey_west_covariance,
)

# Potentially import from utils.py if it contains public utilities
# from .utils import ...

__all__ = [
    # From regression.py
    "ordinary_least_squares",
    "weighted_least_squares",
    # From kalman_filter.py
    "KalmanFilter",
    # From matrix_ops.py
    "shrink_eigenvalues_spiked_model",
    "shrink_eigenvalues_linear",
    "align_eigenvectors_procrustes",
    # From linalg.py
    "woodbury_sherman_morrison_inverse",
    "matrix_determinant_lemma",
    # From preprocessing.py
    "winsorize_series",
    # From covariance_estimation.py
    "calculate_ledoit_wolf_shrinkage",
    "calculate_newey_west_covariance",
]
