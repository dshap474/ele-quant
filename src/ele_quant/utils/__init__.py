"""
Utilities for the Elements of Quantitative Investing project.
"""
from .kalman_filter import KalmanFilter
# from .slugify import slugify # Commented out: Missing src/ele_quant/utils/slugify.py
from .preprocessing import winsorize_series

__all__ = [
    "KalmanFilter",
    # "slugify", # Commented out: Missing src/ele_quant/utils/slugify.py
    "winsorize_series",
]
