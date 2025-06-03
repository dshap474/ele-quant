"""
Utilities for the Elements of Quantitative Investing project.
"""
from .kalman_filter import KalmanFilter
from .slugify import slugify

__all__ = [
    "KalmanFilter",
    "slugify",
]
