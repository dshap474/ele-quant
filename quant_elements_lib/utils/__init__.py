# quant_elements_lib/utils/__init__.py
from .regression import ordinary_least_squares, weighted_least_squares
# Add other future imports like kalman_filter, matrix_ops, preprocessing etc.
__all__ = [
    'ordinary_least_squares',
    'weighted_least_squares'
]
