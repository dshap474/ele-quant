# In quant_elements_lib/utils/__init__.py

# Attempt to import from matrix_ops, assuming it might be created later
_has_matrix_ops = False
_has_align_eigenvectors = False
_has_shrinkage = False

try:
    from . import matrix_ops # Check if module exists
    _has_matrix_ops = True

    try:
        from .matrix_ops import align_eigenvectors
        _has_align_eigenvectors = True
    except ImportError:
        # align_eigenvectors not in matrix_ops or matrix_ops not yet fully defined
        pass

    try:
        from .matrix_ops import shrink_eigenvalues_spiked_model, shrink_eigenvalues_linear
        _has_shrinkage = True
    except ImportError:
        # shrinkage functions not in matrix_ops or matrix_ops not yet fully defined
        pass
except ImportError:
    # matrix_ops.py does not exist
    pass

from .regression import ordinary_least_squares, weighted_least_squares
# ... other imports from utils submodules like preprocessing, linalg, kalman_filter etc.

__all__ = [
    "ordinary_least_squares", # from Ch4
    "weighted_least_squares", # from Ch4
    # ... other functions from utils
]

if _has_align_eigenvectors:
    __all__.append("align_eigenvectors")

if _has_shrinkage:
    __all__.extend(["shrink_eigenvalues_spiked_model", "shrink_eigenvalues_linear"])

# Ensure __all__ is sorted for consistency and uniqueness
__all__ = sorted(list(set(__all__)))
