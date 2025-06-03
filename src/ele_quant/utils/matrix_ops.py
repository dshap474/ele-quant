import numpy as np

def shrink_eigenvalues_spiked_model(
    eigenvalues: np.ndarray, gamma_N_T_ratio: float
) -> np.ndarray:
    """
    Shrinks eigenvalues using the spiked covariance model formula.

    The formula applied is ell_lambda = lambda_val - (gamma_N_T_ratio * lambda_val) / (lambda_val - 1),
    derived from Ledoit & Wolf (2004) type estimators under certain assumptions,
    or related to random matrix theory results for spiked models where lambda_val > 1 + sqrt(gamma).
    This specific simplified form is often used for eigenvalues > 1.

    Args:
        eigenvalues: A 1D numpy array of eigenvalues.
        gamma_N_T_ratio: The ratio N/T (number of variables / number of observations).

    Returns:
        A new numpy array with shrunk eigenvalues. Eigenvalues <= 1 are not shrunk.
    """
    if not isinstance(eigenvalues, np.ndarray):
        raise TypeError("eigenvalues must be a numpy array.")
    if eigenvalues.ndim != 1:
        raise ValueError("eigenvalues must be a 1D numpy array.")
    if gamma_N_T_ratio < 0:
        raise ValueError("gamma_N_T_ratio must be non-negative.")

    shrunk_eigenvalues = np.copy(eigenvalues).astype(float) # Ensure float for calculations

    for i, lambda_val in enumerate(shrunk_eigenvalues):
        if lambda_val > 1.0: # Apply formula only if lambda_val > 1
            denominator = lambda_val - 1.0
            if np.isclose(denominator, 0): # Avoid division by zero if lambda_val is very close to 1
                # In this case, shrinkage term would be very large, effectively pushing lambda to 0 or negative.
                # Depending on interpretation, could set to a small positive value, or 1.
                # For now, if lambda_val is extremely close to 1, we don't shrink it significantly.
                # Or, one might argue the formula is ill-defined here.
                # Let's keep it as original value if denominator is too small.
                # Or, more robustly, the theory implies these small eigenvalues are noise.
                # However, the provided formula is usually for "spikes" (large eigenvalues).
                # Let's assume eigenvalues significantly greater than 1 are of interest for this formula.
                pass # Keep original if lambda_val is ~1
            else:
                shrunk_lambda = lambda_val - (gamma_N_T_ratio * lambda_val) / denominator
                # Ensure non-negativity, though this formula for lambda_val > 1 and reasonable gamma
                # should ideally yield positive if lambda_val is large enough.
                # If shrunk_lambda becomes non-positive, it might indicate the eigenvalue is
                # not large enough relative to gamma for the formula to be stable.
                shrunk_eigenvalues[i] = max(0.0, shrunk_lambda) # Ensure non-negative
        # else: lambda_val <= 1, do not shrink (or could be set to a floor like max(epsilon, lambda_val))
        # For now, values <=1 are returned as is.

    return shrunk_eigenvalues

def shrink_eigenvalues_linear(
    eigenvalues: np.ndarray, k1_coeff: float, k2_coeff: float
) -> np.ndarray:
    """
    Shrinks eigenvalues using a linear shrinkage formula.

    The formula applied is shrunk_lambda = k1_coeff * lambda_val - k2_coeff.
    The result is floored at 0 to ensure non-negativity.

    Args:
        eigenvalues: A 1D numpy array of eigenvalues.
        k1_coeff: Multiplicative coefficient for linear shrinkage.
        k2_coeff: Subtractive coefficient for linear shrinkage.

    Returns:
        A new numpy array with shrunk, non-negative eigenvalues.
    """
    if not isinstance(eigenvalues, np.ndarray):
        raise TypeError("eigenvalues must be a numpy array.")
    if eigenvalues.ndim != 1:
        raise ValueError("eigenvalues must be a 1D numpy array.")

    shrunk_lambda = k1_coeff * eigenvalues - k2_coeff
    return np.maximum(0, shrunk_lambda)

def align_eigenvectors_procrustes(
    B_current: np.ndarray, B_previous: np.ndarray
) -> np.ndarray:
    """
    Aligns current eigenvectors (or factor loadings) to previous ones using Procrustes analysis.

    This method finds an optimal rotation matrix X_rotation such that
    ||B_previous - B_current @ X_rotation||_F is minimized, where ||.||_F is the Frobenius norm.
    The solution involves the SVD of B_previous.T @ B_current.

    Args:
        B_current: Numpy array (N x K) of current eigenvectors or factor loadings.
        B_previous: Numpy array (N x K) of previous period's eigenvectors or factor loadings
                    to which B_current should be aligned.

    Returns:
        Numpy array (N x K) of B_current aligned to B_previous.
    """
    if not isinstance(B_current, np.ndarray) or not isinstance(B_previous, np.ndarray):
        raise TypeError("B_current and B_previous must be numpy arrays.")
    if B_current.shape != B_previous.shape:
        raise ValueError("B_current and B_previous must have the same shape.")
    if B_current.ndim != 2:
        raise ValueError("B_current and B_previous must be 2D arrays (N x K).")

    # Calculate A = B_previous.T @ B_current
    A = B_previous.T @ B_current

    # Perform SVD on A: U_svd @ diag(s_svd) @ Vt_svd = A
    U_svd, s_svd, Vt_svd = np.linalg.svd(A) # Vt_svd is V.T

    # Calculate the optimal rotation matrix X_rotation = V_svd @ U_svd.T
    # Note: numpy.linalg.svd returns V.T as Vt_svd
    X_rotation = Vt_svd.T @ U_svd.T

    # Apply the rotation: B_aligned = B_current @ X_rotation
    B_aligned = B_current @ X_rotation

    return B_aligned
