import numpy as np
import pandas as pd # Keep for potential future use in this file
from typing import List, Tuple # Keep for potential future use

# Content from Task B.2 of Chapter 4 (decompose_alpha)
def decompose_alpha(alpha_total: pd.Series, B_loadings: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Decomposes the total alpha into a spanned part (explained by factor loadings B)
    and an orthogonal part (unexplained by B).

    alpha_total = alpha_spanned + alpha_orthogonal
    alpha_spanned = B @ lambda_spanned

    Args:
        alpha_total (pd.Series): Series of total alphas for N assets (index=asset_names).
        B_loadings (pd.DataFrame): Factor loadings matrix (N x K) for N assets and K factors
                                   (index=asset_names, columns=factor_names).

    Returns:
        tuple[pd.Series, pd.Series, pd.Series]:
            - alpha_spanned (pd.Series): The part of alpha explained by factor loadings B.
            - alpha_orthogonal (pd.Series): The part of alpha orthogonal to factor loadings B.
            - lambda_spanned (pd.Series): The factor exposures that generate alpha_spanned (Kx1).

    Raises:
        ValueError: If indices of alpha_total and B_loadings do not align.
    """
    if not alpha_total.index.equals(B_loadings.index):
        raise ValueError("Index of alpha_total must match index of B_loadings.")

    # Ensure N x K for B_loadings
    B = B_loadings.values
    # Ensure N x 1 for alpha_total
    a_total_vals = alpha_total.values.reshape(-1, 1)

    # lambda_spanned = (B^T B)^-1 B^T alpha_total
    try:
        # Check for perfect multicollinearity or non-invertibility
        BtB = B.T @ B
        if np.linalg.cond(BtB) > 1e10: # Condition number check for stability
             # Handle near-singular cases, e.g. by pseudo-inverse or raising specific error
             # For now, let linalg.solve raise the error or use pseudo-inverse
             # print("Warning: High condition number for B.T @ B in decompose_alpha")
             lambda_s_vals = np.linalg.pinv(BtB) @ (B.T @ a_total_vals)

        else:
            lambda_s_vals = np.linalg.solve(BtB, B.T @ a_total_vals)

    except np.linalg.LinAlgError as e:
        # Fallback to pseudo-inverse if B.T @ B is singular (e.g. K > N or perfect multicollinearity)
        # This can happen if factors are not independent or N < K
        # print(f"Warning: Using pseudo-inverse in decompose_alpha due to LinAlgError: {e}")
        lambda_s_vals = np.linalg.pinv(B.T @ B) @ (B.T @ a_total_vals)


    lambda_s = pd.Series(lambda_s_vals.flatten(), index=B_loadings.columns, name="lambda_spanned")

    # alpha_spanned = B @ lambda_spanned
    alpha_s_vals = B @ lambda_s_vals
    alpha_s = pd.Series(alpha_s_vals.flatten(), index=alpha_total.index, name="alpha_spanned")

    # alpha_orthogonal = alpha_total - alpha_spanned
    alpha_o_vals = a_total_vals - alpha_s_vals
    alpha_o = pd.Series(alpha_o_vals.flatten(), index=alpha_total.index, name="alpha_orthogonal")

    return alpha_s, alpha_o, lambda_s

# New function to be added / updated
def choose_num_factors_threshold(eigenvalues: np.ndarray, N_assets: int, T_observations: int) -> int:
    """
    Chooses the number of factors based on the eigenvalue threshold method.
    Condition: eigenvalue >= 1 + sqrt(N/T)
    (Reference: Textbook Eq. 7.34, implies gamma = N/T for eigenvalues of sample cov matrix R'R/T,
     or T/N if eigenvalues of R R'/N)
    Assuming eigenvalues are from (1/T)R'R where R is T x N (demeaned returns).

    Args:
        eigenvalues (np.ndarray): Sorted array of eigenvalues of the sample
                                  covariance matrix (descending order).
        N_assets (int): Number of assets (N).
        T_observations (int): Number of time series observations (T).

    Returns:
        int: The suggested number of factors.
    """
    if N_assets <= 0 or T_observations <= 0:
        raise ValueError("N_assets and T_observations must be positive.")
    # Redundant due to above check, but explicit: if T_observations == 0: return 0

    gamma = N_assets / T_observations
    threshold = 1 + np.sqrt(gamma)

    num_factors = 0
    for eig_val in eigenvalues:
        if eig_val >= threshold:
            num_factors += 1
        else:
            break # Eigenvalues are sorted

    # Ensure at least 1 factor if any eigenvalue meets threshold, but not more than N_assets
    # And not more than available eigenvalues
    # If num_factors is 0 (no eigenvalue met threshold), max(0,1 if 0 > 0 else 0) = 0.
    # If num_factors is >0, max(num_factors, 1) = num_factors.
    # So, max(num_factors, 1 if num_factors > 0 else 0) simplifies to just num_factors if num_factors must be >=0.
    # However, the logic "Ensure at least 1 factor if any eigenvalue meets threshold" means if num_factors > 0, it should be at least 1.
    # This is already true if num_factors is calculated by incrementing.
    # The main constraints are:
    # 1. Not more than N_assets.
    # 2. Not more than the number of available eigenvalues.
    # 3. If after thresholding num_factors is 0, it should remain 0.
    # Let's refine the return statement to be clearer and match the logic:

    # If no eigenvalues meet the threshold, num_factors is 0.
    # Otherwise, it's the count, capped by len(eigenvalues) and N_assets.
    if num_factors == 0:
        return 0
    else:
        return min(num_factors, len(eigenvalues), N_assets)

# Note: Implementation of Bai & Ng (Eq. 7.38) is deferred due to complexity.
