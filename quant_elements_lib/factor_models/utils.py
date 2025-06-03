import numpy as np
import pandas as pd

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


def choose_num_factors_threshold(eigenvalues: np.ndarray, N_assets: int, T_observations: int) -> int:
    """
    Estimates the number of factors using a threshold rule based on eigenvalue magnitudes.

    This method is related to the Marcenko-Pastur law and is described in
    "Elements of Quantitative Investing" Chapter 7, around Eq. 7.34.
    The threshold (1 + sqrt(N/T)) is typically applied to eigenvalues of the
    sample covariance matrix of standardized residuals, where N is the number of assets
    and T is the number of observations.

    Args:
        eigenvalues (np.ndarray): A 1D NumPy array of eigenvalues of the sample
                                  covariance matrix, assumed to be sorted in
                                  descending order.
        N_assets (int): The number of assets (N).
        T_observations (int): The number of time series observations (T).

    Returns:
        int: The estimated number of factors.

    Raises:
        ValueError: If T_observations is zero or negative, N_assets is negative,
                    or if eigenvalues is not a 1D NumPy array.
    """
    if T_observations <= 0:
        raise ValueError("T_observations must be positive.")
    if N_assets < 0:
        raise ValueError("N_assets cannot be negative.")
    if N_assets == 0: # If there are no assets, there are no factors
        return 0

    if not isinstance(eigenvalues, np.ndarray) or eigenvalues.ndim != 1:
        raise ValueError("Eigenvalues must be a 1D NumPy array.")

    if len(eigenvalues) == 0:
        return 0

    gamma = N_assets / T_observations
    threshold = 1 + np.sqrt(gamma)

    num_factors = 0
    for eig_val in eigenvalues:
        if eig_val >= threshold:
            num_factors += 1
        else:
            break

    return num_factors
