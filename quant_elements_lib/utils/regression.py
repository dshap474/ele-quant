import pandas as pd
import numpy as np
from typing import Tuple

def ordinary_least_squares(
    Y: pd.Series,
    X: pd.DataFrame,
    add_intercept: bool = True
) -> Tuple[pd.Series, pd.Series, float]:
    """
    Performs Ordinary Least Squares (OLS) regression.

    Solves Y = Xβ + ε for β_hat.

    Args:
        Y (pd.Series): Dependent variable (N x 1). Index should represent observations.
        X (pd.DataFrame): Predictor variables (N x P). Index should match Y.
        add_intercept (bool): If True, an intercept column (named 'intercept')
                              is added to the X matrix. Defaults to True.

    Returns:
        Tuple[pd.Series, pd.Series, float]: A tuple containing:
            - beta_hat (pd.Series): Estimated coefficients ((P+intercept) x 1).
                                    Index corresponds to X's columns (plus 'intercept').
            - residuals (pd.Series): Residuals (Y - X @ beta_hat) (N x 1).
                                     Index matches Y.
            - r_squared (float): R-squared value of the regression.

    Raises:
        ValueError: If Y and X do not have matching indices or if X is empty.
    """
    if not Y.index.equals(X.index):
        raise ValueError("Y and X must have the same index (observations).")
    if X.empty:
        raise ValueError("X DataFrame cannot be empty.")

    X_final = X.copy()
    if add_intercept:
        X_final['intercept'] = 1.0
        # Ensure intercept is the first column for conventional representation, though not strictly necessary for lstsq
        cols = ['intercept'] + [col for col in X.columns if col != 'intercept']
        X_final = X_final[cols]


    # Using numpy.linalg.lstsq
    # lstsq returns: coefficients, sum_of_squared_residuals, rank, singular_values
    beta_hat_values, sum_sq_res, rank, s = np.linalg.lstsq(X_final.values, Y.values, rcond=None)

    beta_hat = pd.Series(beta_hat_values, index=X_final.columns, name="coefficients")

    # Calculate residuals
    residuals_values = Y.values - (X_final.values @ beta_hat_values)
    residuals = pd.Series(residuals_values, index=Y.index, name="residuals")

    # Calculate R-squared
    y_mean = np.mean(Y.values)
    total_sum_squares = np.sum((Y.values - y_mean)**2)

    # If total_sum_squares is zero (e.g., Y is constant), R-squared is undefined or 0.
    # np.linalg.lstsq provides sum_sq_res directly for the fitted model.
    # However, the sum_sq_res from lstsq might be an empty array if N < P,
    # so direct calculation is more robust.
    residual_sum_squares = np.sum(residuals_values**2)

    if total_sum_squares == 0:
        if residual_sum_squares == 0 : # Perfect fit to a constant
             r_squared = 1.0
        else: # Cannot explain variation in a constant Y if residuals exist
             r_squared = 0.0 # Or undefined, depending on convention
    else:
        r_squared = 1 - (residual_sum_squares / total_sum_squares)

    return beta_hat, residuals, r_squared


def weighted_least_squares(
    Y: pd.Series,
    X: pd.DataFrame,
    weights_Omega_epsilon_inv_diag: pd.Series,
    add_intercept: bool = True
) -> Tuple[pd.Series, pd.Series, float]:
    """
    Performs Weighted Least Squares (WLS) regression.

    Solves Y = Xβ + ε for β_hat, minimizing (Y - Xβ)' Ω_ε⁻¹ (Y - Xβ).
    This is equivalent to OLS on transformed variables: Y*sqrt(w) = X*sqrt(w)β + ε*sqrt(w).

    Args:
        Y (pd.Series): Dependent variable (N x 1).
        X (pd.DataFrame): Predictor variables (N x P).
        weights_Omega_epsilon_inv_diag (pd.Series): Diagonal elements of the inverse
                                                    of the error covariance matrix (Ω_ε⁻¹).
                                                    These are the weights (N x 1).
        add_intercept (bool): If True, an intercept column is added to X.
                              Defaults to True.

    Returns:
        Tuple[pd.Series, pd.Series, float]: A tuple containing:
            - beta_hat (pd.Series): Estimated coefficients ((P+intercept) x 1).
            - residuals (pd.Series): Residuals on the original scale (Y - X @ beta_hat).
            - r_squared (float): R-squared value, calculated on the original Y.

    Raises:
        ValueError: If Y, X, and weights do not have matching indices or if X is empty.
    """
    if not Y.index.equals(X.index) or not Y.index.equals(weights_Omega_epsilon_inv_diag.index):
        raise ValueError("Y, X, and weights_Omega_epsilon_inv_diag must have the same index.")
    if X.empty:
        raise ValueError("X DataFrame cannot be empty.")
    if np.any(weights_Omega_epsilon_inv_diag < 0):
        raise ValueError("Weights must be non-negative.")

    X_final = X.copy()
    if add_intercept:
        X_final['intercept'] = 1.0
        # Ensure intercept is the first column
        cols = ['intercept'] + [col for col in X.columns if col != 'intercept']
        X_final = X_final[cols]

    sqrt_weights = np.sqrt(weights_Omega_epsilon_inv_diag.values)

    # Transform Y and X
    # Y_transformed is (N,)
    Y_transformed_values = Y.values * sqrt_weights
    # X_final.values is (N, P_final). sqrt_weights[:, np.newaxis] is (N,1) for broadcasting
    X_transformed_values = X_final.values * sqrt_weights[:, np.newaxis]

    # Perform OLS on transformed variables
    # beta_hat_wls, _, _ = ordinary_least_squares(
    #     pd.Series(Y_transformed_values, index=Y.index),
    #     pd.DataFrame(X_transformed_values, index=X_final.index, columns=X_final.columns),
    #     add_intercept=False # Intercept already handled in X_final
    # )
    # Using lstsq directly for WLS:
    beta_hat_values, _, _, _ = np.linalg.lstsq(X_transformed_values, Y_transformed_values, rcond=None)
    beta_hat_wls = pd.Series(beta_hat_values, index=X_final.columns, name="coefficients")


    # Calculate residuals on the original scale
    residuals_values = Y.values - (X_final.values @ beta_hat_wls.values)
    residuals = pd.Series(residuals_values, index=Y.index, name="residuals")

    # Calculate R-squared on the original Y scale
    y_mean_original = np.mean(Y.values)
    total_sum_squares_original = np.sum((Y.values - y_mean_original)**2)
    residual_sum_squares_original = np.sum(residuals_values**2)

    if total_sum_squares_original == 0:
        if residual_sum_squares_original == 0: # Perfect fit to a constant
            r_squared = 1.0
        else:
            r_squared = 0.0 # Or undefined
    else:
        r_squared = 1 - (residual_sum_squares_original / total_sum_squares_original)

    # Note: For WLS, R-squared can be defined in multiple ways.
    # This is one common definition, based on the original (unweighted) variance of Y.
    # Another definition uses weighted sums of squares:
    # R_weighted_squared = 1 - (np.sum(weights_Omega_epsilon_inv_diag.values * residuals_values**2) /
    #                           np.sum(weights_Omega_epsilon_inv_diag.values * (Y.values - np.average(Y.values, weights=weights_Omega_epsilon_inv_diag.values))**2))
    # The current implementation uses the unweighted R-squared on original Y.

    return beta_hat_wls, residuals, r_squared
