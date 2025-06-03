import numpy as np
import pandas as pd
import statsmodels.api as sm # For sm.add_constant
from typing import Tuple

def ordinary_least_squares(
    Y: pd.Series, X: pd.DataFrame, add_intercept: bool = True
) -> Tuple[pd.Series, pd.Series]:
    """
    Performs Ordinary Least Squares (OLS) regression.

    The OLS estimator `beta_hat` is calculated as:
    beta_hat = (X_processed' @ X_processed)⁻¹ @ X_processed' @ Y

    Args:
        Y: Dependent variable (N x 1 pandas Series).
        X: Matrix of predictors (N x P pandas DataFrame).
        add_intercept: If True, adds a constant column to X.

    Returns:
        A tuple containing:
            - beta_hat: Estimated coefficients (pandas Series with index from X's columns or 'const' for intercept).
            - residuals: Residuals of the regression (Y - X_processed @ beta_hat) as a pandas Series.
    """
    X_processed = X.copy()
    if add_intercept:
        X_processed = sm.add_constant(X_processed, prepend=True, has_constant='skip')

    beta_hat_np, _, _, _ = np.linalg.lstsq(X_processed.values, Y.values, rcond=None)

    beta_hat = pd.Series(beta_hat_np, index=X_processed.columns)
    residuals = Y - X_processed @ beta_hat
    residuals = residuals.rename("residuals") # Give the series a name

    return beta_hat, residuals

def weighted_least_squares(
    Y: pd.Series,
    X: pd.DataFrame,
    weights_omega_epsilon_inv_diag: pd.Series,
    add_intercept: bool = True,
) -> Tuple[pd.Series, pd.Series]:
    """
    Performs Weighted Least Squares (WLS) regression.

    WLS is used when the error terms have non-constant variance (heteroscedasticity).
    The observations are weighted by the inverse of the error covariance matrix.
    This implementation assumes a diagonal error covariance matrix, so weights
    are the diagonal elements of Ω_ε⁻¹.

    The WLS estimator is found by transforming Y and X and performing OLS:
    Y_transformed = sqrt(weights) * Y
    X_transformed_i = sqrt(weights) * X_i
    beta_hat = OLS(Y_transformed, X_transformed)

    Args:
        Y: Dependent variable (N x 1 pandas Series).
        X: Matrix of predictors (N x P pandas DataFrame).
        weights_omega_epsilon_inv_diag: Diagonal of the inverse of the error
                                         covariance matrix (N x 1 pandas Series). These are the weights.
        add_intercept: If True, adds a constant column to X before transformation.

    Returns:
        A tuple containing:
            - beta_hat: Estimated coefficients (pandas Series with index from X's columns or 'const' for intercept).
            - residuals: Residuals of the regression on the original scale (Y - X_processed @ beta_hat) as a pandas Series.
    """
    if not isinstance(Y, pd.Series):
        raise TypeError("Y must be a pandas Series.")
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")
    if not isinstance(weights_omega_epsilon_inv_diag, pd.Series):
        raise TypeError("weights_omega_epsilon_inv_diag must be a pandas Series.")
    if not Y.index.equals(X.index) or not Y.index.equals(weights_omega_epsilon_inv_diag.index):
        raise ValueError("Y, X, and weights_omega_epsilon_inv_diag must have the same index.")

    sqrt_weights = np.sqrt(weights_omega_epsilon_inv_diag)

    Y_transformed = Y * sqrt_weights
    X_transformed = X.multiply(sqrt_weights, axis=0)

    X_processed_original = X.copy() # For calculating residuals on original scale

    if add_intercept:
        # Add intercept to original X for final residual calculation
        X_processed_original = sm.add_constant(X_processed_original, prepend=True, has_constant='skip')
        # Add transformed intercept for estimation
        X_transformed = pd.concat([pd.Series(sqrt_weights, index=X.index, name="const"), X_transformed], axis=1)

    # Perform OLS on transformed data
    # We can call ordinary_least_squares or directly use np.linalg.lstsq
    # Using lstsq directly to avoid re-adding intercept if it's already handled
    beta_hat_np, _, _, _ = np.linalg.lstsq(X_transformed.values, Y_transformed.values, rcond=None)
    beta_hat = pd.Series(beta_hat_np, index=X_transformed.columns)

    # Calculate residuals on the original scale
    # If intercept was added, X_processed_original already has it.
    # If not, it's just original X.
    residuals = Y - X_processed_original @ beta_hat[X_processed_original.columns] # Align beta_hat for multiplication
    residuals = residuals.rename("residuals") # Give the series a name

    return beta_hat, residuals
