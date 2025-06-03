import pandas as pd
import numpy as np
from typing import List, Optional

# Attempt to import OLS from quant_elements_lib.utils.regression
# This is a placeholder import. The actual OLS function needs to be
# available in the specified path for calculate_beta_prediction_error to work.
try:
    from quant_elements_lib.utils.regression import ordinary_least_squares
except ImportError:
    # Define a placeholder if the actual function isn't available,
    # so the file can be parsed, but it will raise an error if used.
    def ordinary_least_squares(*args, **kwargs): # type: ignore
        raise NotImplementedError(
            "The function 'ordinary_least_squares' is not available. "
            "Please ensure it is implemented in quant_elements_lib.utils.regression."
        )

def calculate_fmp_turnover(fmp_weights_series: List[pd.DataFrame]) -> float:
    """
    Calculate the Factor Mimicking Portfolio (FMP) turnover.

    Turnover is measured as the average squared Frobenius norm of the change
    in FMP weights (P_t) over time. The formula is:
    (1/(T-1)) * Σ ||P_t - P_{t-1}||_F² for t=1 to T-1.

    Parameters
    ----------
    fmp_weights_series : List[pd.DataFrame]
        A list of T DataFrames. Each DataFrame P_t is N x K, where N is the
        number of assets and K is the number of factors. These represent the
        FMP weights at different time periods t = 0, 1, ..., T-1.

    Returns
    -------
    float
        The average squared Frobenius norm of the changes in FMP weights.
        Returns np.nan if there are fewer than 2 time periods of weights.

    Notes
    -----
    - Reference: Page 189 of "Elements of Quantitative Investing".
    - Requires `numpy.linalg.norm` for Frobenius norm calculation.
    - Assumes all DataFrames in `fmp_weights_series` have the same dimensions (N x K).
    """
    if not isinstance(fmp_weights_series, list) or \
       not all(isinstance(df, pd.DataFrame) for df in fmp_weights_series):
        raise TypeError("fmp_weights_series must be a list of pandas DataFrames.")

    T = len(fmp_weights_series)
    if T < 2:
        # Turnover cannot be calculated with fewer than two weight matrices
        return np.nan

    # Check for consistent shapes (optional, but good practice)
    # Assuming N and K are consistent across all DataFrames based on the first one
    if T > 0:
        first_shape = fmp_weights_series[0].shape
        if not all(df.shape == first_shape for df in fmp_weights_series):
            raise ValueError("All DataFrames in fmp_weights_series must have the same dimensions.")

    sum_squared_frobenius_norm_changes = 0.0
    for t in range(1, T):
        p_t = fmp_weights_series[t].values
        p_t_minus_1 = fmp_weights_series[t-1].values

        delta_p = p_t - p_t_minus_1
        squared_frobenius_norm = np.linalg.norm(delta_p, 'fro')**2
        sum_squared_frobenius_norm_changes += squared_frobenius_norm

    return sum_squared_frobenius_norm_changes / (T - 1)


def calculate_beta_prediction_error(
    predicted_betas_ts: pd.Series,
    realized_returns_portfolio_ts: pd.Series,
    realized_returns_benchmark_ts: pd.Series,
    estimation_window: int
) -> float:
    """
    Calculate the sum of squared errors between predicted and realized betas.

    Realized betas are calculated using a rolling Ordinary Least Squares (OLS)
    regression of the portfolio's realized returns on the benchmark's realized
    returns over a specified estimation window.

    Parameters
    ----------
    predicted_betas_ts : pd.Series
        A T_pred x 1 Series of ex-ante predicted betas.
    realized_returns_portfolio_ts : pd.Series
        A T_ret x 1 Series of the portfolio's realized returns.
    realized_returns_benchmark_ts : pd.Series
        A T_ret x 1 Series of the benchmark's realized returns.
    estimation_window : int
        The number of periods to use for the rolling OLS regression to
        calculate realized betas.

    Returns
    -------
    float
        The sum of squared errors: Σ (β_predicted,t - β_realized,t)².
        Returns np.nan if calculations cannot be performed (e.g., not enough data).

    Notes
    -----
    - Reference: Page 190 of "Elements of Quantitative Investing".
    - This function relies on an `ordinary_least_squares` function expected to be
      available at `quant_elements_lib.utils.regression.ordinary_least_squares`.
      This OLS function should be capable of handling inputs `X` (benchmark returns)
      and `y` (portfolio returns) and return at least the beta coefficient.
      It's assumed it can be called as:
      `ols_results = ordinary_least_squares(X_window, y_window)`
      and `beta = ols_results['coefficients'][X_window.columns[0]]` (or similar access).
    - The `predicted_betas_ts` and the calculated `realized_betas_ts` are aligned
      based on their time indices before calculating the error.
    """
    if not isinstance(predicted_betas_ts, pd.Series):
        raise TypeError("predicted_betas_ts must be a pandas Series.")
    if not isinstance(realized_returns_portfolio_ts, pd.Series):
        raise TypeError("realized_returns_portfolio_ts must be a pandas Series.")
    if not isinstance(realized_returns_benchmark_ts, pd.Series):
        raise TypeError("realized_returns_benchmark_ts must be a pandas Series.")
    if not isinstance(estimation_window, int) or estimation_window <= 0:
        raise ValueError("estimation_window must be a positive integer.")

    if realized_returns_portfolio_ts.shape[0] != realized_returns_benchmark_ts.shape[0]:
        raise ValueError("Portfolio and benchmark return series must have the same length.")

    if realized_returns_portfolio_ts.shape[0] < estimation_window:
        # Not enough data to perform even one rolling regression
        return np.nan

    # Calculate ex-post realized betas using rolling OLS
    realized_betas_list = []
    # The index for realized betas will correspond to the *end* of each window
    realized_betas_index = []

    # Ensure benchmark returns are structured as DataFrame for OLS (X)
    # and portfolio returns as Series (y)
    # Add a constant for the regression intercept
    X_benchmark_full = pd.DataFrame({
        'benchmark': realized_returns_benchmark_ts.values,
        'const': 1.0 # Assuming OLS utility handles or expects constant
    }, index=realized_returns_benchmark_ts.index)

    y_portfolio_full = realized_returns_portfolio_ts

    for i in range(estimation_window -1 , len(realized_returns_portfolio_ts)):
        start_idx = i - estimation_window + 1
        end_idx = i + 1 # pandas slice upper bound is exclusive

        y_window = y_portfolio_full.iloc[start_idx:end_idx]
        X_window = X_benchmark_full.iloc[start_idx:end_idx]

        current_window_index = y_portfolio_full.index[i]

        if y_window.shape[0] < estimation_window: # Should not happen with correct loop bounds
            continue

        try:
            # This part assumes how ordinary_least_squares works.
            # It needs to take X (DataFrame, independent vars) and y (Series, dependent var)
            # and return coefficients. The benchmark beta is the coefficient of 'benchmark'.
            # If X_window has multiple columns, ensure 'benchmark' is correctly identified.
            ols_results = ordinary_least_squares(X_window[['benchmark']], y_window, add_intercept=True)

            # The structure of ols_results and access to beta needs to match
            # the actual implementation of ordinary_least_squares.
            # Assuming it returns a dict with 'coefficients' Series,
            # and 'benchmark' is the name of the coefficient for the benchmark returns.
            # If OLS adds its own constant, X_benchmark_full might not need 'const'.
            if 'coefficients' in ols_results and 'benchmark' in ols_results['coefficients'].index:
                 beta_realized = ols_results['coefficients']['benchmark']
            else:
                # Fallback if the structure is different, or specific handling for single regressor
                # This part is highly dependent on the OLS utility's output.
                # If only one regressor (benchmark) + intercept, beta might be the first coeff.
                # This is a guess and needs to be confirmed.
                # For robustness, we expect 'benchmark' named coefficient.
                print(f"Warning: Could not extract 'benchmark' coefficient at window ending {current_window_index}. OLS results: {ols_results}")
                beta_realized = np.nan

            realized_betas_list.append(beta_realized)
            realized_betas_index.append(current_window_index)

        except Exception as e:
            # Catch errors during OLS, e.g., singular matrix if returns are collinear
            print(f"Warning: OLS regression failed for window ending {current_window_index}: {e}")
            realized_betas_list.append(np.nan)
            realized_betas_index.append(current_window_index)

    if not realized_betas_list:
        return np.nan

    realized_betas_ts = pd.Series(realized_betas_list, index=pd.Index(realized_betas_index, name="Date"))
    realized_betas_ts = realized_betas_ts.dropna()

    # Align predicted_betas_ts with realized_betas_ts
    # The predicted betas are "ex-ante", so predicted_betas_ts.loc[t] is the prediction *for* period t.
    # The realized_betas_ts.loc[t] is calculated using data *up to and including* t.
    # So, a direct alignment on index should be appropriate.
    aligned_predicted, aligned_realized = predicted_betas_ts.align(realized_betas_ts, join='inner')

    if aligned_predicted.empty or aligned_realized.empty:
        # No overlapping periods after alignment
        return np.nan

    # Calculate sum of squared errors
    squared_errors = (aligned_predicted - aligned_realized)**2
    sum_sq_error = np.sum(squared_errors)

    return float(sum_sq_error)
