import pandas as pd
import numpy as np
from typing import List, Union, Callable, Optional, Tuple
from quant_elements_lib.risk_evaluation.loss_functions import calculate_qlike_loss, calculate_mse_variance_ratio_loss

# Epsilon for numerical stability in matrix inversion or operations involving determinants
EPSILON = 1e-12

def evaluate_random_portfolio_variance(
    estimated_cov_matrix: pd.DataFrame,
    realized_returns_series: pd.DataFrame,
    loss_function: Callable[[pd.Series, pd.Series], float],
    num_random_portfolios: int = 1000
) -> float:
    """
    Evaluate the covariance matrix estimate using random portfolios.

    This function generates a number of random portfolios and calculates the
    average loss (e.g., QLIKE) between the predicted variance of these
    portfolios (derived from the estimated covariance matrix) and their
    realized variance (derived from subsequent realized returns).

    Parameters
    ----------
    estimated_cov_matrix : pd.DataFrame
        The estimated N x N covariance matrix.
    realized_returns_series : pd.DataFrame
        A T x N DataFrame of realized returns, where T is the number of
        time periods and N is the number of assets.
    loss_function : Callable[[pd.Series, pd.Series], float]
        A callable loss function (e.g., `calculate_qlike_loss` or
        `calculate_mse_variance_ratio_loss`) that takes two pd.Series
        (realized variances, predicted variances) and returns a float.
    num_random_portfolios : int, optional
        The number of random portfolios to generate, by default 1000.

    Returns
    -------
    float
        The average loss value across all random portfolios.

    Notes
    -----
    - Procedure 5.1, page 180 of "Elements of Quantitative Investing".
    - Random portfolio weights are generated from a standard normal distribution
      and then normalized to have a unit L2 norm (||W||₂=1).
    - The predicted variance for a portfolio `w` is `w.T @ estimated_cov_matrix @ w`.
    - The realized variance for a portfolio `w` at time `t` is
      `(w.T @ realized_returns_series.iloc[t])**2`.
    """
    if not isinstance(estimated_cov_matrix, pd.DataFrame):
        raise TypeError("estimated_cov_matrix must be a pandas DataFrame.")
    if not isinstance(realized_returns_series, pd.DataFrame):
        raise TypeError("realized_returns_series must be a pandas DataFrame.")
    if not callable(loss_function):
        raise TypeError("loss_function must be a callable.")
    if not isinstance(num_random_portfolios, int) or num_random_portfolios <= 0:
        raise ValueError("num_random_portfolios must be a positive integer.")

    num_assets = estimated_cov_matrix.shape[0]
    if num_assets != estimated_cov_matrix.shape[1]:
        raise ValueError("estimated_cov_matrix must be square.")
    if num_assets != realized_returns_series.shape[1]:
        raise ValueError(
            "Number of assets in estimated_cov_matrix and "
            "realized_returns_series must match."
        )

    T = realized_returns_series.shape[0]
    if T == 0:
        return np.nan # Or raise error, depending on desired behavior for no returns

    total_loss = 0.0
    for _ in range(num_random_portfolios):
        # Generate random weights and normalize (L2 norm)
        w = np.random.randn(num_assets, 1)
        w = w / np.linalg.norm(w)

        # Calculate predicted portfolio variance
        # Ensure estimated_cov_matrix values are used for calculation
        pred_var_scalar = w.T @ estimated_cov_matrix.values @ w
        if pred_var_scalar.ndim > 1 : # Should be a scalar
             pred_var_scalar = pred_var_scalar[0,0]


        # Create a Series of predicted variances (repeated T times)
        predicted_variances_series = pd.Series([pred_var_scalar] * T, index=realized_returns_series.index)

        # Calculate realized portfolio variance for each period t
        # (w.T @ r_t)**2
        realized_portfolio_returns = realized_returns_series.values @ w # T x 1
        realized_variances_portfolio_series = pd.Series(
            (realized_portfolio_returns**2).flatten(), index=realized_returns_series.index
        )

        # Apply the loss function
        loss = loss_function(realized_variances_portfolio_series, predicted_variances_series)
        total_loss += loss

    return total_loss / num_random_portfolios


def evaluate_mvo_portfolio_variance(
    estimated_cov_matrix_list: List[pd.DataFrame],
    realized_returns_series: pd.DataFrame,
    alpha_forecast_series: pd.DataFrame, # T x N or K x N
    risk_aversion: float
) -> pd.DataFrame:
    """
    Evaluate Mean-Variance Optimized (MVO) portfolios' realized variances.

    For each estimated covariance matrix and corresponding alpha forecast,
    this function constructs an MVO portfolio and then calculates its
    realized variance over subsequent periods.

    Parameters
    ----------
    estimated_cov_matrix_list : List[pd.DataFrame]
        A list of K estimated N x N covariance matrices. K can be the number
        of models or time periods for which covariance is estimated.
    realized_returns_series : pd.DataFrame
        A T x N DataFrame of realized returns, where T is the number of
        time periods and N is the number of assets.
    alpha_forecast_series : pd.DataFrame
        A K x N or T x N DataFrame of alpha forecasts.
        If K x N, it's assumed that the k-th alpha forecast corresponds to the
        k-th covariance matrix in `estimated_cov_matrix_list`.
        If T x N, it's assumed that `alpha_forecast_series.iloc[t]` is used
        with the covariance matrix relevant for predictions at time t.
        The length K of `estimated_cov_matrix_list` must match the first
        dimension of `alpha_forecast_series` if KxN.
    risk_aversion : float
        The risk aversion coefficient (lambda) used in MVO portfolio construction.

    Returns
    -------
    pd.DataFrame
        A K x T_eval DataFrame, where K is the number of MVO portfolios
        (corresponding to each item in `estimated_cov_matrix_list`), and T_eval
        is the number of periods for which realized variance is evaluated.
        Each row contains the time series of realized variances for an MVO
        portfolio. The exact definition of T_eval depends on indexing and
        whether forecasts are for the next period or multiple periods.
        Typically, if a cov matrix and alpha at t are used to form a portfolio,
        its variance is realized at t+1.

    Notes
    -----
    - Procedure 5.3, page 182 of "Elements of Quantitative Investing".
    - MVO portfolio weights `w_t = estimated_cov_matrix_inv @ alpha_t / risk_aversion`.
    - Weights `w_t` are normalized to sum to 1.
    - Realized variance at `t+1`: `(w_t.T @ realized_returns_series.iloc[t+1])**2`.
    - Matrix inversion is handled using `np.linalg.pinv` for pseudo-inverse
      to improve stability.
    """
    if not isinstance(estimated_cov_matrix_list, list) or \
       not all(isinstance(m, pd.DataFrame) for m in estimated_cov_matrix_list):
        raise TypeError("estimated_cov_matrix_list must be a list of pandas DataFrames.")
    if not isinstance(realized_returns_series, pd.DataFrame):
        raise TypeError("realized_returns_series must be a pandas DataFrame.")
    if not isinstance(alpha_forecast_series, pd.DataFrame):
        raise TypeError("alpha_forecast_series must be a pandas DataFrame.")
    if not isinstance(risk_aversion, (int, float)) or risk_aversion == 0:
        raise ValueError("risk_aversion must be a non-zero number.")

    num_assets = realized_returns_series.shape[1]
    num_realized_periods = realized_returns_series.shape[0]

    if not estimated_cov_matrix_list:
        return pd.DataFrame()

    # Validate dimensions
    for cov_matrix in estimated_cov_matrix_list:
        if cov_matrix.shape[0] != cov_matrix.shape[1]:
            raise ValueError("Each estimated_cov_matrix must be square.")
        if cov_matrix.shape[0] != num_assets:
            raise ValueError(
                "Asset count mismatch between covariance matrices and realized returns."
            )
    if alpha_forecast_series.shape[1] != num_assets:
        raise ValueError(
            "Asset count mismatch between alpha forecasts and realized returns."
        )
    if len(estimated_cov_matrix_list) != alpha_forecast_series.shape[0] and \
       alpha_forecast_series.shape[0] != num_realized_periods : # KxN or TxN
         raise ValueError(
            "Alpha forecast series rows must match either number of cov matrices (K) "
            "or number of realized return periods (T)."
        )


    all_mvo_realized_variances = []

    for i, estimated_cov_matrix in enumerate(estimated_cov_matrix_list):
        try:
            # Use pseudo-inverse for stability
            inv_cov_matrix = np.linalg.pinv(estimated_cov_matrix.values + EPSILON * np.eye(num_assets))
        except np.linalg.LinAlgError:
            # Handle cases where matrix is singular even with epsilon
            # For example, by skipping or logging
            print(f"Warning: Could not invert covariance matrix at index {i}. Skipping.")
            all_mvo_realized_variances.append(pd.Series([np.nan] * (num_realized_periods -1), name=f"mvo_portfolio_{i}")) # Or some other length
            continue

        # Determine which alpha forecast to use
        if len(estimated_cov_matrix_list) == alpha_forecast_series.shape[0]: # KxN alphas
            alpha_t = alpha_forecast_series.iloc[i].values.reshape(-1, 1) # N x 1
        elif num_realized_periods == alpha_forecast_series.shape[0]: # TxN alphas
            # This case implies cov matrix might be fixed or slowly changing,
            # and alphas are per-period. We assume we use alpha from time `t`
            # to form portfolio for `t+1`.
            # This part of logic depends on how cov matrices are indexed/meant to be used.
            # For now, let's assume if K < T, then the i-th cov matrix is used with i-th alpha
            # to predict for *all* subsequent t. This might need refinement based on exact use case.
            # A common setup: estimated_cov_matrix_list has 1 element (the model)
            # and alpha_forecast_series is TxN.
            # If len(estimated_cov_matrix_list) == 1, we use this one matrix for all alphas.
            if len(estimated_cov_matrix_list) == 1:
                 inv_cov_matrix = np.linalg.pinv(estimated_cov_matrix_list[0].values + EPSILON * np.eye(num_assets))
                 # This loop structure will be problematic if K=1 and alphas are TxN
                 # The current outer loop is on K. This function might need rethink for TxN alphas
                 # if K > 1. For now, assuming KxN or K=1 and alphas can be TxN implicitly.
                 # The problem asks for KxT output. So K portfolios, each has T realized variances.
                 # This means one w_k is formed and its variance is tracked over T periods.

            # This interpretation matches Procedure 5.3 where a single Omega_hat
            # is used with a sequence of alphas alpha_t to form w_t, then evaluate (w_t' r_{t+1})^2
            # However, the input `estimated_cov_matrix_list` suggests K different Omega_hats.
            # Let's assume the k-th Omega_hat forms portfolio k, and its alpha is alpha_k.
            # This portfolio w_k is then held, and its variance is calculated over T periods.
            alpha_t = alpha_forecast_series.iloc[i].values.reshape(-1, 1) # N x 1

        else: # Fallback or error for ambiguous alpha scenario
             raise ValueError("Mismatch in alpha_forecast_series dimensions relative to cov matrices and returns.")


        w_t_unnormalized = inv_cov_matrix @ alpha_t / risk_aversion # N x 1

        # Normalize weights to sum to 1
        w_t_sum = np.sum(w_t_unnormalized)
        if np.abs(w_t_sum) < EPSILON: # Avoid division by zero if sum of weights is tiny
            # Handle this case: e.g., equal weights, or skip, or error
            # For now, let's assume it doesn't happen or use zero weights (or skip)
            print(f"Warning: Sum of unnormalized weights for portfolio {i} is near zero. Using zero weights.")
            w_t = np.zeros_like(w_t_unnormalized)
        else:
            w_t = w_t_unnormalized / w_t_sum


        # Calculate realized variance of this portfolio w_t for each subsequent period
        # (w_t.T @ realized_returns_series.iloc[t_plus_1])**2
        # The problem asks for KxT output. This means for each of K MVO portfolios,
        # we calculate its realized variance using *all* T periods of realized_returns_series.
        portfolio_realized_returns = realized_returns_series.values @ w_t # T x 1
        portfolio_realized_variances = (portfolio_realized_returns**2).flatten() # Array of T values

        all_mvo_realized_variances.append(
            pd.Series(portfolio_realized_variances, index=realized_returns_series.index, name=f"mvo_portfolio_{i}")
        )

    if not all_mvo_realized_variances:
        return pd.DataFrame()

    result_df = pd.concat(all_mvo_realized_variances, axis=1)
    return result_df.T # Transpose to get KxT


def calculate_qdist_likelihood(
    estimated_cov_matrices_ts: Union[pd.DataFrame, List[pd.DataFrame]],
    realized_returns_series: pd.DataFrame
) -> float:
    """
    Calculate the QDIST likelihood value (Gaussian likelihood up to a constant).

    The QDIST statistic is defined as: Σ_t (r_t.T @ Ω_t⁻¹ @ r_t + log(det(Ω_t))),
    where Ω_t is the estimated covariance matrix at time t and r_t is the
    vector of realized returns at time t.

    Parameters
    ----------
    estimated_cov_matrices_ts : Union[pd.DataFrame, List[pd.DataFrame]]
        Either a single N x N covariance matrix (pd.DataFrame) to be used for all
        time periods, or a list of T (number of time periods) N x N
        covariance matrices (List[pd.DataFrame]).
    realized_returns_series : pd.DataFrame
        A T x N DataFrame of realized returns, where T is the number of
        time periods and N is the number of assets.

    Returns
    -------
    float
        The QDIST likelihood value.

    Notes
    -----
    - Equation 5.2, page 183 of "Elements of Quantitative Investing".
    - Uses `np.linalg.solve` for `r_t.T @ Ω_t⁻¹ @ r_t` (by solving Ω_t @ x = r_t, then r_t.T @ x).
    - Uses `np.linalg.slogdet` for `log(det(Ω_t))`.
    - Adds a small epsilon to the diagonal of covariance matrices before inversion
      or determinant calculation for numerical stability.
    """
    if not isinstance(realized_returns_series, pd.DataFrame):
        raise TypeError("realized_returns_series must be a pandas DataFrame.")
    if realized_returns_series.empty:
        return np.nan

    T, N = realized_returns_series.shape
    qdist_sum = 0.0

    is_single_matrix = isinstance(estimated_cov_matrices_ts, pd.DataFrame)

    if is_single_matrix:
        if estimated_cov_matrices_ts.shape != (N, N):
            raise ValueError("Single estimated_cov_matrix must be N x N.")
    elif isinstance(estimated_cov_matrices_ts, list):
        if not all(isinstance(m, pd.DataFrame) for m in estimated_cov_matrices_ts):
            raise TypeError("If list, estimated_cov_matrices_ts must contain pandas DataFrames.")
        if len(estimated_cov_matrices_ts) != T:
            raise ValueError("List of cov matrices must have length T (number of return periods).")
        if not all(m.shape == (N, N) for m in estimated_cov_matrices_ts):
            raise ValueError("Each cov matrix in the list must be N x N.")
    else:
        raise TypeError("estimated_cov_matrices_ts must be a DataFrame or a list of DataFrames.")


    for t in range(T):
        r_t = realized_returns_series.iloc[t].values # N x 1 vector

        if is_single_matrix:
            # Use .copy() to avoid modifying the original DataFrame if it's passed multiple times
            current_cov_matrix = estimated_cov_matrices_ts.copy().values
        else: # list of matrices
            current_cov_matrix = estimated_cov_matrices_ts[t].copy().values

        # Add epsilon for stability
        current_cov_matrix_stable = current_cov_matrix + EPSILON * np.eye(N)

        try:
            # Calculate r_t.T @ inv_Omega_t @ r_t
            # This is equivalent to r_t.T @ x where Omega_t @ x = r_t
            # Or, solve Omega_t @ x = r_t for x, then compute r_t.T @ x
            x = np.linalg.solve(current_cov_matrix_stable, r_t)
            quadratic_term = r_t.T @ x

            # Calculate log(det(Omega_t))
            sign, log_det = np.linalg.slogdet(current_cov_matrix_stable)

            if sign <= 0: # Determinant is not positive, problem with matrix
                # This might happen if matrix is not positive definite
                # Depending on strictness, could return np.nan, raise error, or log
                print(f"Warning: Covariance matrix at period {t} (or the single one) is not positive definite. log_det may be invalid.")
                # If sign is -1, log_det is log(abs(det)). If 0, log_det is -inf.
                # For QDIST, we need det > 0.
                return np.nan # Or handle as per specific requirements for non-PD matrices

            log_det_term = log_det
            qdist_sum += quadratic_term + log_det_term

        except np.linalg.LinAlgError:
            # Matrix is singular or other issue
            print(f"Warning: Linear algebra error (e.g., singular matrix) for period {t}. Returning NaN for QDIST.")
            return np.nan

    return qdist_sum


def test_minimum_variance_portfolios(
    estimated_cov_matrix: pd.DataFrame,
    realized_returns_one_period: pd.Series,
    constraint_vector: Optional[np.ndarray] = None
) -> Tuple[float, pd.Series]:
    """
    Test Minimum Variance Portfolios (MVP).

    Calculates the weights of the Minimum Variance Portfolio given an estimated
    covariance matrix and an optional constraint vector (typically a vector of ones
    for a fully invested portfolio). Then, it computes the realized variance of
    this MVP using one period of realized returns.

    Parameters
    ----------
    estimated_cov_matrix : pd.DataFrame
        The N x N estimated covariance matrix (Ω).
    realized_returns_one_period : pd.Series
        An N x 1 Series of realized returns for a single period.
    constraint_vector : Optional[np.ndarray], optional
        An N x 1 constraint vector (e.g., `np.ones(N)` for sum of weights = 1).
        If None, defaults to `np.ones(N)`.

    Returns
    -------
    Tuple[float, pd.Series]
        A tuple containing:
        - realized_variance_mvp (float): The realized variance of the MVP,
          calculated as `(w_mvp.T @ realized_returns_one_period)**2`.
        - mvp_weights (pd.Series): An N x 1 Series of MVP weights.

    Notes
    -----
    - Based on Theorem 5.1 context, page 196 of "Elements of Quantitative Investing".
    - MVP weights `w_mvp = (Ω⁻¹ @ c) / (c.T @ Ω⁻¹ @ c)`, where `c` is the
      constraint vector.
    - Uses pseudo-inverse `np.linalg.pinv` for Ω⁻¹ for numerical stability.
    """
    if not isinstance(estimated_cov_matrix, pd.DataFrame):
        raise TypeError("estimated_cov_matrix must be a pandas DataFrame.")
    if not estimated_cov_matrix.shape[0] == estimated_cov_matrix.shape[1]:
        raise ValueError("estimated_cov_matrix must be square.")
    N = estimated_cov_matrix.shape[0]

    if not isinstance(realized_returns_one_period, pd.Series):
        raise TypeError("realized_returns_one_period must be a pandas Series.")
    if realized_returns_one_period.shape[0] != N:
        raise ValueError("Mismatch in dimensions between cov matrix and returns.")

    if constraint_vector is None:
        c = np.ones(N)
    else:
        if not isinstance(constraint_vector, np.ndarray):
            raise TypeError("constraint_vector must be a numpy ndarray.")
        if constraint_vector.shape != (N,) and constraint_vector.shape != (N,1):
            raise ValueError(f"constraint_vector must have shape ({N},) or ({N},1).")
        c = constraint_vector.flatten() # Ensure it's 1D for consistent dot products

    # Add epsilon for stability before inversion
    cov_matrix_stable = estimated_cov_matrix.values + EPSILON * np.eye(N)
    try:
        inv_omega = np.linalg.pinv(cov_matrix_stable)
    except np.linalg.LinAlgError:
        print("Error: Could not pseudo-invert the covariance matrix.")
        nan_weights = pd.Series([np.nan] * N, index=estimated_cov_matrix.columns)
        return np.nan, nan_weights

    # Numerator: inv_Omega @ c
    numerator = inv_omega @ c

    # Denominator: c.T @ inv_Omega @ c
    denominator = c.T @ inv_omega @ c

    if np.abs(denominator) < EPSILON:
        print("Warning: Denominator in MVP weight calculation is close to zero. MVP weights may be unstable.")
        # Handle this case: e.g., return NaN weights or specific error
        mvp_weights_values = np.full(N, np.nan)
    else:
        mvp_weights_values = numerator / denominator

    mvp_weights = pd.Series(mvp_weights_values, index=estimated_cov_matrix.columns)

    # Calculate realized variance of MVP
    # (w_mvp.T @ realized_returns_one_period)**2
    # Ensure realized_returns_one_period is treated as a column vector if mvp_weights is 1D
    realized_portfolio_return = mvp_weights.values.T @ realized_returns_one_period.values
    realized_variance_mvp = realized_portfolio_return**2

    return float(realized_variance_mvp), mvp_weights


def calculate_malv_statistic(
    estimated_precision_matrices_ts: Union[pd.DataFrame, List[pd.DataFrame]],
    realized_returns_series: pd.DataFrame
) -> float:
    """
    Calculate the Mean Asset Likelihood Variance (MALV) statistic.

    The MALV statistic is defined as: (1/T) Σ_t (r_t.T @ Ω_t⁻¹ @ r_t),
    where Ω_t⁻¹ is the estimated precision matrix (inverse covariance) at time t,
    and r_t is the vector of realized returns at time t.

    Parameters
    ----------
    estimated_precision_matrices_ts : Union[pd.DataFrame, List[pd.DataFrame]]
        Either a single N x N precision matrix (Ω⁻¹) (pd.DataFrame) to be used
        for all time periods, or a list of T (number of time periods) N x N
        precision matrices (List[pd.DataFrame]).
    realized_returns_series : pd.DataFrame
        A T x N DataFrame of realized returns, where T is the number of
        time periods and N is the number of assets.

    Returns
    -------
    float
        The MALV statistic value.

    Notes
    -----
    - Reference: Page 186 of "Elements of Quantitative Investing".
    - This function assumes Ω_t⁻¹ (precision matrix) is provided directly.
      If covariance matrices are provided, they need to be inverted first (with care).
    """
    if not isinstance(realized_returns_series, pd.DataFrame):
        raise TypeError("realized_returns_series must be a pandas DataFrame.")
    if realized_returns_series.empty:
        return np.nan

    T, N = realized_returns_series.shape
    malv_sum = 0.0

    is_single_matrix = isinstance(estimated_precision_matrices_ts, pd.DataFrame)

    if is_single_matrix:
        if estimated_precision_matrices_ts.shape != (N, N):
            raise ValueError("Single estimated_precision_matrix must be N x N.")
    elif isinstance(estimated_precision_matrices_ts, list):
        if not all(isinstance(m, pd.DataFrame) for m in estimated_precision_matrices_ts):
            raise TypeError("If list, estimated_precision_matrices_ts must contain pandas DataFrames.")
        if len(estimated_precision_matrices_ts) != T:
            raise ValueError("List of precision matrices must have length T (number of return periods).")
        if not all(m.shape == (N, N) for m in estimated_precision_matrices_ts):
            raise ValueError("Each precision matrix in the list must be N x N.")
    else:
        raise TypeError(
            "estimated_precision_matrices_ts must be a DataFrame or a list of DataFrames."
        )

    for t in range(T):
        r_t = realized_returns_series.iloc[t].values # N vector

        if is_single_matrix:
            current_precision_matrix = estimated_precision_matrices_ts.values
        else: # list of matrices
            current_precision_matrix = estimated_precision_matrices_ts[t].values

        # Calculate r_t.T @ precision_matrix_t @ r_t
        # r_t is 1xN (after .T), current_precision_matrix is NxN, r_t is Nx1
        # (r_t.reshape(1, -1) @ current_precision_matrix @ r_t.reshape(-1, 1))[0,0]
        term = r_t.T @ current_precision_matrix @ r_t
        malv_sum += term

    if T == 0: # Avoid division by zero if no returns
        return np.nan

    return malv_sum / T
