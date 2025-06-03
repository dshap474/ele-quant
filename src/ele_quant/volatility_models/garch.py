import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Optional, List, Tuple, Dict

def garch_1_1_conditional_variance(
    log_returns: pd.Series,
    alpha_0: float,
    alpha_1: float,
    beta_1: float,
    initial_h_sq: Optional[float] = None
) -> pd.Series:
    """
    Calculates the GARCH(1,1) conditional variance series (h_t^2).

    The formula is h_t^2 = alpha_0 + alpha_1 * r_{t-1}^2 + beta_1 * h_{t-1}^2.
    (Eq. 2.8, p. 73, "Elements of Quantitative Investing")

    Parameters
    ----------
    log_returns : pd.Series
        Series of log returns (r_t).
    alpha_0 : float
        GARCH parameter omega (or alpha_0).
    alpha_1 : float
        GARCH parameter alpha_1 (ARCH term).
    beta_1 : float
        GARCH parameter beta_1 (GARCH term).
    initial_h_sq : Optional[float], default None
        The initial conditional variance h_0^2 (or h_1^2 depending on indexing).
        If None, it's calculated as the unconditional variance:
        alpha_0 / (1 - alpha_1 - beta_1). If 1 - alpha_1 - beta_1 is not positive,
        the sample variance of log_returns**2 is used.

    Returns
    -------
    pd.Series
        A pandas Series of conditional variances h_t^2, with the same index as log_returns.
    """
    n = len(log_returns)
    if n == 0:
        return pd.Series([], dtype=float)

    h_sq = np.zeros(n)
    squared_returns = log_returns**2

    if initial_h_sq is None:
        denominator = 1 - alpha_1 - beta_1
        if denominator > 1e-8: # Check for positive denominator
            h0_sq = alpha_0 / denominator
        else:
            h0_sq = np.var(squared_returns.dropna()) # Fallback to sample variance of r_t^2
            if np.isnan(h0_sq) or h0_sq <= 1e-8: # Further fallback if sample variance is non-positive
                 h0_sq = 1e-4 # A small positive number
    else:
        h0_sq = initial_h_sq

    if h0_sq <= 0: # Ensure initial variance is positive
        h0_sq = 1e-4

    # h_sq[0] is h_1^2, which uses r_0^2 and h_0^2 (initial_h_sq)
    # However, GARCH is typically h_t^2 based on r_{t-1}^2 and h_{t-1}^2.
    # So, h_sq[0] (for time t=0) should be the initial variance.
    # Then h_sq[1] (for time t=1) uses r_0^2 and h_sq[0].

    h_sq[0] = h0_sq
    for t in range(1, n):
        # h_t^2 = alpha_0 + alpha_1 * r_{t-1}^2 + beta_1 * h_{t-1}^2
        h_sq[t] = alpha_0 + alpha_1 * squared_returns.iloc[t-1] + beta_1 * h_sq[t-1]
        if h_sq[t] <= 1e-8: # Ensure positivity, prevent issues in log-likelihood
            h_sq[t] = 1e-8 # Floor at a small positive number

    return pd.Series(h_sq, index=log_returns.index)


def garch_1_1_log_likelihood(
    params: Tuple[float, float, float],
    log_returns: pd.Series,
    initial_h_sq_method: str = 'unconditional'
) -> float:
    """
    Calculates the negative log-likelihood for a GARCH(1,1) model.

    The log-likelihood for a single observation t, assuming r_t ~ N(0, h_t^2), is:
    l_t = -0.5 * (log(2*pi) + log(h_t^2) + r_t^2 / h_t^2).
    The constant term log(2*pi) can be ignored for maximization.
    We minimize the negative sum: sum(log(h_t^2) + r_t^2 / h_t^2).
    (Based on p. 77, "Elements of Quantitative Investing", though specific formula varies)

    Parameters
    ----------
    params : Tuple[float, float, float]
        GARCH parameters (alpha_0, alpha_1, beta_1).
    log_returns : pd.Series
        Series of log returns.
    initial_h_sq_method : str, default 'unconditional'
        Method to determine initial h_sq for `garch_1_1_conditional_variance`.
        'unconditional': uses alpha_0 / (1 - alpha_1 - beta_1).
        'sample_var': uses sample variance of log_returns**2.

    Returns
    -------
    float
        The negative log-likelihood value. Returns a large positive number
        if parameters are invalid or variances become non-positive.
    """
    alpha_0, alpha_1, beta_1 = params
    epsilon = 1e-8 # Small constant for stability checks

    # Parameter constraints
    if alpha_0 <= epsilon: return 1e7 # alpha_0 must be > 0
    if alpha_1 < 0: return 1e7      # alpha_1 must be >= 0
    if beta_1 < 0: return 1e7       # beta_1 must be >= 0
    if alpha_1 + beta_1 >= 1.0 - epsilon: return 1e7 # Stationarity condition alpha_1 + beta_1 < 1

    initial_h_sq_val: Optional[float] = None
    if initial_h_sq_method == 'unconditional':
        denom = 1 - alpha_1 - beta_1
        if denom > epsilon:
            initial_h_sq_val = alpha_0 / denom
        else: # Fallback if non-stationary or at boundary
            initial_h_sq_val = np.var(log_returns**2)
            if np.isnan(initial_h_sq_val) or initial_h_sq_val <= epsilon:
                initial_h_sq_val = 1e-4
    elif initial_h_sq_method == 'sample_var':
        initial_h_sq_val = np.var(log_returns**2)
        if np.isnan(initial_h_sq_val) or initial_h_sq_val <= epsilon:
             initial_h_sq_val = 1e-4
    else: # Should not happen with defined choices
        initial_h_sq_val = 1e-4

    if initial_h_sq_val <= epsilon: # Ensure initial variance is positive
        initial_h_sq_val = epsilon

    h_sq_series = garch_1_1_conditional_variance(
        log_returns, alpha_0, alpha_1, beta_1, initial_h_sq=initial_h_sq_val
    )

    # Check for non-positive variances in the series (already floored in conditional_variance)
    if (h_sq_series <= epsilon).any():
        return 1e7 # Penalize parameters leading to non-positive variance

    # Calculate log-likelihood components
    log_h_sq = np.log(h_sq_series)
    r_sq_div_h_sq = (log_returns**2) / h_sq_series

    # Summing up for the total log-likelihood (ignoring constant 0.5*N*log(2*pi))
    # We want to minimize the negative log likelihood.
    neg_log_likelihood = 0.5 * np.sum(log_h_sq + r_sq_div_h_sq)

    if np.isnan(neg_log_likelihood) or np.isinf(neg_log_likelihood):
        return 1e7 # Penalize if likelihood calculation fails

    return neg_log_likelihood


def estimate_garch_1_1_parameters(
    log_returns: pd.Series,
    initial_guess: Optional[List[float]] = None,
    initial_h_sq_method: str = 'unconditional'
) -> Dict[str, float]:
    """
    Estimates GARCH(1,1) parameters (alpha_0, alpha_1, beta_1) using MLE.

    Minimizes the negative log-likelihood obtained from `garch_1_1_log_likelihood`.

    Parameters
    ----------
    log_returns : pd.Series
        Series of log returns.
    initial_guess : Optional[List[float]], default None
        Initial guess for [alpha_0, alpha_1, beta_1]. If None, a default guess like
        `[np.var(log_returns**2) * 0.05, 0.1, 0.85]` is used.
    initial_h_sq_method : str, default 'unconditional'
        Method for `initial_h_sq` in likelihood calculation.
        'unconditional' or 'sample_var'.

    Returns
    -------
    Dict[str, float]
        A dictionary containing the estimated parameters and the final log-likelihood:
        {'alpha_0': float, 'alpha_1': float, 'beta_1': float, 'log_likelihood': float}.
        Returns NaNs for parameters if optimization fails.
    """
    if log_returns.empty:
        return {'alpha_0': np.nan, 'alpha_1': np.nan, 'beta_1': np.nan, 'log_likelihood': np.nan}

    if initial_guess is None:
        # Heuristic initial guess: variance of r^2 * small factor, common values for alpha1, beta1
        var_r_sq = np.var(log_returns**2)
        if np.isnan(var_r_sq) or var_r_sq < 1e-8: var_r_sq = 1e-4 # Handle case of zero/low variance returns
        initial_guess = [var_r_sq * 0.05, 0.1, 0.85]
        # Ensure sum alpha1+beta1 < 1 for initial guess if possible
        if initial_guess[1] + initial_guess[2] >= 0.99:
            initial_guess[1] = 0.1
            initial_guess[2] = 0.85


    epsilon = 1e-8 # For bounds and constraints to avoid exact zero or one

    # Bounds for parameters: alpha_0 > 0, alpha_1 >= 0, beta_1 >= 0
    bounds = [
        (epsilon, None),      # alpha_0 > epsilon
        (0, 1.0 - epsilon),   # 0 <= alpha_1 < 1
        (0, 1.0 - epsilon),   # 0 <= beta_1 < 1
    ]

    # Constraint: alpha_1 + beta_1 < 1 (for stationarity)
    #minimize wants g(x) >= 0, so 1 - alpha_1 - beta_1 >= epsilon
    constraints = ({'type': 'ineq', 'fun': lambda x: 1.0 - x[1] - x[2] - epsilon})

    # Objective function for minimize
    objective_func = lambda params: garch_1_1_log_likelihood(
        params, log_returns, initial_h_sq_method=initial_h_sq_method
    )

    result = minimize(
        objective_func,
        initial_guess,
        method='SLSQP', # Sequential Least Squares Programming, good for constrained optimization
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-7, 'disp': False} # Adjust ftol for precision, disp for verbosity
    )

    if result.success:
        est_alpha_0, est_alpha_1, est_beta_1 = result.x
        final_log_likelihood = result.fun
        return {
            'alpha_0': est_alpha_0,
            'alpha_1': est_alpha_1,
            'beta_1': est_beta_1,
            'log_likelihood': final_log_likelihood
        }
    else:
        # print(f"GARCH estimation failed: {result.message}") # Optional: for debugging
        return {'alpha_0': np.nan, 'alpha_1': np.nan, 'beta_1': np.nan, 'log_likelihood': np.nan}
