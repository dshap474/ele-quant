import numpy as np
import pandas as pd
from typing import Union

def calculate_empirical_rademacher_complexity(
    performance_matrix_X: pd.DataFrame,
    num_rademacher_samples: int = 1000
) -> float:
    """
    Calculates the empirical Rademacher complexity for a set of strategies/signals.
    The performance_matrix_X should have observations (time periods) as rows
    and strategies/signals as columns.

    Args:
        performance_matrix_X (pd.DataFrame): A T x N_strategies DataFrame where T is
                                             the number of observations and N_strategies
                                             is the number of strategies/signals.
                                             Values are the performance metric (e.g., ICs, SRs)
                                             for each strategy at each time point.
        num_rademacher_samples (int): The number of Rademacher vectors (epsilon samples)
                                      to average over for the expectation.

    Returns:
        float: The estimated empirical Rademacher complexity.

    Raises:
        ValueError: If performance_matrix_X is empty or num_rademacher_samples is not positive.
    """
    if performance_matrix_X.empty:
        raise ValueError("performance_matrix_X cannot be empty.")
    if num_rademacher_samples <= 0:
        raise ValueError("num_rademacher_samples must be positive.")

    T, N_strategies = performance_matrix_X.shape
    if T == 0:
        # Should be caught by .empty but as an explicit check for clarity
        raise ValueError("performance_matrix_X must have at least one observation (row).")

    X_values = performance_matrix_X.values  # Work with NumPy array for efficiency

    sum_sup_eps_X = 0.0

    for _ in range(num_rademacher_samples):
        epsilon_vector = np.random.choice([-1, 1], size=T)
        # Calculate epsilon.T @ X_s for all strategies s
        # This is equivalent to (epsilon_vector @ X_values)
        # The result is a vector of length N_strategies
        eps_T_X_s = epsilon_vector @ X_values  # Broadcasting if X_values is T x N_strategies

        # Find the supremum (maximum) over all strategies s
        sup_eps_X = np.max(eps_T_X_s)
        sum_sup_eps_X += sup_eps_X

    rademacher_complexity = sum_sup_eps_X / (num_rademacher_samples * T)

    return rademacher_complexity

def calculate_ras_lower_bound_ic(
    empirical_ic_series: Union[pd.Series, np.ndarray],
    rademacher_complexity_R_hat_T: float,
    num_observations_T: int,
    delta_confidence: float = 0.05
) -> Union[pd.Series, np.ndarray]:
    """
    Calculates the Rademacher Anti-Serum (RAS) lower bound for Information Coefficients (ICs).
    This uses the formula: θ_S ≥ θ̂_S - 2R̂_T(S) - sqrt((2 * log(2/δ)) / T)
    (Based on Eq. 8.3 context from "Elements of Quantitative Investing")

    Args:
        empirical_ic_series (Union[pd.Series, np.ndarray]): A series or array of empirical
                                                           average ICs (θ̂_S) for one or
                                                           more signals.
        rademacher_complexity_R_hat_T (float): The estimated empirical Rademacher
                                               complexity for the set of signals.
        num_observations_T (int): The number of observations (time periods, T) used
                                  to calculate the empirical ICs and R_hat_T.
        delta_confidence (float): The desired confidence level (1 - δ is the probability).
                                  E.g., 0.05 for 95% confidence.

    Returns:
        Union[pd.Series, np.ndarray]: The RAS lower bound(s) for the IC(s).
                                       Type matches input empirical_ic_series.

    Raises:
        ValueError: If inputs are invalid (e.g., T <= 0, delta not in (0,1)).
    """
    if num_observations_T <= 0:
        raise ValueError("num_observations_T must be positive.")
    if not (0 < delta_confidence < 1):
        raise ValueError("delta_confidence must be between 0 and 1 (exclusive).")
    if rademacher_complexity_R_hat_T < 0:
        # While theoretically possible for R_hat_T to be slightly negative due to sampling,
        # it's usually non-negative. A large negative value would be suspicious.
        # For the formula, it's used as a positive penalty.
        # Let's assume it should be non-negative as a penalty.
        pass # Allowing non-strictly positive R_hat_T for now

    estimation_error_term = np.sqrt((2 * np.log(2 / delta_confidence)) / num_observations_T)

    lower_bound = empirical_ic_series - 2 * rademacher_complexity_R_hat_T - estimation_error_term

    return lower_bound

def calculate_ras_lower_bound_sr(
    empirical_sr_series: Union[pd.Series, np.ndarray],
    rademacher_complexity_R_hat_T: float,
    num_observations_T: int,
    delta_confidence: float = 0.05
) -> Union[pd.Series, np.ndarray]:
    """
    Calculates the Rademacher Anti-Serum (RAS) lower bound for Sharpe Ratios (SRs).
    This uses the formula: θ_S ≥ θ̂_S - 2R̂_T(S) - sqrt((2 * log(2/δ) * (1 + θ̂_S²/2)) / T)
    (Based on Eq. 8.4 context from "Elements of Quantitative Investing")
    Note: empirical_sr_series (θ̂_S) should be the non-annualized Sharpe Ratios if T
    is the number of non-annualized periods.

    Args:
        empirical_sr_series (Union[pd.Series, np.ndarray]): A series or array of empirical
                                                           non-annualized Sharpe Ratios (θ̂_S)
                                                           for one or more strategies.
        rademacher_complexity_R_hat_T (float): The estimated empirical Rademacher
                                               complexity for the set of strategies.
        num_observations_T (int): The number of observations (time periods, T) used
                                  to calculate the empirical SRs and R_hat_T.
        delta_confidence (float): The desired confidence level (1 - δ is the probability).
                                  E.g., 0.05 for 95% confidence.

    Returns:
        Union[pd.Series, np.ndarray]: The RAS lower bound(s) for the SR(s).
                                       Type matches input empirical_sr_series.

    Raises:
        ValueError: If inputs are invalid (e.g., T <= 0, delta not in (0,1)).
    """
    if num_observations_T <= 0:
        raise ValueError("num_observations_T must be positive.")
    if not (0 < delta_confidence < 1):
        raise ValueError("delta_confidence must be between 0 and 1 (exclusive).")

    # Ensure empirical_sr_series is a NumPy array for vectorized operations
    theta_hat_S_sq = np.square(np.asarray(empirical_sr_series))

    estimation_error_term = np.sqrt(
        (2 * np.log(2 / delta_confidence) * (1 + theta_hat_S_sq / 2)) / num_observations_T
    )

    lower_bound = empirical_sr_series - 2 * rademacher_complexity_R_hat_T - estimation_error_term

    return lower_bound
