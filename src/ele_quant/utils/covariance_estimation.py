import pandas as pd
import numpy as np
from typing import Tuple, Optional, Union
from sklearn.covariance import ledoit_wolf as sklearn_ledoit_wolf

def calculate_ledoit_wolf_shrinkage(
    returns_data: pd.DataFrame,
    assume_centered: bool = False,
    block_size: int = 1000,
) -> Tuple[pd.DataFrame, float]:
    """
    Estimates the Ledoit-Wolf shrunk covariance matrix.

    This function uses the scikit-learn implementation of Ledoit-Wolf shrinkage.
    The shrinkage target is the identity matrix scaled by the average variance of the assets.

    Args:
        returns_data: A T x N pandas DataFrame of time series data (T observations, N assets/variables).
                      The function expects asset returns, not a pre-computed sample covariance.
        assume_centered: If True, data will not be centered before computation.
                         Useful if data is already demeaned. Defaults to False.
        block_size: Size of blocks into which the covariance matrix will be split.
                    This is purely a memory optimization and does not affect results.

    Returns:
        A tuple containing:
            - shrunk_cov_matrix: The N x N shrunk covariance matrix (pandas DataFrame).
            - shrinkage_coefficient: The shrinkage coefficient (rho) applied (float).
    """
    if not isinstance(returns_data, pd.DataFrame):
        raise TypeError("returns_data must be a pandas DataFrame.")
    if returns_data.empty:
        raise ValueError("returns_data cannot be empty.")
    if not returns_data.applymap(np.isreal).all().all(): # Checks if all elements are real numbers
        raise ValueError("All elements in returns_data must be real numbers.")


    # Scikit-learn's ledoit_wolf function takes X (T x N array-like)
    # It returns: shrunk_cov, shrinkage_coef
    # X : array-like of shape (n_samples, n_features)
    # The function expects that n_samples > 1
    if returns_data.shape[0] <= 1:
        raise ValueError("Number of samples (rows) in returns_data must be greater than 1.")


    shrunk_cov_np, shrinkage_coefficient = sklearn_ledoit_wolf(
        X=returns_data.values,
        assume_centered=assume_centered,
        block_size=block_size
    )

    shrunk_cov_matrix = pd.DataFrame(
        shrunk_cov_np,
        index=returns_data.columns,
        columns=returns_data.columns
    )

    return shrunk_cov_matrix, shrinkage_coefficient

def calculate_newey_west_covariance(
    returns_ts: pd.DataFrame,
    max_lags: Optional[int] = None,
    kernel: str = 'bartlett',
    demean: bool = True,
) -> pd.DataFrame:
    """
    Calculates the Newey-West heteroscedasticity and autocorrelation consistent (HAC)
    covariance matrix for a time series of returns.

    Args:
        returns_ts: A T x N pandas DataFrame of time series data (T observations, N variables/assets).
        max_lags: The maximum number of lags to consider for autocovariance.
                  If None, a common default is floor(4 * (T/100)^(2/9)) or T^(1/3).
                  This implementation uses floor(T^(1/3)).
        kernel: The weighting kernel to use. Currently, only 'bartlett' (triangular) is supported.
                Bartlett weight: w_j = 1 - j / (max_lags + 1)
        demean: If True, demean the returns before calculating covariance. Defaults to True.

    Returns:
        The N x N Newey-West covariance matrix (pandas DataFrame).

    Raises:
        ValueError: If an unsupported kernel is specified or returns_ts is invalid.
    """
    if not isinstance(returns_ts, pd.DataFrame):
        raise TypeError("returns_ts must be a pandas DataFrame.")
    if returns_ts.empty:
        raise ValueError("returns_ts cannot be empty.")
    if not returns_ts.applymap(np.isreal).all().all():
         raise ValueError("All elements in returns_ts must be real numbers.")


    T, N = returns_ts.shape

    if T <= 1:
        raise ValueError("Number of observations T must be greater than 1.")

    if max_lags is None:
        # Common rule of thumb, e.g., from Greene (2003) or Stock & Watson (2007)
        # max_lags = int(np.floor(4 * (T / 100)**(2/9)))
        # Another common one is T^(1/3) or T^(1/4)
        max_lags = int(np.floor(T**(1/3)))

    if max_lags < 0:
        raise ValueError("max_lags must be non-negative.")

    if max_lags >= T:
        # print(f"Warning: max_lags ({max_lags}) >= T ({T}). Setting max_lags to T-1.")
        max_lags = T - 1


    if kernel.lower() != 'bartlett':
        raise ValueError("Currently, only 'bartlett' kernel is supported.")

    # Ensure returns are numpy arrays for calculations
    if demean:
        data = returns_ts.values - returns_ts.mean(axis=0).values
    else:
        data = returns_ts.values

    # Calculate Gamma_0 (contemporaneous covariance)
    # Note: Some definitions of Gamma_0 use (1/T), others (1/(T-N)) or similar for unbiasedness.
    # For HAC estimators, (1/T) is common for the S_0 term.
    Gamma_0 = (data.T @ data) / T # N x N matrix

    Omega_NW = Gamma_0.copy()

    for j in range(1, max_lags + 1):
        # Calculate Gamma_j (j-th order autocovariance)
        # Gamma_j = (1/T) * sum_{t=j+1 to T} (x_t * x_{t-j}')
        # data_lagged = data[j:] (shape T-j x N)
        # data_current = data[:-j] (shape T-j x N) - NO, this is wrong.
        # It should be data[t] @ data[t-j].T
        # data[j:] means from index j to end. data[:-j] means from start up to (not including) -j index.

        # Product sum: X_t' * X_{t-j}
        # X_t is data[j:, :]
        # X_{t-j} is data[:T-j, :]

        # Sum over t from j to T-1 (0-indexed) of (data[t,:]' @ data[t-j,:])
        # data[t,:] is a 1xN vector (after reshape). data[t-j,:] is 1xN.
        # data[t,:].reshape(N,1) @ data[t-j,:].reshape(1,N)

        # Gamma_j = sum_{t=j}^{T-1} (data[t].reshape(N,1) @ data[t-j].reshape(1,N)) / T
        Gamma_j = np.zeros((N, N))
        for t_idx in range(j, T):
            # data[t_idx] is row for time t
            # data[t_idx-j] is row for time t-j
            Gamma_j += np.outer(data[t_idx, :], data[t_idx - j, :])
        Gamma_j /= T


        # Bartlett kernel weight
        weight = 1.0 - (j / (max_lags + 1.0))

        Omega_NW += weight * (Gamma_j + Gamma_j.T)

    return pd.DataFrame(Omega_NW, index=returns_ts.columns, columns=returns_ts.columns)
