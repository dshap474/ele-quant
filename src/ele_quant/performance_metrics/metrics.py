import pandas as pd
import numpy as np
from typing import Union

def calculate_expected_return(returns: pd.Series) -> float:
    """Calculate the expected return of a series.

    Parameters
    ----------
    returns : pd.Series
        A pandas Series of asset returns.

    Returns
    -------
    float
        The mean of the returns series.
    """
    return returns.mean()

def calculate_volatility(returns: pd.Series, trading_periods_per_year: int = 252, annualize: bool = True) -> float:
    """Calculate the volatility of a series of returns.

    Parameters
    ----------
    returns : pd.Series
        A pandas Series of asset returns.
    trading_periods_per_year : int, optional
        The number of trading periods in a year, by default 252.
    annualize : bool, optional
        Whether to annualize the volatility, by default True.

    Returns
    -------
    float
        The calculated volatility.
    """
    std_dev = returns.std()
    if annualize:
        return std_dev * np.sqrt(trading_periods_per_year)
    return std_dev

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: Union[float, pd.Series], trading_periods_per_year: int = 252, annualize: bool = True) -> float:
    """Calculate the Sharpe Ratio.

    Parameters
    ----------
    returns : pd.Series
        A pandas Series of asset returns.
    risk_free_rate : Union[float, pd.Series]
        The risk-free rate of return. Can be a scalar or a series.
    trading_periods_per_year : int, optional
        The number of trading periods in a year, by default 252.
    annualize : bool, optional
        Whether to annualize the Sharpe Ratio, by default True.

    Returns
    -------
    float
        The calculated Sharpe Ratio. Returns np.nan if the standard
        deviation of excess returns is zero.
    """
    excess_returns = returns - risk_free_rate
    mean_excess_return = excess_returns.mean()
    std_dev_excess_return = excess_returns.std()

    if std_dev_excess_return == 0:
        return np.nan

    raw_sharpe_ratio = mean_excess_return / std_dev_excess_return

    if annualize:
        return raw_sharpe_ratio * np.sqrt(trading_periods_per_year)
    return raw_sharpe_ratio

def calculate_information_ratio(active_returns: pd.Series, trading_periods_per_year: int = 252, annualize: bool = True) -> float:
    """Calculate the Information Ratio.

    Parameters
    ----------
    active_returns : pd.Series
        A pandas Series of active returns (portfolio returns - benchmark returns).
    trading_periods_per_year : int, optional
        The number of trading periods in a year, by default 252.
    annualize : bool, optional
        Whether to annualize the Information Ratio, by default True.

    Returns
    -------
    float
        The calculated Information Ratio. Returns np.nan if the standard
        deviation of active returns is zero.
    """
    mean_active_return = active_returns.mean()
    std_dev_active_return = active_returns.std()

    if std_dev_active_return == 0:
        return np.nan

    raw_ir = mean_active_return / std_dev_active_return

    if annualize:
        return raw_ir * np.sqrt(trading_periods_per_year)
    return raw_ir

def calculate_sharpe_ratio_se(estimated_sharpe_ratio: float, num_observations: int, trading_periods_per_year: int = 252, annualize_se: bool = True) -> float:
    """Calculate the Standard Error (SE) of the Sharpe Ratio.

    Parameters
    ----------
    estimated_sharpe_ratio : float
        The non-annualized Sharpe Ratio estimated from the sample.
    num_observations : int
        The number of observations (e.g., days, weeks, months) used to
        calculate the `estimated_sharpe_ratio`.
    trading_periods_per_year : int, optional
        The number of trading periods in a year, used for annualizing
        the SE. By default 252.
    annualize_se : bool, optional
        Whether to annualize the Standard Error, by default True.

    Returns
    -------
    float
        The calculated Standard Error of the Sharpe Ratio. Returns np.nan
        if `num_observations` is not positive.
    """
    if num_observations <= 0:
        return np.nan

    se_sr_non_annualized = np.sqrt((1 + (estimated_sharpe_ratio**2) / 2) / num_observations)

    if annualize_se:
        return se_sr_non_annualized * np.sqrt(trading_periods_per_year)
    return se_sr_non_annualized

def adjust_sharpe_ratio_for_autocorrelation(sharpe_ratio: float, lag_one_autocorrelation: float) -> float:
    """Adjust the Sharpe Ratio for lag-1 autocorrelation in returns.

    Parameters
    ----------
    sharpe_ratio : float
        The original Sharpe Ratio (can be annualized or non-annualized,
        the adjustment factor is independent of the annualization period).
    lag_one_autocorrelation : float
        The lag-1 autocorrelation (rho) of the returns series.

    Returns
    -------
    float
        The adjusted Sharpe Ratio. Returns np.nan if the adjustment factor
        cannot be calculated (e.g., `1 + lag_one_autocorrelation` is zero or
        the term under square root is negative).
    """
    rho = lag_one_autocorrelation
    if (1 + rho) == 0:
        return np.nan

    adjustment_factor_squared = (1 - rho) / (1 + rho)
    if adjustment_factor_squared < 0:
        return np.nan

    adjustment_factor = np.sqrt(adjustment_factor_squared)
    return sharpe_ratio * adjustment_factor
