import pandas as pd
from typing import Union
import numpy as np

def calculate_simple_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate simple returns from a series of prices.

    Parameters
    ----------
    prices : pd.Series
        A pandas Series of prices.

    Returns
    -------
    pd.Series
        A pandas Series of simple returns.
        The first element will be NaN as it has no prior price.
        (p. 60)
    """
    return prices.pct_change()

def calculate_dividend_adjusted_returns(prices: pd.Series, dividends: pd.Series) -> pd.Series:
    """
    Calculate dividend-adjusted returns.

    This formula considers the price appreciation and the dividend received.
    Return = (P_t - P_{t-1} + D_t) / P_{t-1}

    Parameters
    ----------
    prices : pd.Series
        A pandas Series of prices.
    dividends : pd.Series
        A pandas Series of dividends, indexed consistently with prices.
        Assumes dividends are paid at time t, associated with holding period t-1 to t.

    Returns
    -------
    pd.Series
        A pandas Series of dividend-adjusted returns.
        The first element will be NaN.
        (p. 60)
    """
    # Ensure alignment, then fill NaNs in dividends with 0 for calculation
    aligned_prices, aligned_dividends = prices.align(dividends, join='left')
    aligned_dividends = aligned_dividends.fillna(0)

    price_diff = aligned_prices.diff()
    returns = (price_diff + aligned_dividends) / aligned_prices.shift(1)
    return returns

def calculate_log_returns(prices_or_simple_returns: pd.Series) -> pd.Series:
    """
    Calculate log returns from a series of prices or simple returns.

    If prices are provided, log return is ln(P_t / P_{t-1}).
    If simple returns are provided, log return is ln(1 + R_t).

    Parameters
    ----------
    prices_or_simple_returns : pd.Series
        A pandas Series of prices or simple returns.
        The function attempts to auto-detect if the input is prices or returns
        based on typical value ranges, but it's more robust if the nature
        of the series is known. For this implementation, we assume if values
        are typically much larger than 1, they are prices.
        A more robust check could involve checking if any value is < -1,
        which is impossible for simple returns if prices are positive.

    Returns
    -------
    pd.Series
        A pandas Series of log returns.
        The first element will be NaN if calculated from prices.
        (p. 61)
    """
    # A simple heuristic: if mean is very far from 0, assume prices.
    # This is not perfectly robust. A better way is to have the user specify.
    # However, for this exercise, we'll use a common convention.
    # If the input series contains negative values (except for potentially the first NaN),
    # it's more likely to be simple returns.
    # For prices, log(P_t / P_{t-1})
    # For simple returns R_t, log(1 + R_t)

    # Check if it's likely simple returns (values mostly between -1 and 1, for example)
    # or prices (values generally > 0, often > 1).
    # If there's a value <= -1, it cannot be 1+R unless R = -2 which means price became negative.
    # We'll assume positive prices.

    # If the series could be simple returns (e.g., mean close to 0, typical values small)
    # and not prices (e.g. values are not all positive or very large)
    # A common way to calculate log returns is from prices directly:
    # log_returns = np.log(prices) - np.log(prices.shift(1))
    # Or from simple returns:
    # log_returns = np.log(1 + simple_returns)

    # Given the ambiguity, the most direct interpretation from "prices_or_simple_returns"
    # is to provide flexibility. If we assume positive prices,
    # simple returns R_t = P_t/P_{t-1} - 1, so 1 + R_t = P_t/P_{t-1}.
    # Thus, ln(1 + R_t) = ln(P_t/P_{t-1}).
    # So, if the input is simple returns, we calculate ln(1 + R_t).
    # If the input is prices, we calculate ln(P_t / P_{t-1}).

    # Let's assume if the mean of the series (ignoring NaNs) is small (e.g. < 1),
    # it's simple returns. Otherwise, it's prices. This is a heuristic.
    if prices_or_simple_returns.mean() < 1 and prices_or_simple_returns.abs().max() < 2 : # Heuristic for simple returns
        return np.log(1 + prices_or_simple_returns)
    else: # Assumed to be prices
        return np.log(prices_or_simple_returns / prices_or_simple_returns.shift(1))


def calculate_excess_returns(asset_returns: pd.Series, risk_free_rate: Union[pd.Series, float]) -> pd.Series:
    """
    Calculate excess returns of an asset over a risk-free rate.

    Excess Return = Asset Return - Risk-Free Rate

    Parameters
    ----------
    asset_returns : pd.Series
        A pandas Series of asset returns (simple or log).
    risk_free_rate : Union[pd.Series, float]
        A pandas Series of risk-free rates (aligned with asset_returns) or a constant float.
        The risk-free rate should be of the same periodicity as asset_returns.

    Returns
    -------
    pd.Series
        A pandas Series of excess returns.
        (p. 62)
    """
    if isinstance(risk_free_rate, pd.Series):
        # Ensure series are aligned if risk_free_rate is a Series
        aligned_asset_returns, aligned_risk_free_rate = asset_returns.align(risk_free_rate, join='left')
        return aligned_asset_returns - aligned_risk_free_rate
    else: # float
        return asset_returns - risk_free_rate

def calculate_compounded_return_series(simple_returns: pd.Series) -> pd.Series:
    """
    Calculates a series of compounded returns from simple returns.

    This shows the growth of a $1 investment over time.
    Compounded Value_t = (1 + R_1) * (1 + R_2) * ... * (1 + R_t)

    Parameters
    ----------
    simple_returns : pd.Series
        A pandas Series of simple returns. NaNs (e.g. first element) will be handled.

    Returns
    -------
    pd.Series
        A pandas Series representing the compounded growth of $1.
        The first valid return point will show 1 + R_1.
        Points prior to the first valid simple return will be NaN.
        (p. 62)
    """
    # (1 + simple_returns).cumprod() handles NaNs by propagating them initially,
    # but once a valid number is hit, it computes correctly.
    # If the first simple_return is NaN, (1+NaN) is NaN.
    # We want the series to start from 1 at the point before the first return.
    # For example, if returns are [NaN, 0.1, 0.05]
    # (1+returns).cumprod() -> [NaN, 1.1, 1.1*1.05 = 1.155]
    # This is the standard behavior and typically what's desired.
    return (1 + simple_returns).cumprod()

def calculate_total_compounded_return(simple_returns: pd.Series) -> float:
    """
    Calculates the total compounded return over a period.

    Total Compounded Return = [(1 + R_1) * (1 + R_2) * ... * (1 + R_n)] - 1

    Parameters
    ----------
    simple_returns : pd.Series
        A pandas Series of simple returns. NaNs will be ignored in the product.

    Returns
    -------
    float
        The total compounded return as a single float.
        (p. 62)
    """
    # Drop NaNs before computation, as they would make the product NaN.
    compounded_product = (1 + simple_returns.dropna()).prod()
    return compounded_product - 1
