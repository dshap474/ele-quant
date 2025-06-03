import numpy as np
import pandas as pd

def calculate_realized_variance_from_sum_sq_returns(
    sum_squared_intraperiod_returns: float,
    num_intraperiod_obs_n: int,
    period_length_T: float = 1.0
) -> float:
    """
    Calculates realized variance from the sum of squared intra-period returns.

    The formula used is RV(n) = (1/T) * sum_j (r_j^2) when r_j are returns over
    sub-intervals of length T/n. If r_j are returns over intervals of length 'delta_t',
    and T is the total period length we are normalizing to (e.g. T=1 for daily RV from
    minute returns, where delta_t = 1 minute), then the sum of r_j^2 is an estimate of
    variance over T if returns are scaled appropriately.

    The formula on p.78 is RV(n) = (n/T) * sum_j r(j)^2, where r(j) are returns over
    intervals of length T/n. This implies sum_j r(j)^2 is the sum of squared returns
    over these small intervals. If period_length_T corresponds to the period for which
    the variance is being annualized (e.g. T=1 for daily, T=252 for annualized daily),
    and num_intraperiod_obs_n is the number of such returns *within that period T*,
    then the formula (1/T) * sum_squared_intraperiod_returns would be appropriate if
    sum_squared_intraperiod_returns refers to returns already scaled to T.

    Let's assume:
    - `sum_squared_intraperiod_returns` is Σ r_j^2, where r_j are returns over sub-periods.
    - `num_intraperiod_obs_n` is the count of these sub-periods.
    - `period_length_T` is the length of the target period for which variance is estimated.
      If T=1, this is variance per one unit of time over which n observations occurred.
      If r_j are returns over intervals of length dt = target_period / n,
      then sum(r_j^2) is an estimate of variance over target_period.
      The formula RV = (1/T) * sum_of_squared_returns is often used where T is the
      number of periods in the target horizon (e.g. T=1 for daily RV if returns are daily).
      If we have n intra-period returns, r_1, ..., r_n, each over a time step dt.
      The total time is n*dt. If we want variance per unit time (where unit time = n*dt),
      then sum(r_j^2) is the variance.
      If we want variance per a different period T, then scaling is needed.

    The prompt's `(n/T) * sum_squared_intraperiod_returns` seems to imply that `T` is a
    normalizing factor for the number of observations `n`.
    Let's take the textbook formula directly: RV(n) = (n/T) * Σ_j r(j)^2.
    Here, T is the length of the "overall period" and n is the number of "sub-periods" within T.
    Example: Daily RV (T=1 day) from 1-minute returns (n=390 for a 6.5hr day).
    RV_daily = (n / 1) * sum(r_minute^2) would be incorrect if r_minute were daily returns.
    If r_j are returns over intervals of length delta_t = T/n, then RV_T = sum(r_j^2).
    The formula (n/T) * sum_j r(j)^2 from p.78 refers to r(j) as "returns in period j of length T/n".
    So if T=1 (e.g. 1 day), and n is number of intraday returns, then RV = n * sum(r_j^2). This seems high.

    Re-interpreting p.78: "RV(n) = (n/T) sum_{j=1 to T*n_obs_per_unit_T} r(j)^2" where r(j) are returns
    over intervals of length 1/n_obs_per_unit_T.
    If T is the number of days, and n is obs per day, then total obs is T*n.
    Realized variance for period T is sum_{i=1 to N_total} r_i^2.
    If we want *average daily* variance, it is (1/T) * sum_{i=1 to N_total} r_i^2.

    Let's assume `sum_squared_intraperiod_returns` is Σ r_j^2 for `n` returns,
    and we want the variance for the total period covered by these `n` returns, scaled by `period_length_T`.
    If `period_length_T = 1`, it's just the sum of squared returns.
    If `period_length_T` is, for example, the number of trading days in a year (252),
    and `sum_squared_intraperiod_returns` is for one day, and `num_intraperiod_obs_n`
    is effectively 1 (for daily returns), then `(1/252) * daily_variance` doesn't make sense.

    The most standard definition for realized variance over a period (e.g., one day)
    using `n` intra-period returns (e.g., 1-minute returns within that day) is simply
    the sum of the squared intra-period returns: RV_day = Σ r_minute^2.
    If an *annualized* RV is needed, then RV_annual = RV_day * Days_per_Year.

    Given the parameters: `sum_squared_intraperiod_returns`, `num_intraperiod_obs_n`, `period_length_T`.
    If `sum_squared_intraperiod_returns` is Σ r_j^2 for `num_intraperiod_obs_n` returns,
    and these `n` returns span a time `P`. We want variance for a period of length `T`.
    The variance for period `P` is `S = sum_squared_intraperiod_returns`.
    The variance per sub-period is `S / num_intraperiod_obs_n`.
    If `period_length_T` is a multiplier (e.g. T=252 for annualizing daily variance),
    then the result should be `S * period_length_T`.

    Let's stick to the formula given in the prompt description: `(n/T) * sum_squared_intraperiod_returns`.
    This implies `T` is a denominator for scaling. If `T=1` (e.g. variance per day), then `n * sum_sq_returns`.
    This is what the textbook says on p.78, `RV(n) = (n/T) Σ r_j^2`.
    This formula is for when `r_j` are returns over intervals of length `T/n`.
    E.g. T=1 (1 day), n=390 (minutes in a day). Interval length is `1/390` of a day.
    `r_j` is return in minute `j`.
    `RV(n) = n * Σ r_j^2` (if T=1). This is the daily variance.

    Parameters
    ----------
    sum_squared_intraperiod_returns : float
        The sum of squared returns within a base period (e.g., sum of squared 1-minute
        log returns within a trading day). This is Σ r_j^2.
    num_intraperiod_obs_n : int
        The number of intra-period observations used to calculate the sum. (n in the formula).
    period_length_T : float, default 1.0
        The length of the period for which the variance is being estimated, relative
        to the definition of n. If n is number of observations per day, and T=1,
        this gives daily variance. If T is number of days in year, and n is obs per T,
        this definition becomes tricky.
        Let's assume T is a scaling factor for the "density" of observations.
        As per p.78, T is the length of the overall period, and n is the number of
        sub-periods of length T/n. So n/T is the sampling frequency.

    Returns
    -------
    float
        The calculated realized variance.

    References
    ----------
    - p. 78, "Elements of Quantitative Investing". RV(n) = (n/T) * Σ_j r(j)^2
    """
    if num_intraperiod_obs_n <= 0:
        return np.nan
    if period_length_T <= 0:
        return np.nan

    # Assuming sum_squared_intraperiod_returns is Σ r_j^2, where r_j are returns
    # over intervals of length delta_t = period_length_T / num_intraperiod_obs_n.
    # The formula RV(n) = (n/T) * Σ r_j^2 from p.78 is stated where r(j) are
    # "returns in period j of length T/n".
    # This implies if T=1 (e.g. 1 day), and n is number of intraday returns,
    # then RV_daily = n * sum(r_j^2). This is a common but sometimes debated scaling.
    # A more common definition of daily RV from intraday returns is just sum(r_j^2).
    # Let's use the book's formula directly as requested.
    realized_variance = (num_intraperiod_obs_n / period_length_T) * sum_squared_intraperiod_returns
    return realized_variance


def estimate_variance_mle_diffusion(
    intra_period_returns: pd.Series,
    period_length_T: float = 1.0
) -> float:
    """
    Estimates variance of a diffusion process using Maximum Likelihood Estimation.

    The formula is sigma_hat_sq_n = (1/T) * sum_j ((r_j - alpha_hat/n)^2),
    where alpha_hat = sum_j r_j, and n is the number of intra-period returns.
    (Eq. 2.16, p. 79, "Elements of Quantitative Investing")

    Parameters
    ----------
    intra_period_returns : pd.Series
        A pandas Series of intra-period log returns (r_j).
    period_length_T : float, default 1.0
        The total length of the period T over which the n returns r_j occurred.
        For example, if r_j are 1-minute returns over a day, T=1 (for daily variance).
        If returns are daily and T=252, this would annualize the variance.

    Returns
    -------
    float
        The MLE estimate of the variance (sigma_hat_sq_n).
    """
    n = len(intra_period_returns)
    if n == 0:
        return np.nan
    if period_length_T <= 0:
        return np.nan

    alpha_hat = intra_period_returns.sum()
    mean_return_contribution = alpha_hat / n

    sum_sq_deviations = ((intra_period_returns - mean_return_contribution)**2).sum()

    sigma_hat_sq_n = (1 / period_length_T) * sum_sq_deviations

    return sigma_hat_sq_n
