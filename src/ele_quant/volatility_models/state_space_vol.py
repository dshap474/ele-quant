import numpy as np
import pandas as pd
from typing import Optional
from ele_quant.utils.kalman_filter import KalmanFilter

def ewma_variance_forecast(
    squared_returns: pd.Series,
    smoothing_factor_K: float,
    initial_variance: Optional[float] = None
) -> pd.Series:
    """
    Calculates Exponentially Weighted Moving Average (EWMA) variance forecasts.

    The formula is sigma_hat_sq_t = (1-K) * r_{t-1}^2 + K * sigma_hat_sq_{t-1}.
    (Eq. 2.22, p. 82, "Elements of Quantitative Investing")
    Note: The equation implies sigma_hat_sq_t is the forecast for time t,
    using squared return from t-1 and forecast from t-1.

    Parameters
    ----------
    squared_returns : pd.Series
        A pandas Series of squared log returns (r_t^2). The index should align
        with typical time series indexing (e.g., t-1 observation is used for t forecast).
    smoothing_factor_K : float
        The smoothing factor K (lambda in some notations, often close to 1, e.g., 0.94 for RiskMetrics).
        Here K corresponds to lambda in RiskMetrics formulation sigma_t^2 = (1-lambda)*r_{t-1}^2 + lambda*sigma_{t-1}^2.
    initial_variance : Optional[float], default None
        The initial variance estimate (sigma_hat_sq_0 or sigma_hat_sq_1).
        If None, the first value of `squared_returns` is used as the initial actual
        variance, and the first forecast will use this. A more robust approach might be
        the mean of `squared_returns` if the series is long enough. For simplicity,
        using the first squared return to initialize the previous period's variance.

    Returns
    -------
    pd.Series
        A pandas Series of EWMA variance forecasts (sigma_hat_sq_t). The series will have
        the same index as `squared_returns`. The first element is the initial_variance.
    """
    n = len(squared_returns)
    if n == 0:
        return pd.Series([], dtype=float)

    variance_forecasts = np.zeros(n)

    if initial_variance is None:
        # Use the first squared return as the initial variance (sigma_hat_sq_{t-1} for the first r_{t-1}^2)
        # So, variance_forecasts[0] will be this initial value.
        # And the forecast for time 1 (index 1) will use r_0^2 and sigma_0^2.
        current_sigma_hat_sq = squared_returns.iloc[0] if n > 0 else 1e-4 # Fallback for empty
        if pd.isna(current_sigma_hat_sq) or current_sigma_hat_sq <= 0:
            current_sigma_hat_sq = np.mean(squared_returns.dropna()) # Try mean if first is problematic
            if pd.isna(current_sigma_hat_sq) or current_sigma_hat_sq <=0:
                 current_sigma_hat_sq = 1e-4 # Default small positive
    else:
        current_sigma_hat_sq = initial_variance
        if current_sigma_hat_sq <=0: # Ensure positivity
            current_sigma_hat_sq = 1e-4


    variance_forecasts[0] = current_sigma_hat_sq # This is sigma_hat_sq_0

    for t in range(1, n):
        # sigma_hat_sq_t = (1-K) * r_{t-1}^2 + K * sigma_hat_sq_{t-1}
        # squared_returns.iloc[t-1] is r_{t-1}^2
        # variance_forecasts[t-1] is sigma_hat_sq_{t-1}
        r_tm1_sq = squared_returns.iloc[t-1]
        if pd.isna(r_tm1_sq): # Handle potential NaNs in returns
            r_tm1_sq = variance_forecasts[t-1] # Assume if return is NaN, variance contribution is based on forecast

        variance_forecasts[t] = (1 - smoothing_factor_K) * r_tm1_sq + \
                                 smoothing_factor_K * variance_forecasts[t-1]
        if variance_forecasts[t] <=0: # Ensure positivity
            variance_forecasts[t] = 1e-8

    return pd.Series(variance_forecasts, index=squared_returns.index)


def muth_model_variance_estimation(
    observed_squared_returns: pd.Series,
    tau_w_sq: float,
    tau_v_sq: float,
    initial_x_hat: Optional[float] = None,
    initial_sigma_sq_x_hat: Optional[float] = None
) -> pd.Series:
    """
    Estimates true underlying variance using Muth's model with a Kalman filter.

    State equation: x_t = x_{t-1} + w_t,  w_t ~ N(0, tau_w_sq) (x_t is true variance)
    Observation equation: y_t = x_t + v_t, v_t ~ N(0, tau_v_sq) (y_t is observed_squared_returns)
    (p. 83-84, "Elements of Quantitative Investing")

    Parameters
    ----------
    observed_squared_returns : pd.Series
        Pandas Series of observed squared returns (y_t).
    tau_w_sq : float
        Variance of the process noise for true variance (w_t). This is Q.
    tau_v_sq : float
        Variance of the measurement noise (v_t). This is R.
    initial_x_hat : Optional[float], default None
        Initial estimate for the true variance (x_0). If None, uses the first
        observed_squared_returns value.
    initial_sigma_sq_x_hat : Optional[float], default None
        Initial uncertainty (variance) of x_0. If None, uses tau_w_sq + tau_v_sq.

    Returns
    -------
    pd.Series
        A pandas Series of estimated true variances (x_t_hat), with the same index
        as `observed_squared_returns`.
    """
    n = len(observed_squared_returns)
    if n == 0:
        return pd.Series([], dtype=float)

    A = np.array([[1.0]])
    B = np.array([[1.0]])
    Q = np.array([[tau_w_sq]])
    R = np.array([[tau_v_sq]])

    if initial_x_hat is None:
        x0_hat = observed_squared_returns.iloc[0] if n > 0 else 1e-4
        if pd.isna(x0_hat) or x0_hat <=0: x0_hat = 1e-4
    else:
        x0_hat = initial_x_hat
        if x0_hat <=0: x0_hat = 1e-4


    if initial_sigma_sq_x_hat is None:
        sigma0_sq_x_hat = tau_w_sq + tau_v_sq
        if sigma0_sq_x_hat <=0 : sigma0_sq_x_hat = tau_w_sq # if tau_v_sq is negative
    else:
        sigma0_sq_x_hat = initial_sigma_sq_x_hat
    if sigma0_sq_x_hat <=0: sigma0_sq_x_hat = 1e-4


    kf = KalmanFilter(A, B, Q, R, np.array([x0_hat]), np.array([[sigma0_sq_x_hat]]))

    estimated_true_variances = np.zeros(n)
    estimated_true_variances[0] = x0_hat # Store initial estimate

    for t in range(n): # Iterate through all observations
        y_t_observed = np.array([observed_squared_returns.iloc[t]])
        if pd.isna(y_t_observed[0]): # Skip update if observation is NaN
            if t > 0: # If not first, carry forward previous estimate
                 estimated_true_variances[t] = estimated_true_variances[t-1]
            # If t=0 and y_t is NaN, x0_hat is already stored.
            # No predict/update, KF state remains as is for next valid observation.
            # Or, one might choose to predict only:
            # kf.predict()
            # estimated_true_variances[t] = kf.x_hat_k_k_minus_1[0,0]
            # kf.Sigma_k_minus_1_k_minus_1 = kf.Sigma_k_k_minus_1 # Update Sigma for next step too
            # For simplicity, just carry forward if observation is missing.
            if t > 0 : estimated_true_variances[t] = estimated_true_variances[t-1]
            else: estimated_true_variances[t] = x0_hat # already set
            if t < n-1 : # if not the last observation, update kf state for next iteration
                kf.predict() # advance state based on model
                kf.x_hat_k_minus_1_k_minus_1 = kf.x_hat_k_k_minus_1 # Update for next step (no new info)
                kf.Sigma_k_minus_1_k_minus_1 = kf.Sigma_k_k_minus_1
            continue


        if t == 0: # For the first observation, it was used as initial_x_hat
            # We can consider kf.update(y_t_observed) to refine x0_hat using y0
            # The current KF design: x_hat_k_minus_1_k_minus_1 is x0_hat.
            # predict() would give x_hat_0|-1 (if previous existed) or x_1|0 if current is t=0.
            # Let's align with standard loop: predict then update.
            # The first value in estimated_true_variances is the posterior from P_0.
            pass # x0_hat is already kf.x_hat_k_minus_1_k_minus_1

        kf.predict()
        x_hat_t_t, _ = kf.update(y_t_observed)
        estimated_true_variances[t] = x_hat_t_t[0,0]
        if estimated_true_variances[t] <=0: # Ensure positivity
            estimated_true_variances[t] = 1e-8
            kf.x_hat_k_minus_1_k_minus_1[0,0] = 1e-8 # also update filter state


    return pd.Series(estimated_true_variances, index=observed_squared_returns.index)


def harvey_shephard_volatility_estimation(
    log_returns: pd.Series,
    b_state: float,
    a_state: float,
    sigma_epsilon_sq_state: float,
    beta_hs: float = 0.0,
    initial_x_hat: Optional[float] = None,
    initial_sigma_sq_x_hat: Optional[float] = None
) -> pd.Series:
    """
    Estimates volatility using the Harvey-Shephard state-space model.

    State equation: x_t = b_state + a_state * x_{t-1} + epsilon_t_state (x_t = log(h_t^2))
    Observation: y_t = log(r_t^2) = x_t + log(epsilon_t^2) (log(epsilon_t^2) is obs noise)
    The constant beta_hs can be added to y_t to approximate E[log(epsilon_t^2)].
    (p. 85-86, "Elements of Quantitative Investing")

    Parameters
    ----------
    log_returns : pd.Series
        Pandas Series of log returns (r_t).
    b_state : float
        Constant term in the state equation for log(h_t^2).
    a_state : float
        Autoregressive coefficient for log(h_t^2) (persistence).
    sigma_epsilon_sq_state : float
        Variance of the process noise for log(h_t^2) (epsilon_t_state). This is Q.
    beta_hs : float, default 0.0
        Approximation for E[log(chi_1_square)], often -1.27. Added to log(r_t^2).
        If your y_t definition already includes it, set to 0.
    initial_x_hat : Optional[float], default None
        Initial estimate for x_0 = log(h_0^2). If None, uses log of sample variance of returns.
    initial_sigma_sq_x_hat : Optional[float], default None
        Initial uncertainty (variance) of x_0. If None, uses unconditional variance of state
        (sigma_epsilon_sq_state / (1 - a_state^2)) if |a_state|<1, else sigma_epsilon_sq_state.

    Returns
    -------
    pd.Series
        A pandas Series of estimated volatilities (h_t = exp(x_t_hat/2)), with the
        same index as `log_returns`.
    """
    n = len(log_returns)
    if n == 0:
        return pd.Series([], dtype=float)

    # Prepare y_t = log(r_t^2 + offset) + beta_hs
    small_offset = 1e-16 # To prevent log(0)
    y_t_observed_series = np.log(log_returns**2 + small_offset) + beta_hs

    A_hs = np.array([[a_state]])
    B_hs = np.array([[1.0]])
    Q_hs = np.array([[sigma_epsilon_sq_state]])
    R_hs = np.array([[np.pi**2 / 2]]) # Variance of log(chi_1_sq) noise term

    if initial_x_hat is None:
        sample_var_ret = np.var(log_returns.dropna())
        if sample_var_ret <= small_offset : sample_var_ret = small_offset * 10
        x0_hat = np.log(sample_var_ret) # log of sample variance of returns
    else:
        x0_hat = initial_x_hat

    if initial_sigma_sq_x_hat is None:
        if np.abs(a_state) < 1.0 - 1e-8: # Check for stationarity
            sigma0_sq_x_hat = sigma_epsilon_sq_state / (1 - a_state**2)
        else:
            sigma0_sq_x_hat = sigma_epsilon_sq_state # If non-stationary or at boundary
        if sigma0_sq_x_hat <=0: sigma0_sq_x_hat = 1e-4
    else:
        sigma0_sq_x_hat = initial_sigma_sq_x_hat
    if sigma0_sq_x_hat <=0: sigma0_sq_x_hat = 1e-4


    kf = KalmanFilter(A_hs, B_hs, Q_hs, R_hs, np.array([x0_hat]), np.array([[sigma0_sq_x_hat]]))

    estimated_log_h_sq = np.zeros(n)
    estimated_log_h_sq[0] = x0_hat # Store initial estimate for x_0

    for t in range(n):
        y_t = np.array([y_t_observed_series.iloc[t]])

        if pd.isna(y_t[0]): # Skip if transformed observation is NaN
            if t > 0: estimated_log_h_sq[t] = estimated_log_h_sq[t-1]
            else: estimated_log_h_sq[t] = x0_hat # already set
            if t < n-1 :
                # State prediction: x_{t|t-1} = b_state + a_state * x_{t-1|t-1}
                # KF predict gives: A @ x_hat_k_minus_1_k_minus_1
                # So, add b_state manually.
                kf.predict()
                kf.x_hat_k_k_minus_1[0,0] += b_state # Add control input b_state
                kf.x_hat_k_minus_1_k_minus_1 = kf.x_hat_k_k_minus_1 # Update for next step
                kf.Sigma_k_minus_1_k_minus_1 = kf.Sigma_k_k_minus_1
            continue

        # Predict step: x_hat_{t|t-1} = a_state * x_hat_{t-1|t-1} + Q_hs (done by kf.predict)
        # Then add b_state: x_hat_{t|t-1} = b_state + a_state * x_hat_{t-1|t-1}
        kf.predict()
        kf.x_hat_k_k_minus_1[0,0] += b_state # Manually add control input b_state to predicted state
                                          # This modifies the internal state of KF for the update step.

        # Update step using y_t = log(r_t^2)
        x_hat_t_t, _ = kf.update(y_t)
        estimated_log_h_sq[t] = x_hat_t_t[0,0]

    # Convert log(h_t^2) back to h_t (volatility)
    estimated_volatilities = np.exp(estimated_log_h_sq / 2)

    return pd.Series(estimated_volatilities, index=log_returns.index)
