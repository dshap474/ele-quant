import numpy as np
import pandas as pd
from ele_quant.utils.kalman_filter import KalmanFilter # Corrected import path

def estimate_true_price_roll_model(
    observed_prices: pd.Series,
    sigma_m_sq: float,
    sigma_eta_sq: float
) -> pd.Series:
    """
    Estimates the true underlying prices using the Roll model with a Kalman filter.

    The Roll model assumes the observed price P_t is the sum of the true price m_t
    and a bid-ask bounce component eta_t. The true price m_t is assumed to follow
    a random walk: m_t = m_{t-1} + epsilon_t.

    State-space representation:
    - State equation: m_t = m_{t-1} + epsilon_t  (epsilon_t ~ N(0, sigma_m_sq))
    - Observation equation: P_t = m_t + eta_t     (eta_t ~ N(0, sigma_eta_sq))

    Parameters
    ----------
    observed_prices : pd.Series
        A pandas Series of observed market prices (P_t).
    sigma_m_sq : float
        Variance of the random walk increments for the true price (epsilon_t), i.e., Q.
        This represents the process noise variance. (p.63, p.93)
    sigma_eta_sq : float
        Variance of the bid-ask bounce component (eta_t), i.e., R.
        This represents the measurement noise variance. (p.63, p.93)

    Returns
    -------
    pd.Series
        A pandas Series of estimated true prices (m_t_hat), with the same index
        as `observed_prices`.

    References
    ----------
    - Based on Roll's model of market microstructure.
    - Example 2.2, p.93, "Elements of Quantitative Investing" (for Kalman filter application).
    - General model discussion p.63-64.
    """
    if observed_prices.empty:
        return pd.Series([], dtype=float)

    # State-space model matrices
    # m_t = 1 * m_{t-1} + epsilon_t
    A = np.array([[1.0]])  # State transition matrix for m_t

    # P_t = 1 * m_t + eta_t
    B = np.array([[1.0]])  # Observation matrix

    # Covariance of process noise epsilon_t
    Q = np.array([[sigma_m_sq]])

    # Covariance of measurement noise eta_t
    R = np.array([[sigma_eta_sq]])

    # Initial state estimate and covariance
    # x_hat_initial: Best guess for m_0. Use the first observed price.
    x_hat_initial = np.array([observed_prices.iloc[0]])

    # Sigma_initial: Initial uncertainty about m_0.
    # Example 2.2 (p.93) suggests P_0 = Q if m_0 is unknown and estimated.
    # Or, a larger variance like sigma_m_sq + sigma_eta_sq could be used.
    # Let's use Q as per typical initialization when m_0 is a rough guess.
    Sigma_initial = np.array([[sigma_m_sq]]) # Or Q.copy()

    # Initialize Kalman Filter
    kf = KalmanFilter(A=A, B=B, Q=Q, R=R, x_hat_initial=x_hat_initial, Sigma_initial=Sigma_initial)

    num_observations = len(observed_prices)
    estimated_true_prices = np.zeros(num_observations)

    # The initial state x_hat_initial (m_0_hat) is our estimate for m_0.
    # So, estimated_true_prices[0] can be x_hat_initial[0].
    # Then, we iterate from t=1 to update.
    # The KalmanFilter internal state x_hat_k_minus_1_k_minus_1 is initialized with x_hat_initial.

    # For the first observation P_0, we can consider it as 'y_0'.
    # The filter is initialized with x_hat_0_0 = P_0 and Sigma_0_0 = Q.
    # For P_1 (i.e., observed_prices.iloc[1]):
    # 1. Predict: x_hat_1_0, Sigma_1_0 = kf.predict() (uses x_hat_0_0, Sigma_0_0)
    # 2. Update: x_hat_1_1, Sigma_1_1 = kf.update(P_1) (uses x_hat_1_0, Sigma_1_0 and P_1)
    # Store x_hat_1_1.

    # The current KF implementation initializes x_hat_k_minus_1_k_minus_1 = x_hat_initial.
    # So, for the first call to predict(), it uses x_hat_initial.
    # Let's store the initial guess for the first price point.
    estimated_true_prices[0] = x_hat_initial[0]

    # Iterate through observed prices starting from the first one (index 0)
    # The filter's internal state x_hat_k_minus_1_k_minus_1 is already P_0
    # and Sigma_k_minus_1_k_minus_1 is Q.
    # So for P_0, we need to effectively run an "update" if we consider P_0 as the first measurement 'y_0'
    # to refine the initial guess x_hat_initial if it were different from P_0.
    # However, since x_hat_initial = P_0, an update with P_0 would refine P_0 using P_0.
    # The typical KF loop is predict then update.
    # If x_hat_0_0 is our estimate for m_0, then for m_1 (based on P_1):
    #   x_hat_1_0, Sigma_1_0 = kf.predict() (based on x_hat_0_0, Sigma_0_0)
    #   x_hat_1_1, Sigma_1_1 = kf.update(P_1)
    # This means estimated_true_prices[0] = x_hat_0_0
    # estimated_true_prices[1] = x_hat_1_1 etc.

    # The first price P_0 was used as x_hat_0_0. So, kf.x_hat_k_minus_1_k_minus_1 is x_hat_0_0.
    # We loop from the first observation.
    for t in range(num_observations):
        y_t = np.array([observed_prices.iloc[t]])

        if t == 0:
            # For the very first observation P_0, our initial guess is x_hat_0_0 = P_0.
            # We can consider this the result of an "update" step at t=0.
            # The KF is initialized with x_hat_0_0 and Sigma_0_0.
            # For the next step (t=1), we first predict m_1 using m_0.
            # So, the value at t=0 is x_hat_initial.
            estimated_true_prices[t] = kf.x_hat_k_minus_1_k_minus_1[0,0]
            # Or, one could run predict then update for t=0.
            # kf.predict() # x_hat_0|-1 from some prior, or x_1|0 from x_0|0
            # kf.update(y_0) # x_0|0
            # This depends on the exact timing convention for initialization.
            # Given x_hat_initial = P_0, this is x_hat_0_0. So store it.
        else:
            # Standard predict-update cycle for t > 0
            kf.predict()
            x_hat_t_t, _ = kf.update(y_t)
            estimated_true_prices[t] = x_hat_t_t[0,0] # KalmanFilter returns state as column vector

    return pd.Series(estimated_true_prices, index=observed_prices.index)
