import numpy as np

class KalmanFilter:
    """
    A generic Kalman Filter implementation.

    This filter estimates the state of a linear dynamic system perturbed by Gaussian noise,
    using a series of measurements observed over time, themselves corrupted by Gaussian noise.

    Attributes
    ----------
    A : np.ndarray
        State transition matrix. Relates the state at the previous time step k-1
        to the state at the current step k, in the absence of a control input or process noise.
    B : np.ndarray
        Observation matrix (often denoted as H in other literature). Maps the true
        state space into the observed space.
    Q : np.ndarray
        Process noise covariance matrix (Sigma_epsilon). Covariance of the process noise.
    R : np.ndarray
        Measurement noise covariance matrix (Sigma_eta). Covariance of the observation noise.
    x_hat_t_minus_1_t_minus_1 : np.ndarray
        State estimate at time t-1 given observations up to t-1 (posterior estimate at t-1).
        This is initialized with `x_hat_initial`. It is updated after each predict or update call.
        Represents x̂_{k|k} after an update, or x̂_{k|k-1} after a predict.
    Sigma_t_minus_1_t_minus_1 : np.ndarray
        State estimate covariance at time t-1 given observations up to t-1 (posterior covariance at t-1).
        This is initialized with `Sigma_initial`. It is updated after each predict or update call.
        Represents Σ_{k|k} after an update, or Σ_{k|k-1} after a predict.
    _I : np.ndarray
        Identity matrix, cached for use in the update step. Size is determined by state dimension.

    # Current state of the filter (after predict, these are prior; after update, these are posterior)
    # To be more precise with naming conventions in methods:
    # x_hat_k_k_minus_1 is the prior state estimate for time k
    # Sigma_k_k_minus_1 is the prior covariance for time k
    # x_hat_k_k is the posterior state estimate for time k
    # Sigma_k_k is the posterior covariance for time k
    # The attributes x_hat_t_minus_1_t_minus_1 and Sigma_t_minus_1_t_minus_1 will store the latest estimate
    # and its covariance, whether it's a prior (after predict) or posterior (after update).
    """

    def __init__(self,
                 A: np.ndarray,
                 B: np.ndarray,
                 Q: np.ndarray,
                 R: np.ndarray,
                 x_hat_initial: np.ndarray,
                 Sigma_initial: np.ndarray):
        """
        Initializes the Kalman Filter.

        Parameters
        ----------
        A : np.ndarray
            State transition matrix. Shape (n_states, n_states).
        B : np.ndarray
            Observation matrix. Shape (n_obs, n_states).
        Q : np.ndarray
            Process noise covariance matrix. Shape (n_states, n_states).
        R : np.ndarray
            Measurement noise covariance matrix. Shape (n_obs, n_obs).
        x_hat_initial : np.ndarray
            Initial state estimate. Shape (n_states, 1).
        Sigma_initial : np.ndarray
            Initial state estimate covariance. Shape (n_states, n_states).
        """
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R

        # These represent the state estimate and covariance from the PREVIOUS time step,
        # or the latest available estimate.
        # Initially, these are x̂_{0|0} and Σ_{0|0}.
        self.x_hat_k_minus_1_k_minus_1 = x_hat_initial # x̂_{t-1|t-1}
        self.Sigma_k_minus_1_k_minus_1 = Sigma_initial # Σ_{t-1|t-1}

        # These will store the results of the predict step (prior for current time k)
        self.x_hat_k_k_minus_1 = None # x̂_{t|t-1}
        self.Sigma_k_k_minus_1 = None # Σ_{t|t-1}

        # Cache identity matrix of the size of the state vector
        self._I = np.eye(A.shape[0])

    def predict(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Predicts the state and covariance for the current time step k,
        based on the state at k-1.

        Equations:
        - Predicted state estimate: x̂_{k|k-1} = A * x̂_{k-1|k-1} (Eq. 2.28)
        - Predicted estimate covariance: Σ_{k|k-1} = A * Σ_{k-1|k-1} * A^T + Q (Eq. 2.29)

        The results of this prediction (x̂_{k|k-1}, Σ_{k|k-1}) are stored internally
        and are used by the `update` method.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            - x_hat_k_k_minus_1 : Predicted state estimate for time k given observations up to k-1.
            - Sigma_k_k_minus_1 : Predicted estimate covariance for time k given observations up to k-1.
        """
        # Predict state estimate: x̂_{k|k-1} = A * x̂_{k-1|k-1}
        self.x_hat_k_k_minus_1 = self.A @ self.x_hat_k_minus_1_k_minus_1

        # Predict estimate covariance: Σ_{k|k-1} = A * Σ_{k-1|k-1} * A^T + Q
        self.Sigma_k_k_minus_1 = self.A @ self.Sigma_k_minus_1_k_minus_1 @ self.A.T + self.Q

        return self.x_hat_k_k_minus_1, self.Sigma_k_k_minus_1

    def update(self, y_k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Updates the state estimate and covariance with a new observation y_k at time k.

        This method must be called after `predict()`. It uses the predicted state
        (x̂_{k|k-1}) and predicted covariance (Σ_{k|k-1}) from the `predict` step.

        Equations:
        - Innovation (measurement residual): v_k = y_k - B * x̂_{k|k-1}
        - Innovation covariance: S_k = B * Σ_{k|k-1} * B^T + R
        - Kalman Gain: K_k = Σ_{k|k-1} * B^T * S_k^-1 (Eq. 2.30)
        - Updated state estimate: x̂_{k|k} = x̂_{k|k-1} + K_k * v_k (Eq. 2.31)
        - Updated estimate covariance: Σ_{k|k} = (I - K_k * B) * Σ_{k|k-1} (Eq. 2.32)

        The updated estimate (x̂_{k|k}) and covariance (Σ_{k|k}) become the basis
        for the next prediction (i.e., they become x̂_{k-1|k-1} and Σ_{k-1|k-1} for step k+1).

        Parameters
        ----------
        y_k : np.ndarray
            Current observation at time k. Shape (n_obs, 1).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            - x_hat_k_k : Updated (posterior) state estimate for time k.
            - Sigma_k_k : Updated (posterior) estimate covariance for time k.

        Raises
        ------
        ValueError
            If `predict()` was not called before `update()`.
        """
        if self.x_hat_k_k_minus_1 is None or self.Sigma_k_k_minus_1 is None:
            raise ValueError("predict() must be called before update().")

        # Innovation (measurement residual): v_k = y_k - B * x̂_{k|k-1}
        innovation_v_k = y_k - self.B @ self.x_hat_k_k_minus_1

        # Innovation covariance (or measurement prediction covariance): S_k = B * Σ_{k|k-1} * B^T + R
        innovation_cov_S_k = self.B @ self.Sigma_k_k_minus_1 @ self.B.T + self.R

        # Kalman Gain: K_k = Σ_{k|k-1} * B^T * S_k^-1 (Eq. 2.30)
        # Using np.linalg.solve for potentially better stability than np.linalg.inv
        # K_k * S_k = Sigma_k_k_minus_1 @ B.T
        # K_k = (Sigma_k_k_minus_1 @ B.T) @ np.linalg.inv(S_k)
        K_k = (Sigma_k_k_minus_1 @ self.B.T) @ np.linalg.inv(innovation_cov_S_k)

        # Updated state estimate: x̂_{k|k} = x̂_{k|k-1} + K_k * v_k (Eq. 2.31)
        x_hat_k_k = self.x_hat_k_k_minus_1 + K_k @ innovation_v_k

        # Updated estimate covariance: Σ_{k|k} = (I - K_k * B) * Σ_{k|k-1} (Eq. 2.32)
        Sigma_k_k = (self._I - K_k @ self.B) @ self.Sigma_k_k_minus_1

        # Update the state for the next iteration: x̂_{k-1|k-1} for the next step becomes x̂_{k|k}
        self.x_hat_k_minus_1_k_minus_1 = x_hat_k_k
        self.Sigma_k_minus_1_k_minus_1 = Sigma_k_k

        # Clear the prior for the next iteration, to enforce predict-update sequence
        self.x_hat_k_k_minus_1 = None
        self.Sigma_k_k_minus_1 = None

        return x_hat_k_k, Sigma_k_k
