import numpy as np
import pandas as pd
from typing import Tuple, Union

class KalmanFilter:
    """
    Implements a standard Kalman Filter.

    The filter estimates the state of a linear dynamic system from a series of
    noisy measurements.
    """

    def __init__(
        self,
        A: Union[np.ndarray, pd.DataFrame],
        B: Union[np.ndarray, pd.DataFrame],
        Sigma_epsilon: Union[np.ndarray, pd.DataFrame],
        Sigma_eta: Union[np.ndarray, pd.DataFrame],
    ):
        """
        Initializes the Kalman Filter with system parameters.

        Args:
            A: State transition matrix (P x P). Defines how the state evolves from one time step to the next
               in the absence of noise and control inputs.
            B: Observation matrix (N x P). Defines how the state is transformed into observations.
               (Often denoted as H in some literature).
            Sigma_epsilon: Process noise covariance matrix (P x P). Covariance of the process noise.
                           (Often denoted as Q in some literature).
            Sigma_eta: Measurement noise covariance matrix (N x N). Covariance of the measurement noise.
                       (Often denoted as R in some literature).

        P is the dimension of the state space.
        N is the dimension of the observation space.
        """
        self.A = A.values if isinstance(A, pd.DataFrame) else A
        self.B = B.values if isinstance(B, pd.DataFrame) else B
        self.Sigma_epsilon = Sigma_epsilon.values if isinstance(Sigma_epsilon, pd.DataFrame) else Sigma_epsilon
        self.Sigma_eta = Sigma_eta.values if isinstance(Sigma_eta, pd.DataFrame) else Sigma_eta

        # Basic dimension checks
        if self.A.shape[0] != self.A.shape[1]:
            raise ValueError("State transition matrix A must be square.")
        if self.Sigma_epsilon.shape[0] != self.Sigma_epsilon.shape[1] or self.Sigma_epsilon.shape[0] != self.A.shape[0]:
            raise ValueError("Process noise covariance matrix Sigma_epsilon must be square and match A's dimensions.")
        if self.B.shape[1] != self.A.shape[0]:
            raise ValueError("Observation matrix B must have P columns (A.shape[0]).")
        if self.Sigma_eta.shape[0] != self.Sigma_eta.shape[1] or self.Sigma_eta.shape[0] != self.B.shape[0]:
            raise ValueError("Measurement noise covariance matrix Sigma_eta must be square and match B's row dimension (N).")

    def predict(
        self, x_hat_prev: np.ndarray, Sigma_prev: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs the predict step of the Kalman Filter.

        Args:
            x_hat_prev: Previous state estimate (P x 1 numpy array).
            Sigma_prev: Previous state estimate covariance (P x P numpy array).

        Returns:
            A tuple containing:
                - x_hat_pred: Predicted state estimate (P x 1 numpy array). (Eq. 2.28)
                - Sigma_pred: Predicted state estimate covariance (P x P numpy array). (Eq. 2.29)
        """
        if x_hat_prev.shape[0] != self.A.shape[0] or (x_hat_prev.ndim > 1 and x_hat_prev.shape[1] != 1) :
             raise ValueError(f"Previous state estimate x_hat_prev must be a Px1 vector (P={self.A.shape[0]})")
        if Sigma_prev.shape != (self.A.shape[0], self.A.shape[0]):
            raise ValueError(f"Previous state covariance Sigma_prev must be a PxP matrix (P={self.A.shape[0]})")


        # Predicted state estimate: x_hat_pred = A @ x_hat_prev
        x_hat_pred = self.A @ x_hat_prev

        # Predicted state covariance: Sigma_pred = A @ Sigma_prev @ A.T + Sigma_epsilon
        Sigma_pred = self.A @ Sigma_prev @ self.A.T + self.Sigma_epsilon

        return x_hat_pred, Sigma_pred

    def update(
        self,
        x_hat_pred: np.ndarray,
        Sigma_pred: np.ndarray,
        y_observed: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Performs the update step of the Kalman Filter.

        Args:
            x_hat_pred: Predicted state estimate (P x 1 numpy array from predict step).
            Sigma_pred: Predicted state estimate covariance (P x P numpy array from predict step).
            y_observed: Current observation (N x 1 numpy array).

        Returns:
            A tuple containing:
                - x_hat_updated: Updated state estimate (P x 1 numpy array). (Eq. 2.32, part 1)
                - Sigma_updated: Updated state estimate covariance (P x P numpy array). (Eq. 2.32, part 2)
                - K: Kalman gain (P x N numpy array). (Eq. 2.31)
        """
        if x_hat_pred.shape[0] != self.A.shape[0] or (x_hat_pred.ndim > 1 and x_hat_pred.shape[1] != 1) :
             raise ValueError(f"Predicted state estimate x_hat_pred must be a Px1 vector (P={self.A.shape[0]})")
        if Sigma_pred.shape != (self.A.shape[0], self.A.shape[0]):
            raise ValueError(f"Predicted state covariance Sigma_pred must be a PxP matrix (P={self.A.shape[0]})")
        if y_observed.shape[0] != self.B.shape[0] or (y_observed.ndim > 1 and y_observed.shape[1] != 1):
             raise ValueError(f"Observation y_observed must be an Nx1 vector (N={self.B.shape[0]})")

        # Innovation residual: v = y_observed - B @ x_hat_pred (Eq. 2.30, part 1)
        v = y_observed - self.B @ x_hat_pred

        # Innovation covariance: S = B @ Sigma_pred @ B.T + Sigma_eta (Eq. 2.30, part 2)
        S = self.B @ Sigma_pred @ self.B.T + self.Sigma_eta

        # Kalman gain: K = Sigma_pred @ B.T @ np.linalg.inv(S) (Eq. 2.31)
        # Using np.linalg.solve for potentially better stability and efficiency with S @ K.T
        # K = Sigma_pred @ B.T @ np.linalg.inv(S)
        # Sigma_pred @ B.T = K @ S
        # K = (Sigma_pred @ B.T) @ S_inv
        # S.T @ K.T = B @ Sigma_pred.T
        # S @ K.T = B @ Sigma_pred  (since Sigma_pred is symmetric)
        # K.T = np.linalg.solve(S, B @ Sigma_pred)
        # K = np.linalg.solve(S.T, (B @ Sigma_pred).T).T # This is one way if S is not symmetric
        # Or more directly:
        K = Sigma_pred @ self.B.T @ np.linalg.inv(S)


        # Updated state estimate: x_hat_updated = x_hat_pred + K @ v (Eq. 2.32, part 1)
        x_hat_updated = x_hat_pred + K @ v

        # Updated state covariance: Sigma_updated = Sigma_pred - K @ S @ K.T (Eq. 2.32, part 2)
        # This can also be written as (I - K @ B) @ Sigma_pred for better numerical stability in some cases,
        # but the form K @ S @ K.T is also common.
        Sigma_updated = Sigma_pred - K @ S @ K.T
        # Numerically stable version (Joseph form) for Sigma_updated, though more computations:
        # I_minus_KB = np.eye(self.A.shape[0]) - K @ self.B
        # Sigma_updated = I_minus_KB @ Sigma_pred @ I_minus_KB.T + K @ self.Sigma_eta @ K.T


        return x_hat_updated, Sigma_updated, K
