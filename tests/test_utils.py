import unittest
import numpy as np
import pandas as pd # Though not directly used here, common in other tests

from ele_quant.utils.kalman_filter import KalmanFilter

class TestKalmanFilter(unittest.TestCase):

    def test_kalman_filter_known_case_1d(self):
        """
        Test Kalman Filter with a simple 1D system tracking a constant.
        Checks if state estimate converges and covariance decreases.
        """
        true_state = 5.0
        A = np.array([[1.0]])
        B = np.array([[1.0]])
        Q = np.array([[0.01]]) # Process noise (uncertainty in state evolution)
        R = np.array([[0.25]])  # Measurement noise (uncertainty in observation), made it larger for clearer effect

        x_initial = np.array([[0.0]]) # Initial guess for the state
        Sigma_initial = np.array([[10.0]]) # Initial uncertainty of the guess, made it larger

        kf = KalmanFilter(A, B, Q, R, x_initial, Sigma_initial)

        # Using fixed observations for repeatable tests
        observations_fixed = [4.8, 5.1, 4.95, 5.05, 5.0]

        current_x_hat_val = x_initial[0,0]
        current_Sigma_val = Sigma_initial[0,0]

        # print(f"Initial: x_hat={current_x_hat_val:.4f}, Sigma={current_Sigma_val:.4f}")

        for i, y_t_val in enumerate(observations_fixed):
            y_t = np.array([[y_t_val]])

            x_hat_k_minus_1_k_minus_1 = kf.x_hat_k_minus_1_k_minus_1 # Current best estimate before predict
            Sigma_k_minus_1_k_minus_1 = kf.Sigma_k_minus_1_k_minus_1

            # Predict step
            x_hat_pred, Sigma_pred = kf.predict() # Predicts x_k|k-1 and Sigma_k|k-1

            # Predicted uncertainty (Sigma_pred) should be greater than previous posterior uncertainty (Sigma_k_minus_1_k_minus_1) + Q
            # Sigma_pred = A * Sigma_k-1|k-1 * A' + Q
            self.assertAlmostEqual(Sigma_pred[0,0], A[0,0] * Sigma_k_minus_1_k_minus_1[0,0] * A[0,0].T + Q[0,0])
            if Q[0,0] > 1e-9 : # If there is process noise
                 self.assertGreater(Sigma_pred[0,0], Sigma_k_minus_1_k_minus_1[0,0] )


            # Update step
            x_hat_updated, Sigma_updated = kf.update(y_t) # Updates to x_k|k and Sigma_k|k

            # print(f"Obs {i+1}: {y_t_val:.2f} -> Pred x_hat: {x_hat_pred[0,0]:.4f}, Sigma_pred: {Sigma_pred[0,0]:.4f} -> Updated x_hat: {x_hat_updated[0,0]:.4f}, Sigma_updated: {Sigma_updated[0,0]:.4f}")

            # Check that the updated estimate has incorporated observation.
            # If R is very small, x_hat_updated should be close to y_t.
            # If R is very large, x_hat_updated should be close to x_hat_pred.
            # Key property: Updated uncertainty Sigma_updated should be less than predicted uncertainty Sigma_pred.
            self.assertLess(Sigma_updated[0,0], Sigma_pred[0,0])
            # And also less than measurement noise R (if state is observable)
            self.assertLess(Sigma_updated[0,0], R[0,0] + 1e-9)


            current_x_hat_val = x_hat_updated[0,0]
            current_Sigma_val = Sigma_updated[0,0]
            # print(f"After Upd {i+1}: x_hat={current_x_hat_val:.4f}, Sigma={current_Sigma_val:.4f}")


        # After a few steps, the estimate should be closer to the true_state (or mean of observations)
        self.assertAlmostEqual(current_x_hat_val, np.mean(observations_fixed), delta=0.5) # Looser check on mean
        # Uncertainty Sigma should have decreased significantly from initial Sigma.
        self.assertLess(current_Sigma_val, Sigma_initial[0,0] / 2) # Expect substantial reduction

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
