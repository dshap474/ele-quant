import unittest
import pandas as pd
import numpy as np

# Functions to be tested
from quant_elements_lib.risk_evaluation import (
    calculate_qlike_loss,
    calculate_mse_variance_ratio_loss,
    evaluate_random_portfolio_variance, # Smoke test due to randomness
    calculate_qdist_likelihood,
    test_minimum_variance_portfolios,
    calculate_malv_statistic,
    calculate_fmp_turnover
)

# Small epsilon for floating point comparisons
EPSILON = 1e-7 # Standard for assertAlmostEqual
ASSET_NAMES = ['AssetA', 'AssetB']

class TestRiskEvaluation(unittest.TestCase):

    def test_qlike_loss_known_values(self):
        print("\nRunning test_qlike_loss_known_values...")
        # Case 1: Perfect match
        realized_perfect = pd.Series([1.0, 2.0], index=ASSET_NAMES)
        predicted_perfect = pd.Series([1.0, 2.0], index=ASSET_NAMES)
        # Loss = (1/T) * sum( (r/p) - log(r/p) - 1 )
        # If r=p, then (1 - log(1) - 1) = 1 - 0 - 1 = 0. So loss is 0.
        self.assertAlmostEqual(calculate_qlike_loss(realized_perfect, predicted_perfect), 0.0, places=6)

        # Case 2: Simple mismatch
        realized_mismatch = pd.Series([1.0, 2.0], index=ASSET_NAMES)
        predicted_mismatch = pd.Series([0.8, 2.5], index=ASSET_NAMES)
        # r1/p1 = 1.0/0.8 = 1.25; log(1.25) approx 0.22314
        # term1 = 1.25 - 0.22314355131 - 1 = 0.02685644869
        # r2/p2 = 2.0/2.5 = 0.8; log(0.8) approx -0.22314
        # term2 = 0.8 - (-0.22314355131) - 1 = 0.02314355131
        # QLIKE = (term1 + term2) / 2
        expected_qlike = ( (1.0/0.8 - np.log(1.0/0.8) - 1) + (2.0/2.5 - np.log(2.0/2.5) - 1) ) / 2
        self.assertAlmostEqual(calculate_qlike_loss(realized_mismatch, predicted_mismatch), expected_qlike, places=6)

        # Case 3: Handling of zero or negative predicted values (via epsilon in function)
        # The function adds epsilon, so direct log(0) or division by 0 is avoided.
        # Let's test with values close to zero that would be affected by internal epsilon.
        # Assume internal epsilon is 1e-8.
        # If predicted is 0, it becomes epsilon. r/epsilon - log(r/epsilon) -1. This can be large.
        realized_edge = pd.Series([1.0], index=['AssetA'])
        predicted_near_zero = pd.Series([1e-10], index=['AssetA']) # Smaller than internal epsilon
        # The function adds its own EPSILON (default 1e-8) to predicted.
        # So predicted becomes effectively 1e-8 + 1e-10 approx 1e-8.
        # Ratio = 1.0 / (1e-10 + 1e-8) approx 1.0 / 1e-8 = 1e8
        # log(Ratio) = log(1e8) = 8 * log(10) approx 8 * 2.302585 = 18.42068
        # Loss = (1e8 - 18.42068 - 1) approx 1e8. This is a large number.
        # The exact value depends on the function's internal epsilon.
        # For this test, let's just ensure it runs and returns a positive value.
        loss_near_zero = calculate_qlike_loss(realized_edge, predicted_near_zero)
        self.assertTrue(loss_near_zero > 0)
        # More precise test if we know the exact epsilon used in calculate_qlike_loss
        func_epsilon = 1e-8 # from loss_functions.py
        effective_pred = predicted_near_zero.iloc[0] + func_epsilon
        ratio = realized_edge.iloc[0] / effective_pred
        expected_loss_edge = ratio - np.log(ratio) - 1
        self.assertAlmostEqual(loss_near_zero, expected_loss_edge, places=6)


    def test_mse_variance_ratio_loss_known_values(self):
        print("\nRunning test_mse_variance_ratio_loss_known_values...")
        # Case 1: Perfect match
        realized_perfect = pd.Series([1.0, 2.0], index=ASSET_NAMES)
        predicted_perfect = pd.Series([1.0, 2.0], index=ASSET_NAMES)
        # Loss = (1/T) * sum( (r/p - 1)^2 )
        # If r=p, then (1-1)^2 = 0. Loss is 0.
        self.assertAlmostEqual(calculate_mse_variance_ratio_loss(realized_perfect, predicted_perfect), 0.0, places=6)

        # Case 2: Simple mismatch
        realized_mismatch = pd.Series([1.0, 2.0], index=ASSET_NAMES)
        predicted_mismatch = pd.Series([0.8, 2.5], index=ASSET_NAMES)
        # r1/p1 = 1.25; (1.25-1)^2 = 0.25^2 = 0.0625
        # r2/p2 = 0.8; (0.8-1)^2 = (-0.2)^2 = 0.04
        # MSE = (0.0625 + 0.04) / 2 = 0.1025 / 2 = 0.05125
        expected_mse = ( ((1.0/0.8 - 1)**2) + ((2.0/2.5 - 1)**2) ) / 2
        self.assertAlmostEqual(calculate_mse_variance_ratio_loss(realized_mismatch, predicted_mismatch), expected_mse, places=6)

    def test_qdist_simple_case(self):
        print("\nRunning test_qdist_simple_case...")
        # Omega = [[2, 0.5], [0.5, 1]]
        omega_df = pd.DataFrame([[2.0, 0.5], [0.5, 1.0]], columns=ASSET_NAMES, index=ASSET_NAMES)
        # r_t = [0.1, -0.05] for one period
        r_t_series = pd.DataFrame([[0.1, -0.05]], columns=ASSET_NAMES, index=[pd.Timestamp('2023-01-01')])

        # Manual calculation
        omega_val = omega_df.values
        r_t_val = r_t_series.iloc[0].values

        # Add epsilon for stability as in the function
        func_epsilon_qdist = 1e-12 # from matrix_evaluation.py
        omega_stable = omega_val + func_epsilon_qdist * np.eye(omega_val.shape[0])

        inv_omega = np.linalg.inv(omega_stable)
        det_omega = np.linalg.det(omega_stable)
        log_det_omega = np.log(det_omega)

        quadratic_term = r_t_val.T @ inv_omega @ r_t_val
        expected_qdist = quadratic_term + log_det_omega

        self.assertAlmostEqual(calculate_qdist_likelihood(omega_df, r_t_series), expected_qdist, places=6)

        # Test with list of matrices (single entry list for simplicity)
        self.assertAlmostEqual(calculate_qdist_likelihood([omega_df], r_t_series), expected_qdist, places=6)


    def test_minimum_variance_portfolios_2assets(self):
        print("\nRunning test_minimum_variance_portfolios_2assets...")
        omega_df = pd.DataFrame([[0.01, 0.0012], [0.0012, 0.0025]], columns=ASSET_NAMES, index=ASSET_NAMES)
        r_t = pd.Series([0.01, -0.005], index=ASSET_NAMES) # Realized returns for one period

        # Manual MVP calculation
        # Add epsilon for stability as in the function
        func_epsilon_mvp = 1e-12 # from matrix_evaluation.py (used as EPSILON there)
        omega_stable = omega_df.values + func_epsilon_mvp * np.eye(omega_df.shape[0])

        inv_omega = np.linalg.pinv(omega_stable) # Use pinv as in function
        ones_vector = np.ones(omega_df.shape[0])

        numerator = inv_omega @ ones_vector
        denominator = ones_vector.T @ inv_omega @ ones_vector
        expected_w_mvp_val = numerator / denominator
        expected_w_mvp = pd.Series(expected_w_mvp_val, index=ASSET_NAMES)

        expected_real_var = (expected_w_mvp_val.T @ r_t.values)**2

        real_var, w_mvp = test_minimum_variance_portfolios(omega_df, r_t)

        self.assertAlmostEqual(real_var, expected_real_var, places=6)
        pd.testing.assert_series_equal(w_mvp, expected_w_mvp, rtol=1e-5) # Higher rtol for pinv differences

    def test_malv_simple_case(self):
        print("\nRunning test_malv_simple_case...")
        # P = [[100, -20], [-20, 200]] (a sample precision matrix)
        precision_matrix_df = pd.DataFrame([[100.0, -20.0], [-20.0, 200.0]], columns=ASSET_NAMES, index=ASSET_NAMES)
        # r_t = [0.01, 0.02] for one period
        r_t_series = pd.DataFrame([[0.01, 0.02]], columns=ASSET_NAMES, index=[pd.Timestamp('2023-01-01')])

        # Manual calculation: (1/T) * sum(r_t.T @ P @ r_t)
        # Here T=1
        r_t_val = r_t_series.iloc[0].values
        precision_val = precision_matrix_df.values

        expected_malv = r_t_val.T @ precision_val @ r_t_val # Since T=1

        self.assertAlmostEqual(calculate_malv_statistic(precision_matrix_df, r_t_series), expected_malv, places=6)

        # Test with list of matrices (single entry list for simplicity)
        self.assertAlmostEqual(calculate_malv_statistic([precision_matrix_df], r_t_series), expected_malv, places=6)


    def test_fmp_turnover_basic(self):
        print("\nRunning test_fmp_turnover_basic...")
        factor_names = ['Factor1']
        p_t0 = pd.DataFrame({factor_names[0]: [0.6, 0.4]}, index=ASSET_NAMES)
        p_t1 = pd.DataFrame({factor_names[0]: [0.7, 0.3]}, index=ASSET_NAMES)

        fmp_weights_list = [p_t0, p_t1]

        # Manual calculation: ||P_t1 - P_t0||_F^2 / (T-1)
        # T-1 = 1 here
        delta_p = p_t1.values - p_t0.values
        # delta_p = [[0.1], [-0.1]]
        # Frobenius norm squared = sum of squares of elements = (0.1)^2 + (-0.1)^2 = 0.01 + 0.01 = 0.02
        expected_turnover = np.linalg.norm(delta_p, 'fro')**2

        self.assertAlmostEqual(calculate_fmp_turnover(fmp_weights_list), expected_turnover, places=6)

        # Test with T < 2 (should return nan as per implementation)
        self.assertTrue(np.isnan(calculate_fmp_turnover([p_t0])))
        self.assertTrue(np.isnan(calculate_fmp_turnover([])))

    def test_evaluate_random_portfolio_variance_smoke(self):
        print("\nRunning test_evaluate_random_portfolio_variance_smoke...")
        omega_df = pd.DataFrame([[0.01, 0.0012], [0.0012, 0.0025]], columns=ASSET_NAMES, index=ASSET_NAMES)
        # Ensure omega_df is positive definite for the test
        omega_df = omega_df + np.eye(omega_df.shape[0]) * 1e-8

        # 5 periods x 2 assets
        np.random.seed(42) # for reproducibility if needed, though output not checked precisely
        log_returns_eval = pd.DataFrame(np.random.randn(5, 2) * 0.01, columns=ASSET_NAMES)

        # Use a small number of portfolios for speed
        result = evaluate_random_portfolio_variance(
            omega_df, log_returns_eval, calculate_qlike_loss, num_random_portfolios=10
        )

        self.assertIsInstance(result, float)
        self.assertFalse(np.isnan(result))
        self.assertFalse(np.isinf(result))
        # We can't easily check the exact value due to randomness in portfolio weights.
        # A possible check is if the loss is non-negative, but QLIKE can be negative.
        # For QLIKE, if predicted variance is much larger than realized, (r/p) is small,
        # log(r/p) is large negative. So (r/p - log(r/p) -1) can be positive.
        # If predicted variance is much smaller than realized, (r/p) is large,
        # log(r/p) is large positive. (r/p - log(r/p) -1) can be positive.

if __name__ == '__main__':
    unittest.main()
