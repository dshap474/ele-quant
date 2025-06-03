import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

# Import functions to be tested
from ele_quant.volatility_models.garch import (
    garch_1_1_conditional_variance,
    estimate_garch_1_1_parameters,
    # garch_1_1_log_likelihood # Not directly tested but used by estimate
)
from ele_quant.volatility_models.realized_volatility import (
    calculate_realized_variance_from_sum_sq_returns,
    estimate_variance_mle_diffusion,
)
from ele_quant.volatility_models.state_space_vol import (
    ewma_variance_forecast,
    muth_model_variance_estimation,
    harvey_shephard_volatility_estimation,
)
from ele_quant.utils.kalman_filter import KalmanFilter # For state-space models

class TestGARCH(unittest.TestCase):

    def test_garch_1_1_conditional_variance_step(self):
        log_returns = pd.Series([0.01, -0.02, 0.015]) # r0, r1, r2
        alpha_0 = 0.00001
        alpha_1 = 0.05
        beta_1 = 0.90

        # Unconditional variance: alpha_0 / (1 - alpha_1 - beta_1)
        # = 0.00001 / (1 - 0.05 - 0.90) = 0.00001 / 0.05 = 0.0002
        initial_h_sq = alpha_0 / (1 - alpha_1 - beta_1) # Should be 0.0002
        self.assertAlmostEqual(initial_h_sq, 0.0002)

        h_sq_series = garch_1_1_conditional_variance(
            log_returns, alpha_0, alpha_1, beta_1, initial_h_sq=None # Test auto-calc of initial
        )

        # h_sq[0] is initial_h_sq (h_0^2 in notation where t starts from 0 for h)
        # h_sq[0] = h_1^2 if h_1^2 is the first element of the series
        # The implementation: h_sq[0] = initial_h_sq (h_0^2 for the code)
        # h_sq[1] = alpha_0 + alpha_1 * r_0^2 + beta_1 * h_0^2
        # h_sq[2] = alpha_0 + alpha_1 * r_1^2 + beta_1 * h_1^2

        expected_h_sq = np.zeros(3)
        expected_h_sq[0] = initial_h_sq # h_0^2
        expected_h_sq[1] = alpha_0 + alpha_1 * log_returns.iloc[0]**2 + beta_1 * expected_h_sq[0]
        # expected_h_sq[1] = 0.00001 + 0.05 * (0.01)**2 + 0.90 * 0.0002
        #                  = 0.00001 + 0.05 * 0.0001 + 0.90 * 0.0002
        #                  = 0.00001 + 0.000005 + 0.00018 = 0.000195
        self.assertAlmostEqual(expected_h_sq[1], 0.000195)

        expected_h_sq[2] = alpha_0 + alpha_1 * log_returns.iloc[1]**2 + beta_1 * expected_h_sq[1]
        # expected_h_sq[2] = 0.00001 + 0.05 * (-0.02)**2 + 0.90 * 0.000195
        #                  = 0.00001 + 0.05 * 0.0004 + 0.90 * 0.000195
        #                  = 0.00001 + 0.00002 + 0.0001755 = 0.0002055
        self.assertAlmostEqual(expected_h_sq[2], 0.0002055)

        assert_series_equal(h_sq_series, pd.Series(expected_h_sq, index=log_returns.index), rtol=1e-5)

    def test_estimate_garch_1_1_parameters_runs(self):
        # Generate some plausible log returns
        np.random.seed(42)
        log_returns_sample = pd.Series(np.random.normal(0, 0.02, 200)) # Longer series for stability

        params = estimate_garch_1_1_parameters(log_returns_sample)

        self.assertIsInstance(params, dict)
        self.assertTrue('alpha_0' in params)
        self.assertTrue('alpha_1' in params)
        self.assertTrue('beta_1' in params)
        self.assertTrue('log_likelihood' in params)

        if not np.isnan(params['alpha_0']): # If estimation succeeded
            self.assertGreater(params['alpha_0'], 0)
            self.assertGreaterEqual(params['alpha_1'], 0)
            self.assertLess(params['alpha_1'], 1.0) # alpha_1 < 1
            self.assertGreaterEqual(params['beta_1'], 0)
            self.assertLess(params['beta_1'], 1.0) # beta_1 < 1
            self.assertLess(params['alpha_1'] + params['beta_1'], 1.0) # Stationarity
        else:
            print("GARCH estimation returned NaNs, which can happen for some random series. Test primarily checks it runs.")


class TestRealizedVolatility(unittest.TestCase):

    def test_calculate_realized_variance_from_sum_sq_returns(self):
        # RV(n) = (n/T) * sum_squared_intraperiod_returns
        # If T=1 (e.g. daily variance from intraday), RV = n * sum_sq_ret
        sum_sq = 0.005
        n_obs = 10
        T_period = 1.0
        expected_rv = (n_obs / T_period) * sum_sq # 10 * 0.005 = 0.05
        calculated_rv = calculate_realized_variance_from_sum_sq_returns(sum_sq, n_obs, T_period)
        self.assertAlmostEqual(calculated_rv, expected_rv, places=5)

        T_period_annual = 252 # Example: annualizing
        # This formula's interpretation of T might not be for annualization directly.
        # If sum_sq is daily, n_obs=1 (1 daily observation), T=252.
        # Then RV = (1/252) * daily_sum_sq. This is variance per day, if T means "number of days in year".
        # The docstring tries to clarify this. For the test, we use the formula directly.
        # If sum_sq is from 1 day, n=1, T=1/252 (if T is in years), then (1/(1/252)) * sum_sq = 252 * sum_sq.
        # Let's test with T=0.5 (e.g. variance per half-period)
        T_period_half = 0.5
        expected_rv_half = (n_obs / T_period_half) * sum_sq # (10 / 0.5) * 0.005 = 20 * 0.005 = 0.1
        calculated_rv_half = calculate_realized_variance_from_sum_sq_returns(sum_sq, n_obs, T_period_half)
        self.assertAlmostEqual(calculated_rv_half, expected_rv_half, places=5)


    def test_estimate_variance_mle_diffusion(self):
        # sigma_hat_sq_n = (1/T) * sum((r_j - alpha_hat/n)**2)
        # alpha_hat = sum r_j
        returns = pd.Series([0.001, -0.0005, 0.0003, 0.0001])
        T_period = 1.0
        n = len(returns)
        alpha_hat = returns.sum() # 0.001 - 0.0005 + 0.0003 + 0.0001 = 0.0009
        mean_ret_contrib = alpha_hat / n # 0.0009 / 4 = 0.000225

        deviations_sq = ((returns - mean_ret_contrib)**2).sum()
        # (0.001-m)^2 = (0.000775)^2 = 6.00625e-07
        # (-0.0005-m)^2 = (-0.000725)^2 = 5.25625e-07
        # (0.0003-m)^2 = (0.000075)^2 = 0.05625e-07
        # (0.0001-m)^2 = (-0.000125)^2 = 0.15625e-07
        # Sum = (6.00625 + 5.25625 + 0.05625 + 0.15625)e-07 = 11.475e-07 = 1.1475e-06

        expected_mle_var = (1 / T_period) * deviations_sq # 1.1475e-06
        calculated_mle_var = estimate_variance_mle_diffusion(returns, T_period)
        self.assertAlmostEqual(calculated_mle_var, expected_mle_var, places=10)


class TestStateSpaceVol(unittest.TestCase):

    def test_ewma_variance_forecast(self):
        # sigma_hat_sq_t = (1-K) * r_{t-1}^2 + K * sigma_hat_sq_{t-1}
        sq_returns = pd.Series([0.0001, 0.0004, 0.00005]) # r0^2, r1^2, r2^2
        K = 0.9
        initial_var = 0.0001 # sigma_hat_sq_0

        forecasts = ewma_variance_forecast(sq_returns, K, initial_var)

        expected_forecasts = np.zeros(3)
        expected_forecasts[0] = initial_var # sigma_hat_sq_0
        # sigma_hat_sq_1 = (1-K)*r0^2 + K*sigma_hat_sq_0
        #                = 0.1 * 0.0001 + 0.9 * 0.0001 = 0.00001 + 0.00009 = 0.0001
        expected_forecasts[1] = (1-K)*sq_returns.iloc[0] + K*expected_forecasts[0]
        self.assertAlmostEqual(expected_forecasts[1], 0.0001)

        # sigma_hat_sq_2 = (1-K)*r1^2 + K*sigma_hat_sq_1
        #                = 0.1 * 0.0004 + 0.9 * 0.0001 = 0.00004 + 0.00009 = 0.00013
        expected_forecasts[2] = (1-K)*sq_returns.iloc[1] + K*expected_forecasts[1]
        self.assertAlmostEqual(expected_forecasts[2], 0.00013)

        assert_series_equal(forecasts, pd.Series(expected_forecasts, index=sq_returns.index), rtol=1e-5)

    def test_muth_model_runs(self):
        np.random.seed(42)
        sq_returns_sample = pd.Series(np.abs(np.random.normal(0.001, 0.005, 50))) # Needs positive values
        sq_returns_sample = sq_returns_sample.replace(0, 1e-8) # ensure positive for log if used elsewhere

        tau_w_sq = 1e-7
        tau_v_sq = 1e-6

        estimated_variances = muth_model_variance_estimation(sq_returns_sample, tau_w_sq, tau_v_sq)

        self.assertEqual(len(estimated_variances), len(sq_returns_sample))
        self.assertFalse(estimated_variances.isna().any())
        self.assertTrue((estimated_variances >= 0).all()) # Variances should be non-negative

    def test_harvey_shephard_model_runs(self):
        np.random.seed(42)
        log_returns_sample = pd.Series(np.random.normal(0, 0.02, 50))

        b_state = -0.1
        a_state = 0.95
        sigma_epsilon_sq_state = 1e-5
        beta_hs = -1.27 # Typical value for E[log(chi_1_sq)]

        estimated_volatilities = harvey_shephard_volatility_estimation(
            log_returns_sample, b_state, a_state, sigma_epsilon_sq_state, beta_hs=beta_hs
        )

        self.assertEqual(len(estimated_volatilities), len(log_returns_sample))
        self.assertFalse(estimated_volatilities.isna().any())
        self.assertTrue((estimated_volatilities >= 0).all()) # Volatilities should be non-negative

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
