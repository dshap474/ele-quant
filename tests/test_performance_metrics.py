import unittest
import pandas as pd
import numpy as np
from ele_quant.performance_metrics import (
    calculate_expected_return,
    calculate_volatility,
    calculate_sharpe_ratio,
    calculate_information_ratio,
    calculate_sharpe_ratio_se,
    adjust_sharpe_ratio_for_autocorrelation,
)

class TestPerformanceMetrics(unittest.TestCase):
    """
    Unit tests for performance metrics functions.
    """

    def setUp(self):
        """
        Set up common data for tests.
        """
        self.sample_returns_list = [0.01, 0.02, -0.01, 0.03, 0.005]
        self.sample_returns = pd.Series(self.sample_returns_list)

        self.active_returns_list = [0.005, -0.002, 0.01, -0.005, 0.003]
        self.active_returns = pd.Series(self.active_returns_list)

        self.trading_periods_per_year = 252
        self.sqrt_tppy = np.sqrt(self.trading_periods_per_year)

    def test_calculate_expected_return(self):
        """Test calculation of expected return."""
        expected = np.mean(self.sample_returns_list)
        result = calculate_expected_return(self.sample_returns)
        self.assertAlmostEqual(result, expected, places=7)

        empty_returns = pd.Series([], dtype=float)
        result_empty = calculate_expected_return(empty_returns)
        self.assertTrue(np.isnan(result_empty)) # Mean of empty series is NaN

    def test_calculate_volatility(self):
        """Test calculation of volatility."""
        # Non-annualized
        expected_std_dev_non_annualized = np.std(self.sample_returns_list, ddof=1)
        result_non_annualized = calculate_volatility(self.sample_returns, annualize=False)
        self.assertAlmostEqual(result_non_annualized, expected_std_dev_non_annualized, places=7)

        # Annualized
        expected_std_dev_annualized = expected_std_dev_non_annualized * self.sqrt_tppy
        result_annualized = calculate_volatility(
            self.sample_returns,
            trading_periods_per_year=self.trading_periods_per_year,
            annualize=True
        )
        self.assertAlmostEqual(result_annualized, expected_std_dev_annualized, places=7)

        # Annualized with trading_periods_per_year = 1
        result_annualized_tppy1 = calculate_volatility(
            self.sample_returns,
            trading_periods_per_year=1,
            annualize=True
        )
        self.assertAlmostEqual(result_annualized_tppy1, expected_std_dev_non_annualized, places=7)

        # Test with very few data points (std of 1 value is NaN, 2 values is calculable)
        single_return = pd.Series([0.01])
        self.assertTrue(np.isnan(calculate_volatility(single_return, annualize=False)))

        two_returns = pd.Series([0.01, 0.02])
        expected_two = np.std([0.01, 0.02], ddof=1)
        self.assertAlmostEqual(calculate_volatility(two_returns, annualize=False), expected_two, places=7)


    def test_calculate_sharpe_ratio(self):
        """Test calculation of Sharpe Ratio."""
        risk_free_rate_scalar = 0.0001 # Daily

        excess_returns_scalar_rf = self.sample_returns - risk_free_rate_scalar
        mean_excess_return = np.mean(excess_returns_scalar_rf)
        std_dev_excess_return = np.std(excess_returns_scalar_rf, ddof=1)

        # Non-annualized, scalar risk-free rate
        expected_sr_non_annualized = mean_excess_return / std_dev_excess_return
        result_sr_non_annualized = calculate_sharpe_ratio(
            self.sample_returns,
            risk_free_rate=risk_free_rate_scalar,
            annualize=False
        )
        self.assertAlmostEqual(result_sr_non_annualized, expected_sr_non_annualized, places=7)

        # Annualized, scalar risk-free rate
        expected_sr_annualized = expected_sr_non_annualized * self.sqrt_tppy
        result_sr_annualized = calculate_sharpe_ratio(
            self.sample_returns,
            risk_free_rate=risk_free_rate_scalar,
            trading_periods_per_year=self.trading_periods_per_year,
            annualize=True
        )
        self.assertAlmostEqual(result_sr_annualized, expected_sr_annualized, places=7)

        # With risk_free_rate as a Series
        risk_free_rate_series = pd.Series([risk_free_rate_scalar] * len(self.sample_returns))
        excess_returns_series_rf = self.sample_returns - risk_free_rate_series
        mean_excess_return_series_rf = np.mean(excess_returns_series_rf)
        std_dev_excess_return_series_rf = np.std(excess_returns_series_rf, ddof=1)

        expected_sr_series_rf_non_annualized = mean_excess_return_series_rf / std_dev_excess_return_series_rf
        result_sr_series_rf_non_annualized = calculate_sharpe_ratio(
            self.sample_returns,
            risk_free_rate=risk_free_rate_series,
            annualize=False
        )
        self.assertAlmostEqual(result_sr_series_rf_non_annualized, expected_sr_series_rf_non_annualized, places=7)

        # Edge case: zero volatility of excess returns
        constant_excess_returns = pd.Series([0.001, 0.001, 0.001]) # std_dev will be 0
        # If mean_excess_return is also 0, then 0/0 -> NaN
        # If mean_excess_return is non-zero, then x/0 -> inf or -inf, then NaN due to implementation
        result_zero_vol_positive_mean = calculate_sharpe_ratio(constant_excess_returns + 0.001, risk_free_rate=0.0, annualize=False)
        self.assertTrue(np.isnan(result_zero_vol_positive_mean))

        result_zero_vol_zero_mean = calculate_sharpe_ratio(constant_excess_returns * 0, risk_free_rate=0.0, annualize=False)
        self.assertTrue(np.isnan(result_zero_vol_zero_mean))

        # Test with risk_free_rate being a float
        result_sr_float_rf = calculate_sharpe_ratio(self.sample_returns, risk_free_rate=float(risk_free_rate_scalar), annualize=False)
        self.assertAlmostEqual(result_sr_float_rf, expected_sr_non_annualized, places=7)


    def test_calculate_information_ratio(self):
        """Test calculation of Information Ratio."""
        mean_active = np.mean(self.active_returns_list)
        std_dev_active = np.std(self.active_returns_list, ddof=1)

        # Non-annualized
        expected_ir_non_annualized = mean_active / std_dev_active
        result_ir_non_annualized = calculate_information_ratio(self.active_returns, annualize=False)
        self.assertAlmostEqual(result_ir_non_annualized, expected_ir_non_annualized, places=7)

        # Annualized
        expected_ir_annualized = expected_ir_non_annualized * self.sqrt_tppy
        result_ir_annualized = calculate_information_ratio(
            self.active_returns,
            trading_periods_per_year=self.trading_periods_per_year,
            annualize=True
        )
        self.assertAlmostEqual(result_ir_annualized, expected_ir_annualized, places=7)

        # Edge case: zero volatility of active returns
        constant_active_returns = pd.Series([0.001, 0.001, 0.001])
        result_zero_vol_ir = calculate_information_ratio(constant_active_returns, annualize=False)
        self.assertTrue(np.isnan(result_zero_vol_ir))

    def test_calculate_sharpe_ratio_se(self):
        """Test calculation of Standard Error of Sharpe Ratio."""
        sr_non_annualized = 0.5
        n_obs = 100

        # Non-annualized SE
        expected_se_non_annualized = np.sqrt((1 + (sr_non_annualized**2) / 2) / n_obs)
        result_se_non_annualized = calculate_sharpe_ratio_se(
            sr_non_annualized,
            n_obs,
            annualize_se=False
        )
        self.assertAlmostEqual(result_se_non_annualized, expected_se_non_annualized, places=7)

        # Annualized SE
        expected_se_annualized = expected_se_non_annualized * self.sqrt_tppy
        result_se_annualized = calculate_sharpe_ratio_se(
            sr_non_annualized,
            n_obs,
            trading_periods_per_year=self.trading_periods_per_year,
            annualize_se=True
        )
        self.assertAlmostEqual(result_se_annualized, expected_se_annualized, places=7)

        # Edge case: num_observations = 0
        result_n_zero = calculate_sharpe_ratio_se(sr_non_annualized, 0)
        self.assertTrue(np.isnan(result_n_zero))

        # Edge case: num_observations < 0
        result_n_neg = calculate_sharpe_ratio_se(sr_non_annualized, -10)
        self.assertTrue(np.isnan(result_n_neg))

    def test_adjust_sharpe_ratio_for_autocorrelation(self):
        """Test adjustment of Sharpe Ratio for autocorrelation."""
        sr = 2.0

        # No autocorrelation
        result_no_autocorr = adjust_sharpe_ratio_for_autocorrelation(sr, 0.0)
        self.assertAlmostEqual(result_no_autocorr, sr, places=7)

        # Positive autocorrelation
        rho_pos = 0.2
        expected_adj_pos = sr * np.sqrt((1 - rho_pos) / (1 + rho_pos))
        result_adj_pos = adjust_sharpe_ratio_for_autocorrelation(sr, rho_pos)
        self.assertAlmostEqual(result_adj_pos, expected_adj_pos, places=7)

        # Negative autocorrelation
        rho_neg = -0.2
        expected_adj_neg = sr * np.sqrt((1 - rho_neg) / (1 + rho_neg))
        result_adj_neg = adjust_sharpe_ratio_for_autocorrelation(sr, rho_neg)
        self.assertAlmostEqual(result_adj_neg, expected_adj_neg, places=7)

        # Edge case: rho = -1.0 (denominator is zero)
        result_rho_minus_one = adjust_sharpe_ratio_for_autocorrelation(sr, -1.0)
        self.assertTrue(np.isnan(result_rho_minus_one))

        # Edge case: rho = 1.0 (numerator is zero, valid)
        result_rho_one = adjust_sharpe_ratio_for_autocorrelation(sr, 1.0)
        self.assertAlmostEqual(result_rho_one, 0.0, places=7)

        # Edge case: rho > 1.0 (term under sqrt is negative)
        result_rho_gt_one = adjust_sharpe_ratio_for_autocorrelation(sr, 1.1)
        self.assertTrue(np.isnan(result_rho_gt_one))

        # Edge case: rho < -1.0 (term under sqrt is negative or 1+rho is negative)
        result_rho_lt_minus_one = adjust_sharpe_ratio_for_autocorrelation(sr, -1.1)
        self.assertTrue(np.isnan(result_rho_lt_minus_one))


if __name__ == '__main__':
    unittest.main()
