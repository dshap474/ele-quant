import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from ele_quant.returns_analysis.calculations import (
    calculate_simple_returns,
    calculate_log_returns,
    calculate_dividend_adjusted_returns,
    calculate_excess_returns,
    calculate_compounded_return_series,
    calculate_total_compounded_return,
)
from ele_quant.returns_analysis.price_estimation import (
    estimate_true_price_roll_model,
)
from ele_quant.utils.kalman_filter import KalmanFilter # For Roll Model which uses KF

class TestReturnCalculations(unittest.TestCase):

    def test_calculate_simple_returns(self):
        prices = pd.Series([10.0, 10.5, 10.29, 10.8045]) # P0, P1, P2, P3
        expected_returns = pd.Series([np.nan, 0.05, -0.02, 0.05]) # (P1-P0)/P0, (P2-P1)/P1, ...
        calculated_returns = calculate_simple_returns(prices)
        assert_series_equal(calculated_returns, expected_returns, rtol=1e-5)

    def test_calculate_log_returns_from_prices(self):
        prices = pd.Series([10.0, 10.5, 10.29])
        # log(10.5/10) = log(1.05) approx 0.04879
        # log(10.29/10.5) = log(0.98) approx -0.02020
        expected_returns = pd.Series([np.nan, np.log(1.05), np.log(0.98)])
        calculated_returns = calculate_log_returns(prices)
        assert_series_equal(calculated_returns, expected_returns, rtol=1e-5)

    def test_calculate_log_returns_from_simple_returns(self):
        simple_returns = pd.Series([np.nan, 0.05, -0.02]) # First element could be NaN or valid
        # log(1 + 0.05) = log(1.05)
        # log(1 - 0.02) = log(0.98)
        # The function heuristic might treat this as prices if the first is not NaN and values are far from 0.
        # Let's test with typical small simple returns.
        simple_returns_for_log = pd.Series([0.05, -0.02, 0.01]) # R1, R2, R3
        expected_log_returns = pd.Series([np.log(1.05), np.log(0.98), np.log(1.01)])
        # The function calculate_log_returns has a heuristic.
        # To reliably test calculation from simple returns, we need to ensure it triggers that path.
        # One way is to ensure the mean is close to 0.
        # prices_or_simple_returns.mean() < 1 and prices_or_simple_returns.abs().max() < 2
        calculated_log_returns = calculate_log_returns(simple_returns_for_log)
        assert_series_equal(calculated_log_returns, expected_log_returns, rtol=1e-5)

        # Test with NaN at the beginning, which is common if derived from prices first
        simple_returns_with_nan = pd.Series([np.nan, 0.05, -0.02])
        expected_log_returns_with_nan = pd.Series([np.nan, np.log(1.05), np.log(0.98)])
        calculated_log_returns_with_nan = calculate_log_returns(simple_returns_with_nan)
        assert_series_equal(calculated_log_returns_with_nan, expected_log_returns_with_nan, rtol=1e-5)


    def test_calculate_dividend_adjusted_returns(self):
        prices = pd.Series([10.0, 10.5, 10.8], index=pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']))
        dividends = pd.Series([0.2], index=pd.to_datetime(['2023-01-02'])) # Dividend D_t for period t-1 to t

        # Align dividends to prices series, fill NaNs with 0
        dividends_aligned = dividends.reindex(prices.index).fillna(0)

        # R1 = (P1 - P0 + D1) / P0 = (10.5 - 10.0 + 0.2) / 10.0 = 0.7 / 10.0 = 0.07
        # R2 = (P2 - P1 + D2) / P1 = (10.8 - 10.5 + 0.0) / 10.5 = 0.3 / 10.5 approx 0.0285714
        expected_returns = pd.Series([np.nan, 0.07, 0.3 / 10.5])
        # The function expects dividends series to be fully aligned.
        calculated_returns = calculate_dividend_adjusted_returns(prices, dividends_aligned)
        assert_series_equal(calculated_returns, expected_returns, rtol=1e-5)

    def test_calculate_excess_returns_scalar_rfr(self):
        asset_returns = pd.Series([0.01, 0.02, -0.005])
        rfr = 0.005
        expected_excess = pd.Series([0.005, 0.015, -0.010])
        calculated_excess = calculate_excess_returns(asset_returns, rfr)
        assert_series_equal(calculated_excess, expected_excess, rtol=1e-5)

    def test_calculate_excess_returns_series_rfr(self):
        asset_returns = pd.Series([0.01, 0.02, -0.005], index=[0,1,2])
        rfr_series = pd.Series([0.004, 0.005, 0.006], index=[0,1,2])
        expected_excess = pd.Series([0.006, 0.015, -0.011])
        calculated_excess = calculate_excess_returns(asset_returns, rfr_series)
        assert_series_equal(calculated_excess, expected_excess, rtol=1e-5)

        # Test with misaligned index (should align and use NaN where no RFR)
        rfr_misaligned = pd.Series([0.004, 0.005], index=[0,1]) # Missing RFR for last asset return
        expected_misaligned = pd.Series([0.006, 0.015, np.nan])
        calculated_misaligned = calculate_excess_returns(asset_returns, rfr_misaligned)
        assert_series_equal(calculated_misaligned, expected_misaligned, rtol=1e-5)


    def test_calculate_compounded_return_series(self):
        # (1+R_t).cumprod()
        simple_returns = pd.Series([np.nan, 0.1, -0.05, 0.02]) # NaN, R1, R2, R3
        # Val0 = 1 (implicit)
        # Val1 = Val0 * (1+R1) = 1 * 1.1 = 1.1
        # Val2 = Val1 * (1+R2) = 1.1 * 0.95 = 1.045
        # Val3 = Val2 * (1+R3) = 1.045 * 1.02 = 1.0659
        expected_series = pd.Series([np.nan, 1.1, 1.045, 1.0659])
        calculated_series = calculate_compounded_return_series(simple_returns)
        assert_series_equal(calculated_series, expected_series, rtol=1e-5)

    def test_calculate_total_compounded_return(self):
        simple_returns = pd.Series([np.nan, 0.1, -0.05, 0.02]) # R1, R2, R3
        # (1+0.1)*(1-0.05)*(1+0.02) - 1
        # 1.1 * 0.95 * 1.02 - 1 = 1.0659 - 1 = 0.0659
        expected_total_return = 0.0659
        calculated_total_return = calculate_total_compounded_return(simple_returns)
        self.assertAlmostEqual(calculated_total_return, expected_total_return, places=5)


class TestRollModel(unittest.TestCase):

    def test_estimate_true_price_roll_model_basic(self):
        observed_prices = pd.Series([10.0, 10.1, 9.9, 10.0, 10.2, 10.15, 10.25])
        sigma_m_sq = 0.01**2  # Variance of true price changes
        sigma_eta_sq = 0.05**2 # Variance of bid-ask bounce (measurement noise)

        estimated_true_prices = estimate_true_price_roll_model(
            observed_prices, sigma_m_sq, sigma_eta_sq
        )

        self.assertEqual(len(estimated_true_prices), len(observed_prices))
        self.assertTrue(observed_prices.index.equals(estimated_true_prices.index))

        # Estimated true prices should be "smoother" than observed prices.
        # This can be subjective, but variance of estimates might be smaller
        # if true price is actually smoother than observations suggest.
        # Or, more simply, check they are not wildly off.
        # For this specific model (random walk true price), estimated true price will still fluctuate.
        # A key check: estimates should generally lie between the extremes of observed prices,
        # and react to them.
        self.assertTrue(estimated_true_prices.mean() > 9.0) # Sanity check lower bound
        self.assertTrue(estimated_true_prices.mean() < 11.0) # Sanity check upper bound

        # First estimated price is the initial guess (first observed price)
        self.assertAlmostEqual(estimated_true_prices.iloc[0], observed_prices.iloc[0])

        # Subsequent prices should be influenced by Kalman filter updates
        # e.g. estimated_true_prices.iloc[1] should be between observed_prices.iloc[0] (as prediction)
        # and observed_prices.iloc[1] (as observation)
        # This depends on Kalman gain, which depends on sigmas.
        # For a simple check, ensure values are not NaN (unless input was all NaN)
        self.assertFalse(estimated_true_prices.isna().any())


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
