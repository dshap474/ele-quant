import unittest
import pandas as pd
import numpy as np
from ele_quant.utils import winsorize_series # Assuming winsorize_series is in utils.preprocessing
                                          # and utils/__init__.py exports it.
                                          # Actual import path might be ele_quant.utils.preprocessing.winsorize_series
                                          # For now, using ele_quant.utils.winsorize_series based on previous __init__ update.

class TestWinsorization(unittest.TestCase):

    def setUp(self):
        self.raw_data = pd.Series([np.nan, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 100.0])
        self.all_nan_data = pd.Series([np.nan, np.nan, np.nan])
        self.all_same_data = pd.Series([5.0, 5.0, 5.0, 5.0])

    def test_winsorize_quantile(self):
        # Test basic quantile winsorization
        # For self.raw_data (excluding NaN): [1,2,3,4,5,6,7,8,9,10,100], count = 11
        # 10%ile: 11*0.1 = 1.1th element => 1.0 (after sorting [1,2...10,100])
        # 90%ile: 11*0.9 = 9.9th element => 10.0
        # The actual quantiles might depend on pandas interpolation method.
        # Let's use lower=0.1, upper=0.9 (which means 0.1 and 0.9 for .quantile())
        # Sorted non-NaN: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 100.0]
        # Lower bound: 10% of 11 values. pd.quantile(0.1) = 2.0
        # Upper bound: 90% of 11 values. pd.quantile(0.9) = 9.0 + (10-9)*0 = 9 + 1 = 10
        # Pandas quantile(0.1) on [1..10, 100] is 2.0. Quantile(0.9) is 10.0.

        winsorized = winsorize_series(self.raw_data, method='quantile', lower_quantile=0.1, upper_quantile=0.9)
        expected_non_nan = self.raw_data.dropna().quantile([0.1, 0.9])
        lower_b = expected_non_nan.iloc[0] # Should be 2.0
        upper_b = expected_non_nan.iloc[1] # Should be 10.0

        expected_series = self.raw_data.copy()
        expected_series[expected_series < lower_b] = lower_b
        expected_series[expected_series > upper_b] = upper_b

        pd.testing.assert_series_equal(winsorized, expected_series, check_dtype=False)
        self.assertTrue(winsorized.isna().sum() == self.raw_data.isna().sum(), "NaNs not preserved")

    def test_winsorize_quantile_edge_cases(self):
        # All NaNs
        winsorized_all_nan = winsorize_series(self.all_nan_data, method='quantile', lower_quantile=0.1, upper_quantile=0.9)
        pd.testing.assert_series_equal(self.all_nan_data, winsorized_all_nan, check_dtype=False)

        # All same values (should not change)
        winsorized_all_same = winsorize_series(self.all_same_data, method='quantile', lower_quantile=0.1, upper_quantile=0.9)
        pd.testing.assert_series_equal(self.all_same_data, winsorized_all_same, check_dtype=False)

    def test_winsorize_z_score_robust(self):
        # Test robust z-score winsorization
        # Data: [NaN, 1,2,3,4,5,6,7,8,9,10,100]
        # Non-NaN: [1,2,3,4,5,6,7,8,9,10,100]
        # Median = 6.0
        # MAD = median(|x_i - median(X)|). Deviations: [5,4,3,2,1,0,1,2,3,4,94]. Sorted: [0,1,1,2,2,3,3,4,4,5,94]. Median = 3.0
        # Lower bound = 6.0 - 2.0 * 3.0 / 0.6745 = 6.0 - 8.895 = -2.895
        # Upper bound = 6.0 + 2.0 * 3.0 / 0.6745 = 6.0 + 8.895 = 14.895
        # So, 100.0 should be capped to approx 14.895. 1.0 should not be capped.

        winsorized = winsorize_series(self.raw_data, method='z_score_robust', z_threshold=2.0)

        series_cleaned = self.raw_data.dropna()
        median = series_cleaned.median()
        mad = (series_cleaned - median).abs().median()
        const = 0.6745 # For normally distributed data, MAD*1.4826 = SD. So SD_est = MAD/0.6745

        if mad == 0: # If MAD is 0, z-scores are inf or nan. Handle this case.
            lower_bound = median
            upper_bound = median
        else:
            lower_bound = median - 2.0 * mad / const
            upper_bound = median + 2.0 * mad / const

        expected_series = self.raw_data.copy()
        expected_series[expected_series < lower_bound] = lower_bound
        expected_series[expected_series > upper_bound] = upper_bound

        pd.testing.assert_series_equal(winsorized, expected_series, check_dtype=False)
        self.assertTrue(winsorized.isna().sum() == self.raw_data.isna().sum(), "NaNs not preserved in z-score")

    def test_winsorize_z_score_robust_mad_zero(self):
        # Test MAD is zero case (all same values after NaNs)
        winsorized_mad_zero = winsorize_series(self.all_same_data, method='z_score_robust', z_threshold=2.0)
        # If MAD is 0, it should not change the data as per current implementation (no clipping occurs as bounds become median)
        pd.testing.assert_series_equal(self.all_same_data, winsorized_mad_zero, check_dtype=False)

    def test_winsorize_input_validation(self):
        with self.assertRaises(ValueError, msg="Invalid method"):
            winsorize_series(self.raw_data, method='invalid_method')
        with self.assertRaises(ValueError, msg="Quantile out of range"):
            winsorize_series(self.raw_data, method='quantile', lower_quantile=-0.1)
        with self.assertRaises(ValueError, msg="Quantile out of range"):
            winsorize_series(self.raw_data, method='quantile', upper_quantile=1.1)
        with self.assertRaises(ValueError, msg="Lower > Upper Quantile"):
            winsorize_series(self.raw_data, method='quantile', lower_quantile=0.6, upper_quantile=0.4)
        with self.assertRaises(ValueError, msg="Z-threshold non-positive"):
            winsorize_series(self.raw_data, method='z_score_robust', z_threshold=0)
        with self.assertRaises(TypeError, msg="Input not a Series"):
            winsorize_series([1,2,3,100])

if __name__ == '__main__':
    unittest.main()
