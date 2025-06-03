import unittest
import pandas as pd
import numpy as np
from quant_elements_lib.returns_analysis import calculate_hedged_returns, calculate_forward_premium
# from ele_quant.returns_analysis.currency_hedging import calculate_hedged_returns, calculate_forward_premium

class TestCurrencyHedging(unittest.TestCase):

    def setUp(self):
        self.dates = pd.date_range(start='2023-01-01', periods=5, freq='D')

        # Asset returns in local currency
        self.asset_returns_eur = pd.DataFrame({'AssetEUR': [0.01, 0.005, -0.002, 0.015, 0.003]}, index=self.dates)
        self.asset_returns_jpy = pd.DataFrame({'AssetJPY': [0.005, -0.001, 0.003, 0.002, -0.004]}, index=self.dates)
        self.asset_returns_multi = pd.concat([self.asset_returns_eur, self.asset_returns_jpy], axis=1)

        # FX Spot Returns (LC per Base Currency, e.g., EUR/USD, JPY/USD)
        # Positive means LC appreciated against Base.
        self.fx_returns_eur_usd = pd.Series([0.002, -0.001, 0.003, -0.0005, 0.001], index=self.dates, name='EUR')
        self.fx_returns_jpy_usd = pd.Series([-0.001, 0.0005, -0.0015, 0.002, 0.0003], index=self.dates, name='JPY')
        self.fx_returns_df = pd.concat([self.fx_returns_eur_usd, self.fx_returns_jpy_usd], axis=1)

        self.asset_to_currency_map = {'AssetEUR': 'EUR', 'AssetJPY': 'JPY'}

    def test_calculate_hedged_returns_basic_series_fx(self):
        # R_hedged = R_local - HR * R_fx - cost
        # AssetEUR: [0.01, 0.005, -0.002, 0.015, 0.003]
        # EUR/USD:  [0.002, -0.001, 0.003, -0.0005, 0.001]
        # Expected (HR=1, cost=0): [0.01-0.002, 0.005-(-0.001), -0.002-0.003, 0.015-(-0.0005), 0.003-0.001]
        # Expected: [0.008, 0.006, -0.005, 0.0155, 0.002]
        expected = pd.Series([0.008, 0.006, -0.005, 0.0155, 0.002], index=self.dates, name='AssetEUR')

        hedged = calculate_hedged_returns(
            asset_returns_local_currency=self.asset_returns_eur,
            fx_spot_returns_local_per_base=self.fx_returns_eur_usd, # Pass as Series
            hedge_ratio=1.0,
            cost_of_hedge_per_period=0.0
        )
        pd.testing.assert_series_equal(hedged['AssetEUR'], expected, check_dtype=False)

    def test_calculate_hedged_returns_basic_df_fx_single_col(self):
        # Same as above, but fx_spot_returns_local_per_base is a DataFrame with one column
        fx_df_single = self.fx_returns_eur_usd.to_frame()
        expected = pd.Series([0.008, 0.006, -0.005, 0.0155, 0.002], index=self.dates, name='AssetEUR')
        hedged = calculate_hedged_returns(
            asset_returns_local_currency=self.asset_returns_eur,
            fx_spot_returns_local_per_base=fx_df_single,
            hedge_ratio=1.0,
            cost_of_hedge_per_period=0.0
            # No map needed if fx_df_single has 1 col, or if asset_col name matches fx_df_single col name
        )
        pd.testing.assert_series_equal(hedged['AssetEUR'], expected, check_dtype=False)


    def test_calculate_hedged_returns_multi_asset_currency(self):
        hedged = calculate_hedged_returns(
            asset_returns_local_currency=self.asset_returns_multi,
            fx_spot_returns_local_per_base=self.fx_returns_df,
            hedge_ratio=1.0,
            cost_of_hedge_per_period=0.0,
            asset_to_currency_map=self.asset_to_currency_map
        )
        expected_eur = pd.Series([0.008, 0.006, -0.005, 0.0155, 0.002], index=self.dates, name='AssetEUR')
        # AssetJPY: [0.005, -0.001, 0.003, 0.002, -0.004]
        # JPY/USD:  [-0.001, 0.0005, -0.0015, 0.002, 0.0003]
        # Expected (HR=1, cost=0): [0.005-(-0.001), -0.001-0.0005, 0.003-(-0.0015), 0.002-0.002, -0.004-0.0003]
        # Expected: [0.006, -0.0015, 0.0045, 0.0, -0.0043]
        expected_jpy = pd.Series([0.006, -0.0015, 0.0045, 0.0, -0.0043], index=self.dates, name='AssetJPY')

        pd.testing.assert_series_equal(hedged['AssetEUR'], expected_eur, check_dtype=False)
        pd.testing.assert_series_equal(hedged['AssetJPY'], expected_jpy, check_dtype=False)

    def test_calculate_hedged_returns_params(self):
        # Test hedge_ratio and cost
        # R_hedged = R_local - HR * R_fx - cost
        # AssetEUR: 0.01 (period 0)
        # EUR/USD: 0.002 (period 0)
        # HR = 0.5, Cost = 0.0001
        # Expected_AssetEUR_p0 = 0.01 - 0.5 * 0.002 - 0.0001 = 0.01 - 0.001 - 0.0001 = 0.0089
        hedged = calculate_hedged_returns(
            asset_returns_local_currency=self.asset_returns_eur,
            fx_spot_returns_local_per_base=self.fx_returns_eur_usd,
            hedge_ratio=0.5,
            cost_of_hedge_per_period=0.0001
        )
        self.assertAlmostEqual(hedged['AssetEUR'].iloc[0], 0.0089)

    def test_calculate_hedged_returns_alignment(self):
        asset_returns_short = self.asset_returns_eur.iloc[:3]
        fx_returns_long = self.fx_returns_eur_usd.copy() # Has 5 periods

        # Default align is 'left' on asset_returns.index
        hedged = calculate_hedged_returns(asset_returns_short, fx_returns_long)
        self.assertEqual(len(hedged), 3) # Result should match shorter asset_returns index
        self.assertFalse(hedged['AssetEUR'].isnull().any())

        # Case: FX returns missing some dates that asset returns has (should result in NaN for those)
        fx_returns_shorter_or_holey = self.fx_returns_eur_usd.drop(self.fx_returns_eur_usd.index[1]) # Drop 2nd date
        hedged_with_nan = calculate_hedged_returns(self.asset_returns_eur, fx_returns_shorter_or_holey)
        self.assertEqual(len(hedged_with_nan), len(self.asset_returns_eur))
        self.assertTrue(hedged_with_nan['AssetEUR'].isnull().sum() > 0) # Expect some NaNs due to missing fx
        self.assertTrue(hedged_with_nan['AssetEUR'].iloc[1:2].isnull().all())


    def test_calculate_hedged_returns_input_validation(self):
        with self.assertRaises(ValueError): # asset_returns not DataFrame
            calculate_hedged_returns([0.01], self.fx_returns_eur_usd)
        with self.assertRaises(ValueError): # fx_returns not DataFrame/Series
            calculate_hedged_returns(self.asset_returns_eur, [0.01])

        # Ambiguous FX returns (multi-col DF, no map, no direct name match)
        ambiguous_fx_df = pd.DataFrame({'FX1': self.fx_returns_eur_usd, 'FX2': self.fx_returns_jpy_usd})
        asset_df_no_match_name = pd.DataFrame({'SomeAsset': self.asset_returns_eur['AssetEUR']}, index=self.dates)
        with self.assertRaises(ValueError):
            calculate_hedged_returns(asset_df_no_match_name, ambiguous_fx_df)

        # Map provided but currency not in FX df
        bad_map = {'AssetEUR': 'NonExistentFX'}
        with self.assertRaises(ValueError):
            calculate_hedged_returns(self.asset_returns_eur, self.fx_returns_df, asset_to_currency_map=bad_map)


    def test_calculate_forward_premium(self):
        spot = pd.Series([1.1, 1.12, 1.15], index=self.dates[:3])
        forward = pd.Series([1.11, 1.13, 1.14], index=self.dates[:3]) # Misaligned on purpose for test
        # Expected: (1.11-1.1)/1.1 = 0.0090909, (1.13-1.12)/1.12 = 0.00892857, (1.14-1.15)/1.15 = -0.00869565

        premium = calculate_forward_premium(spot, forward)
        self.assertIsInstance(premium, pd.Series)
        self.assertEqual(len(premium), 3)
        self.assertAlmostEqual(premium.iloc[0], (1.11-1.1)/1.1)

        # Test alignment: spot_short, forward_long
        spot_short = spot.iloc[:2]
        premium_aligned = calculate_forward_premium(spot_short, forward) # Should be len 2
        self.assertEqual(len(premium_aligned), 2)

        # Test with NaN from zero spot
        spot_with_zero = pd.Series([1.1, 0.0, 1.15], index=self.dates[:3])
        premium_with_nan = calculate_forward_premium(spot_with_zero, forward)
        self.assertTrue(premium_with_nan.iloc[1:2].isnull().all())

        # Test empty result from no overlap
        spot_no_overlap = pd.Series([1.0], index=[pd.Timestamp('1990-01-01')])
        premium_no_overlap = calculate_forward_premium(spot_no_overlap, forward)
        self.assertIsNone(premium_no_overlap) # Or empty series, depends on impl. Current: None

if __name__ == '__main__':
    unittest.main()
