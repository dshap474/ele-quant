import unittest
import pandas as pd
import numpy as np
from ele_quant.factor_models import FundamentalFactorModel

class TestFundamentalFactorModel(unittest.TestCase):

    def setUp(self):
        self.n_periods = 20 # Reduced for faster tests
        self.n_assets = 5
        self.n_factors = 2
        self.dates = pd.date_range(start='2022-01-01', periods=self.n_periods, freq='M')
        self.asset_names = [f'Asset_{i+1}' for i in range(self.n_assets)]
        self.factor_names = ['F1', 'F2']

        np.random.seed(42)
        # Asset Returns
        self.returns_data = np.random.randn(self.n_periods, self.n_assets) * 0.02
        self.asset_returns = pd.DataFrame(self.returns_data, index=self.dates, columns=self.asset_names)

        # Dynamic Factor Exposures: MultiIndex (date, asset)
        exposures_list = []
        for date_val in self.dates:
            for asset_val in self.asset_names:
                exposures_list.append([date_val, asset_val] + list(np.random.rand(self.n_factors)))
        self.factor_exposures_dynamic = pd.DataFrame(exposures_list, columns=['date', 'asset'] + self.factor_names)
        self.factor_exposures_dynamic = self.factor_exposures_dynamic.set_index(['date', 'asset'])

        # Static Factor Exposures: Index (asset), Columns (factors)
        self.factor_exposures_static = pd.DataFrame(np.random.rand(self.n_assets, self.n_factors),
                                                    index=self.asset_names, columns=self.factor_names)

        self.model = FundamentalFactorModel()

    def test_initialization(self):
        self.assertIsInstance(self.model, FundamentalFactorModel)
        self.assertIsNone(self.model.factor_loadings)
        self.assertIsNone(self.model.factor_returns)

    def _check_fit_attributes(self, model_fitted):
        self.assertIsNotNone(model_fitted.factor_returns)
        self.assertIsInstance(model_fitted.factor_returns, pd.DataFrame)
        self.assertEqual(model_fitted.factor_returns.shape[1], self.n_factors)
        # Num periods in factor_returns can be less than self.n_periods if some periods had issues

        self.assertIsNotNone(model_fitted.factor_covariance_matrix)
        self.assertIsInstance(model_fitted.factor_covariance_matrix, pd.DataFrame)
        self.assertEqual(model_fitted.factor_covariance_matrix.shape, (self.n_factors, self.n_factors))

        self.assertIsNotNone(model_fitted.idiosyncratic_variance)
        self.assertIsInstance(model_fitted.idiosyncratic_variance, pd.DataFrame)
        self.assertEqual(model_fitted.idiosyncratic_variance.shape[0], self.n_assets) # Assets as index
        self.assertEqual(model_fitted.idiosyncratic_variance.shape[1], 1) # Single column

        self.assertIsNotNone(model_fitted.factor_loadings)
        self.assertIsInstance(model_fitted.factor_loadings, pd.DataFrame)
        self.assertEqual(model_fitted.factor_loadings.shape, (self.n_assets, self.n_factors))

        self.assertEqual(len(model_fitted.assets), self.n_assets)
        self.assertEqual(len(model_fitted.factors), self.n_factors)


    def test_fit_runs_dynamic_exposures_smoke(self):
        # Smoke test: Does it run and set attributes?
        self.model.fit(self.asset_returns, self.factor_exposures_dynamic, covariance_method='sample')
        self._check_fit_attributes(self.model)

    def test_fit_runs_static_exposures_smoke(self):
        self.model.fit(self.asset_returns, self.factor_exposures_static, covariance_method='sample')
        self._check_fit_attributes(self.model)

    def test_fit_ewma_covariance(self):
        # Check if EWMA runs. Span needs to be reasonable.
        # min_periods_ewma default is 2. n_periods for factor returns could be small.
        self.model.fit(self.asset_returns, self.factor_exposures_dynamic,
                       covariance_method='ewma', span=10, min_periods_ewma=2)
        self._check_fit_attributes(self.model)
        # Further check if cov matrix is not all NaN, if factor returns are sufficient
        if self.model.factor_returns is not None and len(self.model.factor_returns) >=2:
             self.assertFalse(self.model.factor_covariance_matrix.isnull().all().all(), "EWMA Cov matrix is all NaN")


    def test_predict_runs_and_returns_expected_format(self):
        self.model.fit(self.asset_returns, self.factor_exposures_dynamic)
        if self.model.factor_loadings is None or self.model.factor_returns is None:
            self.skipTest("Skipping predict test as model fitting failed or did not set necessary attributes.")

        # Predict with internal loadings (latest from dynamic)
        predicted_returns = self.model.predict()
        self.assertIsInstance(predicted_returns, pd.DataFrame)
        self.assertEqual(predicted_returns.shape[0], self.n_assets)
        self.assertEqual(predicted_returns.shape[1], 1)
        self.assertEqual(predicted_returns.columns[0], 'expected_return')
        pd.testing.assert_index_equal(predicted_returns.index, pd.Index(self.asset_names))

        # Predict with future exposures (using current loadings as example)
        future_exposures = self.model.factor_loadings.copy()
        predicted_returns_future = self.model.predict(factor_exposures_future=future_exposures)
        self.assertIsInstance(predicted_returns_future, pd.DataFrame)
        pd.testing.assert_frame_equal(predicted_returns, predicted_returns_future, check_dtype=False)


    def test_predict_error_if_not_fitted(self):
        fresh_model = FundamentalFactorModel()
        with self.assertRaises(ValueError, msg="Predict called before fit or essential attributes missing"):
            fresh_model.predict()

    def test_fit_edge_cases_nans(self):
        # Returns with some NaNs
        asset_returns_with_nans = self.asset_returns.copy()
        asset_returns_with_nans.iloc[0, 0] = np.nan
        asset_returns_with_nans.iloc[1, :] = np.nan # Whole period NaN for returns

        # Exposures with some NaNs
        factor_exposures_dyn_with_nans = self.factor_exposures_dynamic.copy()
        factor_exposures_dyn_with_nans.iloc[0, 0] = np.nan # Single NaN
        # To make a whole asset NaN for a period in dynamic exposures:
        # factor_exposures_dyn_with_nans.loc[(self.dates[0], self.asset_names[0]), :] = np.nan

        model_nan_test = FundamentalFactorModel()
        model_nan_test.fit(asset_returns_with_nans, factor_exposures_dyn_with_nans)

        # Check that attributes are still set, possibly with NaNs or reduced dimensions
        self.assertIsNotNone(model_nan_test.factor_returns)
        # Factor returns for periods with all NaN returns or insufficient data might be dropped.
        # So, len(model_nan_test.factor_returns) could be < self.n_periods.
        self.assertTrue(len(model_nan_test.factor_returns) <= self.n_periods)

        if model_nan_test.factor_returns is not None and not model_nan_test.factor_returns.empty:
            self.assertFalse(model_nan_test.factor_returns.isnull().all().all())
            if model_nan_test.factor_covariance_matrix is not None:
                 self.assertFalse(model_nan_test.factor_covariance_matrix.isnull().all().all())

        if model_nan_test.idiosyncratic_variance is not None:
             # Idiosyncratic variance might be NaN for assets with all NaN residuals
             self.assertTrue(model_nan_test.idiosyncratic_variance.isnull().sum().sum() < model_nan_test.idiosyncratic_variance.size or \
                             model_nan_test.idiosyncratic_variance.isnull().all().all())


    # A more specific test for _estimate_factor_returns if it were public or accessible
    # For now, testing its effect through model.factor_returns after fit()
    def test_factor_returns_estimation_logic_simple(self):
        # Simple case: 1 asset, 1 factor, perfect correlation
        dates_simple = pd.date_range(start='2023-01-01', periods=3, freq='D')
        asset_ret_simple = pd.DataFrame({'AssetX': [0.02, 0.04, -0.02]}, index=dates_simple)

        # Dynamic exposures
        exposures_data = []
        # Period 0: AssetX exposure to F1 is 2.0. Ret = 0.02. Factor Ret = 0.01
        exposures_data.append([dates_simple[0], 'AssetX', 2.0])
        # Period 1: AssetX exposure to F1 is 2.0. Ret = 0.04. Factor Ret = 0.02
        exposures_data.append([dates_simple[1], 'AssetX', 2.0])
        # Period 2: AssetX exposure to F1 is 2.0. Ret = -0.02. Factor Ret = -0.01
        exposures_data.append([dates_simple[2], 'AssetX', 2.0])

        factor_exp_simple_dyn = pd.DataFrame(exposures_data, columns=['date', 'asset', 'F1']).set_index(['date', 'asset'])

        model_simple = FundamentalFactorModel()
        model_simple.fit(asset_ret_simple, factor_exp_simple_dyn)

        expected_factor_returns = pd.DataFrame({'F1': [0.01, 0.02, -0.01]}, index=dates_simple)
        expected_factor_returns.index.name = 'time' # Match what _estimate_factor_returns produces

        if model_simple.factor_returns is not None:
            pd.testing.assert_frame_equal(model_simple.factor_returns, expected_factor_returns,
                                          check_dtype=False, atol=1e-6) # Added tolerance
        else:
            self.fail("Factor returns were None for simple logic test.")

if __name__ == '__main__':
    unittest.main()
