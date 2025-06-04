import unittest
import pandas as pd
import numpy as np
from quant_elements_lib.factor_models import StatisticalFactorModel

class TestStatisticalFactorModel(unittest.TestCase):

    def setUp(self):
        self.N_assets = 10
        self.N_factors_true = 2
        self.T_obs = 100
        self.asset_names = [f'Asset_{i+1}' for i in range(self.N_assets)]
        self.factor_names_true = [f'Factor_{i+1}' for i in range(self.N_factors_true)]
        # Generate factor names for the model based on num_factors_to_extract
        # This is what the model will use for its column/index names
        self.model_factor_names = [f'Factor_{i+1}' for i in range(self.N_factors_true)]


        self.dates = pd.date_range(start='2020-01-01', periods=self.T_obs, freq='B')

        # Generate synthetic data
        np.random.seed(42) # for reproducibility
        # True factor loadings (N x K_true)
        # Using asset_names for index and factor_names_true for columns
        self.B_true = pd.DataFrame(np.random.rand(self.N_assets, self.N_factors_true),
                                   index=self.asset_names, columns=self.factor_names_true)
        # True factor returns (T x K_true)
        # Using dates for index and factor_names_true for columns
        self.F_true = pd.DataFrame(np.random.randn(self.T_obs, self.N_factors_true),
                                   index=self.dates, columns=self.factor_names_true)
        # Noise (T x N)
        E_noise = pd.DataFrame(np.random.randn(self.T_obs, self.N_assets) * 0.1,
                               index=self.dates, columns=self.asset_names)

        # Synthetic asset returns: R = F_true @ B_true.T + E_noise
        self.true_alpha_values = np.random.rand(self.N_assets) * 0.0001 # Small random alpha values
        self.true_alpha_series = pd.Series(self.true_alpha_values, index=self.asset_names)

        self.synthetic_returns_df = pd.DataFrame(
            self.F_true.values @ self.B_true.values.T + E_noise.values,
            index=self.dates,
            columns=self.asset_names
        )
        # Add the true alpha to the returns generated from factors and noise
        self.synthetic_returns_df = self.synthetic_returns_df + self.true_alpha_series

    def test_fit_pca_synthetic_data(self):
        model = StatisticalFactorModel(
            asset_universe=self.asset_names,
            num_factors_to_extract=self.N_factors_true # This will generate factor_names like Factor_1, Factor_2
        )
        model.fit(self.synthetic_returns_df, estimation_method='PCA')

        self.assertTrue(model.is_fitted)
        self.assertIsNotNone(model.B_loadings)
        self.assertEqual(model.B_loadings.shape, (self.N_assets, self.N_factors_true))
        self.assertListEqual(list(model.B_loadings.columns), self.model_factor_names)

        self.assertIsNotNone(model.factor_returns)
        self.assertEqual(model.factor_returns.shape, (self.T_obs, self.N_factors_true))
        self.assertListEqual(list(model.factor_returns.columns), self.model_factor_names)

        self.assertIsNotNone(model.factor_covariance)
        self.assertEqual(model.factor_covariance.shape, (self.N_factors_true, self.N_factors_true))
        self.assertListEqual(list(model.factor_covariance.columns), self.model_factor_names)
        self.assertListEqual(list(model.factor_covariance.index), self.model_factor_names)

        diag_values = np.diag(model.factor_covariance.values)
        off_diag_values = model.factor_covariance.values - np.diag(diag_values)
        self.assertAlmostEqual(np.sum(np.abs(off_diag_values)), 0.0, places=5,
                               msg="Factor covariance should be diagonal for PCA")

        self.assertIsNotNone(model.idiosyncratic_covariance)
        self.assertEqual(len(model.idiosyncratic_covariance), self.N_assets)
        self.assertTrue(all(model.idiosyncratic_covariance.values >= 0))

        self.assertIsNotNone(model.alpha)
        self.assertEqual(len(model.alpha), self.N_assets)
        # For PCA, model.alpha is the mean of the input returns_data
        pd.testing.assert_series_equal(model.alpha, self.synthetic_returns_df.mean(axis=0),
                                       check_dtype=False, atol=1e-5) # Check with atol due to potential float precision

    def test_fit_ppca_synthetic_data(self):
        num_factors_ppca = self.N_factors_true
        if num_factors_ppca >= self.N_assets:
             num_factors_ppca = self.N_assets - 1

        if num_factors_ppca <=0:
            self.skipTest(f"Not enough assets ({self.N_assets}) for PPCA with num_factors < N_assets (K_true={self.N_factors_true})")

        ppca_model_factor_names = [f'Factor_{i+1}' for i in range(num_factors_ppca)]
        model = StatisticalFactorModel(
            asset_universe=self.asset_names,
            num_factors_to_extract=num_factors_ppca
        )
        model.fit(self.synthetic_returns_df, estimation_method='PPCA')

        self.assertTrue(model.is_fitted)
        self.assertIsNotNone(model.B_loadings)
        self.assertEqual(model.B_loadings.shape, (self.N_assets, num_factors_ppca))
        self.assertListEqual(list(model.B_loadings.columns), ppca_model_factor_names)

        self.assertIsNotNone(model.factor_returns)
        self.assertEqual(model.factor_returns.shape, (self.T_obs, num_factors_ppca))
        self.assertListEqual(list(model.factor_returns.columns), ppca_model_factor_names)

        self.assertIsNotNone(model.factor_covariance)
        self.assertEqual(model.factor_covariance.shape, (num_factors_ppca, num_factors_ppca))
        self.assertListEqual(list(model.factor_covariance.columns), ppca_model_factor_names)
        self.assertListEqual(list(model.factor_covariance.index), ppca_model_factor_names)

        expected_factor_cov_df = pd.DataFrame(np.eye(num_factors_ppca),
                                               index=ppca_model_factor_names,
                                               columns=ppca_model_factor_names)
        pd.testing.assert_frame_equal(model.factor_covariance, expected_factor_cov_df,
                                      check_dtype=False, atol=1e-5)

        self.assertIsNotNone(model.idiosyncratic_covariance)
        self.assertEqual(len(model.idiosyncratic_covariance), self.N_assets)
        self.assertTrue(all(model.idiosyncratic_covariance.values >= 0))
        if len(model.idiosyncratic_covariance) > 0: # Check that all idio variances are similar (sigma_sq_hat)
             self.assertAlmostEqual(model.idiosyncratic_covariance.std(), 0.0, places=5,
                                    msg="Idio variances should be almost identical for PPCA")

        self.assertIsNotNone(model.alpha)
        self.assertEqual(len(model.alpha), self.N_assets)
        pd.testing.assert_series_equal(model.alpha, self.synthetic_returns_df.mean(axis=0),
                                       check_dtype=False, atol=1e-5)

if __name__ == '__main__':
    unittest.main()
