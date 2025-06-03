import unittest
import numpy as np
import pandas as pd
from quant_elements_lib.factor_models import StatisticalFactorModel

class TestStatisticalFactorModelPCA(unittest.TestCase):
    def setUp(self):
        # Setup synthetic data: 2 factors, 5 assets, 100 observations
        self.num_obs = 100
        self.num_assets = 5
        self.num_true_factors = 2
        np.random.seed(42)

        # True factor returns (T x K_true)
        self.true_factor_returns = pd.DataFrame(
            np.random.randn(self.num_obs, self.num_true_factors),
            index=pd.date_range(start="2020-01-01", periods=self.num_obs, freq='B'),
            columns=[f'TrueFactor{i+1}' for i in range(self.num_true_factors)]
        )
        # True loadings (N x K_true)
        self.true_B_loadings = pd.DataFrame(
            np.random.rand(self.num_assets, self.num_true_factors) * 2 - 1,
            index=[f'Asset{i+1}' for i in range(self.num_assets)],
            columns=[f'TrueFactor{i+1}' for i in range(self.num_true_factors)]
        )
        # Idiosyncratic returns (T x N)
        idio_returns = pd.DataFrame(
            np.random.randn(self.num_obs, self.num_assets) * 0.005, # Low idio variance
            index=self.true_factor_returns.index,
            columns=self.true_B_loadings.index
        )
        # Total returns R = F B^T + E (T x N = T x K_true @ K_true x N + T x N)
        self.returns_data = pd.DataFrame(
            self.true_factor_returns.values @ self.true_B_loadings.T.values + idio_returns.values,
            index=self.true_factor_returns.index,
            columns=self.true_B_loadings.index
        )
        self.asset_universe = self.returns_data.columns.tolist()

    def test_fit_pca(self):
        num_factors_to_extract = self.num_true_factors
        model = StatisticalFactorModel(
            asset_universe=self.asset_universe,
            num_factors_to_extract=num_factors_to_extract
        )
        model.fit(self.returns_data, estimation_method='PCA')

        self.assertIsNotNone(model.B_loadings)
        self.assertEqual(model.B_loadings.shape, (self.num_assets, num_factors_to_extract))
        self.assertListEqual(model.B_loadings.columns.tolist(), [f'Factor_{i+1}' for i in range(num_factors_to_extract)])

        self.assertIsNotNone(model.factor_returns)
        self.assertEqual(model.factor_returns.shape, (self.num_obs, num_factors_to_extract))
        self.assertListEqual(model.factor_returns.columns.tolist(), [f'Factor_{i+1}' for i in range(num_factors_to_extract)])

        self.assertIsNotNone(model.factor_covariance)
        self.assertEqual(model.factor_covariance.shape, (num_factors_to_extract, num_factors_to_extract))
        self.assertListEqual(model.factor_covariance.columns.tolist(), [f'Factor_{i+1}' for i in range(num_factors_to_extract)])
        self.assertListEqual(model.factor_covariance.index.tolist(), [f'Factor_{i+1}' for i in range(num_factors_to_extract)])

        # Check factor covariance is diagonal and positive
        self.assertTrue(np.all(np.diag(model.factor_covariance.values) >= 0))
        # Check off-diagonal are zero (or very small due to precision) for PCA factors from SVD
        off_diag_sum_abs = np.sum(np.abs(model.factor_covariance.values - np.diag(np.diag(model.factor_covariance.values))))
        self.assertAlmostEqual(off_diag_sum_abs, 0, places=5)

        self.assertIsNotNone(model.idiosyncratic_covariance)
        self.assertEqual(model.idiosyncratic_covariance.shape, (self.num_assets,))
        self.assertTrue(np.all(model.idiosyncratic_covariance.values >= 0))

        self.assertIsNotNone(model.alpha)
        self.assertEqual(model.alpha.shape, (self.num_assets,))
        # For PCA on demeaned data, alpha should be very close to zero.
        self.assertTrue(np.allclose(model.alpha.values, 0, atol=1e-5))

        if num_factors_to_extract > 1:
            self.assertGreaterEqual(model.factor_covariance.values[0,0], model.factor_covariance.values[1,1])

class TestStatisticalFactorModelPPCA(unittest.TestCase):
    def setUp(self):
        self.num_obs = 100
        self.num_assets = 5
        self.num_true_factors = 2 # K
        np.random.seed(42)

        self.true_factor_returns = pd.DataFrame(
            np.random.randn(self.num_obs, self.num_true_factors),
            index=pd.date_range(start="2020-01-01", periods=self.num_obs, freq='B'),
            columns=[f'TrueFactor{i+1}' for i in range(self.num_true_factors)]
        )
        self.true_B_loadings = pd.DataFrame(
            np.random.rand(self.num_assets, self.num_true_factors) * 2 - 1,
            index=[f'Asset{i+1}' for i in range(self.num_assets)],
            columns=[f'TrueFactor{i+1}' for i in range(self.num_true_factors)]
        )
        self.true_sigma_sq = 0.01**2 # scalar noise variance
        idio_returns = pd.DataFrame(
            np.random.randn(self.num_obs, self.num_assets) * np.sqrt(self.true_sigma_sq),
            index=self.true_factor_returns.index,
            columns=self.true_B_loadings.index
        )
        self.true_alpha = pd.Series(np.random.rand(self.num_assets) * 0.001, index=self.true_B_loadings.index)

        self.returns_data = pd.DataFrame(
            self.true_alpha.values + self.true_factor_returns.values @ self.true_B_loadings.T.values + idio_returns.values,
            index=self.true_factor_returns.index,
            columns=self.true_B_loadings.index
        )
        self.asset_universe = self.returns_data.columns.tolist()

    def test_fit_ppca(self):
        num_factors_to_extract = self.num_true_factors
        model = StatisticalFactorModel(
            asset_universe=self.asset_universe,
            num_factors_to_extract=num_factors_to_extract
        )
        model.fit(self.returns_data, estimation_method='PPCA')

        self.assertIsNotNone(model.B_loadings)
        self.assertEqual(model.B_loadings.shape, (self.num_assets, num_factors_to_extract))
        self.assertListEqual(model.B_loadings.columns.tolist(), [f'Factor_{i+1}' for i in range(num_factors_to_extract)])

        self.assertIsNotNone(model.factor_returns)
        self.assertEqual(model.factor_returns.shape, (self.num_obs, num_factors_to_extract))
        self.assertListEqual(model.factor_returns.columns.tolist(), [f'Factor_{i+1}' for i in range(num_factors_to_extract)])

        self.assertIsNotNone(model.factor_covariance)
        self.assertEqual(model.factor_covariance.shape, (num_factors_to_extract, num_factors_to_extract))
        self.assertListEqual(model.factor_covariance.columns.tolist(), [f'Factor_{i+1}' for i in range(num_factors_to_extract)])
        self.assertListEqual(model.factor_covariance.index.tolist(), [f'Factor_{i+1}' for i in range(num_factors_to_extract)])
        self.assertTrue(np.allclose(model.factor_covariance.values, np.eye(num_factors_to_extract), atol=1e-5))

        self.assertIsNotNone(model.idiosyncratic_covariance) # This is a pd.Series
        self.assertEqual(model.idiosyncratic_covariance.shape, (self.num_assets,))
        self.assertTrue(np.all(model.idiosyncratic_covariance.values > 0))
        # For PPCA, all idiosyncratic variances should be equal to sigma_sq_hat
        self.assertTrue(np.allclose(model.idiosyncratic_covariance.values, model.idiosyncratic_covariance.values[0], atol=1e-5))

        # Check if estimated sigma_sq is somewhat close to true_sigma_sq (can be very rough)
        estimated_sigma_sq = model.idiosyncratic_covariance.mean()
        # print(f"True sigma_sq: {self.true_sigma_sq}, Estimated sigma_sq for PPCA: {estimated_sigma_sq}")
        # This delta is very wide because the estimation of sigma_sq can be noisy,
        # especially with small N and T or if model is misspecified.
        self.assertAlmostEqual(estimated_sigma_sq, self.true_sigma_sq, delta=self.true_sigma_sq * 20 + 1e-4)


        self.assertIsNotNone(model.alpha)
        self.assertEqual(model.alpha.shape, (self.num_assets,))
        # Check if estimated alpha is close to true_alpha (mean of original returns)
        self.assertTrue(np.allclose(model.alpha.values, self.returns_data.mean(axis=0).values, atol=1e-5))

if __name__ == '__main__':
    unittest.main()
