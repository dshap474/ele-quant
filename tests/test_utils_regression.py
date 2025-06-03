import unittest
import pandas as pd
import numpy as np
from pandas.testing import assert_series_equal, assert_frame_equal
from numpy.testing import assert_array_almost_equal

try:
    from quant_elements_lib.utils.regression import ordinary_least_squares, weighted_least_squares
except ImportError:
    import sys
    sys.path.insert(0, '..')
    from quant_elements_lib.utils.regression import ordinary_least_squares, weighted_least_squares

class TestRegressionUtils(unittest.TestCase):

    def setUp(self):
        # Synthetic data for OLS: Y = intercept + c1*X1 + c2*X2 + noise
        self.N = 100
        np.random.seed(42) # for reproducibility

        self.X_df = pd.DataFrame({
            'X1': np.random.rand(self.N) * 10,
            'X2': np.random.rand(self.N) * 5
        })

        self.intercept_true = 2.0
        self.c1_true = 3.5
        self.c2_true = -1.5

        self.noise_std = 0.5
        self.noise = np.random.normal(0, self.noise_std, self.N)

        self.Y_series = pd.Series(
            self.intercept_true + self.c1_true * self.X_df['X1'] + self.c2_true * self.X_df['X2'] + self.noise,
            name="Y"
        )

        # For WLS
        self.weights = pd.Series(np.random.rand(self.N) + 0.1, index=self.Y_series.index) # Ensure positive weights
        # Make weights heterogeneous: some observations are much more important
        self.weights.iloc[::10] = 10.0
        self.weights.iloc[5::10] = 0.01


    def test_ordinary_least_squares_with_intercept(self):
        beta_hat, residuals, r_squared = ordinary_least_squares(self.Y_series, self.X_df, add_intercept=True)

        self.assertIn('intercept', beta_hat.index)
        self.assertEqual(len(beta_hat), 3) # X1, X2, intercept
        self.assertEqual(len(residuals), self.N)

        # Check coefficients (allow some deviation due to noise)
        assert_array_almost_equal([beta_hat['intercept'], beta_hat['X1'], beta_hat['X2']],
                                  [self.intercept_true, self.c1_true, self.c2_true], decimal=0) # low decimal due to noise

        # Residuals sum to near zero (property of OLS with intercept)
        self.assertAlmostEqual(residuals.sum(), 0.0, places=5)

        # R-squared should be high for this model with moderate noise
        self.assertTrue(0.9 < r_squared <= 1.0)

        # Check R-squared calculation consistency
        y_pred = (self.X_df['X1'] * beta_hat['X1'] +
                  self.X_df['X2'] * beta_hat['X2'] +
                  beta_hat['intercept'])
        manual_residuals = self.Y_series - y_pred
        rss = np.sum(manual_residuals**2)
        tss = np.sum((self.Y_series - self.Y_series.mean())**2)
        manual_r_squared = 1 - (rss / tss)
        self.assertAlmostEqual(r_squared, manual_r_squared, places=6)


    def test_ordinary_least_squares_no_intercept(self):
        # Y_no_intercept = c1*X1 + c2*X2 + noise (generate Y without true intercept)
        Y_no_intercept_series = pd.Series(
            self.c1_true * self.X_df['X1'] + self.c2_true * self.X_df['X2'] + self.noise,
            name="Y_no_intercept"
        )

        beta_hat, residuals, r_squared = ordinary_least_squares(Y_no_intercept_series, self.X_df, add_intercept=False)

        self.assertNotIn('intercept', beta_hat.index)
        self.assertEqual(len(beta_hat), 2) # X1, X2

        assert_array_almost_equal([beta_hat['X1'], beta_hat['X2']],
                                  [self.c1_true, self.c2_true], decimal=0)
        # Residuals sum won't necessarily be zero without intercept

    def test_weighted_least_squares_uniform_weights(self):
        # WLS with uniform weights should yield same results as OLS
        uniform_weights = pd.Series(np.ones(self.N), index=self.Y_series.index)

        beta_hat_wls, residuals_wls, r_squared_wls = weighted_least_squares(
            self.Y_series, self.X_df, uniform_weights, add_intercept=True
        )

        beta_hat_ols, residuals_ols, r_squared_ols = ordinary_least_squares(
            self.Y_series, self.X_df, add_intercept=True
        )

        assert_series_equal(beta_hat_wls, beta_hat_ols, atol=1e-6)
        # Residuals and R-squared are calculated on original Y scale, so should also match
        assert_series_equal(residuals_wls, residuals_ols, atol=1e-6)
        self.assertAlmostEqual(r_squared_wls, r_squared_ols, places=6)

    def test_weighted_least_squares_heterogeneous_weights(self):
        beta_hat, residuals, r_squared = weighted_least_squares(
            self.Y_series, self.X_df, self.weights, add_intercept=True
        )

        self.assertIn('intercept', beta_hat.index)
        self.assertEqual(len(beta_hat), 3)
        self.assertEqual(len(residuals), self.N)

        # Coefficients should still be somewhat close to true values, but will differ from OLS
        # due to weighting. Harder to give exact bounds without deriving weighted solution.
        # For now, primarily a smoke test that it runs and returns correct shapes/types.
        self.assertTrue(isinstance(beta_hat, pd.Series))
        self.assertTrue(isinstance(residuals, pd.Series))
        self.assertTrue(isinstance(r_squared, float))

        # R-squared is on original Y, so it's comparable to OLS R-squared,
        # but might be higher or lower depending on how weights correlate with errors.
        self.assertTrue(0.0 <= r_squared <= 1.0)

        # Check that residuals are on original scale: Y - X @ beta_hat
        X_final_wls = self.X_df.copy()
        X_final_wls['intercept'] = 1.0
        cols = ['intercept'] + [col for col in self.X_df.columns if col != 'intercept']
        X_final_wls = X_final_wls[cols]

        expected_residuals_values = self.Y_series.values - (X_final_wls.values @ beta_hat.values)
        assert_array_almost_equal(residuals.values, expected_residuals_values, decimal=6)


    def test_ols_edge_cases_perfect_fit(self):
        Y_perfect = pd.Series(2.0 * self.X_df['X1'] + 3.0) # Perfect fit, no noise
        X_perfect = pd.DataFrame({'X1': self.X_df['X1']})

        beta_hat, residuals, r_squared = ordinary_least_squares(Y_perfect, X_perfect, add_intercept=True)

        self.assertAlmostEqual(beta_hat['intercept'], 3.0, places=6)
        self.assertAlmostEqual(beta_hat['X1'], 2.0, places=6)
        self.assertAlmostEqual(r_squared, 1.0, places=6)
        self.assertTrue(np.allclose(residuals.values, 0.0, atol=1e-6))

    def test_ols_constant_y(self):
        Y_constant = pd.Series(np.full(self.N, 5.0), name="Y_const")
        beta_hat, residuals, r_squared = ordinary_least_squares(Y_constant, self.X_df, add_intercept=True)

        # Model should predict Y_mean. Intercept = Y_mean, other betas = 0.
        self.assertAlmostEqual(beta_hat['intercept'], 5.0, places=6)
        self.assertAlmostEqual(beta_hat['X1'], 0.0, places=6)
        self.assertAlmostEqual(beta_hat['X2'], 0.0, places=6)
        self.assertTrue(np.allclose(residuals.values, 0.0, atol=1e-6))
        self.assertAlmostEqual(r_squared, 1.0, places=6) # Perfect fit to a constant

    def test_ols_constant_y_no_intercept(self):
        Y_constant = pd.Series(np.full(self.N, 5.0), name="Y_const")
        # If no intercept, OLS tries to fit Y = Xb. If Y is constant and X varies,
        # this might result in non-zero betas and non-zero residuals.
        # R-squared might be negative or low.
        beta_hat, residuals, r_squared = ordinary_least_squares(Y_constant, self.X_df, add_intercept=False)

        # Check that TSS = 0 results in R^2 = 1 if RSS = 0, else 0.
        # Here RSS will likely not be 0.
        # total_sum_squares = np.sum((Y_constant.values - np.mean(Y_constant.values))**2) -> 0
        # residual_sum_squares = np.sum(residuals**2)
        # The code has: if total_sum_squares == 0: r_squared = 1.0 if residual_sum_squares == 0 else 0.0
        if np.sum(residuals**2) == 0:
             self.assertAlmostEqual(r_squared, 1.0, places=6)
        else:
             self.assertAlmostEqual(r_squared, 0.0, places=6)


if __name__ == '__main__':
    unittest.main()
