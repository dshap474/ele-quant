import unittest
import pandas as pd
import numpy as np
from pandas.testing import assert_series_equal, assert_frame_equal
from numpy.testing import assert_array_almost_equal

# Use try-except for import paths, similar to the example script
try:
    from quant_elements_lib.core import FactorModelBase
except ImportError:
    import sys
    sys.path.insert(0, '..') # Adjust if tests are run from a different depth
    from quant_elements_lib.core.factor_model_base import FactorModelBase

# A minimal concrete class for testing, as FactorModelBase might be abstract
# or require components not directly tested here.
class ConcreteFactorModel(FactorModelBase):
    def __init__(self, alpha=None, B_loadings=None, factor_covariance=None, idiosyncratic_covariance=None, factor_returns=None):
        super().__init__(alpha, B_loadings, factor_covariance, idiosyncratic_covariance)
        self.factor_returns = factor_returns # For predict/decompose methods

class TestFactorModelBase(unittest.TestCase):

    def setUp(self):
        """Set up a simple 2-asset, 1-factor model for testing."""
        self.asset_names = ['Asset1', 'Asset2']
        self.factor_names = ['Factor1']

        self.alpha = pd.Series([0.01, 0.005], index=self.asset_names, name='alpha')
        self.B_loadings = pd.DataFrame([[1.0], [0.8]], index=self.asset_names, columns=self.factor_names)
        self.factor_covariance = pd.DataFrame([[0.04]], index=self.factor_names, columns=self.factor_names)

        # Idiosyncratic covariance - diagonal (as Series)
        self.idio_cov_diag = pd.Series([0.01, 0.005], index=self.asset_names, name='idio_var')

        # Idiosyncratic covariance - full (as DataFrame)
        self.idio_cov_full = pd.DataFrame(np.diag([0.01, 0.005]), index=self.asset_names, columns=self.asset_names)

        # Sample factor returns (T=3, K=1)
        self.factor_returns_sample = pd.DataFrame({
            'Factor1': [0.02, -0.01, 0.015]
        }, index=pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']))

        # Sample total returns (T=3, N=2)
        # r = alpha + B*f + epsilon. For simplicity, assume epsilon is small or zero for this construction
        # This is just for testing the mechanics of decompose_total_returns
        T = len(self.factor_returns_sample)
        alpha_broadcasted = pd.DataFrame(np.tile(self.alpha.values, (T, 1)),
                                         index=self.factor_returns_sample.index,
                                         columns=self.asset_names)
        systematic_part = self.factor_returns_sample.values @ self.B_loadings.T.values
        self.total_returns_sample = pd.DataFrame(
            alpha_broadcasted.values + systematic_part,
            index=self.factor_returns_sample.index,
            columns=self.asset_names
        )
        # Add some small noise to make it more realistic for decomposition
        self.total_returns_sample += np.random.normal(0, 0.001, self.total_returns_sample.shape)


    def test_calculate_total_asset_covariance_matrix_diag_idio(self):
        model = ConcreteFactorModel(
            alpha=self.alpha,
            B_loadings=self.B_loadings,
            factor_covariance=self.factor_covariance,
            idiosyncratic_covariance=self.idio_cov_diag # Diagonal
        )
        total_cov = model.calculate_total_asset_covariance_matrix()

        self.assertIsNotNone(total_cov)
        self.assertEqual(total_cov.shape, (2, 2))

        # Expected: B @ Ω_f @ B.T + Ω_ε (diag)
        # B @ Ω_f @ B.T = [[1.0], [0.8]] @ [[0.04]] @ [[1.0, 0.8]]
        # = [[1.0], [0.8]] @ [[0.04, 0.032]]
        # = [[0.04,   0.032],
        #    [0.032,  0.0256]]
        # Ω_ε (diag) = [[0.01, 0], [0, 0.005]]
        # Expected total_cov = [[0.05,   0.032],
        #                       [0.032,  0.0306]]
        expected_B_Omega_Bt = self.B_loadings.values @ self.factor_covariance.values @ self.B_loadings.values.T
        expected_Omega_eps = np.diag(self.idio_cov_diag.values)
        expected_total_cov_values = expected_B_Omega_Bt + expected_Omega_eps

        assert_frame_equal(total_cov, pd.DataFrame(expected_total_cov_values, index=self.asset_names, columns=self.asset_names),
                           check_dtype=False, atol=1e-6)

    def test_calculate_total_asset_covariance_matrix_full_idio(self):
        model = ConcreteFactorModel(
            alpha=self.alpha,
            B_loadings=self.B_loadings,
            factor_covariance=self.factor_covariance,
            idiosyncratic_covariance=self.idio_cov_full # Full
        )
        total_cov = model.calculate_total_asset_covariance_matrix()

        self.assertIsNotNone(total_cov)
        self.assertEqual(total_cov.shape, (2, 2))

        expected_B_Omega_Bt = self.B_loadings.values @ self.factor_covariance.values @ self.B_loadings.values.T
        expected_Omega_eps = self.idio_cov_full.values
        expected_total_cov_values = expected_B_Omega_Bt + expected_Omega_eps

        assert_frame_equal(total_cov, pd.DataFrame(expected_total_cov_values, index=self.asset_names, columns=self.asset_names),
                           check_dtype=False, atol=1e-6)

    def test_predict_systematic_returns(self):
        model = ConcreteFactorModel(B_loadings=self.B_loadings) # Alpha not needed for this method

        predicted_sys_returns = model.predict_systematic_returns(self.factor_returns_sample)

        self.assertIsNotNone(predicted_sys_returns)
        self.assertEqual(predicted_sys_returns.shape, (3, 2)) # T x N

        # Expected: factor_returns @ B_loadings.T
        # [[0.02], [-0.01], [0.015]] @ [[1.0, 0.8]]
        # = [[0.02,   0.016],
        #    [-0.01, -0.008],
        #    [0.015,  0.012]]
        expected_sys_returns_values = self.factor_returns_sample.values @ self.B_loadings.values.T
        expected_sys_returns_df = pd.DataFrame(expected_sys_returns_values,
                                               index=self.factor_returns_sample.index,
                                               columns=self.asset_names)
        assert_frame_equal(predicted_sys_returns, expected_sys_returns_df, check_dtype=False, atol=1e-6)

    def test_decompose_total_returns(self):
        model = ConcreteFactorModel(
            alpha=self.alpha,
            B_loadings=self.B_loadings,
            factor_returns=self.factor_returns_sample # Not directly used by decompose, but good practice
        )

        systematic_comp, idiosyncratic_comp = model.decompose_total_returns(
            self.total_returns_sample,
            self.factor_returns_sample
        )

        self.assertIsNotNone(systematic_comp)
        self.assertIsNotNone(idiosyncratic_comp)
        self.assertEqual(systematic_comp.shape, (3, 2)) # T x N
        self.assertEqual(idiosyncratic_comp.shape, (3, 2)) # T x N

        # Check: total_returns ≈ systematic_comp + idiosyncratic_comp + alpha_broadcasted
        T = len(self.total_returns_sample)
        alpha_broadcasted_df = pd.DataFrame(
            np.tile(self.alpha.values, (T, 1)),
            index=self.total_returns_sample.index,
            columns=self.asset_names
        )

        reconstructed_total = systematic_comp + idiosyncratic_comp + alpha_broadcasted_df
        assert_frame_equal(self.total_returns_sample, reconstructed_total, check_dtype=False, atol=1e-3) # Higher atol due to noise in total_returns_sample

    def test_decompose_total_returns_no_alpha_in_model(self):
        model_no_alpha = ConcreteFactorModel(
            B_loadings=self.B_loadings # Alpha is None in the model
        )

        # For this test, total_returns_no_alpha = systematic_part + epsilon_part
        # (i.e., it doesn't inherently have the model's alpha in it)
        systematic_part = self.factor_returns_sample.values @ self.B_loadings.T.values
        total_returns_no_alpha_model = pd.DataFrame(
            systematic_part,
            index=self.factor_returns_sample.index,
            columns=self.asset_names
        ) + np.random.normal(0, 0.001, (3,2))


        systematic_comp, idiosyncratic_comp = model_no_alpha.decompose_total_returns(
            total_returns_no_alpha_model,
            self.factor_returns_sample
        )

        # Check: total_returns_no_alpha_model ≈ systematic_comp + idiosyncratic_comp
        # (since model.alpha is None, it won't be subtracted during idiosyncratic estimation)
        reconstructed_total = systematic_comp + idiosyncratic_comp
        assert_frame_equal(total_returns_no_alpha_model, reconstructed_total, check_dtype=False, atol=1e-3)


if __name__ == '__main__':
    unittest.main()
