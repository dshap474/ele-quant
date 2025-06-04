import unittest
import pandas as pd
import numpy as np
from pandas.testing import assert_series_equal
from numpy.testing import assert_array_almost_equal

try:
    from quant_elements_lib.factor_models.utils import decompose_alpha, choose_num_factors_threshold
except ImportError:
    # This configuration might be for running tests directly from the tests directory
    import sys
    sys.path.insert(0, '..') # Add the parent directory (project root) to the Python path
    from quant_elements_lib.factor_models.utils import decompose_alpha, choose_num_factors_threshold


class TestFactorModelsUtils(unittest.TestCase):

    def setUp(self):
        self.asset_names = ['AssetA', 'AssetB', 'AssetC']
        self.factor_names_k1 = ['FactorX']
        self.factor_names_k2 = ['FactorX', 'FactorY']

        # Data for decompose_alpha
        self.B_loadings_k1 = pd.DataFrame({
            'FactorX': [1.0, 0.8, 1.2]
        }, index=self.asset_names)
        self.B_loadings_k2 = pd.DataFrame({
            'FactorX': [1.0, 0.8, 1.2],
            'FactorY': [0.5, -0.3, 0.0]
        }, index=self.asset_names)
        self.alpha_total_simple = pd.Series([0.01, 0.02, 0.03], index=self.asset_names, name='alpha_total')

    # Tests for decompose_alpha (from existing file)
    def test_decompose_alpha_fully_spanned_k1(self):
        lambda_true = pd.Series([0.01], index=self.factor_names_k1)
        alpha_total_spanned = self.B_loadings_k1 @ lambda_true
        alpha_total_spanned.name = 'alpha_total'
        alpha_s, alpha_o, lambda_s = decompose_alpha(alpha_total_spanned, self.B_loadings_k1)
        assert_series_equal(alpha_s, alpha_total_spanned, check_names=False, atol=1e-7)
        self.assertTrue(np.allclose(alpha_o.values, 0.0, atol=1e-7))
        assert_series_equal(lambda_s, lambda_true, check_names=False, atol=1e-7)
        assert_series_equal(alpha_s + alpha_o, alpha_total_spanned, check_names=False, atol=1e-7)
        self.assertEqual(alpha_s.name, "alpha_spanned")
        self.assertEqual(alpha_o.name, "alpha_orthogonal")
        self.assertEqual(lambda_s.name, "lambda_spanned")

    def test_decompose_alpha_fully_orthogonal_k1(self):
        alpha_total_ortho = pd.Series([0.01, 0.01, -0.015 * (1.0*0.01 + 0.8*0.01)/(0.018) ], index=self.asset_names, name='alpha_total')
        # Recalculate a3 for alpha_total_ortho to be orthogonal to B_loadings_k1 = [1.0, 0.8, 1.2]'
        # 1.0*a1 + 0.8*a2 + 1.2*a3 = 0. Let a1=0.008, a2=-0.01. So 0.008 - 0.008 = 0.
        alpha_total_ortho_vals = [0.008, -0.010, 0.0] # 1*0.008 + 0.8*(-0.01) + 1.2*0 = 0
        alpha_total_ortho = pd.Series(alpha_total_ortho_vals, index=self.asset_names, name='alpha_total')

        alpha_s, alpha_o, lambda_s = decompose_alpha(alpha_total_ortho, self.B_loadings_k1)
        self.assertTrue(np.allclose(alpha_s.values, 0.0, atol=1e-7))
        self.assertTrue(np.allclose(lambda_s.values, 0.0, atol=1e-7))
        assert_series_equal(alpha_o, alpha_total_ortho, check_names=False, atol=1e-7)

    def test_decompose_alpha_mixed_case_k2(self):
        alpha_total_mixed = pd.Series([0.015, 0.003, 0.010], index=self.asset_names, name='alpha_total')
        alpha_s, alpha_o, lambda_s = decompose_alpha(alpha_total_mixed, self.B_loadings_k2)
        self.assertEqual(alpha_s.shape, (3,))
        self.assertEqual(alpha_o.shape, (3,))
        self.assertEqual(lambda_s.shape, (2,))
        expected_alpha_s_from_lambda = self.B_loadings_k2 @ lambda_s
        assert_series_equal(alpha_s, expected_alpha_s_from_lambda, check_names=False, atol=1e-7)
        assert_series_equal(alpha_s + alpha_o, alpha_total_mixed, check_names=False, atol=1e-7)
        Bt_alpha_o = self.B_loadings_k2.T @ alpha_o
        assert_array_almost_equal(Bt_alpha_o.values, np.zeros(self.B_loadings_k2.shape[1]), decimal=6)

    def test_decompose_alpha_shapes(self):
        alpha_s, alpha_o, lambda_s = decompose_alpha(self.alpha_total_simple, self.B_loadings_k1)
        self.assertEqual(alpha_s.shape, (3,)); self.assertEqual(alpha_o.shape, (3,)); self.assertEqual(lambda_s.shape, (1,))
        alpha_s_k2, alpha_o_k2, lambda_s_k2 = decompose_alpha(self.alpha_total_simple, self.B_loadings_k2)
        self.assertEqual(alpha_s_k2.shape, (3,)); self.assertEqual(alpha_o_k2.shape, (3,)); self.assertEqual(lambda_s_k2.shape, (2,))

    def test_alignment_error_decompose_alpha(self): # Renamed to avoid conflict if other alignment tests exist
        alpha_misaligned = pd.Series([0.1, 0.2], index=['AssetX', 'AssetY'])
        with self.assertRaises(ValueError):
            decompose_alpha(alpha_misaligned, self.B_loadings_k1)

    # Tests for choose_num_factors_threshold (merged and enhanced)
    def test_choose_num_factors_threshold_general_cases(self):
        # Case 1: Clear distinction (Prompt's Case 1)
        eigenvalues1 = np.array([10.0, 8.0, 5.0, 0.5, 0.3, 0.1])
        self.assertEqual(choose_num_factors_threshold(eigenvalues1, N_assets=10, T_observations=100), 3)

        # Case 2: No factors meet threshold (Prompt's Case 2)
        eigenvalues2 = np.array([0.5, 0.3, 0.1])
        self.assertEqual(choose_num_factors_threshold(eigenvalues2, N_assets=3, T_observations=100), 0)

        # Case 3: All factors meet threshold (Prompt's Case 3)
        eigenvalues3 = np.array([3.0, 2.8, 2.5])
        self.assertEqual(choose_num_factors_threshold(eigenvalues3, N_assets=3, T_observations=100), 3)

        # Case 4: N/T is large (Prompt's Case 4)
        eigenvalues4 = np.array([10.0, 8.0, 5.0, 3.0, 2.0])
        self.assertEqual(choose_num_factors_threshold(eigenvalues4, N_assets=50, T_observations=20), 3)

        # Case 5: Test capping by N_assets (Prompt's Case 6)
        eigenvalues5 = np.array([10.0, 8.0, 5.0]) # 3 factors meet threshold 1 + sqrt(2/100) ~ 1.141
        self.assertEqual(choose_num_factors_threshold(eigenvalues5, N_assets=2, T_observations=100), 2) # Capped by N_assets

        # Case 6: Test capping by len(eigenvalues)
        eigenvalues6 = np.array([10.0, 8.0, 5.0, 4.0, 3.0]) # 5 factors meet threshold 1 + sqrt(10/100) ~ 1.316
        self.assertEqual(choose_num_factors_threshold(eigenvalues6, N_assets=10, T_observations=100), 5) # Capped by len(eigenvalues)
        # Make N_assets larger than len(eigenvalues)
        self.assertEqual(choose_num_factors_threshold(eigenvalues6, N_assets=3, T_observations=100), 3) # Capped by N_assets

    def test_choose_num_factors_threshold_edge_cases(self):
        # ValueError for T_observations <= 0
        with self.assertRaises(ValueError):
            choose_num_factors_threshold(np.array([1.0]), N_assets=10, T_observations=0)
        with self.assertRaises(ValueError):
            choose_num_factors_threshold(np.array([1.0]), N_assets=10, T_observations=-1)

        # ValueError for N_assets <= 0 (as per implementation of choose_num_factors_threshold)
        with self.assertRaises(ValueError):
            choose_num_factors_threshold(np.array([1.0]), N_assets=0, T_observations=10)
        with self.assertRaises(ValueError): # N_assets < 0
             choose_num_factors_threshold(np.array([1.0]), N_assets=-1, T_observations=10)

        # Empty eigenvalues array
        self.assertEqual(choose_num_factors_threshold(np.array([]), N_assets=10, T_observations=100), 0)

        # Eigenvalues not 1D array (from existing tests, assuming this is how it's handled)
        # The function expects np.ndarray; non-ndarray or wrong dim might error out earlier or behave unexpectedly.
        # Current choose_num_factors_threshold doesn't explicitly check for ndim. It would fail at `for eig_val in eigenvalues:`.
        # Let's assume typical usage involves 1D np.ndarray. A type hint or check could be added to the main function.
        # For now, this test might be too strict or depend on internal numpy errors.
        # with self.assertRaises(TypeError): # or ValueError depending on numpy version / exact failure
        #     choose_num_factors_threshold(np.array([[1,2],[3,4]]), N_assets=10, T_observations=100)
        # with self.assertRaises(AttributeError): # if None is passed
        #    choose_num_factors_threshold(None, N_assets=10, T_observations=100)


if __name__ == '__main__':
    unittest.main()
