import unittest
import pandas as pd
import numpy as np
from pandas.testing import assert_series_equal, assert_frame_equal
from numpy.testing import assert_array_almost_equal

try:
    from quant_elements_lib.factor_models.utils import decompose_alpha
except ImportError:
    import sys
    sys.path.insert(0, '..')
    from quant_elements_lib.factor_models.utils import decompose_alpha

class TestFactorModelsUtils(unittest.TestCase):

    def setUp(self):
        self.asset_names = ['AssetA', 'AssetB', 'AssetC']
        self.factor_names_k1 = ['FactorX']
        self.factor_names_k2 = ['FactorX', 'FactorY']

        # Case 1: Single factor (K=1)
        self.B_loadings_k1 = pd.DataFrame({
            'FactorX': [1.0, 0.8, 1.2]
        }, index=self.asset_names)

        # Case 2: Two factors (K=2)
        self.B_loadings_k2 = pd.DataFrame({
            'FactorX': [1.0, 0.8, 1.2],
            'FactorY': [0.5, -0.3, 0.0]
        }, index=self.asset_names)

        self.alpha_total_simple = pd.Series([0.01, 0.02, 0.03], index=self.asset_names, name='alpha_total')

    def test_decompose_alpha_fully_spanned_k1(self):
        # alpha_total is constructed to be perfectly spanned by B_loadings_k1
        # Let lambda_spanned = [0.01] (Kx1)
        # alpha_spanned = B @ lambda_spanned = [[1.0],[0.8],[1.2]] @ [0.01] = [[0.01],[0.008],[0.012]]
        lambda_true = pd.Series([0.01], index=self.factor_names_k1)
        alpha_total_spanned = self.B_loadings_k1 @ lambda_true
        alpha_total_spanned.name = 'alpha_total'

        alpha_s, alpha_o, lambda_s = decompose_alpha(alpha_total_spanned, self.B_loadings_k1)

        assert_series_equal(alpha_s, alpha_total_spanned, check_names=False, atol=1e-7)
        self.assertTrue(np.allclose(alpha_o.values, 0.0, atol=1e-7)) # Orthogonal part should be zero
        assert_series_equal(lambda_s, lambda_true, check_names=False, atol=1e-7)
        assert_series_equal(alpha_s + alpha_o, alpha_total_spanned, check_names=False, atol=1e-7)
        self.assertEqual(alpha_s.name, "alpha_spanned")
        self.assertEqual(alpha_o.name, "alpha_orthogonal")
        self.assertEqual(lambda_s.name, "lambda_spanned")


    def test_decompose_alpha_fully_orthogonal_k1(self):
        # Construct alpha_total to be orthogonal to B_loadings_k1 columns.
        # For K=1, if B = [b1, b2, b3]', then alpha_o needs to be in null_space(B.T)
        # which means B.T @ alpha_o = 0.  [b1,b2,b3] @ [a1,a2,a3]' = 0.
        # e.g. B = [1, 0.8, 1.2]'. Let alpha_o = [0.8, -1, 0]' (scaled to be small)
        # No, this is wrong. alpha_o should be orthogonal to columns of B, i.e. B_loadings_k1.T @ alpha_o should be 0.
        # This is not quite right. alpha_orthogonal means it cannot be represented as B @ lambda.
        # If alpha_total is orthogonal to columns of B, then B.T @ alpha_total = 0.
        # This implies (B.T @ B) @ inv(B.T @ B) @ B.T @ alpha_total = 0, so lambda_spanned = 0.

        # Example: B = [1,1,1]'. alpha_o = [1, -1, 0]'. B.T @ alpha_o = 0.
        # For self.B_loadings_k1 = [1.0, 0.8, 1.2]'
        # Choose alpha_o such that B_loadings_k1.T @ alpha_o = 0
        # 1.0*a1 + 0.8*a2 + 1.2*a3 = 0. Let a1=0.01, a2=0.01. Then 1.0*0.01 + 0.8*0.01 = -1.2*a3 => 0.018 = -1.2*a3 => a3 = -0.015
        alpha_total_ortho = pd.Series([0.01, 0.01, -0.015], index=self.asset_names, name='alpha_total')
        # Verify orthogonality:
        # print(self.B_loadings_k1.T @ alpha_total_ortho) # Should be close to 0

        alpha_s, alpha_o, lambda_s = decompose_alpha(alpha_total_ortho, self.B_loadings_k1)

        self.assertTrue(np.allclose(alpha_s.values, 0.0, atol=1e-7)) # Spanned part should be zero
        self.assertTrue(np.allclose(lambda_s.values, 0.0, atol=1e-7)) # Lambda should be zero
        assert_series_equal(alpha_o, alpha_total_ortho, check_names=False, atol=1e-7)
        assert_series_equal(alpha_s + alpha_o, alpha_total_ortho, check_names=False, atol=1e-7)

    def test_decompose_alpha_mixed_case_k2(self):
        # alpha_total = some_spanned_part + some_orthogonal_part
        lambda_true_spanned_part = pd.Series([0.01, -0.005], index=self.factor_names_k2)
        alpha_spanned_part = self.B_loadings_k2 @ lambda_true_spanned_part

        # Construct an orthogonal part. For K=2, N=3. B is 3x2.
        # We need alpha_o such that B.T @ alpha_o is close to zero.
        # This is harder to construct directly.
        # Let's use a known alpha_total and see the decomposition.
        alpha_total_mixed = pd.Series([0.015, 0.003, 0.010], index=self.asset_names, name='alpha_total')

        alpha_s, alpha_o, lambda_s = decompose_alpha(alpha_total_mixed, self.B_loadings_k2)

        self.assertEqual(alpha_s.shape, (3,))
        self.assertEqual(alpha_o.shape, (3,))
        self.assertEqual(lambda_s.shape, (2,)) # K=2

        # Check that alpha_spanned is indeed B @ lambda_spanned
        expected_alpha_s_from_lambda = self.B_loadings_k2 @ lambda_s
        assert_series_equal(alpha_s, expected_alpha_s_from_lambda, check_names=False, atol=1e-7)

        # Check that alpha_total = spanned + orthogonal
        assert_series_equal(alpha_s + alpha_o, alpha_total_mixed, check_names=False, atol=1e-7)

        # Check that the calculated alpha_orthogonal is actually orthogonal to B_loadings columns
        # B.T @ alpha_o should be close to zero vector (Kx1)
        Bt_alpha_o = self.B_loadings_k2.T @ alpha_o
        assert_array_almost_equal(Bt_alpha_o.values, np.zeros(self.B_loadings_k2.shape[1]), decimal=6)

    def test_decompose_alpha_shapes(self):
        alpha_s, alpha_o, lambda_s = decompose_alpha(self.alpha_total_simple, self.B_loadings_k1)
        self.assertEqual(alpha_s.shape, (3,)) # N
        self.assertEqual(alpha_o.shape, (3,)) # N
        self.assertEqual(lambda_s.shape, (1,)) # K

        alpha_s_k2, alpha_o_k2, lambda_s_k2 = decompose_alpha(self.alpha_total_simple, self.B_loadings_k2)
        self.assertEqual(alpha_s_k2.shape, (3,)) # N
        self.assertEqual(alpha_o_k2.shape, (3,)) # N
        self.assertEqual(lambda_s_k2.shape, (2,)) # K

    def test_alignment_error(self):
        alpha_misaligned = pd.Series([0.1, 0.2], index=['AssetX', 'AssetY']) # Different assets
        with self.assertRaises(ValueError):
            decompose_alpha(alpha_misaligned, self.B_loadings_k1)

if __name__ == '__main__':
    unittest.main()
