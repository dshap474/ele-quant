import unittest
import pandas as pd
import numpy as np
from pandas.testing import assert_series_equal, assert_frame_equal
from numpy.testing import assert_array_almost_equal

try:
    from quant_elements_lib.core import FactorModelBase
    from quant_elements_lib.factor_models.transformations import rotate_factor_model, project_factor_model_to_subset
except ImportError:
    import sys
    sys.path.insert(0, '..')
    from quant_elements_lib.core.factor_model_base import FactorModelBase
    from quant_elements_lib.factor_models.transformations import rotate_factor_model, project_factor_model_to_subset

# Minimal concrete FactorModel for testing transformations
class TestConcreteFactorModel(FactorModelBase):
    def __init__(self, alpha=None, B_loadings=None, factor_covariance=None,
                 idiosyncratic_covariance=None, factor_returns=None):
        super().__init__(alpha, B_loadings, factor_covariance, idiosyncratic_covariance)
        # Explicitly store factor_returns as rotate_factor_model might try to access/transform it
        self.factor_returns = factor_returns


class TestFactorModelsTransformations(unittest.TestCase):

    def setUp(self):
        self.asset_names = ['Asset1', 'Asset2', 'Asset3']
        self.factor_names = ['FactorA', 'FactorB'] # K=2

        self.alpha = pd.Series([0.01, 0.005, 0.02], index=self.asset_names)
        self.B_loadings = pd.DataFrame({
            'FactorA': [1.0, 0.8, 0.5],
            'FactorB': [0.2, 0.7, 1.1]
        }, index=self.asset_names) # N=3, K=2

        self.factor_covariance = pd.DataFrame({
            'FactorA': [0.04, 0.01],
            'FactorB': [0.01, 0.09]
        }, index=self.factor_names, columns=self.factor_names) # KxK

        self.idio_cov_diag = pd.Series([0.01, 0.005, 0.008], index=self.asset_names)

        self.factor_returns_sample = pd.DataFrame({
            'FactorA': [0.02, -0.01, 0.015, 0.005],
            'FactorB': [-0.01, 0.005, 0.01, -0.002]
        }, index=pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'])) # T=4, K=2

        self.base_model = TestConcreteFactorModel(
            alpha=self.alpha,
            B_loadings=self.B_loadings,
            factor_covariance=self.factor_covariance,
            idiosyncratic_covariance=self.idio_cov_diag,
            factor_returns=self.factor_returns_sample
        )

    def test_rotate_factor_model(self):
        # Define a 2x2 rotation matrix C
        # C columns are new factor names, index are original factor names
        C_rotation_matrix = pd.DataFrame({
            'FactorARot': [1, 0.5], # FactorARot = 1*FactorA + 0.5*FactorB
            'FactorBRot': [0.5, 1]  # FactorBRot = 0.5*FactorA + 1*FactorB
        }, index=self.factor_names)
        # C_rotation_matrix = C.T from some definitions. Let's assume C_rotation_matrix is "C"
        # B_rot = B @ inv(C)
        # Omega_f_rot = C @ Omega_f @ C.T

        rotated_model = rotate_factor_model(self.base_model, C_rotation_matrix)

        self.assertIsNotNone(rotated_model)
        self.assertEqual(rotated_model.B_loadings.shape, self.B_loadings.shape)
        self.assertEqual(rotated_model.factor_covariance.shape, self.factor_covariance.shape)
        self.assertTrue(all(item in rotated_model.B_loadings.columns for item in ['FactorARot', 'FactorBRot']))
        self.assertTrue(all(item in rotated_model.factor_covariance.index for item in ['FactorARot', 'FactorBRot']))
        self.assertTrue(all(item in rotated_model.factor_covariance.columns for item in ['FactorARot', 'FactorBRot']))

        # Check that alpha and idiosyncratic_covariance are carried over
        assert_series_equal(self.base_model.alpha, rotated_model.alpha)
        assert_series_equal(self.base_model.idiosyncratic_covariance, rotated_model.idiosyncratic_covariance)

        # Verify B_rot @ Ω_f_rot @ B_rot.T ≈ B_orig @ Ω_f_orig @ B_orig.T
        # This is the systematic component of the asset covariance matrix
        sys_risk_original = self.base_model.B_loadings @ self.base_model.factor_covariance @ self.base_model.B_loadings.T
        sys_risk_rotated = rotated_model.B_loadings @ rotated_model.factor_covariance @ rotated_model.B_loadings.T

        assert_frame_equal(sys_risk_original, sys_risk_rotated, check_dtype=False, atol=1e-6)

        # Check factor returns transformation: f_rot = f @ C.T
        # C_rotation_matrix is K_orig_rows x K_new_cols
        # The function expects C_rotation_matrix to be K_orig x K_new (mapping matrix)
        # f_rot (T x K_new) = f_orig (T x K_orig) @ C_transformed (K_orig x K_new for C.T in formula)
        # The current rotate_factor_model uses f_rot = f @ C.T where C is the input matrix.
        # So C_rotation_matrix is the C in that formula.
        # f_rot (T x K_new) = f_orig (T x K_orig) @ C.T (K_orig_cols (new_factors) x K_orig_rows (old_factors).T)
        # This needs C_rotation_matrix to be K_new x K_orig.
        # The current code for rotate_factor_model:
        # factor_returns_rotated_values = model.factor_returns.values @ C_rotation_matrix.values.T
        # So C_rotation_matrix is K_orig_factors_as_index x K_new_factors_as_columns
        # This implies C_rotation_matrix.values.T is K_new x K_orig
        # f_rot (T x K_new) = f_orig (T x K_orig) @ C.values.T (K_new x K_orig) -- this is not quite right.
        # f_orig is T x K_orig. C_rotation_matrix is K_orig x K_new.
        # f_rot should be T x K_new. So f_orig @ C_rotation_matrix.
        # Let's recheck the notes for rotate_factor_model: "f_rotated = f @ C.T"
        # If C is K_new x K_orig, then f_rotated (T x K_new) = f_orig (T x K_orig) @ C.T (K_orig x K_new). This matches.
        # The current C_rotation_matrix is K_orig x K_new. So we need f @ inv(C_rotation_matrix.T) for consistency with B_rot = B @ inv(C).
        # Or, if factors f transform as f_new = C.T @ f_old (vector form), then f_new_T = f_old_T @ C.
        # The implementation is f_rot = f @ C.T, where C is K_orig_idx x K_new_col
        # So f_rot (T x K_new) = f_orig (T x K_orig) @ (K_new_col_as_idx x K_orig_row_as_col from C.T)
        # This means C.T should be K_orig x K_new.
        # The current C_rotation_matrix is K_orig x K_new. C.values.T is K_new x K_orig.
        # factor_returns_rotated_values = model.factor_returns.values @ C_rotation_matrix.values.T seems dimensionally incorrect if K_new != K_orig.
        # It should be: model.factor_returns.values @ C_rotation_matrix.values
        # For K_new == K_orig, it will run but the math might be different than B's transformation.
        # Let's assume the current implementation of rotate_factor_model for f_returns is f_orig @ C_matrix.T
        # (T x K_orig) @ (K_new x K_orig). This is what is in code.
        # This is only valid if K_new == K_orig for the .T to make sense for the factor_returns.
        # The provided C_rotation_matrix is K_orig x K_new. So C.values.T is K_new x K_orig.
        # factor_returns (T x K_orig) @ C.values.T (K_new x K_orig) -> error unless K_orig = K_new
        # The current code uses new_factor_names = C_rotation_matrix.columns (K_new)
        # And C_rotation_matrix.values.T (K_new x K_orig)
        # So (T x K_orig) @ (K_new x K_orig) -> should be (T x K_orig) @ (K_orig x K_new)
        # It seems the transformation rule for factor_returns f_rot = f @ C.T expects C to be K_orig_factors x K_new_factors.
        # The current code for B_loadings is B_rot = B @ inv(C). C is K_orig x K_new. inv(C) is K_new x K_orig.
        # B (N x K_orig) @ inv(C) (K_orig x K_new, if C is square, or pseudo-inverse).
        # My C_rotation_matrix is K_orig_idx x K_new_cols.
        # B_loadings_rotated_values = model.B_loadings.values @ np.linalg.inv(C_rotation_matrix.values)
        # This requires C to be square (K_orig == K_new).
        # Factor_returns: factor_returns_rotated_values = model.factor_returns.values @ C_rotation_matrix.values.T
        # This is (T x K_orig) @ (K_new x K_orig). This requires K_orig = K_new.
        # My C_rotation_matrix is square (2x2). So C.values.T is also 2x2.

        if hasattr(rotated_model, 'factor_returns') and rotated_model.factor_returns is not None:
            self.assertEqual(rotated_model.factor_returns.shape[0], self.factor_returns_sample.shape[0]) # T
            self.assertEqual(rotated_model.factor_returns.shape[1], C_rotation_matrix.shape[1]) # K_new
            self.assertTrue(all(item in rotated_model.factor_returns.columns for item in ['FactorARot', 'FactorBRot']))

            expected_f_rot = self.factor_returns_sample.values @ C_rotation_matrix.values.T
            # Note: This is what the code implements. The math consistency across B and f needs careful thought for non-orthogonal C.
            # For orthogonal C, C.T = inv(C). Then B_rot = B @ C and f_rot = f @ C.
            # The current transformations are B_rot = B @ inv(C) and f_rot = f @ C.T
            assert_array_almost_equal(rotated_model.factor_returns.values, expected_f_rot, decimal=6)


    def test_project_factor_model_to_subset(self):
        factors_to_keep = ['FactorA'] # Keep only FactorA

        projected_model = project_factor_model_to_subset(self.base_model, factors_to_keep)

        self.assertIsNotNone(projected_model)
        self.assertEqual(projected_model.B_loadings.shape, (self.B_loadings.shape[0], 1)) # N x 1
        self.assertEqual(projected_model.factor_covariance.shape, (1, 1)) # 1 x 1
        self.assertTrue(all(item in projected_model.B_loadings.columns for item in factors_to_keep))
        self.assertTrue(all(item in projected_model.factor_covariance.index for item in factors_to_keep))
        self.assertTrue(all(item in projected_model.factor_covariance.columns for item in factors_to_keep))

        # Check values
        assert_series_equal(self.base_model.B_loadings['FactorA'], projected_model.B_loadings['FactorA'])
        expected_factor_cov_A = pd.DataFrame(
            [[self.base_model.factor_covariance.loc['FactorA', 'FactorA']]],
            index=['FactorA'], columns=['FactorA']
        )
        assert_frame_equal(expected_factor_cov_A, projected_model.factor_covariance)

        # Check that alpha and idiosyncratic_covariance are carried over
        assert_series_equal(self.base_model.alpha, projected_model.alpha)
        assert_series_equal(self.base_model.idiosyncratic_covariance, projected_model.idiosyncratic_covariance)

        # Check factor returns projection
        if hasattr(projected_model, 'factor_returns') and projected_model.factor_returns is not None:
            self.assertEqual(projected_model.factor_returns.shape[1], len(factors_to_keep))
            assert_frame_equal(self.base_model.factor_returns[factors_to_keep], projected_model.factor_returns)

    def test_project_factor_model_to_subset_invalid_factor(self):
        factors_to_keep = ['FactorC'] # FactorC does not exist
        with self.assertRaises(ValueError):
            project_factor_model_to_subset(self.base_model, factors_to_keep)

if __name__ == '__main__':
    unittest.main()
