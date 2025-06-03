# tests/test_backtesting.py

import unittest
import numpy as np
import pandas as pd

from quant_elements_lib.backtesting.framework import (
    generate_k_fold_cv_indices_ts,
    generate_walk_forward_indices,
)
from quant_elements_lib.backtesting.ras_evaluation import (
    calculate_empirical_rademacher_complexity,
    calculate_ras_lower_bound_ic,
    calculate_ras_lower_bound_sr,
)

class TestBacktestingFramework(unittest.TestCase):

    def test_generate_k_fold_cv_indices_ts_basic(self):
        num_obs = 100
        k_folds = 5
        splits = generate_k_fold_cv_indices_ts(num_obs, k_folds, gap_between_folds=0)
        self.assertEqual(len(splits), k_folds)

        # Check properties of TimeSeriesSplit-like behavior
        # (initial train + k test blocks, train expands)
        # num_test_blocks = k_folds
        # first_test_start_index = num_obs // (k_folds + 1)

        for i, (train_idx, test_idx) in enumerate(splits):
            self.assertTrue(len(train_idx) > 0)
            self.assertTrue(len(test_idx) > 0)
            if i > 0: # Second fold onwards, training set should be larger
                self.assertTrue(len(train_idx) > len(splits[i-1][0]))
            if len(train_idx) and len(test_idx):
                 self.assertTrue(train_idx[-1] < test_idx[0]) # Train must precede test
            # Test sets should be contiguous or gapped, and not overlap
            if i > 0 and len(splits[i-1][1]) and len(test_idx):
                 self.assertTrue(splits[i-1][1][-1] < test_idx[0])


    def test_generate_k_fold_cv_indices_ts_with_gap(self):
        num_obs = 100
        k_folds = 3
        gap = 5
        splits = generate_k_fold_cv_indices_ts(num_obs, k_folds, gap_between_folds=gap)
        self.assertEqual(len(splits), k_folds)
        for train_idx, test_idx in splits:
            if len(train_idx) and len(test_idx):
                self.assertTrue(train_idx[-1] + gap < test_idx[0])

    def test_generate_k_fold_cv_indices_ts_errors(self):
        with self.assertRaises(ValueError): # k_folds <= 0
            generate_k_fold_cv_indices_ts(100, 0)
        with self.assertRaises(ValueError): # gap < 0
            generate_k_fold_cv_indices_ts(100, 5, -1)
        with self.assertRaises(ValueError): # num_obs too small for folds
            generate_k_fold_cv_indices_ts(5, 5) # Needs at least k+1 = 6
        with self.assertRaises(ValueError): # gap too large
            generate_k_fold_cv_indices_ts(20, 3, 10) # first train segment size = 20 // 4 = 5. gap 10 > 5

    def test_generate_walk_forward_indices_fixed_window(self):
        num_obs = 100
        train_size = 20
        test_size = 5
        splits = generate_walk_forward_indices(num_obs, train_size, test_size, fixed_train_window=True)

        self.assertTrue(len(splits) > 0)
        expected_num_splits = (num_obs - train_size) // test_size
        self.assertEqual(len(splits), expected_num_splits)

        for train_idx, test_idx in splits:
            self.assertEqual(len(train_idx), train_size)
            self.assertEqual(len(test_idx), test_size)
            if len(train_idx) and len(test_idx):
                self.assertTrue(train_idx[-1] < test_idx[0])

    def test_generate_walk_forward_indices_expanding_window(self):
        num_obs = 100
        initial_train_size = 20
        test_size = 5
        splits = generate_walk_forward_indices(num_obs, train_window_size=initial_train_size,
                                               test_window_size=test_size, fixed_train_window=False,
                                               initial_train_size=initial_train_size)
        self.assertTrue(len(splits) > 0)
        expected_num_splits = (num_obs - initial_train_size) // test_size
        self.assertEqual(len(splits), expected_num_splits)

        for i, (train_idx, test_idx) in enumerate(splits):
            self.assertEqual(len(train_idx), initial_train_size + i * test_size)
            self.assertEqual(len(test_idx), test_size)

    def test_generate_walk_forward_indices_errors(self):
        with self.assertRaises(ValueError): # num_obs <=0
            generate_walk_forward_indices(0, 10, 1)
        with self.assertRaises(ValueError): # test_window_size <=0
            generate_walk_forward_indices(100, 10, 0)
        with self.assertRaises(ValueError): # train_window_size <=0 (for fixed)
            generate_walk_forward_indices(100, 0, 1, fixed_train_window=True)
        with self.assertRaises(ValueError): # initial_train_size <=0 (for expanding)
            generate_walk_forward_indices(100, 10, 1, fixed_train_window=False, initial_train_size=0)

        # Case where no splits can be formed (valid, returns empty list)
        splits = generate_walk_forward_indices(30, train_window_size=25, test_window_size=10)
        self.assertEqual(len(splits), 0)


class TestRASEvaluation(unittest.TestCase):

    def test_calculate_empirical_rademacher_complexity_single_strategy(self):
        # For a single strategy (one column), R_hat_T should be E_eps[ (eps.T @ X) / T ]
        # If X is all ones, sum(eps_t * 1) / T. E[sum(eps_t)] = 0. So R_hat_T should be near 0.
        T = 100
        X_single_ones = pd.DataFrame(np.ones((T, 1)), columns=['strat1'])
        r_hat_T = calculate_empirical_rademacher_complexity(X_single_ones, num_rademacher_samples=2000)
        self.assertAlmostEqual(r_hat_T, 0.0, delta=0.05) # Expect near zero

        # If X is random mean zero
        np.random.seed(42)
        X_single_random = pd.DataFrame(np.random.randn(T, 1) * 0.1, columns=['strat1'])
        r_hat_T_random = calculate_empirical_rademacher_complexity(X_single_random, num_rademacher_samples=2000)
        # This should also be small, as sup over one item is just that item.
        # E[ (eps.T @ X) / T ]. If X is mean zero, this should be near zero.
        self.assertAlmostEqual(r_hat_T_random, 0.0, delta=0.05)


    def test_calculate_empirical_rademacher_complexity_zero_matrix(self):
        T = 100
        N_strat = 5
        X_zeros = pd.DataFrame(np.zeros((T, N_strat)))
        r_hat_T = calculate_empirical_rademacher_complexity(X_zeros, num_rademacher_samples=100)
        self.assertEqual(r_hat_T, 0.0)

    def test_calculate_empirical_rademacher_complexity_multiple_strategies(self):
        T = 100
        N_strat = 10
        np.random.seed(42)
        # Create some correlated strategies
        common_factor = np.random.randn(T,1) * 0.05
        X_data = common_factor + np.random.randn(T, N_strat) * 0.01
        X_df = pd.DataFrame(X_data)

        r_hat_T = calculate_empirical_rademacher_complexity(X_df, num_rademacher_samples=1000)
        self.assertTrue(r_hat_T >= 0) # Should be non-negative
        # Massart's bound (approx): sqrt(2*logN/T) * max_norm_X_s
        # max_norm_X_s is roughly sqrt(T * var(X_s)). If var(X_s) ~ 0.05^2 = 0.0025
        # max_norm_X_s ~ sqrt(100 * 0.0025) = sqrt(0.25) = 0.5
        # bound ~ sqrt(2*log10/100) * 0.5 = sqrt(0.046) * 0.5 = 0.21 * 0.5 = 0.1
        # This is a loose upper bound, r_hat_T should be less.
        self.assertLess(r_hat_T, 0.1)


    def test_ras_bounds_known_values_ic(self):
        empirical_ic = 0.05
        r_hat_T = 0.01
        num_obs = 1000
        delta = 0.05

        # estimation_error = sqrt((2 * log(2/0.05)) / 1000) = sqrt(2 * log(40) / 1000)
        # = sqrt(2 * 3.689 / 1000) = sqrt(7.378 / 1000) = sqrt(0.007378) approx 0.08589
        # expected_bound = 0.05 - 2 * 0.01 - 0.08589 = 0.05 - 0.02 - 0.08589 = 0.03 - 0.08589 = -0.05589
        expected_est_error = np.sqrt((2 * np.log(2 / delta)) / num_obs)
        expected_bound = empirical_ic - 2 * r_hat_T - expected_est_error

        bound = calculate_ras_lower_bound_ic(empirical_ic, r_hat_T, num_obs, delta)
        self.assertAlmostEqual(bound, expected_bound, places=5)

    def test_ras_bounds_known_values_sr(self):
        empirical_sr = 0.5 # non-annualized
        r_hat_T = 0.02
        num_obs = 1000
        delta = 0.05

        # theta_hat_S_sq_half = (0.5**2) / 2 = 0.25 / 2 = 0.125
        # estimation_error = sqrt((2 * log(40) * (1 + 0.125)) / 1000)
        # = sqrt(7.378 * 1.125 / 1000) = sqrt(8.30025 / 1000) = sqrt(0.00830025) approx 0.09110
        # expected_bound = 0.5 - 2 * 0.02 - 0.09110 = 0.5 - 0.04 - 0.09110 = 0.46 - 0.09110 = 0.3689
        theta_hat_S_sq_half = (empirical_sr**2) / 2
        expected_est_error = np.sqrt(
            (2 * np.log(2 / delta) * (1 + theta_hat_S_sq_half)) / num_obs
        )
        expected_bound = empirical_sr - 2 * r_hat_T - expected_est_error

        bound = calculate_ras_lower_bound_sr(empirical_sr, r_hat_T, num_obs, delta)
        self.assertAlmostEqual(bound, expected_bound, places=5)

    def test_ras_errors(self):
        with self.assertRaises(ValueError):
            calculate_ras_lower_bound_ic(0.1, 0.01, 0, 0.05) # T <= 0
        with self.assertRaises(ValueError):
            calculate_ras_lower_bound_sr(0.5, 0.01, 100, 0) # delta not in (0,1)
        with self.assertRaises(ValueError):
            calculate_ras_lower_bound_sr(0.5, 0.01, 100, 1) # delta not in (0,1)
        with self.assertRaises(ValueError):
            calculate_empirical_rademacher_complexity(pd.DataFrame(), 100) # empty df
        with self.assertRaises(ValueError):
            calculate_empirical_rademacher_complexity(pd.DataFrame({'a':[1]}), 0) # samples <=0


if __name__ == '__main__':
    unittest.main()
