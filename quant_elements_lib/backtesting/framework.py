import numpy as np
from typing import List, Tuple, Optional

def generate_k_fold_cv_indices_ts(num_observations: int, k_folds: int, gap_between_folds: int = 0) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generates train/test indices for K-fold cross-validation on time series data.
    This implementation is inspired by sklearn.model_selection.TimeSeriesSplit,
    where training sets expand and test sets are sequential and contiguous.
    The data is split into k_folds + 1 segments. The first segment is used for
    the initial training set, and each subsequent segment is used as a test set
    in each split. The training set expands to include all prior segments.

    Args:
        num_observations: Total number of observations in the dataset.
        k_folds: The number of splits (folds).
        gap_between_folds: Number of observations to skip between the end of
                           a training set and the beginning of its corresponding test set.

    Returns:
        A list of tuples. Each tuple contains two NumPy arrays:
        the first for training indices and the second for testing indices.

    Raises:
        ValueError: If k_folds is not positive, or if num_observations is too small
                    to create the specified number of folds with meaningful train/test sets,
                    or if gap_between_folds is too large.
    """
    if k_folds <= 0:
        raise ValueError("k_folds must be positive.")
    if gap_between_folds < 0:
        raise ValueError("gap_between_folds cannot be negative.")

    indices = np.arange(num_observations)
    splits = []

    # Determine the size of each of the (k_folds + 1) segments
    # n_segments = k_folds + 1
    # segment_size = num_observations // n_segments
    # remainder = num_observations % n_segments

    # test_fold_size determines the size of each test set
    # initial_train_size is the size of the first training set segment

    # Simplified logic based on sklearn's TimeSeriesSplit:
    # Test sets are the *last* k_folds blocks of a certain size.
    # Training sets are everything before the test set (minus gap).

    # Calculate the number of samples in each test fold.
    # (k_folds) test folds + 1 initial training block = (k_folds + 1) total blocks.
    test_fold_size = num_observations // (k_folds + 1)
    if test_fold_size == 0:
            raise ValueError(
            f"num_observations ({num_observations}) too small for {k_folds} splits. "
            f"Each test fold would be of size 0. Minimum {k_folds + 1} samples required."
        )

    for i in range(k_folds):
        # Training set ends before the gap preceding the current test set
        # The start of the first test set is after the first segment_size samples
        train_end_idx = (i + 1) * test_fold_size - gap_between_folds

        # Test set starts after the initial training segment and previous test segments
        test_start_idx = (i + 1) * test_fold_size
        test_end_idx = test_start_idx + test_fold_size

        if train_end_idx < 0:
            # This means the gap is too large for any training data to exist before the first test set
            raise ValueError(
                f"Fold {i + 1}: Gap ({gap_between_folds}) is too large, resulting in no training data "
                f"as train_end_idx is {train_end_idx}."
            )

        train_set = indices[0:train_end_idx]

        # Adjust last test set if it overshoots (can happen if num_observations is not perfectly divisible)
        if i == k_folds - 1: # If it's the last fold
            if test_end_idx < num_observations : # if default test_end_idx is not covering remaining
                 test_end_idx = num_observations # Make the last test fold consume all remaining samples

        if test_start_idx >= num_observations: # Not enough data for current test fold to even start
            raise ValueError(
                f"Cannot form {k_folds} folds. Not enough data for test set of fold {i + 1} to start. "
                f"test_start_idx ({test_start_idx}) >= num_observations ({num_observations})."
            )

        if test_start_idx >= test_end_idx : # Current test fold is empty or invalid
             raise ValueError(
                f"Cannot form {k_folds} folds. Test set for fold {i + 1} is empty or invalid. "
                f"test_start_idx ({test_start_idx}) >= test_end_idx ({test_end_idx})."
            )

        test_set = indices[test_start_idx:test_end_idx]

        if len(train_set) == 0 and test_start_idx > 0 : # No training data but test data exists
             # This condition can be hit if train_end_idx calculated to < 0 and then clipped to 0.
             # For a valid split, training data should exist unless it's the very first possible point.
             # The check `train_end_idx < 0` above should catch if gap makes all training impossible.
             # This specific check is for cases where train_set becomes empty due to slicing at 0.
             # Generally, TimeSeriesSplit implies the first training set has size `test_fold_size`.
             # If train_end_idx is 0, train_set will be empty.
             # This implies (i+1)*test_fold_size - gap_between_folds <= 0.
             # For i=0, test_fold_size <= gap_between_folds.
            if test_fold_size <= gap_between_folds and i==0:
                 raise ValueError(
                    f"Fold {i+1}: Gap ({gap_between_folds}) is >= first training segment size ({test_fold_size}), "
                    "resulting in no training data for the first split."
                )


        # Ensure train does not overlap test or come after test
        if len(train_set) > 0 and len(test_set) > 0 and train_set[-1] >= test_set[0]:
                raise ValueError(
                f"Fold {i + 1}: Training data (max index {train_set[-1]}) overlaps or "
                f"is later than test data (start index {test_set[0]}). Check gap."
            )

        splits.append((train_set, test_set))

    if len(splits) != k_folds:
        # This condition should ideally not be met if the logic for calculating fold sizes and iterating is robust.
        # It indicates that the loop terminated prematurely or parameters were inconsistent in a way not caught earlier.
        raise RuntimeError(
            f"Could not generate the required {k_folds} splits. Generated {len(splits)}. "
            f"Check num_observations ({num_observations}), k_folds, and gap_between_folds ({gap_between_folds})."
        )
    return splits


def generate_walk_forward_indices(
    num_observations: int,
    train_window_size: int,
    test_window_size: int = 1,
    fixed_train_window: bool = True,
    initial_train_size: Optional[int] = None
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generates train/test indices for walk-forward validation.
    This method simulates how models are often retrained and evaluated in production.
    Training data always precedes test data.

    Args:
        num_observations: Total number of observations in the dataset.
        train_window_size: For fixed window: the size of the rolling training window.
                           For expanding window: this is the minimum training window size.
        test_window_size: The size of the test window (number of forward steps to predict).
        fixed_train_window: If True, use a rolling training window of size `train_window_size`.
                            If False, use an expanding window.
        initial_train_size: For expanding window: the size of the first training set.
                            If None, the first training set size will be `train_window_size`.
                            For fixed window: this parameter is ignored.

    Returns:
        A list of tuples. Each tuple contains two NumPy arrays:
        the first for training indices and the second for testing indices.

    Raises:
        ValueError: If parameters are inconsistent or do not allow for any splits.
    """
    if num_observations <= 0:
        raise ValueError("num_observations must be positive.")
    if test_window_size <= 0:
        raise ValueError("test_window_size must be positive.")
    if train_window_size <= 0 : # train_window_size must be positive for fixed, and for initial in expanding
        raise ValueError("train_window_size must be positive.")


    indices = np.arange(num_observations)
    splits = []

    start_train_offset = 0

    if fixed_train_window:
        if train_window_size > num_observations:
             raise ValueError(f"train_window_size ({train_window_size}) cannot exceed num_observations ({num_observations}).")
        start_train_offset = train_window_size
    else: # Expanding window
        if initial_train_size is not None:
            if initial_train_size <= 0:
                raise ValueError("initial_train_size for expanding window must be positive.")
            if initial_train_size > num_observations:
                raise ValueError(f"initial_train_size ({initial_train_size}) cannot exceed num_observations ({num_observations}).")
            start_train_offset = initial_train_size
        else:
            # If initial_train_size is not given for expanding window,
            # train_window_size acts as the first training period's length.
            if train_window_size > num_observations:
                 raise ValueError(
                    f"train_window_size ({train_window_size}) as initial for expanding window "
                    f"cannot exceed num_observations ({num_observations})."
                )
            start_train_offset = train_window_size

    # current_pos is the first possible start of a test set
    current_pos = start_train_offset

    while current_pos + test_window_size <= num_observations:
        test_start_index = current_pos
        test_end_index = current_pos + test_window_size

        if fixed_train_window:
            # Rolling window: train_start_index moves with test_start_index
            train_start_index = test_start_index - train_window_size
        else: # Expanding window
            train_start_index = 0 # Expanding window always starts from index 0

        train_set = indices[train_start_index : test_start_index]
        test_set = indices[test_start_index : test_end_index]

        if len(train_set) == 0:
            # This should only happen if initial_train_size (or train_window_size for fixed)
            # is such that the first test window starts at index 0, leaving no room for training.
            # The initial current_pos calculation should prevent this.
            # If current_pos (test_start_index) is 0, train_set is empty.
            # This implies start_train_offset was 0, which is prevented by checks.
            raise ValueError(
                "A split resulted in an empty training set. "
                "This should not happen with valid initial parameters. "
                f"test_start_index: {test_start_index}, train_start_index: {train_start_index}"
            )

        splits.append((train_set, test_set))
        current_pos += test_window_size # Move to the next potential test window start

    # No specific error if no splits are formed, list will be empty.
    # This is valid if num_observations is too small for even one split.
    # e.g. num_obs = 10, train_window = 8, test_window = 3. current_pos = 8. 8+3 > 10. Loop doesn't run.
    return splits
