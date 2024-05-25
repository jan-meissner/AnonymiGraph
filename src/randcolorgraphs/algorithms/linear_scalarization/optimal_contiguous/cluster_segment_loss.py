import numpy as np
from numba import njit


@njit
def _concatenate_arrays_axis0(arr1: np.ndarray, arr2: np.ndarray):
    # !!!!
    # !!!! Written by Copilot
    # !!!

    rows1, cols1 = arr1.shape
    rows2, _ = arr2.shape

    result = np.empty((rows1 + rows2, cols1), dtype=arr1.dtype)
    result[:rows1, :] = arr1
    result[rows1:, :] = arr2

    return result


@njit
def _cum_sum_numba_axis0(arr: np.ndarray):
    # !!!!
    # !!!! Written by Copilot
    # !!!!

    cum_sum = np.empty_like(arr)
    cum_sum[0, :] = arr[0, :]
    for i in range(1, arr.shape[0]):
        cum_sum[i, :] = cum_sum[i - 1, :] + arr[i, :]
    return cum_sum


@njit
def _compute_cluster_cost_projection_perpendicular(X: np.ndarray):
    n, n_feats = X.shape
    S = np.zeros((n, n))

    pre_sum = _concatenate_arrays_axis0(np.zeros((1, n_feats)), X)
    cum_sum = _cum_sum_numba_axis0(pre_sum)
    pre_sum_sq = _concatenate_arrays_axis0(np.zeros((1, n_feats)), X**2)
    cum_sum_squares = _cum_sum_numba_axis0(pre_sum_sq)

    for i in range(n):
        for j in range(i, n):
            num_elements = j - i + 1

            segment_sum_squares = 0
            for idx_feat in range(n_feats):
                segment_sum = cum_sum[j + 1, idx_feat] - cum_sum[i, idx_feat]
                segment_sum_squares_feat = cum_sum_squares[j + 1, idx_feat] - cum_sum_squares[i, idx_feat]

                segment_sum_squares += segment_sum_squares_feat - (segment_sum**2) / num_elements

            S[i][j] = segment_sum_squares

    return S


@njit
def _compute_cluster_cost_projection_parallel(X: np.ndarray):
    n, n_feats = X.shape
    S = np.zeros((n, n))

    pre_sum = _concatenate_arrays_axis0(np.zeros((1, n_feats)), X)
    cum_sum = _cum_sum_numba_axis0(pre_sum)

    for i in range(n):
        for j in range(i, n):
            num_elements = j - i + 1

            total_segment_sum = 0
            for idx_feat in range(n_feats):
                segment_sum = cum_sum[j + 1, idx_feat] - cum_sum[i, idx_feat]
                total_segment_sum += (segment_sum**2) / num_elements

            S[i][j] = total_segment_sum

    return S


@njit
def get_cluster_segment_cost(sorted_x, sorted_A, w):
    """
    x,A both sorted by x ascending.
    """
    assert np.array_equal(sorted_x, np.sort(sorted_x)), "x is not sorted!"

    Sx = _compute_cluster_cost_projection_perpendicular(sorted_x.reshape(-1, 1))
    SA = _compute_cluster_cost_projection_parallel(sorted_A.T)

    return Sx + w * SA
