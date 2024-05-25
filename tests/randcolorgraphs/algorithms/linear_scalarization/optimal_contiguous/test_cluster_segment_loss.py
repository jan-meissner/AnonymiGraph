import numpy as np

from randcolorgraphs.algorithms.linear_scalarization.compute_objective import (
    dense_compute_linear_scalarization_objective,
)
from randcolorgraphs.algorithms.linear_scalarization.optimal_contiguous.cluster_segment_loss import (
    _compute_cluster_cost_projection_parallel,
    _compute_cluster_cost_projection_perpendicular,
    get_cluster_segment_cost,
)
from randcolorgraphs.utils.calculate_katz import calculate_katz


def compute_sum_of_squares_naive(X):
    n, n_feats = X.shape
    S = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            segment = X[i : j + 1, :]
            sum_of_squares = 0
            for feat in range(n_feats):
                mean = np.mean(segment[:, feat])
                sum_of_squares += np.sum((segment[:, feat] - mean) ** 2)
            S[i][j] = sum_of_squares

    return S


def test_cluster_costs():
    np.random.seed(33242)
    X = np.random.rand(240, 15)

    S = _compute_cluster_cost_projection_perpendicular(X)  # Time-Complexity O(n^3)
    S_naive = compute_sum_of_squares_naive(X)  # if X is a nxn matrix: Time-Complexity O(n^4)

    assert np.allclose(S, S_naive)

    S_parallel = _compute_cluster_cost_projection_parallel(X)

    # Check Pythagoras for Frobenius norm
    assert np.allclose(S_parallel[0, len(S_parallel) - 1] + S[0, len(S_parallel) - 1], np.sum(X**2))


def test_get_cluster_segment_cost():
    n = 199
    p = 2 / n

    np.random.seed(33242)
    A = np.random.binomial(1, p, (n, n))
    katz_centrality = calculate_katz(A, alpha=0.3)
    w = np.exp(-5)

    # Sort katz centrality and A
    sorted_indices = np.argsort(katz_centrality)
    sorted_x = katz_centrality[sorted_indices]
    sorted_A = A[:, sorted_indices][sorted_indices, :]
    S = get_cluster_segment_cost(sorted_x, sorted_A, w)

    for i in range(0, len(sorted_x) - 1):
        np.allclose(
            S[0, i] + S[i + 1, len(sorted_x) - 1],
            dense_compute_linear_scalarization_objective(
                sorted_x, sorted_A, np.array([1] * (i + 1) + [0] * (len(sorted_x) - i - 1)), w
            ),
        )
