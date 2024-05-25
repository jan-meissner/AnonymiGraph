import numpy as np

from randcolorgraphs.algorithms.linear_scalarization.compute_objective import (
    dense_compute_linear_scalarization_objective,
)


def naive_loss(x, A, clusters, w):
    """
    The brute force naive implementation. O(n^3)
    """

    n = len(clusters)

    unique_clusters, cluster_indices = np.unique(clusters, return_inverse=True)
    H = np.identity(len(unique_clusters))[cluster_indices]

    HtHinv = np.linalg.inv(H.T @ H)
    Ppar = H @ HtHinv @ H.T

    Id = np.identity(n)
    ell = np.linalg.norm((Id - Ppar) @ x) ** 2

    # A @ Ppar!!! Not Ppar @ A
    expected_edge_overlap = np.linalg.norm(A @ Ppar, "fro") ** 2
    return ell + w * expected_edge_overlap


def test_dense_compute_linear_scalarization_objective():
    np.random.seed(33242)

    v_K = np.random.rand(10)
    A = np.random.rand(10, 10)
    clusters = np.array([0, 0, 1, 1, 1, 2, 2, 0, 2, 1])

    w = 0.5

    assert np.allclose(
        dense_compute_linear_scalarization_objective(v_K, A, clusters, w), naive_loss(v_K, A, clusters, w)
    )
