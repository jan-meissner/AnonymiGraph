import numpy as np


def _recurse_generate_clusters(node, current, n):
    #
    # !!! Copilot generated Code
    #
    if node > n:
        yield current
        return

    max_label = max(current) if current else 0
    used_labels = set(current)

    # Option 1: Assign to any existing cluster
    for label in used_labels:
        yield from _recurse_generate_clusters(node + 1, current + [label], n)

    # Option 2: Start a new cluster if possible
    if max_label < n:
        yield from _recurse_generate_clusters(node + 1, current + [max_label + 1], n)


def generate_clusters(n: int):
    """Efficiently generates all possible cluster assignments of size n"""
    return _recurse_generate_clusters(1, [], n=n)


def dense_compute_metrics(x, A, cluster_assignment):
    """
    Calculates ell and expected_edge_overlap in O(n^3) for a dense matrix A.
    """
    unique_clusters, cluster_indices = np.unique(cluster_assignment, return_inverse=True)
    H = np.identity(len(unique_clusters))[cluster_indices]

    HtHinv = np.linalg.inv(H.T @ H)

    ell = np.linalg.norm(x - H @ (HtHinv @ (H.T @ x)))

    N = A @ H  # O(n^3)
    expected_edge_overlap = np.trace(HtHinv @ (N.T @ N))
    return ell, expected_edge_overlap


def is_contiguous(x):
    return all(x[i] <= x[i + 1] for i in range(len(x) - 1))


def evaluate_cluster_assignment(A, cluster_assignment, katz_centrality):
    ell, expected_edge_overlap = dense_compute_metrics(katz_centrality, A, cluster_assignment)
    return [cluster_assignment, is_contiguous(cluster_assignment), ell, expected_edge_overlap]
