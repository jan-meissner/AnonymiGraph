import numpy as np
from numba import njit


@njit
def dense_compute_linear_scalarization_objective(x, A, clusters, w):
    """
    Efficient dense computation of the linear scalarization  objective function value given the cluster assignment.

    Doesn't consider sparsity of A and thus is O(n^2).

    Parameters:
    x (np.ndarray): The centrality to preserve (n_samples, n_features).
    A (np.ndarray): The dense adjacency matrix (n_features, n_features).
    clusters (list or np.ndarray): Cluster assignments for each data point (n_samples,).
    w (float): The scalarization weight parameter.

    Returns:
    float: The value of the objective function.
    """
    unique_clusters = np.unique(clusters)

    # Compute the norms for x and A
    norm_x = np.sum(x**2)

    sum_Hx = 0
    sum_HAT = 0

    for cluster in unique_clusters:
        indices = np.where(clusters == cluster)[0]
        n = len(indices)

        x_cluster_sum = np.sum(x[indices], axis=0)
        A_cluster_sum = np.sum(A.T[indices, :], axis=0)
        sum_Hx += (x_cluster_sum**2) / n
        sum_HAT += (np.sum(A_cluster_sum**2)) / n

    return norm_x - sum_Hx + w * sum_HAT
