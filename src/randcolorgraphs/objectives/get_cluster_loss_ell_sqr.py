import numpy as np
from numba import njit

from randcolorgraphs.objectives.get_cluster_to_indices import get_cluster_to_indices
from randcolorgraphs.objectives.get_scalar_cluster_sums import get_scalar_cluster_sums


@njit
def get_cluster_loss_ell_sqr(x, clusters):
    """
    Compute the cluster loss `ell` (WCSS) to the power of 2 for the given clusters and data points.

    Args:
        x (np.ndarray): 1D array of numerical values associated with the data points.
        clusters (np.ndarray): 1D array of cluster labels where each element indicates the cluster label of the corresponding point.

    Returns:
        float: The computed loss `ell` to the power of 2.
    """
    cluster_to_indices = get_cluster_to_indices(clusters)  # O(n)
    Sx = get_scalar_cluster_sums(x, cluster_to_indices)  # O(n)

    ell_sqr = np.sum(x**2)
    for c in cluster_to_indices:
        n_cluster = len(cluster_to_indices[c])
        ell_sqr += -Sx[c] ** 2 / n_cluster

    return ell_sqr
