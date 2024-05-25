import numpy as np
from numba import njit, types
from numba.typed import Dict


@njit
def get_scalar_cluster_sums(x: np.ndarray, cluster_to_indices):
    """
    Compute the sum of elements in `x` for each cluster in `cluster_to_indices`.

    Args:
        x (np.ndarray): 1D array of numerical values.
        cluster_to_indices (Dict[int, List[int]]): Dictionary mapping cluster labels to lists of indices,
            indicating which points in `x` belong to which cluster.

    Returns:
        Dict[int, float]: Dictionary mapping cluster labels to the sum of elements in `x` for each cluster.
    """
    S_scalar = Dict.empty(
        key_type=types.int64,
        value_type=types.float64,
    )
    for c in cluster_to_indices:
        for i in cluster_to_indices[c]:
            if c not in S_scalar:
                S_scalar[c] = x[i]
            else:
                S_scalar[c] += x[i]

    return S_scalar
