from numba import njit, types
from numba.typed import Dict, List

integer_list_type = types.ListType(types.int64)


@njit
def get_cluster_to_indices(cluster) -> dict:
    """
    Returns a dictionary mapping each cluster label to the indices of vertices inside the cluster.
    Returned indices are sorted in ascending order in the list.

    Args:
        cluster (np.ndarray): A 1D array of cluster labels where each element indicates the cluster
            label of the corresponding vertex.

    Returns:
        Dict[int, List[int]]: A dictionary where each key is a cluster label (int) and each value
            is a list of indices (List[int]) of vertices in that cluster. The indices in the list
            are sorted in ascending order.

    """
    cluster_to_indices = Dict.empty(
        key_type=types.int64,
        value_type=integer_list_type,
    )

    for idx, label in enumerate(cluster):
        if label not in cluster_to_indices:
            cluster_to_indices[label] = List.empty_list(types.int64)
        cluster_to_indices[label].append(idx)

    return cluster_to_indices
