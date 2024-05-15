import numpy as np
from numba import njit


@njit
def compute_obj(v_K, A, clusters, w):
    """
    Compute the objective function value given the cluster assignment.

    Parameters:
    v_K (np.ndarray): The input data matrix (n_samples, n_features).
    A (np.ndarray): The matrix A in the objective function (n_features, n_features).
    clusters (list or np.ndarray): Cluster assignments for each data point (n_samples,).
    w (float): The weight parameter.

    Returns:
    float: The value of the objective function.
    """
    unique_clusters = np.unique(clusters)

    # Compute the norms for v_K and A
    norm_v_K = np.sum(v_K**2)

    sum_HvK = 0
    sum_HAT = 0

    for cluster in unique_clusters:
        indices = np.where(clusters == cluster)[0]
        n_j = len(indices)
        assert n_j > 0

        v_K_cluster_sum = np.sum(v_K[indices], axis=0)
        A_cluster_sum = np.sum(A.T[indices, :], axis=0)
        sum_HvK += (v_K_cluster_sum**2) / n_j
        sum_HAT += (np.sum(A_cluster_sum**2)) / n_j
    obj_value = norm_v_K - sum_HvK + w * sum_HAT
    return obj_value


@njit
def generate_unbalanced_splits(state):
    """
    Generate all possible unbalanced splits for each cluster.

    Parameters:
    state (np.ndarray): The current state of the clustering.

    Returns:
    list of np.ndarray: Each array represents a new state after an unbalanced split.
    """
    unique_clusters = np.unique(state)
    unbalanced_splits = []

    for cluster in unique_clusters:
        cluster_indices = np.where(state == cluster)[0]
        n = cluster_indices.size

        if n > 1:
            for m in range(1, n):
                indices_to_split = cluster_indices[:m]
                new_state = state.copy()
                new_cluster = np.max(state) + 1
                for idx in indices_to_split:
                    new_state[idx] = new_cluster
                unbalanced_splits.append(new_state)

    return unbalanced_splits


@njit
def generate_2_swaps(clusters, max_dist):
    """
    Generate all possible 2-swap moves.

    Parameters:
    state (np.ndarray): The current state of the clustering.
    max_dist (int): The maximum distance between two points that are swapped as defined by the order of points in `clusters`

    Returns:
    list of np.ndarray: Each array represents a new state after a 2-swap.
    """
    swaps = []
    n = clusters.size

    for i in range(n):
        for j in range(max(i - max_dist + 1, 0), min(i + max_dist, n)):
            if clusters[i] != clusters[j]:
                new_state = clusters.copy()
                new_state[i], new_state[j] = new_state[j], new_state[i]
                swaps.append(new_state)

    return swaps


@njit
def generate_1_swaps(clusters, max_dist):
    """
    Generate all possible 1-swap moves.
    I.e. Instead of doing a swap simply reassign the cluster of only one node with the other one that is within max_dist.

    Parameters:
    state (np.ndarray): The current state of the clustering.

    Returns:
    list of np.ndarray: Each array represents a new state after a 1-swap.
    """
    swaps = []
    n = clusters.size

    for i in range(n):
        for j in range(max(i - max_dist + 1, 0), min(i + max_dist, n)):
            if clusters[i] != clusters[j]:
                new_state = clusters.copy()
                new_state[i] = new_state[j]
                swaps.append(new_state)

    return swaps


@njit
def greedy_algorithm_inefficient(v_K, A, inital_clusters, w, two_swap_max_dist, one_swap_max_dist):
    """
    It is recommended v_K is sorted. (And A also in that same node order)
    """
    current_clusters = inital_clusters
    best_state = current_clusters
    best_clusters_objective = compute_obj(v_K, A, current_clusters, w)

    while True:
        best_next_clusters_objective = best_clusters_objective
        best_next_clusters = current_clusters

        # Go through all Unbalanced Split
        # for cluster in clusters: for possible unbalanced split in all possible splits: ... (update best_move and best_move_objective here)
        unbalanced_split_clusters = generate_unbalanced_splits(current_clusters)
        for one_swap_cluster in unbalanced_split_clusters:
            obj_value = compute_obj(v_K, A, one_swap_cluster, w)
            if obj_value < best_next_clusters_objective:
                best_next_clusters = one_swap_cluster
                best_next_clusters_objective = obj_value

        # Go through all 2-Swaps
        one_swap_clusters = generate_2_swaps(current_clusters, two_swap_max_dist)
        for one_swap_cluster in one_swap_clusters:
            obj_value = compute_obj(v_K, A, one_swap_cluster, w)
            if obj_value < best_next_clusters_objective:
                best_next_clusters = one_swap_cluster
                best_next_clusters_objective = obj_value

        # Go through all 1-Swaps
        one_swap_clusters = generate_1_swaps(current_clusters, one_swap_max_dist)
        for one_swap_cluster in one_swap_clusters:
            obj_value = compute_obj(v_K, A, one_swap_cluster, w)
            if obj_value < best_next_clusters_objective:
                best_next_clusters = one_swap_cluster
                best_next_clusters_objective = obj_value

        # best_next_state, best_next_state_objective
        if best_next_clusters_objective >= best_clusters_objective:
            break

        # Update the current state and the best objective found
        current_clusters = best_next_clusters
        best_state = current_clusters
        best_clusters_objective = best_next_clusters_objective

    return best_state, best_clusters_objective
