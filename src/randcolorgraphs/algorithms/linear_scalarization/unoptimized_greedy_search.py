import numpy as np
from numba import njit, types
from numba.typed import Dict, List

from randcolorgraphs.algorithms.linear_scalarization.compute_objective import (
    dense_compute_linear_scalarization_objective,
)

array_type = types.int64[:]


@njit
def _get_sorted_unique_clusters(current_clusters):
    cluster_to_index_of_unique_cluster_sorted = Dict.empty(
        key_type=types.int64,
        value_type=types.int64,
    )

    unique_cluster_sorted = List.empty_list(types.int64)

    i = 0
    for c in current_clusters:
        if c not in cluster_to_index_of_unique_cluster_sorted:
            cluster_to_index_of_unique_cluster_sorted[c] = i
            unique_cluster_sorted.append(c)
            i += 1

    return unique_cluster_sorted, cluster_to_index_of_unique_cluster_sorted


@njit
def _generate_partial_merges(state):
    """
    Generate all possible partial merges between any two clusters,
    with one cluster being the "upper" cluster (having the biggest max index)
    and the other being the "lower" cluster.

    Parameters:
    state (np.ndarray): The current state of the clustering.

    Returns:
    list of np.ndarray: Each array represents a new state after a partial merge.
    """

    unique_cluster_sorted, _ = _get_sorted_unique_clusters(state)

    unbalanced_splits = []
    for i in range(len(unique_cluster_sorted) - 1):
        new_state = state.copy()
        # Merge Phase; Merge first and second cluster
        first_cluster = unique_cluster_sorted[i]
        second_cluster = unique_cluster_sorted[i + 1]
        new_state[new_state == second_cluster] = first_cluster

        # Split the clusters (with single difference that we now includes the 0vsn split)
        cluster_indices = np.where(new_state == first_cluster)[0]
        n = cluster_indices.size

        new_cluster_index = np.max(state) + 1
        if n > 1:
            for m in range(0, n):
                indices_to_split = cluster_indices[:m]
                new_split_state = new_state.copy()
                for idx in indices_to_split:
                    new_split_state[idx] = new_cluster_index

                # Avoid adding the original state
                n_c1_to_c2, n_c1_to_c1, n_c2_to_c1, n_c2_to_c2 = 0, 0, 0, 0
                for idx in cluster_indices:
                    if state[idx] == first_cluster:
                        if new_split_state[idx] == first_cluster:
                            n_c1_to_c1 += 1
                        else:
                            n_c1_to_c2 += 1
                    else:
                        if new_split_state[idx] == first_cluster:
                            n_c2_to_c1 += 1
                        else:
                            n_c2_to_c2 += 1
                if n_c1_to_c2 + n_c2_to_c1 == 0 or n_c1_to_c1 + n_c2_to_c2 == 0:
                    # new_split_state is state up to relabeling
                    continue
                else:
                    unbalanced_splits.append(new_split_state)

    return unbalanced_splits


@njit
def _generate_unbalanced_splits(state):
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
                # if np.random.uniform() > 0.1: continue
                indices_to_split = cluster_indices[:m]
                new_state = state.copy()
                new_cluster = np.max(state) + 1
                for idx in indices_to_split:
                    new_state[idx] = new_cluster
                unbalanced_splits.append(new_state)

    return unbalanced_splits


@njit
def _generate_pam_swaps(clusters, max_dist):
    """
    Generate all possible PAM swap moves.
    I.e. Instead of doing a swap simply reassign the cluster of only one node with the other one that is within max_dist.

    Parameters:
    state (np.ndarray): The current state of the clustering.

    Returns:
    list of np.ndarray: Each array represents a new state after a 1-swap.
    """
    swaps = []
    n = clusters.size
    unique_cluster_sorted, cluster_to_index_of_unique_cluster_sorted = _get_sorted_unique_clusters(clusters)
    num_clusters = len(unique_cluster_sorted)

    for i in range(n):
        c = clusters[i]

        c_idx = cluster_to_index_of_unique_cluster_sorted[c]
        assert unique_cluster_sorted[c_idx] == c

        for c_prime_idx in range(max(0, c_idx - max_dist), min(num_clusters, c_idx + max_dist + 1)):
            if c_prime_idx != c_idx:
                new_state = clusters.copy()
                new_state[i] = unique_cluster_sorted[c_prime_idx]
                swaps.append(new_state)

    return swaps


@njit
def unoptimized_greedy_search_linear_scalarization(v_K, A, inital_clusters, w, pam_cluster_dist=1):
    """
    It is recommended v_K is sorted. (And A also in that same node order)
    """
    assert inital_clusters.dtype == np.dtype(np.int64)
    current_clusters = inital_clusters
    best_state = current_clusters
    best_clusters_objective = dense_compute_linear_scalarization_objective(v_K, A, current_clusters, w)

    print("init obj:", best_clusters_objective)

    iteration = 0
    while True:
        best_next_clusters_objective = best_clusters_objective
        best_next_clusters = current_clusters
        best_move_type = ""

        # Go through all Unbalanced Split
        # for cluster in clusters: for possible unbalanced split in all possible splits: ... (update best_move and best_move_objective here)
        unbalanced_split_clusters = _generate_unbalanced_splits(current_clusters)
        for one_swap_cluster in unbalanced_split_clusters:
            obj_value = dense_compute_linear_scalarization_objective(v_K, A, one_swap_cluster, w)
            if obj_value < best_next_clusters_objective:
                best_next_clusters = one_swap_cluster
                best_next_clusters_objective = obj_value
                best_move_type = "split"

        # (SWAP) from PAM O(n*pam_cluster_dist)
        # swap_clusters = _generate_pam_swaps(current_clusters, pam_cluster_dist)
        # for swap_cluster in swap_clusters:
        #    obj_value = dense_compute_linear_scalarization_objective(v_K, A, swap_cluster, w)
        #    if obj_value < best_next_clusters_objective:
        #        best_next_clusters = swap_cluster
        #        best_next_clusters_objective = obj_value
        #        best_move_type = "swap"
        #
        # # Partial Merge;
        # # can be seen as a merge with an instantly followed unbalanced split, only adjacent clusters are merged
        # # correct bad initial splits
        partial_merge_clusters = _generate_partial_merges(current_clusters)
        for partial_merge_cluster in partial_merge_clusters:
            obj_value = dense_compute_linear_scalarization_objective(v_K, A, partial_merge_cluster, w)
            if obj_value < best_next_clusters_objective:
                best_next_clusters = partial_merge_cluster
                best_next_clusters_objective = obj_value
                best_move_type = "merge-split"

        # Go through all 2-Swaps
        # one_swap_clusters = generate_2_swaps(current_clusters, two_swap_max_dist)
        # for one_swap_cluster in one_swap_clusters:
        #     obj_value = compute_obj(v_K, A, one_swap_cluster, w)
        #     if obj_value < best_next_clusters_objective:
        #         best_next_clusters = one_swap_cluster
        #         best_next_clusters_objective = obj_value

        # best_next_state, best_next_state_objective
        if best_next_clusters_objective >= best_clusters_objective:
            break

        # Update the current state and the best objective found
        current_clusters = best_next_clusters
        best_state = current_clusters
        best_clusters_objective = best_next_clusters_objective

        print("Iteration", iteration, "Objective:", best_clusters_objective, "move_type", best_move_type)
        # if iteration > 0:
        #    break
        iteration += 1

    print("!!!!!!!!!!!!!!!!! RUNNING UNOPTIMIZED_GREEDY_SEARCH IN DEBUG MODE")
    return best_state, best_clusters_objective
