import logging
import time

import numpy as np
from numba import njit, types
from numba.typed import Dict, List

from randcolorgraphs.objectives.get_adjacency_dict import get_adjacency_dict
from randcolorgraphs.objectives.get_cluster_to_indices import get_cluster_to_indices
from randcolorgraphs.objectives.numba_sparse_vector_ops import (
    inplace_add_sparse_vecs,
    inplace_subtract_sparse_vecs,
)

logger = logging.getLogger(__name__)

#######################
# Generic Type Definitions
#######################

sparse_int_vector_type = types.DictType(types.int64, types.int64)
int_list_type = types.ListType(types.int64)

#######################
# cluster states
#######################
# Define the type for the sparse int vector

# Define a simple enum mapping for state components
# Define the type for the state tuple
STATE_N, STATE_SX, STATE_P_POWER_SUM_SAT, STATE_SAT, STATE_CLUSTER_INDICES = 0, 1, 2, 3, 4
cluster_state_type = types.Tuple((types.int64, types.float64, types.float64, sparse_int_vector_type, int_list_type))


@njit
def create_empty_cluster_states():
    return Dict.empty(key_type=types.int64, value_type=cluster_state_type)


@njit
def create_empty_sparse_int_vector():
    """
    Currently using an Dict instead of an actual sparse vector, we do know that the keys in the dict are int from 0...n-1
    so this is a bit inefficient but it's ok for now
    """
    return Dict.empty(key_type=types.int64, value_type=types.int64)


@njit
def create_cluster_state(n, Sx, p_power_sum_SAt, SAt, cluster_indices):
    """
    For improved readability
    """
    return (n, Sx, p_power_sum_SAt, SAt, cluster_indices)


@njit
def get_cluster_state(cluster_indices, x, At_sparse, p=2):
    # Check that the cluster_indices are sorted
    for i in range(1, len(cluster_indices)):
        if cluster_indices[i - 1] > cluster_indices[i]:
            raise ValueError("cluster_indices must be sorted in ascending order")

    # n
    n = len(cluster_indices)

    # Sx (With outer loops: O(|V|))
    Sx = 0.0
    for i in cluster_indices:
        Sx += x[i]

    # SAt (With outer loops: O(|E|))
    SAt = Dict.empty(key_type=types.int64, value_type=types.int64)
    for i in cluster_indices:
        inplace_add_sparse_vecs(SAt, At_sparse[i])

    # p_power_sum_SAt (With outer loops: O(|E|))
    p_power_sum_SAt = 0.0
    for j in SAt:
        p_power_sum_SAt += abs(SAt[j]) ** p

    return create_cluster_state(n=n, Sx=Sx, p_power_sum_SAt=p_power_sum_SAt, SAt=SAt, cluster_indices=cluster_indices)


@njit
def get_initial_cluster_state(inital_clusters, x, At_sparse, p=2):
    cluster_states = create_empty_cluster_states()

    for c, cluster_indices in get_cluster_to_indices(inital_clusters).items():
        cluster_states[c] = get_cluster_state(cluster_indices, x, At_sparse, p=p)

    return cluster_states


#######################
# Compute Delta Objective
#######################


@njit
def compute_delta_obj(
    S_j_prime_p_power_sum,
    S_k_prime_p_power_sum,
    n_j_prime,
    n_k_prime,  # new cluster states
    S_j_p_power_sum,
    S_k_p_power_sum,
    n_j,
    n_k,  # old cluster states
):
    """Calculate the objective if clusters j and k are replaced by j_prime and k_prime based on their cluster states"""
    partial_objective_before = S_j_p_power_sum / n_j + S_k_p_power_sum / n_k
    partial_objective_after = S_j_prime_p_power_sum / n_j_prime + S_k_prime_p_power_sum / n_k_prime
    return partial_objective_after - partial_objective_before


#######################
# Interactions
#######################


@njit
def get_add_change_in_p_power_sum(potential_dense_vec, sparse_vec, p=2):
    change = 0.0
    for key, add_value in sparse_vec.items():
        if key in potential_dense_vec:
            base_value = potential_dense_vec[key]
        else:
            base_value = 0.0

        change += (add_value + base_value) ** p - base_value**p

    return change


@njit
def get_subtract_change_in_p_power_sum(potential_dense_vec, sparse_vec, p=2):
    change = 0.0
    for key, sub_value in sparse_vec.items():
        if key in potential_dense_vec:
            base_value = potential_dense_vec[key]
        else:
            base_value = 0.0

        change += (base_value - sub_value) ** p - base_value**p

    return change


# Define the type for the state tuple
MOVE_NONE, MOVE_SPLIT, MOVE_MERGE_SPLIT, MOVE_SWAP = -1, 0, 1, 2
INTER_DELTA_OBJ, INTER_CLUSTER_UPDATE, INTER_MOVE_TYPE = 0, 1, 2
interaction_type = types.Tuple((types.float64, sparse_int_vector_type, types.int64))
int_two_tuple = types.Tuple((types.int64, types.int64))


@njit
def create_interaction_move(delta_obj, cluster_update, move_type):
    """
    For improved readability
    """
    return (delta_obj, cluster_update, move_type)


@njit
def create_empty_one_interactions():
    return Dict.empty(key_type=types.int64, value_type=interaction_type)


@njit
def create_empty_two_interactions():
    return Dict.empty(key_type=int_two_tuple, value_type=interaction_type)


@njit
def move_split(cluster_indices, Sx, SAt, p_power_sum_SAt, x, At_sparse, w, p=2):
    """Calculates the best way to split a cluster into two clusters
    All splits are contiguous in the katz centrality i.e. only consider splitting

    Important: x[cluster_indices] must be sorted in ascending order!
    """
    assert len(cluster_indices) > 0

    best_cluster_update = Dict.empty(key_type=types.int64, value_type=types.int64)
    best_delta_obj = 0

    # Notation: clusters j and k are transformed into j_prime and k_prime (here k is empty so omitted)
    n_j = len(cluster_indices)
    Sx_j = Sx
    SAt_j = SAt
    p_power_sum_SAt_j = p_power_sum_SAt

    n_j_prime = n_j
    n_k_prime = 0

    Sx_j_prime = Sx_j
    Sx_k_prime = 0

    SAt_j_prime = SAt_j.copy()
    SAt_k_prime = Dict.empty(key_type=types.int64, value_type=types.int64)

    # Further speed up; avoid recalculating the power sum instead which might be O(n)
    p_power_sum_SAt_j_prime = p_power_sum_SAt
    p_power_sum_SAt_k_prime = 0.0

    best_loop_idx = -1  # -1 indicates no improvement found at all

    for loop_idx in range(len(cluster_indices) - 1):
        # Check the cluster splitting of the first (sorted by x) loop_idx+1 many in the first cluster and all other in second cluster
        i = cluster_indices[loop_idx]

        n_j_prime = n_j_prime - 1
        n_k_prime = n_k_prime + 1

        #########
        # Calculate x objective loss component, Complexity with outer for loops: O(n)
        #########
        Sx_j_prime = Sx_j_prime - x[i]
        Sx_k_prime = Sx_k_prime + x[i]

        delta_obj_x = compute_delta_obj(
            Sx_j_prime**2, Sx_k_prime**2, n_j_prime, n_k_prime, Sx_j**2, 0, n_j, 1
        )  # Sx_k is 0 so n_k can be arbitrary

        #########
        # Calculate A^t objective loss component, Complexity with outer for loops: O(|E|)
        #########
        p_power_sum_SAt_j_prime += get_subtract_change_in_p_power_sum(SAt_j_prime, At_sparse[i], p=p)
        p_power_sum_SAt_k_prime += get_add_change_in_p_power_sum(SAt_k_prime, At_sparse[i], p=p)

        inplace_subtract_sparse_vecs(SAt_j_prime, At_sparse[i])
        inplace_add_sparse_vecs(SAt_k_prime, At_sparse[i])

        # EXTREMELY SLOW ASSERT REMOVE WHEN RUNNING SERIOUSLY
        # assert np.allclose(sparse_p_power_sum(SAt_j_prime), p_power_sum_SAt_j_prime)
        # assert np.allclose(sparse_p_power_sum(SAt_k_prime), p_power_sum_SAt_k_prime)

        delta_obj_A = compute_delta_obj(
            p_power_sum_SAt_j_prime, p_power_sum_SAt_k_prime, n_j_prime, n_k_prime, p_power_sum_SAt_j, 0, n_j, 1
        )  # SA_k is 0 so n_k can be arbitrary

        delta_obj = -delta_obj_x + w * delta_obj_A
        if delta_obj < best_delta_obj:
            best_delta_obj = delta_obj
            best_loop_idx = loop_idx

    # assert np.allclose(Sx_j_prime, x[cluster_indices[len(cluster_indices) - 1]])

    # Found a split that was better than currently best
    if best_loop_idx > -1:
        best_cluster_update = Dict.empty(key_type=types.int64, value_type=types.int64)
        for j in range(best_loop_idx + 1):
            best_cluster_update[
                cluster_indices[j]
            ] = -1  # -1 is convention for new cluster actual positive label will be inferred dynamically

    return best_delta_obj, best_cluster_update


@njit
def calc_best_one_move(cluster_state, x, At_sparse, w, p=2):
    best_delta_obj, best_cluster_update = move_split(
        cluster_state[STATE_CLUSTER_INDICES],
        cluster_state[STATE_SX],
        cluster_state[STATE_SAT],
        cluster_state[STATE_P_POWER_SUM_SAT],
        x,
        At_sparse,
        w,
        p=p,
    )

    return create_interaction_move(delta_obj=best_delta_obj, cluster_update=best_cluster_update, move_type=MOVE_SPLIT)


@njit
def merge_cluster_states(cluster_state1, cluster_state2, p=2):
    # Calculate the merged cluster indices
    merged_cluster_indices = List.empty_list(types.int64)
    for i in cluster_state1[STATE_CLUSTER_INDICES]:
        merged_cluster_indices.append(i)
    for i in cluster_state2[STATE_CLUSTER_INDICES]:
        merged_cluster_indices.append(i)
    merged_cluster_indices.sort()

    # Calculate the merged SAt
    merged_SAt = cluster_state1[STATE_SAT].copy()
    inplace_add_sparse_vecs(merged_SAt, cluster_state2[STATE_SAT])

    # Calculate the merged_p_power_sum_SAt
    merged_p_power_sum_SAt = 0.0
    for j in merged_SAt:
        merged_p_power_sum_SAt += abs(merged_SAt[j]) ** p

    # Calculate the merged p_power_sum_SAt
    merged_Sx = cluster_state1[STATE_SX] + cluster_state2[STATE_SX]
    merged_N = cluster_state1[STATE_N] + cluster_state2[STATE_N]

    return create_cluster_state(
        n=merged_N,
        Sx=merged_Sx,
        p_power_sum_SAt=merged_p_power_sum_SAt,
        SAt=merged_SAt,
        cluster_indices=merged_cluster_indices,
    )


@njit
def move_merge_split(cluster_state1, c1, cluster_state2, c2, x, At_sparse, w, p=2):
    merged_cluster = merge_cluster_states(cluster_state1, cluster_state2, p=p)

    best_split_delta_obj, cluster_update = move_split(
        merged_cluster[STATE_CLUSTER_INDICES],
        merged_cluster[STATE_SX],
        merged_cluster[STATE_SAT],
        merged_cluster[STATE_P_POWER_SUM_SAT],
        x,
        At_sparse,
        w,
        p=p,
    )

    # The delta_obj improvement from the move split is relative to the merged cluster so we need to account for that by subtracting the delta_obj of the two unmerged clusters
    # Notice the delta_obj can be now positive but all downstream checks always init the best_delta_obj as 0 so if this is positive it will be ignored
    # TODO This is slightly inefficient because it can be cached across run, but doesn't cause complexity to increase
    current_split_delta_obj = (
        get_objective_contributation_of_cluster(cluster_state1, w, p=p)
        + get_objective_contributation_of_cluster(cluster_state2, w, p=p)
        - get_objective_contributation_of_cluster(merged_cluster, w, p=p)
    )
    delta_obj = best_split_delta_obj - current_split_delta_obj

    # Also consider the full merge as a move here; It's equivalent to best_split_delta_obj = 0 and
    # cluster update = empty, as the below code will then assign c2 to all points
    if -current_split_delta_obj < delta_obj:
        delta_obj = -current_split_delta_obj
        cluster_update = Dict.empty(key_type=types.int64, value_type=types.int64)
        print("!!!")
        print("TODO DID A FULL MERGE: THIS CODE ISN'T WELL TESTED CLUSTERS ARE ALMOST NEVER EXECUTED!!!")
        print("!!!")

    # As the move_split uses -1 as a convention for the new clusters we replace it with c1 to avoid adding a new cluster

    # Due to floating point in the delta_obj we need to check if the new clusters are actually different we can do this
    # by calculating the 2x2 confusion matrix and checking if one of the diagonals sums to 0
    n_c1_to_c2, n_c1_to_c1, n_c2_to_c1, n_c2_to_c2 = 0, 0, 0, 0

    for i in cluster_state1[STATE_CLUSTER_INDICES]:
        if i in cluster_update:
            n_c1_to_c1 += 1
            cluster_update[i] = c1
        else:
            n_c1_to_c2 += 1
            cluster_update[i] = c2

    for i in cluster_state2[STATE_CLUSTER_INDICES]:
        if i in cluster_update:
            n_c2_to_c1 += 1
            cluster_update[i] = c1
        else:
            n_c2_to_c2 += 1
            cluster_update[i] = c2

    # The clusters didn't change up to relabeling; in order to avoid delta_obj < 0 floating errors force to return delta_obj = 0
    if n_c1_to_c2 + n_c2_to_c1 == 0 or n_c1_to_c1 + n_c2_to_c2 == 0:
        return 0, Dict.empty(key_type=types.int64, value_type=types.int64)
    else:
        return delta_obj, cluster_update


@njit
def move_swap(cluster_state1, c1, cluster_state2, c2, x, At_sparse, w, p=2):
    best_cluster_update = Dict.empty(key_type=types.int64, value_type=types.int64)
    best_delta_obj = 0

    best_swap_index = -1  # -1 indicates no improvement found at all
    best_cluster_to_swap_to = -1

    # Move points from c1 to c2
    if cluster_state1[STATE_N] > 1:  # Ignore Swaps that remove clusters
        for i in cluster_state1[STATE_CLUSTER_INDICES]:
            delta_obj_x = compute_delta_obj(
                (cluster_state1[STATE_SX] - x[i]) ** p,
                (cluster_state2[STATE_SX] + x[i]) ** p,
                cluster_state1[STATE_N] - 1,
                cluster_state2[STATE_N] + 1,
                cluster_state1[STATE_SX] ** p,
                cluster_state2[STATE_SX] ** p,
                cluster_state1[STATE_N],
                cluster_state2[STATE_N],
            )
            delta_obj_A = compute_delta_obj(
                cluster_state1[STATE_P_POWER_SUM_SAT]
                + get_subtract_change_in_p_power_sum(cluster_state1[STATE_SAT], At_sparse[i], p=p),
                cluster_state2[STATE_P_POWER_SUM_SAT]
                + get_add_change_in_p_power_sum(cluster_state2[STATE_SAT], At_sparse[i], p=p),
                cluster_state1[STATE_N] - 1,
                cluster_state2[STATE_N] + 1,
                cluster_state1[STATE_P_POWER_SUM_SAT],
                cluster_state2[STATE_P_POWER_SUM_SAT],
                cluster_state1[STATE_N],
                cluster_state2[STATE_N],
            )

            delta_obj = -delta_obj_x + w * delta_obj_A
            if delta_obj < best_delta_obj:
                best_delta_obj = delta_obj
                best_swap_index = i
                best_cluster_to_swap_to = c2

    # Move points from c2 to c1
    if cluster_state2[STATE_N] > 1:  # Ignore Swaps that remove clusters
        for i in cluster_state2[STATE_CLUSTER_INDICES]:
            delta_obj_x = compute_delta_obj(
                (cluster_state1[STATE_SX] + x[i]) ** p,
                (cluster_state2[STATE_SX] - x[i]) ** p,
                cluster_state1[STATE_N] + 1,
                cluster_state2[STATE_N] - 1,
                cluster_state1[STATE_SX] ** p,
                cluster_state2[STATE_SX] ** p,
                cluster_state1[STATE_N],
                cluster_state2[STATE_N],
            )
            delta_obj_A = compute_delta_obj(
                cluster_state1[STATE_P_POWER_SUM_SAT]
                + get_add_change_in_p_power_sum(cluster_state1[STATE_SAT], At_sparse[i], p=p),
                cluster_state2[STATE_P_POWER_SUM_SAT]
                + get_subtract_change_in_p_power_sum(cluster_state2[STATE_SAT], At_sparse[i], p=p),
                cluster_state1[STATE_N] + 1,
                cluster_state2[STATE_N] - 1,
                cluster_state1[STATE_P_POWER_SUM_SAT],
                cluster_state2[STATE_P_POWER_SUM_SAT],
                cluster_state1[STATE_N],
                cluster_state2[STATE_N],
            )

            delta_obj = -delta_obj_x + w * delta_obj_A
            if delta_obj < best_delta_obj:
                best_delta_obj = delta_obj
                best_swap_index = i
                best_cluster_to_swap_to = c1

    # Found a split that was better than currently best
    if best_swap_index > -1:
        best_cluster_update[best_swap_index] = best_cluster_to_swap_to

    return best_delta_obj, best_cluster_update


@njit
def calc_best_two_move(cluster_state1, c1, cluster_state2, c2, x, At_sparse, w, p=2):
    best_cluster_update = Dict.empty(key_type=types.int64, value_type=types.int64)
    best_delta_obj = 0
    best_move_type = MOVE_NONE

    # logger.info("Calculating a merge split move")
    merge_split_delta_obj, merge_split_cluster_update = move_merge_split(
        cluster_state1, c1, cluster_state2, c2, x, At_sparse, w, p=p
    )
    if merge_split_delta_obj < best_delta_obj:
        best_delta_obj = merge_split_delta_obj
        best_cluster_update = merge_split_cluster_update
        best_move_type = MOVE_MERGE_SPLIT

    swap_delta_obj, swap_cluster_update = move_swap(cluster_state1, c1, cluster_state2, c2, x, At_sparse, w, p=p)
    if swap_delta_obj < best_delta_obj:
        best_delta_obj = swap_delta_obj
        best_cluster_update = swap_cluster_update
        best_move_type = MOVE_SWAP

    return create_interaction_move(
        delta_obj=best_delta_obj, cluster_update=best_cluster_update, move_type=best_move_type
    )


@njit
def get_changed_clusters_to_indices(curr_cluster_states, best_cluster_update, curr_clusters):
    changed_clusters = set()
    for i in best_cluster_update:
        changed_clusters.add(best_cluster_update[i])
        changed_clusters.add(curr_clusters[i])

    changed_clusters_to_indices = Dict.empty(key_type=types.int64, value_type=int_list_type)

    # Ensure all changed_clusters are keyed so we can recover deleted clusters
    for c in changed_clusters:
        changed_clusters_to_indices[c] = List.empty_list(types.int64)

    # Gather new cluster indices for changed_clusters
    for c in changed_clusters:
        # If c is not a newly added cluster
        if c in curr_cluster_states:
            cluster_indices = curr_cluster_states[c][STATE_CLUSTER_INDICES]
            for i in cluster_indices:
                # Use updated cluster label
                if i in best_cluster_update:
                    changed_clusters_to_indices[best_cluster_update[i]].append(i)

                # Otherwise use old label
                else:
                    changed_clusters_to_indices[curr_clusters[i]].append(i)

    # Sort the cluster_indices as required but the move functions
    for c in changed_clusters:
        changed_clusters_to_indices[c].sort()

    return changed_clusters_to_indices


@njit
def get_objective_contributation_of_cluster(cluster_state, w, p=2):
    n_j = cluster_state[STATE_N]
    Sx = cluster_state[STATE_SX]
    p_power_sum_SAt = cluster_state[STATE_P_POWER_SUM_SAT]
    return -(Sx**p) / n_j + w * p_power_sum_SAt / n_j


@njit
def get_objective(curr_cluster_states, x, w, p=2):
    obj = np.sum(x**p)
    for _, cluster_state in curr_cluster_states.items():
        obj += get_objective_contributation_of_cluster(cluster_state, w, p=p)

    return obj


@njit
def get_expected_edge_overlap(curr_cluster_states):
    expected_edge_overlap = 0.0
    for cluster_state in curr_cluster_states.values():
        expected_edge_overlap += cluster_state[STATE_P_POWER_SUM_SAT] / cluster_state[STATE_N]
    return expected_edge_overlap


@njit
def get_sorted_unique_clusters(current_clusters):
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


# @njit #(parallel=True)
def greedy_search(x, edges_out_to_in, inital_clusters, w, max_iter=100000000, p=2, max_interaction_dist=1):
    assert np.array_equal(x, np.sort(x)), "x is not sorted!"
    assert inital_clusters.dtype.kind == "i"
    assert x.dtype.kind == "f"
    assert edges_out_to_in.dtype.kind == "i"

    n = len(x)
    edges_in_to_out = edges_out_to_in[:, ::-1]
    At_sparse = get_adjacency_dict(edges_in_to_out, n)

    # logger.info("Getting Initial Objective")
    curr_clusters = inital_clusters
    curr_cluster_states = get_initial_cluster_state(curr_clusters, x, At_sparse, p=p)
    curr_objective = get_objective(curr_cluster_states, x, w, p=p)

    # logger.info(f"init obj: {curr_objective}")
    # Includes added
    changed_clusters = List.empty_list(types.int64)
    deleted_clusters = List.empty_list(types.int64)

    one_interactions = create_empty_one_interactions()
    two_interactions = create_empty_two_interactions()

    unique_cluster_sorted, cluster_to_index_of_unique_cluster_sorted = get_sorted_unique_clusters(curr_clusters)

    # logger.info("Starting to iterate...")
    for iteration in range(max_iter):
        # Derivation for this section:
        # You need to track for all c1 and c2: best_move(c1 [,c2])
        # 1. either c1 or c2 was deleted need to stop tracking them
        # 2. either c1 or c2 was updated -> update best_move(c)
        # 3. We only actually allow moves that are between two clusters that are "close enough" based on some distance
        # 3.a Update best_move if they are "close enough" and c1 or c2 was updated
        # 3-b Stop tracking them if NOT "close enough"
        # Finally we assume all best_move(c1, c2) are symmetric, thus we only track c1 < c2
        # Note: The code below is quadratic in number of clusters but can be optimized to avoid the double foor loop through clusters
        # Note: In practice that double for loop is irrelevant has the dist calculation is fast (i.e. i-j) and number_of_clusters <= 10_000 almost always

        # Stop tracking moves from clusters that have been removed in the last iteration
        # Must do this manually because "clusters" does not contain the deleted_cluster and it thus can't be directly removed

        # timer.start_total()
        # logger.info("Calculating best moves")
        # timer.start_block('calc_moves')

        for c1 in deleted_clusters:
            print("!!!")
            print("TODO DELETING CLUSTERS: THIS CODE ISN'T WELL TESTED CLUSTERS ARE ALMOST NEVER EXECUTED!!!")
            print("!!!")
            if c1 in one_interactions:
                del one_interactions[c1]

            for c2 in curr_cluster_states.keys():
                c_a, c_b = sorted((c1, c2))
                if (c_a, c_b) in two_interactions:
                    del two_interactions[(c_a, c_b)]

        # logger.info("Calculating best one_moves")
        # Update the one_interactions (e.g. split) of a clusters that were changed
        for c1 in curr_cluster_states.keys():
            if (c1 not in one_interactions) or (
                c1 in changed_clusters
            ):  # handle newly added clusters OR changed clusters
                # timer.start_block('calc_best_one_move')
                one_interactions[c1] = calc_best_one_move(curr_cluster_states[c1], x, At_sparse, w, p=p)
                # timer.stop_block('calc_best_one_move')

        # Remove two_interactions that are no longer within interaction distance
        for c1, c2 in two_interactions.keys():
            c1_idx = cluster_to_index_of_unique_cluster_sorted[c1]
            c2_idx = cluster_to_index_of_unique_cluster_sorted[c2]

            if not (abs(c2_idx - c1_idx) <= max_interaction_dist):
                del two_interactions[(c1, c2)]

        # Update the two_interactions (e.g. merge-split, sym-swap, exchange) of clusters that were change
        for c2 in unique_cluster_sorted:
            c2_idx = cluster_to_index_of_unique_cluster_sorted[c2]

            # Clusters can interact with each other if they are within max_interaction_dist, as we assume that all
            # two interactions are symmetric we ensure c1_idx<c2_idx
            for c1_idx in range(max(0, c2_idx - max_interaction_dist), c2_idx):
                c1 = unique_cluster_sorted[c1_idx]
                c_a, c_b = sorted((c1, c2))  # we only consider c_a < c_b as all of our moves are symmetric

                # Clusters got closer to each other so add them new to two_interactions or c2 or c1 is a new cluster
                # (for example due to a full merge reducing distance by 1 or a new cluster being added)
                # OR
                # one of the clusters was changed
                if ((c_a, c_b) not in two_interactions) or ((c_a in changed_clusters) or (c_b in changed_clusters)):
                    assert curr_cluster_states[c_a][STATE_N] > 0
                    assert curr_cluster_states[c_b][STATE_N] > 0

                    # timer.start_block('calc_best_two_move')
                    two_interactions[(c_a, c_b)] = calc_best_two_move(
                        curr_cluster_states[c_a], c_a, curr_cluster_states[c_b], c_b, x, At_sparse, w, p=p
                    )
                    # timer.stop_block('calc_best_two_move')

        # timer.stop_block('calc_moves')

        # timer.start_block('find best move')
        # Get the best move. (Kinda slow but still linear. Could use a RMQ Tree for O(moves_updated*ln(number_of_all_moves)))
        best_interaction_move = create_interaction_move(
            delta_obj=0, cluster_update=Dict.empty(key_type=types.int64, value_type=types.int64), move_type=MOVE_NONE
        )

        for c in one_interactions:
            if one_interactions[c][INTER_DELTA_OBJ] < best_interaction_move[INTER_DELTA_OBJ]:
                best_interaction_move = one_interactions[c]

        for c1, c2 in two_interactions:
            if two_interactions[(c1, c2)][INTER_DELTA_OBJ] < best_interaction_move[INTER_DELTA_OBJ]:
                best_interaction_move = two_interactions[(c1, c2)]

        best_delta_obj = best_interaction_move[INTER_DELTA_OBJ]
        best_cluster_update = best_interaction_move[INTER_CLUSTER_UPDATE]
        best_move_type = best_interaction_move[INTER_MOVE_TYPE]

        if best_delta_obj >= 0:
            break  # No improvement found, stop the search
        # timer.stop_block('find best move')

        ##################
        # UPDATE STEP
        ##################
        # timer.start_block('update_step')
        # logger.info("Updating for next iterations")
        # best_cluster_update may contain -1 values which indicate that the indices are assigned to a new cluster.
        # We need to decide on a positive label here
        # TODO do this faster O(1) by tracking a new_label var across iterations as np.max is O(n)
        new_label = np.max(curr_clusters) + 1
        for i in best_cluster_update:
            if best_cluster_update[i] < 0:
                best_cluster_update[i] = new_label

        # Task calculate the indices of all new clusters, also track clusters that will have now n=0 and add them to deleted
        # Such that we can calculate get_cluster_state(cluster_indices, x, At_sparse)
        # NOTICE THAT NEW CLUSTER_INDICES MUST BE SORTED (assertion!)
        # logger.info("Getting clusters that were changed and their new indices")
        changed_clusters_to_indices = get_changed_clusters_to_indices(
            curr_cluster_states, best_cluster_update, curr_clusters
        )

        # logger.info("Updating cluster states")
        deleted_clusters.clear()
        changed_clusters.clear()
        for c, cluster_indices in changed_clusters_to_indices.items():
            # Update delete_clusters
            if len(cluster_indices) == 0:
                print("!!!")
                print(
                    "TODO THIS CODE ISN'T WELL TESTED CLUSTERS ARE ALMOST NEVER EXECUTED!!! Cluster",
                    c,
                    "that was deleted!",
                )
                print("!!!")
                deleted_clusters.append(c)
                del curr_cluster_states[c]

            # Update cluster state otherwise
            else:
                # print("Cluster", c, "was updated!")
                changed_clusters.append(c)
                curr_cluster_states[c] = get_cluster_state(cluster_indices, x, At_sparse, p=p)

        # Update curr_clusters
        for i in best_cluster_update:
            curr_clusters[i] = best_cluster_update[i]

        # Update curr_objective
        curr_objective += best_delta_obj

        # Calculate the new interaction_pos
        # TODO O(n) can be done faster
        # logger.info("Updating the interaction position")
        unique_cluster_sorted, cluster_to_index_of_unique_cluster_sorted = get_sorted_unique_clusters(curr_clusters)

        # timer.stop_block('update_step')

        move_type_str = {MOVE_SWAP: "swap", MOVE_SPLIT: "split", MOVE_MERGE_SPLIT: "merge-split"}
        print(
            "Iteration",
            iteration,
            "Objective:",
            curr_objective,
            "move_type",
            move_type_str.get(best_move_type, "ERROR"),
            "expected_edge_overlap",
            get_expected_edge_overlap(curr_cluster_states),
        )

        assert curr_objective >= 0, "The objective must be positive at all times."

        # timer.stop_and_report()

    return curr_clusters, curr_objective


class Timer:
    def __init__(self):
        self.total_start = None
        self.blocks = {}

    def start_total(self):
        self.total_start = time.time()

    def start_block(self, block_name):
        if block_name not in self.blocks:
            self.blocks[block_name] = {"start": None, "total": 0}
        self.blocks[block_name]["start"] = time.time()

    def stop_block(self, block_name):
        if block_name in self.blocks and self.blocks[block_name]["start"] is not None:
            elapsed = time.time() - self.blocks[block_name]["start"]
            self.blocks[block_name]["total"] += elapsed
            self.blocks[block_name]["start"] = None

    def stop_and_report(self):
        total_time = time.time() - self.total_start
        for block_name, times in self.blocks.items():
            percentage = (times["total"] / total_time) * 100
            print(f"Block '{block_name}' took {percentage:.2f}% of total time")

        self.reset()

    def reset(self):
        self.total_start = None
        self.blocks = {}


timer = Timer()
