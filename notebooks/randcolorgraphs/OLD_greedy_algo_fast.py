import numpy as np
from numba import njit, types
from numba.typed import Dict, List

integer_list_type = types.ListType(types.int64)


@njit
def get_cluster_to_indices(cluster):
    """
    Returns a dict that holds for each cluster the indices of vertices inside the cluster.
    Important: Returned indices are sorted in ascending order in the list.
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


# Example usage
# cluster = np.array([0, 1, 0, 2, 1, 2, 0, 5, 5, 5])
# cluster_to_indices = get_cluster_to_indices(cluster)
# print(cluster_to_indices)


@njit
def get_scalar_cluster_sums(x, cluster_to_indices):
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


# Sx = get_scalar_cluster_sums(cluster, cluster_to_indices)
# print(Sx)

int_sparse_vector_type = types.DictType(types.int64, types.int64)


@njit
def get_adjacency_dicts(edges, n):
    adjacency_dict = Dict.empty(key_type=types.int64, value_type=int_sparse_vector_type)

    # Populate the outer dictionary with edges data
    for edge in edges:
        out_node = edge[0]
        in_node = edge[1]

        # Get the color multi set or create a new one if out_node not yet seen.
        if out_node not in adjacency_dict:
            out_neigh_sparse_vec = Dict.empty(key_type=types.int64, value_type=types.int64)
            adjacency_dict[out_node] = out_neigh_sparse_vec

        adjacency_dict[out_node][in_node] = 1

    # Ensure nodes without out edges are also initialized
    for i in range(n):
        if i not in adjacency_dict:
            adjacency_dict[i] = Dict.empty(key_type=types.int64, value_type=types.int64)

    return adjacency_dict


# edges = np.array([
#    (0, 1),
#    (0, 2),
#    (0, 5),
#    (1, 0),
#    (1, 3),
#    (2, 0),
#    (2, 3),
#    (3, 1),
#    (3, 2),
#    (3, 4),
#    (4, 3),
#    (5, 0)
# ])
#
# Create the adjacency matrix
# adjacency_matrix = get_adjacency_dicts(edges, 10)
# print(adjacency_matrix)

int_sparse_vector_type = types.DictType(types.int64, types.int64)


@njit
def inplace_add_sparse_vecs(sparse_vec_1, sparse_vec_2):
    for key, value in sparse_vec_2.items():
        if key in sparse_vec_1:
            sparse_vec_1[key] += value
        else:
            sparse_vec_1[key] = value

    return sparse_vec_1


@njit
def inplace_subtract_sparse_vecs(sparse_vec_1, sparse_vec_2):
    for key, value in sparse_vec_2.items():
        if key in sparse_vec_1:
            sparse_vec_1[key] -= value
        else:
            sparse_vec_1[key] = -value

    return sparse_vec_1


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


@njit
def get_sparse_cluster_sums(adjacency_dicts, cluster_to_indices):
    """
    Efficient sums
    """
    S_sparse = Dict.empty(
        key_type=types.int64,
        value_type=int_sparse_vector_type,
    )
    for c in cluster_to_indices:
        for i in cluster_to_indices[c]:
            if c not in S_sparse:
                S_sparse[c] = Dict.empty(key_type=types.int64, value_type=types.int64)

            inplace_add_sparse_vecs(S_sparse[c], adjacency_dicts[i])

    return S_sparse


# edges = np.array([
#    (0, 1),
#    (0, 2),
#    (0, 5),
#    (1, 0),
#    (1, 3),
#    (2, 0),
#    (2, 3),
#    (3, 1),
#    (3, 2),
#    (3, 4),
#    (4, 3),
#    (5, 0)
# ])
#
# cluster = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
#
# Create the adjacency matrix
# adjacency_dicts = get_adjacency_dicts(edges, 10)
# cluster_to_indices = get_cluster_to_indices(cluster)
# sparse_cluster_sum = get_sparse_cluster_sums(adjacency_dicts, cluster_to_indices)
# print(sparse_cluster_sum)


@njit
def compute_delta_obj(S_j_prime, S_k_prime, n_j_prime, n_k_prime, S_j, S_k, n_j, n_k):
    partial_objective_before = S_j / n_j + S_k / n_k
    partial_objective_after = S_j_prime / n_j_prime + S_k_prime / n_k_prime
    return partial_objective_after - partial_objective_before


@njit
def sparse_p_power_sum(sparse_vector, p=2):
    total = 0.0
    for index in sparse_vector:
        total += sparse_vector[index] ** p

    return total


@njit
def get_sparse_p_power_sum(dict_of_sparse_vectors, p=2):
    p_power_sum = Dict.empty(
        key_type=types.int64,
        value_type=types.float64,
    )
    for c in dict_of_sparse_vectors:
        sparse_vec = dict_of_sparse_vectors[c]
        total = 0.0
        for i in sparse_vec:
            total += sparse_vec[i] ** p
        p_power_sum[c] = total

    return p_power_sum


@njit
def get_objective(current_clusters, x, At_sparse, w, p=2):
    cluster_to_indices = get_cluster_to_indices(current_clusters)  # O(n)
    Sx = get_scalar_cluster_sums(x, cluster_to_indices)  # O(n)
    SAt = get_sparse_cluster_sums(At_sparse, cluster_to_indices)  # O(|E|)
    p_power_sum_SAt = get_sparse_p_power_sum(SAt, p=p)  # O(|E|)

    obj = np.sum(x**p)
    for c in cluster_to_indices:
        cluster_indices = cluster_to_indices[c]
        n_j = len(cluster_indices)

        obj += -Sx[c] ** p / n_j + w * p_power_sum_SAt[c] / n_j

    return obj


@njit
def move_split_cluster(current_clusters, cluster_to_indices, Sx, SAt, p_power_sum_SAt, At_sparse, x, w):
    best_cluster_update = Dict.empty(key_type=types.int64, value_type=types.int64)
    best_delta_obj = 0

    # Try split moves
    for c in cluster_to_indices:
        cluster_indices = cluster_to_indices[c]

        # Cluster indices are assumed to be in
        assert len(cluster_indices) > 0

        n_j = len(cluster_indices)
        Sx_j = Sx[c]  # S hold the sum over the vectors of all points in cluster j
        SAt_j = SAt[c]
        p_power_sum_SAt_j = p_power_sum_SAt[c]

        n_j_prime = n_j
        n_k_prime = 0

        Sx_j_prime = Sx_j
        Sx_k_prime = 0

        SAt_j_prime = SAt_j.copy()
        SAt_k_prime = Dict.empty(key_type=types.int64, value_type=types.int64)

        # Further speed up
        p_power_sum_SAt_j_prime = p_power_sum_SAt[c]
        p_power_sum_SAt_k_prime = 0.0

        best_loop_idx = -1
        for loop_idx in range(len(cluster_indices) - 1):
            # Check the cluster splitting of the first (sorted by x) loop_idx+1 many in the first cluster and all other in second cluster
            i = cluster_indices[loop_idx]

            n_j_prime = n_j_prime - 1
            n_k_prime = n_k_prime + 1

            # Calculate x objective loss component, Complexity with outer for loops: O(n)

            Sx_j_prime = Sx_j_prime - x[i]
            Sx_k_prime = Sx_k_prime + x[i]

            delta_obj_x = compute_delta_obj(
                Sx_j_prime**2, Sx_k_prime**2, n_j_prime, n_k_prime, Sx_j**2, 0, n_j, 1
            )  # Sx_k is 0 so n_k can be arbitrary

            # Calculate A^t objective loss component, Complexity with outer for loops: O(|E|)

            p_power_sum_SAt_j_prime += get_subtract_change_in_p_power_sum(SAt_j_prime, At_sparse[i])
            p_power_sum_SAt_k_prime += get_add_change_in_p_power_sum(SAt_k_prime, At_sparse[i])

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
            # print(
            #    "Found a better split at vertex best_split_index",
            #    best_loop_idx,
            #    "of",
            #    len(cluster_indices),
            #    "for cluster",
            #    c,
            #    "Best delta obj is now ",
            #    best_delta_obj,
            # )
            best_cluster_update = Dict.empty(key_type=types.int64, value_type=types.int64)
            new_cluster_index = np.max(current_clusters) + 1
            for j in range(best_loop_idx + 1):
                best_cluster_update[cluster_indices[j]] = new_cluster_index

    return best_delta_obj, best_cluster_update


# def move_swap_cluster(current_clusters, cluster_to_indices, Sx, SAt, p_power_sum_SAt, At_sparse, x, w):
#    get_sorted_unique_clusters(current_clusters)#
#
#    return cluster_to_indices


#    best_cluster_update = Dict.empty(key_type=types.int64, value_type=types.int64)
#    best_delta_obj = 0

# Try split moves


# 11111111111222222222233333333344444444444455
# 11111111111222222222233333333344444444444455


# TODO Speed up by only recalculating changed clusters between iterations, hence we also want to cache all the move types and make a queue out of them; re-adding them when they are update (i.e. the cluster they are for changed)
# @njit(cache=False)
def greedy_search(x, edges_out_to_in, inital_clusters, w, max_iter=100000000):
    assert np.array_equal(x, np.sort(x)), "x is not sorted!"
    assert inital_clusters.dtype.kind == "i"
    assert x.dtype.kind == "f"
    assert edges_out_to_in.dtype.kind == "i"

    n = len(x)
    edges_in_to_out = edges_out_to_in[:, ::-1]
    At_sparse = get_adjacency_dicts(edges_in_to_out, n)

    print("Getting Initial Objective")
    current_clusters = inital_clusters
    current_objective = get_objective(current_clusters, x, At_sparse, w, p=2)
    print("init objective", current_objective)

    print("Starting Iterating")
    for iteration in range(max_iter):
        print("Starting new iteration")
        cluster_to_indices = get_cluster_to_indices(current_clusters)  # O(n)
        Sx = get_scalar_cluster_sums(x, cluster_to_indices)  # O(n)
        SAt = get_sparse_cluster_sums(At_sparse, cluster_to_indices)  # O(|E|)
        p_power_sum_SAt = get_sparse_p_power_sum(SAt)  # O(|E|)
        print("Finished Iteration Init")

        best_cluster_update = Dict.empty(key_type=types.int64, value_type=types.int64)
        best_delta_obj = 0

        # Uneven Split Moves;
        # Complexity: O(|E|)
        # Split 1vn-1 2vn-2 3vn-3 ... (sorted by katz centrality)
        print("Performing split moves")
        split_best_delta_obj, split_best_cluster_update = move_split_cluster(
            current_clusters, cluster_to_indices, Sx, SAt, p_power_sum_SAt, At_sparse, x, w
        )
        if split_best_delta_obj < best_delta_obj:
            best_delta_obj = split_best_delta_obj
            best_cluster_update = split_best_cluster_update

        # SWAP (from PAM for k-medoids)
        # swap_best_delta_obj, swap_best_cluster_update = move_swap_cluster(current_clusters, cluster_to_indices, Sx, SAt, p_power_sum_SAt, At_sparse, x, w)
        # if swap_best_delta_obj < best_delta_obj:
        #    best_delta_obj = swap_best_delta_obj
        #    best_cluster_update = swap_best_cluster_update

        # Partial Merge;
        # Complexity O(|E|)
        # Expand one cluster at the cost of another cluster
        # Example: 11111111133334444444444 -> 11111444433334444444444
        # Allows to undo "mistakes" of early uneven splits easily

        # BAD Try 2-swap moves O(n^2) (the minimum range idea doesn't work as for large graph clusters might be large too)
        # for i in range(len(data)):
        #   for j in range(i+1, len(data)):
        #       a, b = initial_assignments[i], initial_assignments[j]
        #       if a != b:
        #           delta_obj = two_swap_delta(state_variables, i, j, a, b)
        #           if delta_obj < best_delta_obj:
        #               best_move = ('2-swap', i, j, a, b, delta_obj)
        #               best_delta_obj = delta_obj

        # The k-color adjacent color flip

        # REPLACED BY PARTIAL MERGE: Try merge moves
        # for j in clusters:
        #    for k in clusters:
        #        if j != k and state_variables['n'][k] > 0:
        #            delta_obj = merge_clusters_delta(state_variables, j, k)
        #            if delta_obj < best_delta_obj:
        #                best_move = ('merge', j, k)
        #                best_delta_obj = delta_obj

        # BAD O(n^2) Try 1-swap moves
        # for i in range(len(data)):
        #    for k in clusters:
        #        if initial_assignments[i] != k:
        #            j = initial_assignments[i]
        #            delta_obj = one_swap_delta(state_variables, i, j, k)
        #            if delta_obj < best_delta_obj:
        #                best_move = ('1-swap', i, j, k, delta_obj)
        #                best_delta_obj = delta_obj

        if best_delta_obj >= 0:
            break  # No improvement found, stop the search

        # Update clusters
        for i in best_cluster_update:
            current_clusters[i] = best_cluster_update[i]

        current_objective += best_delta_obj
        print("Iteration", iteration, "Objective:", current_objective)

    return current_clusters, current_objective


# 2 swap worst
