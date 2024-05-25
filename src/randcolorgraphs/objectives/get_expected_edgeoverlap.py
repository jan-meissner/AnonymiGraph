from numba import njit, types
from numba.typed import Dict

from randcolorgraphs.objectives.get_adjacency_dict import get_adjacency_dict
from randcolorgraphs.objectives.get_cluster_to_indices import get_cluster_to_indices
from randcolorgraphs.objectives.get_edges_from_adj_matrix import (
    get_edges_from_adj_matrix,
)
from randcolorgraphs.objectives.get_sparse_cluster_sums import get_sparse_cluster_sums


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
            total += abs(sparse_vec[i]) ** p
        p_power_sum[c] = total

    return p_power_sum


@njit
def _get_expected_edgeoverlap(At_neigh_dict, clusters, p=2):
    cluster_to_indices = get_cluster_to_indices(clusters)  # O(n)
    SAt = get_sparse_cluster_sums(At_neigh_dict, cluster_to_indices)  # O(|E|)
    p_power_sum_SAt = get_sparse_p_power_sum(SAt, p=p)  # O(|E|)

    expected_edgeoverlap = 0.0
    for c in cluster_to_indices:
        cluster_indices = cluster_to_indices[c]
        n_cluster = len(cluster_indices)

        expected_edgeoverlap += p_power_sum_SAt[c] / n_cluster

    return expected_edgeoverlap


def get_expected_edgeoverlap(A_sparse, clusters):
    n = A_sparse.shape[0]
    edges_out_to_in = get_edges_from_adj_matrix(A_sparse)
    edges_in_to_out = edges_out_to_in[:, ::-1]
    At_neigh_dict = get_adjacency_dict(edges_in_to_out, n)
    return _get_expected_edgeoverlap(At_neigh_dict, clusters)
