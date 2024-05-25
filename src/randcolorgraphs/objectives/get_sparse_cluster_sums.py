from numba import njit, types
from numba.typed import Dict

from randcolorgraphs.objectives.numba_sparse_vector_ops import inplace_add_sparse_vecs

int_sparse_vector_type = types.DictType(types.int64, types.int64)


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
