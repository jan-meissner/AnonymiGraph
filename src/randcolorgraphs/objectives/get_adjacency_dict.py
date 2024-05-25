from numba import njit, types
from numba.typed import Dict

int_sparse_vector_type = types.DictType(types.int64, types.int64)


@njit
def get_adjacency_dict(edges, n):
    adjacency_dict = Dict.empty(key_type=types.int64, value_type=int_sparse_vector_type)

    for edge in edges:
        out_node = edge[0]
        in_node = edge[1]

        if out_node not in adjacency_dict:
            adjacency_dict[out_node] = Dict.empty(key_type=types.int64, value_type=types.int64)

        adjacency_dict[out_node][in_node] = 1

    # Ensure nodes without out edges are also initialized
    for i in range(n):
        if i not in adjacency_dict:
            adjacency_dict[i] = Dict.empty(key_type=types.int64, value_type=types.int64)

    return adjacency_dict
