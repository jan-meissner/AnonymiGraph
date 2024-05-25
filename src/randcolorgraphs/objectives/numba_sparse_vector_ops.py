from numba import njit


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
