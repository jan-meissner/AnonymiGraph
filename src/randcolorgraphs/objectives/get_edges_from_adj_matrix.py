import numpy as np
import scipy.sparse as sp


def get_edges_from_adj_matrix(adj_matrix):
    sparse_matrix = sp.csr_matrix(adj_matrix)
    row_indices, col_indices = sparse_matrix.nonzero()
    edge_vector = np.vstack((row_indices, col_indices)).T
    return edge_vector
