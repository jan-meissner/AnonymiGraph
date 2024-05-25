import networkx as nx
import numpy as np

from randcolorgraphs.objectives.get_expected_edgeoverlap import get_expected_edgeoverlap


def naive_expected_edge_overlap(A, clusters, p=2):
    unique_clusters, cluster_indices = np.unique(clusters, return_inverse=True)
    H = np.identity(len(unique_clusters))[cluster_indices]

    HtHinv = np.linalg.inv(H.T @ H)
    Ppar = H @ HtHinv @ H.T

    expected_edge_overlap = np.linalg.norm(A @ Ppar, "fro") ** 2
    return expected_edge_overlap


def test_get_expected_edgeoverlap():
    seed = 33242
    np.random.seed(seed)
    n = 10
    p = 3 / n
    G = nx.erdos_renyi_graph(n, p, directed=True, seed=seed)
    A_G = nx.adjacency_matrix(G).astype(np.float64)

    clusters = np.array([0, 0, 1, 1, 1, 2, 2, 0, 2, 1])

    assert np.allclose(get_expected_edgeoverlap(A_G, clusters), naive_expected_edge_overlap(A_G, clusters))


def test_get_cluster_loss_ell_power_p_large():
    seed = 33242
    np.random.seed(seed)
    n = 100
    p = 15 / n
    G = nx.erdos_renyi_graph(n, p, directed=True, seed=seed)
    A_G = nx.adjacency_matrix(G).astype(np.float64)

    clusters = np.random.randint(0, 20, size=n)

    assert np.allclose(get_expected_edgeoverlap(A_G, clusters), naive_expected_edge_overlap(A_G, clusters))
