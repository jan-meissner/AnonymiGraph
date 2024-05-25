import numpy as np

from randcolorgraphs.objectives.get_cluster_loss_ell_sqr import get_cluster_loss_ell_sqr


def naive_ell(x, clusters, p=2):
    n = len(clusters)

    unique_clusters, cluster_indices = np.unique(clusters, return_inverse=True)
    H = np.identity(len(unique_clusters))[cluster_indices]

    HtHinv = np.linalg.inv(H.T @ H)
    Ppar = H @ HtHinv @ H.T

    Id = np.identity(n)
    ell = np.sum(np.abs((Id - Ppar) @ x) ** p)

    return ell


def test_get_cluster_loss_ell_sqr():
    np.random.seed(33242)
    x = np.random.rand(10)
    clusters = np.array([0, 0, 1, 1, 1, 2, 2, 0, 2, 1])

    assert np.allclose(get_cluster_loss_ell_sqr(x, clusters), naive_ell(x, clusters))


def test_get_cluster_loss_ell_sqr_large():
    np.random.seed(33242)
    n = 100
    x = np.random.rand(n)
    clusters = np.random.randint(0, 20, size=n)

    assert np.allclose(get_cluster_loss_ell_sqr(x, clusters), naive_ell(x, clusters))
