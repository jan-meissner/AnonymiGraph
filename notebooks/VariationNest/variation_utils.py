import numpy as np


def calculate_kmeans_cluster_loss(x, cluster_labels, centroids, mode=2):
    assert mode == 2
    total_loss = 0
    for i, color in enumerate(cluster_labels):
        centroid = centroids[color]
        distance = (x[i] - centroid) ** 2
        total_loss += distance
    return np.sqrt(total_loss)
