import numpy as np


def colour_refinement_out_neigh_algorithm(A, depth):
    n = A.shape[0]
    clusters = np.zeros(n, dtype=int)

    for d in range(depth):
        unique_clusters, cluster_indices = np.unique(clusters, return_inverse=True)
        H = np.identity(len(unique_clusters))[cluster_indices]

        AH = A @ H

        new_unique_clusters, new_clusters = np.unique(AH, axis=0, return_inverse=True)

        if len(new_unique_clusters) == len(unique_clusters):
            print(f"Converged at depth {d} with {len(unique_clusters)} unique clusters")
            break

        clusters = new_clusters

    return clusters
