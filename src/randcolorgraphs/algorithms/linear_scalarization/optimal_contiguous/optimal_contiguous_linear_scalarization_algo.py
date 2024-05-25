import numpy as np
from numba import njit

from randcolorgraphs.algorithms.linear_scalarization.optimal_contiguous.cluster_segment_loss import (
    get_cluster_segment_cost,
)


@njit
def optimal_contiguous_linear_scalarization_algo(x, A, w):
    #     !!!
    # TODO !!!This function was written by Copilot, either rewrite or get a source!!
    #     !!!
    sorted_indices = np.argsort(x)
    sorted_x = x[sorted_indices]
    sorted_A = A[:, sorted_indices][sorted_indices, :]
    S = get_cluster_segment_cost(sorted_x, sorted_A, w)

    n = len(sorted_x)
    D = np.zeros((n + 1, n + 1)) + float("inf")
    T = np.zeros((n + 1, n + 1), dtype=int)

    # Base case
    for m in range(1, n + 1):
        D[1][m] = S[0][m - 1]

    # DP to fill D and T
    for i in range(2, n + 1):
        for m in range(1, n + 1):
            for j in range(1, m + 1):
                cost = D[i - 1][j - 1] + S[j - 1][m - 1]
                if cost < D[i][m]:
                    D[i][m] = cost
                    T[i][m] = j

            # Add j = m+1 case which corresponds to an empty cluster
            cost = D[i - 1][m]
            if cost < D[i][m]:
                D[i][m] = cost
                T[i][m] = m + 1

    # Backtrack to find the optimal clusters
    clusters = []
    current = n
    for i in range(n, 0, -1):
        start = T[i][current]
        lower = max(0, start - 1)
        clusters.extend(
            [
                i,
            ]
            * (current - lower)
        )
        current = lower

    clusters.reverse()
    clusters = np.array(clusters)
    return clusters[np.argsort(sorted_indices)]
