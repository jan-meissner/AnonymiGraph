import numpy as np


def calculate_katz(A, alpha=0.1, beta=1.0, num_iters=10000, tol=None):
    if num_iters is None:
        num_iters = float("inf")  # By default only stop once tolerance criteria is met
    if tol is None:
        tol = 1e-12

    n = A.shape[0]  # Size of the matrix
    vec = np.zeros(n)  # Starting with a constant vector

    iter_count = 0  # Initialize iteration counter
    while iter_count < num_iters:  # Loop until num_iters or convergence
        vec_next = alpha * A @ vec + beta

        # Check for convergence
        if np.linalg.norm(vec_next - vec) < tol:
            print(f"Katz converged after {iter_count} iterations.")
            break

        vec = vec_next

        # Warn if exceeding 10,000 iterations
        if iter_count == 10000:
            print(
                "Iteration count exceeded 10,000. Consider increasing tolerance or checking the matrix for convergence properties."
            )
        if iter_count == 100000:
            print(
                "Iteration count exceeded 100,000. Consider increasing tolerance or checking the matrix for convergence properties."
            )
        if iter_count == 1000000:
            print(
                "Iteration count exceeded 1,000,000. Consider increasing tolerance or checking the matrix for convergence properties."
            )

        if iter_count >= num_iters:
            print("Number of iterations exceeded the maximum (num_iters) allowed iterations.")
            break

        iter_count += 1

    return vec


def calculate_kmeans_cluster_loss(x, cluster_labels, centroids, mode=2):
    assert mode == 2
    total_loss = 0
    for i, color in enumerate(cluster_labels):
        centroid = centroids[color]
        distance = (x[i] - centroid) ** 2
        total_loss += distance
    return np.sqrt(total_loss)
