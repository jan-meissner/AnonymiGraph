from __future__ import annotations

import networkx as nx
import numpy as np

from .external.k_degree_anonymity import k_degree_anonymity as _k_deg


def k_degree_anonymity(G: nx.Graph, k: int, noise: int = 10, with_deletions: bool = True, random_seed=None) -> nx.Graph:
    """
    Performs k-degree anonymity on a given graph G, modifying it to ensure that each node shares its degree with at
    least k-1 other nodes.

    The function implements the method described in Liu & Terzi's paper on k-degree anonymity (see reference [1]).

    Parameters:
    - G (networkx.Graph): The graph to be anonymized.
    - k (int): The anonymity parameter, ensuring each node's degree is shared by at least k-1 other nodes.
    - noise (int, optional): The level of noise added after failed anonymization attempts.
    - with_deletions (bool, optional): Allows deletion of edges to achieve k-degree anonymity.
    - random_seed (optional): Seed for random number generation for reproducibility.

    Returns:
    - networkx.Graph: The anonymized graph.

    References:
    [1] Liu, K., & Terzi, E. (2008). Towards identity anonymization on graphs.
        ACM SIGMOD International Conference on Management of Data. https://dl.acm.org/doi/10.1145/1376616.1376629

    Implementation:
    Uses the implementation by Rossi as a base. https://github.com/blextar/graph-k-degree-anonymity/tree/master
    """

    if random_seed:
        np.random.seed(random_seed)

    Ga = _k_deg(G, k, noise=noise, with_deletions=with_deletions)

    for node, data in G.nodes(data=True):
        Ga.nodes[node].update(data)

    return Ga
