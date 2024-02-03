from __future__ import annotations

import networkx as nx
import numpy as np

from ._external.nest_model.fast_rewire import get_block_indices, rewire_fast, sort_edges
from ._external.nest_model.fast_wl import WL_fast


def nest_model(
    G: nx.Graph,
    depth: int,
    r: int = 1,
    initial_colors: np.ndarray = None,
    parallel: bool = False,
    random_seed: int = None,
):
    """
    Generates a synthetic graph that preserves the d-hop neighborhood structure of a given graph.
    This function implements the nest model as defined in Stamm et al. (2023).

    Parameters:
        G (nx.Graph): The input graph from which to sample the synthetic graph.
        depth (int): Specifies the 'hop' depth in the graph to be preserved in the synthetic version.
                     For example a depth of 1 preserves the 1-hop structure i.e. the degree.
        r (int, optional): Multiplier for the number of edge swap attempts.
                           Total swap attempts are r * num_edges.
        initial_colors (np.ndarray, optional): Initial color labels for the Weisfeiler-Lehman algorithm.
        parallel (bool, optional): Enables parallelization of the process.
        random_seed (int, optional): Seed for the random number generator.

    Returns:
        nx.Graph: The generated synthetic graph.

    References:
        Stamm, F. I., Scholkemper, M., Strohmaier, M., & Schaub, M. T. (2023).
        Neighborhood Structure Configuration Models. Proceedings of the ACM Web Conference 2023.
        [URL: http://arxiv.org/abs/2210.06843]

    Implementation based on:
        https://github.com/Feelx234/nestmodel
    """
    if depth == 0:
        raise ValueError("Algorithm undefined for d=0, please choose d>0.")

    edges = np.array(G.edges, dtype=np.uint32)
    bidirectional_edges = np.row_stack((edges, edges[:, [1, 0]]))

    all_depth_colors = WL_fast(bidirectional_edges, num_nodes=None, labels=initial_colors, max_iter=depth)

    edges_rewired = _rewire(
        edges,
        all_depth_colors[-1].reshape(1, -1),
        r=r,
        parallel=parallel,
        random_seed=random_seed,
    )

    G_out = nx.Graph()
    G_out.add_nodes_from(G.nodes())
    G_out.add_edges_from(edges_rewired)

    # _validate_nest(G, G_out, depth, initial_colors)

    return G_out


def _rewire(edges, colors, r=1, parallel=True, random_seed=None):
    """
    Rewires an undirected graph's subgraphs based on partition labels.

    Parameters:
    - edges: np.ndarray (num_edges, 2), unique, arbitrarily oriented edges (each edge appears once)
    - colors: np.ndarray (num_nodes,), color labels for nodes
    - r: int, multiplier of edge swaps attempted (total swap attempts are r*num_edges)
    - parallel: bool, enables parallelization
    - random_seed: int, seed for random number generator. If set parallel also needs to be set to false.
    """
    edges_ordered, edges_classes, dead_arr, is_mono = sort_edges(edges, colors, is_directed=False)
    block_indices = get_block_indices(edges_classes, dead_arr)

    rewire_fast(
        edges_ordered,
        edges_classes[:, 0],
        is_mono[0],
        block_indices[0],
        is_directed=False,
        seed=random_seed,
        num_flip_attempts_in=r,
        parallel=parallel,
    )

    return edges_ordered


def _validate_nest(G: nx.Graph, G_out: nx.Graph, depth: int, initial_colors: np.ndarray = None):
    """
    Validate the Nest Model. Checks if all

    :param G: Original graph.
    :param G_out: Output graph from nest_model.
    :param depth: Same parameter as in nest_model.
    :param initial_colors  Same parameter as in nest_model.
    """
    edges = np.array(G.edges, dtype=np.uint32)
    edges_rewired = np.array(G_out.edges, dtype=np.uint32)

    bidirectional_edges = np.row_stack((edges, edges[:, [1, 0]]))
    bidirectional_edges_rewired = np.row_stack((edges_rewired, edges_rewired[:, [1, 0]]))

    all_depth_colors = WL_fast(bidirectional_edges, num_nodes=None, labels=initial_colors, max_iter=depth + 1 + 100)
    all_depth_colors_rewired = WL_fast(
        bidirectional_edges_rewired, num_nodes=None, labels=initial_colors, max_iter=depth + 1 + 100
    )

    for i in range(min(depth + 1, len(all_depth_colors))):
        # print(i, all_depth_colors[i])
        # print(i, all_depth_colors_rewired[i])
        if not np.all(all_depth_colors[i] == all_depth_colors_rewired[i]):
            raise AssertionError(f"Validation failed at depth {i}")
