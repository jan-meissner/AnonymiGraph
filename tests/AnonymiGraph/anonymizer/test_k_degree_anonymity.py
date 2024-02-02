from __future__ import annotations

import json
import os

import networkx as nx

from anonymigraph.anonymization import k_degree_anonymity


def write_graph(G, fname):
    """
    Use this function to save graphs used for the test case.
    """
    # write_graph(G, "graph.json")
    # write_graph(Ga,"graph_anon.json")
    with open(fname, "w") as file:
        json.dump(list(G.edges()), file)


def load_graph(fname):
    with open(fname, "r") as file:
        edges = json.load(file)

    G = nx.Graph()
    max_node = max(edge[0] for edge in edges)  # undirected graph so we can just look at the source node
    G.add_nodes_from(range(max_node + 1))
    G.add_edges_from(edges)
    return G


def test_ensuring_equivalence_with_original():
    current_dir = os.path.dirname(__file__)
    graph_path = os.path.join(current_dir, "testdata/graph.json")
    graph_anon_path = os.path.join(current_dir, "testdata/graph_anon.json")

    # Generated by the original algorithm from https://github.com/blextar/graph-k-degree-anonymity/tree/master
    G = load_graph(graph_path)
    expected_Ga = load_graph(graph_anon_path)

    Ga = k_degree_anonymity(G, k=100, noise=10, with_deletions=True, random_seed=42)

    assert expected_Ga.edges() == Ga.edges()