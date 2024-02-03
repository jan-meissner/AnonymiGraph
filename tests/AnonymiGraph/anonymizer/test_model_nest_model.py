from __future__ import annotations

import networkx as nx

import anonymigraph.anonymization as anon


def test_nest_model():
    G = nx.erdos_renyi_graph(1000, 15 / 1000)
    G.add_nodes_from([1000000])  # add isolate

    depth = 2
    G_nest = anon.nest_model(G, depth=depth, r=80)
    anon.method_nest_model._validate_nest(G, G_nest, depth)
