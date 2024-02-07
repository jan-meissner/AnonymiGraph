import random
from copy import deepcopy

import networkx as nx

from .abstract_anonymizer import AbstractAnonymizer


class ConfigurationModelAnonymizer(AbstractAnonymizer):
    def anonymize(self, graph: nx.Graph) -> nx.Graph:
        """
        Anonymizes a given undirected network graph using a configuration model approach.

        Based on the algorithm outlined in Tews, Mara. “Privacy-Aware Network Sharing with High Utility,” 2023.

        Args:
            graph (nx.Graph): A NetworkX graph object representing the network to be anonymized.

        Returns:
            nx.Graph: The anonymized graph, where node connections have been reconfigured.
        """
        if nx.is_empty(graph):
            return graph

        if graph.is_directed():
            raise nx.NetworkXNotImplemented("Not implemented for directed graphs.")

        # Deepcopy to prevent graph from being modified in-place
        graph = deepcopy(graph)

        node_labels, degrees = zip(*graph.degree())
        node_labels, degrees = list(node_labels), list(degrees)
        random.shuffle(degrees)

        stublist = [n for n, d in zip(node_labels, degrees) for _ in range(d)]
        random.shuffle(stublist)

        # Split the stublist into two halves for creating new edges
        half = len(stublist) // 2
        out_stublist, in_stublist = stublist[:half], stublist[half:]

        # Remove old edges, add new edges and remove self loops
        graph.remove_edges_from(graph.edges())
        graph.add_edges_from(zip(out_stublist, in_stublist))
        graph.remove_edges_from(nx.selfloop_edges(graph))
        return graph
