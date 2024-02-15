import kmeans1d
import networkx as nx
import numpy as np

from anonymigraph.anonymization._external.nest_model._rewire import _rewire
from anonymigraph.utils import _validate_input_graph

from .abstract_anonymizer import AbstractAnonymizer


class VariantNestModelAnonymizer(AbstractAnonymizer):
    """
    Generates a synthetic graph that preserves; similar to the nest model as defined in Stamm et al. (2023).
    Core difference is that instead of the WL colors a different notion of node roles is used that is based on
    minimizing the long-term loss as defined by Scholkemper et al. (2023).

    Allows fine tuning of the number of different node roles, unlike NeSt.

    Args:
        G (nx.Graph): The input graph from which to sample the synthetic graph.
        k (int): Number of different node roles.
        r (int, optional): Multiplier for the number of edge swap attempts.
                           Total swap attempts are r * num_edges.
        parallel (bool, optional): Enables parallelization of the process.

    References:
        Scholkemper, M., & Schaub, M. T. (2023). An Optimization-based Approach To Node Role Discovery in Networks:
        Approximating Equitable Partitions. arXiv preprint arXiv:2305.19087.
    """

    def __init__(self, k, r=1, parallel=False):
        self.k = k
        self.r = r
        self.parallel = parallel

        if self.k == 0:
            raise ValueError("Algorithm undefined for k=0, please choose d>0.")

    def anonymize(self, G: nx.Graph, random_seed: int = None) -> nx.Graph:
        """
        Anonymize and return the anonymized graph.

        Args:
            graph (nx.Graph): The original graph that is to be anonymized.
            random_seed (int, optional): Seed for the random number generator.

        Returns:
            nx.Graph: The anonymized graph.
        """

        _validate_input_graph(G)

        colors = self._get_node_roles(G)  # colors = node roles

        edges = np.array(G.edges(), dtype=np.uint32)
        edges_rewired = _rewire(edges, colors.reshape(1, -1), r=self.r, parallel=self.parallel, random_seed=random_seed)

        Ga = nx.Graph()
        Ga.add_nodes_from(G.nodes(data=True))
        Ga.add_edges_from(edges_rewired)

        return Ga

    def _get_node_roles(self, G: nx.Graph):
        "Get k different node roles by minimizing the long term loss."
        _validate_input_graph(G)
        centrality_map = nx.eigenvector_centrality(G, max_iter=10000)
        centralities_ordered = [centrality_map[i] for i in range(G.number_of_nodes())]
        colors, _ = kmeans1d.cluster(centralities_ordered, self.k)
        return np.array(colors)
