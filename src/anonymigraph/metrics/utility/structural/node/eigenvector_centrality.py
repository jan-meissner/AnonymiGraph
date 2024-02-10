import networkx as nx

from .metric_distribution import DistributionMetric


class EigenvectorMetric(DistributionMetric):
    def __init__(self):
        super().__init__("Eigenvector Centrality")

    def compute_centrality(self, G: nx.Graph):
        return list(nx.eigenvector_centrality(G, max_iter=1000).values())  # max_iter may need adjustment
