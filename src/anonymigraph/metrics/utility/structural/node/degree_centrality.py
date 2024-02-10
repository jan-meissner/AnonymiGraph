import networkx as nx

from .metric_distribution import DistributionMetric


class DegreeCentralityMetric(DistributionMetric):
    def __init__(self):
        super().__init__("Degree Centrality")

    def compute_centrality(self, G: nx.Graph):
        return list(nx.degree_centrality(G).values())
