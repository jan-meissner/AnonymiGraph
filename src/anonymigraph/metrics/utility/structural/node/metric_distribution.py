import networkx as nx
from scipy.stats import wasserstein_distance

from anonymigraph.metrics.metric import Metric


class DistributionMetric(Metric):
    def compute_centrality(self, G: nx.Graph):
        raise NotImplementedError("Subclass must implement abstract method")

    def evaluate(self, G: nx.Graph, Ga: nx.Graph):
        centrality_G = self.compute_centrality(G)
        centrality_Ga = self.compute_centrality(Ga)
        return self.distribution_distance_func(centrality_G, centrality_Ga)

    def distribution_distance_func(self, valsP, valsQ):
        return wasserstein_distance(valsP, valsQ)
