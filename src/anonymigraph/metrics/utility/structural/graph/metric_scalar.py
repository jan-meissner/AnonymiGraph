import networkx as nx

from anonymigraph.metrics.metric import Metric


class ScalarMetric(Metric):
    def compute_scalar(self, G: nx.Graph):
        raise NotImplementedError("Subclass must implement abstract method")

    def evaluate(self, G: nx.Graph, Ga: nx.Graph):
        num_G = self.compute_scalar(G)
        num_Ga = self.compute_scalar(Ga)

        return {"G": num_G, "Ga": num_Ga}
