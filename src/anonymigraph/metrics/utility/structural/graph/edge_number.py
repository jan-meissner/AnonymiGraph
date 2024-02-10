import networkx as nx

from .metric_scalar import ScalarMetric


class NumberOfEdgesMetric(ScalarMetric):
    def __init__(self):
        super().__init__("|Edges|")

    def compute_scalar(self, G: nx.Graph):
        return G.number_of_edges()
