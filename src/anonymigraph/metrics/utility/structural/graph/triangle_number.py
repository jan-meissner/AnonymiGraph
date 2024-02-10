import networkx as nx

from .metric_scalar import ScalarMetric


class NumberOfTrianglesMetric(ScalarMetric):
    def __init__(self):
        super().__init__("|Triangles|")

    def compute_scalar(self, G: nx.Graph):
        return sum(nx.triangles(G).values()) / 3
