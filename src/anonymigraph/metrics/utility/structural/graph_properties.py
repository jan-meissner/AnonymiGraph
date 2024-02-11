import networkx as nx

from .abstract_graph_metric import AbstractGraphMetric


class ConnectedComponentsMetric(AbstractGraphMetric):
    def __init__(self):
        super().__init__("|Connected Components|")

    def compute_scalar(self, G: nx.Graph):
        return nx.number_connected_components(G)


class NumberOfEdgesMetric(AbstractGraphMetric):
    def __init__(self):
        super().__init__("|Edges|")

    def compute_scalar(self, G: nx.Graph):
        return G.number_of_edges()


class NumberOfNodesMetric(AbstractGraphMetric):
    def __init__(self):
        super().__init__("|Nodes|")

    def compute_scalar(self, G: nx.Graph):
        return G.number_of_nodes()


class NumberOfTrianglesMetric(AbstractGraphMetric):
    def __init__(self):
        super().__init__("|Triangles|", pass_graph_as_graphblas=True)

    def compute_scalar(self, G: nx.Graph):
        return sum(nx.triangles(G).values()) / 3


class AverageClusteringCoefficientMetric(AbstractGraphMetric):
    def __init__(self):
        super().__init__("Average Clustering Coefficient", pass_graph_as_graphblas=True)

    def compute_scalar(self, G: nx.Graph):
        return nx.average_clustering(G)


class TransitivityMetric(AbstractGraphMetric):
    def __init__(self):
        super().__init__("Transitivity", pass_graph_as_graphblas=True)

    def compute_scalar(self, G: nx.Graph):
        return nx.transitivity(G)
