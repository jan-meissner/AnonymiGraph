import networkx as nx

from .abstract_graph_metric import AbstractGraphMetric


class ConnectedComponentsMetric(AbstractGraphMetric):
    """Compute the number of connected components in a graph."""

    def compute_scalar(self, G: nx.Graph):
        return nx.number_connected_components(G)


class NumberOfEdgesMetric(AbstractGraphMetric):
    """Calculate the total number of edges in a graph."""

    def compute_scalar(self, G: nx.Graph):
        return G.number_of_edges()


class NumberOfNodesMetric(AbstractGraphMetric):
    """Determine the total number of nodes in a graph."""

    def compute_scalar(self, G: nx.Graph):
        return G.number_of_nodes()


class NumberOfTrianglesMetric(AbstractGraphMetric):
    """Count the total number of triangles in a graph. Uses GraphBLAS to accelerate the calculation."""

    def __init__(self):
        super().__init__(pass_graph_as_graphblas=True)

    def compute_scalar(self, G: nx.Graph):
        return sum(nx.triangles(G).values()) / 3


class AverageClusteringCoefficientMetric(AbstractGraphMetric):
    """Calculate the average clustering coefficient of a graph. Uses GraphBLAS to accelerate the calculation."""

    def __init__(self):
        super().__init__(pass_graph_as_graphblas=True)

    def compute_scalar(self, G: nx.Graph):
        return nx.average_clustering(G)


class TransitivityMetric(AbstractGraphMetric):
    """Compute the transitivity of a graph. Uses GraphBLAS to accelerate the calculation."""

    def __init__(self):
        super().__init__(pass_graph_as_graphblas=True)

    def compute_scalar(self, G: nx.Graph):
        return nx.transitivity(G)
