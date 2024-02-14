import networkx as nx

from anonymigraph.metrics.utility.structural.abstract_node_metric import (
    AbstractNodeMetric,
)


class DegreeCentralityMetric(AbstractNodeMetric):
    """
    This class calculates and compares the degree centrality of two graphs.
    The comparison is done using the default distribution distance function used by AbstractNodeMetric.
    Uses GraphBLAS to accelerate the calculation.
    """

    def __init__(self):
        super().__init__()  # graphblas = True causes floating point errors

    def compute_node_distribution(self, G: nx.Graph):
        return list(nx.degree_centrality(G).values())


class EigenvectorMetric(AbstractNodeMetric):
    """
    This class calculates and compares the eigenvector centralities of two graphs.
    The comparison is done using the default distribution distance function used by AbstractNodeMetric.
    Uses GraphBLAS to accelerate the calculation.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the EigenvectorMetric.

        Args:
            *args: Additional positional arguments to be passed to nx.eigenvector_centrality.
            **kwargs: Additional keyword arguments to be passed to nx.eigenvector_centrality.
        """
        super().__init__(pass_graph_as_graphblas=True)
        self.args = args
        self.kwargs = kwargs

        if "max_iter" not in self.kwargs:
            self.kwargs["max_iter"] = 1000

    def compute_node_distribution(self, G: nx.Graph):
        return list(nx.eigenvector_centrality(G, *self.args, **self.kwargs).values())


class LocalClusteringCoefficientMetric(AbstractNodeMetric):
    """
    This class calculates and compares the local clustering coefficients of two graphs.
    The comparison is done using the default distribution distance function used by AbstractNodeMetric.
    Uses GraphBLAS to accelerate the calculation.
    """

    def __init__(self):
        super().__init__(pass_graph_as_graphblas=True)

    def compute_node_distribution(self, G: nx.Graph):
        return list(nx.clustering(G).values())


class PageRankMetric(AbstractNodeMetric):
    """
    This class calculates and compares the PageRank centralities of two graphs.
    The comparison is done using the default distribution distance function used by AbstractNodeMetric.
    Uses GraphBLAS to accelerate the calculation.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the PageRankMetric.

        Args:
            *args: Additional positional arguments to be passed to nx.pagerank.
            **kwargs: Additional keyword arguments to be passed to nx.pagerank.
        """
        super().__init__(pass_graph_as_graphblas=True)
        # Store args and kwargs to be used in compute_node_distribution
        self.args = args
        self.kwargs = kwargs

    def compute_node_distribution(self, G: nx.Graph):
        return list(nx.pagerank(G, *self.args, **self.kwargs).values())


class ClosenessCentralityMetric(AbstractNodeMetric):
    """
    This class calculates and compares the closeness centralities of two graphs.
    The comparison is done using the default distribution distance function used by AbstractNodeMetric.
    """

    def __init__(self):
        super().__init__()

    def compute_node_distribution(self, G: nx.Graph):
        return list(nx.closeness_centrality(G).values())
