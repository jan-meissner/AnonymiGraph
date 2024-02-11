import networkx as nx

from anonymigraph.metrics.utility.structural.abstract_node_metric import (
    AbstractNodeMetric,
)


class DegreeCentralityMetric(AbstractNodeMetric):
    def __init__(self):
        super().__init__("Degree Centrality", pass_graph_as_graphblas=True)

    def compute_node_distribution(self, G: nx.Graph):
        return list(nx.degree_centrality(G).values())


class EigenvectorMetric(AbstractNodeMetric):
    def __init__(self):
        super().__init__("Eigenvector Centrality", pass_graph_as_graphblas=True)

    def compute_node_distribution(self, G: nx.Graph):
        return list(nx.eigenvector_centrality(G, max_iter=1000).values())  # max_iter may need adjustment


class LocalClusteringCoefficientMetric(AbstractNodeMetric):
    def __init__(self):
        super().__init__("Local Clustering Coefficient", pass_graph_as_graphblas=True)

    def compute_node_distribution(self, G: nx.Graph):
        return list(nx.clustering(G).values())


class PageRankMetric(AbstractNodeMetric):
    def __init__(self):
        super().__init__("PageRank", pass_graph_as_graphblas=True)

    def compute_node_distribution(self, G: nx.Graph):
        return list(nx.pagerank(G).values())


class ClosenessCentralityMetric(AbstractNodeMetric):
    def __init__(self):
        super().__init__("Closeness Centrality")

    def compute_node_distribution(self, G: nx.Graph):
        return list(nx.closeness_centrality(G).values())
