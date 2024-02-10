import networkx as nx

from .metric_scalar import ScalarMetric


class ConnectedComponentsMetric(ScalarMetric):
    def __init__(self):
        super().__init__("|Connected Components|")

    def compute_scalar(self, G: nx.Graph):
        return nx.number_connected_components(G)
