import networkx as nx


class Metric:
    def __init__(self, name, pass_graph_as_graphblas=False):
        self.name = name
        self.pass_graph_as_graphblas = pass_graph_as_graphblas

    def evaluate(self, G: nx.Graph, Ga: nx.Graph):
        raise NotImplementedError("Subclass must implement abstract method")
