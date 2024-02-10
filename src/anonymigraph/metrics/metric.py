import networkx as nx


class Metric:
    def __init__(self, name):
        self.name = name

    def evaluate(self, G: nx.Graph, Ga: nx.Graph):
        raise NotImplementedError("Subclass must implement abstract method")
