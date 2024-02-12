from typing import Any

import networkx as nx


class AbstractMetric:
    """
    Base class for graph metrics evaluation with optional GraphBLAS acceleration support.
    """

    def __init__(self, pass_graph_as_graphblas=False):
        """
        Initialize the Metric.

        Parameters:
            pass_graph_as_graphblas (boolean): True if graphblas graphs should be passed to evaluate.
        """
        self.pass_graph_as_graphblas = pass_graph_as_graphblas

    def evaluate(self, G: nx.Graph, Ga: nx.Graph) -> Any:
        """
        Evaluates the metric for the original graph G and the anonymized graph Ga. If self.pass_graph_as_graphblas is
        set to true G and Ga are graphblas graphs.

        Parameters:
            G (nx.Graph): The original graph.
            Ga (nx.Graph): The anonymized graph.

        Returns:
            Any: The result of the metric evaluation, either a number or a dictionary with keys G and Ga.
        """
        raise NotImplementedError("Subclass must implement abstract method")
