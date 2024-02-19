import logging
from typing import Dict

import graphblas_algorithms as ga
import networkx as nx

from anonymigraph.metrics.abstract_metric import AbstractMetric

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluator class for evaluating anonymization techniques of graphs.

    Args:
        metrics (Dict[str, AbstractMetric]): A dictionary of metrics to be evaluated.
        use_graphblas (bool, optional): Flag indicating whether to use graphblas if a metric can use it.
                                        Defaults to True.
    """

    def __init__(self, metrics: Dict[str, AbstractMetric], use_graphblas=True):
        self.metrics = metrics
        self.use_graphblas = use_graphblas

    def evaluate(self, G: nx.Graph, Ga: nx.Graph):
        """
        Evaluate the metrics on the given graphs.

        Args:
            G (nx.Graph): The original graph.
            Ga (nx.Graph): The anonymized graph.

        Returns:
            dict: A dictionary containing the evaluation results for each metric.
        """
        logger.info("Converting graphs to graphblas")
        if self.use_graphblas:
            G_blas = ga.Graph.from_networkx(G)
            Ga_blas = ga.Graph.from_networkx(Ga)

        results = {}
        for metric_name, metric in self.metrics.items():
            logger.info(f"Evaluating Metric {metric_name}")
            if metric.pass_graph_as_graphblas and self.use_graphblas:
                result = metric.evaluate(G_blas, Ga_blas)  # Assuming compute_scalar is the method to use
            else:
                result = metric.evaluate(G, Ga)
            results[metric_name] = result
        return results
