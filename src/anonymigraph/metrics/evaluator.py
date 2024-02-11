import logging
from typing import List

import graphblas_algorithms as ga
import networkx as nx

from anonymigraph.metrics.metric import Metric

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, metrics_list: List[Metric]):
        self.metrics = metrics_list

    def evaluate(self, G: nx.Graph, Ga: nx.Graph):
        logger.info("Converting Graphs to graphblas.")
        G_blas = ga.Graph.from_networkx(G)
        Ga_blas = ga.Graph.from_networkx(Ga)

        results = {}
        for metric in self.metrics:
            logger.info(f"Evaluating Metric {metric.name}.")
            if metric.pass_graph_as_graphblas:
                result = metric.evaluate(G_blas, Ga_blas)
            else:
                result = metric.evaluate(G, Ga)
            results[metric.name] = result
        return results
