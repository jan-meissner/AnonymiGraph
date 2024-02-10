import logging

import networkx as nx

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, metrics_dict):
        self.metrics = metrics_dict

    def evaluate(self, G: nx.Graph, Ga: nx.Graph):
        results = {}
        for metric in self.metrics:
            result = metric.evaluate(G, Ga)
            results[metric.name] = result
            logger.info("Metric '%s' evaluated. Result: %s", metric.name, result)
        return results
