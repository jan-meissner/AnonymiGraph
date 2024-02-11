import logging

import networkx as nx
import numpy as np

from anonymigraph.anonymization._external.nest_model.fast_wl import WL_fast
from anonymigraph.utils import _validate_input_graph

from .abstract_node_metric import AbstractNodeMetric

logger = logging.getLogger(__name__)


class WLColorMetric(AbstractNodeMetric):
    def __init__(self, depth):
        """
        Metric that is computed by comparing the distance between two color Weisfeiler Lehman Coloring

        - depth: the depth of the wl algorithm: depth = 0 doesnt consider 0 hop subtree, depth = 1 is equivalent to
        degrees (1-hop neigh), ...
        """
        self.depth = depth
        super().__init__(f"TVD WL Colors d={self.depth}")

    def _total_variation_distance(self, p, q):
        """Calculate the Total Variation Distance."""
        return 0.5 * np.sum(np.abs(p - q))

    def evaluate(self, G: nx.Graph, Ga: nx.Graph):
        _validate_input_graph(G)
        _validate_input_graph(Ga)

        mapping_Ga = {node: i + len(G) for i, node in enumerate(Ga.nodes())}
        Ga_relabelled = nx.relabel_nodes(Ga, mapping_Ga)

        G_union = nx.union(G, Ga_relabelled)
        _validate_input_graph(G_union)
        # assert G_union.number_of_nodes() == G.number_of_nodes() + Ga.number_of_nodes()

        logger.info("Calculating WL colors.")
        edges = np.array(G_union.edges, dtype=np.uint32)
        bidirectional_edges = np.row_stack((edges, edges[:, [1, 0]]))
        colors = WL_fast(bidirectional_edges, G_union.number_of_nodes(), labels=None, max_iter=self.depth + 1)[-1]
        logger.info("Comparing distributions of WL colors using TVD.")
        G_dist, Ga_dist = _labels_to_prob_dists(colors[: len(G)], colors[len(G) :])

        # Can't use Wasserstein distance as there is no distance between colors
        # Instead using TVD (Which is wasserstein but with d(x,y) = 1_x!=y)
        return self._total_variation_distance(G_dist, Ga_dist)


def _labels_to_prob_dists(p_labels, q_labels):
    """Convert label vectors to probability distributions considering all unique labels in both vectors."""
    all_labels = np.concatenate((p_labels, q_labels))
    unique_labels = np.unique(all_labels)

    p_indices = np.searchsorted(unique_labels, p_labels)
    q_indices = np.searchsorted(unique_labels, q_labels)

    p_prob_dist = np.zeros(len(unique_labels))
    np.add.at(p_prob_dist, p_indices, 1)

    q_prob_dist = np.zeros(len(unique_labels))
    np.add.at(q_prob_dist, q_indices, 1)

    return p_prob_dist / len(p_labels), q_prob_dist / len(q_labels)


# def _kl_divergence(p, q):
#    """Calculate the Kullback-Leibler divergence."""
#    return np.sum(np.where(p != 0, p * np.log(p / q), 0))
#
# def _jensen_shannon_divergence(p, q):
#    """Calculate the Jensen-Shannon Divergence."""
#    m = 0.5 * (p + q)
#    return 0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m)
#
# def _hellinger_distance(p, q):
#    """Calculate the Hellinger Distance."""
#    return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))
#
# def _bhattacharyya_distance(p, q):
#    """Calculate the Bhattacharyya Distance."""
#    return -np.log(np.sum(np.sqrt(p * q)))
#
