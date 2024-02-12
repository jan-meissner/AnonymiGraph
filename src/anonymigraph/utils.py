import networkx as nx


def _validate_input_graph(G: nx.Graph):
    if not isinstance(G, nx.Graph):
        raise TypeError("The graph must be undirected. Please provide an undirected graph.")

    expected_labels = set(range(len(G)))
    actual_labels = set(G.nodes())

    if expected_labels != actual_labels:
        raise ValueError(
            "Graph nodes must be labeled with integers from 1 to G.number_of_nodes() - 1. Please relabel the graph"
            "accordingly or use `utils.relabel_graph`."
        )


def relabel_graph(G: nx.Graph) -> nx.Graph:
    """
    Returns a copy of the graph G with nodes relabeled to integers from 0 to G.number_of_nodes() - 1.

    Parameters:
    - G (nx.Graph): The graph to relabel.

    Returns:
    - nx.Graph: A copy of G with sequentially labeled nodes.
    """
    mapping = {node: i for i, node in enumerate(G.nodes())}
    return nx.relabel_nodes(G, mapping, copy=True)
