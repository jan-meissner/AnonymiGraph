import networkx as nx
import numpy as np


def generate_power_law_graph(num_nodes, exponent):
    degrees = np.random.zipf(a=exponent, size=num_nodes)
    degrees = [d for d in degrees if 0 < d < num_nodes]
    if sum(degrees) % 2 == 1:
        degrees[-1] += 1

    print(f"Max Degree: {np.max(degrees)} - Median {np.median(degrees)} - Mean {np.mean(degrees)}")

    G = nx.configuration_model(degrees)
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))

    return G


def get_ring_of_rings(n_cliques, nodes_per_clique, r_clique, r_graph):
    G = nx.Graph()

    for i in range(n_cliques):
        clique_nodes = list(range(i * nodes_per_clique, (i + 1) * nodes_per_clique))
        G.add_nodes_from(clique_nodes)

        for node1, node2 in zip(clique_nodes, clique_nodes[1:]):
            G.add_edge(node1, node2)
        G.add_edge(clique_nodes[0], clique_nodes[-1])

    clique_edge_offset = nodes_per_clique // 4
    for i in range(n_cliques):
        source_node = nodes_per_clique - clique_edge_offset + i * nodes_per_clique
        target_node = (clique_edge_offset + (i + 1) * nodes_per_clique) % (nodes_per_clique * n_cliques)
        G.add_edge(source_node, target_node)

    pos = {}

    angles = np.linspace(-np.pi, np.pi, n_cliques, endpoint=False)

    for i in range(n_cliques):
        cx, cy = r_graph * np.cos(angles[i]), r_graph * np.sin(angles[i])
        small_angles = np.linspace(-np.pi, np.pi, nodes_per_clique, endpoint=False)
        for j in range(nodes_per_clique):
            x = cx + r_clique * np.cos(angles[i] + small_angles[j])
            y = cy + r_clique * np.sin(angles[i] + small_angles[j])
            pos[nodes_per_clique * i + j] = (x, y)

    return G, pos
