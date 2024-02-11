import networkx as nx
import numpy as np


def generate_power_law_graph(num_nodes, exponent):
    degrees = np.random.zipf(a=exponent, size=num_nodes)
    degrees = [d for d in degrees if 0 < d < num_nodes]
    if sum(degrees) % 2 == 1:
        degrees[-1] += 1

    G = nx.configuration_model(degrees)
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))

    return G


def get_ring_of_rings(n_rings, nodes_per_ring, r_ring, r_graph):
    G = nx.Graph()

    for i in range(n_rings):
        clique_nodes = list(range(i * nodes_per_ring, (i + 1) * nodes_per_ring))
        G.add_nodes_from(clique_nodes)

        for node1, node2 in zip(clique_nodes, clique_nodes[1:]):
            G.add_edge(node1, node2)
        G.add_edge(clique_nodes[0], clique_nodes[-1])

    clique_edge_offset = nodes_per_ring // 4
    for i in range(n_rings):
        source_node = nodes_per_ring - clique_edge_offset + i * nodes_per_ring
        target_node = (clique_edge_offset + (i + 1) * nodes_per_ring) % (nodes_per_ring * n_rings)
        G.add_edge(source_node, target_node)

    pos = {}

    angles = np.linspace(-np.pi, np.pi, n_rings, endpoint=False)

    for i in range(n_rings):
        cx, cy = r_graph * np.cos(angles[i]), r_graph * np.sin(angles[i])
        small_angles = np.linspace(-np.pi, np.pi, nodes_per_ring, endpoint=False)
        for j in range(nodes_per_ring):
            x = cx + r_ring * np.cos(angles[i] + small_angles[j])
            y = cy + r_ring * np.sin(angles[i] + small_angles[j])
            pos[nodes_per_ring * i + j] = (x, y)

    return G, pos
